# Spec: whisper duration cap (`--max-whisper-duration`)

**Status:** proposed (2026-10-17). Requested by the n8n_transcript pipeline
(`~/n8n_projects/n8n_transcript/docs/design/gpu-fallback.md`).

## Why

The n8n pipeline runs scriber on a CPU-only box with 7.6 GB RAM. When a YouTube
video has no usable captions, `handle_url` falls back to whisper. On that box whisper
runs at ~0.35× real time and is killed for lack of memory above ~1h45 of audio.

Long caption-less videos will be transcribed on a GPU machine instead. The box keeps
scriber's normal flow (captions first, whisper fallback) and only needs scriber to
**refuse the whisper fallback above a duration**, so the caller can hand the video to
the GPU machine.

## 1. Whisper duration cap (must)

### Behaviour

- New setting `max_whisper_duration_sec` (default: none = no cap).
- Applies to YouTube URLs, at the point where `handle_url` is about to call
  `_transcribe_url_via_whisper` (caption fallback, and the `--diarize` path, which
  always uses whisper).
- Before any audio download: get the duration with `pya.fetch_video_metadata(url)`
  (`SourceMetadata.duration_seconds`; metadata only, no download).
- Duration > cap, or duration unknown while a cap is set: do not download, do not run
  whisper. Raise a new `WhisperDeferredError(duration_sec, cap_sec)`.
- Duration ≤ cap: unchanged (download + whisper).
- Captions found: unchanged, whatever the duration. The cap only gates whisper.
- Local media and text inputs: unaffected.

### Output contract

- Log line at WARNING level, stable and greppable:
  `Whisper deferred: duration=<sec>s cap=<sec>s — no captions, video exceeds the whisper cap`
  (`duration=unknown` when yt-dlp gives none).
- Single input: exit code **3**. Not used today (0 ok, 1 batch failure, 2 preflight).
- Batch with `--continue-on-error`: the input is recorded as failed with that message
  in the ✓/✗ summary; the run exits non-zero as for any failure.
- No transcript file is written.

### Config (DESIGN.md §11 pattern)

- `Settings.max_whisper_duration_sec: int | None = None`.
- Env `MAX_WHISPER_DURATION_SEC` (unset or empty = no cap).
- CLI `--max-whisper-duration SECONDS` on `transcribe` and `summarize`, overlaid in
  `main._apply_cli_overrides`.

Deployment intent: the box container sets `MAX_WHISPER_DURATION_SEC=2700`; the GPU
machine leaves it unset.

### Tests

- Cap set, no captions, duration over the cap → `WhisperDeferredError`; neither the
  audio download nor whisper is called.
- Cap set, no captions, duration under the cap → whisper path unchanged.
- Cap set, no captions, duration unknown → deferred.
- Cap set, captions found, long video → caption transcript returned (cap ignored).
- Cap set, `--diarize`, long video → deferred.
- No cap → behaviour unchanged (regression).
- Precedence: CLI flag over env over default.
- Exit code 3 and the log line for a single input.
- `tests/test_base_install_imports.py` still passes (no new imports on the base path).

### Docs

- README: flag in the CLI reference; the caption ladder's step 7 ("No captions → fall
  back to whisper") gains "unless the video exceeds `--max-whisper-duration`".
- DESIGN.md §10 (error handling): `WhisperDeferredError`; §11: the new setting.

## 2. Retry a rate-limited caption download (should)

Both recent pipeline failures started with
`Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests`.
The track existed; YouTube throttled the request, and scriber went to whisper. One
delayed retry would turn some of these into caption transcripts.

- In the caption download step only (reason `download_failed`), when the yt-dlp error
  contains `HTTP Error 429`: wait, then retry once.
- Delay: 60 s default, env `CAPTIONS_RETRY_DELAY_SEC`, `0` disables the retry.
- One retry, not a loop: repeated requests extend YouTube's throttling.
- Log both attempts at WARNING.

Tests: 429 then success → transcript returned, one sleep; 429 twice → original
`download_failed` behaviour (then whisper or deferral); non-429 error → no retry;
delay `0` → no retry. Patch the sleep.

## Out of scope

- Output filenames keyed by video id. Callers already isolate each run in its own
  `--output-dir` and glob the single `* transcript.txt`.
- Any change to whisper itself, diarization or summarization.

## Consumers

- n8n_transcript scriber shim (Homelab `stacks/rootless/n8n/scriber-companion/app/main.py`):
  maps exit code 3 to `422 {"status":"no_transcript","reason":"whisper_deferred"}`. The
  scriber container sets `MAX_WHISPER_DURATION_SEC=2700`. Needs `SCRIBER_REF` bumped to
  the commit carrying the feature.
- The GPU worker on the Windows PC runs scriber without a cap (captions first, then
  whisper on CUDA).
