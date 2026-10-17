# Spec: `--captions-only` (no whisper fallback)

**Status:** proposed (2026-10-17). Requested by the n8n_transcript pipeline
(`~/n8n_projects/n8n_transcript/docs/design/gpu-fallback.md`).

## Why

The n8n pipeline runs scriber on a CPU-only box with 7.6 GB RAM. When a YouTube
video's captions can't be fetched, `handle_url` falls back to whisper. On that box
whisper runs at ~0.35× real time and is killed for lack of memory above ~1h45 of audio.

The pipeline will route long caption-less videos to a GPU machine instead. To do that
it needs scriber to try captions and **stop** when there are none, rather than
downloading audio and starting whisper.

## 1. `--captions-only` (must)

### Behaviour

- New flag on `transcribe` and `summarize`: `--captions-only`.
- In `handle_url`, when `get_youtube_transcript` raises `TranscriptUnavailableError`
  and the flag is set: do not call `_transcribe_url_via_whisper`, do not download
  audio. Raise a new `CaptionsUnavailableError(reason, message)` that carries the
  original `reason` tag (`list_failed`, `lang_not_found`, `download_failed`,
  `empty_payload`).
- Caption selection is unchanged (manual > auto, language ladder, `-l` hint).
- The caption-success path is unchanged: same transcript file, same output contract.
- Local media and text inputs ignore the flag (no captions involved).
- `--captions-only` with `--diarize` is a parse error: diarization needs the audio.

### Output contract

- Log line at WARNING level, stable and greppable:
  `Captions unavailable (captions-only): reason=<tag> — <message>`.
- Single input: exit code **3**. Not used today (0 ok, 1 batch failure, 2 preflight).
- Batch with `--continue-on-error`: the input is recorded as failed with the reason
  in the ✓/✗ summary; the run exits non-zero as for any failure.
- No transcript file is written.

### Config

CLI flag only. It is a per-call decision taken by the caller (the pipeline decides
per video, from its duration), so no `Settings` field or env var.

### Tests

- Flag set + `TranscriptUnavailableError` → `CaptionsUnavailableError` with the same
  reason; `_transcribe_url_via_whisper` and the audio download are not called.
- Flag set + captions found → same `Transcript` as without the flag.
- Flag unset → whisper fallback unchanged (regression).
- `--captions-only --diarize` → parser error.
- Exit code 3 and the log line for a single input.
- `tests/test_base_install_imports.py` still passes (no new imports on the base path).

### Docs

- README: flag in the CLI reference.
- DESIGN.md §10 (error handling): add `CaptionsUnavailableError` next to
  `TranscriptUnavailableError`.

## 2. Retry a rate-limited caption download (should)

Both recent pipeline failures started with
`Unable to download video subtitles for 'en': HTTP Error 429: Too Many Requests`.
The track exists; YouTube throttled the request. One delayed retry would turn some of
these into caption successes instead of a whisper run or a GPU job.

- In the caption download step only (reason `download_failed`), when the yt-dlp error
  contains `HTTP Error 429`: wait, then retry once.
- Delay: 60 s default, env `CAPTIONS_RETRY_DELAY_SEC`, `0` disables the retry.
- One retry, not a loop: repeated requests extend YouTube's throttling.
- Log both attempts at WARNING.
- Applies with or without `--captions-only`.

Tests: 429 then success → transcript returned, one sleep; 429 twice → original
`download_failed` behaviour; non-429 error → no retry; delay `0` → no retry. Patch
the sleep.

## Out of scope

- Output filenames keyed by video id. Callers already isolate each run in its own
  `--output-dir` and glob the single `* transcript.txt`.
- Any change to whisper, diarization or summarization.

## Consumers

- n8n_transcript scriber shim (Homelab `stacks/rootless/n8n/scriber-companion/app/main.py`):
  passes `--captions-only` when the request sets `captionsOnly`, maps exit code 3 to
  `422 {"status":"no_transcript","reason":"captions_unavailable"}`. Needs `SCRIBER_REF`
  bumped to the commit carrying the flag.
- The GPU worker on the Windows PC runs scriber **without** the flag (captions first,
  then whisper on CUDA).
