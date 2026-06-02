# Installing scriber as a lean transcription provider (BDC OS)

scriber's default install is **transcription-only** and stays GPU-capable for
its own users. Consumers that only need transcription — and want a small,
CPU-only footprint — install the base package and select the CPU torch backend
at install time. This is exactly what BDC OS does: it shells out to
`scriber transcribe` and never touches the summarization path.

## TL;DR

```bash
# transcription-only, CPU torch (~1.3 GB)
./scripts/install-bdcos.sh

# + speaker diarization (pyannote + torchaudio, ~+0.5 GB; needs HUGGINGFACE_TOKEN)
./scripts/install-bdcos.sh --diarize
```

Or directly, without the script:

```bash
UV_TORCH_BACKEND=cpu uv tool install "scriber @ git+https://github.com/arnobeutch/scriber.git"
# optional diarization extra:
UV_TORCH_BACKEND=cpu uv tool install "scriber[diarize] @ git+https://github.com/arnobeutch/scriber.git"
```

## What gets installed

| Install | Pulls | Footprint |
| --- | --- | --- |
| base (`scriber`) | whisper + CPU torch + ffmpeg-python + yt-dlp | ~1.3 GB |
| `scriber[diarize]` | + pyannote-audio + torchaudio | ~+0.5 GB |
| `scriber[summarize]` | + chromadb/langchain/openai/textblob | (BDC OS never installs this) |

The base install excludes both the `diarize` and `summarize` extras. Their
imports are lazy, so `scriber transcribe` runs cleanly with neither installed;
requesting `--diarize` without the extra fails with an actionable message
pointing at `scriber[diarize]`.

## Why CPU torch is selected here, not pinned upstream

openai-whisper is a torch app, so torch can't be removed. But the default torch
wheel carries ~3.5 GB of CUDA baggage (`nvidia/*`, `triton`, `cusparselt`) that
CPU-only consumers never use. A CPU-only wheel sheds it while keeping
openai-whisper byte-for-byte identical — the whisper tuning in
[`WHISPER_SETUP.md`](WHISPER_SETUP.md) stays valid.

scriber deliberately does **not** pin torch to a CPU index in `pyproject.toml`:
that would force CPU on every scriber user, including those who want GPU whisper.
Instead the CPU choice is made per-install via `UV_TORCH_BACKEND=cpu`.

> **uv version note:** `UV_TORCH_BACKEND` is honored by uv's resolver
> (tested on uv 0.11). If a future uv ignores it for `uv tool install`, fall
> back to an explicit CPU index:
> `uv tool install --index https://download.pytorch.org/whl/cpu "scriber @ <repo>"`.

## Diarization needs a Hugging Face token

`--diarize` loads gated pyannote models
(`speaker-diarization-3.1`, `voice-activity-detection`). Export a token with
access to them before running:

```bash
export HUGGINGFACE_TOKEN=hf_...
scriber transcribe ./meeting.mp4 --language fr --diarize
```

Without the token, the diarize path raises a clear error and stops.

## Calling scriber from BDC OS

```bash
scriber transcribe <input> --language <fr|en> [--diarize]
```

Subprocess call, no shared environment, no fixed path. `<input>` is a YouTube
URL or a local media/text path. The transcript is written to the output
directory; BDC OS does its own summarization downstream.
