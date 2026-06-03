# Installing scriber — BDC OS's transcription provider

**BDC OS owns this integration.** The canonical copy lives in the bdc_os repo
(`docs/scriber-install.md` + `scripts/install-scriber.sh`); this scriber-side
copy is kept in sync so scriber's maintainer can see how bdc_os consumes scriber
and avoid breaking the contract.

BDC OS uses scriber for transcript acquisition: any skill that ingests a
video/audio source shells out to the `scriber` CLI in **`transcribe` mode only**.
BDC OS never calls scriber's summarization path — it summarizes downstream with
its own skills.

scriber is consumed as an **isolated `uv` tool installed from GitHub**. It is
**not** assumed to be checked out locally; the install pulls it from its git
remote into its own environment.

## TL;DR

```bash
# reduced scriber: CPU torch, base + diarization, from GitHub (~2.1 GB)
./scripts/install-bdcos.sh

# transcription only, no speaker labels (~1.3 GB)
./scripts/install-bdcos.sh --no-diarize
```

Or directly, without the script:

```bash
# default (base + diarization), CPU torch, from GitHub
UV_TORCH_BACKEND=cpu uv tool install "scriber[diarize] @ git+https://github.com/arnobeutch/scriber.git"

# transcription only
UV_TORCH_BACKEND=cpu uv tool install "scriber @ git+https://github.com/arnobeutch/scriber.git"
```

## What gets installed

| Install | Pulls | Footprint |
| --- | --- | --- |
| `scriber[diarize]` (default) | whisper + CPU torch + pyannote + torchaudio + ffmpeg-python + yt-dlp | ~2.1 GB |
| `scriber` (`--no-diarize`) | whisper + CPU torch + ffmpeg-python + yt-dlp | ~1.3 GB |
| `scriber[summarize]` | + chromadb/langchain/openai/textblob | BDC OS never installs this |

The summarization backends are never installed for BDC OS. Their imports (and the
diarization imports, under `--no-diarize`) are lazy, so each install runs cleanly
without the omitted packages; requesting `--diarize` on a `--no-diarize` install
fails with an actionable message pointing at `scriber[diarize]`.

## Why CPU torch is selected here, not pinned upstream

openai-whisper is a torch app, so torch can't be removed. But the default torch
wheel carries ~2.7 GB of CUDA baggage (`nvidia/*`) that CPU-only consumers never
use. Selecting the CPU wheel sheds it while keeping openai-whisper byte-for-byte
identical — the whisper tuning in [WHISPER_SETUP.md](WHISPER_SETUP.md) stays
valid. (`triton`, ~545 MB, is a direct openai-whisper dep on linux-x86_64, not
GPU baggage — it stays on the CPU path.)

scriber deliberately does **not** pin torch to a CPU index in its `pyproject.toml`:
that would force CPU on every scriber user, including those who want GPU whisper.
The CPU choice is made per-install via `UV_TORCH_BACKEND=cpu`, which is what the
install script sets.

> **uv version note:** `UV_TORCH_BACKEND` is honored by uv's resolver (tested on
> uv 0.11). If a future uv ignores it for `uv tool install`, fall back to an
> explicit CPU index:
> `uv tool install --index https://download.pytorch.org/whl/cpu "scriber[diarize] @ <repo>"`.

## Diarization needs a Hugging Face token

The default install includes diarization, but **diarized runs** load gated
pyannote models (`speaker-diarization-3.1`, `voice-activity-detection`). Export a
token with access to them before running with `--diarize`:

```bash
export HUGGINGFACE_TOKEN=hf_...
scriber transcribe ./meeting.mp4 --language fr --diarize
```

Without the token, the diarize path raises a clear error and stops. Plain
transcription (no `--diarize`) needs no token.

**How scriber sources the token.** `diarize.py` reads `os.getenv("HUGGINGFACE_TOKEN")`
straight from the **process environment**; `settings.py` additionally seeds it from
a `.env` in the **current working directory** (via `setdefault`, so a real env var
wins). The installed `uv` tool has no "scriber root" `.env` — that path only exists
in scriber's dev checkout. So the robust contract is: **the caller exports the token
into scriber's process env.** An install script can't do this (env vars don't persist
past the process).

**In BDC OS:** the token lives in `.bdcos.env` (the same file skills read for API
keys). The scriber-invoking skill exports it before the subprocess call:

```bash
HUGGINGFACE_TOKEN="$(grep -E '^HUGGINGFACE_TOKEN=' "$BDCOS_ENV" | cut -d= -f2-)" \
  scriber transcribe <input> --language <fr|en> --diarize
```

## Calling scriber from BDC OS

```bash
scriber transcribe <input> --language <fr|en> [--diarize]
```

Subprocess call, no shared environment, no fixed path. `<input>` is a YouTube URL
or a local media/text path. The transcript is written to the output directory;
BDC OS does its own summarization downstream.
