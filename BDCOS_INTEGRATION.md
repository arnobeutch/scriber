# Handoff: lean transcription-only install for BDC OS

**Audience:** the scriber maintainer (you, in a scriber session).
**Author:** Claude, from the `bdc_os` project, 2026-06-01.
**Status:** ✅ **implemented** (2026-06-02). Base is transcription-only; `diarize`
and `summarize` are extras; optional imports are lazy with actionable errors.
Two deviations from this note: (1) torch is **not** CPU-pinned in `pyproject.toml`
— scriber stays GPU-capable, and CPU-only consumers select the backend at
install time via `UV_TORCH_BACKEND=cpu` (see `docs/BDCOS_INSTALL.md` +
`scripts/install-bdcos.sh`); (2) `triton` is a *direct* openai-whisper dep on
linux-x86_64, not torch-wheel baggage, so it stays in the base — the CPU wheel
sheds the ~2.7 GB `nvidia/*` stack but not triton.

> **Historical snapshot.** This is the original handoff note; the embedded
> `pyproject.toml` excerpt is point-in-time (e.g. `diarize` is now
> `pyannote-audio>=4.0`, and Windows pins a win32-only CUDA torch index). For
> current dependency versions and install steps, `pyproject.toml` and
> [docs/BDCOS_INSTALL.md](docs/BDCOS_INSTALL.md) are canonical.

---

## Why this note exists

BDC OS (a separate project at `~/Documents/dev/bdc_os/`) is adopting scriber as
its **transcription provider**. Its `workflow.meeting_summarizer` — and any skill
that ingests a video/audio source — will shell out to the `scriber` CLI in
**`transcribe` mode only**. BDC OS does its own summarization (own summarizer
skill, vault integration, FR/EN chain, templates), so it never calls scriber's
`summarize` path.

BDC OS consumes scriber as an **isolated `uv tool`** (`uv tool install` from this
git repo), not as a fixed local path and not as a library — so scriber's
dependency footprint becomes BDC OS's install cost. That's what motivates this.

### The measurement that drove the decision

A full `uv sync` of scriber today is **6.3 GB**. Breakdown of the heavy hitters:

| Package | Size | Pulled by |
| --- | --- | --- |
| `nvidia/*` | 2.7 GB | default (GPU) torch wheel |
| `torch` | 1.7 GB | `openai-whisper` |
| `triton` | 545 MB | default torch wheel |
| `cusparselt` | 227 MB | default torch wheel |
| `llvmlite`+`numba` | 149 MB | `pyannote` → librosa |
| `scipy`/`sklearn`/`pandas`/`matplotlib` | ~230 MB | `pyannote` stack |
| `chromadb_rust`/`onnxruntime`/`kubernetes` | ~113 MB | summarization (RAG) stack |

Key facts:

- **openai-whisper *is* a torch app** — torch can't be removed while keeping the
  current engine. But ~3.5 GB of that (`nvidia` + `triton` + `cusparselt`) is
  **GPU baggage** from the default torch wheel. A **CPU-only torch wheel** sheds
  it while keeping openai-whisper byte-for-byte identical (so the tuning in
  `docs/WHISPER_SETUP.md` stays 100 % valid).
- **Diarization is torch-bound** (pyannote is a torch app), so it can't move to a
  torch-free engine. It stays available but **opt-in**.
- **Summarization** (chromadb/langchain/openai/textblob) is pure weight BDC OS
  never needs.

### The decision (made with the BDC OS owner)

Target the **CPU-torch + opt-in-diarize** shape:

- **Base install = transcription-only, CPU-torch (~1.3 GB).** openai-whisper
  unchanged.
- **`diarize` extra (~+0.5 GB, needs `HUGGINGFACE_TOKEN`)** — installed only when
  speaker-attributed notes are wanted.
- **`summarize` extra** — scriber's existing summarization backends, preserved for
  scriber's own CLI users, but not installed for BDC OS.

> **Design call for you:** this makes scriber's *default* install lean, which is a
> behavior change for scriber's own users (`scriber summarize` would then require
> `scriber[summarize]`). Recommended: accept that, make base = transcription-only,
> and update the README's install/usage so full users run
> `uv sync --all-extras` (or `scriber[diarize,summarize]`). If you'd rather keep
> base = full for scriber's own ergonomics, the alternative is a `lean` story for
> BDC OS instead — but extras-off-by-default is cleaner and what the pyproject
> below assumes. Your repo, your call.

`uv tool install` itself needs **nothing new** — scriber already has
`[build-system]` (hatchling), `[project.scripts]`, and a `src/` layout. The base
becomes lean automatically once the deps move to extras.

---

## TODO

### 1. `pyproject.toml` — split deps into extras + pin CPU torch

Move the optional deps out of `dependencies` and add a CPU torch index:

```toml
dependencies = [                       # transcription-only base
    "bs4>=0.0.2",
    "colorama>=0.4.6",
    "ffmpeg-python>=0.2.0",
    "langdetect>=1.0.9",
    "numpy>=2.2.4",
    "openai-whisper>=20240930",
    "pyyaml>=6.0.3",
    "requests>=2.32.3",
    "tqdm>=4.67.1",
    "unidecode>=1.3.8",
    "yt-dlp>=2026.3.17",
]

[project.optional-dependencies]
diarize   = ["pyannote-audio>=3.3.2", "torchaudio>=2.6.0"]
summarize = [
    "chromadb>=1.0.4",
    "langchain>=0.3.23",
    "langchain-chroma>=0.2.5",
    "langchain-community>=0.3.21",
    "langchain-ollama>=0.3.1",
    "openai>=1.67.0",
    "textblob>=0.19.0",
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch      = { index = "pytorch-cpu" }
torchaudio = { index = "pytorch-cpu" }
```

Pinning torch to the CPU index drops `nvidia/*`, `triton`, and `cusparselt`
transitively — they're deps of the GPU wheel only. (`torchaudio` is in the
`diarize` extra; pin it too so the extra is also CPU-only.)

### 2. Make the optional imports lazy (the part that isn't just metadata)

The deps are currently imported **eagerly**, so a base-only install would crash on
import even for plain transcription. Three sites:

- **`src/scriber/transcription/local.py:17-18`** —
  `from pyannote.audio import Pipeline` / `from pyannote.core import Segment, Timeline`
  are at module top, but this is the **same module as the whisper path**. Move them
  into the diarization function body, behind the `--diarize` branch. On `ImportError`,
  raise a clear message: *"diarization requires `uv tool install 'scriber[diarize]'`
  (or `uv sync --extra diarize`)."*
- **`src/scriber/main.py:16`** — `from scriber.summarizers import MissingAPIKeyError, make_summarizer`
  is top-level, so importing `scriber.main` pulls langchain/openai/textblob. Defer it
  into the `summarize`-subcommand path (the `will_summarize` branch around
  `main.py:108`/`142`). On `ImportError`, point at `scriber[summarize]`.
- **`src/scriber/summarizers/`** — `base.py:8` (textblob), `engine.py:5-8`
  (langchain), `openai_compatible.py:8-9` (openai) are fine *inside* the package;
  just ensure nothing on the `transcribe` path imports the `summarizers` package at
  all (check `summarizers/__init__.py` re-exports aren't dragged in by a shared
  module).

### 3. Verify

- `uv sync` (no extras) → `just all` green, and `uv run scriber transcribe <media>`
  runs clean with **no** pyannote/langchain/openai installed.
- `uv sync --extra diarize` → `uv run scriber transcribe <media> --diarize` works.
- `uv sync --extra summarize` → `uv run scriber summarize <media>` works.
- `uv sync --all-extras` → full `just all` (271 tests) green. Mind the pyright-strict
  boundary: conditional imports may need `# pyright: ...` headers like the existing
  ML-boundary files.
- Sanity-check the wheel still builds for tool install:
  `uvx --from . scriber --help` (or `uv build`).

### 4. Constraints

- Keep scriber's existing CLI behavior intact under the full install. This is a
  packaging/lazy-import refactor, **not** a feature change.
- Don't auto-commit (scriber rule). Run the pre-commit gate.
- `docs/WHISPER_SETUP.md` needs no change — the engine is unchanged.

---

## What BDC OS will do once this lands

```bash
uv tool install git+https://github.com/arnobeutch/scriber.git          # ~1.3 GB base
# optional, when speaker labels are wanted (extra on the main package):
uv tool install "scriber[diarize] @ git+https://github.com/arnobeutch/scriber.git"
```

Then BDC OS calls `scriber transcribe <input> --language <fr|en> [--diarize]` as a
subprocess. No fixed path, no shared environment, no summarization deps.
