# scriber — Claude Reference

## Project

CLI tool to transcribe and summarize YouTube videos, local audio/video files, or pre-existing text transcripts. Two subcommands: `scriber transcribe <input>...` (transcript only) and `scriber summarize <input>...` (transcribe + summarize). Pipeline: fetch/transcribe → optional diarization → sentiment/polarity → LLM summary (OpenAI API, OpenRouter, or local RAG via langchain + Ollama).

## Tech stack

- Python `>=3.11` (3.11–3.14 verified: resolve + cp314 wheels + live import on 3.14.5). The old `==3.11.9` pin is gone — modern `openai-whisper` / `numba` / `torch` / `pyannote-audio` all span 3.11–3.14. **Diarization requires `pyannote-audio>=4.0`**: `torchaudio>=2.11` (needed for 3.14 wheels) removed `torchaudio.AudioMetaData`, which every pyannote 3.x imports — so 3.x and 3.14 are mutually exclusive. pyannote 4.x uses `token=` (not `use_auth_token=`) and the torchcodec audio backend.
- `uv` for dependency/env management.
- `ruff` lint (`select = ["ALL"]`; ignores in `pyproject.toml`, `target-version = "py311"` — the supported floor; keep code 3.11-compatible).
- `pyright` strict mode (`[tool.pyright]` in `pyproject.toml`, `pythonVersion = "3.11"` — type-check against the oldest supported Python) — matches VSCode Pylance.
- `.claude/hooks/python-quality.sh` runs after every `Edit`/`Write` on a `.py` file. If it exits 2, read the stderr diagnostics and fix before continuing. See `.claude/rules/python_strict.md` for the recurring traps.

## Layout

`src/scriber/` — the installable package (`uv run scriber` invokes `scriber.main:main`).

## File map

| File | Purpose |
| --- | --- |
| `src/scriber/main.py` | Entry point. Parses args, initializes logger, loads settings, loops over `args.input_path` (one or more), dispatches per-path to the right handler, and runs the summarizer when `args.command == "summarize"`. Also: preflight LLM key check (`_preflight_summarizer`, which **lazily** imports `scriber.summarizers` so the base install needn't have the `summarize` extra), GPU warning, `--dry-run` report. |
| `src/scriber/handlers.py` | `handle_url` / `handle_media` / `handle_text` + `write_transcript_file(subtitles=bool)` + `summarize`. The actual orchestration. The `summarize` path and the two `--diarize` branches **lazily** import `scriber.summarizers` / `scriber.transcription.diarize` and raise `_SUMMARIZE_EXTRA_HINT` / `_DIARIZE_EXTRA_HINT` on `ImportError` (so the base install runs transcription without either extra). |
| `src/scriber/model.py` | Shared dataclasses (currently just `Transcript`) — kept import-cycle-free. |
| `src/scriber/formatting.py` | `sanitize_filename`, `wrap_transcript` — text helpers shared by handlers. |
| `src/scriber/language.py` | `derive_summary_language` + `derive_whisper_summary_language`. Pure functions implementing the language-selection ladder (see README). |
| `src/scriber/subtitles.py` | `write_srt` / `write_vtt` from whisper-style segment dicts. Used by `--subtitles`. |
| `src/scriber/summarizers/` | Pluggable summarization backends: `OpenAISummarizer`, `OpenRouterSummarizer`, `RagSummarizer` behind a shared `Summarizer` Protocol; `make_summarizer(settings)` factory; `MissingAPIKeyError` for preflight; `analyze_sentiment`. `modes.py` carries the `meeting`/`source`/`auto` prompts + autodetect heuristic. `markdown.py` formats output; `engine.py` is the RAG chain. |
| `src/scriber/parser.py` | argparse with `transcribe` / `summarize` subparsers. Shared flags live on both; `--with-openai` / `--llm-provider` / `--llm-model` / `--summary-mode` are summarize-only. `classify_input(path)` helper returns `{is_url, is_file, is_media_file, is_text_file}` per path (called by `main.py` per-input and eagerly in `parse_args()` for validation). |
| `src/scriber/logger.py` | Custom logging: `ColorFormatter`, `MyJSONFormatter`, `NonErrorFilter`, `install_excepthook`. |
| `src/scriber/logger_config.yaml` | dictConfig YAML. Handlers: stdout (non-errors), stderr (WARNING+), rotating `logs/scriber.log`. |
| `src/scriber/constants.py` | Section schemas per mode (`MEETING_SECTION_KEYS/LABELS/HEADERS`, `SOURCE_SECTION_*`, combined lookups `SECTION_KEYS/LABELS/HEADERS`) + polarity thresholds. Prompts moved into `summarizers/modes.py`. |
| `src/scriber/settings.py` | Frozen `Settings` dataclass + stdlib `.env` loader (replaces the old `python-dotenv` dep). |
| `src/scriber/transcription/youtube_captions.py` | yt-dlp-backed YouTube caption fetch. Picks manual > auto across `["fr", "en"]`. Raises `TranscriptUnavailableError` on failure. |
| `src/scriber/transcription/youtube_audio.py` | yt-dlp-based audio download + video-id extraction + title metadata (used for the captionless-video fallback path). Smart-caches: returns existing `.wav` unless `force=True`. |
| `src/scriber/transcription/local.py` | ffmpeg → whisper transcription (whisper engine only; **no** pyannote/torchaudio imports). Module-level `_MODEL_CACHE` avoids reloading whisper across calls. `transcribe_audio_full` is the primary entry point (returns text + lang + segments). Default audio pre-processing via `maybe_preprocess` (alimiter+dynaudnorm; gated by `Settings.preprocess_audio`). Shared engine helpers (`get_device`, `load_model`, `detect_language`, `detect_language_probs` (per-window prob dict for the multi-sample probe), `patch_whisper_progress_bar`, `maybe_preprocess`, `extract_audio`, `transcribe_audio_full`) are imported by `diarize.py`. Module constant: `_PREPROCESS_FILTER`. |
| `src/scriber/transcription/diarize.py` | Speaker-diarization path (pyannote 4.x) — the `diarize` extra. **Transcribe-then-assign** (WhisperX-style): one full `transcribe_audio_full` pass produces timestamped segments, pyannote produces speaker turns, then `assign_speakers_to_segments` (via `_best_speaker`: max-overlap, else nearest turn within `_MAX_ASSIGN_GAP`, else drop — which sheds music/silence) labels each segment; `format_diarized` renders `SPEAKER_XX: …` runs. Replaced the old slow per-turn `model.transcribe` loop. Diarization runs on the **raw** (un-preprocessed) audio — dynaudnorm blurs speaker embeddings. `detect_language_from_speech` samples up to `_LID_MAX_WINDOWS` speaker turns (spread via `_evenly_spaced`) and sums Whisper probs, so a music/intro prelude no longer fixes the wrong global language. `relabel_by_appearance` maps pyannote's arbitrary cluster ids to `SPEAKER_00/01/…` in speak order. `diarize_speakers(audio, *, min_speakers, max_speakers)` takes a **preloaded waveform** + optional count hints and feeds pyannote a `{"waveform","sample_rate"}` mapping so it never touches **torchcodec** (see below). Other functions: `decode_audio`, `slice_audio` (now used for LID windows), `group_speaker_segments`, `transcribe_audio_with_diarization`, `transcribe_video_file_with_diarization`. Constants: `MIN_SEGMENT_DURATION` / `_MAX_SPEAKER_GAP` / `_MAX_ASSIGN_GAP` / `_SAMPLE_RATE` / `_LID_MAX_WINDOWS` / `_LID_WINDOW_SEC`. Speaker hints flow from `Settings.min_speakers`/`max_speakers` (env `MIN_SPEAKERS`/`MAX_SPEAKERS`, CLI `--min-speakers`/`--max-speakers`). Imported lazily by `handlers.py` so the base install never pulls pyannote. **torchcodec note:** pyannote 4.x / torchaudio 2.11 decode through torchcodec, which fails closed to a warning on Windows when its FFmpeg "full-shared" DLLs are missing or torch/torchcodec versions mismatch — leaving `AudioDecoder` undefined → `NameError` at first use. The in-memory waveform path sidesteps it; we never call `torchaudio.load`/`torchcodec` from app code. |
| `src/scriber/transcription/preprocess.py` | Cleanup + speaker-name heuristics. |

## Dev workflow

- `uv add`, `uv run`, `uv sync` — never `pip`/`venv` directly.
- **Packaging**: base = transcription-only; `diarize` + `summarize` are optional extras (`[project.optional-dependencies]`). torch is a **direct base dep** (whisper needs it; declared direct so `[tool.uv.sources]` can bind to it — uv sources only apply to direct deps). torch backend is **not** globally CPU-pinned: Linux/macOS resolve from PyPI (GPU-capable on Linux); lean CPU consumers select the backend at install time via `UV_TORCH_BACKEND=cpu` (see `docs/BDCOS_INSTALL.md`). **Windows is the exception**: PyPI ships CPU-only torch there, so `[tool.uv.sources]` pins win32 `torch`/`torchaudio` to the PyTorch CUDA index (`[[tool.uv.index]] pytorch-cuda` = cu126; same 2.12.x/2.11.0 versions as PyPI so only the build differs, runs on a CUDA ≥12.x driver via minor-version compat). The win32 marker keeps Linux/macOS resolution untouched. Dev env needs all extras: `just sync` (= `uv sync --all-extras`); the `just` recipes run with `--all-extras`.
- `just sync`, `just lint`, `just typecheck`, `just test`, `just all`.
- `pre-commit` runs ruff + pyright + pytest on `git commit` once installed (`uv run pre-commit install`).
- `pytest -m integration` runs opt-in ML tests (whisper / pyannote). They need a fixture at `tests/integration/data/hello.wav` (see the test module's docstring) and download whisper models on first run. Skipped by default.
- Do not auto-commit.
- Keep changes minimal and focused.

## Known state

- **App modules + tests are ruff-ALL + pyright-strict clean.** `just all` runs green. Boundary with untyped ML deps (`whisper`, `pyannote`, `torchaudio`, `ffmpeg`, `yt-dlp`, `langchain`, `chromadb`) is handled via file-level `# pyright: reportUnknown... = false` headers in `src/scriber/transcription/local.py`, `src/scriber/transcription/diarize.py`, and `src/scriber/transcription/youtube_audio.py`, and explicit `cast(Any, ...)` at call sites elsewhere. Apply `.claude/rules/python_strict.md` patterns when extending.
- `tests/test_base_install_imports.py` is the **packaging guard**: it blocks the optional-extra deps at import time and asserts the transcribe path still imports — it fails if an eager `pyannote`/`torchaudio`/`langchain`/`openai`/`chromadb`/`textblob` import sneaks onto the base path.
- `results/`, `downloads/`, and `chroma_db/` are runtime outputs (gitignored).
- `.env` holds `OPENAI_API_KEY`, optionally `HUGGINGFACE_TOKEN` (for diarization), and any `LOG_LEVEL` override.
- Improvement plan lives at `/home/mprz/.claude/plans/ok-now-that-we-inherited-pascal.md`. In-flight work is tracked there; `TODO.md` is the grooming backlog.
