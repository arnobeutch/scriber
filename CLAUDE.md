# scriber — Claude Reference

## Project

CLI tool to transcribe and summarize YouTube videos, local audio/video files, or pre-existing text transcripts. Two subcommands: `scriber transcribe <input>...` (transcript only) and `scriber summarize <input>...` (transcribe + summarize). Pipeline: fetch/transcribe → optional diarization → sentiment/polarity → LLM summary (OpenAI API, OpenRouter, or local RAG via langchain + Ollama).

## Tech stack

- Python `==3.11.9` (pinned — `openai-whisper`, `pyannote-audio`, `torchaudio` block 3.12+).
- `uv` for dependency/env management.
- `ruff` lint (`select = ["ALL"]`; ignores in `pyproject.toml`, `target-version = "py311"`).
- `pyright` strict mode (`[tool.pyright]` in `pyproject.toml`, `pythonVersion = "3.11"`) — matches VSCode Pylance.
- `.claude/hooks/python-quality.sh` runs after every `Edit`/`Write` on a `.py` file. If it exits 2, read the stderr diagnostics and fix before continuing. See `.claude/rules/python_strict.md` for the recurring traps.

## Layout

`src/scriber/` — the installable package (`uv run scriber` invokes `scriber.main:main`).

## File map

| File | Purpose |
| --- | --- |
| `src/scriber/main.py` | Entry point. Parses args, initializes logger, loads settings, loops over `args.input_path` (one or more), dispatches per-path to the right handler, and runs the summarizer when `args.command == "summarize"`. Also: preflight LLM key check, GPU warning, `--dry-run` report. |
| `src/scriber/handlers.py` | `handle_url` / `handle_media` / `handle_text` + `write_transcript_file(subtitles=bool)` + `summarize`. The actual orchestration. |
| `src/scriber/model.py` | Shared dataclasses (currently just `Transcript`) — kept import-cycle-free. |
| `src/scriber/formatting.py` | `sanitize_filename`, `wrap_transcript` — text helpers shared by handlers. |
| `src/scriber/language.py` | `derive_summary_language` + `derive_whisper_summary_language`. Pure functions implementing the language-selection ladder (see README). |
| `src/scriber/subtitles.py` | `write_srt` / `write_vtt` from whisper-style segment dicts. Used by `--subtitles`. |
| `src/scriber/summarizers/` | Pluggable summarization backends: `OpenAISummarizer`, `OpenRouterSummarizer`, `RagSummarizer` behind a shared `Summarizer` Protocol; `make_summarizer(settings)` factory; `MissingAPIKeyError` for preflight; `analyze_sentiment`. `modes.py` carries the `meeting`/`source`/`auto` prompts + autodetect heuristic. `markdown.py` formats output; `engine.py` is the RAG chain. |
| `src/scriber/parser.py` | argparse with `transcribe` / `summarize` subparsers. Shared flags live on both; `--with-openai` / `--llm-provider` / `--llm-model` / `--summary-mode` are summarize-only. `classify_input(path)` helper returns `{is_url, is_file, is_media_file, is_text_file}` per path (called by `main.py` per-input and eagerly in `parse_args()` for validation). |
| `src/scriber/logger.py` | Custom logging: `ColorFormatter`, `MyJSONFormatter`, `NonErrorFilter`, `install_excepthook`. |
| `src/scriber/logger_config.yaml` | dictConfig YAML. Handlers: stdout (non-errors), stderr (WARNING+), rotating `logs/scriber.log`. |
| `src/scriber/constants.py` | Prompts (`OPENAI_PROMPT_EN/FR`, `RAG_FRENCH/ENGLISH_PROMPT`) + `RAG_SECTION_KEYS` / `RAG_SECTION_LABELS` / `RAG_SECTION_HEADERS` + polarity thresholds. |
| `src/scriber/settings.py` | Frozen `Settings` dataclass + stdlib `.env` loader (replaces the old `python-dotenv` dep). |
| `src/scriber/transcription/youtube_captions.py` | yt-dlp-backed YouTube caption fetch. Picks manual > auto across `["fr", "en"]`. Raises `TranscriptUnavailableError` on failure. |
| `src/scriber/transcription/youtube_audio.py` | yt-dlp-based audio download + video-id extraction + title metadata (used for the captionless-video fallback path). Smart-caches: returns existing `.wav` unless `force=True`. |
| `src/scriber/transcription/local.py` | ffmpeg → whisper transcription, optional pyannote diarization. Module-level `_MODEL_CACHE` avoids reloading whisper across calls. `transcribe_audio_full` is the primary entry point (returns text + lang + segments). Module constants: `MIN_SEGMENT_DURATION`, `_MAX_SPEAKER_GAP`. |
| `src/scriber/transcription/preprocess.py` | Cleanup + speaker-name heuristics. |

## Dev workflow

- `uv add`, `uv run`, `uv sync` — never `pip`/`venv` directly.
- `just lint`, `just typecheck`, `just test`, `just all`.
- `pre-commit` runs ruff + pyright + pytest on `git commit` once installed (`uv run pre-commit install`).
- `pytest -m integration` runs opt-in ML tests (whisper / pyannote). They need a fixture at `tests/integration/data/hello.wav` (see the test module's docstring) and download whisper models on first run. Skipped by default.
- Do not auto-commit.
- Keep changes minimal and focused.

## Known state

- **App modules + tests are ruff-ALL + pyright-strict clean.** `just all` runs green (271 tests, 2 deselected integration). Boundary with untyped ML deps (`whisper`, `pyannote`, `torchaudio`, `ffmpeg`, `yt-dlp`, `langchain`, `chromadb`) is handled via file-level `# pyright: reportUnknown... = false` headers in `src/scriber/transcription/local.py` and `src/scriber/transcription/youtube_audio.py`, and explicit `cast(Any, ...)` at call sites elsewhere. Apply `.claude/rules/python_strict.md` patterns when extending.
- `results/`, `downloads/`, and `chroma_db/` are runtime outputs (gitignored).
- `.env` holds `OPENAI_API_KEY`, optionally `HUGGINGFACE_TOKEN` (for diarization), and any `LOG_LEVEL` override.
- Improvement plan lives at `/home/mprz/.claude/plans/ok-now-that-we-inherited-pascal.md`. In-flight work is tracked there; `TODO.md` is the grooming backlog.
