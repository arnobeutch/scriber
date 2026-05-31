# scriber

Transcribe and summarize YouTube videos, local audio/video files, or existing text transcripts.

## Features

- `yt-dlp` for both YouTube captions (en / fr; manual preferred over auto) and audio download for the captionless-video fallback.
- `whisper` transcription for local media (`ffmpeg-python` for audio extraction).
- Optional speaker diarization via `pyannote-audio`.
- `textblob` sentiment/polarity (applied to every summary, regardless of backend).
- Pluggable LLM backends: **OpenAI** (default), **OpenRouter** (Minimax, Kimi, Claude, Gemini, …), **Ollama** (local RAG via `langchain` + `langchain-ollama` + `chromadb`).
- Markdown-formatted summary output.

## Usage

Two subcommands:

```bash
uv run scriber transcribe <url | path> [url | path ...] [options]  # transcript only
uv run scriber summarize  <url | path> [url | path ...] [options]  # transcribe + summarize
```

Multiple inputs are processed sequentially in one invocation:

```bash
uv run scriber summarize  https://www.youtube.com/watch?v=VIDEO_ID --with-openai
uv run scriber summarize  ./my_meeting.mp4 --diarize
uv run scriber summarize  ./existing_transcript.txt
uv run scriber transcribe ./my_meeting.mp4 --diarize --subtitles
uv run scriber summarize  https://youtu.be/X https://youtu.be/Y ./local.mp4   # batch
```

### Options

Flags shared by `transcribe` and `summarize`:

| Flag | Description |
| --- | --- |
| `-l`, `--language` | `en` or `fr`. Default: autodetect. Used as a *hint* for caption-track selection and to force whisper's transcription language. The summary always tracks the source's language (English fallback for anything other than en/fr). |
| `--diarize` | Identify speakers when transcribing local media (default: False). |
| `--model-size` | Whisper model: `tiny`, `base`, `small`, `medium`, `large`, `large-v3`, `large-v3-turbo`. Default from `WHISPER_MODEL_SIZE` env or `large-v3-turbo` (best WER among CPU-practical models; faster than `medium` on the same hardware). Pick `small` for a speed-first draft; `medium` is functionally obsolete. |
| `--output-dir` | Where outputs land. Default from `OUTPUT_DIR` env or `./results`. |
| `--downloads-dir` | Where downloaded YT audio is cached. Default from `DOWNLOADS_DIR` env or `./downloads`. |
| `--force` | Re-download audio and re-transcribe even when a cached `.wav` or transcript already exists. |
| `--subtitles` | Also write `.srt` and `.vtt` subtitle files alongside the `.txt` transcript (whisper transcription only — YT captions and diarized output don't carry per-cue timestamps). |
| `--dry-run` | Print what the pipeline would do (input type, model, output dir) without doing any work. |
| `-d`, `--debug` | Enable DEBUG-level logging (default: False). |

Additional flags for `summarize` only:

| Flag | Description |
| --- | --- |
| `--with-openai` | Shortcut for `--llm-provider openai` (default: False). `--with_openai` still works as a legacy alias. |
| `--llm-provider` | `openai`, `openrouter`, `ollama`. Default from `LLM_PROVIDER` env or `openai`. |
| `--llm-model` | Model name for the chosen provider. Default from `LLM_MODEL` env or per-provider default. |
| `--summary-mode` | `meeting` (multi-speaker discussion), `source` (lecture / article / commentary — tags facts vs opinion vs speculation), or `auto` (heuristic). Default from `SUMMARY_MODE` env or `auto`. |

### Caching

Re-running with the same input is fast: the YT audio is reused from `./downloads/<id>.wav` if present, and the whisper transcript is reused from `./results/<title> [diarized] transcript.txt` if present. Pass `--force` to bypass both caches.

### Summary modes

- **`meeting`** — produces a structured summary tailored to discussions: topic, hashtags, takeaways (attributed to speakers), Q&A, decisions, action items.
- **`source`** — produces an evidence-aware summary tailored to a single source (interview, lecture, article reading): TL;DR, key takeaways, **facts** vs **opinions** vs **speculation**, counterpoints / alternatives, and an overall information-quality / reliability rating.
- **`auto`** — picks `meeting` when the transcript is diarized with 2+ distinct speakers; otherwise picks `source`. Logs the choice.

### Language selection

The summary follows the source's language; `--language` is a preference hint, not a hard override.

For YouTube URLs the caption track is picked top-down (manual beats auto across languages). When `--language` isn't set, scriber substitutes the uploader-declared video language (yt-dlp's `info["language"]`) as the implicit preference, so a French video that ships an English subtitle track doesn't accidentally yield an English transcript.

1. Manual captions in `--language` (if set) — or in the declared video language when `--language` is unset
2. Manual captions in English
3. Manual captions in any other language
4. Auto captions in `--language` (if set) — or declared video language when `--language` is unset
5. Auto captions in English
6. Auto captions in any other language
7. *No captions* → fall back to whisper

Once a track is picked, the summary language is derived:

- caption is in the effective preference (CLI flag or declared lang) or English → summary in that language
- caption is in some other language → summary in **English** (translated by the LLM)

When whisper transcribes (no captions available, or local media):

- `--language` set → whisper is forced to that language; summary in that language
- otherwise → whisper auto-detects; if detected ∈ {en, fr} → summary in detected, else → summary in English

For a pre-existing text file (`.txt`, `.srt`, `.vtt`):

- `--language` set → respected
- otherwise → `langdetect` → same en/fr/else rule as whisper

## Configuration

Runtime settings are loaded by `Settings.from_env()` (reads `.env` + `os.environ`; shell env wins over `.env`). Copy `.env.example` to `.env` and adjust. `.env` is gitignored.

| Env var | Default | Purpose |
| --- | --- | --- |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. `-d`/`--debug` forces DEBUG. |
| `OPENAI_API_KEY` | — | Required for `--with-openai` (or `--llm-provider openai`). |
| `OPENROUTER_API_KEY` | — | Required for `--llm-provider openrouter`. |
| `HUGGINGFACE_TOKEN` | — | Required for `--diarize` (pyannote speaker-diarization-3.1 + voice-activity-detection gated models). |
| `LLM_PROVIDER` | `openai` | One of `openai`, `openrouter`, `ollama` (CLI flag overrides). |
| `LLM_MODEL` | provider default | E.g. `gpt-4o`, `anthropic/claude-4.7-sonnet`, `mistral` (CLI flag overrides). |
| `OPENAI_MODEL` | `gpt-4o` | Model for the OpenAI provider. |
| `OLLAMA_MODEL` | `mistral` | Model for the local Ollama/RAG provider. |
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | `tiny`, `base`, `small`, `medium`, `large`, `large-v3`, `large-v3-turbo`. |
| `OUTPUT_DIR` | `results` | Where transcripts and summaries land. |
| `DOWNLOADS_DIR` | `downloads` | Where downloaded YT audio is cached. |
| `WRAP_WIDTH` | `80` | Soft-wrap width for non-diarized transcripts (words are never split). |

## Logging

Configured via `src/scriber/logger_config.yaml`. Handlers:

| Handler | Stream | Level |
| --- | --- | --- |
| stdout | stdout | INFO/DEBUG (non-errors) |
| stderr | stderr | WARNING+ |
| file | `logs/scriber.log` | DEBUG+ (rotating) |

Uncomment `- json_file` under `root.handlers` in `src/scriber/logger_config.yaml` to also emit structured JSON to `logs/scriber.jsonl`. Uncaught exceptions are routed through the logger via `install_excepthook()`.

## Dev workflow

All quality commands run through `just`:

| Command | Does |
| --- | --- |
| `just` | List recipes |
| `just lint` | `ruff check` + `ruff format --check` |
| `just format` | `ruff format` + `ruff check --fix` |
| `just typecheck` | `uv run pyright` |
| `just test` | `uv run pytest` |
| `just all` | `lint` + `typecheck` + `test` |

Install the pre-commit gate once per clone:

```bash
uv run pre-commit install
```

`git commit` will then run ruff + pyright + pytest on staged Python files.

## Roadmap

Grooming backlog with completed items: [TODO.md](TODO.md).
