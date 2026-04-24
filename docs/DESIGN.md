# scriber — Design

Internal reference describing how scriber is wired together. Aimed at a
contributor who has read the README and the CLAUDE.md file map, and now
needs to know *why* the pieces are shaped the way they are.

---

## 1. Purpose & scope

scriber is a one-shot CLI that turns a YouTube URL, a local media file,
or a pre-existing text transcript into a transcript file and
(optionally) a structured markdown summary. Its charter is deliberately
narrow:

- **In scope:** single-user command-line usage, batch of N inputs per
  invocation, file-based caching, per-run transcripts and summaries
  written to disk.
- **Out of scope:** daemons or long-running services, a database or
  vector-index layer beyond the ephemeral Chroma store, multi-user
  concurrency, a web UI, an API surface. If a feature would require a
  service boundary, it doesn't belong here.

Everything the pipeline does is a composition of (a) transcription, (b)
optional diarization, (c) optional summarization via a pluggable LLM
backend. The CLI is two subcommands — `transcribe` and `summarize` —
that share the transcription half and diverge in the summarization half.

---

## 2. Architecture at a glance

```
CLI args ─▶ parser.parse_args ─▶ classify_input (per path)
                                      │
                                      ▼
Settings.from_env + CLI overrides ─▶ main._process_one
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
      handle_url (YT URL)    handle_media (file)    handle_text (file)
                │                     │                     │
                └──────── Transcript dataclass ─────────────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
      write_transcript_file   (optional) handlers.summarize
      [.txt + .srt/.vtt]               │
                                       ▼
                              make_summarizer(settings)
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
           OpenAISummarizer   OpenRouterSummarizer    RagSummarizer
                  │                    │                    │
                  └────────────────────┴────────────────────┘
                                       ▼
                        format_summary_markdown
                         [{title} - summary.md]
```

The central in-memory hand-off is the `Transcript` dataclass. Every
handler produces one; every backend consumes one. That's the system's
waist.

---

## 3. Module boundaries

| Module | Role | Imports from |
|---|---|---|
| `main.py` | Top-level orchestration, batch loop, dry-run, GPU warning, preflight, exit code. | `parser`, `handlers`, `settings`, `summarizers`, `logger` |
| `parser.py` | `argparse` layout; input classification (URL / media / text). | stdlib only |
| `settings.py` | Frozen `Settings` dataclass, `.env` loader, `from_env()`. | stdlib only |
| `handlers.py` | Per-input-type handlers (`handle_url`, `handle_media`, `handle_text`), transcript file writer, summarize dispatcher. | `transcription.*`, `summarizers`, `language`, `formatting`, `subtitles`, `model` |
| `model.py` | Shared dataclasses (`Transcript`, `Chapter`). No business logic. | stdlib only |
| `language.py` | Pure rules for summary-language selection. | none |
| `formatting.py` | `sanitize_filename`, `wrap_transcript`. Text helpers. | stdlib only |
| `subtitles.py` | `.srt` / `.vtt` writers from whisper segments. | stdlib only |
| `logger.py` | dictConfig-driven logger with color + JSON handlers, excepthook install. | stdlib + `yaml` |
| `constants.py` | Section schemas (`SECTION_KEYS/LABELS/HEADERS` per mode), polarity thresholds. | stdlib only |
| `transcription/youtube_captions.py` | `yt-dlp` caption fetch; `CaptionTrack`; `TranscriptUnavailableError`. | `yt-dlp` |
| `transcription/youtube_audio.py` | `yt-dlp` audio download, video-id extraction, metadata (title + chapters). Smart-caches `.wav`. | `yt-dlp`, `model` |
| `transcription/local.py` | `ffmpeg → whisper` pipeline, pyannote diarization, in-process whisper model cache. | `whisper`, `pyannote`, `torchaudio`, `ffmpeg`, `tqdm` |
| `transcription/preprocess.py` | Raw-transcript → `(speaker, text)` utterances; speaker-name resolution heuristic. | `unidecode` |
| `summarizers/base.py` | `Summarizer` Protocol, `make_summarizer` factory, `analyze_sentiment`, `MissingAPIKeyError`. | `textblob`, stdlib |
| `summarizers/modes.py` | `meeting` / `source` prompt templates, `get_prompt(mode, language, context)`, auto-detect heuristic. | stdlib only |
| `summarizers/markdown.py` | `format_summary_markdown` (the structured-sections formatter), `extract_sections`, chapter deep-links. | `constants`, `model`, `logger` |
| `summarizers/openai_compatible.py` | Shared OpenAI-protocol base: streaming Chat Completions, `_consume_stream` to stdout. | `openai`, `markdown`, `modes`, `base` |
| `summarizers/openai_summarizer.py` | Default `OpenAI` backend (thin subclass). | `openai_compatible` |
| `summarizers/openrouter.py` | OpenRouter backend (OpenAI-compatible at a different `base_url`). | `openai_compatible` |
| `summarizers/rag.py` | Local Ollama RAG backend. | `engine`, `markdown`, `modes`, `base`, `preprocess` |
| `summarizers/engine.py` | Chroma vectorstore + `RetrievalQA`; greedy `pack_utterances` chunker. | `langchain`, `langchain-chroma`, `langchain-ollama` |

The direction of dependency is top-down: `main → handlers →
(transcription | summarizers) → model / constants / logger`. Nothing in
`transcription/*` knows about summarizers; nothing in `summarizers/*`
knows about handlers. `model.py` and `constants.py` are the shared
foundations, consumed from both halves but importing neither.

---

## 4. The `Transcript` dataclass

`model.Transcript` is the single hand-off between the transcription half
and the summarization half:

```python
@dataclass(frozen=True)
class Transcript:
    text: str                              # transcript body
    language: str                          # summary language (post-ladder)
    title: str                             # sanitized filename stem
    source: TranscriptSource               # yt_manual | yt_auto | whisper | file
    diarized: bool
    segments: list[dict[str, Any]]         # whisper per-cue data for .srt/.vtt
    chapters: list[Chapter]                # yt-dlp chapters; empty for non-YT
```

Rationale:

- **Frozen**: Transcripts are values, not state. Handlers produce one
  and don't mutate. Summarizers and writers consume and never mutate.
- **`language` means the summary language**, not the detected source
  language. The mapping is done by the language ladder (§8) in the
  handler, so downstream code never has to redo it.
- **`segments` and `chapters` are optional payloads**. `segments` is
  whisper's per-cue list (empty for YT captions and diarized output,
  since neither surfaces per-cue data the writer can use). `chapters`
  is populated only when yt-dlp returns them.
- **`source` is the provenance tag**. It's what lets the auto-mode
  detector distinguish a diarized whisper transcript (likely a meeting)
  from a caption track (almost never a meeting).

The dataclass lives in its own module because both `handlers.py` and
the summarizer package need it, and we don't want an import cycle.

---

## 5. Summarizer Protocol + backends

`summarizers/base.py` defines the contract:

```python
class Summarizer(Protocol):
    def summarize(
        self,
        transcript: Transcript,
        *,
        input_path: str,
        context: str | None = None,
    ) -> Path | None:
```

Three backends implement it today:

- **`OpenAISummarizer`** — default; uses Chat Completions against
  OpenAI's API at the default `base_url`.
- **`OpenRouterSummarizer`** — same protocol, different `base_url`; lets
  the user point at `minimax/minimax-2.7`, `anthropic/claude-4.7-sonnet`,
  etc. Both inherit from `OpenAICompatibleSummarizer`.
- **`RagSummarizer`** — local Ollama via LangChain's `RetrievalQA`;
  packs utterances into ~500-char chunks, indexes in Chroma, retrieves,
  stuffs into the prompt.

`make_summarizer(settings)` is the factory. It **preflights** the
configured backend by checking for required credentials and raises
`MissingAPIKeyError` if they're missing — this runs before the slow
transcription pipeline, so a bad key fails in seconds, not after a
10-minute whisper run.

### Adding a backend

1. Implement the `Summarizer` protocol (synchronous `summarize` that
   writes a markdown file and returns its path).
2. Wire it into `make_summarizer` under a new `llm_provider` value.
3. If the backend needs credentials, raise `MissingAPIKeyError` from the
   factory when they're missing.
4. Reuse `format_summary_markdown` to keep the output contract stable.
5. Reuse `get_prompt(mode, language, context)` for the prompt.

No need to touch `handlers.py` or `main.py`.

---

## 6. Mode + section schemas

There are two summary modes: **meeting** and **source**. Each has a
distinct prompt template and a distinct section schema.

### Three-table layout (per mode)

```python
SECTION_KEYS["meeting"]  == ("topic", "hashtags", "takeaways", ...)
SECTION_LABELS["meeting"]["en"]["topic"] == "Topic"          # regex target
SECTION_LABELS["meeting"]["fr"]["topic"] == "Sujet"
SECTION_HEADERS["meeting"]["en"]["topic"] == "## Meeting Topic"  # rendered
SECTION_HEADERS["meeting"]["fr"]["topic"] == "## Sujet de la réunion"
```

- `KEYS` is the canonical, language-agnostic list of sections in render
  order.
- `LABELS` is what the prompt asks the model to emit as a section
  heading — i.e. what the regex extractor matches against.
- `HEADERS` is what the formatter writes into the final markdown.

**Why three tables?** Early versions conflated "regex target" with
"output header" with a single French-keyed dict that was only correct
for FR output. EN-RAG summaries parsed empty silently. Splitting the
concerns gives us (a) neutral keys that travel with the code, (b) the
ability to match different language labels without re-keying the whole
pipeline, (c) freedom to pick prettier rendered headers than the model's
raw labels (e.g. `## Meeting Topic` vs the model's plain `Topic:`).

### Auto-detect

`modes.detect_mode(transcript)` picks between `meeting` and `source`:

1. Diarized transcript with ≥ 2 distinct speakers → `meeting`.
2. Otherwise → `source`.

A language-opinion heuristic is present in the code but essentially no-ops
today (both dense and sparse opinion content lands in `source`).

### Adding a mode

1. Add a key to `SECTION_KEYS`, `SECTION_LABELS`, `SECTION_HEADERS`.
2. Add an entry in `SECTION_KEYS[mode]` keyed by the new mode name.
3. Add a template + phrase dict in `summarizers/modes.py`; extend
   `get_prompt` to dispatch.
4. Extend `_TITLE_PREFIXES` in `summarizers/markdown.py`.
5. If it should participate in `auto`, extend `detect_mode`.
6. Extend `SummaryMode` / `ResolvedMode` Literals.

---

## 7. Output contract

### Filenames (relative to `settings.output_dir`)

| Kind | Pattern |
|---|---|
| Non-diarized transcript | `{title} transcript.txt` |
| Diarized transcript | `{title} diarized transcript.txt` |
| Subtitles (whisper only) | `{title}.srt` / `{title}.vtt` |
| Summary (EN) | `{title} - summary.md` |
| Summary (FR) | `{title} - résumé.md` |

`{title}` is always the sanitized filename stem. Summary files ending in
` - summary.md` / ` - résumé.md` are the convention both backends now
follow; downstream tooling can pattern-match on the suffix.

### Markdown structure

```
# {Meeting Summary | Résumé de la réunion | Summary | Résumé} — {title}

> Source: {input_path}              (omitted when missing)

## Chapters                          (omitted when chapters = [])
- [MM:SS](url?t=ss) chapter title
- [MM:SS](url?t=ss) chapter title
...

## <Section 1 header>
<content or localized default>

## <Section 2 header>
<content or localized default>
...

## Sentiment                         (omitted when sentiment = None)
{Positive | Neutral | Negative}
```

Section headers come from `SECTION_HEADERS[mode][language]` and render in
`SECTION_KEYS[mode]` order. Missing content is filled with `None` (EN) or
`Aucune` (FR) rather than leaving an empty section — downstream readers
can rely on every key being present when the schema says it should be.

---

## 8. Language ladder

`language.py` holds pure functions; handlers call them at the point
where a transcript is produced.

```
┌─ captions (yt_manual / yt_auto) ────────────────────────────┐
│  derive_summary_language(caption_lang, requested_lang)       │
│    if requested_lang == caption_lang   → requested_lang      │
│    if caption_lang == "en"             → "en"                │
│    else                                → "en"   (forced)     │
└──────────────────────────────────────────────────────────────┘

┌─ whisper output ─────────────────────────────────────────────┐
│  derive_whisper_summary_language(detected_lang, requested)   │
│    if requested                → requested   (whisper forced) │
│    if detected in {"en","fr"}  → detected                     │
│    else                        → "en"   (forced)              │
└──────────────────────────────────────────────────────────────┘

┌─ text file ─────────────────────────────────────────────────┐
│  same ladder as whisper, feeding langdetect output in.       │
└──────────────────────────────────────────────────────────────┘
```

EN is always the safe fallback. The prompt templates are only written
for EN / FR; `get_prompt` raises `ValueError` for anything else, which
is why the ladder has to collapse other languages to EN.

`--language` on the CLI is a **preference hint**, not a hard override.
It biases caption-track selection and forces whisper's input language,
but the summary language is still derived from what was actually
transcribed.

---

## 9. Caching & `--force`

Three caches layered in the transcription half:

1. **Downloaded YT audio** at `downloads/<video_id>.wav`. Skipped
   download if the file exists. `--force` bypasses.
2. **Transcript text** at `{output_dir}/{title} [diarized] transcript.txt`.
   When `handle_url` takes the whisper-fallback branch and this file
   exists, whisper is skipped and the file content is returned as the
   Transcript. `--force` bypasses.
3. **Whisper model** in-process, keyed by `(model_size, device)` in
   `_MODEL_CACHE`. Not user-facing — just avoids re-loading the 80MB+
   model when processing multiple inputs in one run.

Caches at the transcription layer only. Summaries are always regenerated
— they're fast relative to transcription, and the user typically wants
the latest when they invoke `summarize`.

---

## 10. Error handling & batch resilience

`main._process_one(path, ...)` runs one input through the pipeline and
returns an `_InputResult`. The top-level loop collects these:

```python
for path in args.input_path:
    try:
        results.append(_process_one(...))
    except Exception as exc:
        if not args.continue_on_error:
            raise          # default: abort on first failure
        results.append(_InputResult(path, "error", str(exc)))
```

Contract:

- **Default** (no flag): first failure aborts the batch. Matches
  intuition for single-input runs.
- **`--continue-on-error`**: failures are logged, the batch continues,
  and `main()` exits non-zero if *any* input failed — so CI can't miss
  a partial failure.
- `_print_batch_summary` emits a ✓/✗ table only when there's more than
  one input.

Narrower error paths inside the pipeline:

- **`TranscriptUnavailableError`** in the YT caption fetch → falls
  through to whisper audio download in `handle_url`. Not a user-facing
  error.
- **`MissingAPIKeyError`** from the summarizer factory → preflighted
  before any transcription, `sys.exit(2)` with a pointed message.
- **`openai.AuthenticationError` / `APITimeoutError` / `OpenAIError`**
  in the streaming call → logged and the summary is skipped; no summary
  file is written.

---

## 11. Config surface

`Settings` is a frozen dataclass populated by `Settings.from_env()`
(reads `.env` first, then `os.environ`, with shell env winning). CLI
flags layer over env via `main._apply_cli_overrides` using
`dataclasses.replace` — the result is a fresh `Settings` with CLI
fields overlaid, still frozen, still passed through everywhere.

Precedence for every configurable value:

```
CLI flag > env var > .env file > Settings default
```

That single ordering applies to `output_dir`, `downloads_dir`,
`whisper_model_size`, `llm_provider`, `llm_model`, `summary_mode`, and
`log_level`. Adding a new configurable value means:

1. Add a field to `Settings` with a default.
2. Map it in `Settings.from_env`.
3. Add a CLI flag in `parser.py`.
4. Add the overlay in `main._apply_cli_overrides`.

---

## 12. Quality gates

Target versions are pinned and checked:

- **Python `==3.11.9`**. `openai-whisper`, `pyannote-audio`, and
  `torchaudio` block 3.12+ today. Revisit when those lift.
- **`ruff` with `select = ["ALL"]`** plus an explicit ignore list in
  `pyproject.toml` (see `.claude/rules/python_strict.md` for rationale).
- **`pyright` strict mode** (`[tool.pyright]` in `pyproject.toml`).
  Untyped ML boundaries are isolated via file-level
  `# pyright: reportUnknown... = false` in `transcription/local.py` and
  `transcription/youtube_audio.py`, with explicit `cast(Any, ...)` at
  call sites elsewhere.
- **`pytest`** unit tests. Integration tests marked `@pytest.mark.integration`
  are deselected by default; they download whisper models and need a
  fixture at `tests/integration/data/hello.wav`.

Two enforcement points:

1. **`.claude/hooks/python-quality.sh`** runs on every `Edit`/`Write` of
   a `.py` file. Exit 2 means stop and fix before continuing.
2. **`pre-commit`** runs ruff + pyright + pytest at `git commit` time.
   Install once per clone with `uv run pre-commit install`.

`just all` is the local equivalent of what CI would run.

---

## 13. Known boundaries & deferred

Captured in `TODO.md`; the design rationale lives here:

- **Python 3.13 upgrade** — gated by PyTorch-free whisper (`faster-whisper`,
  `whisper.cpp`) and diarization alternatives. The plan is to evaluate
  `faster-whisper` first since it's a near-drop-in.
- **Eval harness** — before any default-model swap (e.g. moving the
  Ollama default from `mistral` to a Gemma 4 variant), we need reference
  `(transcript, expected-sections)` pairs with a cosine-similarity smoke
  test. Without it, default swaps are unreviewed.
- **Packaged binary** — deferred behind feature-flag extras
  (`[summarize]`, `[diarize]`, `[openrouter]`). PyInstaller `--onefile`
  of the minimal build (whisper + yt-dlp + ffmpeg) is the intended
  target.
- **Speaker identification at summarization time** for YT transcripts
  — currently only speaker IDs survive; names would need a diarization
  pass on the downloaded audio layered onto caption timestamps.
- **Structured eval for claim-tagging output** — requires labeled
  `(claim, category)` pairs; deferred with the eval harness.

Things scriber deliberately won't try to do:

- Multi-turn conversation or iterative refinement. One invocation, one
  set of outputs.
- Online learning / summary rewriting. Regenerate by re-running.
- A vector-index across runs. Chroma is scoped to a single summary;
  persistence across runs is an artifact, not a feature.

---

## Conventions recap

- `Settings` is frozen. Replace, don't mutate.
- `Transcript` is frozen. Produce, don't edit.
- Section schemas are keyed by neutral identifiers (`topic`, `tldr`, …);
  languages are a rendering concern.
- The quality hook is ground truth; CLI pyright without config is not.
- New features add tests; `just all` stays green before each commit.
