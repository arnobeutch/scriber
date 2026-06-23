# TODO

Planned improvements, grouped by priority. Items are unordered within each group unless they have an explicit dependency.

## Near-term

All near-term items cleared on 2026-04-24 — see the Completed log below.

## Medium-term

- **Eval harness.** Even 10 reference `(transcript, expected-sections)` pairs with a cosine-similarity smoke test would make LLM provider/model swaps defensible. Currently there's no way to know if a new default degrades output quality — this is a prerequisite for any default-model swap.
- **Feature-flag driven builds** (`[summarize]`, `[diarize]`, `[openrouter]` extras). Enables the packaged-binary plan cheaply and makes contributor onboarding lighter.
- **"Auto" model-size.** Pick `tiny`/`base`/`small`/`medium` from `(hardware, audio duration)`.
- **Speaker identification at summarization time** for YT transcripts (currently only speaker IDs are shown).
- **Multilingual sources.** Handle inputs that switch languages mid-stream (or carry a foreign-language prelude). Even after speech-based multi-sample language detection (done 2026-06-20), a single global language is still forced on the whole transcribe pass — a genuinely bilingual meeting would mis-transcribe the minority language. Options: per-speaker-turn language detection, or chunked detection with language change-points. Motivated by the 2026-06-20 run (FR talk with an EN music intro detected as EN → Whisper *translated* the entire 2h to English).
- **Music / non-speech detection — heavier pass (optional).** A first cut shipped 2026-06-21: `_is_nonspeech_segment` drops whisper hallucinations over non-speech using whisper's own per-segment signals (`no_speech_prob`+`avg_logprob`, `compression_ratio`) — no extra model/dep. This catches looping credit-spam ("Sous-titrage ST' 501") and repetitive song lyrics, and removed the v1/v2 divergence source. Still optional if it proves insufficient on subtler music: an explicit VAD / audio-tagging pass (`silero-vad`, or YAMNet / PANNs / CLAP) to mark `[music]` regions and gate transcription up front rather than filtering after.

## Deferred research (long-horizon)

- **Near-realtime language-ID study ("Murmure").** Investigate how streaming speech-to-text products (the user cited "Murmure"/"Murmurme" — exact product TBC) achieve continuous, near-realtime language detection. Working hypothesis: a dedicated lightweight LID model run on a sliding window with temporal smoothing / majority voting, VAD-gated to speech. Goal: feed back into a more robust offline detector (and the multilingual work above). Spun off from the 2026-06-20 mis-detection.
- **Drop-PyTorch / packaging track.** Python is already `>=3.11` (3.11–3.14 verified), so the old "blocked on newer Python" rationale is resolved. The remaining interest is dropping the PyTorch dependency (size, packaged-binary goal). Evaluate:
  - `faster-whisper` (SYSTRAN) — CTranslate2-based, no PyTorch, typically 4× faster on CPU, same model family. Drops PyTorch and unblocks Python 3.13.
  - `whisper.cpp` bindings (e.g. `pywhispercpp`) — ggml quantized; best fit for the packaged-binary goal below.
  - For diarization alternatives (pyannote pulls PyTorch): `simple-diarizer`, `speechbrain`, or making diarization an optional extra.
- **Better local LLM default** for the Ollama backend. Gemma 4 family (released 2026-04-02, obsoletes Gemma 3):
  - E2B (~2.3B effective via Per-Layer Embeddings) — phone-tier, 128K context.
  - **E4B** (~4.5B effective) — laptop CPU sweet spot, 128K context. Strongest candidate for the new default.
  - 26B A4B (MoE, 4B active) — 256K context, reasoning-grade. Good on 32GB+ laptops.
  - 31B dense — workstation-tier. Overkill for laptop.
  - Other candidates to benchmark: Llama 3.3 70B quantized, Qwen 2.5 (7B / 14B), Phi 3.5 Mini (3.8B).
  - Tentative laptop default: `gemma4:e4b`. Verify via Ollama tag list before wiring. **Prerequisite:** eval harness (above).
- **Portable packaged executable** (single binary, no installer). Allowed deps: whisper (or faster-whisper / whisper.cpp), ffmpeg, yt-dlp. **Prerequisite:** feature-flag extras (above). Approaches:
  - PyInstaller `--onefile` with a "minimal" build flavor (drop pyannote + openai + langchain + chromadb; keep whisper + yt-dlp + ffmpeg).
  - Nuitka — smaller binaries, longer build, better runtime. Head-to-head comparison.
  - whisper.cpp + yt-dlp binary + tiny Go/Python wrapper — smallest footprint, biggest implementation cost.
- **Claim-tagging output** in source-mode — tag each claim `{factual | opinion | speculation}` with confidence.
- **Transcript cleanup** — spaCy / nltk cleanup, Levenshtein on phoneme sequences, NER-filtered corrections.
- **`--summary-style {brief,detailed,bullets,prose}`** orthogonal to `--summary-mode`. 4× the prompt surface to maintain; value unclear — keep deferred or drop.

## Dropped / indefinite defer

- **`langdetect` replacement survey.** Candidate was `lingua-py`. No evidence of failures on short transcripts — revisit only if reliability becomes a real problem.

## Completed

- 2026-04-24 — **Batch-mode resilience.** `--continue-on-error` flag + post-run ✓/✗ summary table; non-zero exit when any input failed so CI surfaces failures.
- 2026-04-24 — **Streaming LLM output.** OpenAI/OpenRouter backend uses `stream=True`; tokens echo to stdout as they generate (long summaries no longer look frozen).
- 2026-04-24 — **Progress bar** during whisper transcription (`verbose=False`) and yt-dlp audio download (native progress line re-enabled). Diarized path wraps the per-segment loop in tqdm.
- 2026-04-24 — **`--context-file path.txt`** for the summarize subcommand. Contents are injected as an "Additional context" block right before the `Transcript:` marker. Wired through the Summarizer Protocol via a new `context: str | None` kwarg.
- 2026-04-24 — **YouTube audio diarization.** `--diarize` on a YT URL now skips the caption fetch entirely and always runs whisper + pyannote on the downloaded audio — captions don't carry speaker attribution.
- 2026-04-24 — **Chapter / TOC extraction** from yt-dlp's `chapters` field. `Transcript` carries a `chapters` list; the formatter renders a `## Chapters` section with `?t=<ss>` deep-links when the source is a URL, plain `MM:SS —` lines otherwise.
- 2026-04-24 — **Unified structured-sections output for OpenAI and RAG.** Both backends write the same Obsidian-friendly markdown via `format_summary_markdown`; source mode got its own section schema (TL;DR / Key takeaways / Facts / Opinions / Speculation / Counterpoints / Reliability). `RAG_SECTION_*` renamed `MEETING_SECTION_*`; combined `SECTION_KEYS/LABELS/HEADERS` lookups. Dead `simple_format_markdown` removed.
- 2026-04-24 — **Unified EN/FR prompt templates.** `MEETING_PROMPT_*` and `SOURCE_PROMPT_*` pairs collapsed into one template per mode + a per-language phrase dict. Dead `OPENAI_PROMPT_EN/FR` and `RAG_FRENCH/ENGLISH_PROMPT` removed.
- 2026-04-24 — **Neutral keys for RAG section titles.** Split `RAG_SECTION_TITLES` into `RAG_SECTION_KEYS` (neutral identifiers) + `RAG_SECTION_LABELS` (per-language regex labels) + `RAG_SECTION_HEADERS` (per-language `##` headers). Side-effect: fixed a latent EN-RAG bug where French labels were matched regardless of language.
- 2026-04-24 — **RAG chunking fix.** Replaced the `RecursiveCharacterTextSplitter(500/50)` that produced one-doc-per-utterance with a greedy utterance-packer: preserves speaker boundaries, overlaps whole-utterance tails.
- 2026-04-24 — **CLI split into `scriber transcribe` / `scriber summarize` subcommands; project renamed `yt-summary` → `scriber`.** `--summarize` and `--transcript-only` flags removed.
- 2026-04-24 — Sentiment added to RAG summaries (parity with OpenAI/OpenRouter backends).
- 2026-04-22 — Unit test suite added (271 tests across all tiers; opt-in `integration` marker for whisper / pyannote).
- 2026-04-22 — App modules + tests brought up to ruff-ALL + pyright-strict clean. `just all` runs green.
- 2026-04-22 — YouTube captionless-video fallback (yt-dlp audio download + whisper transcription) when no captions are available; handles `TranscriptsDisabled`, empty XML payloads, and short-link / embed / shorts URL forms.
- 2026-04-22 — Transcripts soft-wrapped at 80 chars without breaking words when written to disk.
- 2026-04-24 — CLI flag `--with-openai` normalized (hyphen canonical; underscore kept as deprecated alias).
- 2026-04-24 — `Settings` promoted to full config dataclass (all keys, single startup read, `.env` + env override).
- 2026-04-24 — `--model-size`, `--llm-provider`, `--llm-model`, `--output-dir`, `--downloads-dir` CLI flags.
- 2026-04-24 — `TranscriptUnavailableError` exception replaces string-typed sentinel returns.
- 2026-04-24 — `youtube_transcript_api` replaced with yt-dlp for captions (dep dropped).
- 2026-04-24 — Per-source handlers extracted from `main.py` (`handlers.py`); `Transcript` dataclass introduced.
- 2026-04-24 — Full language-selection ladder (manual > auto; requested > en > other; see README).
- 2026-04-24 — Pluggable Summarizer Protocol + OpenRouter backend + sentiment everywhere.
- 2026-04-24 — `--summary-mode {meeting,source,auto}` with autodetect heuristic.
- 2026-04-24 — Smart caching: skip download/transcription when outputs exist; `--force` bypasses.
- 2026-04-24 — `--subtitles` (.srt + .vtt from whisper segments); `--transcript-only` (later superseded by the `transcribe` subcommand).
- 2026-04-24 — Whisper model cache in-process (`_MODEL_CACHE` keyed by model+device); `MIN_SEGMENT_DURATION` and `_MAX_SPEAKER_GAP` promoted to module constants; dead wrappers removed.
- 2026-04-24 — Batch mode (`nargs="+"` multiple inputs), `--dry-run`, GPU-not-detected warning, API-key preflight.
- 2026-04-24 — `langchain_community.vectorstores.Chroma` → `langchain-chroma`.
