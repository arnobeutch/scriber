# Whisper transcription setup — what actually matters

A project-agnostic reference for setting up Whisper-based transcription
pipelines. Distilled from the scriber transcription-tuning bench
(57 runs across 3 stages on a 2h51m French meeting recording; full data
in `experiments/transcription-tuning/RESULTS.md`).

**Use this when:** you're standing up Whisper in a new project and want
to skip the "I'll just use defaults" trap. The defaults are tuned for
the wrong scenario (clean studio audio, English, lots of GPU).

**Don't use this for:** the cases where you need streaming
transcription, real-time word-level alignment, or sub-second latency.
Those use-cases warrant different engines (Vosk, AssemblyAI streaming,
Deepgram). This doc assumes batch transcription of recorded files.

---

## TL;DR

Four knobs matter. Tune them once, copy-paste the result everywhere.

```bash
# 1. Audio pre-pass (ffmpeg) — free quality + free CPU savings on clipped audio
ffmpeg -i source.{mp3,wav,m4a,...} \
       -af "alimiter=limit=0.95:level=disabled,dynaudnorm" \
       -ar 16000 -ac 1 -c:a pcm_s16le \
       source.cleaned.wav
```

```python
# 2-4. Model, params, engine
import whisper

model = whisper.load_model("large-v3-turbo")     # 2. Model: large-v3-turbo
result = model.transcribe(
    "source.cleaned.wav",
    language="fr",                                # 3. Params: force the language
    initial_prompt=DOMAIN_PRIMER,                 #          : seed proper nouns
    condition_on_previous_text=False,             #          : prevent cascades
    # temperature: leave the default ladder ON   #          : DO NOT pass temperature=0.0
    fp16=False,                                   #          : on CPU
    verbose=False,
)
print(result["text"])
```

That's it. The rest of this doc is *why* and *when to deviate*.

---

## Knob 1 — Model choice

### Default: `large-v3-turbo`

**Best WER among CPU-practical models. Faster than `medium`.** This is
the single most important update to your defaults if you're still on
`small` or `medium`.

OpenAI shipped `large-v3-turbo` ("turbo" or "large-v3-turbo" depending
on engine) in late 2024: a 4-decoder-layer distillation atop the
full `large-v3` encoder. On CPU it runs faster than `medium` *and*
beats it on WER. `medium` is functionally obsolete.

### When to deviate

| Scenario | Pick | Why |
|---|---|---|
| Real-time draft, throughput matters more than accuracy | `small` | ~5× faster than turbo, ~25% worse WER |
| Absolute best quality, GPU available | `large-v3` | Slight WER edge over turbo, much slower on CPU |
| Edge / mobile, memory-constrained | `tiny` or `base` | Tradeoffs are obvious |
| Long-form English, GPU | `large-v3` or `faster-whisper large-v3-turbo` | Same as above |

### CPU practicality table (5-min audio clip)

| Model | RTF on Ryzen 7 / int8 | 3h audio projection |
|---|---|---|
| `small` | ~2.2× | ~80min |
| `medium` | ~0.9× | ~3h20 |
| `large-v3-turbo` | ~1.1× | **~2h40m** ← sweet spot |
| `large-v3` | ~0.4× | ~7h30m |

RTF = audio_duration / wallclock. Higher = faster.

### Don't bother benchmarking `medium`

It loses to turbo on both axes. Drop it from your decision tree.

---

## Knob 2 — Audio pre-processing

### The recommended filter chain

```
ffmpeg -i input -af "alimiter=limit=0.95:level=disabled,dynaudnorm" \
       -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

### Why each piece

- **`alimiter=limit=0.95:level=disabled`** — caps audio peaks at 0.95 of
  full scale, defeating any intersample-peak clipping the source may have.
  `level=disabled` keeps the under-the-limit signal at original gain.
  Without this, recorders that ship clipped audio (sensitive room mics
  with loud speakers) feed whisper near-square-wave samples, which the
  model recognizes as low-confidence and **retries 6× per segment via
  the temperature fallback ladder**. That's a ~6× CPU bomb you can avoid
  for free.
- **`dynaudnorm`** — per-frame dynamic gain. Flattens disparity between
  a loud in-room speaker and a faint remote participant on a loudspeaker.
  Whisper does its own internal level normalization, but it's a single
  pass, not adaptive. dynaudnorm matters when speaker levels differ by
  10+ dB across the recording.
- **`-ar 16000 -ac 1`** — whisper's native input. If you skip this, whisper
  will resample internally on every call. Cheap to do up-front.
- **`-c:a pcm_s16le`** — uncompressed 16-bit. Avoids re-encoding losses
  if the temp file gets re-read.

### Filters that look attractive but didn't win in the bench

- **`highpass=f=200`** alone — drops HVAC rumble but also drops real
  signal in some recordings. Caused a WER spike (0.62 → 0.89) on one of
  our test samples. Only use it if you've confirmed your specific
  recording environment has dominant low-frequency noise.
- **`loudnorm` (EBU R128)** alone — no measurable win over `dynaudnorm`,
  costs more CPU. Skip unless you have a downstream consumer that
  requires R128 compliance.
- **`noisereduce` (spectral gating)** — promising but Python-only, not an
  ffmpeg filter. We didn't measure it in stage A. Worth trying if your
  baseline + the chain above leaves audible noise in the output.
- **Demucs vocal separation** — too slow on CPU (minutes per 5-min clip)
  for marginal gains on already-speech-dominated recordings. Reserve for
  music-over-speech scenarios.

### Filters worth adding when the bench's chain isn't enough

- **Silero VAD silence trim** — known to defeat `large-v3`'s
  "thanks for watching" silence-hallucination bug. Costs a single
  Python pass; worth wiring if you see hallucinated content during quiet
  stretches.
- **`compand` (compressor/AGC)** — alternative to dynaudnorm with
  finer control. Pick if dynaudnorm is over-compressing your audio.

### Whisper's hidden cost: temperature-fallback storms

This is the most important thing to understand about whisper's runtime
behavior:

When whisper's internal quality checks reject a segment's first decode
(`compression_ratio > 2.4` or `avg_logprob < -1.0`), it re-runs that
segment at `temperature=0.2`, then `0.4`, ... up to `1.0`. **All six
attempts add up to ~6× the per-segment CPU cost.** A 5-min clip that
should take 60s can take 400s if every segment hits fallback.

**Clipped or noisy audio triggers this storm.** A clean ffmpeg pre-pass
prevents it. **This is where the audio pre-processing pays for itself
in CPU time alone, before considering quality.**

---

## Knob 3 — Whisper parameters

### Always set

```python
language="fr"  # or whatever — never let it auto-detect for known-language audio
fp16=False     # on CPU; True only when device="cuda"
verbose=False  # enables the tqdm progress bar (verbose=None silences entirely)
```

**Why force language:** auto-detect runs per 30-second window. On
French with English loanwords ("le power over Ethernet", "le matter
over thread") or vice-versa, the detector flips mid-meeting and the
model decodes the wrong-language window in its second-language mode.
Quality drops noticeably. Force the language; if your audio is
bilingual, run two passes and merge.

### Default-on (DO NOT disable)

```python
# temperature: omit — defaults to (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) fallback ladder
```

**Critical:** if you pass `temperature=0.0`, whisper disables the
fallback ladder entirely. Greedy decode becomes deterministic — and
when it lands on a token loop (e.g. `"thanks for watching"` ×N), there's
no escape. We hit this in stage C of the bench: sample 01 produced
`"tout tout tout tout ..."` × 24, and `initial_prompt` didn't help
because greedy decode produces the exact same output regardless of
prompt influence on later segments.

The fallback ladder costs CPU on noisy audio but saves you from
permanent loops. **Keep it on in production.** Trade for predictable
slightly-higher CPU vs unpredictable loops.

### Recommended-on for long files

```python
condition_on_previous_text=False  # prevent hallucination cascades
```

Whisper's default is `True`: each segment is conditioned on prior
output. On long files (>30min) this can cascade — once whisper
hallucinates a wrong phrase, the next segment is conditioned on that
wrong phrase, and the error compounds for the rest of the file.
Setting to `False` makes each segment independent. Small quality cost
on short clips, big quality win on long ones.

If your files are <10min and quality on a clean test set degrades with
this off, leave it on. Otherwise off.

### Highest-ROI optional param

```python
initial_prompt="""<a French sentence enumerating the proper nouns,
acronyms, and jargon you expect in this audio>"""
```

**−8% WER, −10% CER, free at decode time.** On samples dense with brand
names or technical terms, the gain can reach −40% WER. The prompt acts
as a vocabulary hint: whisper biases its token probabilities toward
words present in the prompt.

Rules of thumb for the primer text:

- Write it as a natural sentence in the audio's language, not a bare
  list. Whisper uses it as decoder context, not as a separate signal.
- Keep it short. Whisper's prompt window is ~224 tokens (~150 words).
  Past that, it gets truncated.
- Include: proper nouns (people, companies, products), domain-specific
  terms, acronyms, units of measure.
- Don't include: filler words, instructions ("transcribe this video"),
  hints about content ("this is a meeting about..."). Whisper doesn't
  follow instructions, it just biases.
- Same prompt for every clip from the same domain (a meeting series, a
  podcast, a series of lectures). Build a per-domain library.

Example primer (project-specific, French construction meeting):

```
Réunion de chantier pour le projet Le Refuge à Lieurey. Participants :
Christophe, Anne, Gilles Lefebvre. Sujets : bardage Wienerberger Terca
Blockstar, menuiseries K•LINE et Technal, RAL 7021, verrière, pergola,
photovoltaïque, pompe à chaleur, VMC Zehnder, domotique ZigBee MQTT
Matter Thread, visiophone, garde-corps, contremarche.
```

**Recurring-meeting workflow:**

1. **Listen to ~5 minutes** of a typical recording from this series; note
   every proper noun + jargon term you hear.
2. **Compose the primer once.** Stash it alongside the project's other
   docs (e.g. inside your Obsidian vault, your Notion workspace, or a
   plain `~/primers/` folder). Use a descriptive filename:
   `whisper_primer_<project>.<lang>.txt`.
3. **Validate** by running a 5-min sample with vs without the primer;
   eyeball that the new words appear correctly.
4. **Reuse** on every subsequent meeting of the same series. Most
   transcription tools accept either a CLI flag (`--initial-prompt-file
   PATH`) or an env var (`INITIAL_PROMPT_FILE=PATH`).
5. **When the project evolves** (new contractor, new product, new
   acronym): edit the primer. Five-minute task.

**Maintenance cost is real but bounded** — 15 minutes once, then 1-2
minutes of edit per quarter. The output transcript becomes
proper-noun-clean, which materially improves downstream summarization,
search, and reviewability.

**Bootstrapping the primer automatically (scriber).** Instead of step 1
(listen + note by hand), run a transcription once with `--suggest-primer`:
scriber writes a `<title> primer.draft.txt` next to the transcript with the
proper nouns, acronyms, and the words whisper was least sure about (it enables
word timestamps to score confidence). Review it — fix spellings, delete noise —
then feed it back via `--initial-prompt-file` for a second pass. Because the
draft keeps its review notes as `#` comment lines (which the loader ignores), it
is usable as-is after trimming. This also makes spellings *consistent*: on a long
file whisper otherwise spells the same name several ways (it runs with
`condition_on_previous_text=False`), and a primer pins one form throughout.

### Skip these unless you have a reason

- `beam_size`, `patience`, `length_penalty` — second-order. Defaults are
  fine.
- `word_timestamps=True` — adds noticeable CPU. Only enable if you
  actually consume per-word timing (subtitle alignment, karaoke).
- `clip_timestamps` — fine-grained windowing; only useful for very long
  files where you want explicit chunking.
- `hallucination_silence_threshold` — added in recent whisper versions.
  Defer until you've ruled out the simpler fixes (audio pre-pass + VAD).

---

## Knob 4 — Engine choice

### Default: `openai-whisper`

The reference implementation. Pip-installable, no extra deps beyond
PyTorch. Use this when:

- You spawn a fresh Python process per transcription (CLI tools)
- Your files are short enough that per-invocation model load isn't a
  dominant cost
- You want the canonical behavior to match upstream documentation

### Alternative: `faster-whisper` (CTranslate2)

Marketed as 2-4× faster. **Bench finding: faster only when you load the
model once and transcribe many files in one process.** For
spawn-per-file usage, model-load + CTranslate2 init dominate; we
measured `faster-whisper small` at *half* the speed of
`openai-whisper small` on 5-min clips (one process per clip).

Use `faster-whisper` when:

- You have a long-running process that loads the model once
- Quality is identical (it usually is — same model files, different
  inference path)
- You want INT8 quantization (default) for lower memory footprint
- You're on a CPU with AVX-VNNI support (Intel 11th gen+, AMD Zen4+)

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
segments, info = model.transcribe(
    audio_path,
    language="fr",
    initial_prompt=DOMAIN_PRIMER,
    condition_on_previous_text=False,
    beam_size=5,
    vad_filter=False,  # set True to use built-in Silero VAD
)
text = " ".join(s.text for s in segments)
```

**`faster-whisper` ships a built-in `vad_filter` option** that runs
Silero VAD before transcription. Worth enabling if you have silent
stretches you want trimmed.

### Other engines worth knowing

- **whisper.cpp** — GGML quantization, smallest memory footprint, no
  PyTorch dep, easy cross-compile (ARM, Apple Silicon, edge). Slower
  than CTranslate2 on x86 but trivially deployable. Use for embedded /
  packaged-binary scenarios.
- **WhisperX** — adds forced-alignment + diarization on top of
  faster-whisper. Use for podcast-style multi-speaker outputs where you
  want speaker labels.
- **Insanely-fast-whisper / Distil-Whisper** — community projects with
  more aggressive distillation. Worth benchmarking against
  `large-v3-turbo` on your specific audio; results are domain-dependent.

### Engines to avoid for new projects

- **Stable-ts** — extends openai-whisper with timestamp adjustments. Use
  only if you specifically need its timestamp post-processing. Otherwise
  it's just a heavier openai-whisper.
- **Open-source Vosk** for batch French — worse than even
  `whisper small`. Vosk's niche is streaming.

---

## Recommended full pipeline (CPU, batch)

```python
import subprocess
import tempfile
from pathlib import Path

import whisper

PREPROCESS_FILTER = "alimiter=limit=0.95:level=disabled,dynaudnorm"

def preprocess(input_path: Path) -> Path:
    """Apply the recommended ffmpeg filter chain; return a temp wav path."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = Path(tmp.name)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(input_path),
            "-af", PREPROCESS_FILTER,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(out),
        ],
        check=True,
    )
    return out


def transcribe(audio_path: Path, *, language: str, initial_prompt: str | None = None) -> str:
    cleaned = preprocess(audio_path)
    try:
        model = whisper.load_model("large-v3-turbo")
        result = model.transcribe(
            str(cleaned),
            language=language,
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            # temperature: omit — keep default fallback ladder ON
            fp16=False,
            verbose=False,
        )
        return str(result["text"]).strip()
    finally:
        cleaned.unlink(missing_ok=True)
```

For diarization, wrap pyannote around this (separate concern; see
scriber's `transcription/local.py` for a reference impl).

For batch-mode multi-file workflows, load the model **once** and reuse:

```python
model = whisper.load_model("large-v3-turbo")
for audio in inputs:
    result = model.transcribe(str(audio), language="fr", initial_prompt=PRIMER, ...)
```

This is where `faster-whisper` starts winning — same loop, different
model class.

---

## How to validate quality on your audio

You can't tune what you don't measure. Three options, ordered by effort:

### 1. Eyeball (cheapest, weakest signal)

Pick 2-3 representative samples of 30s each. Run two variants
side-by-side, read both transcripts, pick the better one. Works for
quick A/B; doesn't scale.

### 2. Cross-agreement vs second-source transcript (no ground truth needed)

If you have another transcription of the same audio (different
recorder, different engine, human notes), tokenize both, compute WER
against a sliding-window alignment of the reference. Lower WER = more
agreement = (presumed) better quality.

Caveats:

- Both transcripts have errors. WER absolute values cluster around
  0.4-0.7 even between good engines. Use **deltas between variants on
  the same sample**, not absolute values.
- For brand-name heavy content, both transcripts often mangle the same
  names. Add a check for `initial_prompt` recall: count how many primer
  terms appear in each variant's output.

See `scriber/experiments/transcription-tuning/scripts/score.py` for a
reference implementation (≈200 lines: WER + CER + token-set F1 +
sliding-window alignment + repetition flag).

### 3. Hand-transcribed gold (gold standard, expensive)

For one 5-min sample: type the transcript yourself. ~30-60min of
typing. Compute WER vs every variant. Anchor your ranking on that one
sample; A/B everywhere else.

### Always watch for

- **Repetition loops** — n-gram × 4+ in a row. Whisper's classic
  failure mode. Easy to detect heuristically; ugly when it happens.
- **Length ratio** — variant tokens / reference tokens. Far from 1.0
  means the variant either lost (much shorter) or hallucinated (much
  longer) content.
- **RTF** — audio_duration / wallclock. Track it per variant; it's the
  cost axis for any quality/cost trade-off.

---

## Known failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| Whole-file CPU 5-10× expected | Temperature-fallback storm on clipped or noisy audio | Add `alimiter` to the ffmpeg pre-pass |
| `"thanks for watching"` or similar loop in quiet stretches | `large-v3`'s known silence-hallucination | Enable VAD silence-trim (Silero); or use `faster-whisper` with `vad_filter=True` |
| One n-gram repeats N times mid-transcript | Token loop, no fallback to escape | Keep default temperature ladder on (don't pass `temperature=0.0`) |
| Language drifts mid-meeting | Auto-detect re-runs per window | Force `language="xx"` |
| Hallucinated facts not in audio | `condition_on_previous_text=True` cascading from one wrong segment | Set `condition_on_previous_text=False` for files >10min |
| Brand names / proper nouns mangled | No vocabulary hint | Build a domain `initial_prompt` |
| Memory blowup loading the model | Trying to load full `large-v3` on a low-RAM box | Use `large-v3-turbo` (4 decoder layers vs 32) or `faster-whisper` with `compute_type="int8"` |
| OOM on a 4h+ file | Whisper holds the full audio in memory | Chunk into 30-60min segments and concatenate transcripts; use `clip_timestamps` to drive |

---

## Anti-patterns (things people do that don't help)

- **"Just throw `large-v3` at it"** on CPU. RTF ~0.4× = 7.5h for a 3h
  file. `large-v3-turbo` gives near-identical quality at 3× the speed.
- **Running `temperature=0.0` thinking determinism is good.** It is for
  bench reproducibility. It is not for production. You lose the
  fallback escape hatch and gain permanent token loops.
- **Re-implementing VAD in Python.** Use `faster-whisper`'s built-in
  `vad_filter=True` or call Silero directly. Don't roll your own.
- **Pre-processing audio with heavy denoise libraries**
  (`noisereduce`, RNNoise) by default. They can introduce artifacts
  that *worsen* whisper output. Measure first.
- **Splitting audio at fixed time boundaries to avoid memory pressure.**
  Whisper's 30-second internal window means it'll re-split anyway.
  Splitting only matters at the >1h file-level for memory headroom.
- **Trusting RTF benchmarks from blog posts.** Per-clip overhead
  (Python process spawn, model load, CTranslate2 init) is invisible in
  the benchmark but dominates in production. Always measure on your
  actual invocation pattern.

---

## Quick reference card

```text
                ┌─────────────────────────────────────┐
                │  ffmpeg -af alimiter+dynaudnorm     │
                │  -ar 16000 -ac 1 -c:a pcm_s16le    │
                └────────────────┬────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────┐
                │  whisper.load_model("large-v3-turbo")│
                └────────────────┬────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────┐
                │  model.transcribe(                  │
                │    audio,                           │
                │    language="xx",                   │  ← force
                │    initial_prompt=primer,           │  ← seed nouns
                │    condition_on_previous_text=False,│  ← anti-cascade
                │    # temperature: keep default      │  ← DO NOT pass 0.0
                │    fp16=False,                      │
                │  )                                  │
                └─────────────────────────────────────┘
```

If you're implementing transcription from scratch in a new project:

1. Start with the snippet above.
2. Measure RTF + WER (cross-agreement) on 3-5 representative clips.
3. Iterate only if you have a specific failure mode the defaults don't
   handle.
4. Don't optimize before measuring. Don't measure on synthetic test data.

---

## References

- **scriber's transcription-tuning bench** —
  `experiments/transcription-tuning/RESULTS.md`. Full data behind every
  number cited here: 57 runs × 3 stages on a 2h51m French meeting
  recording (Recolx Tap room mic, ~2m from loudspeaker, source clipped
  at +10dBFS peak).
- **OpenAI Whisper paper** — Radford et al., *Robust Speech Recognition
  via Large-Scale Weak Supervision*, 2022.
  https://arxiv.org/abs/2212.04356
- **faster-whisper** —
  https://github.com/SYSTRAN/faster-whisper. CTranslate2-based;
  install via `pip install faster-whisper`.
- **Silero VAD** — https://github.com/snakers4/silero-vad. Python-only,
  small footprint, the de-facto VAD for whisper pipelines.
- **ffmpeg filter docs** — https://ffmpeg.org/ffmpeg-filters.html.
  Authoritative reference for `alimiter`, `dynaudnorm`, and friends.
