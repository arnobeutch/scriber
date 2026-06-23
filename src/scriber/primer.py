"""Build a draft "primer" (whisper ``initial_prompt``) from a transcription.

The primer biases whisper toward your proper nouns / jargon, but writing one by
hand is a chicken-and-egg chore: you don't know up front what the model gets
wrong. This module surfaces candidates *from a finished transcription* — proper
nouns, acronyms, and the words whisper itself was least sure about — into a draft
file the user reviews, fixes, and feeds back via ``--initial-prompt-file`` for a
consolidation pass (which also pins one spelling across a long file, since
transcription runs with ``condition_on_previous_text=False``).

Pure + dependency-free (stdlib only); extraction is heuristic, not NLP-grade.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

# Letter-runs (Unicode-aware, no digits/underscore), allowing internal apostrophe
# (straight or curly — whisper emits curly in French) or hyphen so
# "Marie-Hélène" / "aujourd'hui" stay single tokens.
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")  # noqa: RUF001 — curly apostrophe is intentional
# Sentence boundaries — used only to ignore sentence-initial capitalization.
_SENT_SPLIT_RE = re.compile(r"[.!?…]+|\n+")
_EDGE_PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$")

# Capitalized words that are not proper nouns: sentence openers, politeness,
# connectors (FR + EN, for mixed-language sources). Compared lower-cased.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "le", "la", "les", "l", "un", "une", "des", "de", "du", "d", "et", "ou",
        "mais", "donc", "or", "car", "ni", "alors", "ainsi", "puis", "ensuite",
        "enfin", "bref", "voila", "voilà", "je", "tu", "il", "elle", "on", "nous",
        "vous", "ils", "elles", "ce", "cet", "cette", "ces", "mon", "ma", "mes",
        "ton", "ta", "tes", "son", "sa", "ses", "notre", "nos", "votre", "vos",
        "leur", "leurs", "qui", "que", "quoi", "dont", "quand", "comment",
        "pourquoi", "si", "oui", "non", "bonjour", "bonsoir", "merci", "salut",
        "madame", "monsieur", "messieurs", "mesdames", "aussi", "comme", "très",
        "plus", "moins", "bien", "the", "a", "an", "and", "but", "so",
        "yes", "no", "hello", "hi", "thanks", "thank", "i", "we", "you", "they",
        "he", "she", "it", "this", "that", "these", "those", "well", "okay", "ok",
    },
)

_DEFAULT_MAX_TERMS = 40
_DEFAULT_LOW_CONF_THRESHOLD = 0.5
_MIN_WORD_LEN = 2
_MIN_LOWCONF_LEN = 3  # ignore 1-2 char low-confidence tokens (mostly filler/punctuation)


@dataclass(frozen=True)
class PrimerCandidates:
    """Ranked primer candidates harvested from a transcription."""

    proper_nouns: list[tuple[str, int]]  # (term, frequency), most common first
    acronyms: list[tuple[str, int]]  # (term, frequency), most common first
    low_confidence: list[tuple[str, float]]  # (word, min probability), least confident first


def _segments_text(segments: list[dict[str, Any]]) -> str:
    """Join segment texts (raw whisper cues — no speaker labels to pollute terms)."""
    return " ".join(str(seg.get("text", "")).strip() for seg in segments)


def _extract_terms(text: str) -> tuple[Counter[str], Counter[str]]:
    """Return (proper-noun counter, acronym counter) from `text`."""
    proper: Counter[str] = Counter()
    acronyms: Counter[str] = Counter()
    for sentence in _SENT_SPLIT_RE.split(text):
        tokens = _WORD_RE.findall(sentence)
        for idx, tok in enumerate(tokens):
            if tok.isupper() and len(tok) >= _MIN_WORD_LEN:
                acronyms[tok] += 1
            # idx > 0: sentence-initial capitalization isn't informative.
            elif (
                idx > 0
                and len(tok) >= _MIN_WORD_LEN
                and tok[0].isupper()
                and tok.lower() not in _STOPWORDS
            ):
                proper[tok] += 1
    return proper, acronyms


def _low_confidence_words(
    segments: list[dict[str, Any]],
    threshold: float,
) -> list[tuple[str, float]]:
    """Words whisper scored below `threshold`, keyed to their lowest probability."""
    best: dict[str, float] = {}
    for seg in segments:
        words = seg.get("words") or []
        for word in words:
            token = _EDGE_PUNCT_RE.sub("", str(word.get("word", "")).strip())
            prob = float(word.get("probability", 1.0))
            # A primer wants uncertain *names/terms*, not common words whisper
            # happened to score low — drop stopwords and short tokens.
            if len(token) < _MIN_LOWCONF_LEN or prob >= threshold or token.lower() in _STOPWORDS:
                continue
            if token not in best or prob < best[token]:
                best[token] = prob
    return sorted(best.items(), key=lambda kv: kv[1])


def extract_primer_candidates(
    segments: list[dict[str, Any]],
    *,
    max_terms: int = _DEFAULT_MAX_TERMS,
    low_conf_threshold: float = _DEFAULT_LOW_CONF_THRESHOLD,
) -> PrimerCandidates:
    """Harvest proper nouns, acronyms, and low-confidence words from whisper segments.

    ``low_confidence`` is only populated when segments carry word-level data
    (whisper run with ``word_timestamps=True``).
    """
    proper, acronyms = _extract_terms(_segments_text(segments))
    return PrimerCandidates(
        proper_nouns=proper.most_common(max_terms),
        acronyms=acronyms.most_common(max_terms),
        low_confidence=_low_confidence_words(segments, low_conf_threshold)[:max_terms],
    )


def format_primer_draft(candidates: PrimerCandidates, title: str) -> str:
    """Render a reviewable primer draft.

    Proper nouns + acronyms are written as active (uncommented) primer content;
    low-confidence words are ``#`` comments (they may be misspelled, so the user
    decides whether to add a corrected form). ``#`` lines are stripped when the
    file is loaded as a primer, so the draft is usable as-is after trimming.
    """
    lines: list[str] = [
        f'# Auto-suggested primer for "{title}".',
        "# Review before use: fix spellings, delete noise, keep real proper nouns / jargon.",
        "# '#' lines are ignored when this file is passed to --initial-prompt-file,",
        "# so you can keep these notes inline. See docs/WHISPER_SETUP.md.",
        "#",
        "# --- Proper nouns / names (by frequency) ---",
        ", ".join(term for term, _ in candidates.proper_nouns) or "# (none detected)",
        "#",
        "# --- Acronyms ---",
        ", ".join(term for term, _ in candidates.acronyms) or "# (none detected)",
    ]
    if candidates.low_confidence:
        lines += [
            "#",
            "# --- Words whisper was unsure about (word: confidence) ---",
            "#     If any are real names/terms, add the correct spelling above.",
            *(f"#   {word}: {prob:.2f}" for word, prob in candidates.low_confidence),
        ]
    return "\n".join(lines) + "\n"
