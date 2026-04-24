"""Summary modes (meeting / source / auto-detect) and their prompt templates."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from scriber.model import Transcript

SummaryMode = Literal["meeting", "source", "auto"]
ResolvedMode = Literal["meeting", "source"]

# --- Unified prompt templates ---------------------------------------------
#
# Each template is filled in via ``str.format(**phrases)`` where ``phrases``
# is one of ``_MEETING_PHRASES[lang]`` / ``_SOURCE_PHRASES[lang]``. The
# ``colon`` entry is ``:`` in EN and `` :`` in FR (French typography).

_MEETING_TEMPLATE = """\
{intro}

{topic_label}{colon} {topic_hint}
Hashtags{colon} {hashtags_hint}
{takeaways_label}{colon}
- {bullet} {speaker_attr}
{qa_label}{colon}
- {qa_hint}
{decisions_label}{colon}
- {bullet}
{actions_label}{colon}
- {action_hint}

{transcript_label}{colon}
"""

_MEETING_PHRASES: dict[str, dict[str, str]] = {
    "en": {
        "intro": (
            "You are an expert summarizer of a multi-speaker meeting. Given the\n"
            "transcript below, produce a structured summary in English with these\n"
            "sections (use the exact headers, in this order):"
        ),
        "colon": ":",
        "topic_label": "Topic",
        "topic_hint": "<one-line meeting topic>",
        "hashtags_hint": "<5-8 relevant tags on one line>",
        "takeaways_label": "Main takeaways",
        "bullet": "<bullet>",
        "speaker_attr": "(attribute to the speaker who expressed it when possible)",
        "qa_label": "Questions / Answers",
        "qa_hint": "<Q (asker) — A (answerer)>",
        "decisions_label": "Decisions",
        "actions_label": "Action items",
        "action_hint": "<action> (owner)",
        "transcript_label": "Transcript",
    },
    "fr": {
        "intro": (
            "Vous êtes un expert en résumé de réunion à plusieurs intervenants.\n"
            "À partir de la transcription ci-dessous, produisez un résumé structuré en\n"
            "français avec ces sections (utilisez exactement ces en-têtes, dans cet\n"
            "ordre) :"
        ),
        "colon": " :",
        "topic_label": "Sujet",
        "topic_hint": "<thème de la réunion en une ligne>",
        "hashtags_hint": "<5 à 8 hashtags pertinents sur une ligne>",
        "takeaways_label": "Principaux enseignements",
        "bullet": "<puce>",
        "speaker_attr": "(attribuez la prise de parole quand possible)",
        "qa_label": "Questions / Réponses",
        "qa_hint": "<Q (auteur) — R (répondant)>",
        "decisions_label": "Décisions",
        "actions_label": "Actions à suivre",
        "action_hint": "<action> (responsable)",
        "transcript_label": "Transcription",
    },
}

# Note the ``Topic`` mini-intro uses the word "TL;DR" which contains no
# literal ``{}`` braces, so it is safe inside ``str.format``.
_SOURCE_TEMPLATE = """\
{intro}

TL;DR{colon} {tldr_hint}
{key_takeaways_label}{colon}
- {bullet}
{facts_label}{colon}
- {facts_hint}
{opinions_label}{colon}
- {opinions_hint}
{speculation_label}{colon}
- {speculation_hint}
{counterpoints_label}{colon}
- {counterpoints_hint}
{reliability_label}{colon} {reliability_hint}

{transcript_label}{colon}
"""

_SOURCE_PHRASES: dict[str, dict[str, str]] = {
    "en": {
        "intro": (
            "You are an expert critical summarizer. The transcript below is from a\n"
            "single source (lecture, interview, article reading, commentary). Produce\n"
            "a structured summary in English with these sections (use the exact\n"
            "headers, in this order):"
        ),
        "colon": ":",
        "tldr_hint": "<2-3 sentences>",
        "key_takeaways_label": "Key takeaways",
        "bullet": "<bullet>",
        "facts_label": "Facts",
        "facts_hint": "<claim> — supported by [observation/citation/data referenced in the source]",
        "opinions_label": "Opinions",
        "opinions_hint": "<claim> — speaker/author's opinion (no external evidence offered)",
        "speculation_label": "Speculation / unverified",
        "speculation_hint": "<claim> — speaker speculates or asserts without support",
        "counterpoints_label": "Counterpoints / alternatives",
        "counterpoints_hint": "<alternative perspective the source did not address>",
        "reliability_label": "Information quality / reliability",
        "reliability_hint": (
            "<one short paragraph rating the\n"
            "source's overall reliability — citations, evidence quality, neutrality,\n"
            "acknowledged uncertainty>"
        ),
        "transcript_label": "Transcript",
    },
    "fr": {
        "intro": (
            "Vous êtes un expert en analyse critique. La transcription ci-dessous\n"
            "provient d'une source unique (cours, interview, lecture d'article,\n"
            "commentaire). Produisez un résumé structuré en français avec ces\n"
            "sections (utilisez exactement ces en-têtes, dans cet ordre) :"
        ),
        "colon": " :",
        "tldr_hint": "<2 à 3 phrases>",
        "key_takeaways_label": "Points clés",
        "bullet": "<puce>",
        "facts_label": "Faits",
        "facts_hint": "<affirmation> — étayée par [observation/citation/donnée mentionnée]",
        "opinions_label": "Opinions",
        "opinions_hint": "<affirmation> — opinion de l'auteur (sans preuve externe avancée)",
        "speculation_label": "Spéculations / non vérifié",
        "speculation_hint": "<affirmation> — l'auteur spécule ou affirme sans étayer",
        "counterpoints_label": "Contrepoints / alternatives",
        "counterpoints_hint": "<perspective alternative non abordée par la source>",
        "reliability_label": "Qualité de l'information / fiabilité",
        "reliability_hint": (
            "<court paragraphe évaluant la\n"
            "fiabilité globale — citations, qualité des preuves, neutralité,\n"
            "incertitudes reconnues>"
        ),
        "transcript_label": "Transcription",
    },
}


_CONTEXT_HEADER: dict[str, str] = {
    "en": "Additional context",
    "fr": "Contexte additionnel",
}


def get_prompt(mode: ResolvedMode, language: str, context: str | None = None) -> str:
    """Return the rendered prompt for ``(mode, language)``.

    When ``context`` is a non-empty string, an ``Additional context`` block
    is inserted just before the ``Transcript:`` marker so the model sees
    it as auxiliary material alongside the main transcript.

    Raises ``ValueError`` for unsupported language.
    """
    if language not in {"en", "fr"}:
        err_msg = f"Summarizer language not supported: {language!r}"
        raise ValueError(err_msg)
    phrases = _MEETING_PHRASES if mode == "meeting" else _SOURCE_PHRASES
    template = _MEETING_TEMPLATE if mode == "meeting" else _SOURCE_TEMPLATE
    rendered = template.format(**phrases[language])
    if context and context.strip():
        colon = phrases[language]["colon"]
        transcript_marker = f"{phrases[language]['transcript_label']}{colon}"
        block = f"{_CONTEXT_HEADER[language]}{colon}\n{context.strip()}\n\n"
        rendered = rendered.replace(transcript_marker, block + transcript_marker, 1)
    return rendered


# --- Auto-detect heuristic -------------------------------------------------

# Words/phrases that signal "I'm sharing my view" rather than reporting facts.
_OPINION_MARKERS = re.compile(
    r"\b("
    r"i think|i believe|in my opinion|i feel|"
    r"je pense|je crois|à mon avis|selon moi|d'après moi|"
    r"probably|maybe|perhaps|likely|"
    r"peut-être|probablement|sans doute"
    r")\b",
    flags=re.IGNORECASE,
)

# A diarized line looks like ``SPEAKER_00: text`` or ``Alice: text``.
_DIARIZED_LINE = re.compile(r"^\s*[A-Z][\w\s.-]{0,30}:\s+\S")


def _count_distinct_speakers(text: str) -> int:
    speakers: set[str] = set()
    for line in text.splitlines():
        if _DIARIZED_LINE.match(line):
            speakers.add(line.split(":", 1)[0].strip())
    return len(speakers)


def detect_mode(transcript: Transcript) -> ResolvedMode:
    """Pick ``meeting`` or ``source`` based on transcript shape.

    Order:
      1. Diarized output with 2+ distinct speakers → ``meeting``.
      2. ``opinion`` density above 1 marker per 1000 words → ``source``.
      3. Default → ``source``.

    """
    if transcript.diarized and _count_distinct_speakers(transcript.text) >= 2:
        return "meeting"

    word_count = max(1, len(transcript.text.split()))
    opinion_hits = len(_OPINION_MARKERS.findall(transcript.text))
    if opinion_hits and (opinion_hits / word_count) * 1000 >= 1.0:
        return "source"

    return "source"


def resolve_mode(requested: SummaryMode, transcript: Transcript) -> ResolvedMode:
    """``auto`` triggers ``detect_mode``; otherwise the user choice wins."""
    if requested == "auto":
        return detect_mode(transcript)
    return requested
