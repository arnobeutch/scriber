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

{boundary}

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

_BOUNDARY_EN = (
    "Ground every claim, quote, and attribution in the transcript below.\n"
    "Do NOT supplement with facts, names, statistics, or assertions from your\n"
    "training data. If adding background context genuinely aids comprehension\n"
    '(e.g. defining a technical term), prefix it with "[Background]" so it is\n'
    "visibly distinct from transcript-derived content."
)

_BOUNDARY_FR = (
    "Ancrez chaque affirmation, citation et attribution dans la transcription\n"
    "ci-dessous. N'ajoutez PAS de faits, noms, statistiques ou assertions issus\n"
    "de vos données d'entraînement. Si un contexte d'arrière-plan aide vraiment\n"
    "la compréhension (ex. définir un terme technique), préfixez-le par\n"
    '"[Contexte]" pour le distinguer visiblement du contenu tiré de la transcription.'
)


_MEETING_PHRASES: dict[str, dict[str, str]] = {
    "en": {
        "intro": (
            "You are an expert summarizer of a multi-speaker meeting. Given the\n"
            "transcript below, produce a structured summary in English with these\n"
            "sections (use the exact headers, in this order):"
        ),
        "boundary": _BOUNDARY_EN,
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
        "boundary": _BOUNDARY_FR,
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

_SOURCE_TEMPLATE = """\
{intro}

{boundary}

{summary_label}{colon} {summary_hint}
{claims_label}{colon}
1. {claim_hint}
2. {claim_hint}
3. {claim_hint}
{quotes_label}{colon}
{quote_hint}
{factual_label}{colon}
- {factual_hint}
{likely_label}{colon}
- {likely_hint}
{interpretation_label}{colon}
- {interpretation_hint}
{alternatives_label}{colon}
- {alternatives_hint}
{wrong_label}{colon}
- {wrong_hint}
{applications_label}{colon}
- {applications_hint}
{extensions_label}{colon}
- {extensions_hint}
{keywords_label}{colon} {keywords_hint}
{tags_label}{colon} {tags_hint}

{transcript_label}{colon}
"""

_SOURCE_PHRASES: dict[str, dict[str, str]] = {
    "en": {
        "intro": (
            "You are an expert critical summarizer. The transcript below is from a\n"
            "single source (lecture, interview, article reading, commentary). Produce\n"
            "a structured summary in English with these sections (use the exact\n"
            "headers, in this order). Categorize every substantive claim into exactly\n"
            "one of the four epistemic buckets (Factually correct / Likely but\n"
            "unconfirmed / Interpretation or weakly substantiated / Wrong or\n"
            'misleading). If a bucket has nothing to report, write "None identified".'
        ),
        "boundary": _BOUNDARY_EN,
        "colon": ":",
        "summary_label": "Summary",
        "summary_hint": "<1-2 sentence overview of what the source argues>",
        "claims_label": "Main claims",
        "claim_hint": "<claim>",
        "quotes_label": "Notable quotes",
        "quote_hint": (
            '> "<verbatim quote from the source>"\n'
            ">\n"
            "> — <speaker or author name; 'Unknown' if unattributable>\n"
            '(include 1-3 quotes; if none attributable, write "None identified")'
        ),
        "factual_label": "Factually correct",
        "factual_hint": (
            "<claim> — supported by <observation/citation/data referenced in the source>"
        ),
        "likely_label": "Likely but unconfirmed",
        "likely_hint": "<claim> — <why plausible, without independent confirmation>",
        "interpretation_label": "Interpretation or weakly substantiated",
        "interpretation_hint": "<claim> — <evidence weakness>",
        "alternatives_label": "Alternative interpretations",
        "alternatives_hint": "<alternative reading of one of the weak claims above>",
        "wrong_label": "Wrong or misleading",
        "wrong_hint": "<claim> — <why wrong>, or None identified",
        "applications_label": "Applications / So what",
        "applications_hint": (
            "<practical takeaway or actionable implication>. "
            'If the source is purely descriptive / analytical, write "None identified".'
        ),
        "extensions_label": "How to extend",
        "extensions_hint": ("<suggested follow-up source, research angle, or verification step>"),
        "keywords_label": "Keywords",
        "keywords_hint": "<3-5 kebab-case descriptors, comma-separated>",
        "tags_label": "Tags",
        "tags_hint": "<3-5 Obsidian hashtags, space-separated, e.g. #ml #research>",
        "transcript_label": "Transcript",
    },
    "fr": {
        "intro": (
            "Vous êtes un expert en analyse critique. La transcription ci-dessous\n"
            "provient d'une source unique (cours, interview, lecture d'article,\n"
            "commentaire). Produisez un résumé structuré en français avec ces\n"
            "sections (utilisez exactement ces en-têtes, dans cet ordre). Classez\n"
            "chaque affirmation substantielle dans exactement une des quatre\n"
            "catégories épistémiques (Factuellement correct / Probable mais non\n"
            "confirmé / Interprétation ou faiblement étayé / Faux ou trompeur).\n"
            "Si une catégorie n'a rien à signaler, écrivez « Aucun identifié »."
        ),
        "boundary": _BOUNDARY_FR,
        "colon": " :",
        "summary_label": "Résumé",
        "summary_hint": "<aperçu en 1-2 phrases de la thèse de la source>",
        "claims_label": "Thèses principales",
        "claim_hint": "<thèse>",
        "quotes_label": "Citations notables",
        "quote_hint": (
            "> « <citation verbatim tirée de la source> »\n"
            ">\n"
            "> — <intervenant ou auteur ; « Inconnu » si non identifiable>\n"
            "(inclure 1-3 citations ; si aucune attribuable, écrire « Aucune identifiée »)"
        ),
        "factual_label": "Factuellement correct",
        "factual_hint": (
            "<affirmation> — étayée par <observation/citation/donnée mentionnée dans la source>"
        ),
        "likely_label": "Probable mais non confirmé",
        "likely_hint": "<affirmation> — <pourquoi plausible, sans confirmation indépendante>",
        "interpretation_label": "Interprétation ou faiblement étayé",
        "interpretation_hint": "<affirmation> — <faiblesse de la preuve>",
        "alternatives_label": "Interprétations alternatives",
        "alternatives_hint": "<lecture alternative d'une des affirmations faibles ci-dessus>",
        "wrong_label": "Faux ou trompeur",
        "wrong_hint": "<affirmation> — <pourquoi faux>, ou Aucun identifié",
        "applications_label": "Applications / Et alors",
        "applications_hint": (
            "<enseignement pratique ou implication actionnable>. "
            "Si la source est purement descriptive / analytique, écrire « Aucune identifiée »."
        ),
        "extensions_label": "Pistes d'approfondissement",
        "extensions_hint": ("<source de suivi, angle de recherche ou étape de vérification>"),
        "keywords_label": "Mots-clés",
        "keywords_hint": "<3-5 descripteurs en kebab-case, séparés par des virgules>",
        "tags_label": "Tags",
        "tags_hint": "<3-5 hashtags style Obsidian, séparés par des espaces, ex : #ml #recherche>",
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
