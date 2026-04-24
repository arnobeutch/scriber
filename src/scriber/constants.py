"""Project-wide constants and mutable globals."""

from __future__ import annotations

from typing import Literal

# --- Constants (immutable) ---

POLARITY_POSITIVE_THRESHOLD = 0.2
POLARITY_NEGATIVE_THRESHOLD = -0.2

ResolvedMode = Literal["meeting", "source"]

# --- Meeting-mode section schema ------------------------------------------

MEETING_SECTION_KEYS: tuple[str, ...] = (
    "topic",
    "hashtags",
    "takeaways",
    "qa",
    "decisions",
    "actions",
)

MEETING_SECTION_LABELS: dict[str, dict[str, str]] = {
    "fr": {
        "topic": "Sujet",
        "hashtags": "Hashtags",
        "takeaways": "Principaux enseignements",
        "qa": "Questions / Réponses",
        "decisions": "Décisions",
        "actions": "Actions à suivre",
    },
    "en": {
        "topic": "Topic",
        "hashtags": "Hashtags",
        "takeaways": "Main takeaways",
        "qa": "Questions / Answers",
        "decisions": "Decisions",
        "actions": "Action items",
    },
}

MEETING_SECTION_HEADERS: dict[str, dict[str, str]] = {
    "fr": {
        "topic": "## Sujet de la réunion",
        "hashtags": "## #Hashtags",
        "takeaways": "## Principaux enseignements",
        "qa": "## Questions / Réponses",
        "decisions": "## Décisions",
        "actions": "## Actions à suivre",
    },
    "en": {
        "topic": "## Meeting Topic",
        "hashtags": "## #Hashtags",
        "takeaways": "## Main Takeaways",
        "qa": "## Questions / Answers",
        "decisions": "## Decisions",
        "actions": "## Action Items",
    },
}

# --- Source-mode section schema -------------------------------------------

SOURCE_SECTION_KEYS: tuple[str, ...] = (
    "tldr",
    "takeaways",
    "facts",
    "opinions",
    "speculation",
    "counterpoints",
    "reliability",
)

SOURCE_SECTION_LABELS: dict[str, dict[str, str]] = {
    "fr": {
        "tldr": "TL;DR",
        "takeaways": "Points clés",
        "facts": "Faits",
        "opinions": "Opinions",
        "speculation": "Spéculations / non vérifié",
        "counterpoints": "Contrepoints / alternatives",
        "reliability": "Qualité de l'information / fiabilité",
    },
    "en": {
        "tldr": "TL;DR",
        "takeaways": "Key takeaways",
        "facts": "Facts",
        "opinions": "Opinions",
        "speculation": "Speculation / unverified",
        "counterpoints": "Counterpoints / alternatives",
        "reliability": "Information quality / reliability",
    },
}

SOURCE_SECTION_HEADERS: dict[str, dict[str, str]] = {
    "fr": {
        "tldr": "## TL;DR",
        "takeaways": "## Points clés",
        "facts": "## Faits",
        "opinions": "## Opinions",
        "speculation": "## Spéculations & non vérifié",
        "counterpoints": "## Contrepoints & alternatives",
        "reliability": "## Qualité de l'information",
    },
    "en": {
        "tldr": "## TL;DR",
        "takeaways": "## Key Takeaways",
        "facts": "## Facts",
        "opinions": "## Opinions",
        "speculation": "## Speculation & Unverified",
        "counterpoints": "## Counterpoints & Alternatives",
        "reliability": "## Information Quality",
    },
}

# --- Combined schema lookup ------------------------------------------------

SECTION_KEYS: dict[ResolvedMode, tuple[str, ...]] = {
    "meeting": MEETING_SECTION_KEYS,
    "source": SOURCE_SECTION_KEYS,
}

SECTION_LABELS: dict[ResolvedMode, dict[str, dict[str, str]]] = {
    "meeting": MEETING_SECTION_LABELS,
    "source": SOURCE_SECTION_LABELS,
}

SECTION_HEADERS: dict[ResolvedMode, dict[str, dict[str, str]]] = {
    "meeting": MEETING_SECTION_HEADERS,
    "source": SOURCE_SECTION_HEADERS,
}


# --- Globals (mutable shared state) ---
