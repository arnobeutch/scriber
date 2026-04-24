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
    "summary",
    "claims",
    "quotes",
    "factual",
    "likely",
    "interpretation",
    "alternatives",
    "wrong",
    "keywords",
    "tags",
)

SOURCE_SECTION_LABELS: dict[str, dict[str, str]] = {
    "fr": {
        "summary": "Résumé",
        "claims": "Thèses principales",
        "quotes": "Citations notables",
        "factual": "Factuellement correct",
        "likely": "Probable mais non confirmé",
        "interpretation": "Interprétation ou faiblement étayé",
        "alternatives": "Interprétations alternatives",
        "wrong": "Faux ou trompeur",
        "keywords": "Mots-clés",
        "tags": "Tags",
    },
    "en": {
        "summary": "Summary",
        "claims": "Main claims",
        "quotes": "Notable quotes",
        "factual": "Factually correct",
        "likely": "Likely but unconfirmed",
        "interpretation": "Interpretation or weakly substantiated",
        "alternatives": "Alternative interpretations",
        "wrong": "Wrong or misleading",
        "keywords": "Keywords",
        "tags": "Tags",
    },
}

SOURCE_SECTION_HEADERS: dict[str, dict[str, str]] = {
    "fr": {
        "summary": "## Résumé",
        "claims": "## Thèses principales",
        "quotes": "## Citations notables",
        "factual": "## Ce qui est factuellement correct",
        "likely": "## Ce qui est probable mais non confirmé",
        "interpretation": "## Ce qui relève de l'interprétation ou est faiblement étayé",
        "alternatives": "### Interprétations alternatives",
        "wrong": "## Ce qui est faux ou trompeur",
        "keywords": "",  # rendered only in frontmatter
        "tags": "",
    },
    "en": {
        "summary": "## Summary",
        "claims": "## Main Claims",
        "quotes": "## Notable Quotes",
        "factual": "## What Is Factually Correct",
        "likely": "## What Is Likely but Unconfirmed",
        "interpretation": "## What Is Interpretation or Weakly Substantiated",
        "alternatives": "### Alternative Interpretations",
        "wrong": "## What Is Wrong or Misleading",
        "keywords": "",
        "tags": "",
    },
}

# Neutral keys that are extracted from the model output but routed to the
# YAML frontmatter instead of rendered as a visible section body.
FRONTMATTER_ONLY_KEYS: frozenset[str] = frozenset({"keywords", "tags"})

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
