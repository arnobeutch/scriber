"""Project-wide constants and mutable globals."""

# --- Constants (immutable) ---

POLARITY_POSITIVE_THRESHOLD = 0.2
POLARITY_NEGATIVE_THRESHOLD = -0.2

# Neutral keys — language-independent identifiers used everywhere
# downstream (regex extraction, markdown assembly, tests).
RAG_SECTION_KEYS: tuple[str, ...] = (
    "topic",
    "hashtags",
    "takeaways",
    "qa",
    "decisions",
    "actions",
)

# Language-specific in-transcript labels the model is asked to emit
# (used by the regex in markdown.extract_sections).
RAG_SECTION_LABELS: dict[str, dict[str, str]] = {
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

# Language-specific `##` headers written into the final markdown.
RAG_SECTION_HEADERS: dict[str, dict[str, str]] = {
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


# --- Globals (mutable shared state) ---
