"""Project-wide constants and mutable globals."""

# --- Constants (immutable) ---

POLARITY_POSITIVE_THRESHOLD = 0.2
POLARITY_NEGATIVE_THRESHOLD = -0.2

OPENAI_PROMPT_EN = """
You are an expert summarizer. Given the following YouTube video transcript, provide:

1. **Theme of the video**
2. **Key ideas discussed**
3. **Main takeaways**
4. **Top 5 keywords**
5. **3 main topic categories**

Provide a structured summary.

Transcript:
"""

OPENAI_PROMPT_FR = """
Vous êtes un expert en résumé. Étant donné la transcription vidéo YouTube suivante, fournissez:

1. **Thème de la vidéo**
2. **Idées clés discutées**
3. **Principal à retenir**
4. **5 mots-clés principaux**
5. **3 catégories principales**

Fournir un résumé structuré.

Transcription:
"""

RAG_FRENCH_PROMPT = """
Tu es un assistant intelligent qui aide à résumer des transcriptions de réunions professionnelles. Tu vas analyser le contenu suivant et fournir les éléments suivants en français, de manière claire et concise, avec des informations précises et attribution des intervenants.

Contenu :
{text}

Tâches :
1. Détermine le sujet principal de la réunion.
2. Propose quelques hashtags pertinents (une seule ligne).
3. Résume les points clés discutés, en indiquant qui les a exprimés.
4. Liste les questions posées (avec les auteurs) et les réponses correspondantes (avec les répondants), s'il y en a.
5. Indique les décisions prises, s'il y en a.
6. Détaille les actions à suivre avec les personnes responsables, s'il y en a.

Format :
Sujet : ...
Hashtags : #projetx #client ...
Principaux enseignements :
- ...
Questions / Réponses :
- ...
Décisions :
- ...
Actions à suivre :
- ...
"""

RAG_ENGLISH_PROMPT = """
You are a smart assistant helping to summarize transcripts of professional meetings. Analyze the transcript below and provide the following information in English, with concise yet precise language and speaker attribution.

Content:
{text}

Tasks:
1. Identify the main topic of the meeting.
2. Propose a few relevant hashtags (single line).
3. Summarize the key takeaways, indicating who expressed them.
4. List the questions asked (with who asked) and the answers (with who answered), if any.
5. Note any decisions that were made, if applicable.
6. Outline any action items and who is responsible, if applicable.

Format:
Topic: ...
Hashtags: #projectx #client ...
Main takeaways:
- ...
Questions / Answers:
- ...
Decisions:
- ...
Action items:
- ...
"""

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
