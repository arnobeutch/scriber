"""Markdown formatter for Obsidian-compatible meeting summaries."""

import re

from scriber.constants import RAG_SECTION_HEADERS, RAG_SECTION_KEYS, RAG_SECTION_LABELS
from scriber.logger import my_logger


def extract_sections(summary: str, language: str) -> dict[str, str]:
    """Return mapping of neutral section key -> content from raw text.

    The regex matches the language-specific labels the prompt asks the
    model to emit (``Sujet`` / ``Topic``…); the returned dict is keyed by
    the neutral identifiers in ``RAG_SECTION_KEYS``.
    """
    labels = RAG_SECTION_LABELS[language]
    label_to_key = {label: key for key, label in labels.items()}
    joined_labels = "|".join(re.escape(label) for label in labels.values())
    # Bound each section at the next known label or end-of-string. Using the
    # label alternation in the lookahead (instead of `\n\S+?:`) correctly
    # handles multi-word labels like "Principaux enseignements".
    pattern = rf"({joined_labels})\s*:\s*(.*?)(?=\n(?:{joined_labels})\s*:|\Z)"

    matches = re.findall(pattern, summary, flags=re.DOTALL)
    return {label_to_key[label.strip()]: body.strip() for label, body in matches}


def clean_section(text: str, language: str) -> str:
    """Return cleaned section content, or default if empty.

    Args:
        text (str): Section body.
        language (str): 'fr' or 'en'

    Returns:
        str: Cleaned section body or default filler.

    """
    cleaned = text.strip()
    default = "Aucune" if language == "fr" else "None"
    if not cleaned or cleaned.lower() in {"none", "aucune", "n/a"}:
        return default
    return cleaned


def format_summary_markdown(raw_summary: str, filename_stem: str, language: str) -> str:
    """Return Obsidian-ready markdown summary from raw model output.

    Args:
        raw_summary (str): Summary output from the RAG engine.
        filename_stem (str): Name (stem only) of the input file to use as title.
        language (str): 'fr' or 'en'

    Returns:
        str: Complete markdown string.

    """
    my_logger.debug("Formatting summary markdown...")
    headers = RAG_SECTION_HEADERS[language]
    sections = extract_sections(raw_summary, language)

    title_line = (
        f"# Résumé de la réunion — {filename_stem}"
        if language == "fr"
        else f"# Meeting Summary — {filename_stem}"
    )

    lines: list[str] = [title_line, ""]
    for key in RAG_SECTION_KEYS:
        lines.append(headers[key])
        lines.append(clean_section(sections.get(key, ""), language))
        lines.append("")

    return "\n".join(lines)


def simple_format_markdown(
    video_title: str,
    video_path: str,
    summary: str,
    sentiment: str,
    language: str,
) -> str:
    """Format the final output in Markdown."""
    if language == "en":
        return f"""
## 📺 Video Summary
- Title: {video_title}
- From: {video_path}
- **Sentiment:** {sentiment}
### 🎯 Theme & Summary
{summary}

"""
    if language == "fr":
        return f"""
## 📺 Résumé de la vidéo
- Titre : {video_title}
- De : {video_path}
- **Sentiment :** {sentiment}
### 🎯 Thème & Résumé
{summary}

"""
    return "Error: summarizer language not supported."
