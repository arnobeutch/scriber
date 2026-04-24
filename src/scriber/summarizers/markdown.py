"""Markdown formatter for Obsidian-compatible summaries (meeting & source modes)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from scriber.constants import SECTION_HEADERS, SECTION_KEYS, SECTION_LABELS
from scriber.logger import my_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scriber.constants import ResolvedMode
    from scriber.model import Chapter


def extract_sections(summary: str, mode: ResolvedMode, language: str) -> dict[str, str]:
    """Return mapping of neutral section key → content from raw model text.

    Labels are mode- and language-specific (e.g. ``Sujet`` / ``Topic`` /
    ``TL;DR``). Returned keys are the neutral identifiers from
    ``SECTION_KEYS[mode]``.
    """
    labels = SECTION_LABELS[mode][language]
    label_to_key = {label: key for key, label in labels.items()}
    joined_labels = "|".join(re.escape(label) for label in labels.values())
    pattern = rf"({joined_labels})\s*:\s*(.*?)(?=\n(?:{joined_labels})\s*:|\Z)"

    matches = re.findall(pattern, summary, flags=re.DOTALL)
    return {label_to_key[label.strip()]: body.strip() for label, body in matches}


def clean_section(text: str, language: str) -> str:
    """Return cleaned section content, or a localized default when empty."""
    cleaned = text.strip()
    default = "Aucune" if language == "fr" else "None"
    if not cleaned or cleaned.lower() in {"none", "aucune", "n/a"}:
        return default
    return cleaned


_TITLE_PREFIXES: dict[ResolvedMode, dict[str, str]] = {
    "meeting": {
        "en": "# Meeting Summary",
        "fr": "# Résumé de la réunion",
    },
    "source": {
        "en": "# Summary",
        "fr": "# Résumé",
    },
}

_SENTIMENT_HEADER: dict[str, str] = {"en": "## Sentiment", "fr": "## Sentiment"}
_SOURCE_LABEL: dict[str, str] = {"en": "Source", "fr": "Source"}
_CHAPTERS_HEADER: dict[str, str] = {"en": "## Chapters", "fr": "## Chapitres"}


def _format_timestamp(seconds: float) -> str:
    """Return ``H:MM:SS`` when ≥ 1h, else ``MM:SS``."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _deep_link(base_url: str, seconds: float) -> str | None:
    """Return ``base_url`` with a ``?t=<ss>`` query; ``None`` when not a URL."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    query = parse_qs(parsed.query, keep_blank_values=True)
    query["t"] = [str(int(seconds))]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _render_chapters_section(
    chapters: Sequence[Chapter],
    source_path: str | None,
    language: str,
) -> list[str]:
    if not chapters:
        return []
    lines: list[str] = [_CHAPTERS_HEADER[language]]
    for ch in chapters:
        ts = _format_timestamp(ch.start_time)
        link = _deep_link(source_path, ch.start_time) if source_path else None
        if link:
            lines.append(f"- [{ts}]({link}) {ch.title}")
        else:
            lines.append(f"- {ts} — {ch.title}")
    lines.append("")
    return lines


def format_summary_markdown(
    raw_summary: str,
    filename_stem: str,
    language: str,
    mode: ResolvedMode,
    *,
    source_path: str | None = None,
    sentiment: str | None = None,
    chapters: Sequence[Chapter] | None = None,
) -> str:
    """Return an Obsidian-ready structured-sections markdown for the summary.

    Structure:
        # <mode title> — <filename_stem>
        > Source: <source_path>  (omitted when source_path is None/empty)

        ## Chapters             (omitted when chapters is empty)
        - [MM:SS](url?t=ss) ...

        ## <Section 1 header>
        <content or localized default>

        ...

        ## Sentiment
        <sentiment>  (omitted when sentiment is None)
    """
    my_logger.debug("Formatting summary markdown...")
    headers = SECTION_HEADERS[mode][language]
    sections = extract_sections(raw_summary, mode, language)

    title_line = f"{_TITLE_PREFIXES[mode][language]} — {filename_stem}"

    lines: list[str] = [title_line, ""]
    if source_path:
        lines.append(f"> {_SOURCE_LABEL[language]}: {source_path}")
        lines.append("")

    if chapters:
        lines.extend(_render_chapters_section(chapters, source_path, language))

    for key in SECTION_KEYS[mode]:
        lines.append(headers[key])
        lines.append(clean_section(sections.get(key, ""), language))
        lines.append("")

    if sentiment:
        lines.append(_SENTIMENT_HEADER[language])
        lines.append(sentiment)
        lines.append("")

    return "\n".join(lines)
