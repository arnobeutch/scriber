"""Markdown formatter for Obsidian-compatible summaries (meeting & source modes)."""

from __future__ import annotations

import datetime as dt
import re
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from scriber.constants import SECTION_HEADERS, SECTION_KEYS, SECTION_LABELS
from scriber.logger import my_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scriber.constants import ResolvedMode
    from scriber.model import Chapter, SourceMetadata, Transcript


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


def _yaml_scalar(value: str | float | bool | None) -> str:
    """Render a scalar YAML value safely for our frontmatter use case."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    # Quote strings so special chars don't break YAML. Escape embedded quotes.
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _yaml_list(items: Sequence[str]) -> str:
    """Render a list of strings as an inline YAML flow sequence."""
    if not items:
        return "[]"
    return "[" + ", ".join(_yaml_scalar(item) for item in items) + "]"


def _source_type(transcript: Transcript, source_path: str | None) -> str:
    if transcript.source in {"yt_manual", "yt_auto"}:
        return "youtube"
    if transcript.source == "whisper":
        parsed = urlparse(source_path or "") if source_path else None
        if parsed and parsed.scheme in {"http", "https"}:
            return "youtube"  # YT audio fallback path
        return "media"
    return "text"


def _build_frontmatter(
    transcript: Transcript,
    *,
    source_path: str | None,
    sentiment: str | None,
    processing_date: str,
    mode: ResolvedMode,
    keywords: Sequence[str],
    tags: Sequence[str],
) -> list[str]:
    meta: SourceMetadata = transcript.metadata
    fields: list[tuple[str, str]] = [
        ("title", _yaml_scalar(transcript.title)),
        ("source_url", _yaml_scalar(source_path)),
        ("source_type", _yaml_scalar(_source_type(transcript, source_path))),
        ("transcript_source", _yaml_scalar(transcript.source)),
        ("channel", _yaml_scalar(meta.channel)),
        ("publication_date", _yaml_scalar(meta.publication_date)),
        ("processing_date", _yaml_scalar(processing_date)),
        ("detected_language", _yaml_scalar(meta.detected_language)),
        ("summary_language", _yaml_scalar(transcript.language)),
        ("summary_mode", _yaml_scalar(mode)),
        ("duration_seconds", _yaml_scalar(meta.duration_seconds)),
        ("chapters_count", _yaml_scalar(len(transcript.chapters))),
        ("diarized", _yaml_scalar(transcript.diarized)),
        ("ingestion_status", _yaml_scalar("full")),
        ("extraction_status", _yaml_scalar("ok")),
        ("sentiment", _yaml_scalar(sentiment)),
        ("keywords", _yaml_list(keywords)),
        ("tags", _yaml_list(tags)),
    ]
    return ["---", *(f"{k}: {v}" for k, v in fields), "---", ""]


def format_summary_markdown(
    raw_summary: str,
    transcript: Transcript,
    mode: ResolvedMode,
    *,
    source_path: str | None = None,
    sentiment: str | None = None,
    processing_date: str | None = None,
    keywords: Sequence[str] | None = None,
    tags: Sequence[str] | None = None,
) -> str:
    """Return an Obsidian-ready structured-sections markdown for the summary.

    Structure:
        ---
        <YAML frontmatter: title, source_url, dates, language, sentiment, ...>
        ---

        # <mode title> — <transcript.title>

        > Source: <source_path>  (omitted when source_path is None/empty)

        ## Chapters              (omitted when transcript.chapters = [])
        - [MM:SS](url?t=ss) ...

        ## <Section 1 header>
        <content or localized default>

        ...

        ## Sentiment
        <sentiment>  (omitted when sentiment is None)
    """
    my_logger.debug("Formatting summary markdown...")
    language = transcript.language
    headers = SECTION_HEADERS[mode][language]
    sections = extract_sections(raw_summary, mode, language)

    lines = _build_frontmatter(
        transcript,
        source_path=source_path,
        sentiment=sentiment,
        processing_date=processing_date or dt.datetime.now(tz=dt.UTC).date().isoformat(),
        mode=mode,
        keywords=keywords or [],
        tags=tags or [],
    )

    title_line = f"{_TITLE_PREFIXES[mode][language]} — {transcript.title}"
    lines.extend([title_line, ""])
    if source_path:
        lines.append(f"> {_SOURCE_LABEL[language]}: {source_path}")
        lines.append("")

    if transcript.chapters:
        lines.extend(_render_chapters_section(transcript.chapters, source_path, language))

    for key in SECTION_KEYS[mode]:
        lines.append(headers[key])
        lines.append(clean_section(sections.get(key, ""), language))
        lines.append("")

    if sentiment:
        lines.append(_SENTIMENT_HEADER[language])
        lines.append(sentiment)
        lines.append("")

    return "\n".join(lines)
