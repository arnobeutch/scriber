"""Shared domain dataclasses (no business logic).

Lives in its own module to avoid import cycles between handlers and the
summarizers package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TranscriptSource = Literal["yt_manual", "yt_auto", "whisper", "file"]


@dataclass(frozen=True)
class Chapter:
    """YouTube chapter metadata: start offset in seconds + title."""

    start_time: float
    title: str


@dataclass(frozen=True)
class Transcript:
    """In-memory representation of a transcript ready to be written / summarized."""

    text: str
    language: str  # the *summary* language, derived from source
    title: str
    source: TranscriptSource
    diarized: bool
    segments: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    """Whisper-style per-cue segments for SRT/VTT export. Empty when N/A."""
    chapters: list[Chapter] = field(default_factory=list[Chapter])
    """YouTube chapters when the source URL exposes them; empty otherwise."""
