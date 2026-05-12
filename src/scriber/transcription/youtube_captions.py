# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Get YouTube transcripts via yt-dlp.

We use yt-dlp's caption-download path (rather than a separate API) so we can:
  * benefit from the same yt-dlp we already use for audio,
  * cleanly distinguish *manual* (author-provided) and *automatic* captions,
  * handle the same edge cases yt-dlp handles (member-only videos, region
    blocks, subtitle-disabled videos, etc.).

``get_youtube_transcript`` returns a :class:`CaptionTrack` carrying the
caption text, its actual language code, and whether it came from a manual or
auto-generated track. The pick follows the ladder spelled out in the README's
"Language selection" section (manual beats auto across languages).
"""

from __future__ import annotations

import re
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yt_dlp
from yt_dlp.utils import DownloadError

from scriber.logger import my_logger

CaptionKind = Literal["manual", "auto"]


@dataclass(frozen=True)
class CaptionTrack:
    """A retrieved caption track ready for downstream use."""

    text: str
    lang: str
    kind: CaptionKind
    declared_language: str | None = None
    """Uploader-declared video language from yt-dlp's ``info["language"]``
    (normalised to a 2-letter code). When the user did not pass
    ``--language``, the caller treats this as the implicit requested
    language so the caption ladder doesn't fall through to manual EN on
    a non-English video that happens to ship English subtitles."""


class TranscriptUnavailableError(Exception):
    """Raised when no YouTube transcript can be obtained.

    Callers should catch this and fall back to local transcription.
    The ``reason`` attribute carries a short machine-friendly tag; the
    exception message is human-readable.
    """

    def __init__(self, reason: str, message: str) -> None:
        """Initialize with a tag and a message."""
        super().__init__(message)
        self.reason = reason


def _build_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _match_lang_key(tracks: dict[str, Any], lang: str) -> str | None:
    """Return the actual key in ``tracks`` for ``lang``, or ``None``.

    Tries exact match first, then prefix match so that requesting ``"en"``
    also finds ``"en-US"``, ``"en-GB"``, etc.
    """
    if lang in tracks:
        return lang
    prefix = lang + "-"
    return next((k for k in tracks if k.startswith(prefix)), None)


def _pick_caption(
    info: dict[str, Any],
    requested_lang: str | None = None,
) -> tuple[str, CaptionKind] | None:
    """Return ``(lang_key, kind)`` for the best available caption track.

    ``lang_key`` is the actual key from the yt-dlp info dict (e.g. ``"en-US"``).
    Callers should normalise it to the base language code for downstream use.

    Priority, top-down (manual beats auto across languages):
      1. Manual @ ``requested_lang`` (if set).
      2. Auto @ ``requested_lang`` (if set).
      3. Manual @ ``"en"``.
      4. Auto @ ``"en"``.
      5. Manual @ any other language (first encountered).
      6. Auto @ any other language (first encountered).

    """
    subtitles = cast(dict[str, Any], info.get("subtitles") or {})
    auto = cast(dict[str, Any], info.get("automatic_captions") or {})

    preferred: list[str] = []
    if requested_lang:
        preferred.append(requested_lang)
    if "en" not in preferred:
        preferred.append("en")

    for lang in preferred:
        key = _match_lang_key(subtitles, lang)
        if key is not None:
            return (key, "manual")
    for lang in preferred:
        key = _match_lang_key(auto, lang)
        if key is not None:
            return (key, "auto")
    if subtitles:
        return (next(iter(subtitles)), "manual")
    if auto:
        return (next(iter(auto)), "auto")
    return None


_TIMESTAMP_LINE = re.compile(
    r"^\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}.*$",
)
_INDEX_LINE = re.compile(r"^\d+$")
_VTT_HEADER_LINE = re.compile(r"^(WEBVTT|Kind:|Language:)")
_TAG = re.compile(r"<[^>]+>")


def _extract_text_from_subtitle_file(path: Path) -> str:
    """Strip timestamps + cue indices from an SRT/VTT subtitle file.

    Returns a single string with cue text joined by spaces, deduplicated
    against consecutive identical lines (auto captions are noisy this way).
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines: list[str] = []
    last: str = ""
    for raw_line in raw.splitlines():
        line = _TAG.sub("", raw_line).strip()
        if not line:
            continue
        if _TIMESTAMP_LINE.match(line) or _INDEX_LINE.match(line) or _VTT_HEADER_LINE.match(line):
            continue
        if line == last:
            continue
        lines.append(line)
        last = line
    return " ".join(lines)


def get_youtube_transcript(video_id: str, requested_lang: str | None = None) -> CaptionTrack:
    """Fetch a YouTube transcript honoring the language-selection ladder.

    Args:
        video_id: YouTube video ID (the part after ``v=`` or after
            ``youtu.be/``).
        requested_lang: Optional preferred language code (typically
            ``"fr"`` or ``"en"``). When set, manual @ this lang outranks
            manual @ English.

    Raises:
        TranscriptUnavailableError: No usable caption track is retrievable.
            Caller should route to the whisper-based fallback path.

    """
    url = _build_url(video_id)

    # Languages to enumerate in phase 1: requested lang + English + everything
    # else ("all").  Without hinting yt-dlp with writesubtitles opts it skips
    # the subtitle-metadata fetch entirely, so info["subtitles"] stays empty
    # even when manual captions exist.
    sub_langs: list[str] = []
    if requested_lang:
        sub_langs.append(requested_lang)
    if "en" not in sub_langs:
        sub_langs.append("en")
    sub_langs.append("all")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        # Phase 1: metadata only — populates info["subtitles"] /
        # info["automatic_captions"] without downloading any subtitle file.
        # writesubtitles/writeautomaticsub must be True so yt-dlp runs its
        # subtitle-metadata fetch; download=False ensures no files are written.
        info_opts: Any = {
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": sub_langs,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = cast(dict[str, Any], ydl.extract_info(url, download=False))
        except DownloadError as exc:
            my_logger.warning("yt-dlp could not retrieve video metadata")
            raise TranscriptUnavailableError(
                "list_failed",
                "yt-dlp could not retrieve caption metadata.",
            ) from exc

        manual_langs = sorted(cast(dict[str, Any], info.get("subtitles") or {}))
        auto_langs = sorted(cast(dict[str, Any], info.get("automatic_captions") or {}))
        track_list = (
            "; ".join(
                [f'"{k}" (manual)' for k in manual_langs] + [f'"{k}" (auto)' for k in auto_langs]
            )
            or "none"
        )
        my_logger.info(f"Caption tracks found: {track_list}")

        # Uploader-declared video language (BCP-47 from yt-dlp). Normalise to a
        # 2-letter code for downstream use.
        declared_raw = info.get("language")
        declared_lang: str | None = None
        if isinstance(declared_raw, str) and declared_raw.strip():
            declared_lang = declared_raw.strip().split("-")[0]

        # When the user didn't pass --language, use the declared language as
        # the implicit preference so the ladder prefers manual @ declared over
        # the manual-English fallback. Explicit --language always wins.
        effective_lang = requested_lang or declared_lang
        if effective_lang and effective_lang != requested_lang:
            my_logger.info(
                f"No --language set; using declared language '{declared_lang}' "
                "as the implicit preference.",
            )

        pick = _pick_caption(info, requested_lang=effective_lang)
        if pick is None:
            raise TranscriptUnavailableError(
                "lang_not_found",
                "No caption track found in any language.",
            )

        # lang_key is the raw yt-dlp dict key (e.g. "en-US"); lang is the
        # normalised 2-letter code used downstream (e.g. "en").
        lang_key, kind = pick
        lang = lang_key.split("-")[0]
        my_logger.info(f'Picking: "{lang_key}" ({kind}) as per language selection rules')

        # Phase 2: download only the chosen track.
        dl_opts: Any = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": [lang_key],
            "subtitlesformat": "srt",
            "outtmpl": str(tmpdir / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                ydl.extract_info(url, download=True)
        except DownloadError as exc:
            raise TranscriptUnavailableError(
                "download_failed",
                f"{kind} captions (key='{lang_key}') exist but download failed: {exc}",
            ) from exc

        srt_path = tmpdir / f"{video_id}.{lang_key}.srt"
        if not srt_path.exists():
            raise TranscriptUnavailableError(
                "empty_payload",
                f"yt-dlp listed {kind} captions in '{lang}' but produced no file.",
            )

        text = _extract_text_from_subtitle_file(srt_path)

    if not text.strip():
        raise TranscriptUnavailableError(
            "empty_payload",
            "Caption file existed but contained no usable text.",
        )

    wrapped = textwrap.fill(text, width=80, break_long_words=False, break_on_hyphens=False)
    return CaptionTrack(text=wrapped, lang=lang, kind=kind, declared_language=declared_lang)
