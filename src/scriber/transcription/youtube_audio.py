# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Download YouTube audio for local transcription when captions aren't available."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yt_dlp

from scriber.logger import my_logger
from scriber.model import Chapter, SourceMetadata


def extract_video_id(url: str) -> str:
    """Extract the video ID from any common YouTube URL form.

    Supported: ``watch?v=<id>``, ``youtu.be/<id>``, ``embed/<id>``, ``shorts/<id>``.
    """
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        return parsed.path.lstrip("/").split("/")[0]
    v = parse_qs(parsed.query).get("v")
    if v:
        return v[0]
    parts = parsed.path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] in {"embed", "shorts", "v"}:
        return parts[1]
    err_msg = f"Could not extract video ID from URL: {url}"
    raise ValueError(err_msg)


def _parse_chapters(info: dict[str, Any]) -> list[Chapter]:
    raw: list[dict[str, Any]] = list(info.get("chapters") or [])
    out: list[Chapter] = []
    for c in raw:
        title = str(c.get("title") or "").strip()
        if not title:
            continue
        try:
            start = float(c.get("start_time") or 0.0)
        except (TypeError, ValueError):
            continue
        out.append(Chapter(start_time=start, title=title))
    return out


def _normalize_upload_date(raw: str | None) -> str | None:
    """Convert yt-dlp's ``YYYYMMDD`` upload date to ISO ``YYYY-MM-DD``.

    Returns ``None`` for empty or malformed values.
    """
    _expected_len = 8
    if not raw:
        return None
    s = str(raw).strip()
    if len(s) != _expected_len or not s.isdigit():
        return None
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def _parse_metadata(info: dict[str, Any]) -> SourceMetadata:
    """Pull channel / upload_date / duration from yt-dlp info into ``SourceMetadata``."""
    channel_raw = info.get("channel") or info.get("uploader")
    duration_raw = info.get("duration")
    duration: float | None
    try:
        duration = float(duration_raw) if duration_raw is not None else None
    except (TypeError, ValueError):
        duration = None
    return SourceMetadata(
        channel=str(channel_raw).strip() if channel_raw else None,
        publication_date=_normalize_upload_date(info.get("upload_date")),
        duration_seconds=duration,
    )


def fetch_video_metadata(url: str) -> tuple[str, list[Chapter], SourceMetadata]:
    """Return ``(title, chapters, metadata)`` via yt-dlp metadata (no audio download)."""
    opts: Any = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noprogress": True,
        "extractor_args": {"youtube": {"player_client": ["default", "tv_simply"]}},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = cast(dict[str, Any], ydl.extract_info(url, download=False))
    title = cast(str, info.get("title") or info.get("id") or "unknown")
    return title, _parse_chapters(info), _parse_metadata(info)


def fetch_video_title(url: str) -> str:
    """Return the video title via yt-dlp metadata (no audio download)."""
    return fetch_video_metadata(url)[0]


def download_youtube_audio(
    url: str,
    output_dir: Path,
    *,
    force: bool = False,
) -> tuple[Path, str, list[Chapter], SourceMetadata]:
    """Download the audio track of a YouTube video as a wav file.

    Args:
        url: Full YouTube URL.
        output_dir: Directory to save the downloaded wav file in.
        force: If True, re-download even if the .wav already exists.

    Returns:
        ``(audio_path, video_title, chapters, metadata)`` — path to the
        downloaded wav, the video's unsanitized title, its chapter list
        (empty when absent), and provenance metadata
        (channel / publication_date / duration).

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache hit: the .wav for this video id already exists.
    if not force:
        video_id = extract_video_id(url)
        cached_wav = output_dir / f"{video_id}.wav"
        if cached_wav.exists():
            my_logger.info(f"Using cached audio at {cached_wav}")
            title, chapters, metadata = fetch_video_metadata(url)
            return cached_wav, title, chapters, metadata

    my_logger.info(f"Downloading audio from {url}")

    # Let yt-dlp stream its native progress line to stdout — long downloads
    # are opaque without it. Warnings and non-progress chatter stay silenced.
    opts: Any = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            },
        ],
        "quiet": False,
        "no_warnings": True,
        "noprogress": False,
        "extractor_args": {"youtube": {"player_client": ["default", "tv_simply"]}},
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = cast(dict[str, Any], ydl.extract_info(url, download=True))
    video_id = cast(str, info["id"])
    title = cast(str, info.get("title") or video_id)
    chapters = _parse_chapters(info)
    metadata = _parse_metadata(info)
    audio_path = output_dir / f"{video_id}.wav"
    if not audio_path.exists():
        err_msg = f"yt-dlp reported success but {audio_path} is missing"
        raise FileNotFoundError(err_msg)
    return audio_path, title, chapters, metadata
