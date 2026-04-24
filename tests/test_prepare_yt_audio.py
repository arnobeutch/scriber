"""Tests for prepare_yt_audio — pure helpers (download/title fetch mocked)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from scriber.transcription.youtube_audio import (
    download_youtube_audio,
    extract_video_id,
    fetch_video_metadata,
    fetch_video_title,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestExtractVideoId:
    def test_watch_url(self) -> None:
        assert extract_video_id("https://www.youtube.com/watch?v=f8cfH5XX-XU") == "f8cfH5XX-XU"

    def test_watch_url_with_extra_params(self) -> None:
        assert extract_video_id("https://www.youtube.com/watch?v=abc123&t=30s&list=foo") == "abc123"

    def test_short_link(self) -> None:
        assert extract_video_id("https://youtu.be/f8cfH5XX-XU") == "f8cfH5XX-XU"

    def test_short_link_with_trailing_slash(self) -> None:
        assert extract_video_id("https://youtu.be/f8cfH5XX-XU/") == "f8cfH5XX-XU"

    def test_short_link_with_query(self) -> None:
        assert extract_video_id("https://youtu.be/f8cfH5XX-XU?t=10") == "f8cfH5XX-XU"

    def test_embed_url(self) -> None:
        assert extract_video_id("https://www.youtube.com/embed/f8cfH5XX-XU") == "f8cfH5XX-XU"

    def test_shorts_url(self) -> None:
        assert extract_video_id("https://www.youtube.com/shorts/f8cfH5XX-XU") == "f8cfH5XX-XU"

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not extract"):
            extract_video_id("https://example.com/something")


class TestFetchVideoTitle:
    def test_returns_title_from_info(self) -> None:
        with patch("scriber.transcription.youtube_audio.yt_dlp.YoutubeDL") as dl_cls:
            ctx = MagicMock()
            ctx.extract_info.return_value = {"title": "My Cool Video", "id": "abc"}
            dl_cls.return_value.__enter__.return_value = ctx
            assert fetch_video_title("https://youtu.be/abc") == "My Cool Video"

    def test_falls_back_to_id_when_no_title(self) -> None:
        with patch("scriber.transcription.youtube_audio.yt_dlp.YoutubeDL") as dl_cls:
            ctx = MagicMock()
            ctx.extract_info.return_value = {"id": "abc"}
            dl_cls.return_value.__enter__.return_value = ctx
            assert fetch_video_title("https://youtu.be/abc") == "abc"


class TestFetchVideoMetadata:
    def test_parses_chapters(self) -> None:
        with patch("scriber.transcription.youtube_audio.yt_dlp.YoutubeDL") as dl_cls:
            ctx = MagicMock()
            ctx.extract_info.return_value = {
                "title": "t",
                "id": "abc",
                "chapters": [
                    {"start_time": 0, "title": "Intro", "end_time": 60},
                    {"start_time": 60.5, "title": "Main", "end_time": 300},
                ],
            }
            dl_cls.return_value.__enter__.return_value = ctx
            title, chapters, _ = fetch_video_metadata("https://youtu.be/abc")
        assert title == "t"
        assert len(chapters) == 2
        assert chapters[0].start_time == 0
        assert chapters[0].title == "Intro"
        assert chapters[1].start_time == 60.5

    def test_empty_chapters_when_absent(self) -> None:
        with patch("scriber.transcription.youtube_audio.yt_dlp.YoutubeDL") as dl_cls:
            ctx = MagicMock()
            ctx.extract_info.return_value = {"title": "t", "id": "abc"}
            dl_cls.return_value.__enter__.return_value = ctx
            _, chapters, _ = fetch_video_metadata("https://youtu.be/abc")
        assert chapters == []

    def test_parses_channel_upload_date_duration(self) -> None:
        with patch("scriber.transcription.youtube_audio.yt_dlp.YoutubeDL") as dl_cls:
            ctx = MagicMock()
            ctx.extract_info.return_value = {
                "title": "t",
                "id": "abc",
                "channel": "Some Channel",
                "upload_date": "20260115",
                "duration": 142.7,
            }
            dl_cls.return_value.__enter__.return_value = ctx
            _, _, meta = fetch_video_metadata("https://youtu.be/abc")
        assert meta.channel == "Some Channel"
        assert meta.publication_date == "2026-01-15"
        assert meta.duration_seconds == 142.7

    def test_malformed_upload_date_becomes_none(self) -> None:
        with patch("scriber.transcription.youtube_audio.yt_dlp.YoutubeDL") as dl_cls:
            ctx = MagicMock()
            ctx.extract_info.return_value = {
                "title": "t",
                "id": "abc",
                "upload_date": "nope",
            }
            dl_cls.return_value.__enter__.return_value = ctx
            _, _, meta = fetch_video_metadata("https://youtu.be/abc")
        assert meta.publication_date is None

    def test_skips_malformed_entries(self) -> None:
        with patch("scriber.transcription.youtube_audio.yt_dlp.YoutubeDL") as dl_cls:
            ctx = MagicMock()
            ctx.extract_info.return_value = {
                "title": "t",
                "id": "abc",
                "chapters": [
                    {"start_time": 0, "title": "OK"},
                    {"start_time": "bad", "title": "Bad"},
                    {"title": ""},
                ],
            }
            dl_cls.return_value.__enter__.return_value = ctx
            _, chapters, _ = fetch_video_metadata("https://youtu.be/abc")
        assert len(chapters) == 1
        assert chapters[0].title == "OK"


class TestDownloadYoutubeAudio:
    def test_returns_wav_path_title_and_chapters(self, tmp_path: Path) -> None:
        out = tmp_path / "downloads"
        wav = out / "abc123.wav"

        def fake_extract(_url: str, *, download: bool) -> dict[str, object]:
            _ = download
            out.mkdir(parents=True, exist_ok=True)
            wav.write_bytes(b"")
            return {
                "id": "abc123",
                "title": "My Video",
                "chapters": [{"start_time": 0, "title": "Intro"}],
            }

        with patch("scriber.transcription.youtube_audio.yt_dlp.YoutubeDL") as dl_cls:
            ctx = MagicMock()
            ctx.extract_info.side_effect = fake_extract
            dl_cls.return_value.__enter__.return_value = ctx
            audio_path, title, chapters, metadata = download_youtube_audio(
                "https://youtu.be/abc123",
                out,
            )

        assert audio_path == wav
        assert title == "My Video"
        assert [c.title for c in chapters] == ["Intro"]
        assert audio_path.exists()
        # Metadata fields unset when yt-dlp info is minimal.
        assert metadata.channel is None
        assert metadata.publication_date is None

    def test_missing_output_file_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "downloads"
        with patch("scriber.transcription.youtube_audio.yt_dlp.YoutubeDL") as dl_cls:
            ctx = MagicMock()
            # Simulate yt-dlp reporting success but never creating the file
            ctx.extract_info.return_value = {"id": "missing", "title": "t"}
            dl_cls.return_value.__enter__.return_value = ctx
            with pytest.raises(FileNotFoundError, match="missing"):
                download_youtube_audio("https://youtu.be/missing", out)
