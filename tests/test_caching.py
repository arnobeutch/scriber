"""Tests for smart caching: skip download / skip transcription on re-runs."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from scriber.handlers import handle_url
from scriber.model import SourceMetadata
from scriber.settings import Settings
from scriber.transcription.youtube_audio import download_youtube_audio
from scriber.transcription.youtube_captions import TranscriptUnavailableError


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        log_level="INFO",
        openai_api_key=None,
        openrouter_api_key=None,
        huggingface_token=None,
        llm_provider="openai",
        llm_model=None,
        openai_model="gpt-4o",
        ollama_model="mistral",
        whisper_model_size="small",
        output_dir=tmp_path / "out",
        downloads_dir=tmp_path / "dl",
        wrap_width=80,
        summary_mode="auto",
    )


def _args(**overrides: object) -> MagicMock:
    defaults: dict[str, object] = {
        "input_path": "https://youtu.be/abc",
        "language": None,
        "diarize": False,
        "with_openai": False,
        "force": False,
        "model_size": None,
        "llm_provider": None,
        "llm_model": None,
        "suggest_primer": False,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


class TestDownloadCache:
    def test_returns_cached_wav_without_calling_yt_dlp(self, tmp_path: Path) -> None:
        out = tmp_path / "dl"
        out.mkdir()
        (out / "abc.wav").write_bytes(b"")
        from scriber.model import SourceMetadata

        with (
            patch(
                "scriber.transcription.youtube_audio.fetch_video_metadata",
                return_value=("Cached Title", [], SourceMetadata()),
            ) as fetch,
            patch("scriber.transcription.youtube_audio.yt_dlp.YoutubeDL") as ydl,
        ):
            audio_path, title, chapters, _ = download_youtube_audio("https://youtu.be/abc", out)
        assert audio_path == out / "abc.wav"
        assert title == "Cached Title"
        assert chapters == []
        ydl.assert_not_called()
        fetch.assert_called_once()

    def test_force_bypasses_cache(self, tmp_path: Path) -> None:
        out = tmp_path / "dl"
        out.mkdir()
        cached = out / "abc.wav"
        cached.write_bytes(b"")
        ctx = MagicMock()
        ctx.__enter__.return_value = ctx
        ctx.__exit__.return_value = False

        def _extract_info(_url: str, *, download: bool = True) -> dict[str, object]:
            _ = _url, download
            cached.write_bytes(b"new")
            return {"id": "abc", "title": "Re-downloaded"}

        ctx.extract_info.side_effect = _extract_info
        with patch("scriber.transcription.youtube_audio.yt_dlp.YoutubeDL", return_value=ctx) as ydl:
            audio_path, title, *_ = download_youtube_audio(
                "https://youtu.be/abc",
                out,
                force=True,
            )
        ydl.assert_called_once()
        assert audio_path == cached
        assert title == "Re-downloaded"


class TestHandleUrlCachedTranscript:
    def test_skips_transcription_when_transcript_exists(self, tmp_path: Path) -> None:
        s = _settings(tmp_path)
        s.output_dir.mkdir(parents=True, exist_ok=True)
        # Pre-existing transcript file matching the title download will produce.
        (s.output_dir / "Vid transcript.txt").write_text("cached body", encoding="utf8")
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="abc"),
            patch(
                "scriber.handlers.pytt.get_youtube_transcript",
                side_effect=TranscriptUnavailableError("lang_not_found", "x"),
            ),
            patch(
                "scriber.handlers.pya.download_youtube_audio",
                return_value=(s.downloads_dir / "abc.wav", "Vid", [], SourceMetadata()),
            ),
            patch("scriber.handlers.plt.transcribe_audio_full") as transcribe,
        ):
            t = handle_url(_args(), s)
        transcribe.assert_not_called()
        assert t.text == "cached body"

    def test_force_bypasses_transcript_cache(self, tmp_path: Path) -> None:
        s = _settings(tmp_path)
        s.output_dir.mkdir(parents=True, exist_ok=True)
        (s.output_dir / "Vid transcript.txt").write_text("STALE", encoding="utf8")
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="abc"),
            patch(
                "scriber.handlers.pytt.get_youtube_transcript",
                side_effect=TranscriptUnavailableError("lang_not_found", "x"),
            ),
            patch(
                "scriber.handlers.pya.download_youtube_audio",
                return_value=(s.downloads_dir / "abc.wav", "Vid", [], SourceMetadata()),
            ),
            patch(
                "scriber.handlers.plt.transcribe_audio_full",
                return_value=("fresh body", "en", []),
            ) as transcribe,
        ):
            t = handle_url(_args(force=True), s)
        transcribe.assert_called_once()
        assert t.text == "fresh body"
