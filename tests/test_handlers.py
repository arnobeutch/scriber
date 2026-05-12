"""Tests for the per-source handlers (URL / media / text) + write/summarize."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from scriber.handlers import (
    Transcript,
    handle_media,
    handle_text,
    handle_url,
    summarize,
    write_transcript_file,
)
from scriber.model import SourceMetadata
from scriber.settings import Settings
from scriber.transcription.youtube_captions import CaptionTrack, TranscriptUnavailableError


def _track(
    text: str = "caption text",
    lang: str = "en",
    kind: str = "manual",
    declared_language: str | None = None,
) -> CaptionTrack:
    return CaptionTrack(
        text=text,
        lang=lang,
        kind=kind,  # type: ignore[arg-type]  # kind is Literal in production
        declared_language=declared_language,
    )


def _md(**overrides: object) -> SourceMetadata:
    """Empty-by-default SourceMetadata for mocked yt-dlp returns."""
    return SourceMetadata(**overrides)  # type: ignore[arg-type]


if TYPE_CHECKING:
    from pathlib import Path


def _args(**overrides: object) -> MagicMock:
    defaults: dict[str, object] = {
        "input_path": "",
        "language": None,  # parser default — autodetect
        "diarize": False,
        "with_openai": False,
        "force": False,
        "model_size": None,
        "llm_provider": None,
        "llm_model": None,
        "context_file": None,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "log_level": "INFO",
        "openai_api_key": None,
        "openrouter_api_key": None,
        "huggingface_token": None,
        "llm_provider": "openai",
        "llm_model": None,
        "openai_model": "gpt-4o",
        "ollama_model": "mistral",
        "whisper_model_size": "small",
        "wrap_width": 80,
        "summary_mode": "auto",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]  # frozen dataclass kwargs


class TestHandleUrl:
    def test_caption_happy_path_manual_fr(
        self,
        tmp_path: Path,
    ) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="vid"),
            patch(
                "scriber.handlers.pya.fetch_video_metadata",
                return_value=("My Video", [], _md()),
            ),
            patch(
                "scriber.handlers.pytt.get_youtube_transcript",
                return_value=_track(text="bonjour", lang="fr", kind="manual"),
            ),
        ):
            t = handle_url(_args(input_path="https://y.com/watch?v=vid", language="fr"), s)
        assert t.text == "bonjour"
        assert t.title == "My Video"
        assert t.source == "yt_manual"
        assert t.language == "fr"
        assert t.diarized is False

    def test_declared_lang_drives_summary_when_no_cli_flag(self, tmp_path: Path) -> None:
        # Regression for the Vlmn6j_bIhI case: declared-fr video, no --language
        # set. The picker chose manual fr; the summary should follow.
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="vid"),
            patch(
                "scriber.handlers.pya.fetch_video_metadata",
                return_value=("Le Titre", [], _md()),
            ),
            patch(
                "scriber.handlers.pytt.get_youtube_transcript",
                return_value=_track(
                    text="bonjour",
                    lang="fr",
                    kind="manual",
                    declared_language="fr",
                ),
            ),
        ):
            t = handle_url(_args(input_path="https://y.com/watch?v=vid"), s)
        assert t.language == "fr"  # would be "en" before the fix
        assert t.text == "bonjour"
        assert t.source == "yt_manual"

    def test_caption_other_lang_forces_summary_in_english(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="vid"),
            patch("scriber.handlers.pya.fetch_video_metadata", return_value=("V", [], _md())),
            patch(
                "scriber.handlers.pytt.get_youtube_transcript",
                return_value=_track(text="hallo", lang="de", kind="manual"),
            ),
        ):
            t = handle_url(_args(input_path="https://y.com/watch?v=vid", language="fr"), s)
        # Caption is German → summary forced to English.
        assert t.language == "en"

    def test_caption_auto_marked_as_yt_auto(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="vid"),
            patch("scriber.handlers.pya.fetch_video_metadata", return_value=("V", [], _md())),
            patch(
                "scriber.handlers.pytt.get_youtube_transcript",
                return_value=_track(text="x", lang="en", kind="auto"),
            ),
        ):
            t = handle_url(_args(input_path="https://y.com/watch?v=vid"), s)
        assert t.source == "yt_auto"

    def test_falls_back_to_whisper_when_unavailable(
        self,
        tmp_path: Path,
    ) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="vid"),
            patch(
                "scriber.handlers.pytt.get_youtube_transcript",
                side_effect=TranscriptUnavailableError("lang_not_found", "no caps"),
            ),
            patch(
                "scriber.handlers.pya.download_youtube_audio",
                return_value=(tmp_path / "audio.wav", "Remote Video", [], _md()),
            ),
            patch(
                "scriber.handlers.plt.transcribe_audio_full",
                return_value=("transcribed body", "fr", []),
            ) as transcribe,
        ):
            t = handle_url(_args(input_path="https://y.com/watch?v=vid"), s)
        assert t.text == "transcribed body"
        assert t.language == "fr"  # whisper detected fr; summary follows
        assert t.title == "Remote Video"
        assert t.source == "whisper"
        # Default model size from Settings ("small"); language=None means autodetect.
        transcribe.assert_called_once_with(
            str(tmp_path / "audio.wav"),
            model_size="small",
            language=None,
        )

    def test_diarize_flag_skips_captions_for_yt_urls(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="vid"),
            patch("scriber.handlers.pytt.get_youtube_transcript") as get_captions,
            patch(
                "scriber.handlers.pya.download_youtube_audio",
                return_value=(tmp_path / "audio.wav", "Vid", [], _md()),
            ),
            patch(
                "scriber.handlers.plt.transcribe_audio_with_diarization",
                return_value=("SPEAKER_00: hello", "en"),
            ) as transcribe,
        ):
            t = handle_url(
                _args(input_path="https://y.com/watch?v=vid", diarize=True),
                s,
            )
        # Captions are bypassed entirely when diarization is requested.
        get_captions.assert_not_called()
        transcribe.assert_called_once()
        assert t.diarized is True
        assert t.source == "whisper"

    def test_fallback_with_diarization(
        self,
        tmp_path: Path,
    ) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="vid"),
            patch(
                "scriber.handlers.pytt.get_youtube_transcript",
                side_effect=TranscriptUnavailableError("empty_payload", "empty"),
            ),
            patch(
                "scriber.handlers.pya.download_youtube_audio",
                return_value=(tmp_path / "audio.wav", "Diarized", [], _md()),
            ),
            patch(
                "scriber.handlers.plt.transcribe_audio_with_diarization",
                return_value=("Alice: hi", "en"),
            ) as transcribe,
        ):
            t = handle_url(
                _args(input_path="https://y.com/watch?v=vid", diarize=True),
                s,
            )
        assert t.diarized is True
        assert t.text == "Alice: hi"
        transcribe.assert_called_once()

    def test_settings_model_size_propagates(
        self,
        tmp_path: Path,
    ) -> None:
        s = _settings(
            output_dir=tmp_path / "out",
            downloads_dir=tmp_path / "dl",
            whisper_model_size="medium",
        )
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="vid"),
            patch(
                "scriber.handlers.pytt.get_youtube_transcript",
                side_effect=TranscriptUnavailableError("lang_not_found", "x"),
            ),
            patch(
                "scriber.handlers.pya.download_youtube_audio",
                return_value=(tmp_path / "audio.wav", "X", [], _md()),
            ),
            patch(
                "scriber.handlers.plt.transcribe_audio_full",
                return_value=("body", "en", []),
            ) as transcribe,
        ):
            handle_url(_args(input_path="https://y.com/watch?v=vid"), s)
        transcribe.assert_called_once_with(
            str(tmp_path / "audio.wav"),
            model_size="medium",
            language=None,
        )

    def test_fallback_forces_requested_language_to_whisper(
        self,
        tmp_path: Path,
    ) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="vid"),
            patch(
                "scriber.handlers.pytt.get_youtube_transcript",
                side_effect=TranscriptUnavailableError("lang_not_found", "x"),
            ),
            patch(
                "scriber.handlers.pya.download_youtube_audio",
                return_value=(tmp_path / "audio.wav", "V", [], _md()),
            ),
            patch(
                "scriber.handlers.plt.transcribe_audio_full",
                return_value=("corps", "fr", []),
            ) as transcribe,
        ):
            t = handle_url(_args(input_path="https://y.com/watch?v=vid", language="fr"), s)
        # Whisper called with language=fr (forced), and summary lang follows.
        transcribe.assert_called_once_with(
            str(tmp_path / "audio.wav"),
            model_size="small",
            language="fr",
        )
        assert t.language == "fr"

    def test_title_sanitized(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", downloads_dir=tmp_path / "dl")
        with (
            patch("scriber.handlers.pya.extract_video_id", return_value="vid"),
            patch(
                "scriber.handlers.pya.fetch_video_metadata",
                return_value=("Bad/Name:Here", [], _md()),
            ),
            patch("scriber.handlers.pytt.get_youtube_transcript", return_value=_track()),
        ):
            t = handle_url(_args(input_path="https://y.com/watch?v=vid"), s)
        assert t.title == "Bad_Name_Here"


class TestHandleMedia:
    def test_non_diarize_autodetect(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        media = tmp_path / "video.mp4"
        media.write_text("")
        audio_tmp = str(tmp_path / "audio.wav")
        with (
            patch("scriber.handlers.plt.extract_audio", return_value=audio_tmp),
            patch(
                "scriber.handlers.plt.transcribe_audio_full",
                return_value=("Hello world", "en", []),
            ) as transcribe,
            patch("scriber.handlers.Path.unlink"),
        ):
            t = handle_media(_args(input_path=str(media), language=None), s)
        assert t.text == "Hello world"
        assert t.language == "en"  # detected en → summary en
        assert t.title == "video"
        assert t.source == "whisper"
        transcribe.assert_called_once_with(audio_tmp, model_size="small", language=None)

    def test_explicit_language_forces_whisper(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        media = tmp_path / "video.mp4"
        media.write_text("")
        audio_tmp = str(tmp_path / "audio.wav")
        with (
            patch("scriber.handlers.plt.extract_audio", return_value=audio_tmp),
            patch(
                "scriber.handlers.plt.transcribe_audio_full",
                return_value=("bonjour", "fr", []),
            ) as transcribe,
            patch("scriber.handlers.Path.unlink"),
        ):
            t = handle_media(_args(input_path=str(media), language="fr"), s)
        transcribe.assert_called_once_with(audio_tmp, model_size="small", language="fr")
        assert t.language == "fr"

    def test_detected_other_language_summary_in_english(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        media = tmp_path / "video.mp4"
        media.write_text("")
        audio_tmp = str(tmp_path / "audio.wav")
        with (
            patch("scriber.handlers.plt.extract_audio", return_value=audio_tmp),
            patch("scriber.handlers.plt.transcribe_audio_full", return_value=("hallo", "de", [])),
            patch("scriber.handlers.Path.unlink"),
        ):
            t = handle_media(_args(input_path=str(media), language=None), s)
        assert t.language == "en"

    def test_diarize(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        media = tmp_path / "video.mp4"
        media.write_text("")
        with patch(
            "scriber.handlers.plt.transcribe_video_file_with_diarization",
            return_value=("Alice: hi", "en"),
        ):
            t = handle_media(
                _args(input_path=str(media), language=None, diarize=True),
                s,
            )
        assert t.diarized is True
        assert t.text == "Alice: hi"


class TestHandleText:
    def test_reads_file_with_explicit_language(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        txt = tmp_path / "t.txt"
        txt.write_text("Alice: hi", encoding="utf-8")
        result = handle_text(_args(input_path=str(txt), language="en"), s)
        assert result.text == "Alice: hi"
        assert result.title == "t"
        assert result.source == "file"
        assert result.language == "en"

    def test_reads_file_autodetect_french(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        txt = tmp_path / "t.txt"
        txt.write_text(
            "Bonjour, comment allez-vous aujourd'hui ? J'espère que vous allez bien.",
            encoding="utf-8",
        )
        result = handle_text(_args(input_path=str(txt), language=None), s)
        # langdetect should pick fr; summary follows.
        assert result.language == "fr"

    def test_reads_file_autodetect_failure_defaults_to_en(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        txt = tmp_path / "t.txt"
        # Empty / punctuation-only text trips langdetect.
        txt.write_text("...", encoding="utf-8")
        result = handle_text(_args(input_path=str(txt), language=None), s)
        assert result.language == "en"


class TestWriteTranscriptFile:
    def test_writes_non_diarized(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        t = Transcript(text="hello", language="en", title="vid", source="yt_manual", diarized=False)
        out = write_transcript_file(t, s)
        assert out == tmp_path / "out" / "vid transcript.txt"
        assert out.read_text() == "hello"

    def test_writes_diarized(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        t = Transcript(text="A: x", language="en", title="vid", source="whisper", diarized=True)
        out = write_transcript_file(t, s)
        assert out == tmp_path / "out" / "vid diarized transcript.txt"
        assert out.read_text() == "A: x"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "deep" / "nested")
        t = Transcript(text="x", language="en", title="t", source="file", diarized=False)
        write_transcript_file(t, s)
        assert (tmp_path / "deep" / "nested").is_dir()

    def test_subtitles_written_when_segments_present(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        t = Transcript(
            text="hello",
            language="en",
            title="vid",
            source="whisper",
            diarized=False,
            segments=[{"start": 0.0, "end": 1.0, "text": "hello"}],
        )
        write_transcript_file(t, s, subtitles=True)
        assert (tmp_path / "out" / "vid.srt").exists()
        assert (tmp_path / "out" / "vid.vtt").exists()

    def test_subtitles_skipped_when_no_segments(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        t = Transcript(
            text="hello",
            language="en",
            title="vid",
            source="yt_manual",  # no segments for caption tracks
            diarized=False,
        )
        write_transcript_file(t, s, subtitles=True)
        assert not (tmp_path / "out" / "vid.srt").exists()


class TestSummarize:
    def test_dispatches_through_factory(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        t = Transcript(
            text="body",
            language="en",
            title="title",
            source="yt_manual",
            diarized=False,
        )
        fake = MagicMock()
        with patch("scriber.handlers.make_summarizer", return_value=fake):
            summarize(t, _args(input_path="u"), s)
        fake.summarize.assert_called_once_with(t, input_path="u", context=None)

    def test_context_file_read_and_passed(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        ctx_file = tmp_path / "context.txt"
        ctx_file.write_text("  glossary: foo = bar  \n", encoding="utf-8")
        t = Transcript(
            text="body",
            language="en",
            title="t",
            source="file",
            diarized=False,
        )
        fake = MagicMock()
        with patch("scriber.handlers.make_summarizer", return_value=fake):
            summarize(t, _args(input_path="u", context_file=ctx_file), s)
        _, kwargs = fake.summarize.call_args
        assert kwargs["context"] == "glossary: foo = bar"

    def test_missing_context_file_passes_none(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        t = Transcript(
            text="body",
            language="en",
            title="t",
            source="file",
            diarized=False,
        )
        fake = MagicMock()
        with patch("scriber.handlers.make_summarizer", return_value=fake):
            summarize(
                t,
                _args(input_path="u", context_file=tmp_path / "nope.txt"),
                s,
            )
        _, kwargs = fake.summarize.call_args
        assert kwargs["context"] is None

    def test_empty_context_file_passes_none(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        empty = tmp_path / "empty.txt"
        empty.write_text("   \n\n", encoding="utf-8")
        t = Transcript(
            text="body",
            language="en",
            title="t",
            source="file",
            diarized=False,
        )
        fake = MagicMock()
        with patch("scriber.handlers.make_summarizer", return_value=fake):
            summarize(t, _args(input_path="u", context_file=empty), s)
        _, kwargs = fake.summarize.call_args
        assert kwargs["context"] is None
