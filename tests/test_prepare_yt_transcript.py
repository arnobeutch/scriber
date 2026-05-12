# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Tests for prepare_yt_transcript (yt-dlp backed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from yt_dlp.utils import DownloadError

from scriber.transcription.youtube_captions import (
    CaptionTrack,
    TranscriptUnavailableError,
    _extract_text_from_subtitle_file,
    _pick_caption,
    get_youtube_transcript,
)

# A tiny SRT fixture used by the integration of get_youtube_transcript.
_SAMPLE_SRT = """\
1
00:00:00,000 --> 00:00:02,000
Hello world

2
00:00:02,500 --> 00:00:05,000
This is a transcript

3
00:00:05,500 --> 00:00:07,000
This is a transcript
"""


def _ydl_returning(
    info: dict[str, Any],
    write_files: dict[str, str] | None = None,
) -> MagicMock:
    """Build a YoutubeDL mock that mimics extract_info + writes captions to disk.

    ``write_files`` maps relative filename (e.g. ``"abc.fr.srt"``) to its
    contents; on each ``extract_info`` call the mock writes them under the
    output template's parent directory (parsed from ``opts['outtmpl']``).
    """
    files_to_write = write_files or {}

    def _ctor(opts: dict[str, Any]) -> MagicMock:
        ctx = MagicMock()
        outtmpl = str(opts.get("outtmpl", ""))
        outdir = Path(outtmpl).parent if outtmpl else Path()

        def _extract_info(_url: str, *, download: bool = True) -> dict[str, Any]:
            if download:  # Phase 1 (download=False) must not write files
                for name, body in files_to_write.items():
                    (outdir / name).write_text(body, encoding="utf-8")
            return info

        ctx.extract_info.side_effect = _extract_info
        ctx.__enter__.return_value = ctx
        ctx.__exit__.return_value = False
        return ctx

    return MagicMock(side_effect=_ctor)


class TestPickCaption:
    """Caption-track ladder: manual > auto across languages, requested > en > other."""

    def test_rung_1_manual_at_requested(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {"fr": [{}], "en": [{}]},
            "automatic_captions": {"fr": [{}], "en": [{}]},
        }
        assert _pick_caption(info, requested_lang="fr") == ("fr", "manual")

    def test_rung_2_auto_at_requested(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {},
            "automatic_captions": {"fr": [{}], "en": [{}]},
        }
        assert _pick_caption(info, requested_lang="fr") == ("fr", "auto")

    def test_rung_3_manual_at_en_when_requested_absent(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {"en": [{}]},
            "automatic_captions": {"fr": [{}]},
        }
        # Manual EN beats auto FR even when FR was requested.
        assert _pick_caption(info, requested_lang="fr") == ("en", "manual")

    def test_rung_4_auto_at_en_when_no_manual(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {},
            "automatic_captions": {"en": [{}]},
        }
        assert _pick_caption(info, requested_lang="fr") == ("en", "auto")

    def test_rung_5_manual_other_language(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {"de": [{}]},
            "automatic_captions": {},
        }
        assert _pick_caption(info, requested_lang="fr") == ("de", "manual")

    def test_rung_6_auto_other_language(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {},
            "automatic_captions": {"de": [{}]},
        }
        assert _pick_caption(info, requested_lang="fr") == ("de", "auto")

    def test_no_captions_returns_none(self) -> None:
        empty: dict[str, Any] = {"subtitles": {}, "automatic_captions": {}}
        assert _pick_caption(empty) is None

    def test_no_request_defaults_to_en(self) -> None:
        info: dict[str, Any] = {
            "subtitles": {"en": [{}], "fr": [{}]},
            "automatic_captions": {},
        }
        # Without requested_lang, en is the implicit preference.
        assert _pick_caption(info) == ("en", "manual")

    def test_prefix_match_en_us_beats_auto_en(self) -> None:
        # Real-world case: manual track is "en-US", auto is "en".
        # Without prefix matching the picker misses "en-US" and falls through
        # to auto "en".  With prefix matching, manual "en-US" wins.
        info: dict[str, Any] = {
            "subtitles": {"en-US": [{}]},
            "automatic_captions": {"en": [{}]},
        }
        assert _pick_caption(info) == ("en-US", "manual")

    def test_prefix_match_requested_lang(self) -> None:
        # Requested "fr" should match "fr-FR" in subtitles.
        info: dict[str, Any] = {
            "subtitles": {"fr-FR": [{}]},
            "automatic_captions": {"fr": [{}]},
        }
        assert _pick_caption(info, requested_lang="fr") == ("fr-FR", "manual")


class TestExtractTextFromSubtitleFile:
    def test_strips_timestamps_and_indices(self, tmp_path: Path) -> None:
        srt = tmp_path / "x.srt"
        srt.write_text(_SAMPLE_SRT)
        text = _extract_text_from_subtitle_file(srt)
        assert "Hello world" in text
        assert "transcript" in text
        # No timestamp/index leftovers.
        assert "-->" not in text
        assert "00:00" not in text

    def test_deduplicates_consecutive_identical_lines(self, tmp_path: Path) -> None:
        srt = tmp_path / "x.srt"
        srt.write_text(_SAMPLE_SRT)
        text = _extract_text_from_subtitle_file(srt)
        # "This is a transcript" appears twice in source, once after dedup.
        assert text.count("This is a transcript") == 1

    def test_strips_vtt_header(self, tmp_path: Path) -> None:
        vtt = tmp_path / "x.vtt"
        vtt.write_text(
            "WEBVTT\nKind: captions\nLanguage: en\n\n00:00:00.000 --> 00:00:01.000\nHello\n",
        )
        assert _extract_text_from_subtitle_file(vtt) == "Hello"

    def test_strips_inline_tags(self, tmp_path: Path) -> None:
        vtt = tmp_path / "x.vtt"
        vtt.write_text(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n<c.color1>Hello</c> world\n",
        )
        assert _extract_text_from_subtitle_file(vtt) == "Hello world"


class TestGetYoutubeTranscript:
    def test_happy_path_manual_fr(self) -> None:
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {"fr": [{}]},
            "automatic_captions": {},
        }
        ydl = _ydl_returning(info, write_files={"abc.fr.srt": _SAMPLE_SRT})
        with patch("scriber.transcription.youtube_captions.yt_dlp.YoutubeDL", ydl):
            track = get_youtube_transcript("abc", requested_lang="fr")
        assert isinstance(track, CaptionTrack)
        assert track.lang == "fr"
        assert track.kind == "manual"
        assert "Hello world" in track.text
        for line in track.text.splitlines():
            assert len(line) <= 80

    def test_happy_path_auto_en_when_no_manual(self) -> None:
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {},
            "automatic_captions": {"en": [{}]},
        }
        ydl = _ydl_returning(info, write_files={"abc.en.srt": _SAMPLE_SRT})
        with patch("scriber.transcription.youtube_captions.yt_dlp.YoutubeDL", ydl):
            track = get_youtube_transcript("abc")
        assert track.lang == "en"
        assert track.kind == "auto"
        assert "Hello world" in track.text

    def test_requested_lang_propagates_to_picker(self) -> None:
        # Both manual fr and auto fr present + manual en present.
        # Requested fr should pick manual fr (rung 1), not manual en (rung 3).
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {"fr": [{}], "en": [{}]},
            "automatic_captions": {},
        }
        ydl = _ydl_returning(info, write_files={"abc.fr.srt": _SAMPLE_SRT})
        with patch("scriber.transcription.youtube_captions.yt_dlp.YoutubeDL", ydl):
            track = get_youtube_transcript("abc", requested_lang="fr")
        assert track.lang == "fr"

    def test_declared_language_used_as_implicit_request(self) -> None:
        # Mirrors the Vlmn6j_bIhI case: French video with manual EN+FR subs,
        # no --language flag. Without the implicit-from-declared fallback the
        # picker would land on manual EN; with it, we prefer the declared lang.
        info: dict[str, Any] = {
            "id": "abc",
            "language": "fr",
            "subtitles": {"en": [{}], "fr": [{}]},
            "automatic_captions": {"en": [{}]},
        }
        ydl = _ydl_returning(info, write_files={"abc.fr.srt": _SAMPLE_SRT})
        with patch("scriber.transcription.youtube_captions.yt_dlp.YoutubeDL", ydl):
            track = get_youtube_transcript("abc")  # no requested_lang
        assert track.lang == "fr"
        assert track.kind == "manual"
        assert track.declared_language == "fr"

    def test_explicit_request_overrides_declared(self) -> None:
        # --language en against a declared-fr video should still pick EN.
        info: dict[str, Any] = {
            "id": "abc",
            "language": "fr",
            "subtitles": {"en": [{}], "fr": [{}]},
            "automatic_captions": {},
        }
        ydl = _ydl_returning(info, write_files={"abc.en.srt": _SAMPLE_SRT})
        with patch("scriber.transcription.youtube_captions.yt_dlp.YoutubeDL", ydl):
            track = get_youtube_transcript("abc", requested_lang="en")
        assert track.lang == "en"
        assert track.declared_language == "fr"

    def test_declared_language_bcp47_normalized(self) -> None:
        # ``info["language"]`` may be a BCP-47 tag like ``"pt-BR"``; we keep
        # just the base 2-letter code on the CaptionTrack.
        info: dict[str, Any] = {
            "id": "abc",
            "language": "pt-BR",
            "subtitles": {"en": [{}], "pt": [{}]},
            "automatic_captions": {},
        }
        ydl = _ydl_returning(info, write_files={"abc.pt.srt": _SAMPLE_SRT})
        with patch("scriber.transcription.youtube_captions.yt_dlp.YoutubeDL", ydl):
            track = get_youtube_transcript("abc")
        assert track.lang == "pt"
        assert track.declared_language == "pt"

    def test_no_captions_raises_lang_not_found(self) -> None:
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {},
            "automatic_captions": {},
        }
        ydl = _ydl_returning(info)
        with (
            patch("scriber.transcription.youtube_captions.yt_dlp.YoutubeDL", ydl),
            pytest.raises(TranscriptUnavailableError) as excinfo,
        ):
            get_youtube_transcript("abc")
        assert excinfo.value.reason == "lang_not_found"

    def test_download_error_raises_list_failed(self) -> None:
        ctx = MagicMock()
        ctx.extract_info.side_effect = DownloadError("boom")
        ctx.__enter__.return_value = ctx
        ctx.__exit__.return_value = False
        with (
            patch(
                "scriber.transcription.youtube_captions.yt_dlp.YoutubeDL",
                MagicMock(return_value=ctx),
            ),
            pytest.raises(TranscriptUnavailableError) as excinfo,
        ):
            get_youtube_transcript("abc")
        assert excinfo.value.reason == "list_failed"

    def test_caption_download_error_raises_download_failed(self) -> None:
        # Phase 1 (info, download=False) succeeds; Phase 2 (download=True) gets 429.
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {"en": [{}]},
            "automatic_captions": {},
        }
        ctx_info = MagicMock()
        ctx_info.extract_info.return_value = info
        ctx_info.__enter__.return_value = ctx_info
        ctx_info.__exit__.return_value = False

        ctx_dl = MagicMock()
        ctx_dl.extract_info.side_effect = DownloadError("429")
        ctx_dl.__enter__.return_value = ctx_dl
        ctx_dl.__exit__.return_value = False

        with (
            patch(
                "scriber.transcription.youtube_captions.yt_dlp.YoutubeDL",
                MagicMock(side_effect=[ctx_info, ctx_dl]),
            ),
            pytest.raises(TranscriptUnavailableError) as excinfo,
        ):
            get_youtube_transcript("abc")
        assert excinfo.value.reason == "download_failed"

    def test_listed_but_no_file_raises_empty_payload(self) -> None:
        # Track is listed but yt-dlp doesn't actually write the file.
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {"fr": [{}]},
            "automatic_captions": {},
        }
        ydl = _ydl_returning(info, write_files={})
        with (
            patch("scriber.transcription.youtube_captions.yt_dlp.YoutubeDL", ydl),
            pytest.raises(TranscriptUnavailableError) as excinfo,
        ):
            get_youtube_transcript("abc")
        assert excinfo.value.reason == "empty_payload"

    def test_empty_caption_file_raises_empty_payload(self) -> None:
        info: dict[str, Any] = {
            "id": "abc",
            "subtitles": {"fr": [{}]},
            "automatic_captions": {},
        }
        ydl = _ydl_returning(info, write_files={"abc.fr.srt": "WEBVTT\n\n"})
        with (
            patch("scriber.transcription.youtube_captions.yt_dlp.YoutubeDL", ydl),
            pytest.raises(TranscriptUnavailableError) as excinfo,
        ):
            get_youtube_transcript("abc")
        assert excinfo.value.reason == "empty_payload"
