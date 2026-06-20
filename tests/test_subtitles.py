"""Tests for SRT / VTT writers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scriber.subtitles import _format_timestamp, write_srt, write_vtt

if TYPE_CHECKING:
    from pathlib import Path


_SEGMENTS = [
    {"start": 0.0, "end": 2.4, "text": " Hello world"},
    {"start": 2.5, "end": 5.123, "text": "Second cue"},
]


class TestFormatTimestamp:
    def test_zero(self) -> None:
        assert _format_timestamp(0.0, separator=",") == "00:00:00,000"

    def test_srt_separator(self) -> None:
        assert _format_timestamp(3661.5, separator=",") == "01:01:01,500"

    def test_vtt_separator(self) -> None:
        assert _format_timestamp(3661.5, separator=".") == "01:01:01.500"

    def test_negative_clamps_to_zero(self) -> None:
        assert _format_timestamp(-1.0, separator=",") == "00:00:00,000"

    def test_milliseconds_rounding(self) -> None:
        assert _format_timestamp(0.001, separator=",") == "00:00:00,001"


class TestWriteSrt:
    def test_format(self, tmp_path: Path) -> None:
        path = tmp_path / "out.srt"
        write_srt(_SEGMENTS, path)
        body = path.read_text(encoding="utf-8")
        # First cue
        assert body.startswith("1\n00:00:00,000 --> 00:00:02,400\nHello world\n")
        # Second cue uses comma separator
        assert "2\n00:00:02,500 --> 00:00:05,123\nSecond cue\n" in body


class TestWriteVtt:
    def test_format(self, tmp_path: Path) -> None:
        path = tmp_path / "out.vtt"
        write_vtt(_SEGMENTS, path)
        body = path.read_text(encoding="utf-8")
        assert body.startswith("WEBVTT\n")
        assert "00:00:00.000 --> 00:00:02.400\nHello world" in body
        assert "00:00:02.500 --> 00:00:05.123\nSecond cue" in body
