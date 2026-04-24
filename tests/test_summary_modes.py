"""Tests for summarizers.modes — prompts + auto-detect heuristic."""

from __future__ import annotations

import pytest

from scriber.model import Transcript
from scriber.summarizers.modes import (
    detect_mode,
    get_prompt,
    resolve_mode,
)


def _t(text: str = "x", *, diarized: bool = False) -> Transcript:
    return Transcript(
        text=text,
        language="en",
        title="t",
        source="whisper" if not diarized else "yt_manual",
        diarized=diarized,
    )


class TestGetPrompt:
    def test_meeting_en_has_expected_headers(self) -> None:
        out = get_prompt("meeting", "en")
        assert "Topic:" in out
        assert "Main takeaways:" in out
        assert "Action items:" in out
        assert "Transcript:" in out

    def test_meeting_fr_has_expected_headers(self) -> None:
        out = get_prompt("meeting", "fr")
        assert "Sujet :" in out
        assert "Principaux enseignements :" in out
        assert "Actions à suivre :" in out
        assert "Transcription :" in out

    def test_source_en_has_expected_headers(self) -> None:
        out = get_prompt("source", "en")
        assert "TL;DR:" in out
        assert "Key takeaways:" in out
        assert "Counterpoints / alternatives:" in out

    def test_source_fr_has_expected_headers(self) -> None:
        out = get_prompt("source", "fr")
        assert "TL;DR :" in out
        assert "Points clés :" in out
        assert "Contrepoints / alternatives :" in out

    def test_unsupported_language(self) -> None:
        with pytest.raises(ValueError, match="language not supported"):
            get_prompt("meeting", "de")


class TestDetectMode:
    def test_diarized_two_plus_speakers_is_meeting(self) -> None:
        text = "SPEAKER_00: Hello there\nSPEAKER_01: Hi back\nSPEAKER_00: bye"
        assert detect_mode(_t(text=text, diarized=True)) == "meeting"

    def test_diarized_single_speaker_is_source(self) -> None:
        text = "SPEAKER_00: A long monologue " * 30
        # Even with diarized=True, only one speaker → falls through to source.
        assert detect_mode(_t(text=text, diarized=True)) == "source"

    def test_opinion_dense_text_is_source(self) -> None:
        # Many opinion markers in a short text → source.
        text = (
            "I think this is wrong. In my opinion the data is suspicious. "
            "I believe the conclusion is wrong. Maybe the speaker is right, "
            "perhaps not. I feel uncertain."
        )
        assert detect_mode(_t(text=text)) == "source"

    def test_default_is_source(self) -> None:
        # Plain neutral prose, not diarized → defaults to source.
        text = "The cat sat on the mat. " * 50
        assert detect_mode(_t(text=text)) == "source"


class TestResolveMode:
    def test_explicit_meeting_wins(self) -> None:
        # Even with single-speaker diarized text, explicit choice wins.
        text = "SPEAKER_00: monologue"
        assert resolve_mode("meeting", _t(text=text, diarized=True)) == "meeting"

    def test_explicit_source_wins(self) -> None:
        text = "SPEAKER_00: hi\nSPEAKER_01: yo"
        assert resolve_mode("source", _t(text=text, diarized=True)) == "source"

    def test_auto_routes_to_detect(self) -> None:
        text = "SPEAKER_00: hi\nSPEAKER_01: yo"
        assert resolve_mode("auto", _t(text=text, diarized=True)) == "meeting"
