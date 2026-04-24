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
        assert "Summary:" in out
        assert "Main claims:" in out
        assert "Notable quotes:" in out
        assert "Factually correct:" in out
        assert "Likely but unconfirmed:" in out
        assert "Interpretation or weakly substantiated:" in out
        assert "Alternative interpretations:" in out
        assert "Wrong or misleading:" in out
        assert "Keywords:" in out

    def test_source_fr_has_expected_headers(self) -> None:
        out = get_prompt("source", "fr")
        assert "Résumé :" in out
        assert "Thèses principales :" in out
        assert "Citations notables :" in out
        assert "Factuellement correct :" in out
        assert "Probable mais non confirmé :" in out
        assert "Interprétation ou faiblement étayé :" in out
        assert "Interprétations alternatives :" in out
        assert "Faux ou trompeur :" in out

    def test_unsupported_language(self) -> None:
        with pytest.raises(ValueError, match="language not supported"):
            get_prompt("meeting", "de")

    def test_training_data_boundary_rendered_en(self) -> None:
        for mode in ("meeting", "source"):
            out = get_prompt(mode, "en")  # type: ignore[arg-type]
            assert "Ground every claim" in out
            assert "Do NOT supplement" in out
            assert "[Background]" in out

    def test_training_data_boundary_rendered_fr(self) -> None:
        for mode in ("meeting", "source"):
            out = get_prompt(mode, "fr")  # type: ignore[arg-type]
            assert "Ancrez chaque affirmation" in out
            assert "[Contexte]" in out

    def test_context_injects_before_transcript_marker_en(self) -> None:
        out = get_prompt("source", "en", context="GlossaryX means Y.")
        assert "Additional context:" in out
        assert "GlossaryX means Y." in out
        # Context must precede the Transcript heading in the rendered prompt.
        assert out.index("Additional context") < out.index("Transcript:")

    def test_context_injects_before_transcript_marker_fr(self) -> None:
        out = get_prompt("source", "fr", context="Le projet X est...")
        assert "Contexte additionnel :" in out
        assert "Le projet X est..." in out
        assert out.index("Contexte additionnel") < out.index("Transcription :")

    def test_empty_context_is_ignored(self) -> None:
        assert "Additional context" not in get_prompt("source", "en", context="")
        assert "Additional context" not in get_prompt("source", "en", context="   \n\t  ")
        assert "Additional context" not in get_prompt("source", "en")


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
