"""Tests for prepare_local_transcript — pure helpers only.

Entry points that call whisper / pyannote / ffmpeg are covered by integration
tests (opt-in, ``pytest -m integration``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pyannote.core import Segment

import scriber.transcription.local as plt
from scriber.transcription.diarize import (
    _MAX_ASSIGN_GAP,
    _SAMPLE_RATE,
    assign_speakers_to_segments,
    decode_audio,
    detect_language_from_speech,
    diarize_speakers,
    format_diarized,
    group_speaker_segments,
    relabel_by_appearance,
    slice_audio,
)
from scriber.transcription.local import (
    _MODEL_CACHE,
    _PREPROCESS_FILTER,
    detect_language,
    extract_audio,
    get_device,
    maybe_preprocess,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestGetDevice:
    def test_cuda_when_available(self) -> None:
        with patch("scriber.transcription.local.torch.cuda.is_available", return_value=True):
            assert get_device() == "cuda"

    def test_cpu_fallback(self) -> None:
        with patch("scriber.transcription.local.torch.cuda.is_available", return_value=False):
            assert get_device() == "cpu"


class TestDetectLanguage:
    def test_forwards_model_n_mels_to_spectrogram(self) -> None:
        # Regression: large-v3 / large-v3-turbo encoders expect 128 mel channels;
        # log_mel_spectrogram defaults to 80, which crashed detect_language on the
        # default model. The n_mels must come from the model's own dims.
        model = MagicMock()
        model.dims.n_mels = 128
        model.detect_language.return_value = (None, {"en": 0.9, "fr": 0.1})
        with (
            patch("scriber.transcription.local.whisper.load_audio", return_value="raw"),
            patch("scriber.transcription.local.whisper.pad_or_trim", return_value="trimmed"),
            patch("scriber.transcription.local.whisper.log_mel_spectrogram") as mel,
        ):
            result = detect_language("a.wav", cast("Any", model), "cpu")
        mel.assert_called_once_with("trimmed", n_mels=128)
        assert result == "en"


class TestDiarizeSpeakers:
    def test_passes_token_kwarg_not_use_auth_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regression: pyannote 4.x renamed Pipeline.from_pretrained's
        # use_auth_token -> token. Passing the old kwarg is a TypeError on 4.x.
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test")
        seg = Segment(0.0, 1.0)
        # pyannote 4.x: pipeline(...) -> DiarizeOutput; we read its
        # exclusive_speaker_diarization Annotation, then .itertracks().
        diar_output = MagicMock()
        diar_output.exclusive_speaker_diarization.itertracks.return_value = [
            (seg, "t0", "SPEAKER_00")
        ]
        pipeline_callable = MagicMock(return_value=diar_output)
        audio = np.zeros(_SAMPLE_RATE, dtype=np.float32)
        # Pipeline is imported lazily inside diarize_speakers, so patch the source.
        with patch("pyannote.audio.Pipeline") as pipeline_cls:
            pipeline_cls.from_pretrained.return_value = pipeline_callable
            result = diarize_speakers(audio)
        _, kwargs = pipeline_cls.from_pretrained.call_args
        assert "token" in kwargs
        assert "use_auth_token" not in kwargs
        assert result == [("SPEAKER_00", seg)]

    def test_feeds_in_memory_waveform_not_a_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Regression: pyannote 4.x / torchaudio 2.11 decode through torchcodec,
        # whose native libs fail on Windows (NameError: AudioDecoder). We must
        # hand the pipeline a preloaded {"waveform", "sample_rate"} mapping so it
        # never touches torchcodec — never a file path.
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test")
        diar_output = MagicMock()
        diar_output.exclusive_speaker_diarization.itertracks.return_value = []
        pipeline_callable = MagicMock(return_value=diar_output)
        audio = np.zeros(_SAMPLE_RATE, dtype=np.float32)
        with patch("pyannote.audio.Pipeline") as pipeline_cls:
            pipeline_cls.from_pretrained.return_value = pipeline_callable
            diarize_speakers(audio)
        (call_arg,), _ = pipeline_callable.call_args
        assert isinstance(call_arg, dict)
        mapping = cast("dict[str, Any]", call_arg)
        assert mapping["sample_rate"] == _SAMPLE_RATE
        waveform = mapping["waveform"]
        assert waveform.shape == (1, _SAMPLE_RATE)  # (channel, time)

    def test_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        with pytest.raises(OSError, match="HUGGINGFACE_TOKEN"):
            diarize_speakers(np.zeros(_SAMPLE_RATE, dtype=np.float32))


class TestSliceAudio:
    def test_slices_by_seconds(self) -> None:
        audio = np.arange(_SAMPLE_RATE * 3, dtype=np.float32)  # 3 seconds
        sliced = slice_audio(audio, 1.0, 2.0)
        assert len(sliced) == _SAMPLE_RATE
        assert sliced[0] == float(_SAMPLE_RATE)  # first sample of second 1

    def test_empty_when_start_equals_end(self) -> None:
        audio = np.arange(_SAMPLE_RATE, dtype=np.float32)
        assert len(slice_audio(audio, 0.5, 0.5)) == 0


class TestDecodeAudio:
    def test_delegates_to_whisper_load_audio(self) -> None:
        fake = np.zeros(8, dtype=np.float32)
        with patch("scriber.transcription.diarize.whisper.load_audio", return_value=fake) as load:
            result = decode_audio("a.wav")
        load.assert_called_once_with("a.wav")
        assert result is fake


class TestRelabelByAppearance:
    def test_empty(self) -> None:
        assert relabel_by_appearance([]) == []

    def test_relabels_in_order_of_first_appearance(self) -> None:
        # pyannote hands back arbitrary cluster ids (e.g. SPEAKER_11 first).
        turns = [
            ("SPEAKER_11", Segment(0.0, 1.0)),
            ("SPEAKER_03", Segment(1.0, 2.0)),
            ("SPEAKER_11", Segment(2.0, 3.0)),
        ]
        result = relabel_by_appearance(turns)
        assert [label for label, _ in result] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
        # segments are preserved untouched
        assert [seg.start for _, seg in result] == [0.0, 1.0, 2.0]


class TestAssignSpeakersToSegments:
    def _turns(self) -> list[tuple[str, Segment]]:
        return [("SPEAKER_00", Segment(0.0, 8.0)), ("SPEAKER_01", Segment(20.0, 30.0))]

    def test_assigns_by_max_overlap(self) -> None:
        segments = [{"start": 1.0, "end": 3.0, "text": "hello"}]
        assert assign_speakers_to_segments(segments, self._turns()) == [("SPEAKER_00", "hello")]

    def test_nearest_turn_within_gap_when_no_overlap(self) -> None:
        # segment sits in a short gap just after SPEAKER_00's turn ends (8.0)
        segments = [{"start": 9.0, "end": 10.0, "text": "in the gap"}]
        assert assign_speakers_to_segments(segments, self._turns()) == [
            ("SPEAKER_00", "in the gap")
        ]

    def test_drops_segment_far_from_any_turn(self) -> None:
        # a segment farther than _MAX_ASSIGN_GAP from every turn (e.g. a music
        # interlude diarization left unassigned) is dropped, not mislabelled.
        far = self._turns()[-1][1].end + _MAX_ASSIGN_GAP + 5
        segments = [{"start": far, "end": far + 1, "text": "music"}]
        assert assign_speakers_to_segments(segments, self._turns()) == []

    def test_skips_empty_text(self) -> None:
        segments = [{"start": 1.0, "end": 3.0, "text": "   "}]
        assert assign_speakers_to_segments(segments, self._turns()) == []


class TestFormatDiarized:
    def test_empty(self) -> None:
        assert format_diarized([]) == ""

    def test_groups_consecutive_same_speaker(self) -> None:
        labeled = [
            ("SPEAKER_00", "one"),
            ("SPEAKER_00", "two"),
            ("SPEAKER_01", "three"),
            ("SPEAKER_00", "four"),
        ]
        assert format_diarized(labeled) == (
            "SPEAKER_00: one two\nSPEAKER_01: three\nSPEAKER_00: four"
        )


class TestDetectLanguageFromSpeech:
    def test_votes_across_windows_ignoring_the_head(self) -> None:
        # Long buffer so slices are non-empty; turns are all speech (no head music).
        audio = np.zeros(_SAMPLE_RATE * 400, dtype=np.float32)
        turns = [
            ("SPEAKER_00", Segment(100.0, 140.0)),
            ("SPEAKER_01", Segment(200.0, 240.0)),
        ]
        # window 1 leans en, window 2 leans fr harder → fr wins the sum
        with patch(
            "scriber.transcription.diarize.detect_language_probs",
            side_effect=[{"en": 0.9, "fr": 0.1}, {"en": 0.1, "fr": 0.95}],
        ):
            assert detect_language_from_speech(audio, turns, cast("Any", object()), "cpu") == "fr"

    def test_falls_back_to_head_when_no_turns(self) -> None:
        audio = np.zeros(_SAMPLE_RATE * 5, dtype=np.float32)
        with patch(
            "scriber.transcription.diarize.detect_language_probs",
            return_value={"en": 0.99, "fr": 0.01},
        ) as probs:
            assert detect_language_from_speech(audio, [], cast("Any", object()), "cpu") == "en"
        probs.assert_called_once()  # scored the whole-file head, not a per-turn window


class TestGroupSpeakerSegments:
    def test_empty(self) -> None:
        assert group_speaker_segments([]) == []

    def test_single(self) -> None:
        result = group_speaker_segments([("A", Segment(0.0, 1.0))])
        assert len(result) == 1
        assert result[0][0] == "A"
        assert result[0][1].start == 0.0
        assert result[0][1].end == 1.0

    def test_merges_same_speaker_within_gap(self) -> None:
        segs = [
            ("A", Segment(0.0, 1.0)),
            ("A", Segment(1.5, 2.0)),  # gap of 0.5 ≤ max_gap=1.0
        ]
        result = group_speaker_segments(segs, max_gap=1.0)
        assert len(result) == 1
        assert result[0][1].start == 0.0
        assert result[0][1].end == 2.0

    def test_does_not_merge_across_wide_gap(self) -> None:
        segs = [
            ("A", Segment(0.0, 1.0)),
            ("A", Segment(3.0, 4.0)),  # gap of 2.0 > max_gap=1.0
        ]
        result = group_speaker_segments(segs, max_gap=1.0)
        assert len(result) == 2

    def test_does_not_merge_different_speakers(self) -> None:
        segs = [
            ("A", Segment(0.0, 1.0)),
            ("B", Segment(1.0, 2.0)),
        ]
        result = group_speaker_segments(segs)
        assert len(result) == 2
        assert result[0][0] == "A"
        assert result[1][0] == "B"

    def test_alternating_speakers(self) -> None:
        segs = [
            ("A", Segment(0.0, 1.0)),
            ("B", Segment(1.0, 2.0)),
            ("A", Segment(2.0, 3.0)),
            ("A", Segment(3.2, 4.0)),  # merged with previous A
        ]
        result = group_speaker_segments(segs, max_gap=0.5)
        assert [speaker for speaker, _ in result] == ["A", "B", "A"]
        assert result[-1][1].start == 2.0
        assert result[-1][1].end == 4.0


class TestExtractAudio:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            extract_audio(str(tmp_path / "nope.mp4"))


class TestMaybePreprocess:
    def test_disabled_returns_original_path_no_ownership(self) -> None:
        with patch("scriber.transcription.local.preprocess_audio_file") as pre:
            path, owns = maybe_preprocess("/some/audio.wav", preprocess=False)
        pre.assert_not_called()
        assert path == "/some/audio.wav"
        assert owns is False

    def test_enabled_calls_filter_and_claims_ownership(self) -> None:
        with patch(
            "scriber.transcription.local.preprocess_audio_file",
            return_value="/tmp/filtered.wav",
        ) as pre:
            path, owns = maybe_preprocess("/some/audio.wav", preprocess=True)
        pre.assert_called_once_with("/some/audio.wav")
        assert path == "/tmp/filtered.wav"
        assert owns is True

    def test_filter_chain_is_the_one_recommended_by_bench(self) -> None:
        # Locks the filter to the alimiter+dynaudnorm chain from the bench.
        # If we ever change this, that change should be deliberate.
        assert _PREPROCESS_FILTER == "alimiter=limit=0.95:level=disabled,dynaudnorm"


class TestModelCache:
    def test_model_loaded_once_on_repeated_calls(self) -> None:
        _MODEL_CACHE.clear()
        fake_model = object()
        with patch(
            "scriber.transcription.local.whisper.load_model", return_value=fake_model
        ) as load:
            m1 = plt.load_model("tiny", "cpu")
            m2 = plt.load_model("tiny", "cpu")
        load.assert_called_once_with("tiny", device="cpu")
        assert m1 is m2 is fake_model

    def test_different_keys_load_separate_models(self) -> None:
        _MODEL_CACHE.clear()
        model_a = object()
        model_b = object()
        with patch(
            "scriber.transcription.local.whisper.load_model",
            side_effect=[model_a, model_b],
        ) as load:
            ma = plt.load_model("tiny", "cpu")
            mb = plt.load_model("small", "cpu")
        assert load.call_count == 2
        assert ma is model_a
        assert mb is model_b
