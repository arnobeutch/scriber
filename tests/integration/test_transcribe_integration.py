"""Opt-in integration tests exercising the real ML pipeline.

Skipped by default (see ``addopts`` in ``pyproject.toml``). Run with:

    uv run pytest -m integration

Requires:
    - A small audio fixture at ``tests/integration/data/hello.wav`` (mono, 16kHz
      recommended). Create one with::

          ffmpeg -f lavfi -i "sine=frequency=440:duration=2" \\
                 -ar 16000 -ac 1 tests/integration/data/hello.wav

    - Network access on first run (whisper downloads the model — ``tiny`` ≈75MB).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "data" / "hello.wav"


def test_transcribe_audio_full_tiny_model() -> None:
    if not _FIXTURE.exists():
        pytest.skip(f"Integration fixture missing: {_FIXTURE}")

    from scriber.transcription.local import transcribe_audio_full

    text, language, segments = transcribe_audio_full(str(_FIXTURE), model_size="tiny")
    assert isinstance(text, str)
    assert isinstance(language, str)
    assert len(language) >= 2
    assert isinstance(segments, list)


def test_detect_language_large_v3_turbo() -> None:
    """Regression: detect_language must use the model's own n_mels.

    large-v3 / large-v3-turbo encoders expect 128 mel channels; the helper used
    log_mel_spectrogram's 80-channel default and crashed on the default model.
    Downloads large-v3-turbo (~1.5GB) on first run.
    """
    if not _FIXTURE.exists():
        pytest.skip(f"Integration fixture missing: {_FIXTURE}")

    from scriber.transcription.local import detect_language, load_model

    model = load_model("large-v3-turbo", "cpu")
    language = detect_language(str(_FIXTURE), model, "cpu")
    assert isinstance(language, str)
    assert len(language) >= 2


def test_extract_then_transcribe(tmp_path: Path) -> None:
    """Full pipeline from video-format container → audio → whisper."""
    if not _FIXTURE.exists():
        pytest.skip(f"Integration fixture missing: {_FIXTURE}")

    from scriber.transcription.local import extract_audio, transcribe_audio_full

    audio_path = extract_audio(str(_FIXTURE))
    try:
        text, language, _segments = transcribe_audio_full(audio_path, model_size="tiny")
    finally:
        Path(audio_path).unlink()
    _ = tmp_path
    assert isinstance(text, str)
    assert isinstance(language, str)
