# pyright: reportUnknownVariableType=false
"""Opt-in integration test for the diarized French pipeline.

Skipped by default (see ``addopts`` in ``pyproject.toml``). Run with:

    uv run pytest -m integration

Fixture: ``tests/integration/data/assises_fr_5spk_32-45.wav`` — a 13:20 mono
16 kHz excerpt (32:00-45:20) of a French conference recording with **exactly 5
distinct speakers**. It is *not* committed (binary; see
``tests/integration/data/.gitignore``). Regenerate it from the source video::

    ffmpeg -ss 00:32:00 -i "<source>.mp4" -t 00:13:20 -vn -ar 16000 -ac 1 \\
           -c:a pcm_s16le tests/integration/data/assises_fr_5spk_32-45.wav

This is the regression guard for two bugs found on a real 2h run:
    - language was detected from the first 30s (a music prelude) → the whole
      file was mistranscribed/translated to English. We now sample speech.
    - speaker turns were poor and unlabelled by appearance.

Downloads large-v3-turbo (~1.5 GB) + pyannote ``community-1`` on first run, and
needs ``HUGGINGFACE_TOKEN`` set. Runs on GPU if available (a few minutes), CPU
otherwise (much slower).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langdetect import detect

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "data" / "assises_fr_5spk_32-45.wav"
_EXPECTED_SPEAKERS = 5


def test_french_diarized_excerpt_detects_french_and_five_speakers() -> None:
    if not _FIXTURE.exists():
        pytest.skip(f"Integration fixture missing: {_FIXTURE}")

    from scriber.transcription.diarize import transcribe_audio_with_diarization

    text, language, _segments = transcribe_audio_with_diarization(
        str(_FIXTURE),
        model_size="large-v3-turbo",
        min_speakers=_EXPECTED_SPEAKERS,
        max_speakers=_EXPECTED_SPEAKERS,
    )

    # Language must be detected from speech (French), not the wrong global guess
    # that previously triggered whole-file translation to English.
    assert language == "fr"
    # The transcript text itself is French (end-to-end: no translation).
    assert detect(text) == "fr"

    # With the count pinned to 5, relabel_by_appearance yields SPEAKER_00..04.
    speakers = {line.split(":", 1)[0] for line in text.splitlines() if ":" in line}
    assert speakers == {f"SPEAKER_{i:02d}" for i in range(_EXPECTED_SPEAKERS)}
