# Boundary to untyped ML deps (pyannote, whisper). 2026-06-02:
# suppress unknown-type reports here; keep call/argument/attribute checks on.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""Speaker diarization path (pyannote).

Imported lazily by the handlers only when ``--diarize`` is requested, so the
transcription-only base install (no ``scriber[diarize]``) never pulls
pyannote. The shared whisper helpers live in :mod:`scriber.transcription.local`.

Audio is decoded once with whisper's ffmpeg loader and kept in memory: pyannote
4.x (and torchaudio 2.11) otherwise read audio through ``torchcodec``, whose
native libs are brittle on Windows — they need the FFmpeg "full-shared" DLLs and
a matching torch version, and fail closed to a warning (leaving pyannote's
``AudioDecoder`` undefined → ``NameError`` at first use). Feeding pyannote a
preloaded ``{"waveform", "sample_rate"}`` mapping bypasses torchcodec entirely
(see :class:`pyannote.audio.core.io.Audio`).
"""

import os
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
import whisper
from pyannote.core import Segment

from scriber.logger import my_logger
from scriber.transcription.local import (
    detect_language_probs,
    extract_audio,
    get_device,
    load_model,
    transcribe_audio_full,
)

MIN_SEGMENT_DURATION: float = 1.5  # seconds; ignore speaker turns shorter than this for LID
_MAX_SPEAKER_GAP: float = 1.0  # seconds; merge consecutive same-speaker turns within this gap
_MAX_ASSIGN_GAP: float = 5.0  # seconds; a whisper segment farther than this from any
# speaker turn is dropped (e.g. a music prelude diarization excluded from speech)
_SAMPLE_RATE: int = 16000  # whisper.load_audio always returns mono PCM at 16 kHz
_LID_MAX_WINDOWS: int = 6  # max speech windows to sample for language detection
_LID_WINDOW_SEC: float = 30.0  # whisper scores at most 30s per window


def decode_audio(audio_file: str) -> npt.NDArray[np.float32]:
    """Decode `audio_file` to a mono float32 array at 16 kHz via whisper's ffmpeg loader.

    Keeps the diarization path off torchcodec (see module docstring): the
    decoded array is handed to pyannote in memory and sliced for language probes.
    """
    return cast(npt.NDArray[np.float32], whisper.load_audio(audio_file))


def diarize_speakers(
    audio: npt.NDArray[np.float32],
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[tuple[str, Segment]]:
    """Diarize speakers in a preloaded mono 16 kHz waveform using PyAnnote.

    ``min_speakers`` / ``max_speakers`` constrain clustering when the speaker
    count is known (e.g. a fixed panel) — both optional. Reads
    ``HUGGINGFACE_TOKEN`` from the process env — caller is expected to have
    populated it (e.g. via ``Settings.from_env()``).
    """
    my_logger.info("Diarizing speakers")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        err_msg = "Missing Hugging Face token in HUGGINGFACE_TOKEN env variable"
        raise OSError(err_msg)
    # Import pyannote.audio here, not at module top: it loads torchcodec on
    # import and emits a long multi-traceback UserWarning when torchcodec's
    # native libs can't load (common on Windows — missing FFmpeg "full-shared"
    # DLLs / torch version mismatch). We feed pyannote an in-memory waveform and
    # never use torchcodec, so that warning is pure noise — suppress just it.
    # (pyannote.core, imported at module top, does not trigger it.)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(?s).*torchcodec is not installed",
            category=UserWarning,
        )
        from pyannote.audio import Pipeline
    # pyannote 4.x flagship model. Needs a token with access to the gated repo:
    # - https://huggingface.co/pyannote/speaker-diarization-community-1
    # ``token=`` (4.x renamed use_auth_token); cast over the untyped boundary
    # (pyright reads a stale from_pretrained signature, but token is real at runtime).
    pipeline = cast(Any, Pipeline).from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    # Feed an in-memory (channel, time) waveform so pyannote skips torchcodec
    # decoding (see pyannote.audio.core.io.Audio.validate_file). ascontiguousarray
    # guarantees the C-contiguous float32 buffer torch.from_numpy needs.
    waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
    diar_input = {"waveform": waveform, "sample_rate": _SAMPLE_RATE}
    hints: dict[str, int] = {}
    if min_speakers is not None:
        hints["min_speakers"] = min_speakers
    if max_speakers is not None:
        hints["max_speakers"] = max_speakers
    # 4.x returns a DiarizeOutput, not an Annotation. ``exclusive_speaker_diarization``
    # is the overlap-free turn segmentation pyannote intends for downstream
    # transcription (``speaker_diarization`` keeps overlapping turns). itertracks
    # yields turns in chronological order.
    diarization = pipeline(diar_input, **hints).exclusive_speaker_diarization
    return [(str(label), segment) for segment, _, label in diarization.itertracks(yield_label=True)]


def relabel_by_appearance(turns: list[tuple[str, Segment]]) -> list[tuple[str, Segment]]:
    """Remap pyannote's arbitrary cluster ids to ``SPEAKER_00, 01, ...`` by first appearance.

    pyannote labels (e.g. ``SPEAKER_11`` for the opening voice) are cluster ids
    in no particular order; ``turns`` arrives chronological, so the first
    distinct label becomes ``SPEAKER_00``, the next ``SPEAKER_01``, etc.
    """
    mapping: dict[str, str] = {}
    relabeled: list[tuple[str, Segment]] = []
    for label, segment in turns:
        if label not in mapping:
            mapping[label] = f"SPEAKER_{len(mapping):02d}"
        relabeled.append((mapping[label], segment))
    return relabeled


def slice_audio(
    audio: npt.NDArray[np.float32],
    start: float,
    end: float,
) -> npt.NDArray[np.float32]:
    """Return the mono samples of `audio` between `start` and `end` seconds."""
    return audio[int(start * _SAMPLE_RATE) : int(end * _SAMPLE_RATE)]


def _evenly_spaced(items: list[Segment], count: int) -> list[Segment]:
    """Pick up to `count` items spread evenly across `items` (preserving order)."""
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def detect_language_from_speech(
    audio: npt.NDArray[np.float32],
    speaker_turns: list[tuple[str, Segment]],
    model: whisper.Whisper,
    device: str,
) -> str:
    """Detect language by sampling speech windows spread across the timeline.

    Whisper's stock language detection looks at only the first 30s — which on
    real recordings is often a music prelude, intro jingle, or silence, yielding
    a wrong global language that then mistranslates the whole file. Here we
    sample several actual speaker turns (which exclude non-speech) and sum
    Whisper's per-window probabilities, so the dominant spoken language wins.
    """
    turns = [seg for _, seg in speaker_turns if seg.duration >= MIN_SEGMENT_DURATION]
    if not turns:
        turns = [seg for _, seg in speaker_turns]
    aggregate: dict[str, float] = {}
    for seg in _evenly_spaced(turns, _LID_MAX_WINDOWS):
        window = slice_audio(audio, seg.start, min(seg.end, seg.start + _LID_WINDOW_SEC))
        if window.size == 0:
            continue
        for lang, prob in detect_language_probs(window, model, device).items():
            aggregate[lang] = aggregate.get(lang, 0.0) + prob
    if not aggregate:  # no usable speech (e.g. empty diarization) — fall back to the head
        aggregate = detect_language_probs(audio, model, device)
    return max(aggregate, key=lambda k: aggregate[k])


def _best_speaker(start: float, end: float, speaker_turns: list[tuple[str, Segment]]) -> str | None:
    """Pick the speaker whose turn overlaps [start, end] most.

    Falls back to the nearest turn within ``_MAX_ASSIGN_GAP`` when there is no
    overlap (a whisper segment landing in a short diarization gap), and returns
    ``None`` when the nearest turn is farther than that — i.e. non-speech such as
    a music prelude that diarization excluded, which we then drop.
    """
    best_label: str | None = None
    best_overlap = 0.0
    nearest_label: str | None = None
    nearest_dist = float("inf")
    mid = (start + end) / 2
    for label, seg in speaker_turns:
        overlap = max(0.0, min(end, seg.end) - max(start, seg.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
        dist = 0.0 if seg.start <= mid <= seg.end else min(abs(mid - seg.start), abs(mid - seg.end))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_label = label
    if best_label is not None:
        return best_label
    if nearest_dist <= _MAX_ASSIGN_GAP:
        return nearest_label
    return None


def assign_speakers_to_segments(
    segments: list[dict[str, Any]],
    speaker_turns: list[tuple[str, Segment]],
) -> list[tuple[str, str]]:
    """Label each whisper segment with the speaker whose turn it overlaps.

    This is the "transcribe-then-assign" join: one full whisper pass produces
    ``segments`` (with ``start`` / ``end`` / ``text``), which we map onto the
    diarization ``speaker_turns``. Segments with no nearby turn (music, silence)
    are dropped. Returns ``(speaker, text)`` pairs in transcript order.
    """
    labeled: list[tuple[str, str]] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        speaker = _best_speaker(float(seg["start"]), float(seg["end"]), speaker_turns)
        if speaker is None:
            continue
        labeled.append((speaker, text))
    return labeled


def format_diarized(labeled: list[tuple[str, str]]) -> str:
    """Render ``(speaker, text)`` pairs as ``SPEAKER_XX: ...`` lines, one run per speaker."""
    lines: list[str] = []
    current: str | None = None
    buffer: list[str] = []
    for speaker, text in labeled:
        if speaker != current:
            if current is not None:
                lines.append(f"{current}: {' '.join(buffer)}")
            current = speaker
            buffer = [text]
        else:
            buffer.append(text)
    if current is not None:
        lines.append(f"{current}: {' '.join(buffer)}")
    return "\n".join(lines)


def group_speaker_segments(
    diarized_segments: list[tuple[str, Segment]],
    max_gap: float = 1.0,
) -> list[tuple[str, Segment]]:
    """Group consecutive segments from the same speaker that are close in time.

    Args:
        diarized_segments (list): List of (speaker, Segment) tuples.
        max_gap (float): Max gap in seconds to allow merging.

    Returns:
        list: List of (speaker, merged Segment) tuples.

    """
    grouped_segments: list[tuple[str, Segment]] = []
    last_speaker: str | None = None
    current_start: float = 0.0
    current_end: float = 0.0

    for speaker, segment in diarized_segments:
        if speaker == last_speaker and (segment.start - current_end) <= max_gap:
            current_end = segment.end  # extend current segment
        else:
            if last_speaker is not None:
                grouped_segments.append(
                    (last_speaker, Segment(current_start, current_end)),
                )
            last_speaker = speaker
            current_start = segment.start
            current_end = segment.end

    if last_speaker is not None:
        grouped_segments.append((last_speaker, Segment(current_start, current_end)))

    return grouped_segments


def transcribe_audio_with_diarization(
    audio_file: str,
    model_size: str = "base",
    language: str | None = None,
    *,
    preprocess: bool = True,
    initial_prompt: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> tuple[str, str]:
    """Transcribe audio with speaker diarization (transcribe-then-assign).

    One full whisper pass produces timestamped segments; pyannote produces
    speaker turns; each segment is then labelled with the speaker it overlaps.
    This is far faster than the old per-turn transcription (one pass, full
    decoder context) and lets language detection sample real speech.

    ``language=None`` autodetects from sampled speech windows (see
    :func:`detect_language_from_speech`); pass a code (e.g. ``"fr"``) to force
    it. ``preprocess`` runs the default ffmpeg filter chain
    (``alimiter + dynaudnorm``) before transcription only — diarization runs on
    the raw audio, since loudness normalization can blur speaker embeddings.
    ``initial_prompt`` seeds whisper's decoder with a vocabulary primer.
    ``min_speakers`` / ``max_speakers`` hint pyannote's clustering when known.
    """
    device = get_device()
    my_logger.info(f"\tUsing device: {device}")
    model = load_model(model_size, device)

    # Diarize on the raw (un-preprocessed) audio: dynaudnorm flattens the level
    # cues speaker embeddings rely on. Relabel turns 00, 01, ... by appearance.
    raw_audio = decode_audio(audio_file)
    speaker_turns = relabel_by_appearance(
        diarize_speakers(raw_audio, min_speakers=min_speakers, max_speakers=max_speakers),
    )

    if language is None:
        used_lang = detect_language_from_speech(raw_audio, speaker_turns, model, device)
        my_logger.info(f"Detected language (speech-sampled): {used_lang}")
    else:
        used_lang = language
        my_logger.info(f"Forced language: {used_lang}")

    # One full-file transcription pass (preprocessed audio), then join segments
    # to speaker turns. transcribe_audio_full handles preprocessing + progress bar.
    _text, _lang, segments = transcribe_audio_full(
        audio_file,
        model_size=model_size,
        language=used_lang,
        preprocess=preprocess,
        initial_prompt=initial_prompt,
    )
    grouped_turns = group_speaker_segments(speaker_turns, max_gap=_MAX_SPEAKER_GAP)
    labeled = assign_speakers_to_segments(segments, grouped_turns)
    return format_diarized(labeled), used_lang


def transcribe_video_file_with_diarization(
    video_file: str,
    model_size: str = "base",
    language: str | None = None,
    *,
    preprocess: bool = True,
    initial_prompt: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> tuple[str, str]:
    """Full pipeline: Extract audio from video, transcribe it with diarization."""
    my_logger.info(f"Processing with diarization: {video_file}")
    audio_path = extract_audio(video_file)
    try:
        transcription, used_lang = transcribe_audio_with_diarization(
            audio_path,
            model_size=model_size,
            language=language,
            preprocess=preprocess,
            initial_prompt=initial_prompt,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    finally:
        Path(audio_path).unlink()
    return transcription, used_lang
