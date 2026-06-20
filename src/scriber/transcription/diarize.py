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
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment
from tqdm import tqdm

from scriber.logger import my_logger
from scriber.transcription.local import (
    detect_language,
    extract_audio,
    get_device,
    load_model,
    maybe_preprocess,
    patch_whisper_progress_bar,
)

MIN_SEGMENT_DURATION: float = 1.5  # seconds; skip whisper output shorter than this
_MAX_SPEAKER_GAP: float = 1.0  # seconds; merge consecutive same-speaker segments within this gap
_SAMPLE_RATE: int = 16000  # whisper.load_audio always returns mono PCM at 16 kHz


def decode_audio(audio_file: str) -> npt.NDArray[np.float32]:
    """Decode `audio_file` to a mono float32 array at 16 kHz via whisper's ffmpeg loader.

    Keeps the diarization path off torchcodec (see module docstring): the same
    decoded array is handed to pyannote in memory and sliced per speaker turn.
    """
    return cast(npt.NDArray[np.float32], whisper.load_audio(audio_file))


def diarize_speakers(audio: npt.NDArray[np.float32]) -> list[tuple[str, Segment]]:
    """Diarize speakers in a preloaded mono 16 kHz waveform using PyAnnote.

    Reads ``HUGGINGFACE_TOKEN`` from the process env — caller is expected to
    have populated it (e.g. via ``Settings.from_env()``).
    """
    my_logger.info("Diarizing speakers")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        err_msg = "Missing Hugging Face token in HUGGINGFACE_TOKEN env variable"
        raise OSError(err_msg)
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
    # 4.x returns a DiarizeOutput, not an Annotation. ``exclusive_speaker_diarization``
    # is the overlap-free turn segmentation pyannote intends for downstream
    # transcription (``speaker_diarization`` keeps overlapping turns).
    diarization = pipeline(diar_input).exclusive_speaker_diarization
    return [(str(label), segment) for segment, _, label in diarization.itertracks(yield_label=True)]


def slice_audio(
    audio: npt.NDArray[np.float32],
    start: float,
    end: float,
) -> npt.NDArray[np.float32]:
    """Return the mono samples of `audio` between `start` and `end` seconds."""
    return audio[int(start * _SAMPLE_RATE) : int(end * _SAMPLE_RATE)]


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
) -> tuple[str, str]:
    """Transcribe audio with speaker diarization.

    ``language=None`` autodetects (default behavior); pass a code (e.g.
    ``"fr"``) to force whisper to that language. ``preprocess`` runs the
    default ffmpeg filter chain (``alimiter + dynaudnorm``) before both
    diarization and transcription unless disabled. ``initial_prompt``
    seeds whisper's decoder with a vocabulary primer; applied to every
    per-speaker slice.
    """
    audio_file, owns_temp = maybe_preprocess(audio_file, preprocess=preprocess)
    try:
        device = get_device()
        my_logger.info(f"\tUsing device: {device}")
        patch_whisper_progress_bar()
        model = load_model(model_size, device)

        if language is None:
            used_lang = detect_language(audio_file, model, device)
            my_logger.info(f"Detected language: {used_lang}")
        else:
            used_lang = language
            my_logger.info(f"Forced language: {used_lang}")

        # Decode once; reuse for diarization (in memory) and per-turn slicing.
        audio = decode_audio(audio_file)

        # Diarize speakers. community-1's exclusive_speaker_diarization is already
        # overlap-free and speech-only, so the old separate VAD + crop-filter step
        # (a 3.x workaround) is no longer needed.
        diarized_segments = diarize_speakers(audio)

        # Group segments from the same speaker
        grouped_segments = group_speaker_segments(diarized_segments, max_gap=_MAX_SPEAKER_GAP)
        full_text: list[str] = []
        progress: tqdm[Any] = tqdm(
            grouped_segments,
            desc="Transcribing segments",
            unit="seg",
        )
        for speaker, segment in progress:
            # Skip segments that are too short (silence or noise)
            if segment.end - segment.start < MIN_SEGMENT_DURATION:
                continue
            sliced_audio = slice_audio(audio, segment.start, segment.end)
            segment_result = model.transcribe(
                sliced_audio,
                fp16=(device == "cuda"),
                language=used_lang,
                initial_prompt=initial_prompt,
            )
            text = cast(str, segment_result["text"]).strip()
            if not text:  # Skip empty transcriptions
                continue
            full_text.append(f"{speaker}: {text}")

        return "\n".join(full_text), used_lang
    finally:
        if owns_temp:
            Path(audio_file).unlink(missing_ok=True)


def transcribe_video_file_with_diarization(
    video_file: str,
    model_size: str = "base",
    language: str | None = None,
    *,
    preprocess: bool = True,
    initial_prompt: str | None = None,
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
        )
    finally:
        Path(audio_path).unlink()
    return transcription, used_lang
