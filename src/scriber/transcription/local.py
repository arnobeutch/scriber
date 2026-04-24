# Boundary to untyped ML deps (whisper, pyannote, torchaudio, ffmpeg). 2026-04-22:
# suppress unknown-type reports here; keep call/argument/attribute checks on.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""Retrieve the text transcript from a local media file."""

import os
import tempfile
from pathlib import Path
from typing import Any, cast

import ffmpeg
import numpy as np
import numpy.typing as npt
import torch.cuda
import torchaudio
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline
from tqdm import tqdm

from scriber.logger import my_logger

MIN_SEGMENT_DURATION: float = 1.5  # seconds; skip whisper output shorter than this
_MAX_SPEAKER_GAP: float = 1.0  # seconds; merge consecutive same-speaker segments within this gap
_MODEL_CACHE: dict[tuple[str, str], whisper.Whisper] = {}


class TqdmProgressBar:
    """Replacement for whisper.utils.ProgressBar that uses tqdm."""

    def __init__(self, total: int) -> None:
        """Initialize the progress bar."""
        self._bar: tqdm[Any] = tqdm(total=total, unit="segment")

    def update(self, n: int = 1) -> None:
        """Update the progress bar by n segments."""
        self._bar.update(n)

    def close(self) -> None:
        """Close the progress bar."""
        self._bar.close()


def patch_whisper_progress_bar() -> None:
    """Monkey-patch whisper's ProgressBar with tqdm-based one."""
    cast(Any, whisper).utils.ProgressBar = TqdmProgressBar


def extract_audio(input_file: str, output_format: str = "wav") -> str:
    """Extract audio from a video file and returns the path to the audio file."""
    if not Path(input_file).exists():
        err_msg = f"File not found: {input_file}"
        raise FileNotFoundError(err_msg)

    my_logger.info(f"Extracting audio from {input_file} to {output_format} format")
    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp_audio_file:
        tmp_audio_path = tmp_audio_file.name

    ffmpeg.input(input_file).output(
        tmp_audio_path,
        format=output_format,
        ac=1,
        ar="16000",
    ).run(quiet=True, overwrite_output=True)
    return tmp_audio_path


def get_device() -> str:
    """Return 'cuda' if GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def detect_language(audio_file: str, model: whisper.Whisper, device: str) -> str:
    """Detect the language of the audio file using Whisper."""
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)
    _, probs = model.detect_language(mel)
    probs_dict = cast(dict[str, float], probs)
    return max(probs_dict, key=lambda k: probs_dict[k])


def _load_model(model_size: str, device: str) -> whisper.Whisper:
    """Return a cached Whisper model, loading it on first use."""
    key = (model_size, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = whisper.load_model(model_size, device=device)
    return _MODEL_CACHE[key]


def transcribe_audio_full(
    audio_file: str,
    model_size: str = "base",
    language: str | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Transcribe + return ``(text, language, segments)``.

    ``segments`` is whisper's per-cue list with ``start`` / ``end`` /
    ``text`` keys, suitable for SRT / VTT export.
    """
    my_logger.info(f"Transcribing audio file: {audio_file}")

    device = get_device()
    my_logger.info(f"\tUsing device: {device}")
    patch_whisper_progress_bar()
    model = _load_model(model_size, device)

    if language is None:
        used_lang = detect_language(audio_file, model, device)
        my_logger.info(f"\tDetected language: {used_lang}")
    else:
        used_lang = language
        my_logger.info(f"\tForced language: {used_lang}")

    result = model.transcribe(
        audio_file,
        fp16=(device == "cuda"),
        language=used_lang,
        verbose=False,  # enables whisper's tqdm progress bar
    )
    segments = cast(list[dict[str, Any]], result.get("segments", []))
    return cast(str, result["text"]), used_lang, segments


def diarize_speakers(audio_file: str) -> list[tuple[str, Segment]]:
    """Diarize speakers in the audio file using PyAnnote.

    Reads ``HUGGINGFACE_TOKEN`` from the process env — caller is expected to
    have populated it (e.g. via ``Settings.from_env()``).
    """
    my_logger.info(f"Diarizing speakers in: {audio_file}")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        err_msg = "Missing Hugging Face token in HUGGINGFACE_TOKEN env variable"
        raise OSError(err_msg)
    # Needs token with access to pyannote models:
    # - https://huggingface.co/pyannote/speaker-diarization-3.1
    # - https://huggingface.co/pyannote/segmentation-3.0
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    diarization = pipeline(audio_file)
    return [(str(label), segment) for segment, _, label in diarization.itertracks(yield_label=True)]


def load_audio_slice(audio_path: str, start: float, end: float) -> npt.NDArray[np.floating[Any]]:
    """Load a slice of audio between `start` and `end` seconds."""
    waveform, sample_rate = torchaudio.load(audio_path)
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    sliced_waveform = waveform[:, start_sample:end_sample]
    return sliced_waveform.mean(dim=0).numpy()  # convert to mono np.array


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


def detect_speech_segments(audio_file: str) -> Timeline:
    """Run voice activity detection (VAD) and return speech regions as a Timeline.

    Reads ``HUGGINGFACE_TOKEN`` from the process env — caller is expected to
    have populated it (e.g. via ``Settings.from_env()``).

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        pyannote.core.Timeline: Detected speech segments.

    """
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        err_msg = "Missing Hugging Face token in HUGGINGFACE_TOKEN env variable"
        raise OSError(err_msg)

    # Needs token with access to gated pyannote models:
    # - https://huggingface.co/pyannote/voice-activity-detection
    # - https://huggingface.co/pyannote/segmentation
    vad_pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=token,
    )
    vad_result = vad_pipeline(audio_file)
    return vad_result.get_timeline().support()


def transcribe_audio_with_diarization(
    audio_file: str,
    model_size: str = "base",
    language: str | None = None,
) -> tuple[str, str]:
    """Transcribe audio with speaker diarization.

    ``language=None`` autodetects (default behavior); pass a code (e.g.
    ``"fr"``) to force whisper to that language.
    """
    device = get_device()
    my_logger.info(f"\tUsing device: {device}")
    patch_whisper_progress_bar()
    model = _load_model(model_size, device)

    if language is None:
        used_lang = detect_language(audio_file, model, device)
        my_logger.info(f"Detected language: {used_lang}")
    else:
        used_lang = language
        my_logger.info(f"Forced language: {used_lang}")

    # Diarize speakers
    diarized_segments = diarize_speakers(audio_file)
    speech_timeline = detect_speech_segments(audio_file)

    # Keep only diarized segments that intersect with actual speech
    filtered_segments = [
        (speaker, segment)
        for speaker, segment in diarized_segments
        if speech_timeline.crop(segment)  # returns non-empty Timeline if overlaps
    ]

    # Group segments from the same speaker
    grouped_segments = group_speaker_segments(filtered_segments, max_gap=_MAX_SPEAKER_GAP)
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
        sliced_audio = load_audio_slice(audio_file, segment.start, segment.end)
        segment_result = model.transcribe(
            sliced_audio,
            fp16=(device == "cuda"),
            language=used_lang,
        )
        text = cast(str, segment_result["text"]).strip()
        if not text:  # Skip empty transcriptions
            continue
        full_text.append(f"{speaker}: {text}")

    return "\n".join(full_text), used_lang


def transcribe_video_file_with_diarization(
    video_file: str,
    model_size: str = "base",
    language: str | None = None,
) -> tuple[str, str]:
    """Full pipeline: Extract audio from video, transcribe it with diarization."""
    my_logger.info(f"Processing with diarization: {video_file}")
    audio_path = extract_audio(video_file)
    try:
        transcription, used_lang = transcribe_audio_with_diarization(
            audio_path,
            model_size=model_size,
            language=language,
        )
    finally:
        Path(audio_path).unlink()
    return transcription, used_lang
