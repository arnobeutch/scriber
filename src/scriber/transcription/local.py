# Boundary to untyped ML deps (whisper, ffmpeg). 2026-06-02:
# suppress unknown-type reports here; keep call/argument/attribute checks on.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false
"""Retrieve the text transcript from a local media file (whisper transcription).

The pyannote/torchaudio diarization path lives in
:mod:`scriber.transcription.diarize`, imported lazily so the base install
(without ``scriber[diarize]``) never pulls those deps.
"""

import tempfile
from pathlib import Path
from typing import Any, cast

import ffmpeg
import torch.cuda
import whisper
from tqdm import tqdm

from scriber.logger import my_logger

_MODEL_CACHE: dict[tuple[str, str], whisper.Whisper] = {}

# Audio pre-processing applied before every whisper transcription unless
# explicitly disabled. ``alimiter`` caps intersample peaks (some recorders
# saturate when a speaker raises their voice); ``dynaudnorm`` flattens the
# loud-near / faint-far disparity common in room recordings with a remote
# participant on a loudspeaker. Picked from the transcription-tuning bench
# (see experiments/transcription-tuning/RESULTS.md): the unfiltered
# baseline triggered whisper's temperature-fallback cascade on clipped
# stretches (6x CPU cost) and this filter chain prevented it across every
# test sample.
_PREPROCESS_FILTER: str = "alimiter=limit=0.95:level=disabled,dynaudnorm"


def preprocess_audio_file(input_path: str) -> str:
    """Apply the audio pre-processing filter chain; return a tempfile path.

    The caller owns the returned path and must ``unlink`` it when done.
    Output is 16kHz mono PCM s16le — whisper's native input format.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    my_logger.info(f"Pre-processing audio: {_PREPROCESS_FILTER}")
    ffmpeg.input(input_path).output(
        out_path,
        af=_PREPROCESS_FILTER,
        ar="16000",
        ac=1,
        acodec="pcm_s16le",
        format="wav",
    ).run(quiet=True, overwrite_output=True)
    return out_path


def maybe_preprocess(audio_file: str, *, preprocess: bool) -> tuple[str, bool]:
    """Return ``(path, owns_path)``. Caller unlinks when owns_path is True."""
    if not preprocess:
        return audio_file, False
    return preprocess_audio_file(audio_file), True


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
    # n_mels must match the model: large-v3 / large-v3-turbo encoders expect 128,
    # smaller models 80. log_mel_spectrogram defaults to 80, so feeding a large-v3
    # model an 80-mel tensor raises a channel-mismatch RuntimeError. model.transcribe
    # picks the right n_mels internally; this hand-rolled detect path must too.
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(device)
    _, probs = model.detect_language(mel)
    probs_dict = cast(dict[str, float], probs)
    return max(probs_dict, key=lambda k: probs_dict[k])


def load_model(model_size: str, device: str) -> whisper.Whisper:
    """Return a cached Whisper model, loading it on first use."""
    key = (model_size, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = whisper.load_model(model_size, device=device)
    return _MODEL_CACHE[key]


def transcribe_audio_full(
    audio_file: str,
    model_size: str = "base",
    language: str | None = None,
    *,
    preprocess: bool = True,
    initial_prompt: str | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Transcribe + return ``(text, language, segments)``.

    ``segments`` is whisper's per-cue list with ``start`` / ``end`` /
    ``text`` keys, suitable for SRT / VTT export. ``preprocess`` runs the
    default ffmpeg filter chain (``alimiter + dynaudnorm``) before
    transcription unless disabled. ``initial_prompt`` seeds whisper's
    decoder with a primer text (proper nouns, acronyms, jargon) — see
    docs/WHISPER_SETUP.md.
    """
    my_logger.info(f"Transcribing audio file: {audio_file}")
    audio_file, owns_temp = maybe_preprocess(audio_file, preprocess=preprocess)
    try:
        device = get_device()
        my_logger.info(f"\tUsing device: {device}")
        patch_whisper_progress_bar()
        model = load_model(model_size, device)

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
            # Don't let prior segments bias the next decode. Whisper's
            # default (True) cascades hallucinations on long files — once
            # a wrong phrase enters context, the next segment is biased
            # toward it and the error compounds. See docs/WHISPER_SETUP.md.
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
        )
        segments = cast(list[dict[str, Any]], result.get("segments", []))
        return cast(str, result["text"]), used_lang, segments
    finally:
        if owns_temp:
            Path(audio_file).unlink(missing_ok=True)
