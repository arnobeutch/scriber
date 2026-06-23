# pyright: reportUnknownVariableType=false
"""Per-source handlers: URL, local media, local text file.

Each ``handle_*`` returns a :class:`Transcript` capturing what was produced
and where it came from. ``main.py`` is then a thin orchestrator that picks
the right handler, writes the transcript to disk, and optionally summarizes.

The ``# pyright`` header above suppresses ``reportUnknownVariableType`` across
this file — ``langdetect``'s public ``detect`` returns an annotated-but-
``Unknown`` type, and the pattern propagates everywhere we touch it.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from langdetect import LangDetectException, detect

from scriber.formatting import sanitize_filename, wrap_transcript
from scriber.language import derive_summary_language, derive_whisper_summary_language
from scriber.logger import my_logger
from scriber.model import SourceMetadata, Transcript
from scriber.settings import Settings
from scriber.subtitles import write_srt, write_vtt
from scriber.transcription import local as plt
from scriber.transcription import youtube_audio as pya
from scriber.transcription import youtube_captions as pytt
from scriber.transcription.youtube_captions import TranscriptUnavailableError

# Optional-extra hints — raised when an opt-in dependency group isn't installed.
_DIARIZE_EXTRA_HINT = (
    "Speaker diarization (--diarize) requires the 'diarize' extra. Install it with:\n"
    "  uv sync --extra diarize               (in the scriber repo)\n"
    "  uv tool install 'scriber[diarize]'    (as a standalone tool)\n"
    "and set HUGGINGFACE_TOKEN for the gated pyannote models."
)
_SUMMARIZE_EXTRA_HINT = (
    "Summarization requires the 'summarize' extra. Install it with:\n"
    "  uv sync --extra summarize             (in the scriber repo)\n"
    "  uv tool install 'scriber[summarize]'  (as a standalone tool)"
)


def handle_url(args: argparse.Namespace, settings: Settings) -> Transcript:
    """Fetch a YT transcript honoring the language ladder; whisper-fallback if absent.

    When ``--diarize`` is set the caption shortcut is skipped — captions
    don't carry speaker attribution, so we always go through whisper +
    pyannote on the downloaded audio.
    """
    video_id = pya.extract_video_id(args.input_path)
    my_logger.debug(f"Video ID: {video_id}")
    requested_lang: str | None = args.language
    force: bool = bool(getattr(args, "force", False))

    if args.diarize:
        my_logger.info("--diarize requested — skipping captions, downloading audio.")
        return _transcribe_url_via_whisper(args, settings, requested_lang, force=force)

    try:
        track = pytt.get_youtube_transcript(video_id, requested_lang=requested_lang)
    except TranscriptUnavailableError as exc:
        log = my_logger.warning if exc.reason == "download_failed" else my_logger.info
        log(
            f"No YouTube transcript available ({exc.reason}: {exc}) — "
            f"falling back to local transcription.",
        )
        return _transcribe_url_via_whisper(args, settings, requested_lang, force=force)

    raw_title, chapters, metadata = pya.fetch_video_metadata(args.input_path)
    # When --language wasn't set, the uploader-declared language (relayed via
    # CaptionTrack) acts as the implicit preference for the summary too —
    # otherwise a French video with an English manual sub would yield an
    # English summary even though we just picked the French track.
    effective_lang = requested_lang or track.declared_language
    summary_lang = derive_summary_language(track.lang, effective_lang)
    my_logger.info(
        f"Caption track: {track.kind} '{track.lang}'; summary language: {summary_lang}",
    )
    return Transcript(
        text=track.text,
        language=summary_lang,
        title=sanitize_filename(raw_title),
        source="yt_manual" if track.kind == "manual" else "yt_auto",
        diarized=False,
        chapters=chapters,
        metadata=replace(metadata, detected_language=track.lang),
    )


def _transcribe_url_via_whisper(
    args: argparse.Namespace,
    settings: Settings,
    requested_lang: str | None,
    *,
    force: bool,
) -> Transcript:
    """Download audio and transcribe via whisper (optionally pyannote-diarized)."""
    audio_path, raw_title, chapters, metadata = pya.download_youtube_audio(
        args.input_path,
        settings.downloads_dir,
        force=force,
    )
    title = sanitize_filename(raw_title)

    cached = _try_load_cached_transcript(title, settings, diarize=args.diarize, force=force)
    if cached is not None:
        return Transcript(
            text=cached,
            language=derive_whisper_summary_language(
                requested_lang or "en",
                requested_lang,
            ),
            title=title,
            source="whisper",
            diarized=args.diarize,
            chapters=chapters,
            metadata=replace(metadata, detected_language=requested_lang),
        )

    want_words = bool(getattr(args, "suggest_primer", False))
    segments: list[dict[str, object]] = []
    if args.diarize:
        try:
            from scriber.transcription.diarize import transcribe_audio_with_diarization
        except ImportError as exc:
            raise RuntimeError(_DIARIZE_EXTRA_HINT) from exc
        transcribed_text, used_lang, segments = transcribe_audio_with_diarization(
            str(audio_path),
            model_size=settings.whisper_model_size,
            language=requested_lang,
            preprocess=settings.preprocess_audio,
            initial_prompt=settings.initial_prompt,
            min_speakers=settings.min_speakers,
            max_speakers=settings.max_speakers,
            word_timestamps=want_words,
        )
    else:
        transcribed_text, used_lang, segments = plt.transcribe_audio_full(
            str(audio_path),
            model_size=settings.whisper_model_size,
            language=requested_lang,
            preprocess=settings.preprocess_audio,
            initial_prompt=settings.initial_prompt,
            word_timestamps=want_words,
        )
    summary_lang = derive_whisper_summary_language(used_lang, requested_lang)
    return Transcript(
        text=transcribed_text,
        language=summary_lang,
        title=title,
        source="whisper",
        diarized=args.diarize,
        segments=segments,
        chapters=chapters,
        metadata=replace(metadata, detected_language=used_lang),
    )


def _try_load_cached_transcript(
    title: str,
    settings: Settings,
    *,
    diarize: bool,
    force: bool,
) -> str | None:
    """Return cached transcript text if a matching ``.txt`` already exists."""
    if force:
        return None
    suffix = " diarized transcript" if diarize else " transcript"
    cached = settings.output_dir / f"{title}{suffix}.txt"
    if not cached.exists():
        return None
    my_logger.info(f"Using cached transcript at {cached}")
    return cached.read_text(encoding="utf8")


def handle_media(args: argparse.Namespace, settings: Settings) -> Transcript:
    """Transcribe a local media file via whisper (optionally pyannote-diarized).

    ``--language`` (if set) forces whisper to that language and becomes the
    summary language. Otherwise whisper autodetects; if the detection lands
    on en/fr the summary follows; otherwise summary is forced to English.
    """
    title = sanitize_filename(Path(args.input_path).stem)
    requested_lang: str | None = args.language
    want_words = bool(getattr(args, "suggest_primer", False))
    segments: list[dict[str, object]] = []
    if args.diarize:
        try:
            from scriber.transcription.diarize import transcribe_video_file_with_diarization
        except ImportError as exc:
            raise RuntimeError(_DIARIZE_EXTRA_HINT) from exc
        text, used_lang, segments = transcribe_video_file_with_diarization(
            args.input_path,
            model_size=settings.whisper_model_size,
            language=requested_lang,
            preprocess=settings.preprocess_audio,
            initial_prompt=settings.initial_prompt,
            min_speakers=settings.min_speakers,
            max_speakers=settings.max_speakers,
            word_timestamps=want_words,
        )
    else:
        # transcribe_video_file is a thin wrapper around transcribe_audio_full
        # via tempfile-based ffmpeg extraction; we duplicate the unwrap here
        # so segments are exposed to handle_media too.
        audio_tmp = plt.extract_audio(args.input_path)
        try:
            text, used_lang, segments = plt.transcribe_audio_full(
                audio_tmp,
                model_size=settings.whisper_model_size,
                language=requested_lang,
                preprocess=settings.preprocess_audio,
                initial_prompt=settings.initial_prompt,
                word_timestamps=want_words,
            )
        finally:
            Path(audio_tmp).unlink()
    summary_lang = derive_whisper_summary_language(used_lang, requested_lang)
    my_logger.info(f"Transcribed in '{used_lang}'; summary language: {summary_lang}")
    return Transcript(
        text=text,
        language=summary_lang,
        title=title,
        source="whisper",
        diarized=args.diarize,
        segments=segments,
        metadata=SourceMetadata(detected_language=used_lang),
    )


def _detect_text_language(text: str) -> str:
    """Best-effort language detection; defaults to ``"en"`` on failure."""
    try:
        detected = detect(text)
    except LangDetectException:
        my_logger.warning("Could not detect text language; defaulting to 'en'")
        return "en"
    return cast(str, detected)


def handle_text(args: argparse.Namespace, settings: Settings) -> Transcript:
    """Read a pre-existing transcript from disk."""
    _ = settings  # reserved for future use
    text = Path(args.input_path).read_text(encoding="utf8")
    requested_lang: str | None = args.language
    detected = requested_lang if requested_lang else _detect_text_language(text)
    summary_lang = derive_whisper_summary_language(detected, requested_lang)
    my_logger.info(f"Text-file language: {detected}; summary language: {summary_lang}")
    return Transcript(
        text=text,
        language=summary_lang,
        title=sanitize_filename(Path(args.input_path).stem),
        source="file",
        diarized=False,
        metadata=SourceMetadata(detected_language=detected),
    )


def write_transcript_file(
    transcript: Transcript,
    settings: Settings,
    *,
    subtitles: bool = False,
) -> Path:
    """Write the transcript text to ``<output_dir>/<title> [diarized] transcript.txt``.

    When ``subtitles`` is True and the transcript carries whisper segments,
    also writes ``.srt`` and ``.vtt`` files alongside.
    """
    suffix = " diarized transcript" if transcript.diarized else " transcript"
    p = settings.output_dir / f"{transcript.title}{suffix}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        wrap_transcript(transcript.text, diarize=transcript.diarized, width=settings.wrap_width),
        encoding="utf8",
    )
    my_logger.info(f"Transcript written to {p}")

    if subtitles:
        if not transcript.segments:
            my_logger.warning(
                "--subtitles requested but no segments available "
                "(YT captions or diarized output) — skipping .srt/.vtt.",
            )
        else:
            srt_path = settings.output_dir / f"{transcript.title}.srt"
            vtt_path = settings.output_dir / f"{transcript.title}.vtt"
            write_srt(transcript.segments, srt_path)
            write_vtt(transcript.segments, vtt_path)
            my_logger.info(f"Subtitles written to {srt_path} and {vtt_path}")

    return p


def write_primer_draft(transcript: Transcript, settings: Settings) -> Path | None:
    """Write a reviewable primer draft from the transcription's whisper segments.

    Returns the draft path, or ``None`` when there are no segments to harvest
    (e.g. a cached transcript or YT-caption source). Review and trim the draft,
    then feed it back via ``--initial-prompt-file`` for a consolidation pass.
    """
    from scriber.primer import extract_primer_candidates, format_primer_draft

    if not transcript.segments:
        my_logger.warning(
            "--suggest-primer: no whisper segments available "
            "(cached transcript or YT captions) — skipping primer draft.",
        )
        return None
    candidates = extract_primer_candidates(cast("list[dict[str, Any]]", transcript.segments))
    draft = format_primer_draft(candidates, transcript.title)
    p = settings.output_dir / f"{transcript.title} primer.draft.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(draft, encoding="utf-8")
    my_logger.info(f"Primer draft written to {p} — review, then pass via --initial-prompt-file.")
    return p


def summarize(transcript: Transcript, args: argparse.Namespace, settings: Settings) -> None:
    """Dispatch to the configured Summarizer backend."""
    try:
        from scriber.summarizers import make_summarizer
    except ImportError as exc:
        raise RuntimeError(_SUMMARIZE_EXTRA_HINT) from exc
    summarizer = make_summarizer(settings)
    context = _load_context_file(getattr(args, "context_file", None))
    summarizer.summarize(transcript, input_path=args.input_path, context=context)


def _load_context_file(path: Path | None) -> str | None:
    """Read the ``--context-file`` contents, or return ``None`` when absent."""
    if path is None:
        return None
    if not path.is_file():
        my_logger.warning(f"--context-file {path} not found; ignoring.")
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        my_logger.warning(f"--context-file {path} is empty; ignoring.")
        return None
    my_logger.info(f"Loaded context from {path} ({len(text)} chars)")
    return text
