"""Scriber entry point: transcribe and summarize YouTube videos, local media, or transcripts."""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import torch.cuda

from scriber import handlers, parser
from scriber.logger import initialize_logger, my_logger
from scriber.settings import Settings, load_text_file
from scriber.summarizers import MissingAPIKeyError, make_summarizer


@dataclass(frozen=True)
class _InputResult:
    """One row of the post-run summary table."""

    path: str
    status: str  # "ok" or "error"
    detail: str  # output path on success, error message on failure


def _apply_cli_overrides(args: argparse.Namespace, base: Settings) -> Settings:
    """Return a new ``Settings`` with CLI-provided values overlaid on ``base``.

    Summarize-only flags (``--llm-provider``, ``--llm-model``, ``--summary-mode``,
    ``--with-openai``) are only present when the subcommand is ``summarize``.
    """
    provider = getattr(args, "llm_provider", None) or base.llm_provider
    if getattr(args, "with_openai", False):
        provider = "openai"
    preprocess = base.preprocess_audio
    if getattr(args, "no_preprocess", False):
        preprocess = False
    initial_prompt = base.initial_prompt
    cli_prompt_path = getattr(args, "initial_prompt_file", None)
    if cli_prompt_path is not None:
        loaded = load_text_file(cli_prompt_path)
        if loaded is None:
            my_logger.warning(
                f"--initial-prompt-file {cli_prompt_path!s} not found or empty; "
                "proceeding without a primer.",
            )
        else:
            initial_prompt = loaded
    return dataclasses.replace(
        base,
        output_dir=args.output_dir or base.output_dir,
        downloads_dir=args.downloads_dir or base.downloads_dir,
        whisper_model_size=args.model_size or base.whisper_model_size,
        llm_provider=provider,
        llm_model=getattr(args, "llm_model", None) or base.llm_model,
        summary_mode=getattr(args, "summary_mode", None) or base.summary_mode,
        preprocess_audio=preprocess,
        initial_prompt=initial_prompt,
    )


def _maybe_prompt_for_initial_prompt(
    args: argparse.Namespace,
    settings: Settings,
) -> Settings:
    """Interactively offer a primer file when none was provided.

    Skipped when:
      - a primer is already loaded (CLI flag or env var)
      - ``--dry-run`` is set
      - stdin is not a TTY (scripts / pipes / CI)
      - none of the inputs would route to whisper (text-only batch)
    """
    if settings.initial_prompt is not None:
        return settings
    if args.dry_run:
        return settings
    if not sys.stdin.isatty():
        return settings
    needs_whisper = any(
        parser.classify_input(p)["is_url"] or parser.classify_input(p)["is_media_file"]
        for p in args.input_path
    )
    if not needs_whisper:
        return settings

    sys.stderr.write(
        "\nNo --initial-prompt-file given. A primer file is a short text "
        "(in the audio's language) listing proper nouns, acronyms, and "
        "jargon you expect in the recording — it can substantially improve "
        "transcription of brand-name-dense content.\n"
        "See docs/WHISPER_SETUP.md for what to put in it.\n\n",
    )
    answer = input("Enter primer file path, or press Enter to skip: ").strip()
    if not answer:
        return settings
    loaded = load_text_file(Path(answer))
    if loaded is None:
        my_logger.warning(
            f"Primer file {answer!r} not found or empty; proceeding without.",
        )
        return settings
    return dataclasses.replace(settings, initial_prompt=loaded)


def _gpu_warning() -> None:
    """Warn when nvidia-smi exists but CUDA is unavailable (driver/runtime mismatch)."""
    if shutil.which("nvidia-smi") and not torch.cuda.is_available():
        my_logger.warning(
            "nvidia-smi found but torch.cuda.is_available() is False — "
            "whisper will run on CPU. Check your CUDA driver/runtime installation.",
        )


def _dry_run_report(path: str, classification: dict[str, bool], settings: Settings) -> None:
    """Print a one-line dry-run summary for a single input."""
    if classification["is_url"]:
        kind = "youtube-url"
    elif classification["is_media_file"]:
        kind = "local-media"
    elif classification["is_text_file"]:
        kind = "local-text"
    else:
        kind = "local-file (unknown type)"
    my_logger.info(
        f"[dry-run] {path!r} → {kind} | model: {settings.whisper_model_size} | "
        f"output: {settings.output_dir}/",
    )


def _process_one(
    path: str,
    args: argparse.Namespace,
    settings: Settings,
    *,
    will_summarize: bool,
) -> _InputResult:
    """Transcribe (+ optionally summarize) one input; return a summary row."""
    classification = parser.classify_input(path)
    per_args = argparse.Namespace(**vars(args))
    per_args.input_path = path
    for key, val in classification.items():
        setattr(per_args, key, val)

    if per_args.is_url:
        transcript = handlers.handle_url(per_args, settings)
    elif per_args.is_media_file:
        transcript = handlers.handle_media(per_args, settings)
    elif per_args.is_text_file:
        transcript = handlers.handle_text(per_args, settings)
    else:
        err_msg = f"No handler for the given input type: {path}"
        raise RuntimeError(err_msg)

    transcript_path = handlers.write_transcript_file(
        transcript,
        settings,
        subtitles=args.subtitles,
    )
    my_logger.info(f"Video title: {transcript.title}")

    if will_summarize:
        my_logger.info("Generating summary...")
        handlers.summarize(transcript, per_args, settings)

    return _InputResult(path=path, status="ok", detail=str(transcript_path))


def _print_batch_summary(results: list[_InputResult]) -> None:
    """Print a ✓/✗ table after a multi-input run."""
    if len(results) <= 1:
        return
    ok = sum(1 for r in results if r.status == "ok")
    bad = len(results) - ok
    my_logger.info(f"Batch summary: {ok} ok, {bad} failed")
    for r in results:
        mark = "✓" if r.status == "ok" else "✗"
        my_logger.info(f"  {mark} {r.path}  —  {r.detail}")


def main() -> None:
    """Parse args, build a Transcript for each input, write it, and optionally summarize."""
    args = parser.parse_args()
    initialize_logger(args)
    settings = _apply_cli_overrides(args, Settings.from_env())
    settings = _maybe_prompt_for_initial_prompt(args, settings)

    my_logger.info(f"Script called with the following arguments: {vars(args)}")
    my_logger.debug(f"Loaded settings: {settings}")

    _gpu_warning()

    will_summarize = args.command == "summarize"

    # Preflight the LLM backend BEFORE the slow transcription pipeline so a
    # missing API key fails in seconds, not after a 10-minute whisper run.
    if will_summarize and not args.dry_run:
        try:
            make_summarizer(settings)
        except MissingAPIKeyError as exc:
            my_logger.error(str(exc))
            sys.exit(2)

    results: list[_InputResult] = []
    for path in args.input_path:
        if args.dry_run:
            classification = parser.classify_input(path)
            _dry_run_report(path, classification, settings)
            continue

        try:
            results.append(
                _process_one(path, args, settings, will_summarize=will_summarize),
            )
        except Exception as exc:
            if not getattr(args, "continue_on_error", False):
                raise
            # Top-level batch boundary — capture anything so one bad input
            # doesn't kill the rest of the run.
            my_logger.error(f"Failed on {path!r}: {exc}")
            results.append(_InputResult(path=path, status="error", detail=str(exc)))

    _print_batch_summary(results)
    if any(r.status == "error" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        my_logger.critical("Interrupted by user")
        sys.exit(0)
