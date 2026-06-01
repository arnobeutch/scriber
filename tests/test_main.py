"""Tests for main.main — orchestration only.

Per-handler logic lives in test_handlers.py; per-helper logic in
test_formatting.py. Here we just check that the dispatcher routes to the
right handler and honors `summarize` vs `transcribe` subcommands and
--dry-run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from scriber.main import _maybe_prompt_for_initial_prompt, main
from scriber.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path

_URL = "https://y.com/watch?v=x"
_URL_CLASSIFICATION = {
    "is_url": True,
    "is_file": False,
    "is_media_file": False,
    "is_text_file": False,
}
_MEDIA_CLASSIFICATION = {
    "is_url": False,
    "is_file": True,
    "is_media_file": True,
    "is_text_file": False,
}
_TEXT_CLASSIFICATION = {
    "is_url": False,
    "is_file": True,
    "is_media_file": False,
    "is_text_file": True,
}


def _make_args(**overrides: object) -> MagicMock:
    defaults: dict[str, object] = {
        "command": "transcribe",
        "input_path": [_URL],
        "language": None,
        "diarize": False,
        "with_openai": False,
        "debug": False,
        "model_size": None,
        "llm_provider": None,
        "llm_model": None,
        "output_dir": None,
        "downloads_dir": None,
        "summary_mode": None,
        "force": False,
        "subtitles": False,
        "dry_run": False,
        "continue_on_error": False,
        "context_file": None,
        "no_preprocess": False,
        "initial_prompt_file": None,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_transcript(**overrides: object) -> MagicMock:
    defaults: dict[str, object] = {
        "text": "body",
        "language": "en",
        "title": "T",
        "source": "yt_manual",
        "diarized": False,
    }
    defaults.update(overrides)
    return MagicMock(**defaults)


class TestDispatcher:
    def test_url_branch_routes_to_handle_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("scriber.main.parser.parse_args") as parse,
            patch("scriber.main.initialize_logger"),
            patch("scriber.main.parser.classify_input", return_value=_URL_CLASSIFICATION),
            patch("scriber.main.handlers.handle_url", return_value=_make_transcript()) as h_url,
            patch("scriber.main.handlers.handle_media") as h_media,
            patch("scriber.main.handlers.handle_text") as h_text,
            patch("scriber.main.handlers.write_transcript_file") as write,
            patch("scriber.main.handlers.summarize") as summ,
        ):
            parse.return_value = _make_args(input_path=[_URL])
            main()
        h_url.assert_called_once()
        h_media.assert_not_called()
        h_text.assert_not_called()
        write.assert_called_once()
        summ.assert_not_called()

    def test_media_branch_routes_to_handle_media(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("scriber.main.parser.parse_args") as parse,
            patch("scriber.main.initialize_logger"),
            patch("scriber.main.parser.classify_input", return_value=_MEDIA_CLASSIFICATION),
            patch("scriber.main.handlers.handle_url") as h_url,
            patch("scriber.main.handlers.handle_media", return_value=_make_transcript()) as h_media,
            patch("scriber.main.handlers.handle_text") as h_text,
            patch("scriber.main.handlers.write_transcript_file"),
            patch("scriber.main.handlers.summarize"),
        ):
            parse.return_value = _make_args(input_path=["x.mp4"])
            main()
        h_media.assert_called_once()
        h_url.assert_not_called()
        h_text.assert_not_called()

    def test_text_branch_routes_to_handle_text(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("scriber.main.parser.parse_args") as parse,
            patch("scriber.main.initialize_logger"),
            patch("scriber.main.parser.classify_input", return_value=_TEXT_CLASSIFICATION),
            patch("scriber.main.handlers.handle_url") as h_url,
            patch("scriber.main.handlers.handle_media") as h_media,
            patch("scriber.main.handlers.handle_text", return_value=_make_transcript()) as h_text,
            patch("scriber.main.handlers.write_transcript_file"),
            patch("scriber.main.handlers.summarize"),
        ):
            parse.return_value = _make_args(input_path=["x.txt"])
            main()
        h_text.assert_called_once()
        h_url.assert_not_called()
        h_media.assert_not_called()

    def test_summarize_called_when_subcommand_is_summarize(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # An API-key preflight runs before the pipeline; stub it out.
        with (
            patch("scriber.main.parser.parse_args") as parse,
            patch("scriber.main.initialize_logger"),
            patch("scriber.main.parser.classify_input", return_value=_URL_CLASSIFICATION),
            patch("scriber.main.make_summarizer"),  # preflight ok
            patch("scriber.main.handlers.handle_url", return_value=_make_transcript()),
            patch("scriber.main.handlers.write_transcript_file"),
            patch("scriber.main.handlers.summarize") as summ,
        ):
            parse.return_value = _make_args(input_path=[_URL], command="summarize")
            main()
        summ.assert_called_once()

    def test_transcribe_subcommand_skips_summary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("scriber.main.parser.parse_args") as parse,
            patch("scriber.main.initialize_logger"),
            patch("scriber.main.parser.classify_input", return_value=_URL_CLASSIFICATION),
            patch("scriber.main.make_summarizer") as preflight,
            patch("scriber.main.handlers.handle_url", return_value=_make_transcript()),
            patch("scriber.main.handlers.write_transcript_file"),
            patch("scriber.main.handlers.summarize") as summ,
        ):
            parse.return_value = _make_args(input_path=[_URL], command="transcribe")
            main()
        # Both preflight and summarize must be skipped under the transcribe subcommand.
        preflight.assert_not_called()
        summ.assert_not_called()

    def test_dry_run_skips_all_work(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("scriber.main.parser.parse_args") as parse,
            patch("scriber.main.initialize_logger"),
            patch("scriber.main.parser.classify_input", return_value=_URL_CLASSIFICATION),
            patch("scriber.main.handlers.handle_url") as h_url,
            patch("scriber.main.handlers.write_transcript_file") as write,
        ):
            parse.return_value = _make_args(input_path=[_URL], dry_run=True)
            main()
        h_url.assert_not_called()
        write.assert_not_called()

    def test_batch_mode_processes_multiple_inputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        url2 = "https://y.com/watch?v=y"
        with (
            patch("scriber.main.parser.parse_args") as parse,
            patch("scriber.main.initialize_logger"),
            patch("scriber.main.parser.classify_input", return_value=_URL_CLASSIFICATION),
            patch("scriber.main.handlers.handle_url", return_value=_make_transcript()) as h_url,
            patch("scriber.main.handlers.write_transcript_file"),
        ):
            parse.return_value = _make_args(input_path=[_URL, url2])
            main()
        assert h_url.call_count == 2

    def test_failure_aborts_batch_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        url2 = "https://y.com/watch?v=y"
        with (
            patch("scriber.main.parser.parse_args") as parse,
            patch("scriber.main.initialize_logger"),
            patch("scriber.main.parser.classify_input", return_value=_URL_CLASSIFICATION),
            patch(
                "scriber.main.handlers.handle_url",
                side_effect=[RuntimeError("first failed"), _make_transcript()],
            ) as h_url,
            patch("scriber.main.handlers.write_transcript_file"),
        ):
            parse.return_value = _make_args(input_path=[_URL, url2])
            with pytest.raises(RuntimeError, match="first failed"):
                main()
        # Second input never attempted — default behavior.
        assert h_url.call_count == 1

    def test_continue_on_error_processes_remaining_and_exits_nonzero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        url2 = "https://y.com/watch?v=y"
        with (
            patch("scriber.main.parser.parse_args") as parse,
            patch("scriber.main.initialize_logger"),
            patch("scriber.main.parser.classify_input", return_value=_URL_CLASSIFICATION),
            patch(
                "scriber.main.handlers.handle_url",
                side_effect=[RuntimeError("geo-blocked"), _make_transcript()],
            ) as h_url,
            patch("scriber.main.handlers.write_transcript_file"),
        ):
            parse.return_value = _make_args(
                input_path=[_URL, url2],
                continue_on_error=True,
            )
            with pytest.raises(SystemExit) as excinfo:
                main()
        # Both inputs attempted; non-zero exit because one failed.
        assert h_url.call_count == 2
        assert excinfo.value.code == 1


def _bare_settings() -> Settings:
    """Minimal Settings for prompt-helper tests (no env, no file IO)."""
    return Settings(initial_prompt=None)


class TestMaybePromptForInitialPrompt:
    def test_skipped_when_primer_already_loaded(self) -> None:
        s = _bare_settings()
        s = Settings(initial_prompt="already set")
        with patch("scriber.main.sys.stdin.isatty", return_value=True):
            out = _maybe_prompt_for_initial_prompt(
                _make_args(input_path=[_URL]),
                s,
            )
        assert out.initial_prompt == "already set"

    def test_skipped_when_dry_run(self) -> None:
        with patch("scriber.main.sys.stdin.isatty", return_value=True):
            out = _maybe_prompt_for_initial_prompt(
                _make_args(input_path=[_URL], dry_run=True),
                _bare_settings(),
            )
        assert out.initial_prompt is None

    def test_skipped_when_stdin_not_tty(self) -> None:
        with patch("scriber.main.sys.stdin.isatty", return_value=False):
            out = _maybe_prompt_for_initial_prompt(
                _make_args(input_path=[_URL]),
                _bare_settings(),
            )
        assert out.initial_prompt is None

    def test_skipped_when_only_text_inputs(
        self,
        tmp_path: Path,
    ) -> None:
        txt = tmp_path / "doc.txt"
        txt.write_text("hello", encoding="utf-8")
        with (
            patch("scriber.main.sys.stdin.isatty", return_value=True),
            patch("scriber.main.input") as ip,
        ):
            out = _maybe_prompt_for_initial_prompt(
                _make_args(input_path=[str(txt)]),
                _bare_settings(),
            )
        ip.assert_not_called()
        assert out.initial_prompt is None

    def test_empty_answer_proceeds_without_primer(self, tmp_path: Path) -> None:
        media = tmp_path / "clip.mp4"
        media.write_text("")
        with (
            patch("scriber.main.sys.stdin.isatty", return_value=True),
            patch("scriber.main.input", return_value="   "),
            patch("scriber.main.sys.stderr"),
        ):
            out = _maybe_prompt_for_initial_prompt(
                _make_args(input_path=[str(media)]),
                _bare_settings(),
            )
        assert out.initial_prompt is None

    def test_valid_path_loads_primer(self, tmp_path: Path) -> None:
        media = tmp_path / "clip.mp4"
        media.write_text("")
        primer = tmp_path / "primer.fr.txt"
        primer.write_text("Christophe, Anne, K•LINE.", encoding="utf-8")
        with (
            patch("scriber.main.sys.stdin.isatty", return_value=True),
            patch("scriber.main.input", return_value=str(primer)),
            patch("scriber.main.sys.stderr"),
        ):
            out = _maybe_prompt_for_initial_prompt(
                _make_args(input_path=[str(media)]),
                _bare_settings(),
            )
        assert out.initial_prompt == "Christophe, Anne, K•LINE."

    def test_bogus_path_warns_and_proceeds(self, tmp_path: Path) -> None:
        media = tmp_path / "clip.mp4"
        media.write_text("")
        with (
            patch("scriber.main.sys.stdin.isatty", return_value=True),
            patch("scriber.main.input", return_value="/nonexistent/primer.txt"),
            patch("scriber.main.sys.stderr"),
        ):
            out = _maybe_prompt_for_initial_prompt(
                _make_args(input_path=[str(media)]),
                _bare_settings(),
            )
        assert out.initial_prompt is None
