"""Tests for scriber.parser."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

from scriber.parser import classify_input, is_valid_url, parse_args

if TYPE_CHECKING:
    from pathlib import Path


class TestIsValidUrl:
    def test_http_url(self) -> None:
        assert is_valid_url("http://example.com")

    def test_https_url(self) -> None:
        assert is_valid_url("https://www.youtube.com/watch?v=abc123")

    def test_url_with_path(self) -> None:
        assert is_valid_url("https://example.com/path/to/thing")

    def test_empty_string(self) -> None:
        assert not is_valid_url("")

    def test_plain_string(self) -> None:
        assert not is_valid_url("just-a-string")

    def test_scheme_without_netloc(self) -> None:
        assert not is_valid_url("http://")

    def test_relative_path(self) -> None:
        assert not is_valid_url("./my_video.mp4")


class TestClassifyInput:
    def test_url_is_classified_as_url(self) -> None:
        c = classify_input("https://youtube.com/watch?v=abc")
        assert c["is_url"] is True
        assert c["is_file"] is False
        assert c["is_media_file"] is False
        assert c["is_text_file"] is False

    def test_media_file_mp4(self, tmp_path: Path) -> None:
        f = tmp_path / "clip.mp4"
        f.write_text("")
        c = classify_input(str(f))
        assert c["is_file"] is True
        assert c["is_media_file"] is True
        assert c["is_text_file"] is False
        assert c["is_url"] is False

    def test_media_file_uppercase_ext(self, tmp_path: Path) -> None:
        f = tmp_path / "clip.MP4"
        f.write_text("")
        c = classify_input(str(f))
        assert c["is_media_file"] is True

    @pytest.mark.parametrize("ext", [".mp4", ".mp3", ".wav", ".mkv", ".avi", ".webm", ".m4a"])
    def test_all_media_extensions(self, ext: str, tmp_path: Path) -> None:
        f = tmp_path / f"clip{ext}"
        f.write_text("")
        assert classify_input(str(f))["is_media_file"] is True

    @pytest.mark.parametrize("ext", [".txt", ".srt", ".vtt"])
    def test_all_text_extensions(self, ext: str, tmp_path: Path) -> None:
        f = tmp_path / f"tr{ext}"
        f.write_text("")
        c = classify_input(str(f))
        assert c["is_text_file"] is True
        assert c["is_media_file"] is False

    def test_unknown_extension_is_neither_media_nor_text(self, tmp_path: Path) -> None:
        p = tmp_path / "thing.pdf"
        p.write_text("")
        c = classify_input(str(p))
        assert c["is_file"] is True
        assert c["is_media_file"] is False
        assert c["is_text_file"] is False

    def test_invalid_path_raises(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid input path"):
            classify_input("not-a-url-nor-file")


def _run_parser(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    monkeypatch.setattr("sys.argv", ["scriber", *argv])
    return parse_args()


class TestSubcommands:
    def test_missing_subcommand_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            _run_parser(["https://y.com/watch?v=x"], monkeypatch)

    def test_unknown_subcommand_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            _run_parser(["process", "https://y.com/watch?v=x"], monkeypatch)

    def test_transcribe_command_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["transcribe", "https://y.com/watch?v=x"], monkeypatch)
        assert ns.command == "transcribe"

    def test_summarize_command_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["summarize", "https://y.com/watch?v=x"], monkeypatch)
        assert ns.command == "summarize"


class TestTranscribeSubcommand:
    def test_url_input_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["transcribe", "https://youtube.com/watch?v=abc"], monkeypatch)
        assert ns.input_path == ["https://youtube.com/watch?v=abc"]

    def test_multiple_inputs_accepted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        f = tmp_path / "clip.mp4"
        f.write_text("")
        ns = _run_parser(
            ["transcribe", "https://youtube.com/watch?v=abc", str(f)],
            monkeypatch,
        )
        assert len(ns.input_path) == 2

    def test_invalid_input_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="Invalid input path"):
            _run_parser(["transcribe", "not-a-url-nor-file"], monkeypatch)

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["transcribe", "https://y.com/watch?v=x"], monkeypatch)
        assert ns.language is None  # autodetect
        assert ns.diarize is False
        assert ns.debug is False
        assert ns.dry_run is False
        assert ns.force is False
        assert ns.subtitles is False
        assert ns.no_preprocess is False  # preprocessing is on by default

    def test_summarize_only_flags_rejected_under_transcribe(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for flag in ("--with-openai", "--llm-provider", "--llm-model", "--summary-mode"):
            with pytest.raises(SystemExit):
                _run_parser(
                    ["transcribe", "https://y.com/watch?v=x", flag, "openai"]
                    if flag != "--with-openai"
                    else ["transcribe", "https://y.com/watch?v=x", flag],
                    monkeypatch,
                )

    def test_language_fr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["transcribe", "https://y.com/watch?v=x", "--language", "fr"],
            monkeypatch,
        )
        assert ns.language == "fr"

    def test_language_short_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["transcribe", "https://y.com/watch?v=x", "-l", "fr"], monkeypatch)
        assert ns.language == "fr"

    def test_debug_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["transcribe", "https://y.com/watch?v=x", "-d"], monkeypatch)
        assert ns.debug is True

    def test_force_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["transcribe", "https://y.com/watch?v=x", "--force"], monkeypatch)
        assert ns.force is True

    def test_no_preprocess_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["transcribe", "https://y.com/watch?v=x", "--no-preprocess"],
            monkeypatch,
        )
        assert ns.no_preprocess is True

    def test_initial_prompt_file_default_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["transcribe", "https://y.com/watch?v=x"], monkeypatch)
        assert ns.initial_prompt_file is None

    def test_initial_prompt_file_accepted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        primer = tmp_path / "primer.fr.txt"
        primer.write_text("test", encoding="utf-8")
        ns = _run_parser(
            [
                "transcribe",
                "https://y.com/watch?v=x",
                "--initial-prompt-file",
                str(primer),
            ],
            monkeypatch,
        )
        assert ns.initial_prompt_file == primer

    def test_subtitles_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["transcribe", "https://y.com/watch?v=x", "--subtitles"], monkeypatch)
        assert ns.subtitles is True

    def test_dry_run_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["transcribe", "https://y.com/watch?v=x", "--dry-run"], monkeypatch)
        assert ns.dry_run is True

    def test_model_size_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["transcribe", "https://y.com/watch?v=x", "--model-size", "medium"],
            monkeypatch,
        )
        assert ns.model_size == "medium"

    def test_model_size_invalid_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            _run_parser(
                ["transcribe", "https://y.com/watch?v=x", "--model-size", "huge"],
                monkeypatch,
            )

    def test_output_dir_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "out"
        ns = _run_parser(
            ["transcribe", "https://y.com/watch?v=x", "--output-dir", str(target)],
            monkeypatch,
        )
        assert ns.output_dir == target

    def test_downloads_dir_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "dl"
        ns = _run_parser(
            ["transcribe", "https://y.com/watch?v=x", "--downloads-dir", str(target)],
            monkeypatch,
        )
        assert ns.downloads_dir == target

    def test_unsupported_language_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            _run_parser(["transcribe", "https://y.com/watch?v=x", "-l", "de"], monkeypatch)


class TestSummarizeSubcommand:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(["summarize", "https://y.com/watch?v=x"], monkeypatch)
        assert ns.with_openai is False
        assert ns.llm_provider is None
        assert ns.llm_model is None
        assert ns.summary_mode is None

    def test_all_summarize_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            [
                "summarize",
                "https://y.com/watch?v=x",
                "--diarize",
                "--with-openai",
                "--summary-mode",
                "meeting",
            ],
            monkeypatch,
        )
        assert ns.diarize is True
        assert ns.with_openai is True
        assert ns.summary_mode == "meeting"

    def test_with_openai_canonical_form(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["summarize", "https://y.com/watch?v=x", "--with-openai"],
            monkeypatch,
        )
        assert ns.with_openai is True

    def test_with_openai_legacy_underscore_still_works(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ns = _run_parser(
            ["summarize", "https://y.com/watch?v=x", "--with_openai"],
            monkeypatch,
        )
        assert ns.with_openai is True

    def test_summary_mode_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["summarize", "https://y.com/watch?v=x", "--summary-mode", "meeting"],
            monkeypatch,
        )
        assert ns.summary_mode == "meeting"

    def test_summary_mode_invalid_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            _run_parser(
                ["summarize", "https://y.com/watch?v=x", "--summary-mode", "novel"],
                monkeypatch,
            )

    def test_llm_provider_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["summarize", "https://y.com/watch?v=x", "--llm-provider", "openrouter"],
            monkeypatch,
        )
        assert ns.llm_provider == "openrouter"

    def test_llm_provider_invalid_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            _run_parser(
                ["summarize", "https://y.com/watch?v=x", "--llm-provider", "together"],
                monkeypatch,
            )

    def test_llm_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ns = _run_parser(
            ["summarize", "https://y.com/watch?v=x", "--llm-model", "anthropic/claude-4.7-sonnet"],
            monkeypatch,
        )
        assert ns.llm_model == "anthropic/claude-4.7-sonnet"
