"""Tests for my_settings."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import scriber.settings as my_settings
from scriber.settings import Settings, load_text_file


class TestLoadTextFile:
    def test_none_path_returns_none(self) -> None:
        assert load_text_file(None) is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_text_file(tmp_path / "nope.txt") is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        assert load_text_file(p) is None

    def test_whitespace_only_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "ws.txt"
        p.write_text("   \n\n\t  ", encoding="utf-8")
        assert load_text_file(p) is None

    def test_strips_surrounding_whitespace(self, tmp_path: Path) -> None:
        p = tmp_path / "ok.txt"
        p.write_text("\n\n  hello world  \n", encoding="utf-8")
        assert load_text_file(p) == "hello world"

    def test_utf8_unicode_preserved(self, tmp_path: Path) -> None:
        p = tmp_path / "fr.txt"
        p.write_text("Réunion : K•LINE, Wienerberger.", encoding="utf-8")
        assert load_text_file(p) == "Réunion : K•LINE, Wienerberger."

    def test_strips_hash_comment_lines(self, tmp_path: Path) -> None:
        # A primer draft (--suggest-primer) keeps #-prefixed notes inline; those
        # must be dropped so the file is usable as-is via --initial-prompt-file.
        p = tmp_path / "primer.draft.txt"
        p.write_text(
            "# Auto-suggested primer\nWienerberger, RAL\n#   Culmi: 0.40\n",
            encoding="utf-8",
        )
        assert load_text_file(p) == "Wienerberger, RAL"


class TestLoadDotenv:
    def test_missing_file_is_noop(self, tmp_path: Path) -> None:
        # must not raise
        my_settings._load_dotenv(tmp_path / ".env")

    def test_basic_key_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env = tmp_path / ".env"
        env.write_text("FOO=bar\nBAZ=qux\n")
        monkeypatch.delenv("FOO", raising=False)
        monkeypatch.delenv("BAZ", raising=False)
        my_settings._load_dotenv(env)
        assert os.environ["FOO"] == "bar"
        assert os.environ["BAZ"] == "qux"

    def test_comments_and_blank_lines_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = tmp_path / ".env"
        env.write_text("# comment\n\n  \nKEY=value\n")
        monkeypatch.delenv("KEY", raising=False)
        my_settings._load_dotenv(env)
        assert os.environ["KEY"] == "value"

    def test_strip_quotes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env = tmp_path / ".env"
        env.write_text("Q1=\"quoted\"\nQ2='single'\n")
        monkeypatch.delenv("Q1", raising=False)
        monkeypatch.delenv("Q2", raising=False)
        my_settings._load_dotenv(env)
        assert os.environ["Q1"] == "quoted"
        assert os.environ["Q2"] == "single"

    def test_line_without_equals_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = tmp_path / ".env"
        env.write_text("NOEQUALSLINE\nGOOD=yes\n")
        monkeypatch.delenv("GOOD", raising=False)
        monkeypatch.delenv("NOEQUALSLINE", raising=False)
        my_settings._load_dotenv(env)
        assert os.environ["GOOD"] == "yes"
        assert "NOEQUALSLINE" not in os.environ

    def test_existing_env_wins(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env = tmp_path / ".env"
        env.write_text("PREEXISTING=from_file\n")
        monkeypatch.setenv("PREEXISTING", "from_env")
        my_settings._load_dotenv(env)
        assert os.environ["PREEXISTING"] == "from_env"

    def test_value_with_internal_equals(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = tmp_path / ".env"
        env.write_text("URL=https://x.com/?a=1&b=2\n")
        monkeypatch.delenv("URL", raising=False)
        my_settings._load_dotenv(env)
        assert os.environ["URL"] == "https://x.com/?a=1&b=2"


_ALL_ENV_KEYS = (
    "LOG_LEVEL",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "HUGGINGFACE_TOKEN",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "OPENAI_MODEL",
    "OLLAMA_MODEL",
    "WHISPER_MODEL_SIZE",
    "OUTPUT_DIR",
    "DOWNLOADS_DIR",
    "WRAP_WIDTH",
    "SUMMARY_MODE",
    "MIN_SPEAKERS",
    "MAX_SPEAKERS",
)


def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # no .env in tmp_path
    for key in _ALL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


class TestSettingsFromEnv:
    def test_default_log_level(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch, tmp_path)
        s = Settings.from_env()
        assert s.log_level == "INFO"

    def test_respects_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch, tmp_path)
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        s = Settings.from_env()
        assert s.log_level == "DEBUG"

    def test_loads_from_dotenv(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _clean_env(monkeypatch, tmp_path)
        (tmp_path / ".env").write_text("LOG_LEVEL=WARNING\n")
        s = Settings.from_env()
        assert s.log_level == "WARNING"

    def test_full_config_defaults(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _clean_env(monkeypatch, tmp_path)
        s = Settings.from_env()
        assert s.openai_api_key is None
        assert s.openrouter_api_key is None
        assert s.huggingface_token is None
        assert s.llm_provider == "openai"
        assert s.llm_model is None
        assert s.openai_model == "gpt-4o"
        assert s.ollama_model == "mistral"
        assert s.whisper_model_size == "large-v3-turbo"
        assert s.output_dir == Path("results")
        assert s.downloads_dir == Path("downloads")
        assert s.wrap_width == 80
        assert s.summary_mode == "auto"
        assert s.preprocess_audio is True  # default: on
        assert s.initial_prompt is None  # no env var, no .env, no CLI
        assert s.min_speakers is None
        assert s.max_speakers is None

    def test_full_config_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _clean_env(monkeypatch, tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf-test")
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("LLM_MODEL", "anthropic/claude-4.7-sonnet")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:e4b")
        monkeypatch.setenv("WHISPER_MODEL_SIZE", "medium")
        monkeypatch.setenv("OUTPUT_DIR", "out")
        monkeypatch.setenv("DOWNLOADS_DIR", "dl")
        monkeypatch.setenv("WRAP_WIDTH", "100")
        monkeypatch.setenv("SUMMARY_MODE", "meeting")
        monkeypatch.setenv("PREPROCESS_AUDIO", "false")
        monkeypatch.setenv("MIN_SPEAKERS", "2")
        monkeypatch.setenv("MAX_SPEAKERS", "5")
        primer = tmp_path / "primer.txt"
        primer.write_text("Christophe, Anne, K•LINE, RAL 7021.", encoding="utf-8")
        monkeypatch.setenv("INITIAL_PROMPT_FILE", str(primer))
        s = Settings.from_env()
        assert s.openai_api_key == "sk-test"
        assert s.openrouter_api_key == "or-test"
        assert s.huggingface_token == "hf-test"
        assert s.llm_provider == "openrouter"
        assert s.llm_model == "anthropic/claude-4.7-sonnet"
        assert s.openai_model == "gpt-4o-mini"
        assert s.ollama_model == "gemma4:e4b"
        assert s.whisper_model_size == "medium"
        assert s.output_dir == Path("out")
        assert s.downloads_dir == Path("dl")
        assert s.wrap_width == 100
        assert s.summary_mode == "meeting"
        assert s.preprocess_audio is False  # env override took effect
        assert s.initial_prompt == "Christophe, Anne, K•LINE, RAL 7021."
        assert s.min_speakers == 2
        assert s.max_speakers == 5

    def test_empty_string_env_treated_as_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _clean_env(monkeypatch, tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "")
        s = Settings.from_env()
        assert s.openai_api_key is None

    def test_blank_and_invalid_speaker_counts_are_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _clean_env(monkeypatch, tmp_path)
        monkeypatch.setenv("MIN_SPEAKERS", "   ")
        monkeypatch.setenv("MAX_SPEAKERS", "notanumber")
        s = Settings.from_env()
        assert s.min_speakers is None
        assert s.max_speakers is None
