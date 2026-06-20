"""Tests for the summarizers package (Protocol + factory + each backend)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import openai
import pytest

from scriber.model import Transcript
from scriber.settings import Settings
from scriber.summarizers import MissingAPIKeyError, analyze_sentiment, make_summarizer
from scriber.summarizers.openai_summarizer import OpenAISummarizer
from scriber.summarizers.openrouter import OPENROUTER_BASE_URL, OpenRouterSummarizer
from scriber.summarizers.rag import RagSummarizer


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "log_level": "INFO",
        "openai_api_key": "sk-test",
        "openrouter_api_key": "or-test",
        "huggingface_token": None,
        "llm_provider": "openai",
        "llm_model": None,
        "openai_model": "gpt-4o",
        "ollama_model": "mistral",
        "whisper_model_size": "small",
        "wrap_width": 80,
        "summary_mode": "source",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _transcript(language: str = "en", title: str = "vid") -> Transcript:
    return Transcript(
        text="hello world",
        language=language,
        title=title,
        source="yt_manual",
        diarized=False,
    )


class TestAnalyzeSentiment:
    def test_positive(self) -> None:
        assert analyze_sentiment("This is wonderful, amazing, and excellent.") == "Positive"

    def test_negative(self) -> None:
        assert analyze_sentiment("This is terrible, horrible, awful.") == "Negative"

    def test_neutral(self) -> None:
        assert analyze_sentiment("The cat sat on the mat.") == "Neutral"


class TestMakeSummarizer:
    def test_returns_openai(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", llm_provider="openai")
        assert isinstance(make_summarizer(s), OpenAISummarizer)

    def test_returns_openrouter(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", llm_provider="openrouter")
        assert isinstance(make_summarizer(s), OpenRouterSummarizer)

    def test_returns_rag(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", llm_provider="ollama")
        assert isinstance(make_summarizer(s), RagSummarizer)

    def test_openai_missing_key_raises(self, tmp_path: Path) -> None:
        s = _settings(
            output_dir=tmp_path / "out",
            llm_provider="openai",
            openai_api_key=None,
        )
        with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
            make_summarizer(s)

    def test_openrouter_missing_key_raises(self, tmp_path: Path) -> None:
        s = _settings(
            output_dir=tmp_path / "out",
            llm_provider="openrouter",
            openrouter_api_key=None,
        )
        with pytest.raises(MissingAPIKeyError, match="OPENROUTER_API_KEY"):
            make_summarizer(s)

    def test_unknown_provider_raises(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", llm_provider="anthropic")
        with pytest.raises(ValueError, match="Unknown llm_provider"):
            make_summarizer(s)


class TestOpenAISummarizer:
    def _mock_client(self, content: str | None = "summary body") -> MagicMock:
        """Mock an OpenAI client that streams ``content`` chunk-by-chunk.

        ``content=None`` simulates a stream that yields no text (tests the
        \"empty content\" error path).
        """
        client = MagicMock()

        def make_chunks(text: str | None) -> list[MagicMock]:
            chunks: list[MagicMock] = []
            for piece in (text or "").splitlines(keepends=True) or [""]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = piece
                chunks.append(chunk)
            return chunks

        client.chat.completions.create.return_value = iter(make_chunks(content))
        return client

    def test_writes_summary_file(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        summarizer = OpenAISummarizer(s)
        # Structured source-mode content so sections survive the formatter.
        structured = (
            "Summary: the body of the summary\n"
            "Main claims:\n1. a\n2. b\n3. c\n"
            "Factually correct:\n- F1\n"
            "Likely but unconfirmed:\n- L1\n"
            "Interpretation or weakly substantiated:\n- I1\n"
            "Alternative interpretations:\n- A1\n"
            "Wrong or misleading:\n- None identified\n"
            "Keywords: foo, bar\n"
            "Tags: #ml #research\n"
        )
        with patch(
            "scriber.summarizers.openai_compatible.OpenAI",
            return_value=self._mock_client(content=structured),
        ):
            summarizer.summarize(_transcript(language="en", title="vid"), input_path="u")
        out = tmp_path / "out" / "vid - summary.md"
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "# Summary — vid" in text
        assert "## Summary" in text
        assert "the body of the summary" in text
        assert "## Sentiment" in text

    def test_french_uses_french_prompt(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", summary_mode="source")
        summarizer = OpenAISummarizer(s)
        client = self._mock_client(content="résumé")
        with patch("scriber.summarizers.openai_compatible.OpenAI", return_value=client):
            summarizer.summarize(_transcript(language="fr", title="vidfr"), input_path="u")
        _, kwargs = client.chat.completions.create.call_args
        prompt = kwargs["messages"][1]["content"]
        # FR source prompt uses the "Résumé :" / "Thèses principales :" headers.
        assert "Résumé :" in prompt
        assert "Thèses principales :" in prompt

    def test_meeting_mode_uses_meeting_prompt(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", summary_mode="meeting")
        summarizer = OpenAISummarizer(s)
        client = self._mock_client()
        with patch("scriber.summarizers.openai_compatible.OpenAI", return_value=client):
            summarizer.summarize(_transcript(language="en"), input_path="u")
        _, kwargs = client.chat.completions.create.call_args
        prompt = kwargs["messages"][1]["content"]
        # EN meeting prompt uses "Topic:" header.
        assert "Topic:" in prompt

    def test_unsupported_language_raises(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        summarizer = OpenAISummarizer(s)
        with (
            patch("scriber.summarizers.openai_compatible.OpenAI", return_value=self._mock_client()),
            pytest.raises(ValueError, match="language not supported"),
        ):
            summarizer.summarize(_transcript(language="de"), input_path="u")

    def test_openai_error_swallowed(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        summarizer = OpenAISummarizer(s)
        client = MagicMock()
        client.chat.completions.create.side_effect = openai.OpenAIError("boom")
        with patch("scriber.summarizers.openai_compatible.OpenAI", return_value=client):
            summarizer.summarize(_transcript(), input_path="u")
        assert not list((tmp_path / "out").glob("*.md")) if (tmp_path / "out").exists() else True

    def test_empty_content_does_not_write(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out")
        summarizer = OpenAISummarizer(s)
        with patch(
            "scriber.summarizers.openai_compatible.OpenAI",
            return_value=self._mock_client(content=None),
        ):
            summarizer.summarize(_transcript(), input_path="u")
        assert not list((tmp_path / "out").glob("*.md")) if (tmp_path / "out").exists() else True

    def test_uses_settings_openai_model(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", openai_model="gpt-5")
        client = self._mock_client()
        with patch("scriber.summarizers.openai_compatible.OpenAI", return_value=client):
            OpenAISummarizer(s).summarize(_transcript(), input_path="u")
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-5"

    def test_llm_model_overrides_openai_model(self, tmp_path: Path) -> None:
        s = _settings(
            output_dir=tmp_path / "out",
            openai_model="gpt-4o",
            llm_model="gpt-5",
        )
        client = self._mock_client()
        with patch("scriber.summarizers.openai_compatible.OpenAI", return_value=client):
            OpenAISummarizer(s).summarize(_transcript(), input_path="u")
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-5"


class TestOpenRouterSummarizer:
    def test_uses_openrouter_base_url(self, tmp_path: Path) -> None:
        s = _settings(
            output_dir=tmp_path / "out",
            llm_provider="openrouter",
            openrouter_api_key="or-test",
        )
        client = MagicMock()
        response = MagicMock()
        response.choices[0].message.content = "x"
        client.chat.completions.create.return_value = response
        with patch("scriber.summarizers.openai_compatible.OpenAI", return_value=client) as ctor:
            OpenRouterSummarizer(s).summarize(_transcript(), input_path="u")
        _, kwargs = ctor.call_args
        assert kwargs["api_key"] == "or-test"
        assert kwargs["base_url"] == OPENROUTER_BASE_URL

    def test_default_model_is_provider_prefixed(self, tmp_path: Path) -> None:
        s = _settings(
            output_dir=tmp_path / "out",
            llm_provider="openrouter",
            openrouter_api_key="or-test",
            llm_model=None,
        )
        client = MagicMock()
        response = MagicMock()
        response.choices[0].message.content = "x"
        client.chat.completions.create.return_value = response
        with patch("scriber.summarizers.openai_compatible.OpenAI", return_value=client):
            OpenRouterSummarizer(s).summarize(_transcript(), input_path="u")
        _, kwargs = client.chat.completions.create.call_args
        assert "/" in kwargs["model"]  # provider/model format

    def test_llm_model_used_when_set(self, tmp_path: Path) -> None:
        s = _settings(
            output_dir=tmp_path / "out",
            llm_provider="openrouter",
            openrouter_api_key="or-test",
            llm_model="anthropic/claude-4.7-sonnet",
        )
        client = MagicMock()
        response = MagicMock()
        response.choices[0].message.content = "x"
        client.chat.completions.create.return_value = response
        with patch("scriber.summarizers.openai_compatible.OpenAI", return_value=client):
            OpenRouterSummarizer(s).summarize(_transcript(), input_path="u")
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["model"] == "anthropic/claude-4.7-sonnet"


class TestRagSummarizer:
    def test_writes_structured_markdown_with_sentiment(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", llm_provider="ollama", summary_mode="meeting")
        with patch(
            "scriber.summarizers.rag.generate_summary",
            return_value="Sujet: test\nHashtags: #t\n",
        ):
            out_path = RagSummarizer(s).summarize(
                Transcript(
                    text="Alice: hi",
                    language="fr",
                    title="vid1",
                    source="yt_manual",
                    diarized=False,
                ),
                input_path="u",
            )
        assert out_path.exists()
        body = out_path.read_text(encoding="utf-8")
        assert "# Résumé de la réunion — vid1" in body
        # Sentiment-everywhere: RAG output includes a sentiment section.
        assert "## Sentiment" in body

    def test_english_suffix(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", llm_provider="ollama")
        with patch("scriber.summarizers.rag.generate_summary", return_value="Sujet: x\n"):
            out_path = RagSummarizer(s).summarize(
                Transcript(
                    text="A: x",
                    language="en",
                    title="vid2",
                    source="yt_manual",
                    diarized=False,
                ),
                input_path="u",
            )
        assert out_path.name == "vid2 - summary.md"

    def test_uses_settings_ollama_model(self, tmp_path: Path) -> None:
        s = _settings(
            output_dir=tmp_path / "out",
            llm_provider="ollama",
            ollama_model="gemma4:e4b",
        )
        with patch("scriber.summarizers.rag.generate_summary", return_value="x") as gen:
            RagSummarizer(s).summarize(
                Transcript(
                    text="x",
                    language="en",
                    title="t",
                    source="file",
                    diarized=False,
                ),
                input_path="u",
            )
        _, kwargs = gen.call_args
        assert kwargs["model"] == "gemma4:e4b"

    def test_llm_model_overrides_ollama_model(self, tmp_path: Path) -> None:
        s = _settings(
            output_dir=tmp_path / "out",
            llm_provider="ollama",
            ollama_model="mistral",
            llm_model="gemma4:e4b",
        )
        with patch("scriber.summarizers.rag.generate_summary", return_value="x") as gen:
            RagSummarizer(s).summarize(
                Transcript(
                    text="x",
                    language="en",
                    title="t",
                    source="file",
                    diarized=False,
                ),
                input_path="u",
            )
        _, kwargs = gen.call_args
        assert kwargs["model"] == "gemma4:e4b"

    def test_generate_summary_error_propagates(self, tmp_path: Path) -> None:
        s = _settings(output_dir=tmp_path / "out", llm_provider="ollama")
        with (
            patch("scriber.summarizers.rag.generate_summary", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            RagSummarizer(s).summarize(
                Transcript(
                    text="x: y",
                    language="en",
                    title="v",
                    source="file",
                    diarized=False,
                ),
                input_path="u",
            )
