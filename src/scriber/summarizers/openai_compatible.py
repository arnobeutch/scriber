"""Shared base for OpenAI-API-compatible backends (OpenAI, OpenRouter, ...)."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import openai
from openai import OpenAI

from scriber.logger import my_logger
from scriber.summarizers.markdown import format_summary_markdown

from .base import analyze_sentiment
from .modes import get_prompt, resolve_mode

if TYPE_CHECKING:
    from collections.abc import Iterable

    from openai.types.chat import ChatCompletionChunk

    from scriber.model import Transcript
    from scriber.settings import Settings


def _consume_stream(stream: Iterable[ChatCompletionChunk]) -> str:
    """Pull chunks from a streaming Chat Completions call, echo to stdout.

    Writes the raw token text directly to stdout as it arrives (stdout is
    the summary "loading bar") and accumulates the full content for the
    caller. A trailing newline is written after the stream ends.
    """
    pieces: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        pieces.append(delta)
        sys.stdout.write(delta)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(pieces)


class OpenAICompatibleSummarizer:
    """Base for backends that speak the OpenAI Chat Completions protocol.

    Subclasses set ``api_key`` and ``base_url`` instance attributes via
    ``__init__``; the ``OpenAI`` client is built once per ``summarize`` call
    using those attrs.
    """

    DEFAULT_SYSTEM_PROMPT = "You provide concise and insightful summaries."
    api_key: str | None = None
    base_url: str | None = None

    def __init__(self, settings: Settings) -> None:
        """Bind the Settings instance for later access during ``summarize``."""
        self.settings = settings

    def _build_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _model_name(self) -> str:
        """Return the model id sent in the API call (CLI/env overrides win)."""
        return self.settings.llm_model or self.settings.openai_model

    def summarize(
        self,
        transcript: Transcript,
        *,
        input_path: str,
        context: str | None = None,
    ) -> None:
        """Send the prompt to the API and write the resulting summary to disk."""
        from typing import cast

        from .modes import SummaryMode

        mode = resolve_mode(cast(SummaryMode, self.settings.summary_mode), transcript)
        prompt = get_prompt(mode, transcript.language, context) + transcript.text
        my_logger.info(f"Summary mode: {mode}")

        sentiment = analyze_sentiment(transcript.text)

        try:
            client = self._build_client()
            stream = client.chat.completions.create(
                model=self._model_name(),
                messages=[
                    {"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
            )
            content = _consume_stream(stream)
        except openai.AuthenticationError:
            my_logger.exception("AuthenticationError while performing API request")
            return
        except openai.APITimeoutError:
            my_logger.exception("Timeout while performing API request")
            return
        except openai.OpenAIError:
            my_logger.exception(
                "API error — is the relevant API key set in .env or the environment?",
            )
            return

        if not content:
            my_logger.error("LLM returned empty content")
            return

        markdown_output = format_summary_markdown(
            content,
            filename_stem=transcript.title,
            language=transcript.language,
            mode=mode,
            source_path=input_path,
            sentiment=sentiment,
            chapters=transcript.chapters,
        )
        suffix = "résumé" if transcript.language == "fr" else "summary"
        out_path = self.settings.output_dir / f"{transcript.title} - {suffix}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown_output, encoding="utf8")
        my_logger.info(f"Summary written to {out_path}")
