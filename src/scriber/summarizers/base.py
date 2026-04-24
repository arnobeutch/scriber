"""Summarizer Protocol + factory + sentiment helper."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from textblob import TextBlob

from scriber import constants as my_constants

if TYPE_CHECKING:
    from scriber.model import Transcript
    from scriber.settings import Settings


class MissingAPIKeyError(Exception):
    """Raised by ``make_summarizer`` when the chosen backend lacks credentials."""


class Summarizer(Protocol):
    """Common interface implemented by every summarization backend."""

    def summarize(
        self,
        transcript: Transcript,
        *,
        input_path: str,
        context: str | None = None,
    ) -> Path | None:
        """Produce a markdown summary on disk; return its path when known.

        ``context`` carries optional auxiliary material (glossary, briefing
        notes) injected into the prompt when non-empty.
        """


def analyze_sentiment(text: str) -> str:
    """Classify text as Positive / Neutral / Negative via TextBlob polarity."""
    # textblob's `.sentiment` is a cached_property with partially-unknown typing; cast to skip.
    blob = cast(Any, TextBlob(text))
    polarity: float = blob.sentiment.polarity
    if polarity > my_constants.POLARITY_POSITIVE_THRESHOLD:
        return "Positive"
    if polarity < my_constants.POLARITY_NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def make_summarizer(settings: Settings) -> Summarizer:
    """Build a Summarizer for the chosen ``settings.llm_provider``.

    Raises ``MissingAPIKeyError`` when the provider needs an API key that
    isn't set — call this *before* the slow transcription pipeline so the
    user fails fast.
    """
    provider = settings.llm_provider
    if provider == "openai":
        if not settings.openai_api_key:
            err_msg = (
                "OPENAI_API_KEY is required for --llm-provider openai. "
                "Set it in .env or your shell environment."
            )
            raise MissingAPIKeyError(err_msg)
        from .openai_summarizer import OpenAISummarizer

        return OpenAISummarizer(settings)
    if provider == "openrouter":
        if not settings.openrouter_api_key:
            err_msg = (
                "OPENROUTER_API_KEY is required for --llm-provider openrouter. "
                "Set it in .env or your shell environment."
            )
            raise MissingAPIKeyError(err_msg)
        from .openrouter import OpenRouterSummarizer

        return OpenRouterSummarizer(settings)
    if provider == "ollama":
        from .rag import RagSummarizer

        return RagSummarizer(settings)
    err_msg = (
        f"Unknown llm_provider {provider!r}; expected one of 'openai', 'openrouter', 'ollama'."
    )
    raise ValueError(err_msg)
