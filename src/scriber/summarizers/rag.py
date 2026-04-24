"""Local RAG summarizer (langchain + Ollama + ChromaDB)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from scriber.logger import my_logger
from scriber.summarizers.engine import generate_summary
from scriber.summarizers.markdown import format_summary_markdown
from scriber.transcription.preprocess import parse_transcript, try_resolve_speaker_names

from .base import analyze_sentiment
from .modes import SummaryMode, get_prompt, resolve_mode

if TYPE_CHECKING:
    from scriber.model import Transcript
    from scriber.settings import Settings


class RagSummarizer:
    """Local RAG-based summarizer.

    Always uses Ollama; the model id comes from ``settings.llm_model`` if
    set, otherwise ``settings.ollama_model``. Writes a structured-section
    markdown file (via ``format_summary_markdown``).
    """

    def __init__(self, settings: Settings) -> None:
        """Bind the Settings instance for later access during ``summarize``."""
        self.settings = settings

    def _model_name(self) -> str:
        return self.settings.llm_model or self.settings.ollama_model

    def summarize(
        self,
        transcript: Transcript,
        *,
        input_path: str,
        context: str | None = None,
    ) -> Path:
        """Generate a markdown summary on disk for the given transcript."""
        mode = resolve_mode(cast(SummaryMode, self.settings.summary_mode), transcript)
        prompt = get_prompt(mode, transcript.language, context)
        my_logger.info(f"Summary mode: {mode}")

        my_logger.info("Parsing transcript...")
        utterances = parse_transcript(transcript.text)
        utterances = try_resolve_speaker_names(utterances)

        my_logger.info("Generating summary via RAG...")
        try:
            raw_summary = generate_summary(
                utterances,
                model=self._model_name(),
                prompt=prompt,
            )
        except Exception:
            my_logger.exception("Error generating summary")
            raise

        sentiment = analyze_sentiment(transcript.text)

        my_logger.info("Formatting markdown...")
        formatted = format_summary_markdown(
            raw_summary,
            transcript,
            mode,
            source_path=input_path,
            sentiment=sentiment,
        )

        suffix = "résumé" if transcript.language == "fr" else "summary"
        out_path = self.settings.output_dir / f"{transcript.title} - {suffix}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(formatted, encoding="utf-8")
        my_logger.info(f"Summary written to {out_path}")
        return out_path
