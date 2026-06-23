"""Environment-based settings. Stdlib only — no pydantic dep.

Extend ``Settings`` with your own fields and map them in ``from_env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_WHISPER_MODEL_SIZE = "large-v3-turbo"
_DEFAULT_OPENAI_MODEL = "gpt-4o"
_DEFAULT_OLLAMA_MODEL = "mistral"
_DEFAULT_LLM_PROVIDER = "openai"
_DEFAULT_WRAP_WIDTH = 80
_DEFAULT_SUMMARY_MODE = "auto"
_DEFAULT_PREPROCESS_AUDIO = True


def _bool_from_env(value: str | None, *, default: bool) -> bool:
    """Parse common truthy/falsy strings; fall back to ``default``."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _int_from_env(value: str | None) -> int | None:
    """Parse an int env var; return None when unset/blank/non-numeric."""
    if value is None or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


def load_text_file(path: Path | None) -> str | None:
    """Read a small text file, strip whitespace, return None when missing/empty.

    Full-line ``#`` comments are dropped, so an auto-generated primer draft
    (``--suggest-primer``, with ``#`` review notes) is usable as-is when fed
    back via ``--initial-prompt-file`` — no need to strip the annotations by
    hand. Shared by ``INITIAL_PROMPT_FILE`` env loading and the CLI /
    interactive paths in ``main``; callers only ever see the resolved string.
    """
    if path is None:
        return None
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8")
    body = "\n".join(line for line in raw.splitlines() if not line.lstrip().startswith("#"))
    return body.strip() or None


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Populate os.environ from a .env file (KEY=value per line). Existing vars win."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    """Typed snapshot of process-wide settings.

    Fields with a leading underscore in the env-var name (``_``) are not
    auto-loaded; everything else comes from ``os.environ`` (with ``.env``
    seeding it first).
    """

    log_level: str = "INFO"
    openai_api_key: str | None = None
    openrouter_api_key: str | None = None
    huggingface_token: str | None = None
    llm_provider: str = _DEFAULT_LLM_PROVIDER
    llm_model: str | None = None  # None → provider picks a sensible default
    openai_model: str = _DEFAULT_OPENAI_MODEL
    ollama_model: str = _DEFAULT_OLLAMA_MODEL
    whisper_model_size: str = _DEFAULT_WHISPER_MODEL_SIZE
    output_dir: Path = field(default_factory=lambda: Path("results"))
    downloads_dir: Path = field(default_factory=lambda: Path("downloads"))
    wrap_width: int = _DEFAULT_WRAP_WIDTH
    summary_mode: str = _DEFAULT_SUMMARY_MODE
    preprocess_audio: bool = _DEFAULT_PREPROCESS_AUDIO
    min_speakers: int | None = None  # diarization hint; None lets pyannote decide
    max_speakers: int | None = None
    initial_prompt: str | None = None
    """Resolved primer text (vocabulary hint for whisper). Loaded from the
    ``INITIAL_PROMPT_FILE`` env var, the ``--initial-prompt-file`` CLI flag,
    or an interactive prompt — all three reduce to the loaded *content*,
    not the path. See docs/WHISPER_SETUP.md for what to put in it."""

    @classmethod
    def from_env(cls) -> Settings:
        """Load ``.env`` (if present) then read settings from ``os.environ``."""
        _load_dotenv()
        return cls(
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY") or None,
            huggingface_token=os.environ.get("HUGGINGFACE_TOKEN") or None,
            llm_provider=os.environ.get("LLM_PROVIDER", _DEFAULT_LLM_PROVIDER),
            llm_model=os.environ.get("LLM_MODEL") or None,
            openai_model=os.environ.get("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL),
            ollama_model=os.environ.get("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL),
            whisper_model_size=os.environ.get(
                "WHISPER_MODEL_SIZE",
                _DEFAULT_WHISPER_MODEL_SIZE,
            ),
            output_dir=Path(os.environ.get("OUTPUT_DIR", "results")),
            downloads_dir=Path(os.environ.get("DOWNLOADS_DIR", "downloads")),
            wrap_width=int(os.environ.get("WRAP_WIDTH", str(_DEFAULT_WRAP_WIDTH))),
            summary_mode=os.environ.get("SUMMARY_MODE", _DEFAULT_SUMMARY_MODE),
            preprocess_audio=_bool_from_env(
                os.environ.get("PREPROCESS_AUDIO"),
                default=_DEFAULT_PREPROCESS_AUDIO,
            ),
            min_speakers=_int_from_env(os.environ.get("MIN_SPEAKERS")),
            max_speakers=_int_from_env(os.environ.get("MAX_SPEAKERS")),
            initial_prompt=load_text_file(
                Path(p) if (p := os.environ.get("INITIAL_PROMPT_FILE")) else None,
            ),
        )
