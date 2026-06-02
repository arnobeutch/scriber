"""Guard: the transcription-only base must import without the optional extras.

This pins the central promise of the packaging split — a base install (no
``scriber[diarize]``, no ``scriber[summarize]``) can still run the transcribe
path. It simulates that environment by blocking the optional-extra packages at
import time, so the test fails the moment someone re-introduces an eager import
of pyannote / torchaudio / langchain / openai / chromadb / textblob on a path
the base install exercises.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

# Top-level import names provided only by the diarize / summarize extras.
_BLOCKED = frozenset(
    {
        "pyannote",
        "torchaudio",
        "chromadb",
        "langchain",
        "langchain_community",
        "langchain_chroma",
        "langchain_ollama",
        "openai",
        "textblob",
    },
)


class _BlockExtrasFinder(MetaPathFinder):
    """Meta-path finder that makes the optional-extra packages look uninstalled."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname.split(".", 1)[0] in _BLOCKED:
            msg = f"simulated base install: {fullname} is not installed"
            raise ImportError(msg)
        return None  # decline; let the real finders resolve everything else


@contextmanager
def _simulate_base_install() -> Iterator[None]:
    """Block the optional-extra deps and purge cached scriber + extra modules."""
    saved = dict(sys.modules)
    for name in list(sys.modules):
        top = name.split(".", 1)[0]
        if name.startswith("scriber") or top in _BLOCKED:
            del sys.modules[name]

    finder = _BlockExtrasFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.clear()
        sys.modules.update(saved)


def test_transcribe_path_imports_without_optional_extras() -> None:
    with _simulate_base_install():
        importlib.import_module("scriber.main")
        importlib.import_module("scriber.handlers")
        importlib.import_module("scriber.parser")
        importlib.import_module("scriber.transcription.local")
        importlib.import_module("scriber.transcription.youtube_audio")
        importlib.import_module("scriber.transcription.youtube_captions")


def test_diarize_module_is_gated_behind_its_extra() -> None:
    with _simulate_base_install(), pytest.raises(ImportError):
        importlib.import_module("scriber.transcription.diarize")


def test_summarizers_package_is_gated_behind_its_extra() -> None:
    with _simulate_base_install(), pytest.raises(ImportError):
        importlib.import_module("scriber.summarizers")
