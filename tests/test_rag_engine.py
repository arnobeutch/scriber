"""Unit tests for scriber.summarizers.engine."""

from __future__ import annotations

from scriber.summarizers.engine import pack_utterances


def _utt(speaker: str, text: str) -> tuple[str, str]:
    return (speaker, text)


class TestPackUtterances:
    def test_empty_input_returns_empty(self) -> None:
        assert pack_utterances([]) == []

    def test_short_utterances_merge_into_single_chunk(self) -> None:
        # Three short lines — total far below chunk_size — must pack into one doc.
        utts = [_utt("A", "hi"), _utt("B", "hello"), _utt("A", "ok")]
        chunks = pack_utterances(utts, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert "A : hi" in chunks[0]
        assert "B : hello" in chunks[0]
        assert "A : ok" in chunks[0]

    def test_packs_until_chunk_size(self) -> None:
        # chunk_size=50 → each "A : xxxxxxxxxxxxxxx" (~20 chars) fits 2-ish per chunk.
        utts = [_utt("A", "x" * 15) for _ in range(6)]
        chunks = pack_utterances(utts, chunk_size=50, overlap=0)
        # Must be more than one chunk now (the bug yielded one per utterance).
        assert len(chunks) >= 2
        # And fewer than one-per-utterance (otherwise we reproduced the bug).
        assert len(chunks) < len(utts)

    def test_single_oversized_utterance_is_its_own_chunk(self) -> None:
        # A 2000-char utterance should land in a dedicated chunk, not be dropped.
        utts = [_utt("A", "x" * 2000)]
        chunks = pack_utterances(utts, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert len(chunks[0]) >= 2000

    def test_overlap_prepends_previous_tail(self) -> None:
        # chunk_size=30 forces a split; overlap=20 should echo the last line.
        utts = [
            _utt("A", "aaaaaaaa"),  # "A : aaaaaaaa" = 12 chars
            _utt("B", "bbbbbbbb"),  # 12 chars
            _utt("C", "cccccccc"),  # 12 chars
        ]
        chunks = pack_utterances(utts, chunk_size=30, overlap=15)
        assert len(chunks) >= 2
        # Second chunk must contain the tail of the first for retrieval continuity.
        assert "B : bbbbbbbb" in chunks[1] or "A : aaaaaaaa" in chunks[1]

    def test_preserves_speaker_prefix_format(self) -> None:
        utts = [_utt("Alice", "hello world")]
        chunks = pack_utterances(utts)
        assert chunks == ["Alice : hello world"]
