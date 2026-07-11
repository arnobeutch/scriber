"""Tests for the primer-suggestion extractor."""

from __future__ import annotations

from scriber.primer import (
    PrimerCandidates,
    extract_primer_candidates,
    format_primer_draft,
)


class TestExtractPrimerCandidates:
    def test_proper_nouns_and_acronyms(self) -> None:
        segments = [
            {"text": "Aujourd'hui, Wienerberger présente le projet."},
            {"text": "Ensuite, K-LINE et RAL ont rejoint Wienerberger."},
        ]
        c = extract_primer_candidates(segments)
        proper = dict(c.proper_nouns)
        assert proper.get("Wienerberger") == 2  # mid-sentence, counted both times
        assert "le" not in proper  # lowercase words ignored
        assert "projet" not in proper
        assert set(dict(c.acronyms)) >= {"K-LINE", "RAL"}

    def test_sentence_initial_capitalization_ignored(self) -> None:
        # A word that only ever starts sentences isn't an informative candidate.
        segments = [{"text": "Bonjour tout le monde. Merci beaucoup."}]
        assert extract_primer_candidates(segments).proper_nouns == []

    def test_low_confidence_words(self) -> None:
        segments = [
            {
                "text": "x",
                "words": [
                    {"word": " Wienerberger", "probability": 0.35},
                    {"word": " bonjour", "probability": 0.95},  # confident → ignored
                    {"word": " Culmi", "probability": 0.40},
                    {"word": " a", "probability": 0.10},  # too short → ignored
                    {"word": " merci", "probability": 0.20},  # stopword → ignored
                ],
            },
        ]
        c = extract_primer_candidates(segments)
        assert c.low_confidence == [("Wienerberger", 0.35), ("Culmi", 0.40)]

    def test_no_word_data_means_no_low_confidence(self) -> None:
        assert extract_primer_candidates([{"text": "On voit Wienerberger."}]).low_confidence == []


class TestFormatPrimerDraft:
    def test_active_terms_uncommented_lowconf_commented(self) -> None:
        c = PrimerCandidates(
            proper_nouns=[("Wienerberger", 2)],
            acronyms=[("RAL", 1)],
            low_confidence=[("Culmi", 0.40)],
        )
        out = format_primer_draft(c, "My Talk")
        lines = out.splitlines()
        assert "My Talk" in out  # title in header
        assert "Wienerberger" in lines  # proper-noun line is active (uncommented)
        assert "RAL" in lines  # acronym line is active
        assert "#   Culmi: 0.40" in lines  # low-confidence word is only a comment

    def test_draft_is_usable_after_comment_strip(self) -> None:
        # Mirrors load_text_file's behavior: dropping '#' lines leaves the vocabulary.
        c = PrimerCandidates(
            proper_nouns=[("Wienerberger", 1)],
            acronyms=[("RAL", 1)],
            low_confidence=[("Culmi", 0.40)],
        )
        out = format_primer_draft(c, "T")
        body = "\n".join(
            line for line in out.splitlines() if not line.lstrip().startswith("#")
        ).strip()
        assert "Wienerberger" in body
        assert "RAL" in body
        assert "Culmi" not in body  # the low-confidence comment was stripped

    def test_carried_primer_preserved_and_deduped(self) -> None:
        # A primer fed via --initial-prompt-file is carried into the new draft;
        # its terms survive and newly-harvested duplicates are dropped.
        c = PrimerCandidates(
            proper_nouns=[("Flashnet", 3), ("RecolX", 2)],
            acronyms=[("RAL", 1)],
            low_confidence=[],
        )
        carried = "Flashnet, Jacques Letzelter"
        out = format_primer_draft(c, "Call 2", carried_primer=carried)
        body = "\n".join(
            line for line in out.splitlines() if not line.lstrip().startswith("#")
        ).strip()
        assert "Flashnet, Jacques Letzelter" in body  # carried block preserved verbatim
        assert "RecolX" in body  # new term added
        assert "RAL" in body
        # Flashnet already carried → not repeated in the "New proper nouns" line.
        assert body.count("Flashnet") == 1

    def test_no_carried_primer_keeps_original_headings(self) -> None:
        c = PrimerCandidates(
            proper_nouns=[("Wienerberger", 1)],
            acronyms=[],
            low_confidence=[],
        )
        out = format_primer_draft(c, "T")
        assert "Carried over from previous primer" not in out
        assert "# --- Proper nouns / names (by frequency) ---" in out
