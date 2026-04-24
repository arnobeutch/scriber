"""Tests for markdown_writer.

``extract_sections`` and ``format_summary_markdown`` both require a
``mode`` ("meeting" | "source") alongside ``language``. The regex in
``extract_sections`` matches the mode+language-specific labels the prompt
asks the model to emit.
"""

from scriber.summarizers.markdown import (
    clean_section,
    extract_sections,
    format_summary_markdown,
)

FR_MEETING_SAMPLE = """
Sujet: Lancement produit
Hashtags: #produit #client
Principaux enseignements: Ça avance bien.
Questions / Réponses: Rien.
Décisions: Lancement en juillet.
Actions à suivre: Alice écrit la spec.
"""

EN_MEETING_SAMPLE = """
Topic: Product launch
Hashtags: #product #client
Main takeaways: Going well.
Questions / Answers: None.
Decisions: Launch in July.
Action items: Alice writes the spec.
"""

EN_SOURCE_SAMPLE = """
TL;DR: The author argues that X.
Key takeaways:
- One
Facts:
- F1 — supported by the paper
Opinions:
- O1 — author opines
Speculation / unverified:
- S1 — no evidence
Counterpoints / alternatives:
- C1 — alternative view
Information quality / reliability: Overall well-sourced.
"""


class TestExtractSections:
    def test_extracts_all_meeting_sections_fr(self) -> None:
        result = extract_sections(FR_MEETING_SAMPLE, "meeting", "fr")
        assert "Lancement produit" in result["topic"]
        assert "#produit" in result["hashtags"]
        assert "avance bien" in result["takeaways"]
        assert "Rien" in result["qa"]
        assert "Lancement en juillet" in result["decisions"]
        assert "Alice écrit" in result["actions"]

    def test_extracts_all_meeting_sections_en(self) -> None:
        result = extract_sections(EN_MEETING_SAMPLE, "meeting", "en")
        assert "Product launch" in result["topic"]
        assert "#product" in result["hashtags"]
        assert "Going well" in result["takeaways"]
        assert "Launch in July" in result["decisions"]
        assert "Alice writes" in result["actions"]

    def test_extracts_all_source_sections_en(self) -> None:
        result = extract_sections(EN_SOURCE_SAMPLE, "source", "en")
        assert "author argues" in result["tldr"]
        assert "One" in result["takeaways"]
        assert "F1" in result["facts"]
        assert "O1" in result["opinions"]
        assert "S1" in result["speculation"]
        assert "C1" in result["counterpoints"]
        assert "well-sourced" in result["reliability"]

    def test_language_mismatch_yields_only_shared_labels(self) -> None:
        # Only ``Hashtags`` is spelled identically in FR and EN; everything else
        # falls out on a language mismatch.
        fr_as_en = extract_sections(FR_MEETING_SAMPLE, "meeting", "en")
        en_as_fr = extract_sections(EN_MEETING_SAMPLE, "meeting", "fr")
        assert set(fr_as_en.keys()) <= {"hashtags"}
        assert set(en_as_fr.keys()) <= {"hashtags"}
        assert "topic" not in fr_as_en
        assert "decisions" not in en_as_fr

    def test_missing_section_absent_from_dict(self) -> None:
        partial = "Sujet: Seul sujet ici.\nHashtags: #x\n"
        result = extract_sections(partial, "meeting", "fr")
        assert "Seul sujet" in result.get("topic", "")
        assert "#x" in result.get("hashtags", "")
        assert "decisions" not in result

    def test_empty_summary(self) -> None:
        assert extract_sections("", "meeting", "fr") == {}


class TestCleanSection:
    def test_empty_returns_default_en(self) -> None:
        assert clean_section("", "en") == "None"

    def test_empty_returns_default_fr(self) -> None:
        assert clean_section("", "fr") == "Aucune"

    def test_none_literal(self) -> None:
        assert clean_section("none", "en") == "None"
        assert clean_section("NONE", "en") == "None"

    def test_aucune_literal(self) -> None:
        assert clean_section("aucune", "fr") == "Aucune"

    def test_na_literal(self) -> None:
        assert clean_section("n/a", "en") == "None"
        assert clean_section("N/A", "fr") == "Aucune"

    def test_normal_text_preserved(self) -> None:
        assert clean_section("hello world", "en") == "hello world"

    def test_strips_surrounding_whitespace(self) -> None:
        assert clean_section("  hello  ", "en") == "hello"


class TestFormatSummaryMarkdown:
    def test_meeting_en_output(self) -> None:
        out = format_summary_markdown(EN_MEETING_SAMPLE, "video-1", "en", "meeting")
        assert out.startswith("# Meeting Summary — video-1")
        assert "## Meeting Topic" in out
        assert "Product launch" in out
        assert "## Decisions" in out
        assert "## Action Items" in out

    def test_meeting_fr_output(self) -> None:
        out = format_summary_markdown(FR_MEETING_SAMPLE, "video-1", "fr", "meeting")
        assert out.startswith("# Résumé de la réunion — video-1")
        assert "## Sujet de la réunion" in out
        assert "## Principaux enseignements" in out
        assert "Lancement produit" in out

    def test_source_en_output(self) -> None:
        out = format_summary_markdown(EN_SOURCE_SAMPLE, "vid", "en", "source")
        assert out.startswith("# Summary — vid")
        assert "## TL;DR" in out
        assert "## Facts" in out
        assert "## Counterpoints & Alternatives" in out
        assert "## Information Quality" in out

    def test_source_fr_title(self) -> None:
        out = format_summary_markdown("", "vid", "fr", "source")
        assert out.startswith("# Résumé — vid")

    def test_missing_sections_get_defaults(self) -> None:
        out = format_summary_markdown("", "x", "en", "meeting")
        assert out.count("None") >= 6

    def test_missing_sections_get_fr_defaults(self) -> None:
        out = format_summary_markdown("", "x", "fr", "meeting")
        assert out.count("Aucune") >= 6

    def test_source_path_rendered_when_provided(self) -> None:
        out = format_summary_markdown(
            EN_MEETING_SAMPLE,
            "v",
            "en",
            "meeting",
            source_path="https://youtube.com/watch?v=abc",
        )
        assert "> Source: https://youtube.com/watch?v=abc" in out

    def test_source_path_omitted_when_none(self) -> None:
        out = format_summary_markdown(EN_MEETING_SAMPLE, "v", "en", "meeting")
        assert "Source:" not in out

    def test_sentiment_rendered_when_provided(self) -> None:
        out = format_summary_markdown(
            EN_MEETING_SAMPLE,
            "v",
            "en",
            "meeting",
            sentiment="Positive",
        )
        assert "## Sentiment" in out
        assert "Positive" in out

    def test_sentiment_omitted_when_none(self) -> None:
        out = format_summary_markdown(EN_MEETING_SAMPLE, "v", "en", "meeting")
        assert "## Sentiment" not in out
