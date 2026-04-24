"""Tests for markdown_writer.

``extract_sections`` returns neutral keys (``topic`` / ``hashtags`` / ...).
The regex matches the language-specific labels the prompt asks the model
to emit (``Sujet`` in FR, ``Topic`` in EN), so parsing an FR sample with
``language="en"`` yields no matches and vice versa.
"""

from scriber.summarizers.markdown import (
    clean_section,
    extract_sections,
    format_summary_markdown,
    simple_format_markdown,
)

FR_SAMPLE = """
Sujet: Lancement produit
Hashtags: #produit #client
Principaux enseignements: Ça avance bien.
Questions / Réponses: Rien.
Décisions: Lancement en juillet.
Actions à suivre: Alice écrit la spec.
"""

EN_SAMPLE = """
Topic: Product launch
Hashtags: #product #client
Main takeaways: Going well.
Questions / Answers: None.
Decisions: Launch in July.
Action items: Alice writes the spec.
"""


class TestExtractSections:
    def test_extracts_all_sections_fr(self) -> None:
        result = extract_sections(FR_SAMPLE, "fr")
        assert "Lancement produit" in result["topic"]
        assert "#produit" in result["hashtags"]
        assert "avance bien" in result["takeaways"]
        assert "Rien" in result["qa"]
        assert "Lancement en juillet" in result["decisions"]
        assert "Alice écrit" in result["actions"]

    def test_extracts_all_sections_en(self) -> None:
        result = extract_sections(EN_SAMPLE, "en")
        assert "Product launch" in result["topic"]
        assert "#product" in result["hashtags"]
        assert "Going well" in result["takeaways"]
        assert "Launch in July" in result["decisions"]
        assert "Alice writes" in result["actions"]

    def test_language_mismatch_yields_only_shared_labels(self) -> None:
        # Only ``Hashtags`` is spelled identically in FR and EN, so it's the
        # sole key that survives a language mismatch. The others don't match.
        fr_as_en = extract_sections(FR_SAMPLE, "en")
        en_as_fr = extract_sections(EN_SAMPLE, "fr")
        assert set(fr_as_en.keys()) <= {"hashtags"}
        assert set(en_as_fr.keys()) <= {"hashtags"}
        assert "topic" not in fr_as_en
        assert "decisions" not in en_as_fr

    def test_missing_section_absent_from_dict(self) -> None:
        partial = "Sujet: Seul sujet ici.\nHashtags: #x\n"
        result = extract_sections(partial, "fr")
        assert "Seul sujet" in result.get("topic", "")
        assert "#x" in result.get("hashtags", "")
        assert "decisions" not in result

    def test_empty_summary(self) -> None:
        assert extract_sections("", "fr") == {}


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
    def test_english_output(self) -> None:
        out = format_summary_markdown(EN_SAMPLE, "video-1", "en")
        assert out.startswith("# Meeting Summary — video-1")
        assert "## Meeting Topic" in out
        assert "Product launch" in out
        assert "## Decisions" in out
        assert "## Action Items" in out

    def test_french_output(self) -> None:
        out = format_summary_markdown(FR_SAMPLE, "video-1", "fr")
        assert out.startswith("# Résumé de la réunion — video-1")
        assert "## Sujet de la réunion" in out
        assert "## Principaux enseignements" in out
        assert "Lancement produit" in out

    def test_missing_sections_get_defaults(self) -> None:
        out = format_summary_markdown("", "x", "en")
        # 6 sections total → 6 "None" placeholders
        assert out.count("None") >= 6

    def test_missing_sections_get_fr_defaults(self) -> None:
        out = format_summary_markdown("", "x", "fr")
        assert out.count("Aucune") >= 6


class TestSimpleFormatMarkdown:
    def test_english_output(self) -> None:
        out = simple_format_markdown("Title", "http://v.com", "body", "Positive", "en")
        assert "Video Summary" in out
        assert "Title: Title" in out
        assert "**Sentiment:** Positive" in out
        assert "body" in out

    def test_french_output(self) -> None:
        out = simple_format_markdown("Titre", "http://v.com", "corps", "Neutre", "fr")
        assert "Résumé de la vidéo" in out
        assert "Titre : Titre" in out
        assert "**Sentiment :** Neutre" in out

    def test_unsupported_language(self) -> None:
        out = simple_format_markdown("t", "p", "b", "s", "de")
        assert "Error" in out
