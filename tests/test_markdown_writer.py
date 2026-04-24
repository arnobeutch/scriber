"""Tests for markdown_writer.

``extract_sections`` and ``format_summary_markdown`` both require a
``mode`` ("meeting" | "source") alongside ``language``. The regex in
``extract_sections`` matches the mode+language-specific labels the prompt
asks the model to emit.
"""

from scriber.model import Chapter, SourceMetadata, Transcript
from scriber.summarizers.markdown import (
    clean_section,
    extract_sections,
    format_summary_markdown,
)


def _transcript(
    *,
    title: str = "video-1",
    language: str = "en",
    chapters: list[Chapter] | None = None,
    metadata: SourceMetadata | None = None,
    diarized: bool = False,
    source: str = "yt_manual",
) -> Transcript:
    return Transcript(
        text="",
        language=language,
        title=title,
        source=source,  # type: ignore[arg-type]
        diarized=diarized,
        chapters=chapters or [],
        metadata=metadata or SourceMetadata(),
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
Summary: The author argues that X.
Main claims:
1. Claim one
2. Claim two
3. Claim three
Notable quotes:
> "Scaling laws are log-linear across three orders of magnitude."
>
> — Jane Doe
Factually correct:
- F1 — supported by the paper
Likely but unconfirmed:
- L1 — plausible but no citation
Interpretation or weakly substantiated:
- I1 — evidence is weak
Alternative interpretations:
- A1 — alternative reading
Wrong or misleading:
- W1 — contradicts known data
Keywords: neural-scaling, transformer-attention, lr-warmup
Tags: #ml #research #nlp
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
        assert "author argues" in result["summary"]
        assert "Claim one" in result["claims"]
        assert "Scaling laws" in result["quotes"]
        assert "Jane Doe" in result["quotes"]
        assert "F1" in result["factual"]
        assert "L1" in result["likely"]
        assert "I1" in result["interpretation"]
        assert "A1" in result["alternatives"]
        assert "W1" in result["wrong"]
        assert "neural-scaling" in result["keywords"]
        assert "#ml" in result["tags"]

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
        t = _transcript(title="video-1", language="en")
        out = format_summary_markdown(EN_MEETING_SAMPLE, t, "meeting")
        assert "# Meeting Summary — video-1" in out
        assert "## Meeting Topic" in out
        assert "Product launch" in out
        assert "## Decisions" in out
        assert "## Action Items" in out

    def test_meeting_fr_output(self) -> None:
        t = _transcript(title="video-1", language="fr")
        out = format_summary_markdown(FR_MEETING_SAMPLE, t, "meeting")
        assert "# Résumé de la réunion — video-1" in out
        assert "## Sujet de la réunion" in out
        assert "## Principaux enseignements" in out
        assert "Lancement produit" in out

    def test_source_en_output(self) -> None:
        t = _transcript(title="vid", language="en")
        out = format_summary_markdown(EN_SOURCE_SAMPLE, t, "source")
        assert "# Summary — vid" in out
        assert "## Summary" in out
        assert "## Main Claims" in out
        assert "## Notable Quotes" in out
        assert "Scaling laws are log-linear" in out
        assert "Jane Doe" in out
        assert "## What Is Factually Correct" in out
        assert "## What Is Likely but Unconfirmed" in out
        assert "## What Is Interpretation or Weakly Substantiated" in out
        assert "### Alternative Interpretations" in out
        assert "## What Is Wrong or Misleading" in out

    def test_source_en_keywords_and_tags_route_to_frontmatter(self) -> None:
        t = _transcript(title="vid", language="en")
        out = format_summary_markdown(EN_SOURCE_SAMPLE, t, "source")
        # Routed to YAML frontmatter, not rendered as visible sections.
        assert 'keywords: ["neural-scaling", "transformer-attention", "lr-warmup"]' in out
        assert 'tags: ["#ml", "#research", "#nlp"]' in out
        # And the visible body has no ## Keywords / ## Tags header.
        assert "## Keywords" not in out
        assert "## Tags" not in out

    def test_source_fr_title(self) -> None:
        t = _transcript(title="vid", language="fr")
        out = format_summary_markdown("", t, "source")
        assert "# Résumé — vid" in out

    def test_missing_sections_get_defaults(self) -> None:
        out = format_summary_markdown("", _transcript(title="x"), "meeting")
        assert out.count("None") >= 6

    def test_missing_sections_get_fr_defaults(self) -> None:
        out = format_summary_markdown("", _transcript(title="x", language="fr"), "meeting")
        assert out.count("Aucune") >= 6

    def test_source_path_rendered_when_provided(self) -> None:
        out = format_summary_markdown(
            EN_MEETING_SAMPLE,
            _transcript(title="v"),
            "meeting",
            source_path="https://youtube.com/watch?v=abc",
        )
        assert "> Source: https://youtube.com/watch?v=abc" in out

    def test_source_path_omitted_when_none(self) -> None:
        out = format_summary_markdown(EN_MEETING_SAMPLE, _transcript(title="v"), "meeting")
        assert "> Source:" not in out

    def test_sentiment_rendered_when_provided(self) -> None:
        out = format_summary_markdown(
            EN_MEETING_SAMPLE,
            _transcript(title="v"),
            "meeting",
            sentiment="Positive",
        )
        assert "## Sentiment" in out
        assert "Positive" in out

    def test_sentiment_omitted_when_none(self) -> None:
        out = format_summary_markdown(EN_MEETING_SAMPLE, _transcript(title="v"), "meeting")
        assert "## Sentiment" not in out

    def test_chapters_render_with_youtube_deep_links(self) -> None:
        chapters = [
            Chapter(start_time=0, title="Intro"),
            Chapter(start_time=154.5, title="Main topic"),
            Chapter(start_time=3725, title="Q & A"),
        ]
        out = format_summary_markdown(
            EN_MEETING_SAMPLE,
            _transcript(title="v", chapters=chapters),
            "meeting",
            source_path="https://www.youtube.com/watch?v=abc",
        )
        assert "## Chapters" in out
        assert "[00:00](https://www.youtube.com/watch?v=abc&t=0) Intro" in out
        assert "[02:34](https://www.youtube.com/watch?v=abc&t=154) Main topic" in out
        assert "[1:02:05](https://www.youtube.com/watch?v=abc&t=3725) Q & A" in out

    def test_chapters_render_plain_when_source_not_url(self) -> None:
        chapters = [Chapter(start_time=30, title="Intro")]
        out = format_summary_markdown(
            EN_MEETING_SAMPLE,
            _transcript(title="v", chapters=chapters),
            "meeting",
            source_path="/local/file.mp4",
        )
        assert "## Chapters" in out
        assert "- 00:30 — Intro" in out

    def test_chapters_section_omitted_when_empty(self) -> None:
        out = format_summary_markdown(EN_MEETING_SAMPLE, _transcript(title="v"), "meeting")
        assert "## Chapters" not in out


class TestFrontmatter:
    def test_frontmatter_present_and_closed(self) -> None:
        out = format_summary_markdown(
            EN_MEETING_SAMPLE,
            _transcript(title="v"),
            "meeting",
        )
        lines = out.splitlines()
        assert lines[0] == "---"
        # There must be a closing --- before the first markdown heading.
        closing = lines.index("---", 1)
        assert any(line.startswith("# ") for line in lines[closing + 1 :])

    def test_frontmatter_renders_core_fields(self) -> None:
        t = _transcript(
            title="vid",
            language="en",
            source="yt_manual",
            metadata=SourceMetadata(
                channel="Some Channel",
                publication_date="2026-04-24",
                detected_language="en",
                duration_seconds=180.5,
            ),
        )
        out = format_summary_markdown(
            EN_MEETING_SAMPLE,
            t,
            "meeting",
            source_path="https://youtu.be/abc",
            sentiment="Positive",
            processing_date="2026-04-24",
        )
        assert 'title: "vid"' in out
        assert 'source_url: "https://youtu.be/abc"' in out
        assert 'source_type: "youtube"' in out
        assert 'transcript_source: "yt_manual"' in out
        assert 'channel: "Some Channel"' in out
        assert 'publication_date: "2026-04-24"' in out
        assert 'processing_date: "2026-04-24"' in out
        assert 'detected_language: "en"' in out
        assert 'summary_language: "en"' in out
        assert 'summary_mode: "meeting"' in out
        assert "duration_seconds: 180.5" in out
        assert "chapters_count: 0" in out
        assert "diarized: false" in out
        assert 'ingestion_status: "full"' in out
        assert 'extraction_status: "ok"' in out
        assert 'sentiment: "Positive"' in out
        assert "keywords: []" in out
        assert "tags: []" in out

    def test_frontmatter_nulls_missing_fields(self) -> None:
        out = format_summary_markdown(
            EN_SOURCE_SAMPLE,
            _transcript(title="v", source="file"),
            "source",
        )
        assert "source_url: null" in out
        assert "channel: null" in out
        assert "publication_date: null" in out
        assert "duration_seconds: null" in out
        assert "sentiment: null" in out

    def test_frontmatter_source_type_text_for_file_source(self) -> None:
        out = format_summary_markdown(
            EN_SOURCE_SAMPLE,
            _transcript(title="v", source="file"),
            "source",
        )
        assert 'source_type: "text"' in out

    def test_frontmatter_source_type_media_for_local_whisper(self) -> None:
        out = format_summary_markdown(
            EN_SOURCE_SAMPLE,
            _transcript(title="v", source="whisper"),
            "source",
            source_path="/local/video.mp4",
        )
        assert 'source_type: "media"' in out

    def test_frontmatter_counts_chapters(self) -> None:
        chapters = [Chapter(start_time=0, title="a"), Chapter(start_time=60, title="b")]
        out = format_summary_markdown(
            EN_MEETING_SAMPLE,
            _transcript(title="v", chapters=chapters),
            "meeting",
        )
        assert "chapters_count: 2" in out

    def test_keywords_and_tags_rendered_as_yaml_list(self) -> None:
        out = format_summary_markdown(
            EN_MEETING_SAMPLE,
            _transcript(title="v"),
            "meeting",
            keywords=["alpha", "beta"],
            tags=["#ml", "#research"],
        )
        assert 'keywords: ["alpha", "beta"]' in out
        assert 'tags: ["#ml", "#research"]' in out
