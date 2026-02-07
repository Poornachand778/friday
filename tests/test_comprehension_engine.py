"""
Tests for BookComprehensionEngine
==================================

Comprehensive tests for the book comprehension engine including
knowledge extraction, parsing, deduplication, quality assessment,
domain inference, and both thorough and sampled comprehension modes.

Run with: pytest tests/test_comprehension_engine.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from documents.config import ComprehensionConfig
from documents.models import (
    Chunk,
    ChapterInfo,
    Document,
    DocumentLanguage,
    DocumentMetadata,
    DocumentType,
)
from documents.understanding.comprehension import BookComprehensionEngine
from documents.understanding.models import (
    BookExample,
    BookUnderstanding,
    Concept,
    ConfidenceLevel,
    Principle,
    Technique,
)


# =========================================================================
# Helpers
# =========================================================================


def make_document(
    doc_id="doc-001",
    title="Story",
    author="Robert McKee",
    total_pages=300,
    chapters=None,
):
    """Create a test Document."""
    return Document(
        id=doc_id,
        file_path="/books/story.pdf",
        file_hash="abc123",
        file_size=5000000,
        document_type=DocumentType.BOOK,
        metadata=DocumentMetadata(title=title, author=author),
        language=DocumentLanguage.ENGLISH,
        total_pages=total_pages,
        chapters=chapters or [],
    )


def make_chunk(
    chunk_id="chunk-001",
    document_id="doc-001",
    content="This is test content about inciting incidents.",
    page_range="pp. 10-12",
    chapter=None,
    page_ids=None,
):
    """Create a test Chunk."""
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        page_ids=page_ids or ["p1", "p2"],
        content=content,
        page_range=page_range,
        chapter=chapter,
        chunk_index=0,
    )


def make_chapters():
    """Create standard test chapters."""
    return [
        ChapterInfo(number=1, title="The Inciting Incident", start_page=1, end_page=30),
        ChapterInfo(number=2, title="Character Arc", start_page=31, end_page=60),
        ChapterInfo(number=3, title="Climax", start_page=61, end_page=90),
    ]


def make_config(**overrides):
    """Create a test ComprehensionConfig with sensible defaults."""
    defaults = dict(
        thorough_mode=False,
        max_chunks_per_extraction=20,
        max_chunks_per_chapter=10,
        max_llm_calls_per_book=50,
        deduplication_enabled=True,
        voice_announce_start=True,
        voice_announce_complete=True,
        voice_progress_interval="chapter",
    )
    defaults.update(overrides)
    return ComprehensionConfig(**defaults)


def make_understanding(
    doc_id="doc-001",
    title="Story",
    author="Robert McKee",
    summary="",
    main_argument="",
    concepts=None,
    principles=None,
    techniques=None,
    examples=None,
):
    """Create a BookUnderstanding for testing."""
    u = BookUnderstanding(
        document_id=doc_id,
        title=title,
        author=author,
        summary=summary,
        main_argument=main_argument,
    )
    if concepts is not None:
        u.concepts = concepts
    if principles is not None:
        u.principles = principles
    if techniques is not None:
        u.techniques = techniques
    if examples is not None:
        u.examples = examples
    return u


def _mock_job_tracker():
    """Create a mock StudyJobTracker."""
    tracker = MagicMock()
    tracker.start_job.return_value = "study_test123"
    tracker.update_status = MagicMock()
    tracker.update_progress = MagicMock()
    tracker.complete_job = MagicMock()
    tracker.fail_job = MagicMock()
    return tracker


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_llm():
    """Async mock for llm_complete."""
    return AsyncMock(return_value='{"summary": "A great book about story."}')


@pytest.fixture
def config():
    """Default test config (sampled mode, dedup on)."""
    return make_config()


@pytest.fixture
def mock_tracker():
    """Mock StudyJobTracker."""
    return _mock_job_tracker()


@pytest.fixture
def engine(mock_llm, config, mock_tracker):
    """Create a BookComprehensionEngine with mocked dependencies."""
    with patch(
        "documents.understanding.comprehension.get_job_tracker",
        return_value=mock_tracker,
    ):
        eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
    return eng


# =========================================================================
# 1. TestComprehensionEngineInit
# =========================================================================


class TestComprehensionEngineInit:
    """Tests for engine initialization."""

    def test_stores_llm_complete(self, mock_tracker):
        """llm_complete callable is stored on the engine."""
        llm = AsyncMock()
        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=llm)
        assert eng._llm_complete is llm

    def test_default_config_used_when_none(self, mock_tracker):
        """When config is None, the engine uses the global default config."""
        llm = AsyncMock()
        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            with patch(
                "documents.understanding.comprehension.get_document_config"
            ) as mock_cfg:
                mock_cfg.return_value.comprehension = ComprehensionConfig()
                eng = BookComprehensionEngine(llm_complete=llm, config=None)
        assert eng._config is not None

    def test_custom_config(self, mock_tracker):
        """Custom ComprehensionConfig is used when provided."""
        llm = AsyncMock()
        custom = make_config(thorough_mode=True, max_llm_calls_per_book=10)
        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=llm, config=custom)
        assert eng._config is custom
        assert eng._config.thorough_mode is True
        assert eng._config.max_llm_calls_per_book == 10

    def test_llm_call_count_starts_at_zero(self, engine):
        """LLM call counter initialises to zero."""
        assert engine._llm_call_count == 0


# =========================================================================
# 2. TestSampleChunks
# =========================================================================


class TestSampleChunks:
    """Tests for _sample_chunks helper."""

    def test_returns_all_when_count_le_max(self, engine):
        """All chunks returned when count <= max_chunks."""
        chunks = [make_chunk(chunk_id=f"c{i}") for i in range(5)]
        result = engine._sample_chunks(chunks, 10)
        assert result == chunks

    def test_returns_all_when_equal(self, engine):
        """All chunks returned when count == max_chunks."""
        chunks = [make_chunk(chunk_id=f"c{i}") for i in range(5)]
        result = engine._sample_chunks(chunks, 5)
        assert result == chunks

    def test_returns_evenly_spaced(self, engine):
        """Evenly spaced samples when count > max_chunks."""
        chunks = [
            make_chunk(chunk_id=f"c{i}", content=f"Content {i}") for i in range(20)
        ]
        result = engine._sample_chunks(chunks, 5)
        assert len(result) == 5
        # Step should be 20 // 5 = 4, so indices 0, 4, 8, 12, 16
        assert result[0] is chunks[0]
        assert result[1] is chunks[4]
        assert result[2] is chunks[8]
        assert result[3] is chunks[12]
        assert result[4] is chunks[16]

    def test_single_chunk(self, engine):
        """Single chunk list always returned as-is."""
        chunks = [make_chunk()]
        result = engine._sample_chunks(chunks, 5)
        assert result == chunks

    def test_empty_list(self, engine):
        """Empty list returns empty."""
        result = engine._sample_chunks([], 5)
        assert result == []


# =========================================================================
# 3. TestPrepareContent
# =========================================================================


class TestPrepareContent:
    """Tests for _prepare_content helper."""

    def test_chunks_with_page_range(self, engine):
        """Chunks with page_range get a [page_range] header."""
        chunks = [
            make_chunk(content="First chunk.", page_range="pp. 10-12"),
            make_chunk(content="Second chunk.", page_range="pp. 13-15"),
        ]
        result = engine._prepare_content(chunks)
        assert "[pp. 10-12]" in result
        assert "First chunk." in result
        assert "[pp. 13-15]" in result
        assert "Second chunk." in result

    def test_chunks_without_page_range(self, engine):
        """Chunks without page_range have bare content."""
        chunks = [make_chunk(content="Bare content.", page_range="")]
        result = engine._prepare_content(chunks)
        assert result == "Bare content."
        assert "[" not in result

    def test_empty_list(self, engine):
        """Empty chunk list returns empty string."""
        result = engine._prepare_content([])
        assert result == ""

    def test_separator_between_chunks(self, engine):
        """Chunks are separated by ---."""
        chunks = [
            make_chunk(content="A", page_range=""),
            make_chunk(content="B", page_range=""),
        ]
        result = engine._prepare_content(chunks)
        assert "\n\n---\n\n" in result
        parts = result.split("\n\n---\n\n")
        assert parts[0] == "A"
        assert parts[1] == "B"


# =========================================================================
# 4. TestParseJsonResponse
# =========================================================================


class TestParseJsonResponse:
    """Tests for _parse_json_response."""

    def test_valid_json(self, engine):
        """Plain JSON object is parsed correctly."""
        resp = '{"key": "value", "num": 42}'
        result = engine._parse_json_response(resp)
        assert result == {"key": "value", "num": 42}

    def test_json_embedded_in_text(self, engine):
        """JSON extracted from surrounding prose."""
        resp = 'Here are the results:\n{"concepts": [{"name": "test"}]}\nEnd.'
        result = engine._parse_json_response(resp)
        assert result == {"concepts": [{"name": "test"}]}

    def test_json_array_wrapped(self, engine):
        """JSON array is wrapped in {"items": [...]}."""
        resp = '[{"a": 1}, {"a": 2}]'
        result = engine._parse_json_response(resp)
        assert result == {"items": [{"a": 1}, {"a": 2}]}

    def test_invalid_json_returns_empty(self, engine):
        """Invalid JSON returns empty dict."""
        resp = "This is not JSON at all, just plain text"
        result = engine._parse_json_response(resp)
        assert result == {}

    def test_markdown_code_block(self, engine):
        """JSON inside a markdown code block is extracted."""
        resp = '```json\n{"summary": "Good book"}\n```'
        result = engine._parse_json_response(resp)
        assert result == {"summary": "Good book"}

    def test_empty_response(self, engine):
        """Empty string returns empty dict."""
        result = engine._parse_json_response("")
        assert result == {}


# =========================================================================
# 5. TestInferDomains
# =========================================================================


class TestInferDomains:
    """Tests for _infer_domains."""

    def test_court_drama_keywords(self, engine):
        """Court drama domain detected from trial-related concepts."""
        understanding = make_understanding(
            concepts=[
                Concept(
                    name="Cross-examination",
                    definition="Questioning a witness in trial",
                ),
            ],
        )
        domains = engine._infer_domains(understanding)
        assert "court_drama" in domains

    def test_character_keywords(self, engine):
        """Character domain detected from character-related concepts."""
        understanding = make_understanding(
            concepts=[
                Concept(name="Character Arc", definition="How a protagonist changes"),
            ],
        )
        domains = engine._infer_domains(understanding)
        assert "character" in domains

    def test_multiple_domains(self, engine):
        """Multiple domains detected simultaneously."""
        understanding = make_understanding(
            concepts=[
                Concept(name="Court Strategy", definition="Lawyer tactics in trial"),
                Concept(name="Character Growth", definition="The protagonist arc"),
            ],
            principles=[
                Principle(statement="Every scene must build tension through conflict"),
            ],
        )
        domains = engine._infer_domains(understanding)
        assert "court_drama" in domains
        assert "character" in domains
        assert "tension" in domains

    def test_no_matching_keywords_defaults_to_screenwriting(self, engine):
        """When no keywords match, domain defaults to ['screenwriting']."""
        understanding = make_understanding(
            concepts=[
                Concept(name="Zxywq", definition="Totally made up term"),
            ],
        )
        domains = engine._infer_domains(understanding)
        assert domains == ["screenwriting"]

    def test_empty_concepts_defaults_to_screenwriting(self, engine):
        """Empty concepts and principles default to screenwriting."""
        understanding = make_understanding(concepts=[], principles=[])
        domains = engine._infer_domains(understanding)
        assert domains == ["screenwriting"]

    def test_dialogue_keywords(self, engine):
        """Dialogue domain detected from dialogue-related concepts."""
        understanding = make_understanding(
            concepts=[
                Concept(name="Subtext", definition="Hidden meaning in dialogue"),
            ],
        )
        domains = engine._infer_domains(understanding)
        assert "dialogue" in domains


# =========================================================================
# 6. TestAssessQuality
# =========================================================================


class TestAssessQuality:
    """Tests for _assess_quality."""

    def test_full_coverage_high_score(self, engine):
        """Understanding with good coverage scores high."""
        understanding = make_understanding(
            summary="Full summary of the book.",
            main_argument="The author argues for structured storytelling.",
            concepts=[Concept(name=f"C{i}") for i in range(10)],
            principles=[Principle(statement=f"P{i}") for i in range(15)],
            techniques=[Technique(name=f"T{i}") for i in range(7)],
            examples=[BookExample(work_title=f"E{i}") for i in range(7)],
        )
        score = engine._assess_quality(understanding)
        # summary=0.2 + main_arg=0.1 + concepts(5-20)=0.2 + principles(10-25)=0.2 + tech>=5=0.15 + ex>=5=0.15
        assert score == pytest.approx(1.0)

    def test_no_content_zero(self, engine):
        """Empty understanding scores 0.0."""
        understanding = make_understanding()
        score = engine._assess_quality(understanding)
        assert score == 0.0

    def test_only_summary(self, engine):
        """Only summary gives 0.2."""
        understanding = make_understanding(summary="Just a summary.")
        score = engine._assess_quality(understanding)
        assert score == pytest.approx(0.2)

    def test_summary_and_main_argument(self, engine):
        """Summary + main argument gives 0.3."""
        understanding = make_understanding(
            summary="A summary.", main_argument="A thesis."
        )
        score = engine._assess_quality(understanding)
        assert score == pytest.approx(0.3)

    def test_concepts_in_sweet_spot(self, engine):
        """5-20 concepts gets the full 0.2 concept bonus."""
        understanding = make_understanding(
            concepts=[Concept(name=f"C{i}") for i in range(10)],
        )
        score = engine._assess_quality(understanding)
        # concepts(5-20)=0.2
        assert score == pytest.approx(0.2)

    def test_concepts_outside_sweet_spot(self, engine):
        """Concepts > 0 but < 5 gets only 0.1."""
        understanding = make_understanding(
            concepts=[Concept(name=f"C{i}") for i in range(3)],
        )
        score = engine._assess_quality(understanding)
        assert score == pytest.approx(0.1)

    def test_principles_in_sweet_spot(self, engine):
        """10-25 principles gets 0.2."""
        understanding = make_understanding(
            principles=[Principle(statement=f"P{i}") for i in range(15)],
        )
        score = engine._assess_quality(understanding)
        assert score == pytest.approx(0.2)

    def test_score_capped_at_one(self, engine):
        """Score never exceeds 1.0."""
        understanding = make_understanding(
            summary="Full summary of the book.",
            main_argument="The author argues for structured storytelling.",
            concepts=[Concept(name=f"C{i}") for i in range(10)],
            principles=[Principle(statement=f"P{i}") for i in range(15)],
            techniques=[Technique(name=f"T{i}") for i in range(10)],
            examples=[BookExample(work_title=f"E{i}") for i in range(10)],
        )
        score = engine._assess_quality(understanding)
        assert score <= 1.0


# =========================================================================
# 7. TestParseConcepts
# =========================================================================


class TestParseConcepts:
    """Tests for _parse_concepts."""

    def test_valid_concept_data(self, engine):
        """Correctly structured concept data produces Concept objects."""
        data = {
            "concepts": [
                {
                    "name": "Inciting Incident",
                    "definition": "The event that starts the story",
                    "importance": "Without it, there is no story",
                    "related_concepts": ["Climax", "Resolution"],
                }
            ]
        }
        result = engine._parse_concepts(data, "doc-001")
        assert len(result) == 1
        assert result[0].name == "Inciting Incident"
        assert result[0].definition == "The event that starts the story"
        assert result[0].importance == "Without it, there is no story"
        assert result[0].source_document_id == "doc-001"
        assert result[0].related_concepts == ["Climax", "Resolution"]

    def test_missing_fields_use_defaults(self, engine):
        """Missing fields default to empty strings/lists."""
        data = {"concepts": [{"name": "Minimal"}]}
        result = engine._parse_concepts(data, "doc-001")
        assert len(result) == 1
        assert result[0].name == "Minimal"
        assert result[0].definition == ""
        assert result[0].importance == ""
        assert result[0].related_concepts == []

    def test_empty_concepts_list(self, engine):
        """Empty concepts list returns empty list."""
        result = engine._parse_concepts({"concepts": []}, "doc-001")
        assert result == []

    def test_no_concepts_key(self, engine):
        """Data without 'concepts' key returns empty list."""
        result = engine._parse_concepts({}, "doc-001")
        assert result == []

    def test_multiple_concepts(self, engine):
        """Multiple concepts are all parsed."""
        data = {
            "concepts": [
                {"name": "A", "definition": "Def A"},
                {"name": "B", "definition": "Def B"},
                {"name": "C", "definition": "Def C"},
            ]
        }
        result = engine._parse_concepts(data, "doc-001")
        assert len(result) == 3
        assert [c.name for c in result] == ["A", "B", "C"]


# =========================================================================
# 8. TestParsePrinciples
# =========================================================================


class TestParsePrinciples:
    """Tests for _parse_principles."""

    def test_valid_principle_with_confidence(self, engine):
        """Fully specified principle is parsed correctly."""
        data = {
            "principles": [
                {
                    "statement": "Every scene needs conflict",
                    "rationale": "Conflict drives engagement",
                    "applies_to": ["dialogue", "action"],
                    "exceptions": ["montage"],
                    "confidence_level": "absolute",
                    "check_question": "Does this scene have conflict?",
                }
            ]
        }
        result = engine._parse_principles(data, "doc-001")
        assert len(result) == 1
        p = result[0]
        assert p.statement == "Every scene needs conflict"
        assert p.rationale == "Conflict drives engagement"
        assert p.confidence_level == ConfidenceLevel.ABSOLUTE
        assert p.applies_to == ["dialogue", "action"]
        assert p.exceptions == ["montage"]
        assert p.check_question == "Does this scene have conflict?"
        assert p.source_document_id == "doc-001"

    def test_confidence_level_mapping_strong(self, engine):
        """'strong' maps to ConfidenceLevel.STRONG."""
        data = {"principles": [{"statement": "Test", "confidence_level": "strong"}]}
        result = engine._parse_principles(data, "doc-001")
        assert result[0].confidence_level == ConfidenceLevel.STRONG

    def test_confidence_level_mapping_moderate(self, engine):
        """'moderate' maps to ConfidenceLevel.MODERATE."""
        data = {"principles": [{"statement": "Test", "confidence_level": "moderate"}]}
        result = engine._parse_principles(data, "doc-001")
        assert result[0].confidence_level == ConfidenceLevel.MODERATE

    def test_confidence_level_mapping_suggestion(self, engine):
        """'suggestion' maps to ConfidenceLevel.SUGGESTION."""
        data = {"principles": [{"statement": "Test", "confidence_level": "suggestion"}]}
        result = engine._parse_principles(data, "doc-001")
        assert result[0].confidence_level == ConfidenceLevel.SUGGESTION

    def test_default_confidence_is_strong(self, engine):
        """Missing confidence_level defaults to STRONG."""
        data = {"principles": [{"statement": "No confidence specified"}]}
        result = engine._parse_principles(data, "doc-001")
        assert result[0].confidence_level == ConfidenceLevel.STRONG


# =========================================================================
# 9. TestParseTechniques
# =========================================================================


class TestParseTechniques:
    """Tests for _parse_techniques."""

    def test_valid_technique(self, engine):
        """Fully specified technique is parsed correctly."""
        data = {
            "techniques": [
                {
                    "name": "The Slow Reveal",
                    "description": "Gradually reveal information",
                    "when_to_use": "When building suspense",
                    "example_films": ["Chinatown", "The Sixth Sense"],
                }
            ]
        }
        result = engine._parse_techniques(data, "doc-001")
        assert len(result) == 1
        t = result[0]
        assert t.name == "The Slow Reveal"
        assert t.description == "Gradually reveal information"
        assert t.when_to_use == "When building suspense"
        assert t.example_films == ["Chinatown", "The Sixth Sense"]
        assert t.source_document_id == "doc-001"

    def test_missing_fields_use_defaults(self, engine):
        """Missing fields default to empty strings/lists."""
        data = {"techniques": [{"name": "Minimal"}]}
        result = engine._parse_techniques(data, "doc-001")
        assert len(result) == 1
        assert result[0].description == ""
        assert result[0].when_to_use == ""
        assert result[0].example_films == []

    def test_empty_list(self, engine):
        """Empty techniques list returns empty."""
        result = engine._parse_techniques({"techniques": []}, "doc-001")
        assert result == []

    def test_no_techniques_key(self, engine):
        """Missing 'techniques' key returns empty list."""
        result = engine._parse_techniques({}, "doc-001")
        assert result == []


# =========================================================================
# 10. TestParseExamples
# =========================================================================


class TestParseExamples:
    """Tests for _parse_examples."""

    def test_valid_example(self, engine):
        """Fully specified example is parsed correctly."""
        data = {
            "examples": [
                {
                    "work_title": "12 Angry Men",
                    "scene_or_section": "The final vote",
                    "description": "Juror 8 stands alone",
                    "lesson": "One person can change minds",
                    "demonstrates": ["persuasion", "tension"],
                }
            ]
        }
        result = engine._parse_examples(data, "doc-001")
        assert len(result) == 1
        e = result[0]
        assert e.work_title == "12 Angry Men"
        assert e.scene_or_section == "The final vote"
        assert e.description == "Juror 8 stands alone"
        assert e.lesson == "One person can change minds"
        assert e.demonstrates_concept == ["persuasion", "tension"]
        assert e.source_document_id == "doc-001"

    def test_missing_fields_use_defaults(self, engine):
        """Missing fields default to empty strings/lists."""
        data = {"examples": [{"work_title": "Minimal"}]}
        result = engine._parse_examples(data, "doc-001")
        assert len(result) == 1
        assert result[0].scene_or_section == ""
        assert result[0].description == ""
        assert result[0].lesson == ""
        assert result[0].demonstrates_concept == []

    def test_empty_list(self, engine):
        """Empty examples list returns empty."""
        result = engine._parse_examples({"examples": []}, "doc-001")
        assert result == []

    def test_no_examples_key(self, engine):
        """Missing 'examples' key returns empty list."""
        result = engine._parse_examples({}, "doc-001")
        assert result == []


# =========================================================================
# 11. TestDeduplicateKnowledge
# =========================================================================


class TestDeduplicateKnowledge:
    """Tests for _deduplicate_knowledge."""

    def test_duplicate_concepts_by_name(self, engine):
        """Concepts with same name (case-insensitive) are deduplicated."""
        understanding = make_understanding(
            concepts=[
                Concept(name="Inciting Incident", definition="First def"),
                Concept(name="inciting incident", definition="Second def"),
                Concept(name="Character Arc", definition="Unique"),
            ],
        )
        result = engine._deduplicate_knowledge(understanding)
        assert len(result.concepts) == 2
        names = [c.name for c in result.concepts]
        assert "Inciting Incident" in names
        assert "Character Arc" in names

    def test_duplicate_principles_by_statement_prefix(self, engine):
        """Principles with same first-50-char prefix are deduplicated."""
        long_statement = "A" * 60
        understanding = make_understanding(
            principles=[
                Principle(statement=long_statement + " first version"),
                Principle(statement=long_statement + " second version"),
                Principle(statement="Unique principle statement"),
            ],
        )
        result = engine._deduplicate_knowledge(understanding)
        assert len(result.principles) == 2

    def test_duplicate_techniques_by_name(self, engine):
        """Techniques with same name (case-insensitive) are deduplicated."""
        understanding = make_understanding(
            techniques=[
                Technique(name="Slow Reveal", description="First"),
                Technique(name="slow reveal", description="Second"),
            ],
        )
        result = engine._deduplicate_knowledge(understanding)
        assert len(result.techniques) == 1
        assert result.techniques[0].name == "Slow Reveal"

    def test_duplicate_examples_by_work_title_scene(self, engine):
        """Examples with same work_title:scene_or_section are deduplicated."""
        understanding = make_understanding(
            examples=[
                BookExample(work_title="12 Angry Men", scene_or_section="The vote"),
                BookExample(work_title="12 angry men", scene_or_section="the vote"),
                BookExample(
                    work_title="12 Angry Men", scene_or_section="Opening scene"
                ),
            ],
        )
        result = engine._deduplicate_knowledge(understanding)
        assert len(result.examples) == 2

    def test_case_insensitive_dedup(self, engine):
        """Deduplication is case-insensitive for all knowledge types."""
        understanding = make_understanding(
            concepts=[
                Concept(name="PLOT TWIST"),
                Concept(name="Plot Twist"),
                Concept(name="plot twist"),
            ],
        )
        result = engine._deduplicate_knowledge(understanding)
        assert len(result.concepts) == 1

    def test_no_duplicates_returns_all(self, engine):
        """When there are no duplicates, all items are preserved."""
        understanding = make_understanding(
            concepts=[
                Concept(name="A"),
                Concept(name="B"),
                Concept(name="C"),
            ],
            principles=[
                Principle(statement="P1"),
                Principle(statement="P2"),
            ],
            techniques=[
                Technique(name="T1"),
            ],
            examples=[
                BookExample(work_title="E1", scene_or_section="S1"),
            ],
        )
        result = engine._deduplicate_knowledge(understanding)
        assert len(result.concepts) == 3
        assert len(result.principles) == 2
        assert len(result.techniques) == 1
        assert len(result.examples) == 1


# =========================================================================
# 12. TestGroupChunksByChapter
# =========================================================================


class TestGroupChunksByChapter:
    """Tests for _group_chunks_by_chapter."""

    def test_chunks_with_chapter_set(self, engine):
        """Chunks with chapter attribute are placed in correct group."""
        chapters = make_chapters()
        chunks = [
            make_chunk(chunk_id="c1", chapter="The Inciting Incident"),
            make_chunk(chunk_id="c2", chapter="The Inciting Incident"),
            make_chunk(chunk_id="c3", chapter="Character Arc"),
        ]
        result = engine._group_chunks_by_chapter(chunks, chapters)
        assert len(result["The Inciting Incident"]) == 2
        assert len(result["Character Arc"]) == 1
        assert len(result["Climax"]) == 0

    def test_chunks_without_chapter_assigned_to_first(self, engine):
        """Chunks without chapter fall back to first chapter via page_ids."""
        chapters = make_chapters()
        chunks = [
            make_chunk(chunk_id="c1", chapter=None, page_ids=["p1"]),
        ]
        result = engine._group_chunks_by_chapter(chunks, chapters)
        # Falls back to first chapter
        assert len(result["The Inciting Incident"]) == 1

    def test_no_chapters(self, engine):
        """When no chapters exist, returns empty dict."""
        chunks = [make_chunk(chunk_id="c1")]
        result = engine._group_chunks_by_chapter(chunks, [])
        assert result == {}

    def test_chunk_with_unknown_chapter(self, engine):
        """Chunk whose chapter doesn't match any known chapter falls to first."""
        chapters = make_chapters()
        chunks = [
            make_chunk(chunk_id="c1", chapter="Nonexistent Chapter", page_ids=["p1"]),
        ]
        result = engine._group_chunks_by_chapter(chunks, chapters)
        # chapter doesn't match, falls through to page_ids fallback
        assert len(result["The Inciting Incident"]) == 1


# =========================================================================
# 13. TestComprehendSampled
# =========================================================================


class TestComprehendSampled:
    """Tests for the sampled comprehension flow."""

    @pytest.mark.asyncio
    async def test_sampled_produces_understanding(self, mock_tracker):
        """Sampled mode produces a BookUnderstanding with extracted content."""
        llm_responses = [
            # Summary
            json.dumps(
                {
                    "summary": "A book about story.",
                    "main_argument": "Structure matters.",
                    "target_audience": "Writers",
                }
            ),
            # Concepts
            json.dumps(
                {"concepts": [{"name": "Plot", "definition": "The sequence of events"}]}
            ),
            # Principles
            json.dumps(
                {
                    "principles": [
                        {
                            "statement": "Every scene must turn",
                            "confidence_level": "strong",
                        }
                    ]
                }
            ),
            # Techniques
            json.dumps(
                {
                    "techniques": [
                        {"name": "Foreshadowing", "description": "Hint at what comes"}
                    ]
                }
            ),
            # Examples
            json.dumps(
                {
                    "examples": [
                        {"work_title": "Chinatown", "scene_or_section": "The ending"}
                    ]
                }
            ),
        ]
        mock_llm = AsyncMock(side_effect=llm_responses)
        config = make_config(thorough_mode=False, deduplication_enabled=False)
        doc = make_document()
        chunks = [make_chunk(chunk_id=f"c{i}") for i in range(5)]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            result = await eng.comprehend(doc, chunks)

        assert result.summary == "A book about story."
        assert result.main_argument == "Structure matters."
        assert len(result.concepts) == 1
        assert result.concepts[0].name == "Plot"
        assert len(result.principles) == 1
        assert len(result.techniques) == 1
        assert len(result.examples) == 1

    @pytest.mark.asyncio
    async def test_sampled_calls_llm_five_times(self, mock_tracker):
        """Sampled mode makes 5 LLM calls (summary, concepts, principles, techniques, examples)."""
        mock_llm = AsyncMock(return_value='{"summary": "test"}')
        config = make_config(thorough_mode=False, deduplication_enabled=False)
        doc = make_document()
        chunks = [make_chunk()]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            await eng.comprehend(doc, chunks)

        assert mock_llm.call_count == 5

    @pytest.mark.asyncio
    async def test_sampled_progress_callback_called(self, mock_tracker):
        """Progress callback is called during sampled comprehension."""
        mock_llm = AsyncMock(return_value='{"summary": "test"}')
        config = make_config(thorough_mode=False, deduplication_enabled=False)
        doc = make_document()
        chunks = [make_chunk()]
        progress_cb = MagicMock()

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            await eng.comprehend(doc, chunks, progress_callback=progress_cb)

        # Should be called for each extraction step + "Complete"
        assert progress_cb.call_count >= 5
        # Last call should be ("Complete", 1.0)
        progress_cb.assert_called_with("Complete", 1.0)

    @pytest.mark.asyncio
    async def test_sampled_voice_callback_called(self, mock_tracker):
        """Voice callback is called during sampled comprehension."""
        mock_llm = AsyncMock(return_value='{"summary": "test"}')
        config = make_config(
            thorough_mode=False,
            deduplication_enabled=False,
            voice_announce_start=True,
            voice_announce_complete=True,
        )
        doc = make_document(chapters=[])
        chunks = [make_chunk()]
        voice_cb = MagicMock()

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            await eng.comprehend(doc, chunks, voice_callback=voice_cb)

        # Voice should be called for: start + each extraction step + complete
        assert voice_cb.call_count >= 2  # At least start and complete
        # First call should mention the book title
        first_call_msg = voice_cb.call_args_list[0][0][0]
        assert "Story" in first_call_msg


# =========================================================================
# 14. TestComprehendThorough
# =========================================================================


class TestComprehendThorough:
    """Tests for thorough (chapter-by-chapter) comprehension flow."""

    @pytest.mark.asyncio
    async def test_thorough_processes_each_chapter(self, mock_tracker):
        """Thorough mode processes each chapter individually."""
        chapter_response = json.dumps(
            {
                "chapter_summary": {
                    "summary": "Chapter summary",
                    "key_points": ["Point 1"],
                },
                "concepts": [{"name": "TestConcept", "definition": "A test"}],
                "principles": [
                    {"statement": "Test principle", "confidence_level": "strong"}
                ],
                "techniques": [
                    {"name": "TestTechnique", "description": "A test technique"}
                ],
                "examples": [{"work_title": "TestFilm", "scene_or_section": "Opening"}],
            }
        )
        # Summary call + 3 chapter calls
        mock_llm = AsyncMock(return_value=chapter_response)
        chapters = make_chapters()
        config = make_config(thorough_mode=True, deduplication_enabled=False)
        doc = make_document(chapters=chapters)
        chunks = [
            make_chunk(chunk_id="c1", chapter="The Inciting Incident"),
            make_chunk(chunk_id="c2", chapter="Character Arc"),
            make_chunk(chunk_id="c3", chapter="Climax"),
        ]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            result = await eng.comprehend(doc, chunks)

        # 1 summary call + 3 chapter calls = 4
        assert mock_llm.call_count == 4
        # Each chapter contributes 1 concept, so 3 total
        assert len(result.concepts) == 3

    @pytest.mark.asyncio
    async def test_thorough_llm_call_limit_respected(self, mock_tracker):
        """Thorough mode stops when hitting max_llm_calls_per_book."""
        chapter_response = json.dumps(
            {
                "chapter_summary": {"summary": "Summary", "key_points": []},
                "concepts": [],
                "principles": [],
                "techniques": [],
                "examples": [],
            }
        )
        mock_llm = AsyncMock(return_value=chapter_response)
        chapters = [
            ChapterInfo(
                number=i, title=f"Chapter {i}", start_page=i * 10, end_page=i * 10 + 9
            )
            for i in range(1, 20)
        ]
        # max_llm_calls_per_book=3: 1 for summary + at most 2 chapters before hitting limit
        config = make_config(
            thorough_mode=True, max_llm_calls_per_book=3, deduplication_enabled=False
        )
        doc = make_document(chapters=chapters)
        chunks = [
            make_chunk(chunk_id=f"c{i}", chapter=f"Chapter {i}") for i in range(1, 20)
        ]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            await eng.comprehend(doc, chunks)

        # Should stop around the limit (summary + up to limit chapters)
        assert mock_llm.call_count <= config.max_llm_calls_per_book + 1

    @pytest.mark.asyncio
    async def test_thorough_skips_empty_chapters(self, mock_tracker):
        """Thorough mode skips chapters with no chunks."""
        chapter_response = json.dumps(
            {
                "chapter_summary": {"summary": "Summary", "key_points": []},
                "concepts": [{"name": "Found", "definition": "Test"}],
                "principles": [],
                "techniques": [],
                "examples": [],
            }
        )
        mock_llm = AsyncMock(return_value=chapter_response)
        chapters = make_chapters()  # 3 chapters
        config = make_config(thorough_mode=True, deduplication_enabled=False)
        doc = make_document(chapters=chapters)
        # Only provide chunks for one chapter
        chunks = [make_chunk(chunk_id="c1", chapter="The Inciting Incident")]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            result = await eng.comprehend(doc, chunks)

        # 1 summary call + 1 chapter call (other 2 chapters have no chunks)
        assert mock_llm.call_count == 2
        assert len(result.concepts) == 1

    @pytest.mark.asyncio
    async def test_thorough_voice_progress_per_chapter(self, mock_tracker):
        """Thorough mode calls voice callback for each chapter when interval is 'chapter'."""
        chapter_response = json.dumps(
            {
                "chapter_summary": {"summary": "Summary", "key_points": []},
                "concepts": [],
                "principles": [],
                "techniques": [],
                "examples": [],
            }
        )
        mock_llm = AsyncMock(return_value=chapter_response)
        chapters = make_chapters()
        config = make_config(
            thorough_mode=True,
            deduplication_enabled=False,
            voice_announce_start=True,
            voice_announce_complete=True,
            voice_progress_interval="chapter",
        )
        doc = make_document(chapters=chapters)
        chunks = [
            make_chunk(chunk_id="c1", chapter="The Inciting Incident"),
            make_chunk(chunk_id="c2", chapter="Character Arc"),
            make_chunk(chunk_id="c3", chapter="Climax"),
        ]
        voice_cb = MagicMock()

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            await eng.comprehend(doc, chunks, voice_callback=voice_cb)

        # Voice calls: start + extracting_summary + 3*(chapter_start + chapter_complete) + complete
        # At minimum: start, extracting_summary, and for each chapter start/complete, plus final
        assert voice_cb.call_count >= 7
        # Check that chapter announcements happened
        all_msgs = [call[0][0] for call in voice_cb.call_args_list]
        chapter_starts = [m for m in all_msgs if "Studying Chapter" in m]
        assert len(chapter_starts) == 3


# =========================================================================
# 15. TestExtractFromChapter
# =========================================================================


class TestExtractFromChapter:
    """Tests for _extract_from_chapter."""

    @pytest.mark.asyncio
    async def test_returns_all_knowledge_types(self, mock_tracker):
        """Extracts concepts, principles, techniques, examples, and summary from a chapter."""
        response = json.dumps(
            {
                "chapter_summary": {
                    "summary": "This chapter covers the basics.",
                    "key_points": ["Point A", "Point B"],
                },
                "concepts": [{"name": "Concept1", "definition": "Def1"}],
                "principles": [
                    {"statement": "Principle1", "confidence_level": "strong"}
                ],
                "techniques": [{"name": "Technique1", "description": "Desc1"}],
                "examples": [{"work_title": "Film1", "scene_or_section": "Scene1"}],
            }
        )
        mock_llm = AsyncMock(return_value=response)
        config = make_config()

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)

        concepts, principles, techniques, examples, ch_summary = (
            await eng._extract_from_chapter(
                "Story", "McKee", "Chapter 1", 1, "Content here", "doc-001"
            )
        )

        assert len(concepts) == 1
        assert concepts[0].name == "Concept1"
        assert len(principles) == 1
        assert len(techniques) == 1
        assert len(examples) == 1
        assert ch_summary is not None
        assert ch_summary.summary == "This chapter covers the basics."
        assert ch_summary.key_points == ["Point A", "Point B"]

    @pytest.mark.asyncio
    async def test_increments_llm_call_count(self, mock_tracker):
        """Each call to _extract_from_chapter increments _llm_call_count."""
        mock_llm = AsyncMock(return_value="{}")
        config = make_config()

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)

        assert eng._llm_call_count == 0
        await eng._extract_from_chapter(
            "Book", "Author", "Ch1", 1, "Content", "doc-001"
        )
        assert eng._llm_call_count == 1


# =========================================================================
# 16. TestComprehendIntegration (end-to-end)
# =========================================================================


class TestComprehendIntegration:
    """Integration tests for the main comprehend() method."""

    @pytest.mark.asyncio
    async def test_job_tracker_lifecycle(self, mock_tracker):
        """Job tracker is called with start, update_status, and complete_job."""
        mock_llm = AsyncMock(return_value='{"summary": "test"}')
        config = make_config(thorough_mode=False, deduplication_enabled=False)
        doc = make_document()
        chunks = [make_chunk()]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            await eng.comprehend(doc, chunks)

        mock_tracker.start_job.assert_called_once()
        mock_tracker.update_status.assert_called()
        mock_tracker.complete_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_job_tracker_fail_on_exception(self, mock_tracker):
        """Job tracker fail_job is called if comprehension raises an exception."""
        mock_llm = AsyncMock(side_effect=RuntimeError("LLM connection failed"))
        config = make_config(thorough_mode=False)
        doc = make_document()
        chunks = [make_chunk()]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            with pytest.raises(RuntimeError, match="LLM connection failed"):
                await eng.comprehend(doc, chunks)

        mock_tracker.fail_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_deduplication_runs_when_enabled(self, mock_tracker):
        """Deduplication is applied when config.deduplication_enabled is True."""
        responses = [
            json.dumps({"summary": "Summary"}),
            json.dumps(
                {
                    "concepts": [
                        {"name": "Duplicate", "definition": "A"},
                        {"name": "duplicate", "definition": "B"},
                    ]
                }
            ),
            json.dumps({"principles": []}),
            json.dumps({"techniques": []}),
            json.dumps({"examples": []}),
        ]
        mock_llm = AsyncMock(side_effect=responses)
        config = make_config(thorough_mode=False, deduplication_enabled=True)
        doc = make_document()
        chunks = [make_chunk()]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            result = await eng.comprehend(doc, chunks)

        # Should be deduplicated from 2 to 1
        assert len(result.concepts) == 1

    @pytest.mark.asyncio
    async def test_domains_are_set(self, mock_tracker):
        """Domains are inferred and set on the understanding."""
        responses = [
            json.dumps({"summary": "A book about trials"}),
            json.dumps(
                {"concepts": [{"name": "Trial", "definition": "A court trial"}]}
            ),
            json.dumps({"principles": []}),
            json.dumps({"techniques": []}),
            json.dumps({"examples": []}),
        ]
        mock_llm = AsyncMock(side_effect=responses)
        config = make_config(thorough_mode=False, deduplication_enabled=False)
        doc = make_document()
        chunks = [make_chunk()]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            result = await eng.comprehend(doc, chunks)

        assert len(result.domains) > 0

    @pytest.mark.asyncio
    async def test_quality_is_assessed(self, mock_tracker):
        """Comprehension quality is calculated and set."""
        responses = [
            json.dumps({"summary": "Full summary", "main_argument": "Thesis"}),
            json.dumps(
                {
                    "concepts": [
                        {"name": f"C{i}", "definition": f"D{i}"} for i in range(10)
                    ]
                }
            ),
            json.dumps({"principles": [{"statement": f"P{i}"} for i in range(15)]}),
            json.dumps(
                {
                    "techniques": [
                        {"name": f"T{i}", "description": f"D{i}"} for i in range(7)
                    ]
                }
            ),
            json.dumps({"examples": [{"work_title": f"E{i}"} for i in range(7)]}),
        ]
        mock_llm = AsyncMock(side_effect=responses)
        config = make_config(thorough_mode=False, deduplication_enabled=False)
        doc = make_document()
        chunks = [make_chunk()]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            result = await eng.comprehend(doc, chunks)

        assert result.comprehension_quality > 0.0
        assert result.study_completed_at is not None

    @pytest.mark.asyncio
    async def test_thorough_mode_chosen_when_chapters_present(self, mock_tracker):
        """Thorough mode is used when config.thorough_mode=True and chapters exist."""
        chapter_response = json.dumps(
            {
                "chapter_summary": {"summary": "Ch summary", "key_points": []},
                "concepts": [],
                "principles": [],
                "techniques": [],
                "examples": [],
            }
        )
        mock_llm = AsyncMock(return_value=chapter_response)
        chapters = [ChapterInfo(number=1, title="Ch1", start_page=1, end_page=20)]
        config = make_config(thorough_mode=True, deduplication_enabled=False)
        doc = make_document(chapters=chapters)
        chunks = [make_chunk(chapter="Ch1")]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            result = await eng.comprehend(doc, chunks)

        # Thorough mode: 1 summary + 1 chapter = 2 LLM calls (not 5 for sampled)
        assert mock_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_sampled_mode_used_when_no_chapters(self, mock_tracker):
        """Sampled mode is used when thorough_mode=True but no chapters exist."""
        mock_llm = AsyncMock(return_value='{"summary": "test"}')
        config = make_config(thorough_mode=True, deduplication_enabled=False)
        doc = make_document(chapters=[])  # No chapters
        chunks = [make_chunk()]

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            await eng.comprehend(doc, chunks)

        # Falls back to sampled mode: 5 calls
        assert mock_llm.call_count == 5

    @pytest.mark.asyncio
    async def test_voice_announces_quality(self, mock_tracker):
        """Voice callback completion message includes quality assessment."""
        responses = [
            json.dumps({"summary": "Full summary", "main_argument": "Thesis"}),
            json.dumps(
                {
                    "concepts": [
                        {"name": f"C{i}", "definition": f"D{i}"} for i in range(10)
                    ]
                }
            ),
            json.dumps({"principles": [{"statement": f"P{i}"} for i in range(15)]}),
            json.dumps(
                {
                    "techniques": [
                        {"name": f"T{i}", "description": f"D{i}"} for i in range(7)
                    ]
                }
            ),
            json.dumps({"examples": [{"work_title": f"E{i}"} for i in range(7)]}),
        ]
        mock_llm = AsyncMock(side_effect=responses)
        config = make_config(
            thorough_mode=False,
            deduplication_enabled=False,
            voice_announce_complete=True,
        )
        doc = make_document()
        chunks = [make_chunk()]
        voice_cb = MagicMock()

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            await eng.comprehend(doc, chunks, voice_callback=voice_cb)

        # The last voice call should include a quality message
        last_voice_msg = voice_cb.call_args_list[-1][0][0]
        assert any(
            word in last_voice_msg.lower()
            for word in ["quality", "comprehension", "finished"]
        )

    @pytest.mark.asyncio
    async def test_no_voice_start_when_disabled(self, mock_tracker):
        """Voice start announcement is skipped when voice_announce_start=False."""
        mock_llm = AsyncMock(return_value='{"summary": "test"}')
        config = make_config(
            thorough_mode=False,
            deduplication_enabled=False,
            voice_announce_start=False,
            voice_announce_complete=False,
        )
        doc = make_document()
        chunks = [make_chunk()]
        voice_cb = MagicMock()

        with patch(
            "documents.understanding.comprehension.get_job_tracker",
            return_value=mock_tracker,
        ):
            eng = BookComprehensionEngine(llm_complete=mock_llm, config=config)
            await eng.comprehend(doc, chunks, voice_callback=voice_cb)

        # Voice calls should not include start/complete announcements
        all_msgs = [call[0][0] for call in voice_cb.call_args_list]
        start_msgs = [m for m in all_msgs if "starting to study" in m.lower()]
        complete_msgs = [m for m in all_msgs if "finished studying" in m.lower()]
        assert len(start_msgs) == 0
        assert len(complete_msgs) == 0
