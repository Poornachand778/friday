"""
Comprehensive tests for documents/understanding/models.py
=========================================================

Tests cover all enums, dataclass creation with defaults and custom values,
to_dict() serialization, to_response() formatting, UUID generation,
property methods, and edge cases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import uuid
from datetime import datetime

import pytest

from documents.understanding.models import (
    BookExample,
    BookUnderstanding,
    BrainstormIdea,
    BrainstormResult,
    ChapterSummary,
    Concept,
    ConfidenceLevel,
    Inspiration,
    KnowledgeType,
    MentorAnalysis,
    Principle,
    RuleCheck,
    RuleCheckResult,
    Technique,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestKnowledgeType:
    """Tests for KnowledgeType enum."""

    def test_concept_value(self):
        assert KnowledgeType.CONCEPT == "concept"
        assert KnowledgeType.CONCEPT.value == "concept"

    def test_principle_value(self):
        assert KnowledgeType.PRINCIPLE == "principle"
        assert KnowledgeType.PRINCIPLE.value == "principle"

    def test_technique_value(self):
        assert KnowledgeType.TECHNIQUE == "technique"
        assert KnowledgeType.TECHNIQUE.value == "technique"

    def test_example_value(self):
        assert KnowledgeType.EXAMPLE == "example"
        assert KnowledgeType.EXAMPLE.value == "example"

    def test_framework_value(self):
        assert KnowledgeType.FRAMEWORK == "framework"
        assert KnowledgeType.FRAMEWORK.value == "framework"

    def test_warning_value(self):
        assert KnowledgeType.WARNING == "warning"
        assert KnowledgeType.WARNING.value == "warning"

    def test_all_members_present(self):
        members = set(KnowledgeType)
        assert len(members) == 6
        expected = {
            "concept",
            "principle",
            "technique",
            "example",
            "framework",
            "warning",
        }
        assert {m.value for m in members} == expected

    def test_is_str_subclass(self):
        assert isinstance(KnowledgeType.CONCEPT, str)

    def test_string_comparison(self):
        assert KnowledgeType.CONCEPT == "concept"
        assert KnowledgeType.PRINCIPLE != "concept"


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_absolute_value(self):
        assert ConfidenceLevel.ABSOLUTE == "absolute"
        assert ConfidenceLevel.ABSOLUTE.value == "absolute"

    def test_strong_value(self):
        assert ConfidenceLevel.STRONG == "strong"
        assert ConfidenceLevel.STRONG.value == "strong"

    def test_moderate_value(self):
        assert ConfidenceLevel.MODERATE == "moderate"
        assert ConfidenceLevel.MODERATE.value == "moderate"

    def test_suggestion_value(self):
        assert ConfidenceLevel.SUGGESTION == "suggestion"
        assert ConfidenceLevel.SUGGESTION.value == "suggestion"

    def test_all_members_present(self):
        members = set(ConfidenceLevel)
        assert len(members) == 4
        expected = {"absolute", "strong", "moderate", "suggestion"}
        assert {m.value for m in members} == expected

    def test_is_str_subclass(self):
        assert isinstance(ConfidenceLevel.ABSOLUTE, str)


# =============================================================================
# Concept Tests
# =============================================================================


class TestConcept:
    """Tests for the Concept dataclass."""

    def test_defaults(self):
        c = Concept()
        assert c.name == ""
        assert c.definition == ""
        assert c.importance == ""
        assert c.source_document_id == ""
        assert c.source_pages == ""
        assert c.related_concepts == []
        assert c.parent_concept is None
        assert c.sub_concepts == []
        assert c.synonyms == []
        assert c.domain == "screenwriting"
        assert isinstance(c.extracted_at, datetime)
        assert c.confidence == 0.9

    def test_uuid_generated(self):
        c = Concept()
        assert isinstance(c.id, str)
        # Should be a valid UUID
        uuid.UUID(c.id)

    def test_unique_ids(self):
        c1 = Concept()
        c2 = Concept()
        assert c1.id != c2.id

    def test_custom_values(self):
        dt = datetime(2025, 1, 15, 10, 30)
        c = Concept(
            id="custom-id",
            name="Inciting Incident",
            definition="The event that sets the story in motion",
            importance="Without it, there is no story",
            source_document_id="doc-123",
            source_pages="pp. 45-52",
            related_concepts=["Turning Point", "Climax"],
            parent_concept="Story Structure",
            sub_concepts=["Personal Inciting Incident", "Global Inciting Incident"],
            synonyms=["Catalyst", "Call to Adventure"],
            domain="narrative",
            extracted_at=dt,
            confidence=0.95,
        )
        assert c.id == "custom-id"
        assert c.name == "Inciting Incident"
        assert c.definition == "The event that sets the story in motion"
        assert c.importance == "Without it, there is no story"
        assert c.source_document_id == "doc-123"
        assert c.source_pages == "pp. 45-52"
        assert c.related_concepts == ["Turning Point", "Climax"]
        assert c.parent_concept == "Story Structure"
        assert c.sub_concepts == [
            "Personal Inciting Incident",
            "Global Inciting Incident",
        ]
        assert c.synonyms == ["Catalyst", "Call to Adventure"]
        assert c.domain == "narrative"
        assert c.extracted_at == dt
        assert c.confidence == 0.95

    def test_to_dict_all_fields(self):
        dt = datetime(2025, 6, 1, 12, 0, 0)
        c = Concept(
            id="c1",
            name="Theme",
            definition="Central idea",
            importance="Guides the story",
            source_document_id="doc-1",
            source_pages="p. 10",
            related_concepts=["Motif"],
            parent_concept="Narrative",
            sub_concepts=["Sub-theme"],
            synonyms=["Central Idea"],
            domain="writing",
            extracted_at=dt,
            confidence=0.85,
        )
        d = c.to_dict()
        assert d["id"] == "c1"
        assert d["name"] == "Theme"
        assert d["definition"] == "Central idea"
        assert d["importance"] == "Guides the story"
        assert d["source_document_id"] == "doc-1"
        assert d["source_pages"] == "p. 10"
        assert d["related_concepts"] == ["Motif"]
        assert d["parent_concept"] == "Narrative"
        assert d["sub_concepts"] == ["Sub-theme"]
        assert d["synonyms"] == ["Central Idea"]
        assert d["domain"] == "writing"
        assert d["extracted_at"] == dt.isoformat()
        assert d["confidence"] == 0.85

    def test_to_dict_datetime_serialization(self):
        dt = datetime(2025, 3, 15, 8, 30, 45)
        c = Concept(extracted_at=dt)
        d = c.to_dict()
        assert d["extracted_at"] == "2025-03-15T08:30:45"

    def test_to_dict_returns_dict(self):
        c = Concept()
        assert isinstance(c.to_dict(), dict)

    def test_to_dict_keys(self):
        c = Concept()
        expected_keys = {
            "id",
            "name",
            "definition",
            "importance",
            "source_document_id",
            "source_pages",
            "related_concepts",
            "parent_concept",
            "sub_concepts",
            "synonyms",
            "domain",
            "extracted_at",
            "confidence",
        }
        assert set(c.to_dict().keys()) == expected_keys

    def test_empty_lists_not_shared(self):
        c1 = Concept()
        c2 = Concept()
        c1.related_concepts.append("test")
        assert c2.related_concepts == []


# =============================================================================
# Principle Tests
# =============================================================================


class TestPrinciple:
    """Tests for the Principle dataclass."""

    def test_defaults(self):
        p = Principle()
        assert p.statement == ""
        assert p.rationale == ""
        assert p.source_document_id == ""
        assert p.source_page == ""
        assert p.confidence_level == ConfidenceLevel.STRONG
        assert p.applies_to == []
        assert p.exceptions == []
        assert p.prerequisites == []
        assert p.related_concepts == []
        assert p.related_techniques == []
        assert p.checkable is True
        assert p.check_question == ""
        assert isinstance(p.extracted_at, datetime)

    def test_uuid_generated(self):
        p = Principle()
        uuid.UUID(p.id)

    def test_unique_ids(self):
        p1 = Principle()
        p2 = Principle()
        assert p1.id != p2.id

    def test_custom_values(self):
        dt = datetime(2025, 2, 1, 9, 0)
        p = Principle(
            id="p-custom",
            statement="Every scene must have a turning point",
            rationale="Without turning points, scenes are static",
            source_document_id="doc-mckee",
            source_page="p. 233",
            confidence_level=ConfidenceLevel.ABSOLUTE,
            applies_to=["scenes", "sequences"],
            exceptions=["montage sequences"],
            prerequisites=["understanding of scene structure"],
            related_concepts=["Turning Point", "Scene"],
            related_techniques=["Scene Reversal"],
            checkable=True,
            check_question="Does your scene have a turning point?",
            extracted_at=dt,
        )
        assert p.id == "p-custom"
        assert p.statement == "Every scene must have a turning point"
        assert p.confidence_level == ConfidenceLevel.ABSOLUTE
        assert p.applies_to == ["scenes", "sequences"]
        assert p.exceptions == ["montage sequences"]
        assert p.prerequisites == ["understanding of scene structure"]
        assert p.checkable is True
        assert p.check_question == "Does your scene have a turning point?"
        assert p.extracted_at == dt

    def test_to_dict_all_fields(self):
        dt = datetime(2025, 5, 10, 14, 0)
        p = Principle(
            id="p1",
            statement="Show, don't tell",
            rationale="Visual medium",
            source_document_id="doc-x",
            source_page="p. 50",
            confidence_level=ConfidenceLevel.MODERATE,
            applies_to=["dialogue"],
            exceptions=["exposition scenes"],
            prerequisites=["basic storytelling"],
            related_concepts=["Subtext"],
            related_techniques=["Visual Storytelling"],
            checkable=False,
            check_question="",
            extracted_at=dt,
        )
        d = p.to_dict()
        assert d["id"] == "p1"
        assert d["statement"] == "Show, don't tell"
        assert d["rationale"] == "Visual medium"
        assert d["source_document_id"] == "doc-x"
        assert d["source_page"] == "p. 50"
        assert d["confidence_level"] == "moderate"
        assert d["applies_to"] == ["dialogue"]
        assert d["exceptions"] == ["exposition scenes"]
        assert d["prerequisites"] == ["basic storytelling"]
        assert d["related_concepts"] == ["Subtext"]
        assert d["related_techniques"] == ["Visual Storytelling"]
        assert d["checkable"] is False
        assert d["check_question"] == ""
        assert d["extracted_at"] == dt.isoformat()

    def test_to_dict_confidence_level_serialization(self):
        for level in ConfidenceLevel:
            p = Principle(confidence_level=level)
            d = p.to_dict()
            assert d["confidence_level"] == level.value

    def test_to_dict_keys(self):
        p = Principle()
        expected_keys = {
            "id",
            "statement",
            "rationale",
            "source_document_id",
            "source_page",
            "confidence_level",
            "applies_to",
            "exceptions",
            "prerequisites",
            "related_concepts",
            "related_techniques",
            "checkable",
            "check_question",
            "extracted_at",
        }
        assert set(p.to_dict().keys()) == expected_keys


# =============================================================================
# Technique Tests
# =============================================================================


class TestTechnique:
    """Tests for the Technique dataclass."""

    def test_defaults(self):
        t = Technique()
        assert t.name == ""
        assert t.description == ""
        assert t.steps == []
        assert t.source_document_id == ""
        assert t.source_page == ""
        assert t.use_cases == []
        assert t.when_to_use == ""
        assert t.when_not_to_use == ""
        assert t.example_films == []
        assert t.example_description == ""
        assert t.related_concepts == []
        assert t.related_principles == []
        assert t.alternative_techniques == []
        assert t.difficulty == "intermediate"
        assert isinstance(t.extracted_at, datetime)

    def test_uuid_generated(self):
        t = Technique()
        uuid.UUID(t.id)

    def test_custom_values(self):
        dt = datetime(2025, 4, 20, 16, 0)
        t = Technique(
            id="t-custom",
            name="The Slow Reveal",
            description="Gradually revealing information",
            steps=["Set up mystery", "Drop hints", "Reveal"],
            source_document_id="doc-tension",
            source_page="p. 88",
            use_cases=["building tension", "mystery"],
            when_to_use="When you need to build suspense",
            when_not_to_use="In fast-paced action scenes",
            example_films=["The Sixth Sense", "Usual Suspects"],
            example_description="Bruce Willis reveals to be dead",
            related_concepts=["Suspense", "Mystery"],
            related_principles=["Information Control"],
            alternative_techniques=["Red Herring"],
            difficulty="advanced",
            extracted_at=dt,
        )
        assert t.name == "The Slow Reveal"
        assert t.steps == ["Set up mystery", "Drop hints", "Reveal"]
        assert t.difficulty == "advanced"
        assert t.example_films == ["The Sixth Sense", "Usual Suspects"]

    def test_to_dict_all_fields(self):
        dt = datetime(2025, 7, 1, 10, 0)
        t = Technique(
            id="t1",
            name="Foreshadowing",
            description="Planting clues early",
            steps=["Identify payoff", "Plant seed"],
            source_document_id="doc-y",
            source_page="p. 120",
            use_cases=["drama"],
            when_to_use="When a later reveal needs setup",
            when_not_to_use="Comedy one-liners",
            example_films=["The Shawshank Redemption"],
            example_description="The poster and rock hammer",
            related_concepts=["Setup/Payoff"],
            related_principles=["Chekhov's Gun"],
            alternative_techniques=["Flash Forward"],
            difficulty="beginner",
            extracted_at=dt,
        )
        d = t.to_dict()
        assert d["id"] == "t1"
        assert d["name"] == "Foreshadowing"
        assert d["description"] == "Planting clues early"
        assert d["steps"] == ["Identify payoff", "Plant seed"]
        assert d["source_document_id"] == "doc-y"
        assert d["source_page"] == "p. 120"
        assert d["use_cases"] == ["drama"]
        assert d["when_to_use"] == "When a later reveal needs setup"
        assert d["when_not_to_use"] == "Comedy one-liners"
        assert d["example_films"] == ["The Shawshank Redemption"]
        assert d["example_description"] == "The poster and rock hammer"
        assert d["related_concepts"] == ["Setup/Payoff"]
        assert d["related_principles"] == ["Chekhov's Gun"]
        assert d["alternative_techniques"] == ["Flash Forward"]
        assert d["difficulty"] == "beginner"
        assert d["extracted_at"] == dt.isoformat()

    def test_to_dict_keys(self):
        t = Technique()
        expected_keys = {
            "id",
            "name",
            "description",
            "steps",
            "source_document_id",
            "source_page",
            "use_cases",
            "when_to_use",
            "when_not_to_use",
            "example_films",
            "example_description",
            "related_concepts",
            "related_principles",
            "alternative_techniques",
            "difficulty",
            "extracted_at",
        }
        assert set(t.to_dict().keys()) == expected_keys


# =============================================================================
# BookExample Tests
# =============================================================================


class TestBookExample:
    """Tests for the BookExample dataclass."""

    def test_defaults(self):
        b = BookExample()
        assert b.work_title == ""
        assert b.work_type == "film"
        assert b.scene_or_section == ""
        assert b.source_document_id == ""
        assert b.source_page == ""
        assert b.description == ""
        assert b.lesson == ""
        assert b.what_works == ""
        assert b.demonstrates_concept == []
        assert b.demonstrates_technique == []
        assert b.demonstrates_principle == []
        assert b.situation_type == []
        assert b.emotional_beat == ""
        assert isinstance(b.extracted_at, datetime)

    def test_uuid_generated(self):
        b = BookExample()
        uuid.UUID(b.id)

    def test_custom_values(self):
        dt = datetime(2025, 8, 5, 11, 0)
        b = BookExample(
            id="ex-1",
            work_title="12 Angry Men",
            work_type="film",
            scene_or_section="Final vote scene",
            source_document_id="doc-court",
            source_page="p. 75",
            description="Jurors change their votes one by one",
            lesson="How to build consensus through logic and empathy",
            what_works="Each character has a unique reason to change",
            demonstrates_concept=["Group Dynamics"],
            demonstrates_technique=["The Slow Reveal"],
            demonstrates_principle=["Character Arc"],
            situation_type=["courtroom", "tension"],
            emotional_beat="confrontation",
            extracted_at=dt,
        )
        assert b.work_title == "12 Angry Men"
        assert b.work_type == "film"
        assert b.demonstrates_concept == ["Group Dynamics"]
        assert b.emotional_beat == "confrontation"

    def test_to_dict_all_fields(self):
        dt = datetime(2025, 9, 1, 8, 0)
        b = BookExample(
            id="ex-2",
            work_title="A Few Good Men",
            work_type="play",
            scene_or_section="You can't handle the truth",
            source_document_id="doc-z",
            source_page="p. 150",
            description="Courtroom confrontation",
            lesson="Dramatic irony in confession",
            what_works="Building pressure until it breaks",
            demonstrates_concept=["Dramatic Irony"],
            demonstrates_technique=["Pressure Build"],
            demonstrates_principle=["Truth through conflict"],
            situation_type=["courtroom"],
            emotional_beat="revelation",
            extracted_at=dt,
        )
        d = b.to_dict()
        assert d["id"] == "ex-2"
        assert d["work_title"] == "A Few Good Men"
        assert d["work_type"] == "play"
        assert d["scene_or_section"] == "You can't handle the truth"
        assert d["source_document_id"] == "doc-z"
        assert d["source_page"] == "p. 150"
        assert d["description"] == "Courtroom confrontation"
        assert d["lesson"] == "Dramatic irony in confession"
        assert d["what_works"] == "Building pressure until it breaks"
        assert d["demonstrates_concept"] == ["Dramatic Irony"]
        assert d["demonstrates_technique"] == ["Pressure Build"]
        assert d["demonstrates_principle"] == ["Truth through conflict"]
        assert d["situation_type"] == ["courtroom"]
        assert d["emotional_beat"] == "revelation"
        assert d["extracted_at"] == dt.isoformat()

    def test_to_dict_keys(self):
        b = BookExample()
        expected_keys = {
            "id",
            "work_title",
            "work_type",
            "scene_or_section",
            "source_document_id",
            "source_page",
            "description",
            "lesson",
            "what_works",
            "demonstrates_concept",
            "demonstrates_technique",
            "demonstrates_principle",
            "situation_type",
            "emotional_beat",
            "extracted_at",
        }
        assert set(b.to_dict().keys()) == expected_keys


# =============================================================================
# ChapterSummary Tests
# =============================================================================


class TestChapterSummary:
    """Tests for the ChapterSummary dataclass."""

    def test_required_fields(self):
        ch = ChapterSummary(number=1, title="Introduction", summary="An overview.")
        assert ch.number == 1
        assert ch.title == "Introduction"
        assert ch.summary == "An overview."

    def test_defaults(self):
        ch = ChapterSummary(number=1, title="Ch1", summary="Sum")
        assert ch.key_points == []
        assert ch.concepts_introduced == []
        assert ch.principles_taught == []
        assert ch.page_range == ""

    def test_custom_values(self):
        ch = ChapterSummary(
            number=3,
            title="Scene Design",
            summary="How to construct a scene",
            key_points=["Turning point", "Beats"],
            concepts_introduced=["Beat", "Scene"],
            principles_taught=["Every scene must turn"],
            page_range="pp. 55-80",
        )
        assert ch.number == 3
        assert ch.title == "Scene Design"
        assert ch.key_points == ["Turning point", "Beats"]
        assert ch.page_range == "pp. 55-80"

    def test_to_dict(self):
        ch = ChapterSummary(
            number=2,
            title="Structure",
            summary="Story structure explained",
            key_points=["Three acts"],
            concepts_introduced=["Act"],
            principles_taught=["Rising action"],
            page_range="pp. 20-40",
        )
        d = ch.to_dict()
        assert d["number"] == 2
        assert d["title"] == "Structure"
        assert d["summary"] == "Story structure explained"
        assert d["key_points"] == ["Three acts"]
        assert d["concepts_introduced"] == ["Act"]
        assert d["principles_taught"] == ["Rising action"]
        assert d["page_range"] == "pp. 20-40"

    def test_to_dict_keys(self):
        ch = ChapterSummary(number=1, title="T", summary="S")
        expected_keys = {
            "number",
            "title",
            "summary",
            "key_points",
            "concepts_introduced",
            "principles_taught",
            "page_range",
        }
        assert set(ch.to_dict().keys()) == expected_keys


# =============================================================================
# BookUnderstanding Tests
# =============================================================================


class TestBookUnderstanding:
    """Tests for the BookUnderstanding dataclass."""

    def test_defaults(self):
        bu = BookUnderstanding()
        assert bu.document_id == ""
        assert bu.title == ""
        assert bu.author == ""
        assert bu.summary == ""
        assert bu.main_argument == ""
        assert bu.target_audience == ""
        assert bu.chapters == []
        assert bu.concepts == []
        assert bu.principles == []
        assert bu.techniques == []
        assert bu.examples == []
        assert bu.agrees_with == {}
        assert bu.disagrees_with == {}
        assert bu.extends == {}
        assert bu.domains == []
        assert bu.study_completed_at is None
        assert bu.comprehension_quality == 0.0

    def test_uuid_generated(self):
        bu = BookUnderstanding()
        uuid.UUID(bu.id)

    def test_custom_values(self):
        dt = datetime(2025, 10, 1, 12, 0)
        bu = BookUnderstanding(
            id="bu-1",
            document_id="doc-mckee",
            title="Story",
            author="Robert McKee",
            summary="A comprehensive guide to screenwriting.",
            main_argument="Story is about structure, not formula.",
            target_audience="Screenwriters",
            domains=["screenwriting", "narrative"],
            study_completed_at=dt,
            comprehension_quality=0.85,
        )
        assert bu.id == "bu-1"
        assert bu.title == "Story"
        assert bu.author == "Robert McKee"
        assert bu.study_completed_at == dt
        assert bu.comprehension_quality == 0.85

    def test_total_knowledge_items_empty(self):
        bu = BookUnderstanding()
        assert bu.total_knowledge_items == 0

    def test_total_knowledge_items_with_items(self):
        bu = BookUnderstanding(
            concepts=[Concept(), Concept(), Concept()],
            principles=[Principle(), Principle()],
            techniques=[Technique()],
            examples=[BookExample(), BookExample()],
        )
        assert bu.total_knowledge_items == 8

    def test_total_knowledge_items_only_concepts(self):
        bu = BookUnderstanding(concepts=[Concept()])
        assert bu.total_knowledge_items == 1

    def test_total_knowledge_items_only_principles(self):
        bu = BookUnderstanding(principles=[Principle(), Principle()])
        assert bu.total_knowledge_items == 2

    def test_total_knowledge_items_only_techniques(self):
        bu = BookUnderstanding(techniques=[Technique(), Technique(), Technique()])
        assert bu.total_knowledge_items == 3

    def test_total_knowledge_items_only_examples(self):
        bu = BookUnderstanding(examples=[BookExample()])
        assert bu.total_knowledge_items == 1

    def test_to_dict_basic(self):
        bu = BookUnderstanding(
            id="bu-2",
            document_id="doc-2",
            title="Save the Cat",
            author="Blake Snyder",
        )
        d = bu.to_dict()
        assert d["id"] == "bu-2"
        assert d["document_id"] == "doc-2"
        assert d["title"] == "Save the Cat"
        assert d["author"] == "Blake Snyder"
        assert d["chapters"] == []
        assert d["concepts"] == []
        assert d["study_completed_at"] is None

    def test_to_dict_with_study_completed_at(self):
        dt = datetime(2025, 11, 15, 9, 0, 0)
        bu = BookUnderstanding(study_completed_at=dt)
        d = bu.to_dict()
        assert d["study_completed_at"] == "2025-11-15T09:00:00"

    def test_to_dict_study_completed_at_none(self):
        bu = BookUnderstanding(study_completed_at=None)
        d = bu.to_dict()
        assert d["study_completed_at"] is None

    def test_to_dict_nested_chapters(self):
        ch1 = ChapterSummary(number=1, title="Intro", summary="Intro summary")
        ch2 = ChapterSummary(number=2, title="Body", summary="Body summary")
        bu = BookUnderstanding(chapters=[ch1, ch2])
        d = bu.to_dict()
        assert len(d["chapters"]) == 2
        assert d["chapters"][0]["number"] == 1
        assert d["chapters"][0]["title"] == "Intro"
        assert d["chapters"][1]["number"] == 2

    def test_to_dict_nested_concepts(self):
        c = Concept(id="c-nested", name="Conflict")
        bu = BookUnderstanding(concepts=[c])
        d = bu.to_dict()
        assert len(d["concepts"]) == 1
        assert d["concepts"][0]["id"] == "c-nested"
        assert d["concepts"][0]["name"] == "Conflict"

    def test_to_dict_nested_principles(self):
        p = Principle(id="p-nested", statement="Conflict drives story")
        bu = BookUnderstanding(principles=[p])
        d = bu.to_dict()
        assert len(d["principles"]) == 1
        assert d["principles"][0]["id"] == "p-nested"
        assert d["principles"][0]["statement"] == "Conflict drives story"

    def test_to_dict_nested_techniques(self):
        t = Technique(id="t-nested", name="Plant and Payoff")
        bu = BookUnderstanding(techniques=[t])
        d = bu.to_dict()
        assert len(d["techniques"]) == 1
        assert d["techniques"][0]["id"] == "t-nested"
        assert d["techniques"][0]["name"] == "Plant and Payoff"

    def test_to_dict_nested_examples(self):
        ex = BookExample(id="ex-nested", work_title="Casablanca")
        bu = BookUnderstanding(examples=[ex])
        d = bu.to_dict()
        assert len(d["examples"]) == 1
        assert d["examples"][0]["id"] == "ex-nested"
        assert d["examples"][0]["work_title"] == "Casablanca"

    def test_to_dict_cross_book_relationships(self):
        bu = BookUnderstanding(
            agrees_with={"book-a": ["Theme", "Structure"]},
            disagrees_with={"book-b": ["Pacing"]},
            extends={"book-c": ["Character Arc"]},
        )
        d = bu.to_dict()
        assert d["agrees_with"] == {"book-a": ["Theme", "Structure"]}
        assert d["disagrees_with"] == {"book-b": ["Pacing"]}
        assert d["extends"] == {"book-c": ["Character Arc"]}

    def test_to_dict_keys(self):
        bu = BookUnderstanding()
        expected_keys = {
            "id",
            "document_id",
            "title",
            "author",
            "summary",
            "main_argument",
            "target_audience",
            "chapters",
            "concepts",
            "principles",
            "techniques",
            "examples",
            "agrees_with",
            "disagrees_with",
            "extends",
            "domains",
            "study_completed_at",
            "comprehension_quality",
        }
        assert set(bu.to_dict().keys()) == expected_keys

    def test_to_dict_full_nested(self):
        """Verify to_dict with all nested types populated calls their to_dict."""
        dt = datetime(2025, 12, 25, 0, 0)
        ch = ChapterSummary(number=1, title="Ch1", summary="S1")
        c = Concept(id="c-full", name="Protagonist", extracted_at=dt)
        p = Principle(id="p-full", statement="Show don't tell", extracted_at=dt)
        t = Technique(id="t-full", name="Montage", extracted_at=dt)
        ex = BookExample(id="ex-full", work_title="Rocky", extracted_at=dt)

        bu = BookUnderstanding(
            id="bu-full",
            chapters=[ch],
            concepts=[c],
            principles=[p],
            techniques=[t],
            examples=[ex],
            study_completed_at=dt,
        )
        d = bu.to_dict()

        # All nested items should be dicts (not dataclass instances)
        assert isinstance(d["chapters"][0], dict)
        assert isinstance(d["concepts"][0], dict)
        assert isinstance(d["principles"][0], dict)
        assert isinstance(d["techniques"][0], dict)
        assert isinstance(d["examples"][0], dict)

        # Verify nested datetime serialization
        assert d["concepts"][0]["extracted_at"] == dt.isoformat()
        assert d["principles"][0]["extracted_at"] == dt.isoformat()
        assert d["techniques"][0]["extracted_at"] == dt.isoformat()
        assert d["examples"][0]["extracted_at"] == dt.isoformat()
        assert d["study_completed_at"] == dt.isoformat()


# =============================================================================
# RuleCheck Tests
# =============================================================================


class TestRuleCheck:
    """Tests for the RuleCheck dataclass."""

    def test_creation(self):
        p = Principle(statement="Use subtext")
        rc = RuleCheck(
            principle=p,
            status="followed",
            evidence="The dialogue has layers of meaning",
        )
        assert rc.principle is p
        assert rc.status == "followed"
        assert rc.evidence == "The dialogue has layers of meaning"
        assert rc.suggestion == ""

    def test_creation_with_suggestion(self):
        p = Principle(statement="Every scene must turn")
        rc = RuleCheck(
            principle=p,
            status="violated",
            evidence="Scene 3 has no turning point",
            suggestion="Add a reversal at the midpoint of scene 3",
        )
        assert rc.status == "violated"
        assert rc.suggestion == "Add a reversal at the midpoint of scene 3"

    def test_all_statuses(self):
        p = Principle()
        for status in ["followed", "violated", "unclear", "not_applicable"]:
            rc = RuleCheck(principle=p, status=status, evidence="test")
            assert rc.status == status


# =============================================================================
# RuleCheckResult Tests
# =============================================================================


class TestRuleCheckResult:
    """Tests for the RuleCheckResult dataclass."""

    def test_creation_required_fields(self):
        rcr = RuleCheckResult(
            document_ids=["doc-1", "doc-2"],
            scene_or_work="Act 1 Scene 3",
        )
        assert rcr.document_ids == ["doc-1", "doc-2"]
        assert rcr.scene_or_work == "Act 1 Scene 3"

    def test_defaults(self):
        rcr = RuleCheckResult(document_ids=[], scene_or_work="")
        assert rcr.rules_followed == []
        assert rcr.rules_violated == []
        assert rcr.rules_unclear == []
        assert rcr.rules_not_applicable == []
        assert rcr.overall_assessment == ""
        assert rcr.priority_fixes == []

    def test_with_rule_checks(self):
        p1 = Principle(statement="Rule 1")
        p2 = Principle(statement="Rule 2")
        rc_followed = RuleCheck(principle=p1, status="followed", evidence="Evidence 1")
        rc_violated = RuleCheck(
            principle=p2,
            status="violated",
            evidence="Evidence 2",
            suggestion="Fix it",
        )
        rcr = RuleCheckResult(
            document_ids=["doc-1"],
            scene_or_work="Scene 5",
            rules_followed=[rc_followed],
            rules_violated=[rc_violated],
            overall_assessment="Mostly good, one issue",
            priority_fixes=["Fix scene turning point"],
        )
        assert len(rcr.rules_followed) == 1
        assert len(rcr.rules_violated) == 1
        assert rcr.overall_assessment == "Mostly good, one issue"
        assert rcr.priority_fixes == ["Fix scene turning point"]


# =============================================================================
# Inspiration Tests
# =============================================================================


class TestInspiration:
    """Tests for the Inspiration dataclass."""

    def test_creation(self):
        ex = BookExample(work_title="Chinatown")
        insp = Inspiration(
            example=ex,
            relevance_reason="Similar power dynamic",
            how_to_apply="Use the same reveal structure",
            source_book="Story by McKee",
        )
        assert insp.example is ex
        assert insp.relevance_reason == "Similar power dynamic"
        assert insp.how_to_apply == "Use the same reveal structure"
        assert insp.source_book == "Story by McKee"


# =============================================================================
# MentorAnalysis Tests
# =============================================================================


class TestMentorAnalysis:
    """Tests for the MentorAnalysis dataclass."""

    def test_creation_required_fields(self):
        ma = MentorAnalysis(
            user_input="A courtroom scene where the witness breaks down",
            active_books=["Story", "Screenplay"],
        )
        assert ma.user_input == "A courtroom scene where the witness breaks down"
        assert ma.active_books == ["Story", "Screenplay"]

    def test_defaults(self):
        ma = MentorAnalysis(user_input="test", active_books=[])
        assert ma.elements_present == []
        assert ma.strengths == []
        assert ma.elements_missing == []
        assert ma.potential_issues == []
        assert ma.relevant_principles == []
        assert ma.applicable_techniques == []
        assert ma.similar_examples == []
        assert ma.suggestions == []
        assert ma.questions_to_consider == []
        assert ma.book_agreements == []
        assert ma.book_disagreements == []

    def test_to_response_empty(self):
        ma = MentorAnalysis(user_input="test", active_books=[])
        response = ma.to_response()
        assert response == ""

    def test_to_response_with_strengths(self):
        ma = MentorAnalysis(
            user_input="test",
            active_books=[],
            strengths=["Strong dialogue", "Clear conflict"],
        )
        response = ma.to_response()
        assert "**What's working:**" in response
        assert "Strong dialogue" in response
        assert "Clear conflict" in response

    def test_to_response_with_elements_missing(self):
        ma = MentorAnalysis(
            user_input="test",
            active_books=[],
            elements_missing=["Turning point", "Stakes"],
        )
        response = ma.to_response()
        assert "**Consider adding:**" in response
        assert "Turning point" in response
        assert "Stakes" in response

    def test_to_response_with_principles(self):
        p1 = Principle(statement="Every scene must turn", source_page="p. 233")
        p2 = Principle(statement="Conflict is essential", source_page="p. 100")
        ma = MentorAnalysis(
            user_input="test",
            active_books=[],
            relevant_principles=[p1, p2],
        )
        response = ma.to_response()
        assert "**Relevant principles:**" in response
        assert "Every scene must turn" in response
        assert "p. 233" in response
        assert "Conflict is essential" in response
        assert "p. 100" in response

    def test_to_response_limits_principles_to_three(self):
        principles = [
            Principle(statement=f"Principle {i}", source_page=f"p. {i}")
            for i in range(5)
        ]
        ma = MentorAnalysis(
            user_input="test",
            active_books=[],
            relevant_principles=principles,
        )
        response = ma.to_response()
        assert "Principle 0" in response
        assert "Principle 1" in response
        assert "Principle 2" in response
        assert "Principle 3" not in response
        assert "Principle 4" not in response

    def test_to_response_with_similar_examples(self):
        ex = BookExample(work_title="12 Angry Men")
        insp = Inspiration(
            example=ex,
            relevance_reason="Similar jury dynamics",
            how_to_apply="Use gradual consensus building",
            source_book="Story",
        )
        ma = MentorAnalysis(
            user_input="test",
            active_books=[],
            similar_examples=[insp],
        )
        response = ma.to_response()
        assert "**Inspiration:**" in response
        assert "12 Angry Men" in response
        assert "Similar jury dynamics" in response

    def test_to_response_limits_examples_to_two(self):
        examples = []
        for i in range(4):
            ex = BookExample(work_title=f"Film {i}")
            insp = Inspiration(
                example=ex,
                relevance_reason=f"Reason {i}",
                how_to_apply="Apply it",
                source_book="Book",
            )
            examples.append(insp)
        ma = MentorAnalysis(
            user_input="test",
            active_books=[],
            similar_examples=examples,
        )
        response = ma.to_response()
        assert "Film 0" in response
        assert "Film 1" in response
        assert "Film 2" not in response
        assert "Film 3" not in response

    def test_to_response_with_suggestions(self):
        ma = MentorAnalysis(
            user_input="test",
            active_books=[],
            suggestions=["Add more conflict", "Develop the antagonist"],
        )
        response = ma.to_response()
        assert "**Suggestions:**" in response
        assert "Add more conflict" in response
        assert "Develop the antagonist" in response

    def test_to_response_limits_suggestions_to_three(self):
        suggestions = [f"Suggestion {i}" for i in range(5)]
        ma = MentorAnalysis(
            user_input="test",
            active_books=[],
            suggestions=suggestions,
        )
        response = ma.to_response()
        assert "Suggestion 0" in response
        assert "Suggestion 1" in response
        assert "Suggestion 2" in response
        assert "Suggestion 3" not in response

    def test_to_response_full(self):
        """Test to_response with all sections populated."""
        p = Principle(statement="Conflict drives story", source_page="p. 55")
        ex = BookExample(work_title="The Godfather")
        insp = Inspiration(
            example=ex,
            relevance_reason="Power struggle dynamic",
            how_to_apply="Mirror the escalation pattern",
            source_book="Story",
        )
        ma = MentorAnalysis(
            user_input="My courtroom scene",
            active_books=["Story"],
            strengths=["Great tension"],
            elements_missing=["Subtext"],
            relevant_principles=[p],
            similar_examples=[insp],
            suggestions=["Layer the dialogue"],
        )
        response = ma.to_response()
        # All sections present
        assert "**What's working:**" in response
        assert "**Consider adding:**" in response
        assert "**Relevant principles:**" in response
        assert "**Inspiration:**" in response
        assert "**Suggestions:**" in response
        # Sections separated by double newlines
        assert "\n\n" in response

    def test_to_response_partial_strengths_only(self):
        ma = MentorAnalysis(
            user_input="test",
            active_books=[],
            strengths=["Good pacing"],
        )
        response = ma.to_response()
        assert "**What's working:**" in response
        assert "Good pacing" in response
        # No other sections
        assert "**Consider adding:**" not in response
        assert "**Relevant principles:**" not in response
        assert "**Inspiration:**" not in response
        assert "**Suggestions:**" not in response

    def test_to_response_partial_missing_only(self):
        ma = MentorAnalysis(
            user_input="test",
            active_books=[],
            elements_missing=["Stakes"],
        )
        response = ma.to_response()
        assert "**Consider adding:**" in response
        assert "**What's working:**" not in response


# =============================================================================
# BrainstormIdea Tests
# =============================================================================


class TestBrainstormIdea:
    """Tests for the BrainstormIdea dataclass."""

    def test_creation_required_fields(self):
        bi = BrainstormIdea(
            idea="Use a ticking clock",
            rationale="Creates urgency",
            based_on=["Suspense Theory"],
        )
        assert bi.idea == "Use a ticking clock"
        assert bi.rationale == "Creates urgency"
        assert bi.based_on == ["Suspense Theory"]

    def test_defaults(self):
        bi = BrainstormIdea(idea="test", rationale="test", based_on=[])
        assert bi.source_inspiration is None
        assert bi.potential_risks == []

    def test_custom_values(self):
        bi = BrainstormIdea(
            idea="Unreliable narrator",
            rationale="Keeps audience guessing",
            based_on=["Narrative Theory", "Point of View"],
            source_inspiration="Gone Girl",
            potential_risks=["Can confuse audience", "Hard to execute"],
        )
        assert bi.source_inspiration == "Gone Girl"
        assert bi.potential_risks == ["Can confuse audience", "Hard to execute"]


# =============================================================================
# BrainstormResult Tests
# =============================================================================


class TestBrainstormResult:
    """Tests for the BrainstormResult dataclass."""

    def test_creation_required_fields(self):
        br = BrainstormResult(
            topic="How to open the courtroom scene",
            constraints=["Must be under 3 pages"],
            active_books=["Story"],
        )
        assert br.topic == "How to open the courtroom scene"
        assert br.constraints == ["Must be under 3 pages"]
        assert br.active_books == ["Story"]

    def test_defaults(self):
        br = BrainstormResult(topic="", constraints=[], active_books=[])
        assert br.ideas == []
        assert br.suggested_structure == ""
        assert br.concepts_applied == []
        assert br.techniques_suggested == []

    def test_to_response_empty_ideas(self):
        br = BrainstormResult(topic="Scene opening", constraints=[], active_books=[])
        response = br.to_response()
        assert "**Brainstorming: Scene opening**" in response
        # No idea entries
        assert "1." not in response

    def test_to_response_with_ideas(self):
        idea1 = BrainstormIdea(
            idea="Start with a surprising witness",
            rationale="Grabs audience attention immediately",
            based_on=["Hook Principle"],
        )
        idea2 = BrainstormIdea(
            idea="Open with the verdict, then flashback",
            rationale="Creates dramatic irony",
            based_on=["Non-linear Structure"],
        )
        br = BrainstormResult(
            topic="Courtroom opening",
            constraints=["Must hook audience"],
            active_books=["Story"],
            ideas=[idea1, idea2],
        )
        response = br.to_response()
        assert "**Brainstorming: Courtroom opening**" in response
        assert "1. **Start with a surprising witness**" in response
        assert "Why: Grabs audience attention immediately" in response
        assert "Based on: Hook Principle" in response
        assert "2. **Open with the verdict, then flashback**" in response
        assert "Based on: Non-linear Structure" in response

    def test_to_response_with_suggested_structure(self):
        br = BrainstormResult(
            topic="Plot structure",
            constraints=[],
            active_books=["Save the Cat"],
            suggested_structure="Follow the three-act structure with a midpoint twist",
        )
        response = br.to_response()
        assert "**Suggested approach:**" in response
        assert "Follow the three-act structure with a midpoint twist" in response

    def test_to_response_no_structure(self):
        br = BrainstormResult(
            topic="Plot structure",
            constraints=[],
            active_books=[],
            suggested_structure="",
        )
        response = br.to_response()
        assert "**Suggested approach:**" not in response

    def test_to_response_idea_without_based_on(self):
        idea = BrainstormIdea(
            idea="Ambiguous ending",
            rationale="Leaves audience thinking",
            based_on=[],
        )
        br = BrainstormResult(
            topic="Ending",
            constraints=[],
            active_books=[],
            ideas=[idea],
        )
        response = br.to_response()
        assert "1. **Ambiguous ending**" in response
        assert "Why: Leaves audience thinking" in response
        # "Based on" line should not appear for empty based_on
        assert "Based on:" not in response

    def test_to_response_idea_with_multiple_based_on(self):
        idea = BrainstormIdea(
            idea="Dual timeline",
            rationale="Shows cause and effect",
            based_on=["Parallel Narrative", "Contrast Technique", "Temporal Structure"],
        )
        br = BrainstormResult(
            topic="Structure",
            constraints=[],
            active_books=[],
            ideas=[idea],
        )
        response = br.to_response()
        assert (
            "Based on: Parallel Narrative, Contrast Technique, Temporal Structure"
            in response
        )


# =============================================================================
# UUID Uniqueness Tests
# =============================================================================


class TestUUIDGeneration:
    """Tests that UUID generation produces unique, valid UUIDs."""

    def test_concept_ids_unique_across_many(self):
        ids = {Concept().id for _ in range(50)}
        assert len(ids) == 50

    def test_principle_ids_unique_across_many(self):
        ids = {Principle().id for _ in range(50)}
        assert len(ids) == 50

    def test_technique_ids_unique_across_many(self):
        ids = {Technique().id for _ in range(50)}
        assert len(ids) == 50

    def test_book_example_ids_unique_across_many(self):
        ids = {BookExample().id for _ in range(50)}
        assert len(ids) == 50

    def test_book_understanding_ids_unique_across_many(self):
        ids = {BookUnderstanding().id for _ in range(50)}
        assert len(ids) == 50

    def test_all_ids_are_valid_uuid4(self):
        for cls in [Concept, Principle, Technique, BookExample, BookUnderstanding]:
            obj = cls() if cls != ChapterSummary else None
            if obj:
                parsed = uuid.UUID(obj.id)
                assert parsed.version == 4


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for all models."""

    def test_concept_with_empty_strings(self):
        c = Concept(name="", definition="", importance="")
        d = c.to_dict()
        assert d["name"] == ""
        assert d["definition"] == ""
        assert d["importance"] == ""

    def test_concept_with_none_parent(self):
        c = Concept(parent_concept=None)
        d = c.to_dict()
        assert d["parent_concept"] is None

    def test_principle_checkable_false(self):
        p = Principle(checkable=False)
        d = p.to_dict()
        assert d["checkable"] is False

    def test_technique_with_empty_steps(self):
        t = Technique(steps=[])
        d = t.to_dict()
        assert d["steps"] == []

    def test_book_example_with_empty_lists(self):
        b = BookExample(
            demonstrates_concept=[],
            demonstrates_technique=[],
            demonstrates_principle=[],
            situation_type=[],
        )
        d = b.to_dict()
        assert d["demonstrates_concept"] == []
        assert d["demonstrates_technique"] == []
        assert d["demonstrates_principle"] == []
        assert d["situation_type"] == []

    def test_chapter_summary_zero_number(self):
        ch = ChapterSummary(
            number=0, title="Prologue", summary="Before the story begins"
        )
        d = ch.to_dict()
        assert d["number"] == 0

    def test_book_understanding_empty_dicts(self):
        bu = BookUnderstanding(agrees_with={}, disagrees_with={}, extends={})
        d = bu.to_dict()
        assert d["agrees_with"] == {}
        assert d["disagrees_with"] == {}
        assert d["extends"] == {}

    def test_book_understanding_comprehension_quality_zero(self):
        bu = BookUnderstanding(comprehension_quality=0.0)
        d = bu.to_dict()
        assert d["comprehension_quality"] == 0.0

    def test_book_understanding_comprehension_quality_one(self):
        bu = BookUnderstanding(comprehension_quality=1.0)
        d = bu.to_dict()
        assert d["comprehension_quality"] == 1.0

    def test_concept_confidence_zero(self):
        c = Concept(confidence=0.0)
        d = c.to_dict()
        assert d["confidence"] == 0.0

    def test_concept_confidence_one(self):
        c = Concept(confidence=1.0)
        d = c.to_dict()
        assert d["confidence"] == 1.0

    def test_brainstorm_idea_none_source_inspiration(self):
        bi = BrainstormIdea(
            idea="test",
            rationale="test",
            based_on=[],
            source_inspiration=None,
        )
        assert bi.source_inspiration is None

    def test_rule_check_empty_suggestion(self):
        p = Principle()
        rc = RuleCheck(principle=p, status="followed", evidence="ok")
        assert rc.suggestion == ""

    def test_rule_check_result_empty_lists(self):
        rcr = RuleCheckResult(document_ids=[], scene_or_work="")
        assert rcr.rules_followed == []
        assert rcr.rules_violated == []
        assert rcr.rules_unclear == []
        assert rcr.rules_not_applicable == []
        assert rcr.priority_fixes == []

    def test_mentor_analysis_to_response_returns_string(self):
        ma = MentorAnalysis(user_input="x", active_books=[])
        assert isinstance(ma.to_response(), str)

    def test_brainstorm_result_to_response_returns_string(self):
        br = BrainstormResult(topic="x", constraints=[], active_books=[])
        assert isinstance(br.to_response(), str)

    def test_concept_lists_are_independent_instances(self):
        """Ensure default list factory creates independent lists."""
        c1 = Concept()
        c2 = Concept()
        c1.synonyms.append("test")
        c1.sub_concepts.append("sub")
        assert c2.synonyms == []
        assert c2.sub_concepts == []

    def test_principle_lists_are_independent_instances(self):
        p1 = Principle()
        p2 = Principle()
        p1.applies_to.append("scenes")
        assert p2.applies_to == []

    def test_technique_lists_are_independent_instances(self):
        t1 = Technique()
        t2 = Technique()
        t1.steps.append("step1")
        t1.use_cases.append("case1")
        assert t2.steps == []
        assert t2.use_cases == []

    def test_book_example_work_type_custom(self):
        b = BookExample(work_type="tv_episode")
        assert b.work_type == "tv_episode"
        d = b.to_dict()
        assert d["work_type"] == "tv_episode"

    def test_concept_domain_custom(self):
        c = Concept(domain="legal_drama")
        assert c.domain == "legal_drama"
        d = c.to_dict()
        assert d["domain"] == "legal_drama"

    def test_technique_difficulty_values(self):
        for diff in ["beginner", "intermediate", "advanced"]:
            t = Technique(difficulty=diff)
            assert t.difficulty == diff
            assert t.to_dict()["difficulty"] == diff
