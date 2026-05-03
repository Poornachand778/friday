"""
Tests for Mentor Engine
==========================

Tests the MentorEngine class which applies book knowledge to user's
creative work through scene analysis, brainstorming, rule-checking,
inspiration-finding, and comparison.

Run with: pytest tests/test_mentor_engine.py -v
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from documents.understanding.mentor import MentorEngine
from documents.understanding.models import (
    BookExample,
    BookUnderstanding,
    BrainstormIdea,
    BrainstormResult,
    Concept,
    ConfidenceLevel,
    Inspiration,
    MentorAnalysis,
    Principle,
    RuleCheck,
    RuleCheckResult,
    Technique,
)


# =========================================================================
# Helpers
# =========================================================================


def _run(coro):
    """Run async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_principle(statement="Every scene must turn", page="42"):
    """Create a test Principle."""
    return Principle(
        statement=statement,
        source_page=page,
        rationale="Fundamental storytelling rule",
        confidence_level=ConfidenceLevel.STRONG,
    )


def _make_technique(name="The Slow Reveal", description="Build tension gradually"):
    """Create a test Technique."""
    return Technique(
        name=name,
        description=description,
        source_page="55",
    )


def _make_example(
    title="12 Angry Men",
    scene="Jury deliberation",
    lesson="Shows power of a single dissenting voice",
    page="78",
):
    """Create a test BookExample."""
    return BookExample(
        work_title=title,
        scene_or_section=scene,
        description=f"Analysis of {title}",
        lesson=lesson,
        source_page=page,
    )


def _make_concept(name="Inciting Incident"):
    """Create a test Concept."""
    return Concept(name=name, definition=f"Definition of {name}")


def _make_book(
    book_id="book-1",
    title="Story",
    author="Robert McKee",
    summary="A comprehensive guide to screenwriting" * 20,
    main_argument="Structure is the key to good stories",
    concepts=None,
    principles=None,
    techniques=None,
    examples=None,
    document_id="doc-1",
):
    """Create a test BookUnderstanding with default knowledge items."""
    if concepts is None:
        concepts = [_make_concept("Inciting Incident"), _make_concept("Climax")]
    if principles is None:
        principles = [
            _make_principle("Every scene must turn", "42"),
            _make_principle("Conflict is essential", "67"),
        ]
    if techniques is None:
        techniques = [
            _make_technique("The Slow Reveal", "Build tension gradually"),
            _make_technique("Dialogue Subtext", "What characters mean vs say"),
        ]
    if examples is None:
        examples = [
            _make_example(
                "12 Angry Men", "Jury deliberation", "Power of dissent", "78"
            ),
            _make_example("Chinatown", "Final scene", "Tragic inevitability", "102"),
        ]
    return BookUnderstanding(
        id=book_id,
        document_id=document_id,
        title=title,
        author=author,
        summary=summary,
        main_argument=main_argument,
        concepts=concepts,
        principles=principles,
        techniques=techniques,
        examples=examples,
    )


def _make_second_book():
    """Create a second test book for multi-book tests."""
    return _make_book(
        book_id="book-2",
        title="Save The Cat",
        author="Blake Snyder",
        main_argument="Every story needs a beat sheet",
        document_id="doc-2",
        concepts=[_make_concept("Beat Sheet"), _make_concept("Logline")],
        principles=[
            _make_principle("The hero must save the cat", "15"),
            _make_principle("Stakes must escalate", "33"),
        ],
        techniques=[
            _make_technique("The Board", "Visual planning method"),
        ],
        examples=[
            _make_example("Die Hard", "Opening", "Perfect hero introduction", "45"),
        ],
    )


# =========================================================================
# TestMentorInit
# =========================================================================


class TestMentorInit:
    """Tests for MentorEngine.__init__"""

    def test_init_with_llm_and_store(self):
        """Constructor stores llm_complete and book_store."""
        llm = AsyncMock(return_value="response")
        store = MagicMock()
        engine = MentorEngine(llm_complete=llm, book_store=store)
        assert engine._llm_complete is llm
        assert engine._book_store is store

    def test_init_active_books_empty(self):
        """Active books dict is empty after init."""
        engine = MentorEngine(llm_complete=AsyncMock())
        assert engine._active_books == {}

    def test_init_no_book_store(self):
        """book_store defaults to None when not provided."""
        engine = MentorEngine(llm_complete=AsyncMock())
        assert engine._book_store is None


# =========================================================================
# TestLoadBooks
# =========================================================================


class TestLoadBooks:
    """Tests for load_books and get_active_books."""

    def test_load_single_book(self):
        """Loading a single book stores it by id."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        engine.load_books([book])
        assert "book-1" in engine._active_books
        assert engine._active_books["book-1"] is book

    def test_load_multiple_books(self):
        """Loading multiple books stores all by id."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book1 = _make_book(book_id="a")
        book2 = _make_book(book_id="b")
        engine.load_books([book1, book2])
        assert len(engine._active_books) == 2
        assert engine._active_books["a"] is book1
        assert engine._active_books["b"] is book2

    def test_load_replaces_previous(self):
        """Loading new books replaces the previously loaded set."""
        engine = MentorEngine(llm_complete=AsyncMock())
        engine.load_books([_make_book(book_id="old")])
        assert "old" in engine._active_books

        engine.load_books([_make_book(book_id="new")])
        assert "old" not in engine._active_books
        assert "new" in engine._active_books

    def test_get_active_books_returns_list(self):
        """get_active_books returns a list of BookUnderstanding objects."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        engine.load_books([book])
        active = engine.get_active_books()
        assert isinstance(active, list)
        assert len(active) == 1
        assert active[0] is book


# =========================================================================
# TestGetBooks
# =========================================================================


class TestGetBooks:
    """Tests for _get_books helper."""

    def test_returns_all_active_when_no_ids(self):
        """Without book_ids, returns all active books."""
        engine = MentorEngine(llm_complete=AsyncMock())
        engine.load_books([_make_book(book_id="a"), _make_book(book_id="b")])
        books = engine._get_books(None)
        assert len(books) == 2

    def test_filters_by_ids(self):
        """With book_ids, returns only matching books."""
        engine = MentorEngine(llm_complete=AsyncMock())
        engine.load_books([_make_book(book_id="a"), _make_book(book_id="b")])
        books = engine._get_books(["a"])
        assert len(books) == 1
        assert books[0].id == "a"

    def test_missing_ids_ignored(self):
        """Non-existent book_ids are silently ignored."""
        engine = MentorEngine(llm_complete=AsyncMock())
        engine.load_books([_make_book(book_id="a")])
        books = engine._get_books(["a", "nonexistent"])
        assert len(books) == 1
        assert books[0].id == "a"

    def test_all_ids_missing_returns_empty(self):
        """When all requested IDs are missing, returns empty list."""
        engine = MentorEngine(llm_complete=AsyncMock())
        engine.load_books([_make_book(book_id="a")])
        books = engine._get_books(["x", "y"])
        assert books == []


# =========================================================================
# TestFormatBookSummaries
# =========================================================================


class TestFormatBookSummaries:
    """Tests for _format_book_summaries helper."""

    def test_basic_format(self):
        """Basic formatting includes title, author, argument, summary, concepts."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._format_book_summaries([book])
        assert "## Story by Robert McKee" in result
        assert "Main argument: Structure is the key to good stories" in result
        assert "Summary:" in result
        assert "Key concepts: Inciting Incident, Climax" in result

    def test_with_principles(self):
        """include_principles=True adds principle statements."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._format_book_summaries([book], include_principles=True)
        assert "Key principles:" in result
        assert "Every scene must turn" in result
        assert "Conflict is essential" in result

    def test_without_principles_by_default(self):
        """Principles are not included by default."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._format_book_summaries([book])
        assert "Key principles:" not in result

    def test_with_examples(self):
        """include_examples=True adds example film references."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._format_book_summaries([book], include_examples=True)
        assert "Referenced films/examples:" in result
        assert "12 Angry Men" in result
        assert "Chinatown" in result

    def test_multiple_books_separated(self):
        """Multiple books are separated by horizontal rules."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book1 = _make_book(title="Story", author="McKee")
        book2 = _make_book(title="Save The Cat", author="Snyder")
        result = engine._format_book_summaries([book1, book2])
        assert "\n\n---\n\n" in result
        assert "## Story by McKee" in result
        assert "## Save The Cat by Snyder" in result


# =========================================================================
# TestFindPrinciple
# =========================================================================


class TestFindPrinciple:
    """Tests for _find_principle helper."""

    def test_exact_match(self):
        """Finds principle when statement matches exactly."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_principle("Every scene must turn", [book])
        assert result is not None
        assert result.statement == "Every scene must turn"

    def test_substring_of_statement(self):
        """Finds principle when search text is a substring of the statement."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_principle("scene must turn", [book])
        assert result is not None
        assert result.statement == "Every scene must turn"

    def test_statement_is_substring_of_search(self):
        """Finds principle when the statement is a substring of the search text."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_principle(
            "According to McKee, Every scene must turn in some way", [book]
        )
        assert result is not None
        assert result.statement == "Every scene must turn"

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_principle("EVERY SCENE MUST TURN", [book])
        assert result is not None
        assert result.statement == "Every scene must turn"

    def test_no_match_returns_none(self):
        """Returns None when no principle matches."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_principle("Something completely unrelated", [book])
        assert result is None


# =========================================================================
# TestFindTechnique
# =========================================================================


class TestFindTechnique:
    """Tests for _find_technique helper."""

    def test_exact_match(self):
        """Finds technique with exact name match."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_technique("The Slow Reveal", [book])
        assert result is not None
        assert result.name == "The Slow Reveal"

    def test_substring_match(self):
        """Finds technique with substring match."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_technique("Slow Reveal", [book])
        assert result is not None
        assert result.name == "The Slow Reveal"

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_technique("the slow reveal", [book])
        assert result is not None

    def test_no_match_returns_none(self):
        """Returns None when no technique matches."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_technique("Nonexistent Technique", [book])
        assert result is None


# =========================================================================
# TestFindExample
# =========================================================================


class TestFindExample:
    """Tests for _find_example helper."""

    def test_exact_title_match(self):
        """Finds example by exact film title."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_example("12 Angry Men", [book])
        assert result is not None
        assert result.work_title == "12 Angry Men"

    def test_substring_title_match(self):
        """Finds example when search is substring of title."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_example("Angry Men", [book])
        assert result is not None
        assert result.work_title == "12 Angry Men"

    def test_case_insensitive(self):
        """Matching is case-insensitive for titles."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_example("chinatown", [book])
        assert result is not None
        assert result.work_title == "Chinatown"

    def test_no_match_returns_none(self):
        """Returns None when no example matches."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book()
        result = engine._find_example("The Godfather", [book])
        assert result is None


# =========================================================================
# TestParseJson
# =========================================================================


class TestParseJson:
    """Tests for _parse_json helper."""

    def test_valid_json(self):
        """Parses clean JSON string correctly."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = engine._parse_json('{"key": "value", "items": [1, 2, 3]}')
        assert result == {"key": "value", "items": [1, 2, 3]}

    def test_json_embedded_in_text(self):
        """Extracts JSON from surrounding text (markdown, commentary)."""
        engine = MentorEngine(llm_complete=AsyncMock())
        response = (
            'Here is the analysis:\n```json\n{"key": "value"}\n```\nHope this helps!'
        )
        result = engine._parse_json(response)
        assert result == {"key": "value"}

    def test_invalid_json_returns_empty_dict(self):
        """Returns empty dict when JSON cannot be parsed."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = engine._parse_json("This is not JSON at all")
        assert result == {}

    def test_malformed_json_returns_empty_dict(self):
        """Returns empty dict for malformed JSON."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = engine._parse_json('{"key": "missing closing brace')
        assert result == {}


# =========================================================================
# TestAnalyzeScene
# =========================================================================


class TestAnalyzeScene:
    """Tests for analyze_scene method."""

    def test_no_books_returns_suggestion(self):
        """With no books loaded, returns analysis suggesting to study books first."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = _run(engine.analyze_scene("A courtroom drama scene"))
        assert isinstance(result, MentorAnalysis)
        assert result.user_input == "A courtroom drama scene"
        assert result.active_books == []
        assert len(result.suggestions) == 1
        assert "No books loaded" in result.suggestions[0]

    def test_no_books_does_not_call_llm(self):
        """With no books loaded, does not invoke the LLM."""
        llm = AsyncMock()
        engine = MentorEngine(llm_complete=llm)
        _run(engine.analyze_scene("anything"))
        llm.assert_not_awaited()

    def test_with_books_calls_llm(self):
        """With books loaded, invokes the LLM and returns structured analysis."""
        llm_response = json.dumps(
            {
                "elements_present": ["strong dialogue"],
                "elements_missing": ["subtext"],
                "relevant_principles": [
                    {
                        "statement": "Every scene must turn",
                        "source_book": "Story",
                        "why_relevant": "applies",
                    }
                ],
                "applicable_techniques": [
                    {
                        "name": "The Slow Reveal",
                        "source_book": "Story",
                        "how_to_apply": "build tension",
                    }
                ],
                "similar_examples": [
                    {
                        "film": "12 Angry Men",
                        "scene": "Deliberation",
                        "lesson": "Power of dissent",
                        "source_book": "Story",
                    }
                ],
                "suggestions": ["Add more subtext"],
                "questions_to_consider": ["What does each character want?"],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.analyze_scene("A courtroom scene"))
        llm.assert_awaited_once()
        assert result.user_input == "A courtroom scene"
        assert result.active_books == ["Story"]
        assert "strong dialogue" in result.elements_present
        assert "subtext" in result.elements_missing
        assert "Add more subtext" in result.suggestions
        assert "What does each character want?" in result.questions_to_consider

    def test_principles_found_in_books(self):
        """Principles referenced in LLM response are looked up from loaded books."""
        llm_response = json.dumps(
            {
                "relevant_principles": [
                    {
                        "statement": "Every scene must turn",
                        "source_book": "Story",
                        "why_relevant": "key",
                    }
                ],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.analyze_scene("test"))
        assert len(result.relevant_principles) == 1
        assert result.relevant_principles[0].statement == "Every scene must turn"
        assert result.relevant_principles[0].source_page == "42"

    def test_techniques_found_in_books(self):
        """Techniques referenced in LLM response are looked up from loaded books."""
        llm_response = json.dumps(
            {
                "applicable_techniques": [
                    {
                        "name": "Dialogue Subtext",
                        "source_book": "Story",
                        "how_to_apply": "use it",
                    }
                ],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.analyze_scene("test"))
        assert len(result.applicable_techniques) == 1
        assert result.applicable_techniques[0].name == "Dialogue Subtext"

    def test_examples_as_inspirations(self):
        """Examples from LLM response become Inspiration objects with matched BookExamples."""
        llm_response = json.dumps(
            {
                "similar_examples": [
                    {
                        "film": "Chinatown",
                        "scene": "end",
                        "lesson": "tragic",
                        "source_book": "Story",
                    }
                ],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.analyze_scene("test"))
        assert len(result.similar_examples) == 1
        assert isinstance(result.similar_examples[0], Inspiration)
        assert result.similar_examples[0].example.work_title == "Chinatown"
        assert result.similar_examples[0].source_book == "Story"

    def test_custom_book_ids_filter(self):
        """book_ids parameter filters which books are used."""
        llm_response = json.dumps({"suggestions": ["filtered"]})
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book(book_id="a"), _make_second_book()])

        result = _run(engine.analyze_scene("test", book_ids=["a"]))
        assert result.active_books == ["Story"]

    def test_project_context_passed(self):
        """Project context is included in the LLM prompt."""
        llm_response = json.dumps({"suggestions": ["good"]})
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        _run(engine.analyze_scene("test", project_context="Legal thriller"))
        prompt_used = llm.call_args[0][0]
        assert "Legal thriller" in prompt_used


# =========================================================================
# TestBrainstorm
# =========================================================================


class TestBrainstorm:
    """Tests for brainstorm method."""

    def test_no_books_returns_fallback_idea(self):
        """With no books, returns a single fallback idea."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = _run(engine.brainstorm("courtroom twist"))
        assert isinstance(result, BrainstormResult)
        assert result.topic == "courtroom twist"
        assert result.active_books == []
        assert len(result.ideas) == 1
        assert result.ideas[0].idea == "Load reference books first"

    def test_no_books_does_not_call_llm(self):
        """With no books, does not invoke the LLM."""
        llm = AsyncMock()
        engine = MentorEngine(llm_complete=llm)
        _run(engine.brainstorm("anything"))
        llm.assert_not_awaited()

    def test_with_books_parses_ideas(self):
        """With books loaded, parses LLM response into BrainstormIdea objects."""
        llm_response = json.dumps(
            {
                "ideas": [
                    {
                        "idea": "Use unreliable narrator",
                        "rationale": "Creates tension per McKee",
                        "based_on": ["Inciting Incident"],
                        "source_inspiration": "Fight Club",
                    },
                    {
                        "idea": "Reverse expectations",
                        "rationale": "Subverts audience assumptions",
                        "based_on": ["Climax"],
                        "source_inspiration": None,
                    },
                ],
                "suggested_structure": "Three-act breakdown",
                "concepts_to_apply": ["Inciting Incident", "Climax"],
                "techniques_to_try": ["The Slow Reveal"],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.brainstorm("plot twist"))
        assert result.topic == "plot twist"
        assert result.active_books == ["Story"]
        assert len(result.ideas) == 2
        assert result.ideas[0].idea == "Use unreliable narrator"
        assert result.ideas[0].source_inspiration == "Fight Club"
        assert result.ideas[1].idea == "Reverse expectations"

    def test_constraints_passed(self):
        """Constraints are formatted and passed into the LLM prompt."""
        llm_response = json.dumps({"ideas": []})
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        _run(engine.brainstorm("test", constraints=["no violence", "PG-rated"]))
        prompt_used = llm.call_args[0][0]
        assert "no violence" in prompt_used
        assert "PG-rated" in prompt_used

    def test_suggested_structure_parsed(self):
        """suggested_structure field is correctly extracted from response."""
        llm_response = json.dumps(
            {
                "ideas": [],
                "suggested_structure": "Hero's Journey framework",
                "concepts_to_apply": [],
                "techniques_to_try": [],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.brainstorm("structure"))
        assert result.suggested_structure == "Hero's Journey framework"

    def test_concepts_and_techniques_parsed(self):
        """concepts_applied and techniques_suggested are correctly extracted."""
        llm_response = json.dumps(
            {
                "ideas": [],
                "suggested_structure": "",
                "concepts_to_apply": ["Inciting Incident", "Midpoint"],
                "techniques_to_try": ["The Board", "The Slow Reveal"],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.brainstorm("test"))
        assert result.concepts_applied == ["Inciting Incident", "Midpoint"]
        assert result.techniques_suggested == ["The Board", "The Slow Reveal"]


# =========================================================================
# TestCheckRules
# =========================================================================


class TestCheckRules:
    """Tests for check_rules method."""

    def test_no_books_returns_empty_result(self):
        """With no books, returns result with 'No books loaded' assessment."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = _run(engine.check_rules("My scene"))
        assert isinstance(result, RuleCheckResult)
        assert result.document_ids == []
        assert result.scene_or_work == "My scene"
        assert result.overall_assessment == "No books loaded for reference"
        assert result.rules_followed == []
        assert result.rules_violated == []

    def test_no_books_does_not_call_llm(self):
        """With no books, does not invoke the LLM."""
        llm = AsyncMock()
        engine = MentorEngine(llm_complete=llm)
        _run(engine.check_rules("anything"))
        llm.assert_not_awaited()

    def test_rules_followed_parsed(self):
        """Rules followed in LLM response are matched to loaded principles."""
        llm_response = json.dumps(
            {
                "rules_followed": [
                    {
                        "principle": "Every scene must turn",
                        "evidence": "Scene has clear shift",
                        "source_book": "Story",
                    }
                ],
                "rules_violated": [],
                "rules_unclear": [],
                "overall_assessment": "Strong work",
                "priority_fixes": [],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.check_rules("A well-structured scene"))
        assert len(result.rules_followed) == 1
        assert result.rules_followed[0].status == "followed"
        assert result.rules_followed[0].principle.statement == "Every scene must turn"
        assert result.rules_followed[0].evidence == "Scene has clear shift"

    def test_rules_violated_with_suggestions(self):
        """Violated rules include suggestion for how to fix."""
        llm_response = json.dumps(
            {
                "rules_followed": [],
                "rules_violated": [
                    {
                        "principle": "Conflict is essential",
                        "evidence": "No clear conflict in scene",
                        "suggestion": "Add opposing forces",
                        "source_book": "Story",
                    }
                ],
                "rules_unclear": [],
                "overall_assessment": "Needs work",
                "priority_fixes": ["Add conflict"],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.check_rules("A flat scene"))
        assert len(result.rules_violated) == 1
        assert result.rules_violated[0].status == "violated"
        assert result.rules_violated[0].suggestion == "Add opposing forces"
        assert result.rules_violated[0].principle.statement == "Conflict is essential"

    def test_rules_unclear_parsed(self):
        """Unclear rules are parsed with question as evidence."""
        llm_response = json.dumps(
            {
                "rules_followed": [],
                "rules_violated": [],
                "rules_unclear": [
                    {
                        "principle": "Every scene must turn",
                        "question": "Is the emotional shift sufficient?",
                        "source_book": "Story",
                    }
                ],
                "overall_assessment": "Ambiguous",
                "priority_fixes": [],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.check_rules("An ambiguous scene"))
        assert len(result.rules_unclear) == 1
        assert result.rules_unclear[0].status == "unclear"
        assert result.rules_unclear[0].evidence == "Is the emotional shift sufficient?"

    def test_overall_assessment_and_priority_fixes(self):
        """overall_assessment and priority_fixes are extracted from LLM response."""
        llm_response = json.dumps(
            {
                "rules_followed": [],
                "rules_violated": [],
                "rules_unclear": [],
                "overall_assessment": "Solid scene with minor issues",
                "priority_fixes": ["Sharpen the turning point", "Clarify stakes"],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.check_rules("scene"))
        assert result.overall_assessment == "Solid scene with minor issues"
        assert result.priority_fixes == ["Sharpen the turning point", "Clarify stakes"]

    def test_document_ids_populated(self):
        """document_ids are populated from the books used for checking."""
        llm_response = json.dumps(
            {
                "rules_followed": [],
                "rules_violated": [],
                "rules_unclear": [],
                "overall_assessment": "ok",
                "priority_fixes": [],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book(document_id="doc-abc")])

        result = _run(engine.check_rules("scene"))
        assert result.document_ids == ["doc-abc"]


# =========================================================================
# TestFindInspiration
# =========================================================================


class TestFindInspiration:
    """Tests for find_inspiration method."""

    def test_no_books_returns_empty_list(self):
        """With no books, returns empty list."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = _run(engine.find_inspiration("courtroom tension"))
        assert result == []

    def test_no_books_does_not_call_llm(self):
        """With no books, does not invoke the LLM."""
        llm = AsyncMock()
        engine = MentorEngine(llm_complete=llm)
        _run(engine.find_inspiration("anything"))
        llm.assert_not_awaited()

    def test_example_found_in_books(self):
        """When LLM references a film in the book examples, it is matched."""
        llm_response = json.dumps(
            {
                "inspirations": [
                    {
                        "film": "12 Angry Men",
                        "scene": "Final vote",
                        "description": "The holdout convinces others",
                        "why_relevant": "Shows how to build tension in debate",
                        "how_to_adapt": "Use similar escalation pattern",
                        "source_book": "Story",
                        "page": "78",
                    }
                ]
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.find_inspiration("courtroom scene"))
        assert len(result) == 1
        assert isinstance(result[0], Inspiration)
        # Should match the existing BookExample from the loaded book
        assert result[0].example.work_title == "12 Angry Men"
        assert result[0].relevance_reason == "Shows how to build tension in debate"
        assert result[0].how_to_apply == "Use similar escalation pattern"
        assert result[0].source_book == "Story"

    def test_example_not_found_creates_new(self):
        """When LLM references a film NOT in loaded books, creates a new BookExample."""
        llm_response = json.dumps(
            {
                "inspirations": [
                    {
                        "film": "The Godfather",
                        "scene": "Baptism scene",
                        "description": "Intercuts baptism with murders",
                        "why_relevant": "Brilliant juxtaposition",
                        "how_to_adapt": "Use parallel editing",
                        "source_book": "Story",
                        "page": "150",
                    }
                ]
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.find_inspiration("parallel scenes"))
        assert len(result) == 1
        assert result[0].example.work_title == "The Godfather"
        assert result[0].example.scene_or_section == "Baptism scene"
        assert result[0].example.description == "Intercuts baptism with murders"
        assert result[0].example.lesson == "Brilliant juxtaposition"
        assert result[0].example.source_page == "150"

    def test_multiple_inspirations(self):
        """Multiple inspirations are parsed from LLM response."""
        llm_response = json.dumps(
            {
                "inspirations": [
                    {
                        "film": "12 Angry Men",
                        "scene": "vote",
                        "description": "desc1",
                        "why_relevant": "reason1",
                        "how_to_adapt": "adapt1",
                        "source_book": "Story",
                        "page": "78",
                    },
                    {
                        "film": "Chinatown",
                        "scene": "ending",
                        "description": "desc2",
                        "why_relevant": "reason2",
                        "how_to_adapt": "adapt2",
                        "source_book": "Story",
                        "page": "102",
                    },
                    {
                        "film": "Rashomon",
                        "scene": "testimonies",
                        "description": "desc3",
                        "why_relevant": "reason3",
                        "how_to_adapt": "adapt3",
                        "source_book": "Story",
                        "page": "200",
                    },
                ]
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.find_inspiration("multiple angles"))
        assert len(result) == 3
        # First two match existing book examples
        assert result[0].example.work_title == "12 Angry Men"
        assert result[1].example.work_title == "Chinatown"
        # Third is created since "Rashomon" is not in loaded books
        assert result[2].example.work_title == "Rashomon"
        assert result[2].example.scene_or_section == "testimonies"

    def test_uses_include_examples_in_format(self):
        """find_inspiration passes include_examples=True to _format_book_summaries."""
        llm_response = json.dumps({"inspirations": []})
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        _run(engine.find_inspiration("test"))
        prompt_used = llm.call_args[0][0]
        # The prompt should include film examples because include_examples=True
        assert "Referenced films/examples:" in prompt_used


# =========================================================================
# TestWhatWouldBooksSay
# =========================================================================


class TestWhatWouldBooksSay:
    """Tests for what_would_books_say method."""

    def test_no_books_returns_message(self):
        """With no books, returns a message to study books first."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = _run(engine.what_would_books_say("How do I write dialogue?"))
        assert result == "No books loaded. Study some reference books first."

    def test_no_books_does_not_call_llm(self):
        """With no books, does not invoke the LLM."""
        llm = AsyncMock()
        engine = MentorEngine(llm_complete=llm)
        _run(engine.what_would_books_say("anything"))
        llm.assert_not_awaited()

    def test_with_books_returns_llm_response(self):
        """With books loaded, returns the raw LLM response."""
        expected_response = "According to McKee, dialogue should reveal character."
        llm = AsyncMock(return_value=expected_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.what_would_books_say("How do I write dialogue?"))
        assert result == expected_response
        llm.assert_awaited_once()


# =========================================================================
# TestCompareApproaches
# =========================================================================


class TestCompareApproaches:
    """Tests for compare_approaches method."""

    def test_less_than_two_books_returns_message(self):
        """With fewer than 2 books, returns a message requiring more books."""
        engine = MentorEngine(llm_complete=AsyncMock())
        engine.load_books([_make_book()])
        result = _run(engine.compare_approaches("dialogue"))
        assert result == "Need at least 2 books to compare approaches."

    def test_zero_books_returns_message(self):
        """With zero books, also returns the message."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = _run(engine.compare_approaches("anything"))
        assert result == "Need at least 2 books to compare approaches."

    def test_with_two_books_returns_llm_response(self):
        """With 2+ books loaded, returns the raw LLM comparison response."""
        expected_response = "McKee emphasizes structure while Snyder focuses on beats."
        llm = AsyncMock(return_value=expected_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book(), _make_second_book()])

        result = _run(engine.compare_approaches("story structure"))
        assert result == expected_response
        llm.assert_awaited_once()


# =========================================================================
# Additional edge-case and integration tests
# =========================================================================


class TestAnalyzeSceneEdgeCases:
    """Additional edge cases for analyze_scene."""

    def test_unmatched_principles_excluded(self):
        """Principles not found in loaded books are silently excluded."""
        llm_response = json.dumps(
            {
                "relevant_principles": [
                    {
                        "statement": "Nonexistent principle XYZ",
                        "source_book": "Story",
                        "why_relevant": "test",
                    }
                ],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.analyze_scene("test"))
        assert result.relevant_principles == []

    def test_unmatched_techniques_excluded(self):
        """Techniques not found in loaded books are silently excluded."""
        llm_response = json.dumps(
            {
                "applicable_techniques": [
                    {
                        "name": "Nonexistent Technique",
                        "source_book": "Story",
                        "how_to_apply": "test",
                    }
                ],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.analyze_scene("test"))
        assert result.applicable_techniques == []

    def test_unmatched_examples_excluded(self):
        """Examples not found in loaded books are silently excluded from similar_examples."""
        llm_response = json.dumps(
            {
                "similar_examples": [
                    {
                        "film": "The Godfather",
                        "scene": "opening",
                        "lesson": "power",
                        "source_book": "Story",
                    }
                ],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.analyze_scene("test"))
        # analyze_scene only adds examples if _find_example returns a match
        assert result.similar_examples == []

    def test_default_project_context(self):
        """When no project_context given, defaults to 'General screenplay'."""
        llm_response = json.dumps({"suggestions": []})
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        _run(engine.analyze_scene("test"))
        prompt_used = llm.call_args[0][0]
        assert "General screenplay" in prompt_used

    def test_empty_llm_response(self):
        """Handles LLM returning non-JSON gracefully."""
        llm = AsyncMock(return_value="I don't understand the question")
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.analyze_scene("test"))
        assert isinstance(result, MentorAnalysis)
        assert result.elements_present == []
        assert result.elements_missing == []


class TestBrainstormEdgeCases:
    """Additional edge cases for brainstorm."""

    def test_constraints_none_defaults(self):
        """When constraints is None, uses empty list in result."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = _run(engine.brainstorm("topic", constraints=None))
        assert result.constraints == []

    def test_no_constraints_prompt_text(self):
        """When no constraints given, prompt contains 'None specified'."""
        llm_response = json.dumps({"ideas": []})
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        _run(engine.brainstorm("test"))
        prompt_used = llm.call_args[0][0]
        assert "None specified" in prompt_used

    def test_empty_ideas_list_from_llm(self):
        """Handles LLM returning empty ideas list."""
        llm_response = json.dumps(
            {
                "ideas": [],
                "suggested_structure": "",
                "concepts_to_apply": [],
                "techniques_to_try": [],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.brainstorm("empty topic"))
        assert result.ideas == []


class TestCheckRulesEdgeCases:
    """Additional edge cases for check_rules."""

    def test_unmatched_principle_in_followed_excluded(self):
        """Rules followed referencing unrecognized principles are excluded."""
        llm_response = json.dumps(
            {
                "rules_followed": [
                    {
                        "principle": "Unknown rule 999",
                        "evidence": "no match",
                        "source_book": "Story",
                    }
                ],
                "rules_violated": [],
                "rules_unclear": [],
                "overall_assessment": "ok",
                "priority_fixes": [],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.check_rules("scene"))
        assert result.rules_followed == []

    def test_check_rules_uses_include_principles(self):
        """check_rules passes include_principles=True to _format_book_summaries."""
        llm_response = json.dumps(
            {
                "rules_followed": [],
                "rules_violated": [],
                "rules_unclear": [],
                "overall_assessment": "ok",
                "priority_fixes": [],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        _run(engine.check_rules("test"))
        prompt_used = llm.call_args[0][0]
        # Should include principles since include_principles=True is passed
        assert "Key principles:" in prompt_used


class TestFindInspirationEdgeCases:
    """Additional edge cases for find_inspiration."""

    def test_empty_inspirations_from_llm(self):
        """Handles LLM returning empty inspirations list."""
        llm_response = json.dumps({"inspirations": []})
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.find_inspiration("obscure scenario"))
        assert result == []

    def test_new_example_fields_populated(self):
        """When creating a new BookExample from LLM response, all fields are set."""
        llm_response = json.dumps(
            {
                "inspirations": [
                    {
                        "film": "Memento",
                        "scene": "Reverse chronology opening",
                        "description": "Story told backwards",
                        "why_relevant": "Shows non-linear narrative",
                        "how_to_adapt": "Use selective reveals",
                        "source_book": "Story",
                        "page": "99",
                    }
                ]
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book()])

        result = _run(engine.find_inspiration("non-linear storytelling"))
        assert len(result) == 1
        new_example = result[0].example
        assert new_example.work_title == "Memento"
        assert new_example.scene_or_section == "Reverse chronology opening"
        assert new_example.description == "Story told backwards"
        assert new_example.lesson == "Shows non-linear narrative"
        assert new_example.source_page == "99"


class TestMultiBookIntegration:
    """Tests that use multiple books together."""

    def test_find_principle_across_books(self):
        """_find_principle searches across all loaded books."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book1 = _make_book()
        book2 = _make_second_book()
        result = engine._find_principle("hero must save the cat", [book1, book2])
        assert result is not None
        assert result.statement == "The hero must save the cat"

    def test_find_technique_across_books(self):
        """_find_technique searches across all loaded books."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book1 = _make_book()
        book2 = _make_second_book()
        result = engine._find_technique("The Board", [book1, book2])
        assert result is not None
        assert result.name == "The Board"

    def test_find_example_across_books(self):
        """_find_example searches across all loaded books."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book1 = _make_book()
        book2 = _make_second_book()
        result = engine._find_example("Die Hard", [book1, book2])
        assert result is not None
        assert result.work_title == "Die Hard"

    def test_analyze_scene_with_multiple_books(self):
        """analyze_scene uses all loaded books and lists all titles."""
        llm_response = json.dumps(
            {
                "elements_present": ["tension"],
                "relevant_principles": [
                    {
                        "statement": "Every scene must turn",
                        "source_book": "Story",
                        "why_relevant": "key",
                    },
                    {
                        "statement": "Stakes must escalate",
                        "source_book": "Save The Cat",
                        "why_relevant": "important",
                    },
                ],
                "applicable_techniques": [],
                "similar_examples": [],
                "suggestions": [],
                "questions_to_consider": [],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book(), _make_second_book()])

        result = _run(engine.analyze_scene("tense scene"))
        assert "Story" in result.active_books
        assert "Save The Cat" in result.active_books
        assert len(result.relevant_principles) == 2

    def test_brainstorm_with_multiple_books(self):
        """brainstorm uses all loaded books and lists all titles."""
        llm_response = json.dumps(
            {
                "ideas": [
                    {
                        "idea": "Combine approaches",
                        "rationale": "Both books suggest this",
                        "based_on": ["Beat Sheet"],
                        "source_inspiration": None,
                    }
                ],
                "suggested_structure": "Hybrid approach",
                "concepts_to_apply": [],
                "techniques_to_try": [],
            }
        )
        llm = AsyncMock(return_value=llm_response)
        engine = MentorEngine(llm_complete=llm)
        engine.load_books([_make_book(), _make_second_book()])

        result = _run(engine.brainstorm("hybrid structure"))
        assert "Story" in result.active_books
        assert "Save The Cat" in result.active_books
        assert len(result.ideas) == 1


class TestFormatBookSummariesEdgeCases:
    """Additional edge cases for _format_book_summaries."""

    def test_book_with_no_concepts(self):
        """Book with empty concepts list omits concepts line."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book(concepts=[])
        result = engine._format_book_summaries([book])
        assert "Key concepts:" not in result

    def test_book_with_no_principles_still_shows_header(self):
        """Book with empty principles and include_principles=True shows no principles."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book(principles=[])
        result = engine._format_book_summaries([book], include_principles=True)
        assert "Key principles:" not in result

    def test_book_with_no_examples_and_include_examples(self):
        """Book with empty examples and include_examples=True omits examples section."""
        engine = MentorEngine(llm_complete=AsyncMock())
        book = _make_book(examples=[])
        result = engine._format_book_summaries([book], include_examples=True)
        assert "Referenced films/examples:" not in result

    def test_summary_truncated_to_500_chars(self):
        """Summary is truncated to 500 characters followed by '...'."""
        engine = MentorEngine(llm_complete=AsyncMock())
        long_summary = "A" * 1000
        book = _make_book(summary=long_summary)
        result = engine._format_book_summaries([book])
        # The code uses book.summary[:500] + "..."
        assert "A" * 500 + "..." in result
        assert "A" * 501 not in result


class TestParseJsonEdgeCases:
    """Additional edge cases for _parse_json."""

    def test_nested_json(self):
        """Handles nested JSON structures."""
        engine = MentorEngine(llm_complete=AsyncMock())
        nested = json.dumps(
            {
                "outer": {"inner": [1, 2, 3]},
                "list": [{"a": 1}],
            }
        )
        result = engine._parse_json(nested)
        assert result["outer"]["inner"] == [1, 2, 3]
        assert result["list"][0]["a"] == 1

    def test_json_with_trailing_text(self):
        """Extracts JSON even with trailing text after closing brace."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = engine._parse_json('{"key": "val"} and some more text')
        assert result == {"key": "val"}

    def test_completely_empty_string(self):
        """Empty string returns empty dict."""
        engine = MentorEngine(llm_complete=AsyncMock())
        result = engine._parse_json("")
        assert result == {}
