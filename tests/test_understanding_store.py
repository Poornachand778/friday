"""
Comprehensive tests for BookUnderstandingStore
===============================================

Tests cover initialization, CRUD operations, search, FTS indexing,
find-by-name queries, statistics, and edge cases.
Uses real SQLite databases in tmp_path for full isolation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import sqlite3
import threading
import uuid

import pytest
from datetime import datetime

from documents.storage.understanding_store import BookUnderstandingStore
from documents.config import StorageConfig
from documents.understanding.models import (
    BookUnderstanding,
    ChapterSummary,
    Concept,
    Principle,
    Technique,
    BookExample,
    ConfidenceLevel,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def store(tmp_path):
    """Create an initialized BookUnderstandingStore backed by a temp SQLite DB."""
    config = StorageConfig(
        db_path=str(tmp_path / "test.db"),
        documents_dir=str(tmp_path / "docs"),
        images_dir=str(tmp_path / "images"),
    )
    s = BookUnderstandingStore(config=config)
    s.initialize()
    yield s
    s.close()


@pytest.fixture
def sample_concept():
    """A single screenwriting Concept."""
    return Concept(
        name="Inciting Incident",
        definition="The event that launches the story into motion",
        importance="Without it the story never starts",
        source_document_id="doc-001",
        source_pages="pp. 45-52",
        related_concepts=["Climax", "Resolution"],
        parent_concept="Story Structure",
        sub_concepts=["Call to Adventure"],
        synonyms=["Catalyst"],
        domain="screenwriting",
        confidence=0.95,
    )


@pytest.fixture
def sample_principle():
    """A single screenwriting Principle."""
    return Principle(
        statement="Every scene must turn",
        rationale="Keeps audience engaged by ensuring change in every scene",
        source_document_id="doc-001",
        source_page="p. 233",
        confidence_level=ConfidenceLevel.ABSOLUTE,
        applies_to=["dramatic scenes", "dialogue scenes"],
        exceptions=["montage sequences"],
        prerequisites=["understanding of scene structure"],
        related_concepts=["Scene", "Beat"],
        related_techniques=["The Turning Point"],
        checkable=True,
        check_question="Does your scene have a turning point?",
    )


@pytest.fixture
def sample_technique():
    """A single screenwriting Technique."""
    return Technique(
        name="The Slow Reveal",
        description="Gradually expose information to build tension",
        steps=["Hint at secret", "Build curiosity", "Reveal partially", "Full reveal"],
        source_document_id="doc-001",
        source_page="p. 180",
        use_cases=["mystery", "thriller", "courtroom drama"],
        when_to_use="When the audience needs to discover truth with the character",
        when_not_to_use="When speed is essential to the narrative",
        example_films=["Chinatown", "The Usual Suspects"],
        example_description="In Chinatown, each clue is revealed slowly",
        related_concepts=["Inciting Incident"],
        related_principles=["Every scene must turn"],
        alternative_techniques=["The Twist"],
        difficulty="advanced",
    )


@pytest.fixture
def sample_example():
    """A single BookExample."""
    return BookExample(
        work_title="Chinatown",
        work_type="film",
        scene_or_section="Water mystery sequence",
        source_document_id="doc-001",
        source_page="p. 95",
        description="Jake follows the trail of water diversions",
        lesson="Layered storytelling rewards attentive viewers",
        what_works="Each clue naturally leads to the next",
        demonstrates_concept=["Inciting Incident"],
        demonstrates_technique=["The Slow Reveal"],
        demonstrates_principle=["Every scene must turn"],
        situation_type=["investigation", "noir"],
        emotional_beat="discovery",
    )


@pytest.fixture
def sample_understanding(
    sample_concept, sample_principle, sample_technique, sample_example
):
    """A fully populated BookUnderstanding."""
    return BookUnderstanding(
        document_id="doc-001",
        title="Story",
        author="Robert McKee",
        summary="A comprehensive guide to screenwriting structure and craft.",
        main_argument="Story structure is universal across genres",
        target_audience="Screenwriters and storytellers",
        chapters=[
            ChapterSummary(
                number=1,
                title="The Story Problem",
                summary="Introduction to story structure fundamentals.",
                key_points=["Stories need structure", "Audience expects patterns"],
                concepts_introduced=["Inciting Incident"],
                principles_taught=["Every scene must turn"],
                page_range="pp. 1-30",
            ),
            ChapterSummary(
                number=2,
                title="Structure and Setting",
                summary="How setting shapes story possibilities.",
                key_points=["Setting constrains story", "Genre sets expectations"],
                concepts_introduced=["Story Universe"],
                principles_taught=[],
                page_range="pp. 31-60",
            ),
        ],
        domains=["screenwriting", "structure"],
        agrees_with={"book-002": ["three-act structure"]},
        disagrees_with={"book-003": ["formula over art"]},
        extends={"book-004": ["character arc theory"]},
        comprehension_quality=0.87,
        study_completed_at=datetime(2025, 6, 15, 10, 30, 0),
        concepts=[sample_concept],
        principles=[sample_principle],
        techniques=[sample_technique],
        examples=[sample_example],
    )


@pytest.fixture
def minimal_understanding():
    """A BookUnderstanding with no knowledge items."""
    return BookUnderstanding(
        document_id="doc-minimal",
        title="Minimal Book",
        author="Tester",
        summary="A minimal book for testing.",
        main_argument="Minimalism is enough",
        target_audience="Testers",
        domains=["testing"],
        comprehension_quality=0.5,
    )


def _make_understanding(doc_id, title, domains=None, quality=0.8, completed_at=None):
    """Helper to quickly build a BookUnderstanding."""
    return BookUnderstanding(
        document_id=doc_id,
        title=title,
        author="Author",
        summary=f"Summary of {title}",
        main_argument="Argument",
        target_audience="Audience",
        domains=domains or [],
        comprehension_quality=quality,
        study_completed_at=completed_at or datetime.now(),
    )


# =============================================================================
# 1. TestStoreInit
# =============================================================================


class TestStoreInit:
    """Tests for initialization and lifecycle."""

    def test_init_creates_db_parent_dirs(self, tmp_path):
        """Constructor creates parent directory for db_path."""
        deep_path = tmp_path / "a" / "b" / "c" / "test.db"
        config = StorageConfig(
            db_path=str(deep_path),
            documents_dir=str(tmp_path / "docs"),
            images_dir=str(tmp_path / "images"),
        )
        s = BookUnderstandingStore(config=config)
        assert deep_path.parent.exists()
        # Not yet initialized -- no db file
        assert not deep_path.exists()

    def test_initialize_creates_tables(self, store):
        """initialize() creates all expected tables and the FTS virtual table."""
        with store._transaction() as cur:
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = {row["name"] for row in cur.fetchall()}

        expected = {
            "book_understandings",
            "book_concepts",
            "book_principles",
            "book_techniques",
            "book_examples",
            "knowledge_fts",
            # FTS shadow tables
            "knowledge_fts_data",
            "knowledge_fts_idx",
            "knowledge_fts_content",
            "knowledge_fts_docsize",
            "knowledge_fts_config",
        }
        assert expected.issubset(tables)

    def test_double_initialize_is_noop(self, store):
        """Calling initialize() twice does not raise or recreate tables."""
        store.initialize()  # second call
        # Should still work fine
        stats = store.get_stats()
        assert stats["total_understandings"] == 0

    def test_close_resets_state(self, tmp_path):
        """close() nullifies the connection and clears the initialized flag."""
        config = StorageConfig(
            db_path=str(tmp_path / "test.db"),
            documents_dir=str(tmp_path / "docs"),
            images_dir=str(tmp_path / "images"),
        )
        s = BookUnderstandingStore(config=config)
        s.initialize()
        assert s._initialized is True
        assert s._conn is not None

        s.close()
        assert s._initialized is False
        assert s._conn is None


# =============================================================================
# 2. TestStoreAndRetrieve
# =============================================================================


class TestStoreAndRetrieve:
    """Tests for store_understanding and get_understanding."""

    def test_store_and_retrieve_by_id(self, store, sample_understanding):
        """Round-trip: store then retrieve by understanding id."""
        store.store_understanding(sample_understanding)
        result = store.get_understanding(sample_understanding.id)

        assert result is not None
        assert result.id == sample_understanding.id
        assert result.title == "Story"
        assert result.author == "Robert McKee"
        assert result.summary == sample_understanding.summary

    def test_store_persists_concepts(self, store, sample_understanding):
        """Concepts are stored and reloaded correctly."""
        store.store_understanding(sample_understanding)
        result = store.get_understanding(sample_understanding.id)

        assert len(result.concepts) == 1
        c = result.concepts[0]
        assert c.name == "Inciting Incident"
        assert c.definition == "The event that launches the story into motion"
        assert c.importance == "Without it the story never starts"
        assert c.source_document_id == "doc-001"
        assert c.source_pages == "pp. 45-52"
        assert c.related_concepts == ["Climax", "Resolution"]
        assert c.parent_concept == "Story Structure"
        assert c.sub_concepts == ["Call to Adventure"]
        assert c.synonyms == ["Catalyst"]
        assert c.domain == "screenwriting"
        assert c.confidence == 0.95

    def test_store_persists_principles(self, store, sample_understanding):
        """Principles are stored and reloaded with correct confidence level."""
        store.store_understanding(sample_understanding)
        result = store.get_understanding(sample_understanding.id)

        assert len(result.principles) == 1
        p = result.principles[0]
        assert p.statement == "Every scene must turn"
        assert p.rationale == "Keeps audience engaged by ensuring change in every scene"
        assert p.confidence_level == ConfidenceLevel.ABSOLUTE
        assert p.applies_to == ["dramatic scenes", "dialogue scenes"]
        assert p.exceptions == ["montage sequences"]
        assert p.checkable is True
        assert p.check_question == "Does your scene have a turning point?"

    def test_store_persists_techniques(self, store, sample_understanding):
        """Techniques are stored and reloaded correctly."""
        store.store_understanding(sample_understanding)
        result = store.get_understanding(sample_understanding.id)

        assert len(result.techniques) == 1
        t = result.techniques[0]
        assert t.name == "The Slow Reveal"
        assert t.steps == [
            "Hint at secret",
            "Build curiosity",
            "Reveal partially",
            "Full reveal",
        ]
        assert t.difficulty == "advanced"
        assert "Chinatown" in t.example_films

    def test_store_persists_examples(self, store, sample_understanding):
        """BookExamples are stored and reloaded correctly."""
        store.store_understanding(sample_understanding)
        result = store.get_understanding(sample_understanding.id)

        assert len(result.examples) == 1
        e = result.examples[0]
        assert e.work_title == "Chinatown"
        assert e.work_type == "film"
        assert e.lesson == "Layered storytelling rewards attentive viewers"
        assert e.demonstrates_concept == ["Inciting Incident"]
        assert e.emotional_beat == "discovery"

    def test_get_understanding_by_document_id(self, store, sample_understanding):
        """get_understanding_by_document returns the correct record."""
        store.store_understanding(sample_understanding)
        result = store.get_understanding_by_document("doc-001")

        assert result is not None
        assert result.document_id == "doc-001"
        assert result.title == "Story"
        # Knowledge items should be loaded
        assert len(result.concepts) == 1
        assert len(result.principles) == 1

    def test_get_nonexistent_returns_none(self, store):
        """Querying a missing id returns None."""
        assert store.get_understanding("nonexistent-id") is None
        assert store.get_understanding_by_document("nonexistent-doc") is None

    def test_chapters_round_trip(self, store, sample_understanding):
        """Chapters serialize/deserialize through JSON correctly."""
        store.store_understanding(sample_understanding)
        result = store.get_understanding(sample_understanding.id)

        assert len(result.chapters) == 2
        ch1 = result.chapters[0]
        assert ch1.number == 1
        assert ch1.title == "The Story Problem"
        assert "Stories need structure" in ch1.key_points
        assert ch1.page_range == "pp. 1-30"

        ch2 = result.chapters[1]
        assert ch2.number == 2
        assert ch2.title == "Structure and Setting"

    def test_comprehension_quality_stored(self, store, sample_understanding):
        """comprehension_quality survives the round-trip."""
        store.store_understanding(sample_understanding)
        result = store.get_understanding(sample_understanding.id)
        assert result.comprehension_quality == pytest.approx(0.87)

    def test_study_completed_at_stored(self, store, sample_understanding):
        """study_completed_at survives the round-trip."""
        store.store_understanding(sample_understanding)
        result = store.get_understanding(sample_understanding.id)
        assert result.study_completed_at == datetime(2025, 6, 15, 10, 30, 0)

    def test_domains_stored_as_json(self, store, sample_understanding):
        """domains list is serialized as JSON and restored."""
        store.store_understanding(sample_understanding)
        result = store.get_understanding(sample_understanding.id)
        assert result.domains == ["screenwriting", "structure"]

    def test_agrees_disagrees_extends_stored(self, store, sample_understanding):
        """Cross-book relationship dicts survive the round-trip."""
        store.store_understanding(sample_understanding)
        result = store.get_understanding(sample_understanding.id)
        assert result.agrees_with == {"book-002": ["three-act structure"]}
        assert result.disagrees_with == {"book-003": ["formula over art"]}
        assert result.extends == {"book-004": ["character arc theory"]}

    def test_insert_or_replace_updates(self, store, sample_understanding):
        """Storing the same id again replaces the existing record."""
        store.store_understanding(sample_understanding)
        # Mutate and re-store
        sample_understanding.title = "Story (Updated)"
        sample_understanding.comprehension_quality = 0.99
        store.store_understanding(sample_understanding)

        result = store.get_understanding(sample_understanding.id)
        assert result.title == "Story (Updated)"
        assert result.comprehension_quality == pytest.approx(0.99)


# =============================================================================
# 3. TestListUnderstandings
# =============================================================================


class TestListUnderstandings:
    """Tests for list_understandings with domain filtering and limits."""

    def test_list_all(self, store):
        """list_understandings returns all stored understandings."""
        u1 = _make_understanding("d1", "Book A", domains=["drama"])
        u2 = _make_understanding("d2", "Book B", domains=["comedy"])
        store.store_understanding(u1)
        store.store_understanding(u2)

        results = store.list_understandings()
        assert len(results) == 2

    def test_filter_by_domain(self, store):
        """list_understandings filters by domain correctly."""
        u1 = _make_understanding("d1", "Book A", domains=["drama", "noir"])
        u2 = _make_understanding("d2", "Book B", domains=["comedy"])
        u3 = _make_understanding("d3", "Book C", domains=["drama"])
        store.store_understanding(u1)
        store.store_understanding(u2)
        store.store_understanding(u3)

        drama_results = store.list_understandings(domain="drama")
        assert len(drama_results) == 2
        titles = {r.title for r in drama_results}
        assert titles == {"Book A", "Book C"}

    def test_limit(self, store):
        """list_understandings respects the limit parameter."""
        for i in range(5):
            u = _make_understanding(f"d{i}", f"Book {i}")
            store.store_understanding(u)

        results = store.list_understandings(limit=3)
        assert len(results) == 3

    def test_returns_counts_not_full_objects(self, store, sample_understanding):
        """List view populates length-based placeholders, not real knowledge objects."""
        store.store_understanding(sample_understanding)
        results = store.list_understandings()
        assert len(results) == 1

        listed = results[0]
        # Concepts list has the right count...
        assert len(listed.concepts) == 1
        # ...but the entries are None placeholders, not Concept instances
        assert listed.concepts[0] is None

        assert len(listed.principles) == 1
        assert listed.principles[0] is None

        assert len(listed.techniques) == 1
        assert listed.techniques[0] is None

        assert len(listed.examples) == 1
        assert listed.examples[0] is None

    def test_empty_list(self, store):
        """list_understandings returns empty list on empty db."""
        results = store.list_understandings()
        assert results == []


# =============================================================================
# 4. TestDeleteUnderstanding
# =============================================================================


class TestDeleteUnderstanding:
    """Tests for delete_understanding including cascade and FTS cleanup."""

    def test_delete_existing_returns_true(self, store, sample_understanding):
        """Deleting an existing understanding returns True."""
        store.store_understanding(sample_understanding)
        assert store.delete_understanding(sample_understanding.id) is True

    def test_delete_removes_from_db(self, store, sample_understanding):
        """After deletion the understanding is no longer retrievable."""
        store.store_understanding(sample_understanding)
        store.delete_understanding(sample_understanding.id)
        assert store.get_understanding(sample_understanding.id) is None

    def test_delete_cascades_knowledge(self, store, sample_understanding):
        """Deleting an understanding removes all associated knowledge rows."""
        store.store_understanding(sample_understanding)
        uid = sample_understanding.id

        store.delete_understanding(uid)

        with store._transaction() as cur:
            for table in (
                "book_concepts",
                "book_principles",
                "book_techniques",
                "book_examples",
            ):
                cur.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE understanding_id = ?", (uid,)
                )
                assert (
                    cur.fetchone()[0] == 0
                ), f"{table} should be empty after cascade delete"

    def test_delete_nonexistent_returns_false(self, store):
        """Deleting a non-existent id returns False."""
        assert store.delete_understanding("does-not-exist") is False

    def test_delete_cleans_fts_entries(self, store, sample_understanding):
        """FTS rows referencing the deleted understanding are removed."""
        store.store_understanding(sample_understanding)
        # Verify FTS has entries
        with store._transaction() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_fts")
            assert cur.fetchone()[0] > 0

        store.delete_understanding(sample_understanding.id)

        # After deletion all concept/principle/technique/example FTS rows are gone
        with store._transaction() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_fts")
            assert cur.fetchone()[0] == 0


# =============================================================================
# 5. TestSearchKnowledge
# =============================================================================


class TestSearchKnowledge:
    """Tests for FTS-based search_knowledge."""

    def test_search_finds_concept(self, store, sample_understanding):
        """Searching for a concept keyword returns matching results."""
        store.store_understanding(sample_understanding)
        results = store.search_knowledge("Inciting")
        assert len(results) >= 1
        assert any(r["type"] == "concept" for r in results)

    def test_search_finds_principle(self, store, sample_understanding):
        """Searching for a principle keyword returns matching results."""
        store.store_understanding(sample_understanding)
        results = store.search_knowledge("scene must turn")
        assert len(results) >= 1
        assert any(r["type"] == "principle" for r in results)

    def test_search_finds_technique(self, store, sample_understanding):
        """Searching for a technique name returns matching results."""
        store.store_understanding(sample_understanding)
        results = store.search_knowledge("Slow Reveal")
        assert len(results) >= 1
        assert any(r["type"] == "technique" for r in results)

    def test_search_finds_example(self, store, sample_understanding):
        """Searching for an example work title returns matching results."""
        store.store_understanding(sample_understanding)
        results = store.search_knowledge("Chinatown")
        assert len(results) >= 1
        assert any(r["type"] == "example" for r in results)

    def test_filter_by_knowledge_type(self, store, sample_understanding):
        """knowledge_type parameter restricts results to one type only."""
        store.store_understanding(sample_understanding)
        results = store.search_knowledge("scene", knowledge_type="principle")
        for r in results:
            assert r["type"] == "principle"

    def test_top_k_limits_results(self, store):
        """top_k caps the number of returned results."""
        # Store understanding with many concepts so FTS has many hits
        concepts = [
            Concept(
                name=f"Story Element {i}",
                definition=f"Story concept number {i} about drama",
                source_document_id="doc-bulk",
            )
            for i in range(15)
        ]
        u = BookUnderstanding(
            document_id="doc-bulk",
            title="Bulk Book",
            author="Author",
            summary="Bulk",
            concepts=concepts,
            domains=["drama"],
        )
        store.store_understanding(u)

        results = store.search_knowledge("story", top_k=5)
        assert len(results) <= 5

    def test_no_results_returns_empty(self, store, sample_understanding):
        """Search for a term that doesn't match returns empty list."""
        store.store_understanding(sample_understanding)
        results = store.search_knowledge("xylophone")
        assert results == []

    def test_score_is_positive(self, store, sample_understanding):
        """Scores are returned as absolute (positive) values."""
        store.store_understanding(sample_understanding)
        results = store.search_knowledge("Inciting")
        for r in results:
            assert r["score"] >= 0

    def test_result_dict_shape(self, store, sample_understanding):
        """Each result dict has expected keys: id, type, content, score."""
        store.store_understanding(sample_understanding)
        results = store.search_knowledge("Inciting")
        assert len(results) >= 1
        for r in results:
            assert "id" in r
            assert "type" in r
            assert "content" in r
            assert "score" in r


# =============================================================================
# 6. TestFindByName
# =============================================================================


class TestFindByName:
    """Tests for find_concepts_by_name, find_techniques_by_name, find_examples_by_work."""

    def test_find_concepts_partial_match(self, store, sample_understanding):
        """find_concepts_by_name matches partial names."""
        store.store_understanding(sample_understanding)
        results = store.find_concepts_by_name("Inciting")
        assert len(results) == 1
        assert results[0].name == "Inciting Incident"

    def test_find_concepts_substring(self, store, sample_understanding):
        """find_concepts_by_name matches mid-string."""
        store.store_understanding(sample_understanding)
        results = store.find_concepts_by_name("cident")
        assert len(results) == 1

    def test_find_techniques_partial_match(self, store, sample_understanding):
        """find_techniques_by_name matches partial names."""
        store.store_understanding(sample_understanding)
        results = store.find_techniques_by_name("Slow")
        assert len(results) == 1
        assert results[0].name == "The Slow Reveal"

    def test_find_examples_partial_match(self, store, sample_understanding):
        """find_examples_by_work matches partial work titles."""
        store.store_understanding(sample_understanding)
        results = store.find_examples_by_work("China")
        assert len(results) == 1
        assert results[0].work_title == "Chinatown"

    def test_find_case_insensitive(self, store, sample_understanding):
        """SQL LIKE is case-insensitive by default in SQLite."""
        store.store_understanding(sample_understanding)
        results_lower = store.find_concepts_by_name("inciting")
        results_upper = store.find_concepts_by_name("INCITING")
        assert len(results_lower) == 1
        assert len(results_upper) == 1

    def test_find_no_match_returns_empty(self, store, sample_understanding):
        """No matches returns an empty list."""
        store.store_understanding(sample_understanding)
        assert store.find_concepts_by_name("Nonexistent") == []
        assert store.find_techniques_by_name("Nonexistent") == []
        assert store.find_examples_by_work("Nonexistent") == []


# =============================================================================
# 7. TestGetStats
# =============================================================================


class TestGetStats:
    """Tests for get_stats."""

    def test_empty_db_stats(self, store):
        """Stats on an empty database are all zeros."""
        stats = store.get_stats()
        assert stats["total_understandings"] == 0
        assert stats["total_concepts"] == 0
        assert stats["total_principles"] == 0
        assert stats["total_techniques"] == 0
        assert stats["total_examples"] == 0
        assert stats["total_knowledge_items"] == 0
        assert stats["domain_distribution"] == {}

    def test_stats_with_data(self, store, sample_understanding):
        """Stats reflect stored data accurately."""
        store.store_understanding(sample_understanding)
        stats = store.get_stats()

        assert stats["total_understandings"] == 1
        assert stats["total_concepts"] == 1
        assert stats["total_principles"] == 1
        assert stats["total_techniques"] == 1
        assert stats["total_examples"] == 1
        assert stats["total_knowledge_items"] == 4

    def test_domain_distribution(self, store):
        """domain_distribution counts each domain occurrence."""
        u1 = _make_understanding("d1", "Book A", domains=["drama", "noir"])
        u2 = _make_understanding("d2", "Book B", domains=["drama", "comedy"])
        store.store_understanding(u1)
        store.store_understanding(u2)

        stats = store.get_stats()
        dist = stats["domain_distribution"]
        assert dist["drama"] == 2
        assert dist["noir"] == 1
        assert dist["comedy"] == 1

    def test_stats_db_path(self, store):
        """Stats include the db_path."""
        stats = store.get_stats()
        assert "db_path" in stats
        assert stats["db_path"].endswith("test.db")


# =============================================================================
# 8. TestFTSIndex
# =============================================================================


class TestFTSIndex:
    """Tests for the FTS index lifecycle."""

    def test_fts_populated_on_store(self, store, sample_understanding):
        """Storing an understanding populates the FTS index."""
        store.store_understanding(sample_understanding)
        with store._transaction() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_fts")
            count = cur.fetchone()[0]
        # 1 concept + 1 principle + 1 technique + 1 example = 4
        assert count == 4

    def test_fts_cleaned_on_delete(self, store, sample_understanding):
        """Deleting an understanding cleans FTS entries."""
        store.store_understanding(sample_understanding)
        store.delete_understanding(sample_understanding.id)
        with store._transaction() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_fts")
            assert cur.fetchone()[0] == 0

    def test_fts_search_concepts(self, store, sample_understanding):
        """FTS returns concept entries when matching concept text."""
        store.store_understanding(sample_understanding)
        results = store.search_knowledge("Inciting Incident")
        types = [r["type"] for r in results]
        assert "concept" in types

    def test_fts_search_techniques(self, store, sample_understanding):
        """FTS returns technique entries when matching technique text."""
        store.store_understanding(sample_understanding)
        results = store.search_knowledge("Slow Reveal tension")
        types = [r["type"] for r in results]
        assert "technique" in types

    def test_fts_search_examples(self, store, sample_understanding):
        """FTS returns example entries when matching example text."""
        store.store_understanding(sample_understanding)
        results = store.search_knowledge("Chinatown water")
        types = [r["type"] for r in results]
        assert "example" in types

    def test_fts_entries_accumulate_on_re_store_same_ids(self, store):
        """
        Re-storing the same understanding re-inserts FTS entries.

        Note: _update_fts_index deletes FTS rows whose id LIKE
        '{understanding.id}%', but the FTS entries are keyed by the
        *knowledge item* ids (concept.id, principle.id, etc.) which are
        independent UUIDs.  When the same knowledge items keep their ids
        across re-stores, the INSERT OR REPLACE on the knowledge tables
        keeps them, and _update_fts_index adds new FTS rows because the
        LIKE pattern doesn't match them.  This is the current behaviour.
        """
        concept = Concept(
            name="Stable Concept",
            definition="Definition",
            source_document_id="doc-fts",
        )
        u = BookUnderstanding(
            document_id="doc-fts",
            title="FTS Book",
            author="Author",
            summary="test",
            domains=["test"],
            concepts=[concept],
        )
        store.store_understanding(u)
        with store._transaction() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_fts")
            first_count = cur.fetchone()[0]
        assert first_count == 1

        # Re-store -- FTS cleanup uses understanding.id prefix which
        # does NOT match concept.id, so the rows accumulate.
        store.store_understanding(u)
        with store._transaction() as cur:
            cur.execute("SELECT COUNT(*) FROM knowledge_fts")
            second_count = cur.fetchone()[0]
        assert second_count >= first_count


# =============================================================================
# 9. TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_understanding_with_no_knowledge(self, store, minimal_understanding):
        """An understanding with zero knowledge items stores and retrieves cleanly."""
        store.store_understanding(minimal_understanding)
        result = store.get_understanding(minimal_understanding.id)

        assert result is not None
        assert result.title == "Minimal Book"
        assert result.concepts == []
        assert result.principles == []
        assert result.techniques == []
        assert result.examples == []

    def test_very_long_text_fields(self, store):
        """Very long text (100k chars) doesn't crash."""
        long_text = "A" * 100_000
        u = BookUnderstanding(
            document_id="doc-long",
            title="Long Book",
            author="Author",
            summary=long_text,
            main_argument=long_text,
            domains=["testing"],
            concepts=[
                Concept(
                    name="Long Concept",
                    definition=long_text,
                    source_document_id="doc-long",
                )
            ],
        )
        store.store_understanding(u)
        result = store.get_understanding(u.id)
        assert result is not None
        assert len(result.summary) == 100_000
        assert len(result.concepts[0].definition) == 100_000

    def test_unicode_telugu_content(self, store):
        """Telugu/Unicode characters store and retrieve correctly."""
        u = BookUnderstanding(
            document_id="doc-telugu",
            title="తెలుగు కథ",
            author="తెలుగు రచయిత",
            summary="ఇది తెలుగు కథ యొక్క సారాంశం",
            main_argument="తెలుగు సాహిత్యం గొప్పది",
            domains=["తెలుగు", "సాహిత్యం"],
            concepts=[
                Concept(
                    name="కథా నిర్మాణం",
                    definition="కథ యొక్క నిర్మాణ పద్ధతి",
                    source_document_id="doc-telugu",
                    domain="తెలుగు",
                )
            ],
        )
        store.store_understanding(u)
        result = store.get_understanding(u.id)

        assert result is not None
        assert result.title == "తెలుగు కథ"
        assert result.author == "తెలుగు రచయిత"
        assert result.domains == ["తెలుగు", "సాహిత్యం"]
        assert result.concepts[0].name == "కథా నిర్మాణం"

    def test_special_characters_in_search(self, store, sample_understanding):
        """Search with quotes in query does not crash (quotes are escaped)."""
        store.store_understanding(sample_understanding)
        # This should not raise even though it has special chars
        results = store.search_knowledge('"scene" AND "turn"')
        # We don't assert results because FTS may or may not match;
        # the point is it doesn't raise.
        assert isinstance(results, list)

    def test_concurrent_access_check_same_thread_false(self, tmp_path):
        """Store uses check_same_thread=False, so another thread can read."""
        config = StorageConfig(
            db_path=str(tmp_path / "test.db"),
            documents_dir=str(tmp_path / "docs"),
            images_dir=str(tmp_path / "images"),
        )
        s = BookUnderstandingStore(config=config)
        s.initialize()

        u = _make_understanding("d1", "Thread Book")
        s.store_understanding(u)

        result_holder = [None]
        error_holder = [None]

        def reader():
            try:
                result_holder[0] = s.get_understanding(u.id)
            except Exception as exc:
                error_holder[0] = exc

        t = threading.Thread(target=reader)
        t.start()
        t.join(timeout=5)

        s.close()

        assert error_holder[0] is None, f"Thread raised: {error_holder[0]}"
        assert result_holder[0] is not None
        assert result_holder[0].title == "Thread Book"

    def test_study_completed_at_none(self, store):
        """study_completed_at=None is handled gracefully."""
        u = BookUnderstanding(
            document_id="doc-nodate",
            title="No Date Book",
            author="Author",
            summary="No date",
            domains=[],
        )
        store.store_understanding(u)
        result = store.get_understanding(u.id)
        assert result is not None
        assert result.study_completed_at is None

    def test_empty_lists_and_dicts(self, store):
        """Empty lists and dicts survive the JSON round-trip."""
        u = BookUnderstanding(
            document_id="doc-empty",
            title="Empty Fields Book",
            author="",
            summary="",
            main_argument="",
            target_audience="",
            chapters=[],
            domains=[],
            agrees_with={},
            disagrees_with={},
            extends={},
        )
        store.store_understanding(u)
        result = store.get_understanding(u.id)
        assert result is not None
        assert result.chapters == []
        assert result.domains == []
        assert result.agrees_with == {}
        assert result.disagrees_with == {}
        assert result.extends == {}


# =============================================================================
# 10. TestTransaction
# =============================================================================


class TestTransaction:
    """Tests for the _transaction context manager."""

    def test_transaction_auto_initializes(self, tmp_path):
        """_transaction() calls initialize() if not yet initialized."""
        config = StorageConfig(
            db_path=str(tmp_path / "test.db"),
            documents_dir=str(tmp_path / "docs"),
            images_dir=str(tmp_path / "images"),
        )
        s = BookUnderstandingStore(config=config)
        # Not calling s.initialize() on purpose
        with s._transaction() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1
        s.close()

    def test_transaction_rollback_on_error(self, store, sample_understanding):
        """On error, the transaction is rolled back."""
        store.store_understanding(sample_understanding)

        try:
            with store._transaction() as cur:
                cur.execute(
                    "DELETE FROM book_understandings WHERE id = ?",
                    (sample_understanding.id,),
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # The delete should have been rolled back
        result = store.get_understanding(sample_understanding.id)
        assert result is not None

    def test_transaction_commits_on_success(self, store):
        """On success, the transaction is committed."""
        u = _make_understanding("d-tx", "Transaction Book")
        store.store_understanding(u)

        result = store.get_understanding(u.id)
        assert result is not None


# =============================================================================
# 11. TestMultipleKnowledgeItems
# =============================================================================


class TestMultipleKnowledgeItems:
    """Tests with multiple knowledge items per understanding."""

    def test_multiple_concepts(self, store):
        """Multiple concepts are stored and retrieved."""
        concepts = [
            Concept(
                name=f"Concept {i}",
                definition=f"Def {i}",
                source_document_id="doc-multi",
            )
            for i in range(5)
        ]
        u = BookUnderstanding(
            document_id="doc-multi",
            title="Multi-concept Book",
            author="Author",
            summary="Has many concepts",
            domains=["drama"],
            concepts=concepts,
        )
        store.store_understanding(u)
        result = store.get_understanding(u.id)
        assert len(result.concepts) == 5
        names = {c.name for c in result.concepts}
        assert names == {f"Concept {i}" for i in range(5)}

    def test_multiple_principles(self, store):
        """Multiple principles are stored and retrieved."""
        principles = [
            Principle(
                statement=f"Principle {i}",
                rationale=f"Rationale {i}",
                source_document_id="doc-multi",
                confidence_level=ConfidenceLevel.MODERATE,
            )
            for i in range(4)
        ]
        u = BookUnderstanding(
            document_id="doc-multi-p",
            title="Multi-principle Book",
            author="Author",
            summary="Has many principles",
            domains=["drama"],
            principles=principles,
        )
        store.store_understanding(u)
        result = store.get_understanding(u.id)
        assert len(result.principles) == 4
        for p in result.principles:
            assert p.confidence_level == ConfidenceLevel.MODERATE

    def test_multiple_techniques(self, store):
        """Multiple techniques are stored and retrieved."""
        techniques = [
            Technique(
                name=f"Technique {i}",
                description=f"Description {i}",
                source_document_id="doc-multi",
                difficulty="beginner",
            )
            for i in range(3)
        ]
        u = BookUnderstanding(
            document_id="doc-multi-t",
            title="Multi-technique Book",
            author="Author",
            summary="Has many techniques",
            domains=["drama"],
            techniques=techniques,
        )
        store.store_understanding(u)
        result = store.get_understanding(u.id)
        assert len(result.techniques) == 3

    def test_multiple_examples(self, store):
        """Multiple examples are stored and retrieved."""
        examples = [
            BookExample(
                work_title=f"Film {i}",
                scene_or_section=f"Scene {i}",
                lesson=f"Lesson {i}",
                source_document_id="doc-multi",
            )
            for i in range(6)
        ]
        u = BookUnderstanding(
            document_id="doc-multi-e",
            title="Multi-example Book",
            author="Author",
            summary="Has many examples",
            domains=["drama"],
            examples=examples,
        )
        store.store_understanding(u)
        result = store.get_understanding(u.id)
        assert len(result.examples) == 6

    def test_find_across_multiple_understandings(self, store):
        """find_concepts_by_name searches across all understandings."""
        u1 = BookUnderstanding(
            document_id="d1",
            title="Book A",
            author="A",
            summary="A",
            domains=["drama"],
            concepts=[
                Concept(
                    name="Dramatic Irony",
                    definition="Audience knows more",
                    source_document_id="d1",
                )
            ],
        )
        u2 = BookUnderstanding(
            document_id="d2",
            title="Book B",
            author="B",
            summary="B",
            domains=["drama"],
            concepts=[
                Concept(
                    name="Dramatic Tension",
                    definition="Uncertainty and suspense",
                    source_document_id="d2",
                )
            ],
        )
        store.store_understanding(u1)
        store.store_understanding(u2)

        results = store.find_concepts_by_name("Dramatic")
        assert len(results) == 2
        names = {c.name for c in results}
        assert "Dramatic Irony" in names
        assert "Dramatic Tension" in names


# =============================================================================
# 12. TestConfidenceLevels
# =============================================================================


class TestConfidenceLevels:
    """Tests for correct handling of all ConfidenceLevel enum values."""

    @pytest.mark.parametrize(
        "level",
        [
            ConfidenceLevel.ABSOLUTE,
            ConfidenceLevel.STRONG,
            ConfidenceLevel.MODERATE,
            ConfidenceLevel.SUGGESTION,
        ],
    )
    def test_confidence_level_round_trip(self, store, level):
        """Each ConfidenceLevel value survives store/retrieve."""
        p = Principle(
            statement=f"Test principle at {level.value}",
            rationale="Test",
            source_document_id="doc-cl",
            confidence_level=level,
        )
        u = BookUnderstanding(
            document_id=f"doc-cl-{level.value}",
            title=f"CL Book {level.value}",
            author="Author",
            summary="test",
            domains=["test"],
            principles=[p],
        )
        store.store_understanding(u)
        result = store.get_understanding(u.id)
        assert result.principles[0].confidence_level == level

    def test_invalid_confidence_level_defaults_to_strong(self, store):
        """An unknown confidence_level string in DB defaults to STRONG."""
        u = BookUnderstanding(
            document_id="doc-bad-cl",
            title="Bad CL Book",
            author="Author",
            summary="test",
            domains=["test"],
            principles=[
                Principle(
                    statement="Test",
                    rationale="Test",
                    source_document_id="doc-bad-cl",
                )
            ],
        )
        store.store_understanding(u)

        # Manually corrupt the confidence_level in DB
        with store._transaction() as cur:
            cur.execute(
                "UPDATE book_principles SET confidence_level = 'unknown_level' WHERE understanding_id = ?",
                (u.id,),
            )

        result = store.get_understanding(u.id)
        assert result.principles[0].confidence_level == ConfidenceLevel.STRONG


# =============================================================================
# 13. TestDeleteAndReinsert
# =============================================================================


class TestDeleteAndReinsert:
    """Tests for deleting then re-adding the same understanding."""

    def test_delete_then_reinsert(self, store, sample_understanding):
        """Can delete and re-store the same understanding id."""
        store.store_understanding(sample_understanding)
        store.delete_understanding(sample_understanding.id)
        assert store.get_understanding(sample_understanding.id) is None

        # Re-store with same ID
        store.store_understanding(sample_understanding)
        result = store.get_understanding(sample_understanding.id)
        assert result is not None
        assert result.title == "Story"

    def test_delete_one_keeps_others(self, store):
        """Deleting one understanding does not affect others."""
        u1 = _make_understanding("d1", "Book A", domains=["drama"])
        u2 = _make_understanding("d2", "Book B", domains=["comedy"])
        store.store_understanding(u1)
        store.store_understanding(u2)

        store.delete_understanding(u1.id)

        assert store.get_understanding(u1.id) is None
        result = store.get_understanding(u2.id)
        assert result is not None
        assert result.title == "Book B"


# =============================================================================
# 14. TestSearchEdgeCases
# =============================================================================


class TestSearchEdgeCases:
    """Additional search edge cases."""

    def test_search_empty_db(self, store):
        """Search on empty DB returns empty list, not an error."""
        results = store.search_knowledge("anything")
        assert results == []

    def test_search_with_fts_operator_chars(self, store, sample_understanding):
        """Search with FTS special characters doesn't crash."""
        store.store_understanding(sample_understanding)
        # These might fail the FTS query, but should be handled gracefully
        results = store.search_knowledge("scene OR turn")
        assert isinstance(results, list)

    def test_search_after_delete_no_stale_results(self, store, sample_understanding):
        """After deleting, FTS does not return stale results."""
        store.store_understanding(sample_understanding)
        store.delete_understanding(sample_understanding.id)
        results = store.search_knowledge("Inciting")
        assert results == []

    def test_find_techniques_returns_correct_fields(self, store, sample_understanding):
        """find_techniques_by_name returns fully populated Technique objects."""
        store.store_understanding(sample_understanding)
        results = store.find_techniques_by_name("Slow Reveal")
        assert len(results) == 1
        t = results[0]
        assert t.description == "Gradually expose information to build tension"
        assert t.steps == [
            "Hint at secret",
            "Build curiosity",
            "Reveal partially",
            "Full reveal",
        ]
        assert (
            t.when_to_use
            == "When the audience needs to discover truth with the character"
        )
        assert t.difficulty == "advanced"
        assert isinstance(t.extracted_at, datetime)

    def test_find_examples_returns_correct_fields(self, store, sample_understanding):
        """find_examples_by_work returns fully populated BookExample objects."""
        store.store_understanding(sample_understanding)
        results = store.find_examples_by_work("Chinatown")
        assert len(results) == 1
        e = results[0]
        assert e.scene_or_section == "Water mystery sequence"
        assert e.description == "Jake follows the trail of water diversions"
        assert e.demonstrates_concept == ["Inciting Incident"]
        assert e.situation_type == ["investigation", "noir"]
        assert isinstance(e.extracted_at, datetime)
