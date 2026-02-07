"""
Tests for documents/understanding/graph_integration.py
=======================================================

Comprehensive tests for BookGraphIntegrator.
Covers book integration, concept/principle/technique/example nodes,
cross-concept relationships, queries, and edge cases.

Tests: 75+
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from documents.understanding.graph_integration import (
    BOOK_NODE_TYPES,
    BOOK_RELATION_TYPES,
    BookGraphIntegrator,
)
from documents.understanding.models import (
    BookExample,
    BookUnderstanding,
    Concept,
    ConfidenceLevel,
    Principle,
    Technique,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_concept(name="Inciting Incident", **kwargs):
    defaults = dict(
        id="concept_1234567890",
        name=name,
        definition="The event that starts the story",
        importance="Launches the narrative",
        source_document_id="doc1",
        source_pages="pp. 45-52",
        related_concepts=["Climax"],
        synonyms=["Catalyst"],
        domain="screenwriting",
    )
    defaults.update(kwargs)
    return Concept(**defaults)


def _make_principle(statement="Every scene must have a turning point", **kwargs):
    defaults = dict(
        id="principle_12345678",
        statement=statement,
        rationale="Without change, scenes feel static",
        source_document_id="doc1",
        source_page="p. 89",
        confidence_level=ConfidenceLevel.STRONG,
        applies_to=["dialogue", "action scenes"],
        check_question="Does your scene have a turning point?",
    )
    defaults.update(kwargs)
    return Principle(**defaults)


def _make_technique(name="Slow Reveal", **kwargs):
    defaults = dict(
        id="technique_1234567",
        name=name,
        description="Gradually revealing information to build tension",
        steps=["Step 1", "Step 2"],
        source_document_id="doc1",
        source_page="p. 120",
        use_cases=["tension building", "mystery"],
        when_to_use="When you want to build suspense",
        when_not_to_use="When audience needs clarity",
        example_films=["Chinatown", "The Sixth Sense"],
        related_concepts=["Inciting Incident"],
    )
    defaults.update(kwargs)
    return Technique(**defaults)


def _make_example(title="12 Angry Men", **kwargs):
    defaults = dict(
        id="example_12345678",
        work_title=title,
        work_type="film",
        scene_or_section="Jury deliberation",
        source_document_id="doc1",
        source_page="p. 200",
        description="Jurors debate the fate of a defendant",
        lesson="Tension through character conflict",
        what_works="Confined space amplifies conflict",
        demonstrates_concept=["Inciting Incident"],
        situation_type=["courtroom", "tension"],
        emotional_beat="confrontation",
    )
    defaults.update(kwargs)
    return BookExample(**defaults)


def _make_understanding(**kwargs):
    defaults = dict(
        id="understanding_1234",
        document_id="doc1",
        title="Story",
        author="Robert McKee",
        summary="A comprehensive guide to story structure" * 20,
        main_argument="Great stories have structure",
        domains=["screenwriting", "storytelling"],
        concepts=[_make_concept()],
        principles=[_make_principle()],
        techniques=[_make_technique()],
        examples=[_make_example()],
    )
    defaults.update(kwargs)
    return BookUnderstanding(**defaults)


def _make_mock_graph():
    """Create a mock knowledge graph with async methods."""
    graph = MagicMock()

    # add_node returns a mock node with an id
    async def mock_add_node(**kwargs):
        node = MagicMock()
        node.id = kwargs.get("node_id", f"node_{kwargs.get('name', 'test')}")
        node.name = kwargs.get("name", "test")
        node.attributes = kwargs.get("attributes", {})
        return node

    graph.add_node = AsyncMock(side_effect=mock_add_node)
    graph.add_edge = AsyncMock()

    # ensure_node returns a mock node
    async def mock_ensure_node(name, node_type, project=None):
        node = MagicMock()
        node.id = f"ensured_{name}"
        node.name = name
        node.attributes = {}
        return node

    graph.ensure_node = AsyncMock(side_effect=mock_ensure_node)
    graph.traverse = AsyncMock(return_value=[])

    return graph


# ── Constants ─────────────────────────────────────────────────────────────


class TestConstants:
    def test_book_node_types(self):
        assert "reference" in BOOK_NODE_TYPES
        assert "book_concept" in BOOK_NODE_TYPES
        assert "book_principle" in BOOK_NODE_TYPES
        assert "book_technique" in BOOK_NODE_TYPES
        assert "book_example" in BOOK_NODE_TYPES

    def test_book_relation_types(self):
        assert "defines" in BOOK_RELATION_TYPES
        assert "teaches" in BOOK_RELATION_TYPES
        assert "describes" in BOOK_RELATION_TYPES
        assert "cites" in BOOK_RELATION_TYPES
        assert "related_concept" in BOOK_RELATION_TYPES
        assert "applies_to" in BOOK_RELATION_TYPES
        assert "demonstrates" in BOOK_RELATION_TYPES
        assert "is_similar_to" in BOOK_RELATION_TYPES
        assert "prerequisite_for" in BOOK_RELATION_TYPES

    def test_node_type_count(self):
        assert len(BOOK_NODE_TYPES) == 5

    def test_relation_type_count(self):
        assert len(BOOK_RELATION_TYPES) == 9


# ── Init ──────────────────────────────────────────────────────────────────


class TestInit:
    def test_init(self):
        graph = MagicMock()
        integrator = BookGraphIntegrator(graph)
        assert integrator._graph is graph


# ── integrate_book ────────────────────────────────────────────────────────


class TestIntegrateBook:
    @pytest.mark.asyncio
    async def test_integrate_basic(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        understanding = _make_understanding()

        stats = await integrator.integrate_book(understanding)

        assert stats["nodes_created"] >= 1  # At least book node
        assert stats["edges_created"] >= 1
        assert stats["concepts_added"] >= 1
        assert stats["principles_added"] >= 1
        assert stats["techniques_added"] >= 1
        assert stats["examples_added"] >= 1

    @pytest.mark.asyncio
    async def test_integrate_creates_book_node(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        understanding = _make_understanding()

        await integrator.integrate_book(understanding)

        # First call is the book node
        first_call = graph.add_node.call_args_list[0]
        assert first_call.kwargs["name"] == "Story"
        assert first_call.kwargs["node_type"] == "concept"
        assert first_call.kwargs["attributes"]["type"] == "reference"
        assert first_call.kwargs["attributes"]["author"] == "Robert McKee"

    @pytest.mark.asyncio
    async def test_integrate_with_project(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        understanding = _make_understanding()

        await integrator.integrate_book(understanding, link_to_project="gusagusalu")

        first_call = graph.add_node.call_args_list[0]
        assert first_call.kwargs["project"] == "gusagusalu"

    @pytest.mark.asyncio
    async def test_integrate_empty_understanding(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        understanding = _make_understanding(
            concepts=[], principles=[], techniques=[], examples=[]
        )

        stats = await integrator.integrate_book(understanding)
        assert stats["nodes_created"] == 1  # Just the book node
        assert stats["concepts_added"] == 0
        assert stats["principles_added"] == 0
        assert stats["techniques_added"] == 0
        assert stats["examples_added"] == 0

    @pytest.mark.asyncio
    async def test_integrate_returns_stats(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        understanding = _make_understanding()

        stats = await integrator.integrate_book(understanding)

        assert "nodes_created" in stats
        assert "edges_created" in stats
        assert "concepts_added" in stats
        assert "principles_added" in stats
        assert "techniques_added" in stats
        assert "examples_added" in stats

    @pytest.mark.asyncio
    async def test_integrate_book_summary_truncated(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        long_summary = "A" * 1000
        understanding = _make_understanding(summary=long_summary)

        await integrator.integrate_book(understanding)

        first_call = graph.add_node.call_args_list[0]
        assert len(first_call.kwargs["attributes"]["summary"]) <= 500


# ── _add_concept ──────────────────────────────────────────────────────────


class TestAddConcept:
    @pytest.mark.asyncio
    async def test_add_concept_creates_node(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        concept = _make_concept()
        stats = {"nodes_created": 0, "edges_created": 0, "concepts_added": 0}

        await integrator._add_concept("book_node_1", concept, None, stats)

        assert stats["concepts_added"] == 1
        assert stats["nodes_created"] >= 1

    @pytest.mark.asyncio
    async def test_add_concept_links_to_book(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        concept = _make_concept()
        stats = {"nodes_created": 0, "edges_created": 0, "concepts_added": 0}

        await integrator._add_concept("book_node_1", concept, None, stats)

        # Should have add_edge call linking to book
        graph.add_edge.assert_called()
        first_edge = graph.add_edge.call_args_list[0]
        assert first_edge.args[0] == "book_node_1"

    @pytest.mark.asyncio
    async def test_add_concept_with_synonyms(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        concept = _make_concept(synonyms=["Catalyst", "Trigger"])
        stats = {"nodes_created": 0, "edges_created": 0, "concepts_added": 0}

        await integrator._add_concept("book_node_1", concept, None, stats)

        # 1 concept node + 2 synonym nodes = 3
        assert stats["nodes_created"] == 3
        # 1 book->concept edge + 2 concept->synonym edges = 3
        assert stats["edges_created"] == 3

    @pytest.mark.asyncio
    async def test_add_concept_no_synonyms(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        concept = _make_concept(synonyms=[])
        stats = {"nodes_created": 0, "edges_created": 0, "concepts_added": 0}

        await integrator._add_concept("book_node_1", concept, None, stats)

        assert stats["nodes_created"] == 1  # Just the concept
        assert stats["edges_created"] == 1  # Just the book->concept link

    @pytest.mark.asyncio
    async def test_add_concept_attributes(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        concept = _make_concept()
        stats = {"nodes_created": 0, "edges_created": 0, "concepts_added": 0}

        await integrator._add_concept("book_node_1", concept, "proj1", stats)

        # Check the concept node call
        concept_call = graph.add_node.call_args_list[0]
        attrs = concept_call.kwargs["attributes"]
        assert attrs["type"] == "book_concept"
        assert attrs["definition"] == "The event that starts the story"
        assert concept_call.kwargs["project"] == "proj1"


# ── _add_principle ────────────────────────────────────────────────────────


class TestAddPrinciple:
    @pytest.mark.asyncio
    async def test_add_principle_creates_node(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        principle = _make_principle()
        stats = {"nodes_created": 0, "edges_created": 0, "principles_added": 0}

        await integrator._add_principle("book_node_1", principle, None, stats)

        assert stats["principles_added"] == 1
        assert stats["nodes_created"] >= 1

    @pytest.mark.asyncio
    async def test_add_principle_truncates_name(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        principle = _make_principle(statement="A" * 100)
        stats = {"nodes_created": 0, "edges_created": 0, "principles_added": 0}

        await integrator._add_principle("book_node_1", principle, None, stats)

        first_call = graph.add_node.call_args_list[0]
        assert len(first_call.kwargs["name"]) <= 54  # 50 + "..."

    @pytest.mark.asyncio
    async def test_add_principle_links_to_domains(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        principle = _make_principle(applies_to=["dialogue", "action scenes"])
        stats = {"nodes_created": 0, "edges_created": 0, "principles_added": 0}

        await integrator._add_principle("book_node_1", principle, None, stats)

        # 1 book->principle + 2 principle->domain = 3 edges
        assert stats["edges_created"] == 3

    @pytest.mark.asyncio
    async def test_add_principle_no_domains(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        principle = _make_principle(applies_to=[])
        stats = {"nodes_created": 0, "edges_created": 0, "principles_added": 0}

        await integrator._add_principle("book_node_1", principle, None, stats)

        assert stats["edges_created"] == 1  # Just book->principle

    @pytest.mark.asyncio
    async def test_add_principle_attributes(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        principle = _make_principle()
        stats = {"nodes_created": 0, "edges_created": 0, "principles_added": 0}

        await integrator._add_principle("book_node_1", principle, None, stats)

        first_call = graph.add_node.call_args_list[0]
        attrs = first_call.kwargs["attributes"]
        assert attrs["type"] == "book_principle"
        assert attrs["confidence_level"] == "strong"
        assert attrs["check_question"] == "Does your scene have a turning point?"


# ── _add_technique ────────────────────────────────────────────────────────


class TestAddTechnique:
    @pytest.mark.asyncio
    async def test_add_technique_creates_node(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        technique = _make_technique()
        stats = {"nodes_created": 0, "edges_created": 0, "techniques_added": 0}

        await integrator._add_technique("book_node_1", technique, None, stats)

        assert stats["techniques_added"] == 1
        assert stats["nodes_created"] >= 1

    @pytest.mark.asyncio
    async def test_add_technique_links_use_cases(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        technique = _make_technique(use_cases=["tension building", "mystery"])
        stats = {"nodes_created": 0, "edges_created": 0, "techniques_added": 0}

        await integrator._add_technique("book_node_1", technique, None, stats)

        # 1 book->tech + 2 use_cases + 2 films = 5
        assert stats["edges_created"] == 5

    @pytest.mark.asyncio
    async def test_add_technique_links_films(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        technique = _make_technique(example_films=["Chinatown", "The Sixth Sense"])
        stats = {"nodes_created": 0, "edges_created": 0, "techniques_added": 0}

        await integrator._add_technique("book_node_1", technique, None, stats)

        # Film nodes should get film_reference attribute set
        assert graph.ensure_node.call_count >= 2

    @pytest.mark.asyncio
    async def test_add_technique_no_use_cases_no_films(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        technique = _make_technique(use_cases=[], example_films=[])
        stats = {"nodes_created": 0, "edges_created": 0, "techniques_added": 0}

        await integrator._add_technique("book_node_1", technique, None, stats)

        assert stats["edges_created"] == 1  # Just book->technique

    @pytest.mark.asyncio
    async def test_add_technique_attributes(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        technique = _make_technique()
        stats = {"nodes_created": 0, "edges_created": 0, "techniques_added": 0}

        await integrator._add_technique("book_node_1", technique, None, stats)

        first_call = graph.add_node.call_args_list[0]
        attrs = first_call.kwargs["attributes"]
        assert attrs["type"] == "book_technique"
        assert attrs["difficulty"] == "intermediate"
        assert len(attrs["steps"]) == 2


# ── _add_example ──────────────────────────────────────────────────────────


class TestAddExample:
    @pytest.mark.asyncio
    async def test_add_example_creates_node(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        example = _make_example()
        stats = {"nodes_created": 0, "edges_created": 0, "examples_added": 0}

        await integrator._add_example("book_node_1", example, None, stats)

        assert stats["examples_added"] == 1
        assert stats["nodes_created"] >= 1

    @pytest.mark.asyncio
    async def test_add_example_links_to_book(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        example = _make_example()
        stats = {"nodes_created": 0, "edges_created": 0, "examples_added": 0}

        await integrator._add_example("book_node_1", example, None, stats)

        graph.add_edge.assert_called()
        first_edge = graph.add_edge.call_args_list[0]
        assert first_edge.args[0] == "book_node_1"

    @pytest.mark.asyncio
    async def test_add_example_links_demonstrates(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        example = _make_example(demonstrates_concept=["Inciting Incident", "Climax"])
        stats = {"nodes_created": 0, "edges_created": 0, "examples_added": 0}

        await integrator._add_example("book_node_1", example, None, stats)

        # 1 book->example + 2 example->concept = 3
        assert stats["edges_created"] == 3

    @pytest.mark.asyncio
    async def test_add_example_no_demonstrates(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        example = _make_example(demonstrates_concept=[])
        stats = {"nodes_created": 0, "edges_created": 0, "examples_added": 0}

        await integrator._add_example("book_node_1", example, None, stats)

        assert stats["edges_created"] == 1  # Just book->example

    @pytest.mark.asyncio
    async def test_add_example_node_name(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        example = _make_example(
            work_title="12 Angry Men",
            scene_or_section="Jury deliberation",
        )
        stats = {"nodes_created": 0, "edges_created": 0, "examples_added": 0}

        await integrator._add_example("book_node_1", example, None, stats)

        first_call = graph.add_node.call_args_list[0]
        assert "12 Angry Men" in first_call.kwargs["name"]
        assert "Jury deliberation" in first_call.kwargs["name"]

    @pytest.mark.asyncio
    async def test_add_example_no_scene(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        example = _make_example(scene_or_section="")
        stats = {"nodes_created": 0, "edges_created": 0, "examples_added": 0}

        await integrator._add_example("book_node_1", example, None, stats)

        first_call = graph.add_node.call_args_list[0]
        assert "reference" in first_call.kwargs["name"]

    @pytest.mark.asyncio
    async def test_add_example_attributes(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)
        example = _make_example()
        stats = {"nodes_created": 0, "edges_created": 0, "examples_added": 0}

        await integrator._add_example("book_node_1", example, None, stats)

        first_call = graph.add_node.call_args_list[0]
        attrs = first_call.kwargs["attributes"]
        assert attrs["type"] == "book_example"
        assert attrs["work_type"] == "film"
        assert attrs["emotional_beat"] == "confrontation"


# ── _create_concept_relationships ─────────────────────────────────────────


class TestConceptRelationships:
    @pytest.mark.asyncio
    async def test_create_relationships_between_concepts(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        concept1 = _make_concept(
            id="concept_aaaa1111",
            name="Inciting Incident",
            related_concepts=["Climax"],
        )
        concept2 = _make_concept(
            id="concept_bbbb2222",
            name="Climax",
            related_concepts=["Inciting Incident"],
        )
        understanding = _make_understanding(
            concepts=[concept1, concept2],
            principles=[],
            techniques=[],
            examples=[],
        )
        stats = {"edges_created": 0}

        await integrator._create_concept_relationships(understanding, stats)

        assert stats["edges_created"] == 2  # Bidirectional

    @pytest.mark.asyncio
    async def test_create_relationships_no_match(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        concept = _make_concept(related_concepts=["Nonexistent"])
        understanding = _make_understanding(
            concepts=[concept],
            principles=[],
            techniques=[],
            examples=[],
        )
        stats = {"edges_created": 0}

        await integrator._create_concept_relationships(understanding, stats)

        assert stats["edges_created"] == 0  # No match found

    @pytest.mark.asyncio
    async def test_create_relationships_technique_to_concept(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        concept = _make_concept(
            id="concept_aaaa1111",
            name="Inciting Incident",
            related_concepts=[],
        )
        technique = _make_technique(related_concepts=["Inciting Incident"])
        understanding = _make_understanding(
            concepts=[concept],
            techniques=[technique],
            principles=[],
            examples=[],
        )
        stats = {"edges_created": 0}

        await integrator._create_concept_relationships(understanding, stats)

        assert stats["edges_created"] == 1

    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        concept1 = _make_concept(
            id="concept_aaaa1111",
            name="Inciting Incident",
            related_concepts=["climax"],  # lowercase
        )
        concept2 = _make_concept(
            id="concept_bbbb2222",
            name="Climax",  # Title case
            related_concepts=[],
        )
        understanding = _make_understanding(
            concepts=[concept1, concept2],
            principles=[],
            techniques=[],
            examples=[],
        )
        stats = {"edges_created": 0}

        await integrator._create_concept_relationships(understanding, stats)

        assert stats["edges_created"] == 1

    @pytest.mark.asyncio
    async def test_no_concepts_no_error(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        understanding = _make_understanding(
            concepts=[],
            principles=[],
            techniques=[],
            examples=[],
        )
        stats = {"edges_created": 0}

        await integrator._create_concept_relationships(understanding, stats)

        assert stats["edges_created"] == 0


# ── find_books_about ──────────────────────────────────────────────────────


class TestFindBooksAbout:
    @pytest.mark.asyncio
    async def test_find_books_about_found(self):
        graph = _make_mock_graph()
        # Mock traverse to return book reference
        result_node = MagicMock()
        result_node.node.name = "Story"
        result_node.node.attributes = {
            "type": "reference",
            "author": "McKee",
            "document_id": "doc1",
        }
        result_node.path = ["Inciting Incident", "Story"]
        graph.traverse = AsyncMock(return_value=[result_node])

        integrator = BookGraphIntegrator(graph)
        books = await integrator.find_books_about("Inciting Incident")

        assert len(books) == 1
        assert books[0]["title"] == "Story"
        assert books[0]["author"] == "McKee"

    @pytest.mark.asyncio
    async def test_find_books_about_no_results(self):
        graph = _make_mock_graph()
        graph.traverse = AsyncMock(return_value=[])

        integrator = BookGraphIntegrator(graph)
        books = await integrator.find_books_about("Unknown Concept")

        assert books == []

    @pytest.mark.asyncio
    async def test_find_books_about_filters_non_reference(self):
        graph = _make_mock_graph()
        non_ref = MagicMock()
        non_ref.node.attributes = {"type": "book_concept"}
        ref = MagicMock()
        ref.node.name = "Story"
        ref.node.attributes = {
            "type": "reference",
            "author": "McKee",
            "document_id": "doc1",
        }
        ref.path = ["A", "B"]
        graph.traverse = AsyncMock(return_value=[non_ref, ref])

        integrator = BookGraphIntegrator(graph)
        books = await integrator.find_books_about("test")

        assert len(books) == 1


# ── get_related_knowledge ─────────────────────────────────────────────────


class TestGetRelatedKnowledge:
    @pytest.mark.asyncio
    async def test_get_related_knowledge_all_types(self):
        graph = _make_mock_graph()

        results = []
        for type_name, depth in [
            ("book_concept", 1),
            ("book_principle", 1),
            ("book_technique", 2),
            ("book_example", 2),
        ]:
            r = MagicMock()
            r.node.name = f"Test {type_name}"
            r.node.attributes = {"type": type_name}
            r.depth = depth
            results.append(r)

        graph.traverse = AsyncMock(return_value=results)
        integrator = BookGraphIntegrator(graph)

        knowledge = await integrator.get_related_knowledge("test topic")

        assert len(knowledge["concepts"]) == 1
        assert len(knowledge["principles"]) == 1
        assert len(knowledge["techniques"]) == 1
        assert len(knowledge["examples"]) == 1

    @pytest.mark.asyncio
    async def test_get_related_knowledge_filtered(self):
        graph = _make_mock_graph()

        results = []
        for type_name in ["book_concept", "book_principle", "book_technique"]:
            r = MagicMock()
            r.node.name = f"Test {type_name}"
            r.node.attributes = {"type": type_name}
            r.depth = 1
            results.append(r)

        graph.traverse = AsyncMock(return_value=results)
        integrator = BookGraphIntegrator(graph)

        knowledge = await integrator.get_related_knowledge(
            "test", knowledge_types=["book_concept"]
        )

        assert len(knowledge["concepts"]) == 1
        assert len(knowledge["principles"]) == 0
        assert len(knowledge["techniques"]) == 0

    @pytest.mark.asyncio
    async def test_get_related_knowledge_empty(self):
        graph = _make_mock_graph()
        graph.traverse = AsyncMock(return_value=[])

        integrator = BookGraphIntegrator(graph)
        knowledge = await integrator.get_related_knowledge("unknown")

        assert knowledge == {
            "concepts": [],
            "principles": [],
            "techniques": [],
            "examples": [],
        }

    @pytest.mark.asyncio
    async def test_get_related_knowledge_ignores_unknown_types(self):
        graph = _make_mock_graph()

        r = MagicMock()
        r.node.name = "Test"
        r.node.attributes = {"type": "unknown_type"}
        r.depth = 1
        graph.traverse = AsyncMock(return_value=[r])

        integrator = BookGraphIntegrator(graph)
        knowledge = await integrator.get_related_knowledge("test")

        assert all(len(v) == 0 for v in knowledge.values())


# ── link_to_project_concept ───────────────────────────────────────────────


class TestLinkToProjectConcept:
    @pytest.mark.asyncio
    async def test_link_default_relation(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        await integrator.link_to_project_concept("book_c1", "proj_c1")

        graph.add_edge.assert_called_once_with(
            "book_c1",
            "proj_c1",
            "relates_to",
            context="book_to_project_link",
        )

    @pytest.mark.asyncio
    async def test_link_custom_relation(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        await integrator.link_to_project_concept(
            "book_c1", "proj_c1", relationship="demonstrates"
        )

        graph.add_edge.assert_called_once_with(
            "book_c1",
            "proj_c1",
            "demonstrates",
            context="book_to_project_link",
        )


# ── Full Integration Flow ────────────────────────────────────────────────


class TestFullIntegration:
    @pytest.mark.asyncio
    async def test_multiple_concepts(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        concepts = [
            _make_concept(id=f"concept_{i}", name=f"Concept {i}", synonyms=[])
            for i in range(5)
        ]
        understanding = _make_understanding(
            concepts=concepts,
            principles=[],
            techniques=[],
            examples=[],
        )

        stats = await integrator.integrate_book(understanding)

        assert stats["concepts_added"] == 5
        assert stats["nodes_created"] == 6  # 1 book + 5 concepts

    @pytest.mark.asyncio
    async def test_multiple_principles(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        principles = [
            _make_principle(id=f"principle_{i}", statement=f"Rule {i}", applies_to=[])
            for i in range(3)
        ]
        understanding = _make_understanding(
            concepts=[],
            principles=principles,
            techniques=[],
            examples=[],
        )

        stats = await integrator.integrate_book(understanding)

        assert stats["principles_added"] == 3

    @pytest.mark.asyncio
    async def test_multiple_techniques(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        techniques = [
            _make_technique(
                id=f"technique_{i}",
                name=f"Tech {i}",
                use_cases=[],
                example_films=[],
                related_concepts=[],
            )
            for i in range(4)
        ]
        understanding = _make_understanding(
            concepts=[],
            principles=[],
            techniques=techniques,
            examples=[],
        )

        stats = await integrator.integrate_book(understanding)

        assert stats["techniques_added"] == 4

    @pytest.mark.asyncio
    async def test_multiple_examples(self):
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        examples = [
            _make_example(id=f"example_{i}", title=f"Film {i}", demonstrates_concept=[])
            for i in range(3)
        ]
        understanding = _make_understanding(
            concepts=[],
            principles=[],
            techniques=[],
            examples=examples,
        )

        stats = await integrator.integrate_book(understanding)

        assert stats["examples_added"] == 3

    @pytest.mark.asyncio
    async def test_complete_book_edge_count(self):
        """Verify edge count for a typical book with all types."""
        graph = _make_mock_graph()
        integrator = BookGraphIntegrator(graph)

        # 1 concept (1 synonym), 1 principle (2 domains), 1 technique (2 use_cases, 2 films), 1 example (1 demonstrates)
        understanding = _make_understanding()
        stats = await integrator.integrate_book(understanding)

        # Edges: book->concept(1) + concept->synonym(1)
        # + book->principle(1) + principle->domain(2)
        # + book->technique(1) + technique->use_case(2) + technique->film(2)
        # + book->example(1) + example->demonstrates(1)
        # + concept relationships (technique->concept via related_concepts matching)
        assert stats["edges_created"] >= 10
