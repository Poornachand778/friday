"""
Tests for Knowledge Graph
============================

Tests node/edge operations, traversal, queries, maintenance,
serialization, and graph statistics.

Run with: pytest tests/test_knowledge_graph.py -v
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from memory.layers.knowledge_graph import (
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
    RelationType,
    TraversalResult,
)


# =========================================================================
# Helpers
# =========================================================================


def _run(coro):
    """Run async coroutine synchronously"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def tmp_graph(tmp_path):
    """Create a KnowledgeGraph with a temporary persist path"""
    persist_path = tmp_path / "test_kg.json"
    graph = KnowledgeGraph(persist_path=persist_path)
    _run(graph.initialize())
    return graph


@pytest.fixture
def populated_graph(tmp_graph):
    """Create a graph with some nodes and edges"""
    g = tmp_graph
    _run(g.add_node("Ravi", NodeType.CHARACTER, project="gusagusalu"))
    _run(g.add_node("Father", NodeType.CHARACTER, project="gusagusalu"))
    _run(g.add_node("Climax Scene", NodeType.SCENE, project="gusagusalu"))
    _run(g.add_node("Gusagusalu", NodeType.PROJECT))
    _run(g.add_node("Confrontation", NodeType.CONCEPT))

    _run(g.add_edge("Ravi", "Climax Scene", RelationType.CHARACTER_IN))
    _run(g.add_edge("Father", "Climax Scene", RelationType.CHARACTER_IN))
    _run(
        g.add_edge(
            "Ravi", "Father", RelationType.HAS_RELATIONSHIP, context="father-son"
        )
    )
    _run(g.add_edge("Climax Scene", "Gusagusalu", RelationType.SCENE_IN))
    _run(g.add_edge("Climax Scene", "Confrontation", RelationType.CONTAINS))
    return g


# =========================================================================
# NodeType and RelationType Enums
# =========================================================================


class TestEnums:
    """Test enum definitions"""

    def test_node_types_exist(self):
        assert NodeType.CHARACTER == "character"
        assert NodeType.SCENE == "scene"
        assert NodeType.PROJECT == "project"
        assert NodeType.CONCEPT == "concept"
        assert NodeType.PERSON == "person"
        assert NodeType.EVENT == "event"
        assert NodeType.LOCATION == "location"

    def test_relation_types_exist(self):
        assert RelationType.DISCUSSES == "discusses"
        assert RelationType.CONTAINS == "contains"
        assert RelationType.CHARACTER_IN == "character_in"
        assert RelationType.SCENE_IN == "scene_in"
        assert RelationType.HAS_RELATIONSHIP == "has_relationship"
        assert RelationType.CREATES == "creates"
        assert RelationType.WANTS == "wants"
        assert RelationType.INVOLVES == "involves"

    def test_node_type_is_string(self):
        assert isinstance(NodeType.CHARACTER, str)

    def test_relation_type_is_string(self):
        assert isinstance(RelationType.DISCUSSES, str)


# =========================================================================
# KnowledgeNode
# =========================================================================


class TestKnowledgeNode:
    """Test KnowledgeNode dataclass"""

    def test_create_node(self):
        node = KnowledgeNode(
            id="ravi",
            name="Ravi",
            node_type=NodeType.CHARACTER,
            project="gusagusalu",
        )
        assert node.id == "ravi"
        assert node.name == "Ravi"
        assert node.node_type == NodeType.CHARACTER
        assert node.project == "gusagusalu"

    def test_default_attributes(self):
        node = KnowledgeNode(id="n1", name="N1", node_type=NodeType.CONCEPT)
        assert node.attributes == {}
        assert node.source_memory_ids == []
        assert node.project is None

    def test_to_dict(self):
        node = KnowledgeNode(
            id="ravi",
            name="Ravi",
            node_type=NodeType.CHARACTER,
            attributes={"role": "protagonist"},
        )
        d = node.to_dict()
        assert d["id"] == "ravi"
        assert d["name"] == "Ravi"
        assert d["node_type"] == "character"
        assert d["attributes"]["role"] == "protagonist"
        assert "created_at" in d

    def test_from_dict(self):
        data = {
            "id": "ravi",
            "name": "Ravi",
            "node_type": "character",
            "attributes": {"role": "protagonist"},
            "project": "gusagusalu",
            "created_at": "2025-01-15T10:00:00",
            "source_memory_ids": ["mem-001"],
        }
        node = KnowledgeNode.from_dict(data)
        assert node.id == "ravi"
        assert node.node_type == NodeType.CHARACTER
        assert node.project == "gusagusalu"
        assert node.source_memory_ids == ["mem-001"]

    def test_from_dict_minimal(self):
        data = {"id": "x", "name": "X", "node_type": "concept"}
        node = KnowledgeNode.from_dict(data)
        assert node.id == "x"
        assert node.attributes == {}
        assert node.source_memory_ids == []

    def test_round_trip(self):
        original = KnowledgeNode(
            id="test",
            name="Test",
            node_type=NodeType.EVENT,
            attributes={"date": "2025-03-01"},
            project="proj",
            source_memory_ids=["m1", "m2"],
        )
        restored = KnowledgeNode.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.node_type == original.node_type
        assert restored.attributes == original.attributes
        assert restored.project == original.project
        assert restored.source_memory_ids == original.source_memory_ids


# =========================================================================
# KnowledgeEdge
# =========================================================================


class TestKnowledgeEdge:
    """Test KnowledgeEdge dataclass"""

    def test_create_edge(self):
        edge = KnowledgeEdge(
            source_id="ravi",
            target_id="climax_scene",
            relation=RelationType.CHARACTER_IN,
        )
        assert edge.source_id == "ravi"
        assert edge.target_id == "climax_scene"
        assert edge.relation == RelationType.CHARACTER_IN

    def test_defaults(self):
        edge = KnowledgeEdge(
            source_id="a",
            target_id="b",
            relation=RelationType.RELATES_TO,
        )
        assert edge.weight == 1.0
        assert edge.context == ""
        assert edge.source_memory_id is None

    def test_to_dict(self):
        edge = KnowledgeEdge(
            source_id="ravi",
            target_id="scene1",
            relation=RelationType.CHARACTER_IN,
            weight=0.8,
            context="Ravi appears in scene 1",
        )
        d = edge.to_dict()
        assert d["source_id"] == "ravi"
        assert d["target_id"] == "scene1"
        assert d["relation"] == "character_in"
        assert d["weight"] == 0.8
        assert "created_at" in d

    def test_from_dict(self):
        data = {
            "source_id": "ravi",
            "target_id": "scene1",
            "relation": "character_in",
            "weight": 0.9,
            "context": "test",
            "created_at": "2025-01-15T10:00:00",
            "source_memory_id": "mem-001",
        }
        edge = KnowledgeEdge.from_dict(data)
        assert edge.source_id == "ravi"
        assert edge.relation == RelationType.CHARACTER_IN
        assert edge.source_memory_id == "mem-001"

    def test_round_trip(self):
        original = KnowledgeEdge(
            source_id="a",
            target_id="b",
            relation=RelationType.DISCUSSES,
            weight=1.5,
            context="ctx",
            source_memory_id="m1",
        )
        restored = KnowledgeEdge.from_dict(original.to_dict())
        assert restored.source_id == original.source_id
        assert restored.target_id == original.target_id
        assert restored.relation == original.relation
        assert restored.weight == original.weight
        assert restored.context == original.context


# =========================================================================
# KnowledgeGraph - Initialization
# =========================================================================


class TestGraphInit:
    """Test graph initialization"""

    def test_init_creates_empty_graph(self, tmp_graph):
        assert len(tmp_graph._nodes) == 0
        assert tmp_graph._graph.number_of_edges() == 0
        assert tmp_graph._initialized is True

    def test_init_creates_persist_dir(self, tmp_path):
        persist_path = tmp_path / "sub" / "dir" / "kg.json"
        graph = KnowledgeGraph(persist_path=persist_path)
        _run(graph.initialize())
        assert persist_path.parent.exists()

    def test_double_init_is_noop(self, tmp_graph):
        _run(tmp_graph.add_node("X", NodeType.CONCEPT))
        _run(tmp_graph.initialize())  # Second init
        assert _run(tmp_graph.get_node("X")) is not None

    def test_repr(self, tmp_graph):
        r = repr(tmp_graph)
        assert "KnowledgeGraph(" in r
        assert "nodes=0" in r
        assert "edges=0" in r


# =========================================================================
# KnowledgeGraph - Node Operations
# =========================================================================


class TestNodeOperations:
    """Test add/get/delete/ensure node"""

    def test_add_node(self, tmp_graph):
        node = _run(tmp_graph.add_node("Ravi", NodeType.CHARACTER))
        assert node.name == "Ravi"
        assert node.node_type == NodeType.CHARACTER
        assert node.id == "ravi"

    def test_add_node_with_project(self, tmp_graph):
        node = _run(
            tmp_graph.add_node("Ravi", NodeType.CHARACTER, project="gusagusalu")
        )
        assert node.project == "gusagusalu"

    def test_add_node_with_custom_id(self, tmp_graph):
        node = _run(tmp_graph.add_node("Ravi", NodeType.CHARACTER, node_id="custom-id"))
        assert node.id == "custom-id"

    def test_add_node_with_attributes(self, tmp_graph):
        node = _run(
            tmp_graph.add_node(
                "Ravi",
                NodeType.CHARACTER,
                attributes={"role": "protagonist", "age": 30},
            )
        )
        assert node.attributes["role"] == "protagonist"
        assert node.attributes["age"] == 30

    def test_add_node_with_source_memory(self, tmp_graph):
        node = _run(
            tmp_graph.add_node(
                "Ravi",
                NodeType.CHARACTER,
                source_memory_id="mem-001",
            )
        )
        assert "mem-001" in node.source_memory_ids

    def test_update_existing_node_attributes(self, tmp_graph):
        _run(
            tmp_graph.add_node("Ravi", NodeType.CHARACTER, attributes={"role": "lead"})
        )
        _run(tmp_graph.add_node("Ravi", NodeType.CHARACTER, attributes={"age": 30}))
        node = _run(tmp_graph.get_node("ravi"))
        assert node.attributes["role"] == "lead"
        assert node.attributes["age"] == 30

    def test_update_node_adds_source_memory(self, tmp_graph):
        _run(tmp_graph.add_node("Ravi", NodeType.CHARACTER, source_memory_id="m1"))
        _run(tmp_graph.add_node("Ravi", NodeType.CHARACTER, source_memory_id="m2"))
        node = _run(tmp_graph.get_node("ravi"))
        assert "m1" in node.source_memory_ids
        assert "m2" in node.source_memory_ids

    def test_update_node_no_duplicate_memory(self, tmp_graph):
        _run(tmp_graph.add_node("Ravi", NodeType.CHARACTER, source_memory_id="m1"))
        _run(tmp_graph.add_node("Ravi", NodeType.CHARACTER, source_memory_id="m1"))
        node = _run(tmp_graph.get_node("ravi"))
        assert node.source_memory_ids.count("m1") == 1

    def test_get_node(self, tmp_graph):
        _run(tmp_graph.add_node("Ravi", NodeType.CHARACTER))
        node = _run(tmp_graph.get_node("ravi"))
        assert node is not None
        assert node.name == "Ravi"

    def test_get_node_normalizes_id(self, tmp_graph):
        _run(tmp_graph.add_node("Climax Scene", NodeType.SCENE))
        node = _run(tmp_graph.get_node("Climax Scene"))
        assert node is not None
        assert node.id == "climax_scene"

    def test_get_nonexistent_node(self, tmp_graph):
        node = _run(tmp_graph.get_node("nonexistent"))
        assert node is None

    def test_delete_node(self, tmp_graph):
        _run(tmp_graph.add_node("Temp", NodeType.CONCEPT))
        result = _run(tmp_graph.delete_node("temp"))
        assert result is True
        assert _run(tmp_graph.get_node("temp")) is None

    def test_delete_nonexistent_node(self, tmp_graph):
        result = _run(tmp_graph.delete_node("nonexistent"))
        assert result is False

    def test_delete_node_removes_edges(self, tmp_graph):
        _run(tmp_graph.add_node("A", NodeType.CONCEPT))
        _run(tmp_graph.add_node("B", NodeType.CONCEPT))
        _run(tmp_graph.add_edge("A", "B", RelationType.RELATES_TO))
        _run(tmp_graph.delete_node("A"))
        assert tmp_graph._graph.number_of_edges() == 0

    def test_ensure_node_creates_new(self, tmp_graph):
        node = _run(tmp_graph.ensure_node("New Concept"))
        assert node.name == "New Concept"
        assert node.node_type == NodeType.CONCEPT  # Default type

    def test_ensure_node_returns_existing(self, tmp_graph):
        _run(tmp_graph.add_node("Existing", NodeType.CHARACTER))
        node = _run(tmp_graph.ensure_node("Existing"))
        assert node.node_type == NodeType.CHARACTER  # Keeps original type

    def test_normalize_id(self, tmp_graph):
        assert tmp_graph._normalize_id("Climax Scene") == "climax_scene"
        assert tmp_graph._normalize_id("  Ravi  ") == "ravi"
        assert tmp_graph._normalize_id("a-b-c") == "a_b_c"
        assert tmp_graph._normalize_id("UPPER") == "upper"


# =========================================================================
# KnowledgeGraph - Edge Operations
# =========================================================================


class TestEdgeOperations:
    """Test add_edge, add_triplet, add_triplets"""

    def test_add_edge(self, tmp_graph):
        _run(tmp_graph.add_node("Ravi", NodeType.CHARACTER))
        _run(tmp_graph.add_node("Scene 1", NodeType.SCENE))
        edge = _run(tmp_graph.add_edge("Ravi", "Scene 1", RelationType.CHARACTER_IN))
        assert edge.source_id == "ravi"
        assert edge.target_id == "scene_1"
        assert edge.relation == RelationType.CHARACTER_IN

    def test_add_edge_auto_creates_nodes(self, tmp_graph):
        _run(tmp_graph.add_edge("New Source", "New Target", RelationType.RELATES_TO))
        source = _run(tmp_graph.get_node("new_source"))
        target = _run(tmp_graph.get_node("new_target"))
        assert source is not None
        assert target is not None
        assert source.node_type == NodeType.CONCEPT  # Default type

    def test_add_edge_with_context(self, tmp_graph):
        _run(tmp_graph.add_node("A", NodeType.CONCEPT))
        _run(tmp_graph.add_node("B", NodeType.CONCEPT))
        edge = _run(
            tmp_graph.add_edge(
                "A", "B", RelationType.DISCUSSES, context="talked about it"
            )
        )
        assert edge.context == "talked about it"

    def test_add_edge_with_weight(self, tmp_graph):
        _run(tmp_graph.add_node("A", NodeType.CONCEPT))
        _run(tmp_graph.add_node("B", NodeType.CONCEPT))
        edge = _run(tmp_graph.add_edge("A", "B", RelationType.RELATES_TO, weight=0.5))
        assert edge.weight == 0.5

    def test_duplicate_edge_strengthens(self, tmp_graph):
        _run(tmp_graph.add_node("A", NodeType.CONCEPT))
        _run(tmp_graph.add_node("B", NodeType.CONCEPT))
        _run(tmp_graph.add_edge("A", "B", RelationType.DISCUSSES))
        _run(tmp_graph.add_edge("A", "B", RelationType.DISCUSSES, context="new ctx"))
        edge_data = tmp_graph._graph["a"]["b"]
        assert edge_data["weight"] > 1.0  # Strengthened

    def test_duplicate_edge_appends_context(self, tmp_graph):
        _run(tmp_graph.add_node("A", NodeType.CONCEPT))
        _run(tmp_graph.add_node("B", NodeType.CONCEPT))
        _run(tmp_graph.add_edge("A", "B", RelationType.DISCUSSES, context="first"))
        _run(tmp_graph.add_edge("A", "B", RelationType.DISCUSSES, context="second"))
        edge_data = tmp_graph._graph["a"]["b"]
        assert "first" in edge_data["context"]
        assert "second" in edge_data["context"]

    def test_add_triplet(self, tmp_graph):
        edge = _run(tmp_graph.add_triplet("Boss", "discusses", "Climax"))
        assert edge.source_id == "boss"
        assert edge.target_id == "climax"
        assert edge.relation == RelationType.DISCUSSES

    def test_add_triplet_unknown_relation(self, tmp_graph):
        edge = _run(tmp_graph.add_triplet("A", "unknown_relation", "B"))
        assert edge.relation == RelationType.RELATES_TO  # Default fallback

    def test_add_triplet_with_context(self, tmp_graph):
        edge = _run(
            tmp_graph.add_triplet(
                "Boss",
                "creates",
                "Scene 5",
                context="Boss created scene 5 during brainstorm",
            )
        )
        assert edge.context == "Boss created scene 5 during brainstorm"

    def test_add_triplets_batch(self, tmp_graph):
        triplets = [
            ("Ravi", "character_in", "Gusagusalu"),
            ("Father", "character_in", "Gusagusalu"),
            ("Ravi", "has_relationship", "Father"),
        ]
        edges = _run(tmp_graph.add_triplets(triplets))
        assert len(edges) == 3

    def test_add_triplet_relation_mapping(self, tmp_graph):
        """Test all known relation string mappings"""
        mapping = {
            "discusses": RelationType.DISCUSSES,
            "contains": RelationType.CONTAINS,
            "character_in": RelationType.CHARACTER_IN,
            "scene_in": RelationType.SCENE_IN,
            "creates": RelationType.CREATES,
            "wants": RelationType.WANTS,
            "involves": RelationType.INVOLVES,
        }
        for i, (rel_str, expected) in enumerate(mapping.items()):
            edge = _run(tmp_graph.add_triplet(f"src{i}", rel_str, f"tgt{i}"))
            assert edge.relation == expected


# =========================================================================
# KnowledgeGraph - Query Operations
# =========================================================================


class TestQueryOperations:
    """Test traverse, get_by_type, get_related, find_path, search_by_attribute"""

    def test_traverse_basic(self, populated_graph):
        results = _run(populated_graph.traverse("Ravi", max_depth=1))
        assert len(results) > 0
        reached_ids = {r.node.id for r in results}
        assert "climax_scene" in reached_ids
        assert "father" in reached_ids

    def test_traverse_depth_2(self, populated_graph):
        results = _run(populated_graph.traverse("Ravi", max_depth=2))
        reached_ids = {r.node.id for r in results}
        # Ravi -> Climax Scene -> Gusagusalu (depth 2)
        assert "gusagusalu" in reached_ids

    def test_traverse_nonexistent_start(self, populated_graph):
        results = _run(populated_graph.traverse("nonexistent"))
        assert results == []

    def test_traverse_with_relation_filter(self, populated_graph):
        results = _run(
            populated_graph.traverse(
                "Ravi",
                max_depth=2,
                relation_filter=[RelationType.CHARACTER_IN],
            )
        )
        reached_ids = {r.node.id for r in results}
        assert "climax_scene" in reached_ids
        # Should NOT reach Father (has_relationship, not character_in)
        assert "father" not in reached_ids

    def test_traverse_has_path(self, populated_graph):
        results = _run(populated_graph.traverse("Ravi", max_depth=1))
        for r in results:
            assert len(r.path) > 0
            assert r.depth == 1

    def test_traverse_has_relations(self, populated_graph):
        results = _run(populated_graph.traverse("Ravi", max_depth=1))
        for r in results:
            assert len(r.relations) > 0

    def test_get_by_type(self, populated_graph):
        characters = _run(populated_graph.get_by_type(NodeType.CHARACTER))
        assert len(characters) == 2
        names = {c.name for c in characters}
        assert "Ravi" in names
        assert "Father" in names

    def test_get_by_type_with_project(self, populated_graph):
        chars = _run(
            populated_graph.get_by_type(NodeType.CHARACTER, project="gusagusalu")
        )
        assert len(chars) == 2
        chars_other = _run(
            populated_graph.get_by_type(NodeType.CHARACTER, project="other")
        )
        assert len(chars_other) == 0

    def test_get_by_type_no_results(self, populated_graph):
        events = _run(populated_graph.get_by_type(NodeType.EVENT))
        assert len(events) == 0

    def test_get_related(self, populated_graph):
        related = _run(populated_graph.get_related("climax_scene"))
        assert len(related) > 0
        # Should include both outgoing and incoming edges
        node_ids = {n.id for n, _ in related}
        assert "gusagusalu" in node_ids or "confrontation" in node_ids

    def test_get_related_with_filter(self, populated_graph):
        related = _run(
            populated_graph.get_related(
                "climax_scene",
                relation=RelationType.CONTAINS,
            )
        )
        assert len(related) >= 1
        for node, rel in related:
            assert "contains" in rel

    def test_get_related_nonexistent(self, populated_graph):
        related = _run(populated_graph.get_related("nonexistent"))
        assert related == []

    def test_get_related_includes_reverse(self, populated_graph):
        related = _run(populated_graph.get_related("climax_scene"))
        reverse_rels = [r for _, r in related if r.startswith("reverse_")]
        # Ravi -> climax_scene is an incoming edge, so reverse_character_in
        assert len(reverse_rels) > 0

    def test_find_path(self, populated_graph):
        path = _run(populated_graph.find_path("Ravi", "Gusagusalu"))
        assert path is not None
        assert path[0] == "ravi"
        assert path[-1] == "gusagusalu"

    def test_find_path_no_path(self, tmp_graph):
        _run(tmp_graph.add_node("A", NodeType.CONCEPT))
        _run(tmp_graph.add_node("B", NodeType.CONCEPT))
        # No edge between them
        path = _run(tmp_graph.find_path("A", "B"))
        assert path is None

    def test_find_path_nonexistent_node(self, populated_graph):
        path = _run(populated_graph.find_path("Ravi", "nonexistent"))
        assert path is None

    def test_search_by_attribute(self, tmp_graph):
        _run(
            tmp_graph.add_node(
                "Ravi",
                NodeType.CHARACTER,
                attributes={"role": "protagonist"},
            )
        )
        _run(
            tmp_graph.add_node(
                "Father",
                NodeType.CHARACTER,
                attributes={"role": "antagonist"},
            )
        )
        results = _run(tmp_graph.search_by_attribute("role", "protagonist"))
        assert len(results) == 1
        assert results[0].name == "Ravi"

    def test_search_by_attribute_no_match(self, tmp_graph):
        _run(tmp_graph.add_node("X", NodeType.CONCEPT))
        results = _run(tmp_graph.search_by_attribute("key", "value"))
        assert len(results) == 0


# =========================================================================
# KnowledgeGraph - Maintenance
# =========================================================================


class TestMaintenance:
    """Test orphan cleanup and duplicate merging"""

    def test_get_orphan_nodes(self, tmp_graph):
        _run(tmp_graph.add_node("Connected", NodeType.CONCEPT))
        _run(tmp_graph.add_node("Orphan", NodeType.CONCEPT))
        _run(tmp_graph.add_edge("Connected", "Connected2", RelationType.RELATES_TO))
        orphans = _run(tmp_graph.get_orphan_nodes())
        orphan_ids = {o.id for o in orphans}
        assert "orphan" in orphan_ids

    def test_cleanup_orphans_respects_age(self, tmp_graph):
        node = _run(tmp_graph.add_node("Recent Orphan", NodeType.CONCEPT))
        # Node was just created, so min_age_days=7 should NOT remove it
        deleted = _run(tmp_graph.cleanup_orphans(min_age_days=7))
        assert deleted == 0

    def test_cleanup_orphans_removes_old(self, tmp_graph):
        node = _run(tmp_graph.add_node("Old Orphan", NodeType.CONCEPT))
        # Manually age the node
        tmp_graph._nodes["old_orphan"].created_at = datetime.now() - timedelta(days=10)
        deleted = _run(tmp_graph.cleanup_orphans(min_age_days=7))
        assert deleted == 1
        assert _run(tmp_graph.get_node("old_orphan")) is None

    def test_merge_duplicates_no_duplicates(self, tmp_graph):
        _run(tmp_graph.add_node("A", NodeType.CONCEPT))
        _run(tmp_graph.add_node("B", NodeType.CONCEPT))
        merged = _run(tmp_graph.merge_duplicates())
        assert merged == 0


# =========================================================================
# KnowledgeGraph - Persistence
# =========================================================================


class TestPersistence:
    """Test save/load round-trip"""

    def test_save_creates_file(self, tmp_graph):
        _run(tmp_graph.add_node("Test", NodeType.CONCEPT))
        assert tmp_graph._persist_path.exists()

    def test_save_valid_json(self, tmp_graph):
        _run(tmp_graph.add_node("Test", NodeType.CONCEPT))
        with open(tmp_graph._persist_path) as f:
            data = json.load(f)
        assert "nodes" in data
        assert "edges" in data

    def test_load_restores_nodes(self, tmp_path):
        persist_path = tmp_path / "kg.json"

        # Create and populate
        g1 = KnowledgeGraph(persist_path=persist_path)
        _run(g1.initialize())
        _run(g1.add_node("Ravi", NodeType.CHARACTER, project="gusagusalu"))
        _run(g1.add_node("Scene 1", NodeType.SCENE))
        _run(g1.add_edge("Ravi", "Scene 1", RelationType.CHARACTER_IN))

        # Load from same file
        g2 = KnowledgeGraph(persist_path=persist_path)
        _run(g2.initialize())
        assert len(g2._nodes) == 2
        assert g2._graph.number_of_edges() == 1

    def test_load_restores_node_types(self, tmp_path):
        persist_path = tmp_path / "kg.json"

        g1 = KnowledgeGraph(persist_path=persist_path)
        _run(g1.initialize())
        _run(g1.add_node("Ravi", NodeType.CHARACTER))

        g2 = KnowledgeGraph(persist_path=persist_path)
        _run(g2.initialize())
        node = _run(g2.get_node("ravi"))
        assert node.node_type == NodeType.CHARACTER

    def test_load_restores_edges(self, tmp_path):
        persist_path = tmp_path / "kg.json"

        g1 = KnowledgeGraph(persist_path=persist_path)
        _run(g1.initialize())
        _run(g1.add_node("A", NodeType.CONCEPT))
        _run(g1.add_node("B", NodeType.CONCEPT))
        _run(
            g1.add_edge(
                "A", "B", RelationType.DISCUSSES, context="test ctx", weight=2.0
            )
        )

        g2 = KnowledgeGraph(persist_path=persist_path)
        _run(g2.initialize())
        edge_data = g2._graph["a"]["b"]
        assert edge_data["relation"] == "discusses"
        assert edge_data["context"] == "test ctx"
        assert edge_data["weight"] == 2.0

    def test_close_saves(self, tmp_path):
        persist_path = tmp_path / "kg.json"
        g = KnowledgeGraph(persist_path=persist_path)
        _run(g.initialize())
        _run(g.add_node("X", NodeType.CONCEPT))
        _run(g.close())
        assert persist_path.exists()

    def test_corrupted_file_handled(self, tmp_path):
        persist_path = tmp_path / "kg.json"
        persist_path.write_text("not valid json")

        g = KnowledgeGraph(persist_path=persist_path)
        _run(g.initialize())
        # Should recover gracefully with empty graph
        assert len(g._nodes) == 0


# =========================================================================
# KnowledgeGraph - Stats
# =========================================================================


class TestGraphStats:
    """Test get_stats"""

    def test_empty_stats(self, tmp_graph):
        stats = _run(tmp_graph.get_stats())
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert stats["density"] == 0

    def test_populated_stats(self, populated_graph):
        stats = _run(populated_graph.get_stats())
        assert stats["total_nodes"] == 5
        assert stats["total_edges"] == 5
        assert "character" in stats["nodes_by_type"]
        assert stats["nodes_by_type"]["character"] == 2
        assert "character_in" in stats["edges_by_relation"]

    def test_stats_has_connectivity(self, populated_graph):
        stats = _run(populated_graph.get_stats())
        assert "is_connected" in stats
        assert "density" in stats
        assert isinstance(stats["density"], float)


# =========================================================================
# KnowledgeGraph - Edge Cases
# =========================================================================


class TestGraphEdgeCases:
    """Test edge cases"""

    def test_self_loop(self, tmp_graph):
        _run(tmp_graph.add_node("A", NodeType.CONCEPT))
        _run(tmp_graph.add_edge("A", "A", RelationType.RELATES_TO))
        assert tmp_graph._graph.number_of_edges() == 1

    def test_special_characters_in_name(self, tmp_graph):
        node = _run(tmp_graph.add_node("INT. COURTROOM - DAY", NodeType.SCENE))
        assert node.id == "int._courtroom___day"

    def test_empty_name(self, tmp_graph):
        node = _run(tmp_graph.add_node("", NodeType.CONCEPT))
        assert node.id == ""

    def test_many_nodes(self, tmp_graph):
        for i in range(50):
            _run(tmp_graph.add_node(f"Node {i}", NodeType.CONCEPT))
        assert len(tmp_graph._nodes) == 50

    def test_unicode_name(self, tmp_graph):
        node = _run(tmp_graph.add_node("రవి", NodeType.CHARACTER))
        assert node.name == "రవి"

    def test_traverse_isolated_node(self, tmp_graph):
        _run(tmp_graph.add_node("Lonely", NodeType.CONCEPT))
        results = _run(tmp_graph.traverse("Lonely"))
        assert results == []

    def test_repr_after_populate(self, populated_graph):
        r = repr(populated_graph)
        assert "nodes=5" in r
        assert "edges=5" in r
