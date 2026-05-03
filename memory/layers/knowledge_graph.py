"""
Knowledge Graph Layer
=====================

Cognee-inspired graph for relationship tracking.

Features:
    - NetworkX-based (lightweight, no external DB)
    - Triplet storage (subject-relation-object)
    - Graph traversal for relationship queries
    - Entity type support (character, scene, project, concept)
    - Persistence via JSON serialization

Use Cases:
    - "What scenes involve Ravi?" → Traverse from Ravi node
    - "Everything Boss discussed about climax" → Follow discuss edges
    - "Characters in Gusagusalu" → Query by type + project

Brain Inspiration:
    Semantic memory - conceptual relationships between entities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None

from memory.config import get_memory_config

LOGGER = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of entities in the knowledge graph"""

    CHARACTER = "character"  # Ravi, Father, etc.
    SCENE = "scene"  # Scene 5, Climax, etc.
    PROJECT = "project"  # Gusagusalu, Kitchen
    CONCEPT = "concept"  # Emotion, confrontation, arc
    PERSON = "person"  # Real people (Boss)
    EVENT = "event"  # Deadlines, milestones
    LOCATION = "location"  # Writers room, Kitchen


class RelationType(str, Enum):
    """Types of relationships between entities"""

    DISCUSSES = "discusses"  # Boss discusses climax
    CONTAINS = "contains"  # Scene contains confrontation
    RELATES_TO = "relates_to"  # Generic relationship
    CHARACTER_IN = "character_in"  # Ravi in Gusagusalu
    SCENE_IN = "scene_in"  # Scene in project
    HAS_RELATIONSHIP = "has_relationship"  # Ravi has father
    CREATES = "creates"  # Boss creates scene
    WANTS = "wants"  # Boss wants more emotion
    DEADLINE_FOR = "deadline_for"  # March is deadline for Gusagusalu
    INVOLVES = "involves"  # Confrontation involves Ravi


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph"""

    id: str  # Unique identifier
    name: str  # Display name
    node_type: NodeType  # Type of entity
    attributes: Dict[str, Any] = field(default_factory=dict)
    project: Optional[str] = None  # Associated project
    created_at: datetime = field(default_factory=datetime.now)
    source_memory_ids: List[str] = field(default_factory=list)  # LTM links

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "attributes": self.attributes,
            "project": self.project,
            "created_at": self.created_at.isoformat(),
            "source_memory_ids": self.source_memory_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeNode":
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=NodeType(data["node_type"]),
            attributes=data.get("attributes", {}),
            project=data.get("project"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            source_memory_ids=data.get("source_memory_ids", []),
        )


@dataclass
class KnowledgeEdge:
    """An edge (relationship) in the knowledge graph"""

    source_id: str
    target_id: str
    relation: RelationType
    weight: float = 1.0
    context: str = ""  # Original text context
    created_at: datetime = field(default_factory=datetime.now)
    source_memory_id: Optional[str] = None  # LTM link

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation.value,
            "weight": self.weight,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "source_memory_id": self.source_memory_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEdge":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation=RelationType(data["relation"]),
            weight=data.get("weight", 1.0),
            context=data.get("context", ""),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            source_memory_id=data.get("source_memory_id"),
        )


@dataclass
class TraversalResult:
    """Result from graph traversal"""

    node: KnowledgeNode
    path: List[str]  # Path from start node
    relations: List[str]  # Relations along path
    depth: int  # Distance from start


class KnowledgeGraph:
    """
    NetworkX-based knowledge graph for Friday's memory.

    Stores entities and relationships extracted from conversations.
    Enables queries like:
        - "What scenes involve Ravi?"
        - "Everything discussed about the climax"
        - "Characters in Gusagusalu"

    Usage:
        graph = KnowledgeGraph()
        await graph.initialize()

        # Add entities
        await graph.add_node("ravi", "Ravi", NodeType.CHARACTER, project="gusagusalu")
        await graph.add_node("climax", "Climax Scene", NodeType.SCENE, project="gusagusalu")

        # Add relationship
        await graph.add_edge("ravi", "climax", RelationType.CHARACTER_IN)

        # Traverse
        results = await graph.traverse("ravi", max_depth=2)

        # Query
        characters = await graph.get_by_type(NodeType.CHARACTER, project="gusagusalu")
    """

    def __init__(self, persist_path: Optional[Path] = None):
        if not HAS_NETWORKX:
            raise ImportError(
                "networkx is required for KnowledgeGraph. Install with: pip install networkx"
            )

        config = get_memory_config()
        self._persist_path = (
            persist_path or Path(config.stm.db_path).parent / "knowledge_graph.json"
        )
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize and load persisted graph"""
        if self._initialized:
            return

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)

        if self._persist_path.exists():
            self._load()

        self._initialized = True
        LOGGER.info(
            "Knowledge graph initialized: %d nodes, %d edges",
            len(self._nodes),
            self._graph.number_of_edges(),
        )

    def _load(self) -> None:
        """Load graph from JSON file"""
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load nodes
            for node_data in data.get("nodes", []):
                node = KnowledgeNode.from_dict(node_data)
                self._nodes[node.id] = node
                self._graph.add_node(node.id, **node.to_dict())

            # Load edges
            for edge_data in data.get("edges", []):
                edge = KnowledgeEdge.from_dict(edge_data)
                self._graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    relation=edge.relation.value,
                    weight=edge.weight,
                    context=edge.context,
                    created_at=edge.created_at.isoformat(),
                    source_memory_id=edge.source_memory_id,
                )

            LOGGER.info("Loaded knowledge graph from %s", self._persist_path)

        except Exception as e:
            LOGGER.error("Failed to load knowledge graph: %s", e)
            self._graph = nx.DiGraph()
            self._nodes = {}

    def _save(self) -> None:
        """Save graph to JSON file"""
        nodes = [node.to_dict() for node in self._nodes.values()]

        edges = []
        for source, target, data in self._graph.edges(data=True):
            edges.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "relation": data.get("relation", "relates_to"),
                    "weight": data.get("weight", 1.0),
                    "context": data.get("context", ""),
                    "created_at": data.get("created_at", datetime.now().isoformat()),
                    "source_memory_id": data.get("source_memory_id"),
                }
            )

        with open(self._persist_path, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)

    async def close(self) -> None:
        """Save and close"""
        self._save()
        LOGGER.info("Knowledge graph saved")

    # =========================================================================
    # Node Operations
    # =========================================================================

    def _normalize_id(self, name: str) -> str:
        """Normalize name to ID (lowercase, underscores)"""
        return name.lower().strip().replace(" ", "_").replace("-", "_")

    async def add_node(
        self,
        name: str,
        node_type: NodeType,
        node_id: Optional[str] = None,
        project: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        source_memory_id: Optional[str] = None,
    ) -> KnowledgeNode:
        """
        Add or update a node in the graph.

        Args:
            name: Display name
            node_type: Type of entity
            node_id: Optional custom ID (auto-generated from name if not provided)
            project: Associated project
            attributes: Additional attributes
            source_memory_id: Link to LTM entry

        Returns:
            Created or updated KnowledgeNode
        """
        node_id = node_id or self._normalize_id(name)

        if node_id in self._nodes:
            # Update existing node
            node = self._nodes[node_id]
            if attributes:
                node.attributes.update(attributes)
            if source_memory_id and source_memory_id not in node.source_memory_ids:
                node.source_memory_ids.append(source_memory_id)
            if project and not node.project:
                node.project = project
        else:
            # Create new node
            node = KnowledgeNode(
                id=node_id,
                name=name,
                node_type=node_type,
                project=project,
                attributes=attributes or {},
                source_memory_ids=[source_memory_id] if source_memory_id else [],
            )
            self._nodes[node_id] = node
            self._graph.add_node(node_id, **node.to_dict())

        self._save()
        LOGGER.debug("Added/updated node: %s (%s)", name, node_type.value)
        return node

    async def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID"""
        node_id = self._normalize_id(node_id)
        return self._nodes.get(node_id)

    async def ensure_node(
        self,
        name: str,
        node_type: NodeType = NodeType.CONCEPT,
        project: Optional[str] = None,
    ) -> KnowledgeNode:
        """
        Ensure a node exists, creating if necessary.

        Uses CONCEPT type by default for auto-created nodes.
        """
        node_id = self._normalize_id(name)

        if node_id in self._nodes:
            return self._nodes[node_id]

        return await self.add_node(name, node_type, project=project)

    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its edges"""
        node_id = self._normalize_id(node_id)

        if node_id not in self._nodes:
            return False

        del self._nodes[node_id]
        self._graph.remove_node(node_id)
        self._save()

        LOGGER.debug("Deleted node: %s", node_id)
        return True

    # =========================================================================
    # Edge Operations
    # =========================================================================

    async def add_edge(
        self,
        source: str,
        target: str,
        relation: RelationType,
        weight: float = 1.0,
        context: str = "",
        source_memory_id: Optional[str] = None,
    ) -> KnowledgeEdge:
        """
        Add an edge (relationship) between two nodes.

        Creates nodes if they don't exist.

        Args:
            source: Source node name/ID
            target: Target node name/ID
            relation: Type of relationship
            weight: Relationship strength
            context: Original text context
            source_memory_id: Link to LTM entry

        Returns:
            Created KnowledgeEdge
        """
        source_id = self._normalize_id(source)
        target_id = self._normalize_id(target)

        # Ensure nodes exist
        if source_id not in self._nodes:
            await self.add_node(source, NodeType.CONCEPT)
        if target_id not in self._nodes:
            await self.add_node(target, NodeType.CONCEPT)

        # Check if edge already exists
        if self._graph.has_edge(source_id, target_id):
            # Update weight (strengthen connection)
            existing = self._graph[source_id][target_id]
            existing["weight"] = existing.get("weight", 1.0) + 0.1
            if context and context not in existing.get("context", ""):
                existing["context"] = f"{existing.get('context', '')} | {context}"
        else:
            # Create new edge
            self._graph.add_edge(
                source_id,
                target_id,
                relation=relation.value,
                weight=weight,
                context=context,
                created_at=datetime.now().isoformat(),
                source_memory_id=source_memory_id,
            )

        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            context=context,
            source_memory_id=source_memory_id,
        )

        self._save()
        LOGGER.debug("Added edge: %s -[%s]-> %s", source, relation.value, target)
        return edge

    async def add_triplet(
        self,
        subject: str,
        relation: str,
        obj: str,
        context: str = "",
        source_memory_id: Optional[str] = None,
    ) -> KnowledgeEdge:
        """
        Add a triplet (subject-relation-object) to the graph.

        Convenience method for adding extracted triplets.

        Args:
            subject: Subject entity
            relation: Relationship string (will be mapped to RelationType)
            obj: Object entity
            context: Original text
            source_memory_id: Link to LTM

        Returns:
            Created edge
        """
        # Map relation string to RelationType
        relation_map = {
            "discusses": RelationType.DISCUSSES,
            "contains": RelationType.CONTAINS,
            "relates_to": RelationType.RELATES_TO,
            "character_in": RelationType.CHARACTER_IN,
            "scene_in": RelationType.SCENE_IN,
            "has_relationship": RelationType.HAS_RELATIONSHIP,
            "creates": RelationType.CREATES,
            "wants": RelationType.WANTS,
            "deadline_for": RelationType.DEADLINE_FOR,
            "involves": RelationType.INVOLVES,
        }

        relation_type = relation_map.get(
            relation.lower().replace(" ", "_"), RelationType.RELATES_TO
        )

        return await self.add_edge(
            subject,
            obj,
            relation_type,
            context=context,
            source_memory_id=source_memory_id,
        )

    async def add_triplets(
        self,
        triplets: List[Tuple[str, str, str]],
        context: str = "",
        source_memory_id: Optional[str] = None,
    ) -> List[KnowledgeEdge]:
        """Add multiple triplets at once"""
        edges = []
        for subject, relation, obj in triplets:
            edge = await self.add_triplet(
                subject,
                relation,
                obj,
                context=context,
                source_memory_id=source_memory_id,
            )
            edges.append(edge)
        return edges

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def traverse(
        self,
        start: str,
        max_depth: int = 2,
        relation_filter: Optional[List[RelationType]] = None,
    ) -> List[TraversalResult]:
        """
        Traverse graph from a starting node.

        Args:
            start: Starting node name/ID
            max_depth: Maximum traversal depth
            relation_filter: Only follow these relation types

        Returns:
            List of TraversalResult with reached nodes
        """
        start_id = self._normalize_id(start)

        if start_id not in self._nodes:
            return []

        results: List[TraversalResult] = []
        visited: Set[str] = {start_id}
        queue: List[Tuple[str, List[str], List[str], int]] = [
            (start_id, [start_id], [], 0)
        ]

        while queue:
            current_id, path, relations, depth = queue.pop(0)

            if depth > 0:  # Don't include start node
                node = self._nodes.get(current_id)
                if node:
                    results.append(
                        TraversalResult(
                            node=node,
                            path=path,
                            relations=relations,
                            depth=depth,
                        )
                    )

            if depth >= max_depth:
                continue

            # Explore neighbors
            for neighbor_id in self._graph.neighbors(current_id):
                if neighbor_id in visited:
                    continue

                edge_data = self._graph[current_id][neighbor_id]
                edge_relation = edge_data.get("relation", "relates_to")

                # Apply relation filter
                if relation_filter:
                    if edge_relation not in [r.value for r in relation_filter]:
                        continue

                visited.add(neighbor_id)
                queue.append(
                    (
                        neighbor_id,
                        path + [neighbor_id],
                        relations + [edge_relation],
                        depth + 1,
                    )
                )

        return results

    async def get_by_type(
        self,
        node_type: NodeType,
        project: Optional[str] = None,
    ) -> List[KnowledgeNode]:
        """Get all nodes of a specific type"""
        results = []
        for node in self._nodes.values():
            if node.node_type == node_type:
                if project is None or node.project == project:
                    results.append(node)
        return results

    async def get_related(
        self,
        node_id: str,
        relation: Optional[RelationType] = None,
    ) -> List[Tuple[KnowledgeNode, str]]:
        """
        Get nodes directly related to a given node.

        Returns:
            List of (node, relation_type) tuples
        """
        node_id = self._normalize_id(node_id)

        if node_id not in self._graph:
            return []

        results = []

        # Outgoing edges
        for neighbor_id in self._graph.neighbors(node_id):
            edge_data = self._graph[node_id][neighbor_id]
            edge_relation = edge_data.get("relation", "relates_to")

            if relation and edge_relation != relation.value:
                continue

            neighbor = self._nodes.get(neighbor_id)
            if neighbor:
                results.append((neighbor, edge_relation))

        # Incoming edges
        for predecessor_id in self._graph.predecessors(node_id):
            edge_data = self._graph[predecessor_id][node_id]
            edge_relation = edge_data.get("relation", "relates_to")

            if relation and edge_relation != relation.value:
                continue

            predecessor = self._nodes.get(predecessor_id)
            if predecessor:
                results.append((predecessor, f"reverse_{edge_relation}"))

        return results

    async def find_path(
        self,
        source: str,
        target: str,
    ) -> Optional[List[str]]:
        """
        Find shortest path between two nodes.

        Returns:
            List of node IDs in path, or None if no path exists
        """
        source_id = self._normalize_id(source)
        target_id = self._normalize_id(target)

        try:
            path = nx.shortest_path(self._graph, source_id, target_id)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    async def search_by_attribute(
        self,
        key: str,
        value: Any,
    ) -> List[KnowledgeNode]:
        """Search nodes by attribute value"""
        results = []
        for node in self._nodes.values():
            if node.attributes.get(key) == value:
                results.append(node)
        return results

    # =========================================================================
    # Maintenance Operations
    # =========================================================================

    async def get_orphan_nodes(self) -> List[KnowledgeNode]:
        """Find nodes with no edges"""
        orphans = []
        for node_id, node in self._nodes.items():
            if self._graph.degree(node_id) == 0:
                orphans.append(node)
        return orphans

    async def cleanup_orphans(self, min_age_days: int = 7) -> int:
        """Remove orphan nodes older than threshold"""
        orphans = await self.get_orphan_nodes()
        cutoff = datetime.now()
        deleted = 0

        for node in orphans:
            age_days = (cutoff - node.created_at).days
            if age_days >= min_age_days:
                await self.delete_node(node.id)
                deleted += 1

        if deleted:
            LOGGER.info("Cleaned up %d orphan nodes", deleted)

        return deleted

    async def merge_duplicates(self, similarity_threshold: float = 0.9) -> int:
        """
        Merge nodes that likely represent the same entity.

        Uses name similarity for detection.
        """
        # Simple approach: exact name match after normalization
        name_to_ids: Dict[str, List[str]] = {}

        for node_id, node in self._nodes.items():
            normalized = self._normalize_id(node.name)
            if normalized not in name_to_ids:
                name_to_ids[normalized] = []
            name_to_ids[normalized].append(node_id)

        merged = 0
        for normalized, ids in name_to_ids.items():
            if len(ids) > 1:
                # Keep the first, merge others into it
                primary_id = ids[0]
                for duplicate_id in ids[1:]:
                    await self._merge_nodes(primary_id, duplicate_id)
                    merged += 1

        if merged:
            LOGGER.info("Merged %d duplicate nodes", merged)

        return merged

    async def _merge_nodes(self, keep_id: str, merge_id: str) -> None:
        """Merge one node into another"""
        keep_node = self._nodes.get(keep_id)
        merge_node = self._nodes.get(merge_id)

        if not keep_node or not merge_node:
            return

        # Merge attributes
        keep_node.attributes.update(merge_node.attributes)

        # Merge source memory IDs
        for mid in merge_node.source_memory_ids:
            if mid not in keep_node.source_memory_ids:
                keep_node.source_memory_ids.append(mid)

        # Redirect edges
        for predecessor in list(self._graph.predecessors(merge_id)):
            edge_data = self._graph[predecessor][merge_id]
            if not self._graph.has_edge(predecessor, keep_id):
                self._graph.add_edge(predecessor, keep_id, **edge_data)

        for successor in list(self._graph.neighbors(merge_id)):
            edge_data = self._graph[merge_id][successor]
            if not self._graph.has_edge(keep_id, successor):
                self._graph.add_edge(keep_id, successor, **edge_data)

        # Delete merged node
        del self._nodes[merge_id]
        self._graph.remove_node(merge_id)

        self._save()

    # =========================================================================
    # Stats & Export
    # =========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        type_counts = {}
        for node in self._nodes.values():
            t = node.node_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        relation_counts = {}
        for _, _, data in self._graph.edges(data=True):
            r = data.get("relation", "unknown")
            relation_counts[r] = relation_counts.get(r, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": self._graph.number_of_edges(),
            "nodes_by_type": type_counts,
            "edges_by_relation": relation_counts,
            "is_connected": (
                nx.is_weakly_connected(self._graph) if len(self._nodes) > 0 else True
            ),
            "density": nx.density(self._graph) if len(self._nodes) > 0 else 0,
        }

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes={len(self._nodes)}, edges={self._graph.number_of_edges()})"
