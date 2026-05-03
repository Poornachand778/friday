"""
Knowledge Graph Integration for Book Understanding
===================================================

Connects extracted book knowledge to Friday's Knowledge Graph,
enabling relationship queries across books and projects.

When a book is comprehended, this module:
1. Creates a REFERENCE node for the book itself
2. Adds CONCEPT nodes for all extracted concepts
3. Creates relationship edges (defines, teaches, demonstrates)
4. Links book concepts to existing project concepts

This enables queries like:
- "What books discuss character arcs?"
- "Find all techniques related to tension"
- "What does McKee's book say about the concept in my script?"
"""

from __future__ import annotations

import logging
from typing import List, Optional

from documents.understanding.models import (
    BookUnderstanding,
    Concept,
    Principle,
    Technique,
    BookExample,
)

LOGGER = logging.getLogger(__name__)

# Extended node types for book knowledge
# These augment the base NodeType enum from memory/layers/knowledge_graph.py
BOOK_NODE_TYPES = {
    "reference": "reference",  # The book itself
    "book_concept": "book_concept",  # Concept defined in a book
    "book_principle": "book_principle",  # Principle taught by a book
    "book_technique": "book_technique",  # Technique described in a book
    "book_example": "book_example",  # Example/case study from a book
}

# Extended relation types for book knowledge
BOOK_RELATION_TYPES = {
    "defines": "defines",  # Book defines concept
    "teaches": "teaches",  # Book teaches principle
    "describes": "describes",  # Book describes technique
    "cites": "cites",  # Book cites example
    "related_concept": "related_concept",  # Concept relates to another
    "applies_to": "applies_to",  # Principle applies to domain
    "demonstrates": "demonstrates",  # Example demonstrates concept
    "is_similar_to": "is_similar_to",  # Technique similar to another
    "prerequisite_for": "prerequisite_for",  # Concept is prerequisite
}


class BookGraphIntegrator:
    """
    Integrates book understanding into Friday's Knowledge Graph.

    Creates nodes for book concepts and edges for their relationships,
    enabling semantic queries across books and creative projects.
    """

    def __init__(self, knowledge_graph):
        """
        Initialize integrator.

        Args:
            knowledge_graph: Instance of KnowledgeGraph from memory.layers
        """
        self._graph = knowledge_graph

    async def integrate_book(
        self,
        understanding: BookUnderstanding,
        link_to_project: Optional[str] = None,
    ) -> dict:
        """
        Integrate a book understanding into the knowledge graph.

        Creates:
        - A REFERENCE node for the book
        - CONCEPT nodes for each concept
        - PRINCIPLE nodes for each principle (as CONCEPT type)
        - TECHNIQUE nodes for each technique
        - Edges for all relationships

        Args:
            understanding: The BookUnderstanding to integrate
            link_to_project: Optional project to link book knowledge to

        Returns:
            Stats about what was added
        """
        stats = {
            "nodes_created": 0,
            "edges_created": 0,
            "concepts_added": 0,
            "principles_added": 0,
            "techniques_added": 0,
            "examples_added": 0,
        }

        # 1. Create book reference node
        book_node = await self._graph.add_node(
            name=understanding.title,
            node_type="concept",  # Use CONCEPT since that's what exists
            node_id=f"book_{understanding.id[:8]}",
            project=link_to_project,
            attributes={
                "type": "reference",
                "author": understanding.author,
                "summary": understanding.summary[:500] if understanding.summary else "",
                "main_argument": understanding.main_argument,
                "domains": understanding.domains,
                "document_id": understanding.document_id,
            },
        )
        stats["nodes_created"] += 1
        LOGGER.info("Added book node: %s", understanding.title)

        # 2. Add concepts
        for concept in understanding.concepts:
            await self._add_concept(book_node.id, concept, link_to_project, stats)

        # 3. Add principles (as concept nodes with principle attributes)
        for principle in understanding.principles:
            await self._add_principle(book_node.id, principle, link_to_project, stats)

        # 4. Add techniques
        for technique in understanding.techniques:
            await self._add_technique(book_node.id, technique, link_to_project, stats)

        # 5. Add examples
        for example in understanding.examples:
            await self._add_example(book_node.id, example, link_to_project, stats)

        # 6. Create cross-concept relationships
        await self._create_concept_relationships(understanding, stats)

        LOGGER.info(
            "Integrated '%s': %d nodes, %d edges",
            understanding.title,
            stats["nodes_created"],
            stats["edges_created"],
        )

        return stats

    async def _add_concept(
        self,
        book_node_id: str,
        concept: Concept,
        project: Optional[str],
        stats: dict,
    ) -> str:
        """Add a concept node and link to book"""
        # Create concept node
        node = await self._graph.add_node(
            name=concept.name,
            node_type="concept",
            node_id=f"concept_{concept.id[:8]}",
            project=project,
            attributes={
                "type": "book_concept",
                "definition": concept.definition,
                "importance": concept.importance,
                "source_pages": concept.source_pages,
                "domain": concept.domain,
                "document_id": concept.source_document_id,
            },
        )
        stats["nodes_created"] += 1
        stats["concepts_added"] += 1

        # Link to book (book "defines" concept)
        await self._graph.add_edge(
            book_node_id,
            node.id,
            "relates_to",  # Using existing relation type
            context=f"defines: {concept.definition[:100]}",
        )
        stats["edges_created"] += 1

        # Add synonyms as separate linked nodes
        for synonym in concept.synonyms:
            synonym_node = await self._graph.add_node(
                name=synonym,
                node_type="concept",
                project=project,
                attributes={"type": "synonym", "synonym_of": concept.name},
            )
            await self._graph.add_edge(
                node.id,
                synonym_node.id,
                "relates_to",
                context="synonym",
            )
            stats["nodes_created"] += 1
            stats["edges_created"] += 1

        return node.id

    async def _add_principle(
        self,
        book_node_id: str,
        principle: Principle,
        project: Optional[str],
        stats: dict,
    ) -> str:
        """Add a principle node and link to book"""
        node = await self._graph.add_node(
            name=principle.statement[:50] + "...",  # Truncate for node name
            node_type="concept",
            node_id=f"principle_{principle.id[:8]}",
            project=project,
            attributes={
                "type": "book_principle",
                "statement": principle.statement,
                "rationale": principle.rationale,
                "confidence_level": principle.confidence_level.value,
                "applies_to": principle.applies_to,
                "exceptions": principle.exceptions,
                "check_question": principle.check_question,
                "source_page": principle.source_page,
                "document_id": principle.source_document_id,
            },
        )
        stats["nodes_created"] += 1
        stats["principles_added"] += 1

        # Link to book (book "teaches" principle)
        await self._graph.add_edge(
            book_node_id,
            node.id,
            "relates_to",
            context=f"teaches: {principle.statement[:100]}",
        )
        stats["edges_created"] += 1

        # Link to applicable domains/contexts
        for domain in principle.applies_to:
            domain_node = await self._graph.ensure_node(domain, "concept", project)
            await self._graph.add_edge(
                node.id,
                domain_node.id,
                "relates_to",
                context="applies_to",
            )
            stats["edges_created"] += 1

        return node.id

    async def _add_technique(
        self,
        book_node_id: str,
        technique: Technique,
        project: Optional[str],
        stats: dict,
    ) -> str:
        """Add a technique node and link to book"""
        node = await self._graph.add_node(
            name=technique.name,
            node_type="concept",
            node_id=f"technique_{technique.id[:8]}",
            project=project,
            attributes={
                "type": "book_technique",
                "description": technique.description,
                "steps": technique.steps,
                "when_to_use": technique.when_to_use,
                "when_not_to_use": technique.when_not_to_use,
                "example_films": technique.example_films,
                "difficulty": technique.difficulty,
                "source_page": technique.source_page,
                "document_id": technique.source_document_id,
            },
        )
        stats["nodes_created"] += 1
        stats["techniques_added"] += 1

        # Link to book (book "describes" technique)
        await self._graph.add_edge(
            book_node_id,
            node.id,
            "relates_to",
            context=f"describes technique: {technique.description[:100]}",
        )
        stats["edges_created"] += 1

        # Link to use cases
        for use_case in technique.use_cases:
            use_case_node = await self._graph.ensure_node(use_case, "concept", project)
            await self._graph.add_edge(
                node.id,
                use_case_node.id,
                "relates_to",
                context="use_case",
            )
            stats["edges_created"] += 1

        # Link to example films
        for film in technique.example_films:
            film_node = await self._graph.ensure_node(film, "concept", project)
            film_node.attributes["type"] = "film_reference"
            await self._graph.add_edge(
                node.id,
                film_node.id,
                "relates_to",
                context="demonstrated_in",
            )
            stats["edges_created"] += 1

        return node.id

    async def _add_example(
        self,
        book_node_id: str,
        example: BookExample,
        project: Optional[str],
        stats: dict,
    ) -> str:
        """Add an example node and link to book"""
        node = await self._graph.add_node(
            name=f"{example.work_title} - {example.scene_or_section or 'reference'}",
            node_type="concept",
            node_id=f"example_{example.id[:8]}",
            project=project,
            attributes={
                "type": "book_example",
                "work_title": example.work_title,
                "work_type": example.work_type,
                "scene_or_section": example.scene_or_section,
                "description": example.description,
                "lesson": example.lesson,
                "what_works": example.what_works,
                "situation_type": example.situation_type,
                "emotional_beat": example.emotional_beat,
                "source_page": example.source_page,
                "document_id": example.source_document_id,
            },
        )
        stats["nodes_created"] += 1
        stats["examples_added"] += 1

        # Link to book (book "cites" example)
        await self._graph.add_edge(
            book_node_id,
            node.id,
            "relates_to",
            context=f"cites: {example.lesson[:100]}",
        )
        stats["edges_created"] += 1

        # Link to what it demonstrates
        for concept_name in example.demonstrates_concept:
            concept_node = await self._graph.ensure_node(
                concept_name, "concept", project
            )
            await self._graph.add_edge(
                node.id,
                concept_node.id,
                "relates_to",
                context="demonstrates",
            )
            stats["edges_created"] += 1

        return node.id

    async def _create_concept_relationships(
        self,
        understanding: BookUnderstanding,
        stats: dict,
    ) -> None:
        """Create edges between related concepts within the book"""
        # Map concept names to their node IDs
        concept_nodes = {}
        for concept in understanding.concepts:
            concept_nodes[concept.name.lower()] = f"concept_{concept.id[:8]}"

        # Create related_concept edges
        for concept in understanding.concepts:
            source_id = f"concept_{concept.id[:8]}"
            for related_name in concept.related_concepts:
                related_key = related_name.lower()
                if related_key in concept_nodes:
                    target_id = concept_nodes[related_key]
                    await self._graph.add_edge(
                        source_id,
                        target_id,
                        "relates_to",
                        context="related_concept",
                    )
                    stats["edges_created"] += 1

        # Link techniques to related concepts
        for technique in understanding.techniques:
            tech_id = f"technique_{technique.id[:8]}"
            for concept_name in technique.related_concepts:
                concept_key = concept_name.lower()
                if concept_key in concept_nodes:
                    await self._graph.add_edge(
                        tech_id,
                        concept_nodes[concept_key],
                        "relates_to",
                        context="uses_concept",
                    )
                    stats["edges_created"] += 1

    async def find_books_about(self, concept_name: str) -> List[dict]:
        """
        Find books that discuss a given concept.

        Returns list of book info dicts.
        """
        # Find the concept node
        results = await self._graph.traverse(concept_name, max_depth=2)

        books = []
        for result in results:
            attrs = result.node.attributes
            if attrs.get("type") == "reference":
                books.append(
                    {
                        "title": result.node.name,
                        "author": attrs.get("author", "Unknown"),
                        "document_id": attrs.get("document_id"),
                        "connection": " -> ".join(result.path),
                    }
                )

        return books

    async def get_related_knowledge(
        self,
        topic: str,
        knowledge_types: Optional[List[str]] = None,
    ) -> dict:
        """
        Get all related knowledge for a topic.

        Args:
            topic: The topic to search for
            knowledge_types: Filter by type (concept, principle, technique, example)

        Returns:
            Dict with concepts, principles, techniques, examples
        """
        results = await self._graph.traverse(topic, max_depth=2)

        knowledge = {
            "concepts": [],
            "principles": [],
            "techniques": [],
            "examples": [],
        }

        type_map = {
            "book_concept": "concepts",
            "book_principle": "principles",
            "book_technique": "techniques",
            "book_example": "examples",
        }

        for result in results:
            node_type = result.node.attributes.get("type")
            if node_type in type_map:
                if knowledge_types is None or node_type in knowledge_types:
                    knowledge[type_map[node_type]].append(
                        {
                            "name": result.node.name,
                            "attributes": result.node.attributes,
                            "depth": result.depth,
                        }
                    )

        return knowledge

    async def link_to_project_concept(
        self,
        book_concept_id: str,
        project_concept_id: str,
        relationship: str = "relates_to",
    ) -> None:
        """
        Create a link between a book concept and a project concept.

        This enables queries like "What do the books say about the confrontation
        scene in my script?"
        """
        await self._graph.add_edge(
            book_concept_id,
            project_concept_id,
            relationship,
            context="book_to_project_link",
        )
        LOGGER.info(
            "Linked book concept %s to project concept %s",
            book_concept_id,
            project_concept_id,
        )
