"""
Memory Manager
==============

Central coordinator for Friday's memory system.

Coordinates:
    - Working Memory (active context)
    - Short-Term Memory (7 days)
    - Long-Term Memory (permanent)
    - Knowledge Graph (Cognee-inspired relationships)
    - Profile Store (identity)
    - Telugu-English processing

Usage:
    from memory import MemoryManager

    manager = MemoryManager()
    await manager.initialize()

    # Store conversation turn
    await manager.store_turn(user_msg, assistant_response, session_id)

    # Search across all memory
    results = await manager.search("climax scene")

    # Graph queries
    results = await manager.graph_query("ravi", max_depth=2)

    # Voice commands
    await manager.voice_command("remember this: Ravi hesitates")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from memory.config import MemorySystemConfig, get_memory_config
from memory.layers.working import WorkingMemory, ConversationTurn, PrefetchedMemory
from memory.layers.short_term import ShortTermMemory, STMEntry
from memory.layers.long_term import LongTermMemory, LTMEntry, MemoryType
from memory.layers.profile import ProfileStore
from memory.layers.knowledge_graph import (
    KnowledgeGraph,
    NodeType,
    RelationType,
    TraversalResult,
)
from memory.telugu.processor import TeluguEnglishProcessor
from memory.operations.triplet_extractor import TripletExtractor

LOGGER = logging.getLogger(__name__)


class MemoryManager:
    """
    Central memory coordinator.

    Manages all memory layers and provides unified interface
    for storage, retrieval, and voice commands.

    Architecture:
        ┌───────────────────────────────────────────────────────┐
        │                  MemoryManager                         │
        ├───────────────────────────────────────────────────────┤
        │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐ │
        │  │ Working │  │  STM    │  │  LTM    │  │ Knowledge│ │
        │  │ Memory  │  │ (7 day) │  │(Perm)   │  │  Graph   │ │
        │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬─────┘ │
        │       │            │            │            │        │
        │       └────────────┼────────────┴────────────┘        │
        │                    │                                  │
        │             ┌──────┴──────┐                           │
        │             │   Profile   │                           │
        │             │   Store     │                           │
        │             └─────────────┘                           │
        └───────────────────────────────────────────────────────┘

    Knowledge Graph (Cognee-inspired):
        - Stores entity relationships (character-scene, person-project)
        - Enables "What scenes involve Ravi?" type queries
        - Triplet extraction via GLM-4.7-Flash during LTM consolidation

    Usage:
        manager = MemoryManager()
        await manager.initialize()

        # Process conversation
        await manager.store_turn(
            user_message="What about the climax?",
            assistant_response="The climax needs more punch.",
            session_id="session123",
        )

        # Search
        results = await manager.search("climax")

        # Graph query
        related = await manager.graph_query("ravi")
    """

    def __init__(self, config: Optional[MemorySystemConfig] = None):
        self.config = config or get_memory_config()

        # Memory layers
        self._working: Optional[WorkingMemory] = None
        self._stm: Optional[ShortTermMemory] = None
        self._ltm: Optional[LongTermMemory] = None
        self._profile: Optional[ProfileStore] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None

        # Triplet extractor (lazy extraction on LTM consolidation)
        self._triplet_extractor: Optional[TripletExtractor] = None

        # Telugu processor
        self._telugu = TeluguEnglishProcessor()

        # State
        self._initialized = False
        self._current_session_id: Optional[str] = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def working(self) -> WorkingMemory:
        """Get working memory"""
        if not self._working:
            raise RuntimeError("MemoryManager not initialized")
        return self._working

    @property
    def stm(self) -> ShortTermMemory:
        """Get short-term memory"""
        if not self._stm:
            raise RuntimeError("MemoryManager not initialized")
        return self._stm

    @property
    def ltm(self) -> LongTermMemory:
        """Get long-term memory"""
        if not self._ltm:
            raise RuntimeError("MemoryManager not initialized")
        return self._ltm

    @property
    def profile(self) -> ProfileStore:
        """Get profile store"""
        if not self._profile:
            raise RuntimeError("MemoryManager not initialized")
        return self._profile

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        """Get knowledge graph"""
        if not self._knowledge_graph:
            raise RuntimeError("MemoryManager not initialized")
        return self._knowledge_graph

    async def initialize(self) -> None:
        """Initialize all memory layers"""
        if self._initialized:
            return

        LOGGER.info("Initializing Memory Manager...")

        # Initialize layers
        self._working = WorkingMemory(self.config.working)

        self._stm = ShortTermMemory(self.config.stm)
        await self._stm.initialize()

        self._ltm = LongTermMemory(self.config.ltm)
        await self._ltm.initialize()

        self._profile = ProfileStore(self.config.profile)
        await self._profile.initialize()

        # Initialize knowledge graph (Cognee-inspired)
        self._knowledge_graph = KnowledgeGraph()
        await self._knowledge_graph.initialize()

        # Initialize triplet extractor (for lazy extraction)
        self._triplet_extractor = TripletExtractor()
        LOGGER.info(
            "Triplet extractor configured: %s",
            "GLM" if self._triplet_extractor.is_configured else "fallback",
        )

        self._initialized = True
        LOGGER.info("Memory Manager initialized")

    async def shutdown(self) -> None:
        """Shutdown memory system"""
        LOGGER.info("Shutting down Memory Manager...")

        if self._stm:
            await self._stm.close()

        if self._ltm:
            await self._ltm.close()

        if self._knowledge_graph:
            await self._knowledge_graph.close()

        if self._triplet_extractor:
            await self._triplet_extractor.close()

        self._initialized = False
        LOGGER.info("Memory Manager shutdown complete")

    # =========================================================================
    # Session Management
    # =========================================================================

    def start_session(self, session_id: str) -> None:
        """Start a new conversation session"""
        self._current_session_id = session_id
        self._working.clear()
        LOGGER.info("Started session: %s", session_id[:8])

    def get_session_id(self) -> Optional[str]:
        """Get current session ID"""
        return self._current_session_id

    # =========================================================================
    # Conversation Processing
    # =========================================================================

    async def store_turn(
        self,
        user_message: str,
        assistant_response: str,
        session_id: Optional[str] = None,
        tool_calls: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
        context_type: Optional[str] = None,
    ) -> ConversationTurn:
        """
        Store a conversation turn.

        Adds to working memory and triggers background storage.

        Args:
            user_message: User's message
            assistant_response: Friday's response
            session_id: Session identifier
            tool_calls: Any tool calls made
            tool_results: Results from tool calls
            context_type: Current context (writers_room, etc.)

        Returns:
            The stored ConversationTurn
        """
        session_id = session_id or self._current_session_id
        if not session_id:
            session_id = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self._current_session_id = session_id

        # Process language
        user_processed = self._telugu.process(user_message)
        response_processed = self._telugu.process(assistant_response)

        # Detect dominant language mode
        combined_density = (
            user_processed.telugu_density + response_processed.telugu_density
        ) / 2
        if combined_density > 0.4:
            self._working.set_language_mode("te")
        elif combined_density > 0.1:
            self._working.set_language_mode("mixed")
        else:
            self._working.set_language_mode("en")

        # Add to working memory
        turn = self._working.add_turn(
            user_message=user_message,
            assistant_response=assistant_response,
            tool_calls=tool_calls,
            tool_results=tool_results,
            context_type=context_type or self._working.current_room,
        )

        # Update profile interaction time
        self._profile.record_interaction()

        LOGGER.debug(
            "Stored turn (session: %s, lang: %s)",
            session_id[:8],
            self._working.language_mode,
        )

        return turn

    async def end_session(
        self,
        session_id: Optional[str] = None,
        save_to_stm: bool = True,
    ) -> Optional[STMEntry]:
        """
        End a conversation session.

        Optionally saves session summary to STM.

        Args:
            session_id: Session to end
            save_to_stm: Whether to save summary to STM

        Returns:
            STM entry if saved, None otherwise
        """
        session_id = session_id or self._current_session_id
        if not session_id:
            return None

        entry = None
        if save_to_stm and self._working.turn_count > 0:
            entry = await self._save_session_to_stm(session_id)

        # Clear working memory
        self._working.clear()
        self._current_session_id = None

        LOGGER.info("Ended session: %s", session_id[:8])
        return entry

    async def _save_session_to_stm(self, session_id: str) -> STMEntry:
        """Save current session to short-term memory"""
        turns = self._working.get_turns()

        # Generate summary (simple for now, could use LLM)
        summary_parts = []
        for turn in turns[-5:]:  # Last 5 turns
            summary_parts.append(f"Discussed: {turn.user_message[:50]}...")

        summary = " | ".join(summary_parts)

        # Extract key facts (basic extraction)
        key_facts = []
        for turn in turns:
            if "remember" in turn.user_message.lower():
                key_facts.append(turn.user_message)
            if "important" in turn.user_message.lower():
                key_facts.append(turn.user_message)

        # Detect topics from attention stack
        topics = [item.topic for item in self._working.get_attention_topics()[:5]]

        # Get raw turns
        raw_turns = [
            {"user": t.user_message, "assistant": t.assistant_response} for t in turns
        ]

        entry = await self._stm.store(
            session_id=session_id,
            summary=summary,
            key_facts=key_facts,
            raw_turns=raw_turns,
            room=self._working.current_room,
            project=self._working.current_project,
            topics=topics,
            language=self._working.language_mode,
        )

        LOGGER.info("Saved session to STM: %s", entry.id[:8])
        return entry

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        top_k: int = 10,
        include_stm: bool = True,
        include_ltm: bool = True,
        include_working: bool = True,
        project: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search across all memory layers.

        Args:
            query: Search query
            top_k: Maximum results per layer
            include_stm: Search short-term memory
            include_ltm: Search long-term memory
            include_working: Search working memory
            project: Filter by project
            memory_type: Filter LTM by type

        Returns:
            List of results with source and relevance
        """
        results: List[Dict[str, Any]] = []

        # Search working memory (attention stack)
        if include_working:
            for item in self._working.get_attention_topics():
                if query.lower() in item.topic.lower():
                    results.append(
                        {
                            "source": "working",
                            "content": item.topic,
                            "relevance": item.relevance,
                            "metadata": item.metadata,
                        }
                    )

            # Check prefetched LTM
            for mem in self._working.get_prefetched_ltm():
                if query.lower() in mem.content.lower():
                    results.append(
                        {
                            "source": "prefetched",
                            "content": mem.content,
                            "relevance": mem.relevance,
                            "memory_id": mem.memory_id,
                        }
                    )

        # Search STM
        if include_stm:
            stm_results = await self._stm.search(query, top_k=top_k, project=project)
            for entry in stm_results:
                results.append(
                    {
                        "source": "stm",
                        "content": entry.summary,
                        "relevance": 0.7,  # STM relevance
                        "memory_id": entry.id,
                        "created_at": entry.created_at.isoformat(),
                        "topics": entry.topics,
                    }
                )

        # Search LTM
        if include_ltm:
            ltm_results = await self._ltm.search(
                query, top_k=top_k, project=project, memory_type=memory_type
            )
            for entry, similarity in ltm_results:
                results.append(
                    {
                        "source": "ltm",
                        "content": entry.content,
                        "relevance": similarity,
                        "memory_id": entry.id,
                        "memory_type": entry.memory_type.value,
                        "created_at": entry.created_at.isoformat(),
                    }
                )

        # Sort by relevance
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        return results[:top_k]

    async def prefetch_for_context(
        self,
        context: str,
        project: Optional[str] = None,
        top_k: int = 5,
    ) -> List[PrefetchedMemory]:
        """
        Prefetch relevant memories for current context.

        Called ahead of LLM generation for "think while talk".

        Args:
            context: Current conversation context
            project: Current project
            top_k: Number of memories to prefetch

        Returns:
            List of prefetched memories
        """
        results = await self._ltm.search(context, top_k=top_k, project=project)

        prefetched = []
        for entry, similarity in results:
            prefetched.append(
                PrefetchedMemory(
                    content=entry.content,
                    relevance=similarity,
                    memory_id=entry.id,
                    memory_type=entry.memory_type.value,
                )
            )

        self._working.set_prefetched_ltm(prefetched)

        LOGGER.debug("Prefetched %d memories for context", len(prefetched))
        return prefetched

    # =========================================================================
    # Direct Storage
    # =========================================================================

    async def store_fact(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.5,
        project: Optional[str] = None,
        event_date: Optional[datetime] = None,
        extract_triplets: bool = True,
    ) -> LTMEntry:
        """
        Store a fact directly to long-term memory.

        Also extracts and stores knowledge triplets to the graph
        when extract_triplets=True (default).

        Args:
            content: The fact to store
            memory_type: Type of memory
            importance: Importance score (0-1)
            project: Associated project
            event_date: Date of referenced event
            extract_triplets: Whether to extract and store graph triplets

        Returns:
            Created LTM entry
        """
        # Process language
        processed = self._telugu.process(content)

        entry = await self._ltm.store(
            content=content,
            memory_type=memory_type,
            importance=importance,
            project=project or self._working.current_project,
            event_date=event_date,
            language=processed.dominant_language,
            telugu_keywords=processed.telugu_keywords,
        )

        # Extract triplets and add to knowledge graph (lazy, Cognee-inspired)
        if extract_triplets and self._triplet_extractor:
            await self._extract_and_store_triplets(
                content,
                source_memory_id=entry.id,
                project=project or self._working.current_project,
            )

        LOGGER.info("Stored fact: %s", content[:50])
        return entry

    async def _extract_and_store_triplets(
        self,
        content: str,
        source_memory_id: Optional[str] = None,
        project: Optional[str] = None,
    ) -> int:
        """
        Extract triplets from content and add to knowledge graph.

        Returns:
            Number of triplets added
        """
        if not self._triplet_extractor or not self._knowledge_graph:
            return 0

        try:
            result = await self._triplet_extractor.extract(content, project=project)

            # Add each triplet to the graph
            for triplet in result.high_confidence(0.6):
                await self._knowledge_graph.add_triplet(
                    subject=triplet.subject,
                    relation=triplet.relation,
                    obj=triplet.object,
                    context=content[:200],
                    source_memory_id=source_memory_id,
                )

            if result.count > 0:
                LOGGER.debug(
                    "Extracted %d triplets from: %s", result.count, content[:50]
                )

            return result.count

        except Exception as e:
            LOGGER.warning("Triplet extraction failed: %s", e)
            return 0

    async def boost_memory(self, memory_id: str, boost: float = 0.1) -> bool:
        """
        Boost a memory's importance.

        Called when user reinforces something as important.

        Args:
            memory_id: Memory to boost
            boost: Amount to boost importance

        Returns:
            Whether boost was successful
        """
        entry = await self._ltm.boost_importance(memory_id, boost)
        if entry:
            LOGGER.info(
                "Boosted memory: %s (importance: %.2f)", memory_id[:8], entry.importance
            )
            return True
        return False

    # =========================================================================
    # Voice Commands
    # =========================================================================

    async def voice_command(self, command: str) -> Dict[str, Any]:
        """
        Process a voice command.

        Supported commands:
            - "remember this: {content}" - Store to LTM
            - "this is important" / "ఇది ముఖ్యం" - Boost recent
            - "forget about {topic}" - Delete (with confirmation)
            - "switch to {project}" - Set project context

        Args:
            command: The voice command

        Returns:
            Result of command execution
        """
        command_lower = command.lower()

        # Remember command
        if command_lower.startswith("remember this:") or "గుర్తుంచుకో:" in command:
            content = command.split(":", 1)[-1].strip()
            entry = await self.store_fact(content, importance=0.8)
            return {
                "action": "stored",
                "memory_id": entry.id,
                "content": content,
            }

        # Importance boost
        if "important" in command_lower or "ముఖ్యం" in command:
            # Boost last stored memory or current context
            last_turn = self._working.get_last_turn()
            if last_turn:
                # Store as important fact
                entry = await self.store_fact(
                    last_turn.user_message,
                    importance=0.9,
                )
                return {
                    "action": "boosted",
                    "content": last_turn.user_message[:50],
                }
            return {"action": "nothing_to_boost"}

        # Project switch
        if "switch to" in command_lower or "మీద పని" in command:
            # Extract project name
            if "switch to" in command_lower:
                project = command_lower.split("switch to")[-1].strip()
            else:
                project = command.split("మీద పని")[0].strip().split()[-1]

            self._working.set_project(project)
            self._profile.set_current_project(project)
            return {
                "action": "switched_project",
                "project": project,
            }

        # Forget command (requires confirmation)
        if "forget about" in command_lower or "మర్చిపో" in command:
            topic = (
                command_lower.replace("forget about", "").replace("మర్చిపో", "").strip()
            )
            return {
                "action": "confirm_delete",
                "topic": topic,
                "message": f"Confirm deletion of memories about '{topic}' by saying 'Yes, delete them'",
            }

        return {
            "action": "unknown",
            "message": f"Unknown command: {command}",
        }

    # =========================================================================
    # Knowledge Graph Operations (Cognee-inspired)
    # =========================================================================

    async def graph_query(
        self,
        entity: str,
        max_depth: int = 2,
        relation_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph starting from an entity.

        Args:
            entity: Starting entity name (e.g., "Ravi", "climax")
            max_depth: How many relationship hops to traverse
            relation_filter: Only follow these relation types

        Returns:
            List of related entities with paths

        Example:
            # "What scenes involve Ravi?"
            results = await manager.graph_query("Ravi", max_depth=1)
        """
        if not self._knowledge_graph:
            return []

        # Convert string filters to RelationType if provided
        filters = None
        if relation_filter:
            filters = []
            for r in relation_filter:
                try:
                    filters.append(RelationType(r))
                except ValueError:
                    pass

        results = await self._knowledge_graph.traverse(
            entity, max_depth=max_depth, relation_filter=filters
        )

        return [
            {
                "entity": r.node.name,
                "type": r.node.node_type.value,
                "project": r.node.project,
                "path": r.path,
                "relations": r.relations,
                "depth": r.depth,
            }
            for r in results
        ]

    async def get_related_entities(
        self,
        entity: str,
        relation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get entities directly related to a given entity.

        Args:
            entity: Entity to find relations for
            relation: Optional relation type filter

        Returns:
            List of related entities with relationship type
        """
        if not self._knowledge_graph:
            return []

        rel_type = None
        if relation:
            try:
                rel_type = RelationType(relation)
            except ValueError:
                pass

        results = await self._knowledge_graph.get_related(entity, rel_type)

        return [
            {
                "entity": node.name,
                "type": node.node_type.value,
                "relation": rel,
                "project": node.project,
            }
            for node, rel in results
        ]

    async def get_entities_by_type(
        self,
        entity_type: str,
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all entities of a specific type.

        Args:
            entity_type: Type of entities (character, scene, project, etc.)
            project: Optional project filter

        Returns:
            List of entities

        Example:
            # "List all characters in Gusagusalu"
            chars = await manager.get_entities_by_type("character", project="Gusagusalu")
        """
        if not self._knowledge_graph:
            return []

        try:
            node_type = NodeType(entity_type)
        except ValueError:
            return []

        results = await self._knowledge_graph.get_by_type(node_type, project)

        return [
            {
                "entity": node.name,
                "type": node.node_type.value,
                "project": node.project,
                "attributes": node.attributes,
            }
            for node in results
        ]

    async def add_entity_to_graph(
        self,
        name: str,
        entity_type: str,
        project: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Manually add an entity to the knowledge graph.

        Args:
            name: Entity name
            entity_type: Type (character, scene, project, etc.)
            project: Associated project
            attributes: Additional attributes

        Returns:
            Whether entity was added successfully
        """
        if not self._knowledge_graph:
            return False

        try:
            node_type = NodeType(entity_type)
        except ValueError:
            node_type = NodeType.CONCEPT

        await self._knowledge_graph.add_node(
            name=name,
            node_type=node_type,
            project=project,
            attributes=attributes,
        )

        LOGGER.info("Added entity to graph: %s (%s)", name, entity_type)
        return True

    async def add_relationship(
        self,
        subject: str,
        relation: str,
        obj: str,
        context: str = "",
    ) -> bool:
        """
        Manually add a relationship to the knowledge graph.

        Args:
            subject: Subject entity
            relation: Relationship type
            obj: Object entity
            context: Optional context text

        Returns:
            Whether relationship was added successfully
        """
        if not self._knowledge_graph:
            return False

        await self._knowledge_graph.add_triplet(
            subject=subject,
            relation=relation,
            obj=obj,
            context=context,
        )

        LOGGER.info("Added relationship: %s -[%s]-> %s", subject, relation, obj)
        return True

    # =========================================================================
    # Context for LLM
    # =========================================================================

    def get_context_for_llm(self) -> Dict[str, Any]:
        """
        Get memory context for LLM system prompt.

        Returns:
            Dictionary with relevant context
        """
        context = {
            "profile": self._profile.get_summary(),
            "current_room": self._working.current_room,
            "current_project": self._working.current_project,
            "language_mode": self._working.language_mode,
            "attention": [
                {"topic": a.topic, "relevance": a.relevance}
                for a in self._working.get_attention_topics()[:3]
            ],
            "prefetched_memories": [
                {"content": m.content, "relevance": m.relevance}
                for m in self._working.get_prefetched_ltm()[:3]
            ],
        }

        return context

    # =========================================================================
    # Health & Stats
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all memory components"""
        health = {
            "initialized": self._initialized,
            "working_memory": {
                "turns": self._working.turn_count if self._working else 0,
                "tokens": self._working.token_count if self._working else 0,
                "attention_items": (
                    len(self._working.get_attention_topics()) if self._working else 0
                ),
            },
        }

        if self._stm:
            health["stm"] = await self._stm.get_stats()

        if self._ltm:
            health["ltm"] = await self._ltm.get_stats()

        if self._profile:
            health["profile"] = {
                "version": self._profile.profile.version,
                "projects": len(self._profile.profile.projects),
                "relationships": len(self._profile.profile.relationships),
            }

        if self._knowledge_graph:
            health["knowledge_graph"] = await self._knowledge_graph.get_stats()

        health["triplet_extractor"] = {
            "configured": (
                self._triplet_extractor.is_configured
                if self._triplet_extractor
                else False
            ),
        }

        return health

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = await self.health_check()

        # Add more details
        stats["session"] = {
            "id": self._current_session_id,
            "room": self._working.current_room if self._working else None,
            "project": self._working.current_project if self._working else None,
            "language": self._working.language_mode if self._working else None,
        }

        return stats


# Singleton instance
_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get memory manager singleton"""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager


async def initialize_memory() -> MemoryManager:
    """Initialize and return memory manager"""
    manager = get_memory_manager()
    await manager.initialize()
    return manager
