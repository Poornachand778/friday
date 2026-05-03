"""
Friday AI Memory System
=======================

Brain-inspired, voice-controlled, fully autonomous memory architecture.

Layers:
    - Sensory: Raw audio buffer (100ms-2s)
    - Working: Active conversation context (7±2 items)
    - ShortTerm: Recent conversations (7 days)
    - LongTerm: Consolidated knowledge (permanent)
    - Profile: Identity facts (never decays)
    - KnowledgeGraph: Entity relationships (Cognee-inspired)
    - Timeline: Temporal events
    - Patterns: Learned behaviors

Usage:
    from memory import MemoryManager

    manager = MemoryManager()
    await manager.initialize()

    # Search memories
    results = await manager.search("climax scene discussion")

    # Store memory
    await manager.store_fact("Boss wants more emotional punch in climax")

    # Graph query (Cognee-inspired)
    related = await manager.graph_query("Ravi", max_depth=2)

    # Voice command
    await manager.voice_command("remember this: Ravi hesitates")
"""

from memory.manager import MemoryManager, get_memory_manager, initialize_memory
from memory.config import MemorySystemConfig, get_memory_config
from memory.layers.knowledge_graph import KnowledgeGraph, NodeType, RelationType
from memory.layers.long_term import MemoryType
from memory.operations.triplet_extractor import TripletExtractor, extract_triplets

__all__ = [
    # Manager
    "MemoryManager",
    "get_memory_manager",
    "initialize_memory",
    # Config
    "MemorySystemConfig",
    "get_memory_config",
    # Knowledge Graph
    "KnowledgeGraph",
    "NodeType",
    "RelationType",
    # LTM
    "MemoryType",
    # Triplet extraction
    "TripletExtractor",
    "extract_triplets",
]

__version__ = "1.0.0"
