"""
Memory Layers
=============

Brain-inspired hierarchical memory storage.

Layers:
    - SensoryBuffer: Raw audio capture (100ms-2s)
    - WorkingMemory: Active context (7±2 items)
    - ShortTermMemory: Recent conversations (7 days)
    - LongTermMemory: Consolidated knowledge (permanent)
    - ProfileStore: Identity facts (never decays)
    - KnowledgeGraph: Entity relationships
    - Timeline: Temporal events
    - PatternStore: Learned behaviors
"""

from memory.layers.working import WorkingMemory, AttentionItem
from memory.layers.short_term import ShortTermMemory, STMEntry
from memory.layers.long_term import LongTermMemory, LTMEntry, MemoryType
from memory.layers.profile import ProfileStore, UserProfile

__all__ = [
    "WorkingMemory",
    "AttentionItem",
    "ShortTermMemory",
    "STMEntry",
    "LongTermMemory",
    "LTMEntry",
    "MemoryType",
    "ProfileStore",
    "UserProfile",
]
