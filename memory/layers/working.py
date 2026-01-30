"""
Working Memory Layer
====================

Active conversation context - like human's "7±2 items".

Features:
    - Maintains current conversation turns
    - Attention stack with decay
    - Prefetched LTM for fast access
    - Automatic summarization when exceeding limits

Brain Inspiration:
    Human working memory holds 7±2 items with rapid decay.
    We mirror this with an attention stack that decays unused items.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from memory.config import WorkingMemoryConfig, get_memory_config

LOGGER = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""

    user_message: str
    assistant_response: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    context_type: str = "general"

    # Token estimates
    user_tokens: int = 0
    assistant_tokens: int = 0

    def total_tokens(self) -> int:
        """Estimate total tokens for this turn"""
        if self.user_tokens and self.assistant_tokens:
            return self.user_tokens + self.assistant_tokens
        # Rough estimate: 1 token ≈ 4 characters
        return (len(self.user_message) + len(self.assistant_response)) // 4


@dataclass
class AttentionItem:
    """An item in the attention stack"""

    topic: str
    relevance: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decay(self, rate: float) -> None:
        """Apply decay to relevance"""
        self.relevance *= 1 - rate


@dataclass
class PrefetchedMemory:
    """A memory prefetched from LTM"""

    content: str
    relevance: float
    memory_id: str
    memory_type: str
    fetched_at: datetime = field(default_factory=datetime.now)


class WorkingMemory:
    """
    Active context manager - Friday's "conscious" memory.

    Maintains:
        - Current conversation turns (max 10)
        - Attention stack (max 7 items)
        - Prefetched LTM memories
        - Ephemeral state (current room, project, etc.)

    Usage:
        wm = WorkingMemory()
        wm.add_turn(user_msg, assistant_response)

        # Check attention
        topics = wm.get_attention_topics()

        # Update attention based on conversation
        wm.update_attention("climax scene", relevance=0.9)
    """

    def __init__(self, config: Optional[WorkingMemoryConfig] = None):
        self.config = config or get_memory_config().working

        # Conversation history
        self._turns: List[ConversationTurn] = []
        self._total_tokens: int = 0

        # Attention stack (7±2 items like human working memory)
        self._attention_stack: List[AttentionItem] = []

        # Prefetched LTM memories (ready for use)
        self._prefetched_ltm: List[PrefetchedMemory] = []

        # Ephemeral state
        self._current_room: str = "general"
        self._current_project: Optional[str] = None
        self._active_task: Optional[str] = None
        self._language_mode: str = "mixed"  # en, te, mixed
        self._emotional_context: str = "neutral"

        # Pending operations
        self._pending_tool_calls: List[str] = []

    # =========================================================================
    # Conversation Management
    # =========================================================================

    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
        context_type: Optional[str] = None,
    ) -> ConversationTurn:
        """
        Add a conversation turn.

        Auto-summarizes if exceeding token limit.
        """
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.now(),
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            context_type=context_type or self._current_room,
        )

        self._turns.append(turn)
        self._total_tokens += turn.total_tokens()

        # Check limits
        self._enforce_limits()

        # Update attention based on conversation
        self._extract_attention_from_turn(turn)

        LOGGER.debug(
            "Added turn: %d tokens, %d total turns",
            turn.total_tokens(),
            len(self._turns),
        )

        return turn

    def get_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """Get conversation turns (most recent N or all)"""
        if n is None:
            return self._turns.copy()
        return self._turns[-n:]

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Get most recent turn"""
        return self._turns[-1] if self._turns else None

    @property
    def turn_count(self) -> int:
        """Number of conversation turns"""
        return len(self._turns)

    @property
    def token_count(self) -> int:
        """Estimated total tokens"""
        return self._total_tokens

    def _enforce_limits(self) -> None:
        """Enforce turn and token limits"""
        # Remove old turns if exceeding max
        while len(self._turns) > self.config.max_turns:
            removed = self._turns.pop(0)
            self._total_tokens -= removed.total_tokens()
            LOGGER.debug("Removed old turn, now at %d turns", len(self._turns))

        # Summarize if exceeding token limit
        if self._total_tokens > self.config.max_tokens:
            self._summarize_old_turns()

    def _summarize_old_turns(self) -> None:
        """
        Summarize older turns to save tokens.
        Keeps most recent 3 turns intact.
        """
        if len(self._turns) <= 3:
            return

        # Keep last 3, summarize the rest
        to_summarize = self._turns[:-3]
        to_keep = self._turns[-3:]

        # Simple summary (in production, use LLM)
        summary_parts = []
        for turn in to_summarize:
            # Extract key points
            summary_parts.append(f"- Discussed: {turn.user_message[:50]}...")

        summary = "Previous context: " + " ".join(summary_parts)

        # Replace old turns with summary turn
        summary_turn = ConversationTurn(
            user_message="[Summary of earlier conversation]",
            assistant_response=summary,
            timestamp=to_summarize[0].timestamp,
        )

        self._turns = [summary_turn] + to_keep
        self._total_tokens = sum(t.total_tokens() for t in self._turns)

        LOGGER.info(
            "Summarized %d turns, now at %d tokens",
            len(to_summarize),
            self._total_tokens,
        )

    def clear(self) -> None:
        """Clear all conversation history"""
        self._turns.clear()
        self._total_tokens = 0
        self._attention_stack.clear()
        self._prefetched_ltm.clear()
        LOGGER.info("Working memory cleared")

    # =========================================================================
    # Attention Stack
    # =========================================================================

    def update_attention(
        self,
        topic: str,
        relevance: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update attention on a topic.

        Human-like attention management:
        - Strong signals boost to top
        - Existing topics update in place
        - Weak items decay and prune
        """
        # Apply decay to all existing items
        for item in self._attention_stack:
            item.decay(self.config.attention_decay_rate)

        # Find existing or create new
        existing = next(
            (i for i in self._attention_stack if i.topic.lower() == topic.lower()),
            None,
        )

        if existing:
            # Update existing - take max relevance
            existing.relevance = max(existing.relevance, relevance)
            existing.timestamp = datetime.now()
            if metadata:
                existing.metadata.update(metadata)
        else:
            # Add new item
            self._attention_stack.append(
                AttentionItem(
                    topic=topic,
                    relevance=relevance,
                    metadata=metadata or {},
                )
            )

        # Prune weak attention (below threshold)
        self._attention_stack = [i for i in self._attention_stack if i.relevance > 0.2]

        # Sort by relevance
        self._attention_stack.sort(key=lambda x: x.relevance, reverse=True)

        # Keep top N (human working memory limit: 7±2)
        self._attention_stack = self._attention_stack[: self.config.max_attention_items]

        LOGGER.debug(
            "Attention updated: %s (%.2f), stack size: %d",
            topic,
            relevance,
            len(self._attention_stack),
        )

    def get_attention_topics(self) -> List[AttentionItem]:
        """Get current attention stack"""
        return self._attention_stack.copy()

    def get_top_attention(self) -> Optional[AttentionItem]:
        """Get highest-relevance attention item"""
        return self._attention_stack[0] if self._attention_stack else None

    def is_attending_to(self, topic: str, threshold: float = 0.3) -> bool:
        """Check if currently attending to a topic"""
        for item in self._attention_stack:
            if topic.lower() in item.topic.lower() and item.relevance >= threshold:
                return True
        return False

    def _extract_attention_from_turn(self, turn: ConversationTurn) -> None:
        """Extract attention topics from a conversation turn"""
        # Simple keyword extraction (in production, use NER/LLM)
        text = f"{turn.user_message} {turn.assistant_response}".lower()

        # Project names
        projects = ["gusagusalu", "kitchen project"]
        for proj in projects:
            if proj in text:
                self.update_attention(proj, relevance=0.8, metadata={"type": "project"})

        # Scene-related
        if "scene" in text or "climax" in text or "interval" in text:
            self.update_attention(
                "screenplay", relevance=0.7, metadata={"type": "domain"}
            )

        # Telugu detected - high relevance for language awareness
        if any("\u0c00" <= c <= "\u0c7f" for c in text):
            self.update_attention(
                "telugu_mode", relevance=0.6, metadata={"type": "language"}
            )

    # =========================================================================
    # Prefetched LTM
    # =========================================================================

    def set_prefetched_ltm(self, memories: List[PrefetchedMemory]) -> None:
        """Set prefetched LTM memories"""
        self._prefetched_ltm = memories
        LOGGER.debug("Prefetched %d LTM memories", len(memories))

    def get_prefetched_ltm(self) -> List[PrefetchedMemory]:
        """Get prefetched LTM memories"""
        return self._prefetched_ltm.copy()

    def clear_prefetched_ltm(self) -> None:
        """Clear prefetched memories"""
        self._prefetched_ltm.clear()

    # =========================================================================
    # Ephemeral State
    # =========================================================================

    @property
    def current_room(self) -> str:
        return self._current_room

    def set_room(self, room: str) -> None:
        """Set current room/context"""
        if room != self._current_room:
            LOGGER.info("Room changed: %s -> %s", self._current_room, room)
            self._current_room = room
            self.update_attention(f"room:{room}", relevance=0.5)

    @property
    def current_project(self) -> Optional[str]:
        return self._current_project

    def set_project(self, project: Optional[str]) -> None:
        """Set current project context"""
        if project != self._current_project:
            LOGGER.info("Project changed: %s -> %s", self._current_project, project)
            self._current_project = project
            if project:
                self.update_attention(
                    project, relevance=0.8, metadata={"type": "project"}
                )

    @property
    def language_mode(self) -> str:
        return self._language_mode

    def set_language_mode(self, mode: str) -> None:
        """Set language mode: en, te, or mixed"""
        self._language_mode = mode

    @property
    def emotional_context(self) -> str:
        return self._emotional_context

    def set_emotional_context(self, emotion: str) -> None:
        """Set detected emotional context"""
        self._emotional_context = emotion

    @property
    def active_task(self) -> Optional[str]:
        return self._active_task

    def set_active_task(self, task: Optional[str]) -> None:
        """Set current active task"""
        self._active_task = task
        if task:
            self.update_attention(f"task:{task}", relevance=0.9)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize working memory state"""
        return {
            "turns": [
                {
                    "user": t.user_message,
                    "assistant": t.assistant_response,
                    "timestamp": t.timestamp.isoformat(),
                    "context": t.context_type,
                }
                for t in self._turns
            ],
            "attention": [
                {
                    "topic": a.topic,
                    "relevance": a.relevance,
                    "timestamp": a.timestamp.isoformat(),
                }
                for a in self._attention_stack
            ],
            "state": {
                "room": self._current_room,
                "project": self._current_project,
                "task": self._active_task,
                "language": self._language_mode,
                "emotion": self._emotional_context,
            },
            "token_count": self._total_tokens,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], config: Optional[WorkingMemoryConfig] = None
    ) -> "WorkingMemory":
        """Restore working memory from dict"""
        wm = cls(config)

        # Restore turns
        for t in data.get("turns", []):
            wm._turns.append(
                ConversationTurn(
                    user_message=t["user"],
                    assistant_response=t["assistant"],
                    timestamp=datetime.fromisoformat(t["timestamp"]),
                    context_type=t.get("context", "general"),
                )
            )

        # Restore attention
        for a in data.get("attention", []):
            wm._attention_stack.append(
                AttentionItem(
                    topic=a["topic"],
                    relevance=a["relevance"],
                    timestamp=datetime.fromisoformat(a["timestamp"]),
                )
            )

        # Restore state
        state = data.get("state", {})
        wm._current_room = state.get("room", "general")
        wm._current_project = state.get("project")
        wm._active_task = state.get("task")
        wm._language_mode = state.get("language", "mixed")
        wm._emotional_context = state.get("emotion", "neutral")

        wm._total_tokens = data.get("token_count", 0)

        return wm

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(turns={len(self._turns)}, "
            f"tokens={self._total_tokens}, "
            f"attention={len(self._attention_stack)}, "
            f"room={self._current_room})"
        )
