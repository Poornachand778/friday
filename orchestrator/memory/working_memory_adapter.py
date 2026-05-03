"""
WorkingMemory Adapter for Orchestrator
=======================================

Bridges memory.layers.working.WorkingMemory to the interface
expected by the orchestrator and context builder.

Why an adapter instead of modifying WorkingMemory?
    - WorkingMemory is a pure memory-layer concern (tokens, capacity, poisoning)
    - Orchestrator needs ChatMessage format, session IDs, turn IDs
    - Keeps concerns separated: memory management vs orchestrator protocol

The adapter provides:
    - ConversationMemory-compatible interface (session_id, get_context_messages, etc.)
    - Full WorkingMemory features (capacity zones, poisoning detection, attention)
    - Room/project/language awareness piped to WorkingMemory's ephemeral state
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from memory.config import WorkingMemoryConfig
from memory.layers.working import (
    WorkingMemory,
    ConversationTurn as WMConversationTurn,
    TokenCounter,
)
from orchestrator.inference.local_llm import ChatMessage

LOGGER = logging.getLogger(__name__)


class _AdaptedTurn:
    """
    Thin wrapper that adds turn_id to a WorkingMemory ConversationTurn.

    The orchestrator expects add_turn() to return an object with turn_id.
    WorkingMemory's ConversationTurn doesn't have one.
    """

    def __init__(self, turn_id: int, wm_turn: WMConversationTurn):
        self.turn_id = turn_id
        self._wm_turn = wm_turn

    @property
    def user_message(self) -> str:
        return self._wm_turn.user_message

    @property
    def assistant_response(self) -> str:
        return self._wm_turn.assistant_response

    @property
    def timestamp(self):
        return self._wm_turn.timestamp

    @property
    def tool_calls(self):
        return self._wm_turn.tool_calls

    @property
    def tool_results(self):
        return self._wm_turn.tool_results

    @property
    def context_type(self) -> str:
        return self._wm_turn.context_type

    @property
    def confidence(self) -> float:
        return self._wm_turn.confidence

    @property
    def is_quarantined(self) -> bool:
        return self._wm_turn.is_quarantined

    @property
    def token_estimate(self) -> int:
        return self._wm_turn.total_tokens()


class WorkingMemoryAdapter:
    """
    Adapts WorkingMemory to the ConversationMemory interface.

    Provides:
        - ConversationMemory-compatible API (session_id, get_context_messages, etc.)
        - WorkingMemory advanced features (capacity zones, poisoning, attention)
        - Room/project/language state management

    Usage:
        from memory.layers.working import WorkingMemory
        from orchestrator.memory.working_memory_adapter import WorkingMemoryAdapter

        wm = WorkingMemory()
        adapter = WorkingMemoryAdapter(wm)

        # Use like ConversationMemory
        turn = adapter.add_turn("Hello Boss", "Boss, baagunnanu!")
        messages = adapter.get_context_messages(max_tokens=4000)

        # Access WorkingMemory features
        print(adapter.capacity_zone)  # "normal"
        adapter.update_attention("screenplay", 0.9)
    """

    def __init__(
        self,
        working_memory: Optional[WorkingMemory] = None,
        config: Optional[WorkingMemoryConfig] = None,
    ):
        self._wm = working_memory or WorkingMemory(config=config)

        # Session metadata (not part of WorkingMemory's concern)
        self.session_id: Optional[str] = None
        self.started_at: float = time.time()
        self._turn_counter: int = 0

    # =========================================================================
    # ConversationMemory-compatible interface
    # =========================================================================

    @property
    def current_context(self) -> str:
        """Current context/room."""
        return self._wm.current_room

    @property
    def turn_count(self) -> int:
        """Total turns added to this session."""
        return self._turn_counter

    @property
    def active_turns(self) -> int:
        """Number of verbatim turns in working memory."""
        return self._wm.turn_count

    @property
    def total_tokens(self) -> int:
        """Total tokens currently in working memory."""
        return self._wm.token_count

    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
        context_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> _AdaptedTurn:
        """
        Add a conversation turn.

        Delegates to WorkingMemory which handles:
        - Token counting (tiktoken with Telugu awareness)
        - Context poisoning detection
        - Capacity-based compression (70/85/95% zones)
        - Attention stack updates

        Returns an _AdaptedTurn with turn_id for orchestrator compatibility.
        """
        self._turn_counter += 1

        wm_turn = self._wm.add_turn(
            user_message=user_message,
            assistant_response=assistant_response,
            tool_calls=tool_calls,
            tool_results=tool_results,
            context_type=context_type,
        )

        LOGGER.debug(
            "Adapter: turn %d added, capacity=%.1f%% (%s)",
            self._turn_counter,
            self._wm.capacity_percentage * 100,
            self._wm.capacity_zone,
        )

        return _AdaptedTurn(self._turn_counter, wm_turn)

    def get_last_n_turns(self, n: int = 5) -> List[_AdaptedTurn]:
        """Get the last N turns as adapted turns."""
        wm_turns = self._wm.get_turns(n)
        # Wrap with turn_ids (approximate - use negative offset)
        result = []
        start_id = max(1, self._turn_counter - len(wm_turns) + 1)
        for i, t in enumerate(wm_turns):
            result.append(_AdaptedTurn(start_id + i, t))
        return result

    def set_context(self, context_type: str) -> None:
        """Set current context (maps to WorkingMemory's room)."""
        self._wm.set_room(context_type)

    def get_context_messages(
        self,
        system_prompt: Optional[str] = None,
        include_summary: bool = True,
        max_tokens: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Get messages formatted for LLM context (ChatML format).

        Constructs message list from WorkingMemory's:
        1. Compressed history → system messages
        2. Recent verbatim turns → user/assistant/tool messages

        Respects token budget to avoid context overflow.
        """
        messages: List[ChatMessage] = []
        budget = max_tokens or self._wm.config.max_tokens

        # System prompt
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
            budget -= TokenCounter.count(system_prompt)

        # Compressed history as context
        if include_summary and self._wm._compressed_history:
            history_parts = [ch.summary for ch in self._wm._compressed_history]
            summary_text = "[Prior context:\n" + "\n".join(history_parts) + "]"
            messages.append(ChatMessage(role="system", content=summary_text))
            budget -= TokenCounter.count(summary_text)

        # Add recent verbatim turns (newest first, then reverse)
        recent_messages: List[ChatMessage] = []
        tokens_used = 0

        for turn in reversed(self._wm._turns):
            turn_tokens = turn.total_tokens()
            if tokens_used + turn_tokens > budget:
                break

            turn_msgs = self._turn_to_messages(turn)
            recent_messages.extend(reversed(turn_msgs))
            tokens_used += turn_tokens

        # Reverse to chronological order
        messages.extend(reversed(recent_messages))

        return messages

    def _turn_to_messages(self, turn: WMConversationTurn) -> List[ChatMessage]:
        """Convert a WorkingMemory turn to ChatMessage list."""
        msgs: List[ChatMessage] = []

        msgs.append(ChatMessage(role="user", content=turn.user_message))

        # Tool calls and results
        if turn.tool_calls:
            msgs.append(
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=turn.tool_calls,
                )
            )
            for result in turn.tool_results:
                msgs.append(
                    ChatMessage(
                        role="tool",
                        content=str(result.get("data", result.get("error", ""))),
                        tool_call_id=result.get("tool_call_id", ""),
                        name=result.get("name", ""),
                    )
                )

        msgs.append(ChatMessage(role="assistant", content=turn.assistant_response))

        return msgs

    def clear(self) -> None:
        """Clear all conversation history."""
        self._wm.clear()
        self._turn_counter = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize adapter + working memory state."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "turn_counter": self._turn_counter,
            "working_memory": self._wm.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        config: Optional[WorkingMemoryConfig] = None,
    ) -> "WorkingMemoryAdapter":
        """Restore adapter from serialized state."""
        wm = WorkingMemory.from_dict(data.get("working_memory", {}), config=config)
        adapter = cls(working_memory=wm)
        adapter.session_id = data.get("session_id")
        adapter.started_at = data.get("started_at", time.time())
        adapter._turn_counter = data.get("turn_counter", 0)
        return adapter

    # =========================================================================
    # WorkingMemory-specific features (exposed for orchestrator enrichment)
    # =========================================================================

    @property
    def working_memory(self) -> WorkingMemory:
        """Direct access to underlying WorkingMemory."""
        return self._wm

    @property
    def capacity_zone(self) -> str:
        """Current capacity zone: normal/proactive/aggressive/emergency."""
        return self._wm.capacity_zone

    @property
    def capacity_percentage(self) -> float:
        """Context window usage as 0.0-1.0."""
        return self._wm.capacity_percentage

    @property
    def tokens_available(self) -> int:
        """Tokens remaining before max capacity."""
        return self._wm.tokens_available

    def update_attention(
        self,
        topic: str,
        relevance: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update attention on a topic (7±2 items)."""
        self._wm.update_attention(topic, relevance, metadata)

    def set_project(self, project: Optional[str]) -> None:
        """Set current project context."""
        self._wm.set_project(project)

    def set_language_mode(self, mode: str) -> None:
        """Set language mode: en, te, or mixed."""
        self._wm.set_language_mode(mode)

    def set_emotional_context(self, emotion: str) -> None:
        """Set detected emotional context."""
        self._wm.set_emotional_context(emotion)

    def set_active_task(self, task: Optional[str]) -> None:
        """Set current active task."""
        self._wm.set_active_task(task)

    def set_prefetched_ltm(self, memories) -> None:
        """Set prefetched LTM memories."""
        self._wm.set_prefetched_ltm(memories)

    def get_health_status(self) -> Dict[str, Any]:
        """Get health metrics from WorkingMemory."""
        health = self._wm.get_health_status()
        health["session_id"] = self.session_id
        health["session_turn_count"] = self._turn_counter
        return health

    def get_context_stats(self) -> Dict[str, Any]:
        """Get context usage statistics."""
        stats = self._wm.get_context_stats()
        stats["session_id"] = self.session_id
        stats["session_turn_count"] = self._turn_counter
        return stats

    def set_summarizer(self, summarizer) -> None:
        """Inject LLM-based summarizer for better compression."""
        self._wm.set_summarizer(summarizer)

    def __repr__(self) -> str:
        return (
            f"WorkingMemoryAdapter("
            f"session={self.session_id}, "
            f"turns={self._turn_counter}, "
            f"{self._wm!r})"
        )
