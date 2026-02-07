"""
Conversation Memory for Friday AI
=================================

Manages short-term conversation history with:
- Turn storage and retrieval
- Context window management
- Automatic summarization for long conversations
- Database persistence
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque

from orchestrator.inference.local_llm import ChatMessage

LOGGER = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single conversation turn (user message + assistant response)"""

    turn_id: int
    user_message: str
    assistant_response: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    context_type: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough estimate of tokens in this turn (4 chars ≈ 1 token)"""
        total_chars = len(self.user_message) + len(self.assistant_response)
        for tc in self.tool_calls:
            total_chars += len(str(tc))
        for tr in self.tool_results:
            total_chars += len(str(tr))
        return total_chars // 4

    def to_messages(self) -> List[ChatMessage]:
        """Convert turn to chat messages"""
        messages = [
            ChatMessage(role="user", content=self.user_message),
        ]

        # Add tool calls if any
        if self.tool_calls:
            messages.append(
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=self.tool_calls,
                )
            )
            for result in self.tool_results:
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=str(result.get("data", result.get("error", ""))),
                        tool_call_id=result.get("tool_call_id", ""),
                        name=result.get("name", ""),
                    )
                )

        messages.append(
            ChatMessage(
                role="assistant",
                content=self.assistant_response,
            )
        )

        return messages


class ConversationMemory:
    """
    Manages conversation history for Friday.

    Features:
    - Sliding window of recent turns
    - Token-aware context management
    - Automatic summarization for long histories
    - Optional database persistence

    Usage:
        memory = ConversationMemory(max_turns=20, max_tokens=4000)

        # Add a turn
        memory.add_turn(
            user_message="Boss, show me romantic scenes",
            assistant_response="Found 5 romantic scenes...",
            tool_calls=[...],
        )

        # Get messages for LLM context
        messages = memory.get_context_messages()
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_tokens: int = 4000,
        summarize_threshold: int = 15,
    ):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold

        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)
        self._turn_counter = 0
        self._summary: Optional[str] = None
        self._summarized_turn_count = 0

        # Session info
        self.session_id: Optional[str] = None
        self.started_at: float = time.time()
        self.current_context: str = "general"

    @property
    def turn_count(self) -> int:
        """Total turns in current session"""
        return self._turn_counter

    @property
    def active_turns(self) -> int:
        """Number of turns in active memory"""
        return len(self._turns)

    @property
    def total_tokens(self) -> int:
        """Estimated total tokens in memory"""
        return sum(t.token_estimate for t in self._turns)

    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls: Optional[List[Dict]] = None,
        tool_results: Optional[List[Dict]] = None,
        context_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> ConversationTurn:
        """Add a conversation turn"""
        self._turn_counter += 1

        turn = ConversationTurn(
            turn_id=self._turn_counter,
            user_message=user_message,
            assistant_response=assistant_response,
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            context_type=context_type or self.current_context,
            metadata=metadata or {},
        )

        self._turns.append(turn)

        # Check if we need to summarize
        if len(self._turns) >= self.summarize_threshold:
            self._maybe_summarize()

        LOGGER.debug(
            "Added turn %d (%d tokens, %d active turns)",
            turn.turn_id,
            turn.token_estimate,
            len(self._turns),
        )

        return turn

    def get_context_messages(
        self,
        system_prompt: Optional[str] = None,
        include_summary: bool = True,
        max_tokens: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Get messages for LLM context.

        Returns messages in order:
        1. System prompt (if provided)
        2. Summary of older conversation (if available)
        3. Recent turns (within token limit)
        """
        messages = []
        token_budget = max_tokens or self.max_tokens

        # System prompt
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
            token_budget -= len(system_prompt) // 4

        # Summary of older history
        if include_summary and self._summary:
            summary_msg = f"[Previous conversation summary: {self._summary}]"
            messages.append(ChatMessage(role="system", content=summary_msg))
            token_budget -= len(summary_msg) // 4

        # Collect turns that fit within token budget (most recent first)
        selected_turns = []
        tokens_used = 0

        for turn in reversed(self._turns):
            turn_tokens = turn.token_estimate
            if tokens_used + turn_tokens > token_budget:
                break
            selected_turns.append(turn)
            tokens_used += turn_tokens

        # Add in chronological order (reverse selected turns, then expand messages)
        for turn in reversed(selected_turns):
            messages.extend(turn.to_messages())

        return messages

    def get_last_n_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get the last N turns"""
        return list(self._turns)[-n:]

    def get_turn(self, turn_id: int) -> Optional[ConversationTurn]:
        """Get a specific turn by ID"""
        for turn in self._turns:
            if turn.turn_id == turn_id:
                return turn
        return None

    def set_context(self, context_type: str) -> None:
        """Set current context type"""
        self.current_context = context_type
        LOGGER.info("Context changed to: %s", context_type)

    def clear(self) -> None:
        """Clear all conversation history"""
        self._turns.clear()
        self._turn_counter = 0
        self._summary = None
        self._summarized_turn_count = 0
        LOGGER.info("Conversation memory cleared")

    def _maybe_summarize(self) -> None:
        """Summarize older turns if approaching limit"""
        # Only summarize if we have enough turns
        if len(self._turns) < self.summarize_threshold:
            return

        # Get turns to summarize (oldest half)
        turns_to_summarize = list(self._turns)[: len(self._turns) // 2]

        if not turns_to_summarize:
            return

        # Create simple summary (in production, use LLM)
        topics = set()
        for turn in turns_to_summarize:
            # Extract key words (simple heuristic)
            words = turn.user_message.lower().split()
            topics.update(w for w in words if len(w) > 4)

        self._summary = f"Discussed: {', '.join(list(topics)[:10])}"
        self._summarized_turn_count += len(turns_to_summarize)

        LOGGER.debug("Created summary of %d turns", len(turns_to_summarize))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory state"""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "turn_count": self._turn_counter,
            "current_context": self.current_context,
            "summary": self._summary,
            "summarized_turns": self._summarized_turn_count,
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "user_message": t.user_message,
                    "assistant_response": t.assistant_response,
                    "timestamp": t.timestamp,
                    "tool_calls": t.tool_calls,
                    "context_type": t.context_type,
                }
                for t in self._turns
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """Deserialize memory state"""
        memory = cls()
        memory.session_id = data.get("session_id")
        memory.started_at = data.get("started_at", time.time())
        memory._turn_counter = data.get("turn_count", 0)
        memory.current_context = data.get("current_context", "general")
        memory._summary = data.get("summary")
        memory._summarized_turn_count = data.get("summarized_turns", 0)

        for turn_data in data.get("turns", []):
            turn = ConversationTurn(
                turn_id=turn_data["turn_id"],
                user_message=turn_data["user_message"],
                assistant_response=turn_data["assistant_response"],
                timestamp=turn_data.get("timestamp", time.time()),
                tool_calls=turn_data.get("tool_calls", []),
                context_type=turn_data.get("context_type", "general"),
            )
            memory._turns.append(turn)

        return memory
