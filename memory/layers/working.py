"""
Working Memory Layer - Context Window Manager
==============================================

Active conversation context - like human's "7±2 items".

ARCHITECTURE: Context Window Overflow Prevention
------------------------------------------------

PROBLEM: In 24/7 operation, context windows overflow causing:
    1. Lost-in-the-Middle effect: LLMs weigh start/end heavily, ignore middle
    2. Context Poisoning: One hallucination enters context, gets re-referenced
    3. Token overflow: Exceeding model limits causes hard failures

SOLUTION: Multi-tier sliding window with proactive compression

    ┌─────────────────────────────────────────────────────────────────┐
    │                    CONTEXT WINDOW (max_tokens)                  │
    ├─────────────────────────────────────────────────────────────────┤
    │  [COMPRESSED HISTORY]  │  [RECENT VERBATIM]  │  [ATTENTION]     │
    │  ~20% of capacity      │  ~60% of capacity   │  ~20% reserved   │
    │  Summarized turns      │  Last N full turns  │  Topics + LTM    │
    └─────────────────────────────────────────────────────────────────┘

TRIGGERS:
    - 70% capacity: Proactive summarization begins (not panic mode)
    - 85% capacity: Aggressive compression
    - 95% capacity: Emergency pruning (oldest non-essential removed)

CONTEXT POISONING DETECTION:
    - Track confidence scores per turn
    - Flag repetitive patterns (same claim 3+ times may be hallucination loop)
    - Mark low-confidence turns for exclusion from summaries

Features:
    - Maintains current conversation turns
    - Attention stack with decay
    - Prefetched LTM for fast access
    - Automatic summarization at 70% capacity (proactive, not reactive)
    - Hybrid buffer: recent verbatim + compressed history
    - Context poisoning detection and quarantine

Brain Inspiration:
    Human working memory holds 7±2 items with rapid decay.
    We mirror this with an attention stack that decays unused items.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from memory.config import WorkingMemoryConfig, get_memory_config

LOGGER = logging.getLogger(__name__)

# ============================================================================
# Token Counting Utilities
# ============================================================================

# Try to use tiktoken for accurate counting, fall back to estimation
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
    _ENCODING = tiktoken.get_encoding("cl100k_base")  # GPT-4/Claude tokenizer approx
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    _ENCODING = None
    LOGGER.info("tiktoken not available, using character-based token estimation")


class TokenCounter:
    """
    Accurate token counting for context window management.

    Uses tiktoken when available, falls back to character estimation.
    Character estimation: ~4 chars/token for English, ~2.5 for Telugu.
    """

    # Telugu Unicode range
    TELUGU_RANGE = (0x0C00, 0x0C7F)

    @classmethod
    def count(cls, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0

        if _TIKTOKEN_AVAILABLE and _ENCODING:
            return len(_ENCODING.encode(text))

        # Fallback: Character-based estimation
        return cls._estimate_tokens(text)

    @classmethod
    def _estimate_tokens(cls, text: str) -> int:
        """Estimate tokens using character ratios."""
        if not text:
            return 0

        telugu_chars = sum(
            1 for c in text if cls.TELUGU_RANGE[0] <= ord(c) <= cls.TELUGU_RANGE[1]
        )
        english_chars = len(text) - telugu_chars

        # Telugu: ~2.5 chars per token (more complex script)
        # English: ~4 chars per token
        telugu_tokens = telugu_chars / 2.5
        english_tokens = english_chars / 4.0

        return int(telugu_tokens + english_tokens)

    @classmethod
    def count_messages(cls, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages (ChatML format)."""
        total = 0
        for msg in messages:
            # Add overhead for role markers
            total += 4  # <|role|> ... <|end|> overhead
            total += cls.count(msg.get("role", ""))
            total += cls.count(msg.get("content", ""))
        return total


# ============================================================================
# Context Capacity Thresholds
# ============================================================================


class CapacityThresholds:
    """Context window capacity management thresholds."""

    PROACTIVE = 0.70  # 70%: Start proactive summarization
    AGGRESSIVE = 0.85  # 85%: Aggressive compression
    EMERGENCY = 0.95  # 95%: Emergency pruning

    # Buffer allocations
    COMPRESSED_HISTORY = 0.20  # 20% for summarized history
    RECENT_VERBATIM = 0.60  # 60% for recent full turns
    ATTENTION_RESERVE = 0.20  # 20% for attention + LTM prefetch


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    user_message: str
    assistant_response: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    context_type: str = "general"

    # Token counts (computed on creation)
    user_tokens: int = 0
    assistant_tokens: int = 0
    _cached_total: int = 0

    # Context poisoning detection fields
    confidence: float = 1.0  # 0.0-1.0, lower = potentially hallucinated
    content_hash: str = ""  # For repetition detection
    is_quarantined: bool = False  # Excluded from summaries if True
    repetition_count: int = 0  # How many times similar content appeared

    def __post_init__(self) -> None:
        """Compute tokens and hash after initialization."""
        if not self.user_tokens:
            self.user_tokens = TokenCounter.count(self.user_message)
        if not self.assistant_tokens:
            self.assistant_tokens = TokenCounter.count(self.assistant_response)
        self._cached_total = self.user_tokens + self.assistant_tokens
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash for repetition detection."""
        # Normalize: lowercase, remove extra whitespace
        normalized = " ".join(self.assistant_response.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def total_tokens(self) -> int:
        """Get total tokens for this turn."""
        if self._cached_total:
            return self._cached_total
        return self.user_tokens + self.assistant_tokens

    def mark_quarantined(self, reason: str = "") -> None:
        """Mark this turn as potentially poisoned - exclude from summaries."""
        self.is_quarantined = True
        LOGGER.warning(
            "Turn quarantined: %s... (reason: %s)",
            self.assistant_response[:50],
            reason or "suspected hallucination",
        )


@dataclass
class CompressedHistory:
    """Compressed summary of older conversation turns."""

    summary: str
    turn_count: int  # How many turns were compressed
    timestamp_start: datetime
    timestamp_end: datetime
    tokens: int = 0
    topics_covered: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = TokenCounter.count(self.summary)


class ContextPoisoningDetector:
    """
    Detects potential context poisoning (hallucination loops).

    DETECTION STRATEGIES:
    1. Repetition Detection: Same claim appears 3+ times
    2. Confidence Decay: Track uncertainty markers ("I think", "might be")
    3. Contradiction Detection: Conflicting claims in same session
    4. Self-Reference Loops: Model referencing its own previous (wrong) answers
    """

    # Phrases indicating uncertainty
    UNCERTAINTY_MARKERS = [
        "i think",
        "i believe",
        "might be",
        "could be",
        "possibly",
        "i'm not sure",
        "i'm uncertain",
        "it seems",
        "perhaps",
        "i may have",
        "i might have",
        "if i recall",
        "i assume",
    ]

    # Phrases indicating self-reference (potential loop)
    SELF_REFERENCE_MARKERS = [
        "as i mentioned",
        "as i said",
        "like i said",
        "as stated earlier",
        "i already told you",
        "i explained before",
        "remember when i said",
    ]

    def __init__(self, repetition_threshold: int = 3):
        self.repetition_threshold = repetition_threshold
        self._content_hashes: Counter = Counter()
        self._claim_tracker: Dict[str, int] = {}  # Normalized claim -> count

    def analyze_turn(self, turn: ConversationTurn) -> Tuple[float, List[str]]:
        """
        Analyze a turn for potential poisoning.

        Returns:
            (confidence_score, list_of_warnings)
        """
        warnings = []
        confidence = 1.0

        response = turn.assistant_response.lower()

        # 1. Check for uncertainty markers
        uncertainty_count = sum(
            1 for marker in self.UNCERTAINTY_MARKERS if marker in response
        )
        if uncertainty_count > 0:
            confidence -= 0.1 * uncertainty_count
            warnings.append(f"uncertainty_markers: {uncertainty_count}")

        # 2. Check for self-reference (potential loop)
        self_ref_count = sum(
            1 for marker in self.SELF_REFERENCE_MARKERS if marker in response
        )
        if self_ref_count > 0:
            confidence -= 0.15 * self_ref_count
            warnings.append(f"self_references: {self_ref_count}")

        # 3. Check for content repetition
        self._content_hashes[turn.content_hash] += 1
        if self._content_hashes[turn.content_hash] >= self.repetition_threshold:
            confidence -= 0.3
            warnings.append(
                f"content_repeated: {self._content_hashes[turn.content_hash]} times"
            )

        # 4. Extract and track claims (simple sentence extraction)
        claims = self._extract_claims(response)
        for claim in claims:
            self._claim_tracker[claim] = self._claim_tracker.get(claim, 0) + 1
            if self._claim_tracker[claim] >= self.repetition_threshold:
                confidence -= 0.2
                warnings.append(f"claim_repeated: '{claim[:30]}...'")

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        return confidence, warnings

    def _extract_claims(self, text: str) -> List[str]:
        """Extract potential claims/assertions from text."""
        # Simple: split by sentence-ending punctuation
        sentences = re.split(r"[.!?]", text)
        claims = []
        for s in sentences:
            s = s.strip()
            # Only track declarative sentences of reasonable length
            if 10 < len(s) < 200 and not s.startswith(("?", "how", "what", "why")):
                # Normalize
                normalized = " ".join(s.lower().split())
                claims.append(normalized)
        return claims

    def should_quarantine(self, turn: ConversationTurn) -> bool:
        """Determine if a turn should be quarantined."""
        return (
            turn.confidence < 0.5 or turn.repetition_count >= self.repetition_threshold
        )

    def reset(self) -> None:
        """Reset detector state (call on session end)."""
        self._content_hashes.clear()
        self._claim_tracker.clear()


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

    ARCHITECTURE: Hybrid Buffer with Proactive Compression
    ======================================================

    The context window is divided into three zones:

    1. COMPRESSED HISTORY (~20% of capacity)
       - Summarized older turns
       - Essential facts only, no verbatim content
       - Excluded: quarantined/low-confidence turns

    2. RECENT VERBATIM (~60% of capacity)
       - Last N full conversation turns
       - Preserves nuance, code-switching, exact phrasing
       - Critical for maintaining conversation flow

    3. ATTENTION RESERVE (~20% of capacity)
       - Active topics (7±2 items)
       - Prefetched LTM memories
       - Room for tool results

    CAPACITY MANAGEMENT:
        70%  → Proactive summarization (smooth, no panic)
        85%  → Aggressive compression (compress more, keep less verbatim)
        95%  → Emergency pruning (drop oldest, log warning)

    Maintains:
        - Current conversation turns (max_turns config)
        - Compressed history (older turns summarized)
        - Attention stack (max 7 items)
        - Prefetched LTM memories
        - Ephemeral state (current room, project, etc.)
        - Context poisoning detection

    Usage:
        wm = WorkingMemory()
        wm.add_turn(user_msg, assistant_response)

        # Check capacity
        print(f"Context usage: {wm.capacity_percentage:.1%}")

        # Check attention
        topics = wm.get_attention_topics()

        # Update attention based on conversation
        wm.update_attention("climax scene", relevance=0.9)
    """

    def __init__(
        self,
        config: Optional[WorkingMemoryConfig] = None,
        summarizer: Optional[Callable[[List[ConversationTurn]], str]] = None,
    ):
        self.config = config or get_memory_config().working

        # Hybrid buffer: compressed + recent verbatim
        self._compressed_history: List[CompressedHistory] = []
        self._turns: List[ConversationTurn] = []  # Recent verbatim turns
        self._total_tokens: int = 0

        # Context poisoning detection
        self._poisoning_detector = ContextPoisoningDetector()

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

        # Optional LLM-based summarizer (injected for better summaries)
        self._summarizer = summarizer or self._default_summarizer

        # Capacity tracking
        self._last_capacity_check: float = 0.0

    # =========================================================================
    # Capacity Properties
    # =========================================================================

    @property
    def capacity_percentage(self) -> float:
        """Current context window usage as percentage (0.0 to 1.0+)."""
        return self._total_tokens / self.config.max_tokens

    @property
    def capacity_zone(self) -> str:
        """Current capacity zone: 'normal', 'proactive', 'aggressive', or 'emergency'."""
        pct = self.capacity_percentage
        if pct >= CapacityThresholds.EMERGENCY:
            return "emergency"
        if pct >= CapacityThresholds.AGGRESSIVE:
            return "aggressive"
        if pct >= CapacityThresholds.PROACTIVE:
            return "proactive"
        return "normal"

    @property
    def tokens_available(self) -> int:
        """Tokens remaining before max capacity."""
        return max(0, self.config.max_tokens - self._total_tokens)

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

        Triggers context management based on capacity thresholds:
        - 70%: Proactive summarization (smooth, gradual)
        - 85%: Aggressive compression
        - 95%: Emergency pruning

        Also runs context poisoning detection.
        """
        turn = ConversationTurn(
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.now(),
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            context_type=context_type or self._current_room,
        )

        # Run context poisoning detection
        confidence, warnings = self._poisoning_detector.analyze_turn(turn)
        turn.confidence = confidence
        if warnings:
            LOGGER.debug("Poisoning warnings for turn: %s", warnings)

        # Check if should quarantine
        if self._poisoning_detector.should_quarantine(turn):
            turn.mark_quarantined(f"confidence={confidence:.2f}")

        self._turns.append(turn)
        self._total_tokens += turn.total_tokens()

        # Capacity-based context management
        self._manage_capacity()

        # Update attention based on conversation
        self._extract_attention_from_turn(turn)

        LOGGER.debug(
            "Added turn: %d tokens, total=%d (%.1f%% capacity), zone=%s",
            turn.total_tokens(),
            self._total_tokens,
            self.capacity_percentage * 100,
            self.capacity_zone,
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

    def _manage_capacity(self) -> None:
        """
        Capacity-based context management with three zones.

        This is called after every turn addition to maintain healthy context.

        ZONES:
            Normal (<70%): No action needed
            Proactive (70-85%): Gentle summarization of oldest turns
            Aggressive (85-95%): Compress more, keep fewer verbatim
            Emergency (>95%): Prune oldest, log warning
        """
        zone = self.capacity_zone

        if zone == "normal":
            # Enforce max_turns even in normal zone
            self._enforce_turn_limit()
            return

        if zone == "proactive":
            # Gentle compression - summarize oldest turn(s)
            self._proactive_summarize(turns_to_compress=1)

        elif zone == "aggressive":
            # Aggressive compression - summarize multiple turns
            self._proactive_summarize(turns_to_compress=3)

        elif zone == "emergency":
            # Emergency pruning
            self._emergency_prune()

        # Always enforce turn limit
        self._enforce_turn_limit()

    def _enforce_turn_limit(self) -> None:
        """Enforce max_turns by moving old turns to compressed history."""
        while len(self._turns) > self.config.max_turns:
            oldest = self._turns.pop(0)
            self._total_tokens -= oldest.total_tokens()

            # Add to compressed history instead of dropping
            if not oldest.is_quarantined:
                self._add_to_compressed_history([oldest])

            LOGGER.debug(
                "Moved turn to compressed history, now at %d turns", len(self._turns)
            )

    def _proactive_summarize(self, turns_to_compress: int = 1) -> None:
        """
        Proactively summarize oldest turns while context is healthy.

        This runs at 70% capacity to prevent hitting emergency mode.
        """
        if len(self._turns) <= 3:
            # Keep at least 3 verbatim turns
            return

        # How many can we summarize while keeping 3?
        available = len(self._turns) - 3
        to_compress = min(turns_to_compress, available)

        if to_compress <= 0:
            return

        # Get turns to summarize (excluding quarantined)
        eligible_turns = [t for t in self._turns[:to_compress] if not t.is_quarantined]

        if not eligible_turns:
            # All oldest turns are quarantined, just drop them
            for _ in range(to_compress):
                removed = self._turns.pop(0)
                self._total_tokens -= removed.total_tokens()
            return

        # Summarize eligible turns
        self._add_to_compressed_history(eligible_turns)

        # Remove from verbatim buffer
        for _ in range(to_compress):
            removed = self._turns.pop(0)
            self._total_tokens -= removed.total_tokens()

        LOGGER.info(
            "Proactive summarization: compressed %d turns, now at %.1f%% capacity",
            to_compress,
            self.capacity_percentage * 100,
        )

    def _emergency_prune(self) -> None:
        """
        Emergency pruning when at 95%+ capacity.

        Drops oldest content (compressed history first, then turns).
        """
        LOGGER.warning(
            "Emergency pruning triggered at %.1f%% capacity",
            self.capacity_percentage * 100,
        )

        # First: drop oldest compressed histories
        while (
            self._compressed_history
            and self.capacity_percentage >= CapacityThresholds.AGGRESSIVE
        ):
            removed = self._compressed_history.pop(0)
            self._total_tokens -= removed.tokens
            LOGGER.warning("Dropped compressed history: %d tokens", removed.tokens)

        # If still over, drop oldest verbatim turns (keep at least 2)
        while (
            len(self._turns) > 2
            and self.capacity_percentage >= CapacityThresholds.AGGRESSIVE
        ):
            removed = self._turns.pop(0)
            self._total_tokens -= removed.total_tokens()
            LOGGER.warning("Dropped verbatim turn: %d tokens", removed.total_tokens())

    def _add_to_compressed_history(self, turns: List[ConversationTurn]) -> None:
        """Summarize turns and add to compressed history."""
        if not turns:
            return

        # Use injected summarizer or default
        summary_text = self._summarizer(turns)

        compressed = CompressedHistory(
            summary=summary_text,
            turn_count=len(turns),
            timestamp_start=turns[0].timestamp,
            timestamp_end=turns[-1].timestamp,
            topics_covered=self._extract_topics(turns),
        )

        self._compressed_history.append(compressed)
        self._total_tokens += compressed.tokens

        LOGGER.debug(
            "Added compressed history: %d turns → %d tokens",
            len(turns),
            compressed.tokens,
        )

    def _default_summarizer(self, turns: List[ConversationTurn]) -> str:
        """
        Default summarizer (simple extraction).

        In production, inject an LLM-based summarizer for better results.
        """
        if not turns:
            return ""

        summary_parts = []
        for turn in turns:
            # Extract key content (first 100 chars of each)
            user_snippet = turn.user_message[:100].replace("\n", " ").strip()
            if len(turn.user_message) > 100:
                user_snippet += "..."

            # Only include high-confidence responses in summary
            if turn.confidence >= 0.5:
                summary_parts.append(f"• User asked: {user_snippet}")

        if not summary_parts:
            return "[Earlier conversation - low confidence content omitted]"

        return "Prior context:\n" + "\n".join(summary_parts[:5])  # Max 5 items

    def _extract_topics(self, turns: List[ConversationTurn]) -> List[str]:
        """Extract topic keywords from turns for metadata."""
        topics = set()
        for turn in turns:
            text = f"{turn.user_message} {turn.assistant_response}".lower()

            # Simple keyword extraction
            if "scene" in text or "script" in text:
                topics.add("screenplay")
            if "telugu" in text or any("\u0c00" <= c <= "\u0c7f" for c in text):
                topics.add("telugu")
            if "friday" in text or "boss" in text:
                topics.add("personal")
            if turn.tool_calls:
                topics.add("tools")

        return list(topics)[:5]

    def set_summarizer(
        self, summarizer: Callable[[List[ConversationTurn]], str]
    ) -> None:
        """
        Inject an LLM-based summarizer for better compression.

        The summarizer should:
            1. Accept a list of ConversationTurn objects
            2. Return a concise summary string
            3. Preserve key facts and decisions
            4. Exclude low-confidence content

        Example:
            async def llm_summarizer(turns: List[ConversationTurn]) -> str:
                text = "\\n".join(f"User: {t.user_message}\\nAssistant: {t.assistant_response}"
                                 for t in turns if t.confidence >= 0.5)
                return await llm.complete(f"Summarize this conversation briefly:\\n{text}")

            wm.set_summarizer(llm_summarizer)
        """
        self._summarizer = summarizer
        LOGGER.info("Custom summarizer injected")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health metrics for monitoring."""
        quarantined = [t for t in self._turns if t.is_quarantined]
        low_confidence = [t for t in self._turns if t.confidence < 0.5]

        return {
            "healthy": self.capacity_zone in ("normal", "proactive"),
            "capacity_zone": self.capacity_zone,
            "capacity_percentage": self.capacity_percentage,
            "total_tokens": self._total_tokens,
            "max_tokens": self.config.max_tokens,
            "verbatim_turns": len(self._turns),
            "compressed_blocks": len(self._compressed_history),
            "quarantined_count": len(quarantined),
            "low_confidence_count": len(low_confidence),
            "warnings": self._get_health_warnings(),
        }

    def _get_health_warnings(self) -> List[str]:
        """Generate health warnings based on current state."""
        warnings = []

        if self.capacity_zone == "aggressive":
            warnings.append("Context at 85%+ capacity - aggressive compression active")
        elif self.capacity_zone == "emergency":
            warnings.append("CRITICAL: Context at 95%+ capacity - emergency pruning")

        quarantined_ratio = sum(1 for t in self._turns if t.is_quarantined) / max(
            1, len(self._turns)
        )
        if quarantined_ratio > 0.3:
            warnings.append(
                f"High quarantine rate: {quarantined_ratio:.0%} of turns flagged"
            )

        if len(self._compressed_history) > 10:
            warnings.append("Many compressed blocks - consider session reset")

        return warnings

    def clear(self) -> None:
        """Clear all conversation history and reset state."""
        self._turns.clear()
        self._compressed_history.clear()
        self._total_tokens = 0
        self._attention_stack.clear()
        self._prefetched_ltm.clear()
        self._poisoning_detector.reset()
        LOGGER.info("Working memory cleared")

    def get_full_context(self) -> str:
        """
        Get the complete context string for LLM injection.

        Structure:
            1. Compressed history (if any)
            2. Recent verbatim turns
            3. Attention topics (optional)

        Returns formatted context string.
        """
        parts = []

        # 1. Compressed history
        if self._compressed_history:
            history_parts = []
            for ch in self._compressed_history:
                history_parts.append(ch.summary)
            parts.append("## Earlier Context\n" + "\n".join(history_parts))

        # 2. Recent verbatim turns
        if self._turns:
            turn_parts = []
            for turn in self._turns:
                confidence_marker = "" if turn.confidence >= 0.7 else " [uncertain]"
                turn_parts.append(
                    f"User: {turn.user_message}\n"
                    f"Assistant: {turn.assistant_response}{confidence_marker}"
                )
            parts.append("## Recent Conversation\n" + "\n\n".join(turn_parts))

        # 3. Active attention (optional, for context)
        if self._attention_stack:
            topics = [
                f"- {a.topic} ({a.relevance:.0%})" for a in self._attention_stack[:3]
            ]
            parts.append("## Active Focus\n" + "\n".join(topics))

        return "\n\n".join(parts)

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about current context usage."""
        return {
            "total_tokens": self._total_tokens,
            "max_tokens": self.config.max_tokens,
            "capacity_percentage": self.capacity_percentage,
            "capacity_zone": self.capacity_zone,
            "verbatim_turns": len(self._turns),
            "compressed_histories": len(self._compressed_history),
            "attention_items": len(self._attention_stack),
            "prefetched_ltm": len(self._prefetched_ltm),
            "quarantined_turns": sum(1 for t in self._turns if t.is_quarantined),
        }

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
        """Serialize working memory state."""
        return {
            "compressed_history": [
                {
                    "summary": ch.summary,
                    "turn_count": ch.turn_count,
                    "timestamp_start": ch.timestamp_start.isoformat(),
                    "timestamp_end": ch.timestamp_end.isoformat(),
                    "tokens": ch.tokens,
                    "topics": ch.topics_covered,
                }
                for ch in self._compressed_history
            ],
            "turns": [
                {
                    "user": t.user_message,
                    "assistant": t.assistant_response,
                    "timestamp": t.timestamp.isoformat(),
                    "context": t.context_type,
                    "confidence": t.confidence,
                    "is_quarantined": t.is_quarantined,
                    "content_hash": t.content_hash,
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
            "capacity_zone": self.capacity_zone,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        config: Optional[WorkingMemoryConfig] = None,
        summarizer: Optional[Callable] = None,
    ) -> "WorkingMemory":
        """Restore working memory from dict."""
        wm = cls(config, summarizer)

        # Restore compressed history
        for ch in data.get("compressed_history", []):
            wm._compressed_history.append(
                CompressedHistory(
                    summary=ch["summary"],
                    turn_count=ch["turn_count"],
                    timestamp_start=datetime.fromisoformat(ch["timestamp_start"]),
                    timestamp_end=datetime.fromisoformat(ch["timestamp_end"]),
                    tokens=ch.get("tokens", 0),
                    topics_covered=ch.get("topics", []),
                )
            )

        # Restore turns
        for t in data.get("turns", []):
            turn = ConversationTurn(
                user_message=t["user"],
                assistant_response=t["assistant"],
                timestamp=datetime.fromisoformat(t["timestamp"]),
                context_type=t.get("context", "general"),
            )
            turn.confidence = t.get("confidence", 1.0)
            turn.is_quarantined = t.get("is_quarantined", False)
            turn.content_hash = t.get("content_hash", "")
            wm._turns.append(turn)

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
            f"WorkingMemory("
            f"turns={len(self._turns)}, "
            f"compressed={len(self._compressed_history)}, "
            f"tokens={self._total_tokens}/{self.config.max_tokens} ({self.capacity_percentage:.0%}), "
            f"zone={self.capacity_zone}, "
            f"attention={len(self._attention_stack)}, "
            f"room={self._current_room})"
        )
