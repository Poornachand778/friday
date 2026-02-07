"""
Tests for Working Memory Layer
================================

Tests TokenCounter, ConversationTurn, ContextPoisoningDetector,
AttentionItem, and WorkingMemory capacity management.

Run with: pytest tests/test_working_memory.py -v
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from memory.config import WorkingMemoryConfig
from memory.layers.working import (
    AttentionItem,
    CapacityThresholds,
    CompressedHistory,
    ContextPoisoningDetector,
    ConversationTurn,
    PrefetchedMemory,
    TokenCounter,
    WorkingMemory,
)


# =========================================================================
# TokenCounter
# =========================================================================


class TestTokenCounter:
    """Test token counting utilities"""

    def test_empty_string(self):
        assert TokenCounter.count("") == 0

    def test_english_text(self):
        count = TokenCounter.count("Hello, how are you?")
        assert count > 0

    def test_telugu_text(self):
        """Telugu characters use ~2.5 chars per token estimate"""
        count = TokenCounter.count("\u0c28\u0c2e\u0c38\u0c4d\u0c15\u0c3e\u0c30\u0c02")
        assert count > 0

    def test_mixed_text(self):
        count = TokenCounter.count(
            "Boss, \u0c2c\u0c3e\u0c17\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c30\u0c3e?"
        )
        assert count > 0

    def test_longer_text_more_tokens(self):
        short = TokenCounter.count("hi")
        long = TokenCounter.count(
            "This is a much longer sentence with many more tokens in it"
        )
        assert long > short

    def test_estimate_tokens_english(self):
        """~4 chars per token for English"""
        count = TokenCounter._estimate_tokens("abcdefghijklmnop")  # 16 chars
        assert count == 4  # 16 / 4 = 4

    def test_estimate_tokens_telugu(self):
        """~2.5 chars per token for Telugu"""
        # 5 Telugu chars → 5 / 2.5 = 2 tokens
        count = TokenCounter._estimate_tokens("\u0c05\u0c06\u0c07\u0c08\u0c09")
        assert count == 2

    def test_estimate_tokens_empty(self):
        assert TokenCounter._estimate_tokens("") == 0

    def test_count_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = TokenCounter.count_messages(messages)
        # 2 messages * 4 overhead + role tokens + content tokens
        assert count > 8  # At minimum, overhead is 8

    def test_count_messages_empty_list(self):
        assert TokenCounter.count_messages([]) == 0


# =========================================================================
# CapacityThresholds
# =========================================================================


class TestCapacityThresholds:
    """Test capacity threshold values"""

    def test_proactive_threshold(self):
        assert CapacityThresholds.PROACTIVE == 0.70

    def test_aggressive_threshold(self):
        assert CapacityThresholds.AGGRESSIVE == 0.85

    def test_emergency_threshold(self):
        assert CapacityThresholds.EMERGENCY == 0.95

    def test_buffer_allocations_sum_to_one(self):
        total = (
            CapacityThresholds.COMPRESSED_HISTORY
            + CapacityThresholds.RECENT_VERBATIM
            + CapacityThresholds.ATTENTION_RESERVE
        )
        assert total == pytest.approx(1.0)


# =========================================================================
# ConversationTurn
# =========================================================================


class TestConversationTurn:
    """Test ConversationTurn dataclass"""

    def test_basic_creation(self):
        turn = ConversationTurn(
            user_message="Hello Boss",
            assistant_response="Boss, baagunnanu!",
        )
        assert turn.user_message == "Hello Boss"
        assert turn.assistant_response == "Boss, baagunnanu!"
        assert turn.user_tokens > 0
        assert turn.assistant_tokens > 0

    def test_auto_token_counting(self):
        turn = ConversationTurn(
            user_message="short",
            assistant_response="also short",
        )
        assert turn.user_tokens > 0
        assert turn.assistant_tokens > 0

    def test_total_tokens(self):
        turn = ConversationTurn(
            user_message="Hello",
            assistant_response="Hi",
        )
        assert turn.total_tokens() == turn.user_tokens + turn.assistant_tokens

    def test_content_hash_computed(self):
        turn = ConversationTurn(
            user_message="test",
            assistant_response="response",
        )
        assert turn.content_hash != ""
        assert len(turn.content_hash) == 16  # MD5 truncated to 16 chars

    def test_same_response_same_hash(self):
        turn1 = ConversationTurn(user_message="q1", assistant_response="same response")
        turn2 = ConversationTurn(user_message="q2", assistant_response="same response")
        assert turn1.content_hash == turn2.content_hash

    def test_different_response_different_hash(self):
        turn1 = ConversationTurn(user_message="q", assistant_response="response one")
        turn2 = ConversationTurn(user_message="q", assistant_response="response two")
        assert turn1.content_hash != turn2.content_hash

    def test_default_confidence(self):
        turn = ConversationTurn(user_message="q", assistant_response="a")
        assert turn.confidence == 1.0

    def test_default_not_quarantined(self):
        turn = ConversationTurn(user_message="q", assistant_response="a")
        assert turn.is_quarantined is False

    def test_mark_quarantined(self):
        turn = ConversationTurn(user_message="q", assistant_response="a")
        turn.mark_quarantined("test reason")
        assert turn.is_quarantined is True

    def test_default_context_type(self):
        turn = ConversationTurn(user_message="q", assistant_response="a")
        assert turn.context_type == "general"

    def test_custom_context_type(self):
        turn = ConversationTurn(
            user_message="q", assistant_response="a", context_type="writers_room"
        )
        assert turn.context_type == "writers_room"

    def test_tool_calls_default_empty(self):
        turn = ConversationTurn(user_message="q", assistant_response="a")
        assert turn.tool_calls == []
        assert turn.tool_results == []

    def test_timestamp_auto_set(self):
        turn = ConversationTurn(user_message="q", assistant_response="a")
        assert isinstance(turn.timestamp, datetime)


# =========================================================================
# CompressedHistory
# =========================================================================


class TestCompressedHistory:
    """Test CompressedHistory dataclass"""

    def test_basic_creation(self):
        ch = CompressedHistory(
            summary="User asked about scenes",
            turn_count=3,
            timestamp_start=datetime(2025, 1, 1),
            timestamp_end=datetime(2025, 1, 1),
        )
        assert ch.summary == "User asked about scenes"
        assert ch.turn_count == 3

    def test_auto_token_count(self):
        ch = CompressedHistory(
            summary="Some summary text here",
            turn_count=2,
            timestamp_start=datetime(2025, 1, 1),
            timestamp_end=datetime(2025, 1, 1),
        )
        assert ch.tokens > 0

    def test_topics_default_empty(self):
        ch = CompressedHistory(
            summary="test",
            turn_count=1,
            timestamp_start=datetime(2025, 1, 1),
            timestamp_end=datetime(2025, 1, 1),
        )
        assert ch.topics_covered == []


# =========================================================================
# ContextPoisoningDetector
# =========================================================================


class TestContextPoisoningDetector:
    """Test context poisoning detection"""

    def setup_method(self):
        self.detector = ContextPoisoningDetector(repetition_threshold=3)

    def test_clean_turn_high_confidence(self):
        turn = ConversationTurn(
            user_message="What's the scene about?",
            assistant_response="The scene is about a confrontation.",
        )
        confidence, warnings = self.detector.analyze_turn(turn)
        assert confidence >= 0.9
        assert len(warnings) == 0

    def test_uncertainty_markers_reduce_confidence(self):
        turn = ConversationTurn(
            user_message="What happened?",
            assistant_response="I think it might be something, I'm not sure, possibly related.",
        )
        confidence, warnings = self.detector.analyze_turn(turn)
        assert confidence < 1.0
        assert any("uncertainty" in w for w in warnings)

    def test_self_reference_reduces_confidence(self):
        turn = ConversationTurn(
            user_message="Tell me again",
            assistant_response="As I mentioned earlier, like I said, as stated earlier, this is the case.",
        )
        confidence, warnings = self.detector.analyze_turn(turn)
        assert confidence < 1.0
        assert any("self_ref" in w for w in warnings)

    def test_repetition_detection(self):
        """Same content hash 3+ times triggers warning"""
        turn = ConversationTurn(user_message="q", assistant_response="repeated answer")
        # First two are fine
        self.detector.analyze_turn(turn)
        self.detector.analyze_turn(turn)
        # Third triggers repetition warning
        confidence, warnings = self.detector.analyze_turn(turn)
        assert confidence < 1.0
        assert any("content_repeated" in w for w in warnings)

    def test_should_quarantine_low_confidence(self):
        turn = ConversationTurn(user_message="q", assistant_response="a")
        turn.confidence = 0.3
        assert self.detector.should_quarantine(turn)

    def test_should_not_quarantine_high_confidence(self):
        turn = ConversationTurn(user_message="q", assistant_response="a")
        turn.confidence = 0.8
        assert not self.detector.should_quarantine(turn)

    def test_should_quarantine_high_repetition(self):
        turn = ConversationTurn(user_message="q", assistant_response="a")
        turn.repetition_count = 5
        assert self.detector.should_quarantine(turn)

    def test_reset_clears_state(self):
        turn = ConversationTurn(user_message="q", assistant_response="answer")
        self.detector.analyze_turn(turn)
        self.detector.analyze_turn(turn)
        self.detector.reset()
        # After reset, same content should not trigger repetition
        confidence, warnings = self.detector.analyze_turn(turn)
        assert not any("content_repeated" in w for w in warnings)

    def test_confidence_clamped_to_0_1(self):
        """Confidence should never go below 0.0"""
        turn = ConversationTurn(
            user_message="q",
            assistant_response=(
                "I think I believe it might be, I'm not sure, possibly, perhaps, "
                "I may have assumed, if I recall, it seems, I'm uncertain, could be."
            ),
        )
        confidence, _ = self.detector.analyze_turn(turn)
        assert 0.0 <= confidence <= 1.0


# =========================================================================
# AttentionItem
# =========================================================================


class TestAttentionItem:
    """Test AttentionItem dataclass"""

    def test_basic_creation(self):
        item = AttentionItem(topic="screenplay", relevance=0.8)
        assert item.topic == "screenplay"
        assert item.relevance == 0.8

    def test_decay(self):
        item = AttentionItem(topic="test", relevance=1.0)
        item.decay(0.1)
        assert item.relevance == pytest.approx(0.9)

    def test_decay_multiple(self):
        item = AttentionItem(topic="test", relevance=1.0)
        item.decay(0.1)
        item.decay(0.1)
        assert item.relevance == pytest.approx(0.81)

    def test_zero_decay(self):
        item = AttentionItem(topic="test", relevance=0.5)
        item.decay(0.0)
        assert item.relevance == 0.5

    def test_full_decay(self):
        item = AttentionItem(topic="test", relevance=0.5)
        item.decay(1.0)
        assert item.relevance == 0.0

    def test_auto_timestamp(self):
        item = AttentionItem(topic="test", relevance=0.5)
        assert isinstance(item.timestamp, datetime)

    def test_metadata_default_empty(self):
        item = AttentionItem(topic="test", relevance=0.5)
        assert item.metadata == {}


# =========================================================================
# WorkingMemory - Basic Operations
# =========================================================================


def _make_config(**overrides) -> WorkingMemoryConfig:
    """Create a test config with small limits for fast testing"""
    defaults = {
        "max_turns": 5,
        "max_tokens": 1000,
        "attention_decay_rate": 0.1,
        "max_attention_items": 7,
    }
    defaults.update(overrides)
    return WorkingMemoryConfig(**defaults)


class TestWorkingMemoryBasic:
    """Test basic WorkingMemory operations"""

    def setup_method(self):
        self.config = _make_config()
        self.wm = WorkingMemory(config=self.config)

    def test_initial_state(self):
        assert self.wm.turn_count == 0
        assert self.wm.token_count == 0
        assert self.wm.capacity_zone == "normal"

    def test_add_turn(self):
        turn = self.wm.add_turn("Hello", "Hi there!")
        assert self.wm.turn_count == 1
        assert self.wm.token_count > 0
        assert turn.user_message == "Hello"

    def test_add_multiple_turns(self):
        self.wm.add_turn("q1", "a1")
        self.wm.add_turn("q2", "a2")
        self.wm.add_turn("q3", "a3")
        assert self.wm.turn_count == 3

    def test_get_turns_all(self):
        self.wm.add_turn("q1", "a1")
        self.wm.add_turn("q2", "a2")
        turns = self.wm.get_turns()
        assert len(turns) == 2

    def test_get_turns_limited(self):
        self.wm.add_turn("q1", "a1")
        self.wm.add_turn("q2", "a2")
        self.wm.add_turn("q3", "a3")
        turns = self.wm.get_turns(2)
        assert len(turns) == 2
        assert turns[0].user_message == "q2"
        assert turns[1].user_message == "q3"

    def test_get_last_turn(self):
        self.wm.add_turn("q1", "a1")
        self.wm.add_turn("q2", "a2")
        last = self.wm.get_last_turn()
        assert last is not None
        assert last.user_message == "q2"

    def test_get_last_turn_empty(self):
        assert self.wm.get_last_turn() is None

    def test_clear(self):
        self.wm.add_turn("q1", "a1")
        self.wm.add_turn("q2", "a2")
        self.wm.update_attention("test", 0.8)
        self.wm.clear()
        assert self.wm.turn_count == 0
        assert self.wm.token_count == 0
        assert len(self.wm.get_attention_topics()) == 0

    def test_add_turn_with_tool_calls(self):
        turn = self.wm.add_turn(
            "search for scene 1",
            "Here's scene 1",
            tool_calls=[{"name": "scene_get", "args": {"scene_number": 1}}],
            tool_results=[{"data": {"scene_id": 1, "title": "Opening"}}],
        )
        assert len(turn.tool_calls) == 1
        assert len(turn.tool_results) == 1

    def test_add_turn_with_context_type(self):
        turn = self.wm.add_turn("q", "a", context_type="writers_room")
        assert turn.context_type == "writers_room"


# =========================================================================
# WorkingMemory - Capacity Management
# =========================================================================


class TestWorkingMemoryCapacity:
    """Test capacity zones and compression"""

    def test_capacity_percentage(self):
        config = _make_config(max_tokens=100)
        wm = WorkingMemory(config=config)
        # Add a turn and check capacity is non-zero
        wm.add_turn("hello", "hi")
        assert wm.capacity_percentage > 0.0

    def test_capacity_zone_normal(self):
        config = _make_config(max_tokens=10000)
        wm = WorkingMemory(config=config)
        wm.add_turn("hello", "hi")
        assert wm.capacity_zone == "normal"

    def test_tokens_available(self):
        config = _make_config(max_tokens=10000)
        wm = WorkingMemory(config=config)
        available_before = wm.tokens_available
        wm.add_turn("hello", "hi")
        assert wm.tokens_available < available_before

    def test_enforce_turn_limit(self):
        """Turns exceeding max_turns should be moved to compressed history"""
        config = _make_config(max_turns=3, max_tokens=50000)
        wm = WorkingMemory(config=config)
        for i in range(5):
            wm.add_turn(f"question {i}", f"answer {i}")
        # Should have at most 3 verbatim turns
        assert wm.turn_count <= 3
        # Older turns should be in compressed history
        assert len(wm._compressed_history) > 0

    def test_proactive_summarization_at_70_percent(self):
        """At 70%+ capacity, proactive summarization should compress oldest turns"""
        config = _make_config(max_turns=20, max_tokens=100)
        wm = WorkingMemory(config=config)
        # Add turns until we exceed 70%
        for i in range(10):
            wm.add_turn(
                f"This is a longer question number {i}",
                f"This is a longer answer number {i}",
            )
        # Should have compressed some turns
        if wm.capacity_percentage >= 0.70:
            # Either compressed history exists or turns were pruned
            assert len(wm._compressed_history) > 0 or wm.turn_count < 10

    def test_emergency_prune_keeps_minimum_turns(self):
        """Emergency pruning should keep at least 2 verbatim turns"""
        config = _make_config(max_turns=20, max_tokens=50)
        wm = WorkingMemory(config=config)
        for i in range(10):
            wm.add_turn(f"question {i}", f"answer {i}")
        # After emergency pruning, should have at least 2 turns
        assert wm.turn_count >= 2


# =========================================================================
# WorkingMemory - Attention Stack
# =========================================================================


class TestWorkingMemoryAttention:
    """Test attention stack management"""

    def setup_method(self):
        self.config = _make_config(max_attention_items=7)
        self.wm = WorkingMemory(config=self.config)

    def test_update_attention_adds_item(self):
        self.wm.update_attention("screenplay", 0.8)
        topics = self.wm.get_attention_topics()
        assert len(topics) == 1
        assert topics[0].topic == "screenplay"

    def test_update_attention_existing_topic(self):
        self.wm.update_attention("screenplay", 0.5)
        self.wm.update_attention("screenplay", 0.9)
        topics = self.wm.get_attention_topics()
        # Should update, not duplicate
        screenplay_items = [t for t in topics if t.topic == "screenplay"]
        assert len(screenplay_items) == 1
        assert screenplay_items[0].relevance >= 0.5

    def test_attention_decay_on_update(self):
        self.wm.update_attention("old_topic", 0.5)
        initial_relevance = self.wm.get_attention_topics()[0].relevance
        # Adding new topic decays existing
        self.wm.update_attention("new_topic", 0.8)
        old_item = next(
            (t for t in self.wm.get_attention_topics() if t.topic == "old_topic"),
            None,
        )
        if old_item:
            assert old_item.relevance < initial_relevance

    def test_attention_pruning_below_threshold(self):
        """Items below 0.2 relevance should be pruned"""
        self.wm.update_attention("weak", 0.15)
        # Should be pruned immediately since below 0.2
        topics = self.wm.get_attention_topics()
        weak_items = [t for t in topics if t.topic == "weak"]
        assert len(weak_items) == 0

    def test_attention_max_items(self):
        """Should keep at most max_attention_items (7)"""
        for i in range(10):
            self.wm.update_attention(f"topic_{i}", 0.9)
        topics = self.wm.get_attention_topics()
        assert len(topics) <= 7

    def test_attention_sorted_by_relevance(self):
        self.wm.update_attention("low", 0.4)
        self.wm.update_attention("high", 0.9)
        self.wm.update_attention("mid", 0.6)
        topics = self.wm.get_attention_topics()
        relevances = [t.relevance for t in topics]
        assert relevances == sorted(relevances, reverse=True)

    def test_get_top_attention(self):
        self.wm.update_attention("low", 0.4)
        self.wm.update_attention("high", 0.9)
        top = self.wm.get_top_attention()
        assert top is not None
        assert top.topic == "high"

    def test_get_top_attention_empty(self):
        assert self.wm.get_top_attention() is None

    def test_is_attending_to(self):
        self.wm.update_attention("screenplay", 0.8)
        assert self.wm.is_attending_to("screenplay")
        assert not self.wm.is_attending_to("cooking")

    def test_is_attending_to_partial_match(self):
        self.wm.update_attention("screenplay writing", 0.8)
        assert self.wm.is_attending_to("screenplay")

    def test_is_attending_to_below_threshold(self):
        self.wm.update_attention("weak_topic", 0.25)
        assert not self.wm.is_attending_to("weak_topic", threshold=0.3)

    def test_attention_with_metadata(self):
        self.wm.update_attention("project", 0.8, metadata={"type": "project"})
        topics = self.wm.get_attention_topics()
        assert topics[0].metadata == {"type": "project"}


# =========================================================================
# WorkingMemory - Ephemeral State
# =========================================================================


class TestWorkingMemoryState:
    """Test ephemeral state management"""

    def setup_method(self):
        self.wm = WorkingMemory(config=_make_config())

    def test_default_room(self):
        assert self.wm.current_room == "general"

    def test_set_room(self):
        self.wm.set_room("writers_room")
        assert self.wm.current_room == "writers_room"

    def test_set_room_updates_attention(self):
        self.wm.set_room("kitchen")
        assert self.wm.is_attending_to("room:kitchen")

    def test_set_project(self):
        self.wm.set_project("aa-janta-naduma")
        assert self.wm.current_project == "aa-janta-naduma"

    def test_set_project_updates_attention(self):
        self.wm.set_project("gusagusalu")
        assert self.wm.is_attending_to("gusagusalu")

    def test_set_project_none(self):
        self.wm.set_project("test")
        self.wm.set_project(None)
        assert self.wm.current_project is None

    def test_set_language_mode(self):
        self.wm.set_language_mode("te")
        assert self.wm.language_mode == "te"

    def test_default_language_mode(self):
        assert self.wm.language_mode == "mixed"

    def test_set_emotional_context(self):
        self.wm.set_emotional_context("excited")
        assert self.wm.emotional_context == "excited"

    def test_default_emotional_context(self):
        assert self.wm.emotional_context == "neutral"

    def test_set_active_task(self):
        self.wm.set_active_task("writing scene 5")
        assert self.wm.active_task == "writing scene 5"

    def test_set_active_task_updates_attention(self):
        self.wm.set_active_task("revise climax")
        assert self.wm.is_attending_to("task:revise climax")

    def test_set_active_task_none(self):
        self.wm.set_active_task("test")
        self.wm.set_active_task(None)
        assert self.wm.active_task is None


# =========================================================================
# WorkingMemory - Prefetched LTM
# =========================================================================


class TestWorkingMemoryLTM:
    """Test prefetched LTM management"""

    def setup_method(self):
        self.wm = WorkingMemory(config=_make_config())

    def test_set_prefetched_ltm(self):
        memories = [
            PrefetchedMemory(
                content="test memory",
                relevance=0.8,
                memory_id="mem_001",
                memory_type="fact",
            )
        ]
        self.wm.set_prefetched_ltm(memories)
        assert len(self.wm.get_prefetched_ltm()) == 1

    def test_get_prefetched_ltm_returns_copy(self):
        memories = [
            PrefetchedMemory(
                content="test",
                relevance=0.5,
                memory_id="m1",
                memory_type="fact",
            )
        ]
        self.wm.set_prefetched_ltm(memories)
        result = self.wm.get_prefetched_ltm()
        result.clear()
        # Original should still have items
        assert len(self.wm.get_prefetched_ltm()) == 1

    def test_clear_prefetched_ltm(self):
        self.wm.set_prefetched_ltm(
            [
                PrefetchedMemory(
                    content="x", relevance=0.5, memory_id="m1", memory_type="fact"
                )
            ]
        )
        self.wm.clear_prefetched_ltm()
        assert len(self.wm.get_prefetched_ltm()) == 0


# =========================================================================
# WorkingMemory - Context Poisoning Integration
# =========================================================================


class TestWorkingMemoryPoisoning:
    """Test poisoning detection integration with add_turn"""

    def test_uncertain_response_lowers_confidence(self):
        wm = WorkingMemory(config=_make_config())
        turn = wm.add_turn(
            "What happened?",
            "I think it might be something, I'm not sure.",
        )
        assert turn.confidence < 1.0

    def test_repeated_content_triggers_quarantine(self):
        wm = WorkingMemory(config=_make_config())
        for _ in range(3):
            turn = wm.add_turn("question", "exact same answer")
        # Third repetition should lower confidence
        assert turn.confidence < 1.0

    def test_clean_response_stays_high_confidence(self):
        wm = WorkingMemory(config=_make_config())
        turn = wm.add_turn(
            "What is scene 5 about?",
            "Scene 5 is the confrontation between Arjun and Neelima at the courthouse.",
        )
        assert turn.confidence >= 0.9


# =========================================================================
# WorkingMemory - Health & Stats
# =========================================================================


class TestWorkingMemoryHealth:
    """Test health status and context stats"""

    def setup_method(self):
        self.wm = WorkingMemory(config=_make_config(max_tokens=10000))

    def test_health_status_structure(self):
        health = self.wm.get_health_status()
        assert "healthy" in health
        assert "capacity_zone" in health
        assert "capacity_percentage" in health
        assert "total_tokens" in health
        assert "max_tokens" in health
        assert "verbatim_turns" in health
        assert "warnings" in health

    def test_health_status_healthy_initially(self):
        health = self.wm.get_health_status()
        assert health["healthy"] is True

    def test_context_stats_structure(self):
        stats = self.wm.get_context_stats()
        assert "total_tokens" in stats
        assert "max_tokens" in stats
        assert "capacity_zone" in stats
        assert "verbatim_turns" in stats
        assert "attention_items" in stats
        assert "prefetched_ltm" in stats
        assert "quarantined_turns" in stats

    def test_context_stats_counts(self):
        self.wm.add_turn("q1", "a1")
        self.wm.add_turn("q2", "a2")
        self.wm.update_attention("test", 0.8)
        stats = self.wm.get_context_stats()
        assert stats["verbatim_turns"] == 2
        assert stats["attention_items"] >= 1

    def test_get_full_context(self):
        self.wm.add_turn("hello", "hi there")
        context = self.wm.get_full_context()
        assert "hello" in context
        assert "hi there" in context


# =========================================================================
# WorkingMemory - Serialization
# =========================================================================


class TestWorkingMemorySerialization:
    """Test to_dict / from_dict round-trip"""

    def test_to_dict_structure(self):
        wm = WorkingMemory(config=_make_config())
        wm.add_turn("hello", "hi")
        wm.update_attention("test", 0.8)
        wm.set_room("kitchen")

        data = wm.to_dict()
        assert "compressed_history" in data
        assert "turns" in data
        assert "attention" in data
        assert "state" in data
        assert "token_count" in data
        assert "capacity_zone" in data

    def test_to_dict_turns_content(self):
        wm = WorkingMemory(config=_make_config())
        wm.add_turn("hello", "hi")
        data = wm.to_dict()
        assert len(data["turns"]) == 1
        assert data["turns"][0]["user"] == "hello"
        assert data["turns"][0]["assistant"] == "hi"

    def test_to_dict_state(self):
        wm = WorkingMemory(config=_make_config())
        wm.set_room("kitchen")
        wm.set_language_mode("te")
        wm.set_emotional_context("excited")
        data = wm.to_dict()
        assert data["state"]["room"] == "kitchen"
        assert data["state"]["language"] == "te"
        assert data["state"]["emotion"] == "excited"

    def test_from_dict_restores_turns(self):
        wm = WorkingMemory(config=_make_config())
        wm.add_turn("hello", "hi")
        wm.add_turn("q2", "a2")
        data = wm.to_dict()

        restored = WorkingMemory.from_dict(data, config=_make_config())
        assert restored.turn_count == 2
        turns = restored.get_turns()
        assert turns[0].user_message == "hello"
        assert turns[1].user_message == "q2"

    def test_from_dict_restores_state(self):
        wm = WorkingMemory(config=_make_config())
        wm.set_room("kitchen")
        wm.set_project("gusagusalu")
        wm.set_language_mode("te")
        wm.set_emotional_context("thoughtful")
        wm.set_active_task("writing")
        data = wm.to_dict()

        restored = WorkingMemory.from_dict(data, config=_make_config())
        assert restored.current_room == "kitchen"
        assert restored.current_project == "gusagusalu"
        assert restored.language_mode == "te"
        assert restored.emotional_context == "thoughtful"
        assert restored.active_task == "writing"

    def test_from_dict_restores_attention(self):
        wm = WorkingMemory(config=_make_config())
        wm.update_attention("screenplay", 0.9)
        data = wm.to_dict()

        restored = WorkingMemory.from_dict(data, config=_make_config())
        topics = restored.get_attention_topics()
        assert len(topics) >= 1
        assert any(t.topic == "screenplay" for t in topics)

    def test_from_dict_empty(self):
        restored = WorkingMemory.from_dict({}, config=_make_config())
        assert restored.turn_count == 0
        assert restored.current_room == "general"

    def test_round_trip_preserves_confidence(self):
        wm = WorkingMemory(config=_make_config())
        turn = wm.add_turn(
            "q",
            "I think it might be something, I'm not sure.",
        )
        original_confidence = turn.confidence
        data = wm.to_dict()

        restored = WorkingMemory.from_dict(data, config=_make_config())
        restored_turn = restored.get_last_turn()
        assert restored_turn.confidence == original_confidence


# =========================================================================
# WorkingMemory - Default Summarizer
# =========================================================================


class TestDefaultSummarizer:
    """Test the default summarizer behavior"""

    def test_default_summarizer_produces_output(self):
        wm = WorkingMemory(config=_make_config())
        turns = [
            ConversationTurn(
                user_message="What about scene 1?",
                assistant_response="Scene 1 is the opening.",
            ),
        ]
        summary = wm._default_summarizer(turns)
        assert "User asked" in summary

    def test_default_summarizer_skips_low_confidence(self):
        wm = WorkingMemory(config=_make_config())
        turn = ConversationTurn(user_message="q", assistant_response="a")
        turn.confidence = 0.3
        summary = wm._default_summarizer([turn])
        assert "low confidence" in summary.lower()

    def test_default_summarizer_empty(self):
        wm = WorkingMemory(config=_make_config())
        summary = wm._default_summarizer([])
        assert summary == ""

    def test_set_custom_summarizer(self):
        wm = WorkingMemory(config=_make_config())
        custom_called = []

        def custom_summarizer(turns):
            custom_called.append(True)
            return "custom summary"

        wm.set_summarizer(custom_summarizer)
        # Force summarization by exceeding max_turns
        config = _make_config(max_turns=2, max_tokens=50000)
        wm2 = WorkingMemory(config=config, summarizer=custom_summarizer)
        for i in range(5):
            wm2.add_turn(f"q{i}", f"a{i}")
        assert len(custom_called) > 0


# =========================================================================
# WorkingMemory - Repr
# =========================================================================


class TestWorkingMemoryRepr:
    """Test __repr__ output"""

    def test_repr_contains_key_info(self):
        wm = WorkingMemory(config=_make_config())
        wm.add_turn("q", "a")
        repr_str = repr(wm)
        assert "WorkingMemory(" in repr_str
        assert "turns=" in repr_str
        assert "zone=" in repr_str
        assert "room=" in repr_str
