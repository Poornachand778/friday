"""
Tests for ConversationMemory
==============================

Tests turn creation, memory management, context messages,
summarization, serialization, and edge cases.

Run with: pytest tests/test_conversation_memory.py -v
"""

import sys
import time
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from orchestrator.memory.conversation import ConversationMemory, ConversationTurn


# =========================================================================
# ConversationTurn
# =========================================================================


class TestConversationTurn:
    """Test ConversationTurn dataclass"""

    def test_basic_creation(self):
        turn = ConversationTurn(
            turn_id=1,
            user_message="Hello Boss",
            assistant_response="Hi there",
        )
        assert turn.turn_id == 1
        assert turn.user_message == "Hello Boss"
        assert turn.assistant_response == "Hi there"

    def test_defaults(self):
        turn = ConversationTurn(
            turn_id=1,
            user_message="msg",
            assistant_response="resp",
        )
        assert turn.tool_calls == []
        assert turn.tool_results == []
        assert turn.context_type == "general"
        assert turn.metadata == {}
        assert isinstance(turn.timestamp, float)

    def test_token_estimate(self):
        turn = ConversationTurn(
            turn_id=1,
            user_message="Hello",  # 5 chars
            assistant_response="World",  # 5 chars
        )
        # 10 chars / 4 = 2 tokens
        assert turn.token_estimate == 2

    def test_token_estimate_with_tool_calls(self):
        turn = ConversationTurn(
            turn_id=1,
            user_message="a",
            assistant_response="b",
            tool_calls=[{"name": "scene_search", "args": {"query": "test"}}],
            tool_results=[{"data": "result content here"}],
        )
        # Should include tool_calls and tool_results in estimate
        assert turn.token_estimate > 0

    def test_to_messages_basic(self):
        turn = ConversationTurn(
            turn_id=1,
            user_message="What scenes?",
            assistant_response="Found 3 scenes.",
        )
        messages = turn.to_messages()
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "What scenes?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Found 3 scenes."

    def test_to_messages_with_tool_calls(self):
        turn = ConversationTurn(
            turn_id=1,
            user_message="Search for romantic scenes",
            assistant_response="Found results",
            tool_calls=[{"name": "scene_search", "arguments": '{"query": "romantic"}'}],
            tool_results=[
                {"data": "scene 1", "tool_call_id": "tc_1", "name": "scene_search"}
            ],
        )
        messages = turn.to_messages()
        # user + assistant (tool_calls) + tool result + assistant (response)
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[1].tool_calls is not None
        assert messages[2].role == "tool"
        assert messages[2].tool_call_id == "tc_1"
        assert messages[3].role == "assistant"
        assert messages[3].content == "Found results"

    def test_to_messages_tool_result_error(self):
        turn = ConversationTurn(
            turn_id=1,
            user_message="Search",
            assistant_response="Failed",
            tool_calls=[{"name": "tool1"}],
            tool_results=[
                {"error": "Not found", "tool_call_id": "tc_1", "name": "tool1"}
            ],
        )
        messages = turn.to_messages()
        # The tool message content should come from error field
        tool_msg = [m for m in messages if m.role == "tool"][0]
        assert "Not found" in tool_msg.content

    def test_context_type_custom(self):
        turn = ConversationTurn(
            turn_id=1,
            user_message="msg",
            assistant_response="resp",
            context_type="film_writing",
        )
        assert turn.context_type == "film_writing"

    def test_metadata_stored(self):
        turn = ConversationTurn(
            turn_id=1,
            user_message="msg",
            assistant_response="resp",
            metadata={"language": "te-en", "emotion": "excited"},
        )
        assert turn.metadata["language"] == "te-en"
        assert turn.metadata["emotion"] == "excited"


# =========================================================================
# ConversationMemory - Init
# =========================================================================


class TestConversationMemoryInit:
    """Test memory initialization"""

    def test_default_init(self):
        mem = ConversationMemory()
        assert mem.max_turns == 20
        assert mem.max_tokens == 4000
        assert mem.summarize_threshold == 15
        assert mem.turn_count == 0
        assert mem.active_turns == 0
        assert mem.current_context == "general"

    def test_custom_init(self):
        mem = ConversationMemory(max_turns=10, max_tokens=2000, summarize_threshold=8)
        assert mem.max_turns == 10
        assert mem.max_tokens == 2000
        assert mem.summarize_threshold == 8

    def test_session_id_none(self):
        mem = ConversationMemory()
        assert mem.session_id is None

    def test_started_at_set(self):
        before = time.time()
        mem = ConversationMemory()
        after = time.time()
        assert before <= mem.started_at <= after

    def test_total_tokens_empty(self):
        mem = ConversationMemory()
        assert mem.total_tokens == 0


# =========================================================================
# ConversationMemory - Add Turn
# =========================================================================


class TestAddTurn:
    """Test adding turns to memory"""

    def test_add_turn_returns_turn(self):
        mem = ConversationMemory()
        turn = mem.add_turn("Hello", "Hi")
        assert isinstance(turn, ConversationTurn)

    def test_add_turn_increments_counter(self):
        mem = ConversationMemory()
        mem.add_turn("Hello", "Hi")
        assert mem.turn_count == 1
        mem.add_turn("How?", "Fine")
        assert mem.turn_count == 2

    def test_add_turn_active_turns(self):
        mem = ConversationMemory()
        mem.add_turn("Hello", "Hi")
        mem.add_turn("How?", "Fine")
        assert mem.active_turns == 2

    def test_turn_ids_sequential(self):
        mem = ConversationMemory()
        t1 = mem.add_turn("a", "b")
        t2 = mem.add_turn("c", "d")
        t3 = mem.add_turn("e", "f")
        assert t1.turn_id == 1
        assert t2.turn_id == 2
        assert t3.turn_id == 3

    def test_add_turn_with_tool_calls(self):
        mem = ConversationMemory()
        turn = mem.add_turn(
            "Search scenes",
            "Found 3",
            tool_calls=[{"name": "scene_search"}],
            tool_results=[{"data": "scene1"}],
        )
        assert len(turn.tool_calls) == 1
        assert len(turn.tool_results) == 1

    def test_add_turn_with_context_type(self):
        mem = ConversationMemory()
        turn = mem.add_turn("msg", "resp", context_type="film_writing")
        assert turn.context_type == "film_writing"

    def test_add_turn_inherits_current_context(self):
        mem = ConversationMemory()
        mem.set_context("brainstorming")
        turn = mem.add_turn("msg", "resp")
        assert turn.context_type == "brainstorming"

    def test_add_turn_explicit_context_overrides(self):
        mem = ConversationMemory()
        mem.set_context("brainstorming")
        turn = mem.add_turn("msg", "resp", context_type="film_writing")
        assert turn.context_type == "film_writing"

    def test_add_turn_with_metadata(self):
        mem = ConversationMemory()
        turn = mem.add_turn("msg", "resp", metadata={"key": "val"})
        assert turn.metadata == {"key": "val"}

    def test_total_tokens_increases(self):
        mem = ConversationMemory()
        mem.add_turn("Hello there friend", "I am doing well thank you")
        assert mem.total_tokens > 0


# =========================================================================
# ConversationMemory - Sliding Window
# =========================================================================


class TestSlidingWindow:
    """Test max_turns sliding window behavior"""

    def test_max_turns_deque(self):
        mem = ConversationMemory(max_turns=3)
        mem.add_turn("a", "1")
        mem.add_turn("b", "2")
        mem.add_turn("c", "3")
        mem.add_turn("d", "4")
        # deque maxlen=3, oldest dropped
        assert mem.active_turns == 3
        assert mem.turn_count == 4  # counter still goes up

    def test_oldest_turn_dropped(self):
        mem = ConversationMemory(max_turns=2)
        mem.add_turn("first", "one")
        mem.add_turn("second", "two")
        mem.add_turn("third", "three")
        turns = mem.get_last_n_turns(5)
        messages = [t.user_message for t in turns]
        assert "first" not in messages
        assert "second" in messages
        assert "third" in messages


# =========================================================================
# ConversationMemory - Retrieval
# =========================================================================


class TestRetrieval:
    """Test turn retrieval methods"""

    def test_get_last_n_turns(self):
        mem = ConversationMemory()
        for i in range(5):
            mem.add_turn(f"msg{i}", f"resp{i}")
        turns = mem.get_last_n_turns(3)
        assert len(turns) == 3
        assert turns[0].user_message == "msg2"
        assert turns[2].user_message == "msg4"

    def test_get_last_n_more_than_available(self):
        mem = ConversationMemory()
        mem.add_turn("only", "one")
        turns = mem.get_last_n_turns(10)
        assert len(turns) == 1

    def test_get_turn_by_id(self):
        mem = ConversationMemory()
        mem.add_turn("first", "one")
        t2 = mem.add_turn("second", "two")
        mem.add_turn("third", "three")
        found = mem.get_turn(t2.turn_id)
        assert found is not None
        assert found.user_message == "second"

    def test_get_turn_not_found(self):
        mem = ConversationMemory()
        mem.add_turn("only", "one")
        assert mem.get_turn(999) is None

    def test_get_turn_dropped_from_window(self):
        mem = ConversationMemory(max_turns=2)
        mem.add_turn("first", "one")
        mem.add_turn("second", "two")
        mem.add_turn("third", "three")
        # Turn 1 was dropped from deque
        assert mem.get_turn(1) is None


# =========================================================================
# ConversationMemory - Context Messages
# =========================================================================


class TestContextMessages:
    """Test get_context_messages for LLM context"""

    def test_empty_returns_empty(self):
        mem = ConversationMemory()
        messages = mem.get_context_messages()
        assert messages == []

    def test_system_prompt_added(self):
        mem = ConversationMemory()
        messages = mem.get_context_messages(system_prompt="You are Friday")
        assert len(messages) == 1
        assert messages[0].role == "system"
        assert messages[0].content == "You are Friday"

    def test_turns_as_messages(self):
        mem = ConversationMemory()
        mem.add_turn("Hello", "Hi there")
        messages = mem.get_context_messages()
        assert len(messages) == 2
        # User message comes before assistant response
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_system_prompt_first(self):
        mem = ConversationMemory()
        mem.add_turn("Hello", "Hi")
        messages = mem.get_context_messages(system_prompt="System")
        assert messages[0].role == "system"
        assert messages[0].content == "System"
        # User before assistant within the turn
        assert messages[1].role == "user"
        assert messages[2].role == "assistant"

    def test_multiple_turns_chronological(self):
        mem = ConversationMemory()
        mem.add_turn("first", "one")
        mem.add_turn("second", "two")
        messages = mem.get_context_messages()
        user_msgs = [m.content for m in messages if m.role == "user"]
        assert user_msgs == ["first", "second"]

    def test_max_tokens_limits_output(self):
        mem = ConversationMemory(max_tokens=10000)
        # Add many turns
        for i in range(20):
            mem.add_turn(f"message {i} " * 50, f"response {i} " * 50)
        messages = mem.get_context_messages(max_tokens=100)
        # Should have fewer messages than total turns
        user_msgs = [m for m in messages if m.role == "user"]
        assert len(user_msgs) < 20

    def test_summary_included(self):
        mem = ConversationMemory(max_turns=20, summarize_threshold=3)
        # Add enough turns to trigger summarization
        for i in range(5):
            mem.add_turn(f"discussing topic {i}", f"response about topic {i}")
        messages = mem.get_context_messages(include_summary=True)
        # Check if summary appears as system message
        system_msgs = [m for m in messages if m.role == "system"]
        if mem._summary:
            assert any("summary" in m.content.lower() for m in system_msgs)

    def test_summary_excluded(self):
        mem = ConversationMemory(max_turns=20, summarize_threshold=3)
        for i in range(5):
            mem.add_turn(f"discussing topic {i}", f"response about topic {i}")
        messages = mem.get_context_messages(include_summary=False)
        system_msgs = [m for m in messages if m.role == "system"]
        assert not any("summary" in m.content.lower() for m in system_msgs)


# =========================================================================
# ConversationMemory - Summarization
# =========================================================================


class TestSummarization:
    """Test automatic summarization"""

    def test_summarize_triggered(self):
        mem = ConversationMemory(max_turns=20, summarize_threshold=4)
        for i in range(5):
            mem.add_turn(f"talking about screenplay writing details {i}", f"resp {i}")
        assert mem._summary is not None

    def test_summarize_not_triggered_below_threshold(self):
        mem = ConversationMemory(max_turns=20, summarize_threshold=10)
        for i in range(3):
            mem.add_turn(f"msg {i}", f"resp {i}")
        assert mem._summary is None

    def test_summarized_turn_count_tracked(self):
        mem = ConversationMemory(max_turns=20, summarize_threshold=4)
        for i in range(5):
            mem.add_turn(f"talking about screenplay details {i}", f"resp {i}")
        assert mem._summarized_turn_count > 0

    def test_summary_contains_topics(self):
        mem = ConversationMemory(max_turns=20, summarize_threshold=4)
        for i in range(5):
            mem.add_turn(f"discussing screenplay writing {i}", f"resp {i}")
        if mem._summary:
            assert "Discussed:" in mem._summary


# =========================================================================
# ConversationMemory - Context & Clear
# =========================================================================


class TestContextAndClear:
    """Test context setting and clearing"""

    def test_set_context(self):
        mem = ConversationMemory()
        mem.set_context("film_writing")
        assert mem.current_context == "film_writing"

    def test_clear_resets_all(self):
        mem = ConversationMemory()
        mem.add_turn("msg", "resp")
        mem.add_turn("msg2", "resp2")
        mem._summary = "some summary"
        mem._summarized_turn_count = 5
        mem.clear()
        assert mem.turn_count == 0
        assert mem.active_turns == 0
        assert mem._summary is None
        assert mem._summarized_turn_count == 0

    def test_clear_keeps_config(self):
        mem = ConversationMemory(max_turns=10, max_tokens=2000)
        mem.add_turn("msg", "resp")
        mem.clear()
        assert mem.max_turns == 10
        assert mem.max_tokens == 2000


# =========================================================================
# ConversationMemory - Serialization
# =========================================================================


class TestSerialization:
    """Test to_dict and from_dict"""

    def test_to_dict_structure(self):
        mem = ConversationMemory()
        mem.session_id = "sess-001"
        mem.add_turn("Hello", "Hi")
        d = mem.to_dict()
        assert "session_id" in d
        assert "started_at" in d
        assert "turn_count" in d
        assert "current_context" in d
        assert "summary" in d
        assert "turns" in d

    def test_to_dict_values(self):
        mem = ConversationMemory()
        mem.session_id = "sess-001"
        mem.add_turn("Hello", "Hi", context_type="greeting")
        d = mem.to_dict()
        assert d["session_id"] == "sess-001"
        assert d["turn_count"] == 1
        assert len(d["turns"]) == 1
        assert d["turns"][0]["user_message"] == "Hello"

    def test_from_dict_restores(self):
        mem = ConversationMemory()
        mem.session_id = "sess-001"
        mem.set_context("film_writing")
        mem.add_turn("msg1", "resp1")
        mem.add_turn("msg2", "resp2")
        d = mem.to_dict()

        restored = ConversationMemory.from_dict(d)
        assert restored.session_id == "sess-001"
        assert restored.current_context == "film_writing"
        assert restored.turn_count == 2
        assert restored.active_turns == 2

    def test_from_dict_restores_turns(self):
        mem = ConversationMemory()
        mem.add_turn("Hello", "Hi")
        d = mem.to_dict()
        restored = ConversationMemory.from_dict(d)
        turns = restored.get_last_n_turns(5)
        assert len(turns) == 1
        assert turns[0].user_message == "Hello"
        assert turns[0].assistant_response == "Hi"

    def test_from_dict_restores_summary(self):
        mem = ConversationMemory()
        mem._summary = "Previous topics discussed"
        mem._summarized_turn_count = 3
        d = mem.to_dict()
        restored = ConversationMemory.from_dict(d)
        assert restored._summary == "Previous topics discussed"
        assert restored._summarized_turn_count == 3

    def test_from_dict_empty(self):
        restored = ConversationMemory.from_dict({})
        assert restored.turn_count == 0
        assert restored.active_turns == 0
        assert restored.current_context == "general"

    def test_round_trip(self):
        mem = ConversationMemory()
        mem.session_id = "test-session"
        mem.add_turn("Boss, show scenes", "Found 5 scenes", context_type="film_writing")
        mem.add_turn("Update scene 3", "Done", tool_calls=[{"name": "scene_update"}])
        d = mem.to_dict()
        restored = ConversationMemory.from_dict(d)
        assert restored.turn_count == mem.turn_count
        assert restored.session_id == mem.session_id
        turns = restored.get_last_n_turns(5)
        assert turns[0].user_message == "Boss, show scenes"
        assert turns[1].tool_calls == [{"name": "scene_update"}]


# =========================================================================
# ConversationMemory - Properties
# =========================================================================


class TestProperties:
    """Test computed properties"""

    def test_turn_count_vs_active_turns(self):
        mem = ConversationMemory(max_turns=3)
        for i in range(5):
            mem.add_turn(f"msg{i}", f"resp{i}")
        assert mem.turn_count == 5  # total added
        assert mem.active_turns == 3  # deque maxlen

    def test_total_tokens(self):
        mem = ConversationMemory()
        mem.add_turn("a" * 100, "b" * 100)  # 200 chars / 4 = 50 tokens
        assert mem.total_tokens == 50


# =========================================================================
# Edge Cases
# =========================================================================


class TestEdgeCases:
    """Test edge cases"""

    def test_empty_messages(self):
        mem = ConversationMemory()
        turn = mem.add_turn("", "")
        assert turn.token_estimate == 0

    def test_very_long_message(self):
        mem = ConversationMemory()
        long_msg = "word " * 10000
        turn = mem.add_turn(long_msg, "short")
        assert turn.token_estimate > 1000

    def test_unicode_messages(self):
        mem = ConversationMemory()
        turn = mem.add_turn("Boss, బాగుంది", "నేను ready")
        assert turn.user_message == "Boss, బాగుంది"
        messages = turn.to_messages()
        assert messages[0].content == "Boss, బాగుంది"

    def test_session_id_set(self):
        mem = ConversationMemory()
        mem.session_id = "custom-session-id"
        d = mem.to_dict()
        assert d["session_id"] == "custom-session-id"

    def test_single_turn_max(self):
        mem = ConversationMemory(max_turns=1)
        mem.add_turn("first", "one")
        mem.add_turn("second", "two")
        assert mem.active_turns == 1
        turns = mem.get_last_n_turns(5)
        assert turns[0].user_message == "second"
