"""
Tests for WorkingMemoryAdapter
================================

Tests the adapter layer between WorkingMemory and the orchestrator's
ConversationMemory interface.

Run with: pytest tests/test_working_memory_adapter.py -v
"""

import sys
import time
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from memory.config import WorkingMemoryConfig
from memory.layers.working import WorkingMemory, ConversationTurn
from orchestrator.memory.working_memory_adapter import (
    WorkingMemoryAdapter,
    _AdaptedTurn,
)


# =========================================================================
# Helper
# =========================================================================


def _make_config(**overrides) -> WorkingMemoryConfig:
    defaults = {"max_turns": 10, "max_tokens": 4000, "max_attention_items": 7}
    defaults.update(overrides)
    return WorkingMemoryConfig(**defaults)


def _make_adapter(**overrides) -> WorkingMemoryAdapter:
    config = _make_config(**overrides)
    return WorkingMemoryAdapter(config=config)


# =========================================================================
# _AdaptedTurn
# =========================================================================


class TestAdaptedTurn:
    """Test the _AdaptedTurn wrapper"""

    def _make_wm_turn(self):
        return ConversationTurn(
            user_message="Hello Boss",
            assistant_response="Boss, baagunnanu!",
            context_type="writers_room",
        )

    def test_turn_id(self):
        wm_turn = self._make_wm_turn()
        adapted = _AdaptedTurn(1, wm_turn)
        assert adapted.turn_id == 1

    def test_user_message(self):
        wm_turn = self._make_wm_turn()
        adapted = _AdaptedTurn(1, wm_turn)
        assert adapted.user_message == "Hello Boss"

    def test_assistant_response(self):
        wm_turn = self._make_wm_turn()
        adapted = _AdaptedTurn(1, wm_turn)
        assert adapted.assistant_response == "Boss, baagunnanu!"

    def test_context_type(self):
        wm_turn = self._make_wm_turn()
        adapted = _AdaptedTurn(1, wm_turn)
        assert adapted.context_type == "writers_room"

    def test_confidence(self):
        wm_turn = self._make_wm_turn()
        adapted = _AdaptedTurn(1, wm_turn)
        assert adapted.confidence == 1.0

    def test_is_quarantined(self):
        wm_turn = self._make_wm_turn()
        adapted = _AdaptedTurn(1, wm_turn)
        assert adapted.is_quarantined is False

    def test_token_estimate(self):
        wm_turn = self._make_wm_turn()
        adapted = _AdaptedTurn(1, wm_turn)
        assert adapted.token_estimate > 0
        assert adapted.token_estimate == wm_turn.total_tokens()

    def test_timestamp(self):
        wm_turn = self._make_wm_turn()
        adapted = _AdaptedTurn(1, wm_turn)
        assert adapted.timestamp == wm_turn.timestamp

    def test_tool_calls_empty(self):
        wm_turn = self._make_wm_turn()
        adapted = _AdaptedTurn(1, wm_turn)
        assert adapted.tool_calls == []

    def test_tool_results_empty(self):
        wm_turn = self._make_wm_turn()
        adapted = _AdaptedTurn(1, wm_turn)
        assert adapted.tool_results == []


# =========================================================================
# WorkingMemoryAdapter - Initialization
# =========================================================================


class TestAdapterInit:
    """Test adapter initialization"""

    def test_default_init(self):
        adapter = WorkingMemoryAdapter()
        assert adapter.session_id is None
        assert adapter.turn_count == 0
        assert adapter.active_turns == 0

    def test_init_with_config(self):
        config = _make_config(max_tokens=8000)
        adapter = WorkingMemoryAdapter(config=config)
        assert adapter.working_memory.config.max_tokens == 8000

    def test_init_with_existing_wm(self):
        wm = WorkingMemory(config=_make_config())
        wm.add_turn("q1", "a1")
        adapter = WorkingMemoryAdapter(working_memory=wm)
        assert adapter.active_turns == 1

    def test_started_at_set(self):
        adapter = _make_adapter()
        assert adapter.started_at > 0
        assert adapter.started_at <= time.time()


# =========================================================================
# WorkingMemoryAdapter - ConversationMemory Interface
# =========================================================================


class TestAdapterConversationInterface:
    """Test ConversationMemory-compatible methods"""

    def test_add_turn_returns_adapted(self):
        adapter = _make_adapter()
        turn = adapter.add_turn("Hello", "Hi there!")
        assert isinstance(turn, _AdaptedTurn)
        assert turn.turn_id == 1

    def test_add_turn_increments_counter(self):
        adapter = _make_adapter()
        adapter.add_turn("q1", "a1")
        adapter.add_turn("q2", "a2")
        assert adapter.turn_count == 2

    def test_add_turn_increments_active_turns(self):
        adapter = _make_adapter()
        adapter.add_turn("q1", "a1")
        assert adapter.active_turns == 1

    def test_add_turn_increments_tokens(self):
        adapter = _make_adapter()
        adapter.add_turn("Hello", "Hi")
        assert adapter.total_tokens > 0

    def test_add_turn_with_tool_calls(self):
        adapter = _make_adapter()
        tools = [{"name": "scene_get", "args": {"scene_number": 1}}]
        results = [{"data": {"title": "Opening"}}]
        turn = adapter.add_turn(
            "get scene 1", "Here it is", tool_calls=tools, tool_results=results
        )
        assert len(turn.tool_calls) == 1
        assert len(turn.tool_results) == 1

    def test_add_turn_with_context_type(self):
        adapter = _make_adapter()
        turn = adapter.add_turn("q", "a", context_type="kitchen")
        assert turn.context_type == "kitchen"

    def test_current_context(self):
        adapter = _make_adapter()
        adapter.set_context("writers_room")
        assert adapter.current_context == "writers_room"

    def test_get_last_n_turns(self):
        adapter = _make_adapter()
        adapter.add_turn("q1", "a1")
        adapter.add_turn("q2", "a2")
        adapter.add_turn("q3", "a3")
        turns = adapter.get_last_n_turns(2)
        assert len(turns) == 2
        assert all(isinstance(t, _AdaptedTurn) for t in turns)

    def test_get_last_n_turns_has_turn_ids(self):
        adapter = _make_adapter()
        adapter.add_turn("q1", "a1")
        adapter.add_turn("q2", "a2")
        turns = adapter.get_last_n_turns(2)
        # Turn IDs should be sequential
        assert turns[0].turn_id < turns[1].turn_id

    def test_clear(self):
        adapter = _make_adapter()
        adapter.add_turn("q1", "a1")
        adapter.add_turn("q2", "a2")
        adapter.clear()
        assert adapter.turn_count == 0
        assert adapter.active_turns == 0
        assert adapter.total_tokens == 0


# =========================================================================
# WorkingMemoryAdapter - Context Messages (ChatML)
# =========================================================================


class TestAdapterContextMessages:
    """Test get_context_messages for LLM context building"""

    def test_empty_returns_empty(self):
        adapter = _make_adapter()
        messages = adapter.get_context_messages()
        assert messages == []

    def test_system_prompt_added(self):
        adapter = _make_adapter()
        adapter.add_turn("q", "a")
        messages = adapter.get_context_messages(system_prompt="You are Friday.")
        assert messages[0].role == "system"
        assert messages[0].content == "You are Friday."

    def test_turn_produces_user_assistant_pair(self):
        adapter = _make_adapter()
        adapter.add_turn("Hello", "Hi there!")
        messages = adapter.get_context_messages()
        roles = [m.role for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_user_message_content(self):
        adapter = _make_adapter()
        adapter.add_turn("Hello Boss", "Baagunnanu")
        messages = adapter.get_context_messages()
        user_msgs = [m for m in messages if m.role == "user"]
        assert any(m.content == "Hello Boss" for m in user_msgs)

    def test_assistant_message_content(self):
        adapter = _make_adapter()
        adapter.add_turn("Hello", "Hi there!")
        messages = adapter.get_context_messages()
        asst_msgs = [m for m in messages if m.role == "assistant"]
        assert any(m.content == "Hi there!" for m in asst_msgs)

    def test_multiple_turns_chronological(self):
        adapter = _make_adapter()
        adapter.add_turn("first", "one")
        adapter.add_turn("second", "two")
        messages = adapter.get_context_messages()
        user_msgs = [m for m in messages if m.role == "user"]
        assert user_msgs[0].content == "first"
        assert user_msgs[1].content == "second"

    def test_max_tokens_limits_output(self):
        adapter = _make_adapter(max_tokens=10000)
        for i in range(20):
            adapter.add_turn(f"Question {i} " * 10, f"Answer {i} " * 10)
        messages = adapter.get_context_messages(max_tokens=100)
        # Should have fewer messages due to token budget
        assert len(messages) < 40

    def test_compressed_history_as_system_message(self):
        adapter = _make_adapter(max_turns=3, max_tokens=50000)
        # Add more turns than max to trigger compression
        for i in range(6):
            adapter.add_turn(f"question {i}", f"answer {i}")
        messages = adapter.get_context_messages(include_summary=True)
        system_msgs = [m for m in messages if m.role == "system"]
        # If compression happened, there should be a prior context system message
        if adapter.working_memory._compressed_history:
            assert any("Prior context" in (m.content or "") for m in system_msgs)

    def test_tool_calls_in_messages(self):
        adapter = _make_adapter()
        adapter.add_turn(
            "search scenes",
            "Found 3 scenes",
            tool_calls=[{"name": "scene_search", "args": {"query": "test"}}],
            tool_results=[
                {
                    "data": [{"title": "Scene 1"}],
                    "tool_call_id": "tc1",
                    "name": "scene_search",
                }
            ],
        )
        messages = adapter.get_context_messages()
        # Should have: user, assistant (tool_calls), tool (result), assistant (response)
        roles = [m.role for m in messages]
        assert "tool" in roles


# =========================================================================
# WorkingMemoryAdapter - WorkingMemory Features
# =========================================================================


class TestAdapterWMFeatures:
    """Test WorkingMemory-specific features exposed through adapter"""

    def test_capacity_zone(self):
        adapter = _make_adapter(max_tokens=10000)
        assert adapter.capacity_zone == "normal"

    def test_capacity_percentage(self):
        adapter = _make_adapter(max_tokens=10000)
        assert adapter.capacity_percentage == 0.0

    def test_tokens_available(self):
        adapter = _make_adapter(max_tokens=10000)
        assert adapter.tokens_available == 10000

    def test_update_attention(self):
        adapter = _make_adapter()
        adapter.update_attention("screenplay", 0.9)
        assert adapter.working_memory.is_attending_to("screenplay")

    def test_set_project(self):
        adapter = _make_adapter()
        adapter.set_project("gusagusalu")
        assert adapter.working_memory.current_project == "gusagusalu"

    def test_set_language_mode(self):
        adapter = _make_adapter()
        adapter.set_language_mode("te")
        assert adapter.working_memory.language_mode == "te"

    def test_set_emotional_context(self):
        adapter = _make_adapter()
        adapter.set_emotional_context("excited")
        assert adapter.working_memory.emotional_context == "excited"

    def test_set_active_task(self):
        adapter = _make_adapter()
        adapter.set_active_task("revise scene 5")
        assert adapter.working_memory.active_task == "revise scene 5"

    def test_working_memory_direct_access(self):
        adapter = _make_adapter()
        assert isinstance(adapter.working_memory, WorkingMemory)


# =========================================================================
# WorkingMemoryAdapter - Health & Stats
# =========================================================================


class TestAdapterHealthStats:
    """Test health status and context stats"""

    def test_health_status_includes_session(self):
        adapter = _make_adapter()
        adapter.session_id = "test-session"
        health = adapter.get_health_status()
        assert health["session_id"] == "test-session"
        assert "session_turn_count" in health

    def test_context_stats_includes_session(self):
        adapter = _make_adapter()
        adapter.session_id = "test-session"
        stats = adapter.get_context_stats()
        assert stats["session_id"] == "test-session"
        assert "session_turn_count" in stats

    def test_health_status_turn_count(self):
        adapter = _make_adapter()
        adapter.add_turn("q1", "a1")
        adapter.add_turn("q2", "a2")
        health = adapter.get_health_status()
        assert health["session_turn_count"] == 2


# =========================================================================
# WorkingMemoryAdapter - Serialization
# =========================================================================


class TestAdapterSerialization:
    """Test to_dict / from_dict round-trip"""

    def test_to_dict_structure(self):
        adapter = _make_adapter()
        adapter.session_id = "sess-001"
        adapter.add_turn("q", "a")
        data = adapter.to_dict()
        assert "session_id" in data
        assert "started_at" in data
        assert "turn_counter" in data
        assert "working_memory" in data

    def test_to_dict_values(self):
        adapter = _make_adapter()
        adapter.session_id = "sess-001"
        adapter.add_turn("q1", "a1")
        data = adapter.to_dict()
        assert data["session_id"] == "sess-001"
        assert data["turn_counter"] == 1

    def test_from_dict_restores_session(self):
        adapter = _make_adapter()
        adapter.session_id = "sess-001"
        adapter.add_turn("q1", "a1")
        adapter.add_turn("q2", "a2")
        data = adapter.to_dict()

        restored = WorkingMemoryAdapter.from_dict(data, config=_make_config())
        assert restored.session_id == "sess-001"
        assert restored.turn_count == 2

    def test_from_dict_restores_turns(self):
        adapter = _make_adapter()
        adapter.add_turn("hello", "hi")
        data = adapter.to_dict()

        restored = WorkingMemoryAdapter.from_dict(data, config=_make_config())
        assert restored.active_turns == 1

    def test_from_dict_empty(self):
        restored = WorkingMemoryAdapter.from_dict({}, config=_make_config())
        assert restored.session_id is None
        assert restored.turn_count == 0

    def test_round_trip_preserves_started_at(self):
        adapter = _make_adapter()
        original_time = adapter.started_at
        data = adapter.to_dict()

        restored = WorkingMemoryAdapter.from_dict(data, config=_make_config())
        assert restored.started_at == original_time


# =========================================================================
# WorkingMemoryAdapter - Repr
# =========================================================================


class TestAdapterRepr:
    """Test __repr__ output"""

    def test_repr_contains_session(self):
        adapter = _make_adapter()
        adapter.session_id = "test"
        repr_str = repr(adapter)
        assert "WorkingMemoryAdapter(" in repr_str
        assert "session=test" in repr_str

    def test_repr_contains_turns(self):
        adapter = _make_adapter()
        adapter.add_turn("q", "a")
        repr_str = repr(adapter)
        assert "turns=1" in repr_str
