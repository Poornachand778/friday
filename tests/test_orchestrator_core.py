"""
Tests for Friday AI Orchestrator Core
======================================

Comprehensive tests for orchestrator/core.py - the central coordinator.

All external dependencies are mocked since this module is the integration hub.

Run with: pytest tests/test_orchestrator_core.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock, call
from dataclasses import dataclass, field
from typing import List, Dict, Any

from orchestrator.core import (
    FridayOrchestrator,
    OrchestratorResponse,
    get_orchestrator,
    initialize_orchestrator,
)
from orchestrator.context.contexts import ContextType
from orchestrator.inference.router import TaskType, TaskComplexity


# ---------------------------------------------------------------------------
# Helper factories for mock objects
# ---------------------------------------------------------------------------


def _make_config_mock(router_enabled=False):
    """Create a fully configured OrchestratorConfig mock."""
    config = MagicMock()

    # LLM config
    config.llm = MagicMock()

    # Router config
    config.router = MagicMock()
    config.router.enabled = router_enabled
    config.router.provider = "zhipu"
    config.router.model_name = "glm-4.7-flash"

    # Memory config
    config.memory = MagicMock()
    config.memory.max_history_turns = 20
    config.memory.max_context_tokens = 6000

    # Context config
    config.context = MagicMock()
    config.context.default_context = "writers_room"

    # System prompt
    config.system_prompt_base = "You are Friday."

    return config


def _make_memory_config_mock():
    """Create a MemorySystemConfig mock with a working sub-config."""
    mem_cfg = MagicMock()
    mem_cfg.working = MagicMock()
    mem_cfg.working.max_turns = 10
    mem_cfg.working.max_tokens = 4000
    return mem_cfg


def _make_chat_response(content="Boss, here you go.", has_tools=False, tool_calls=None):
    """Create a mock ChatResponse."""
    resp = MagicMock()
    resp.content = content
    resp.has_tool_calls = has_tools
    resp.tool_calls = tool_calls or []
    resp.usage = {"prompt_tokens": 100, "completion_tokens": 50}
    return resp


def _make_routing_decision(
    confidence=0.9,
    primary_context="general",
    tools=None,
    agent_mode=False,
    expected_turns=1,
):
    """Create a mock RouterDecision."""
    decision = MagicMock()
    decision.task_type = MagicMock()
    decision.task_type.value = "conversation"
    decision.complexity = MagicMock()
    decision.complexity.value = "simple"
    decision.primary_context = primary_context
    decision.suggested_tools = tools or []
    decision.confidence = confidence
    decision.agent_mode = agent_mode
    decision.expected_turns = expected_turns
    return decision


def _make_built_context(messages=None, tools=None):
    """Create a mock BuiltContext."""
    ctx = MagicMock()
    ctx.messages = messages or [
        MagicMock(content="system msg"),
        MagicMock(content="user msg"),
    ]
    ctx.tools = tools or []
    ctx.token_estimate = 200
    ctx.ltm_count = 0
    return ctx


def _make_adapted_turn(turn_id=1):
    """Create a mock _AdaptedTurn returned from add_turn."""
    turn = MagicMock()
    turn.turn_id = turn_id
    turn.user_message = "Hello Boss"
    turn.assistant_response = "Hey Boss!"
    return turn


def _make_tool_call(call_id="call_1", name="scene_search", arguments=None):
    """Create a mock ToolCall object."""
    tc = MagicMock()
    tc.id = call_id
    tc.name = name
    tc.arguments = arguments or {"query": "test"}
    return tc


# ---------------------------------------------------------------------------
# Patch paths (all relative to orchestrator.core module)
# ---------------------------------------------------------------------------

_P = "orchestrator.core"


# ---------------------------------------------------------------------------
# 1. TestOrchestratorResponse
# ---------------------------------------------------------------------------


class TestOrchestratorResponse:
    """Tests for the OrchestratorResponse dataclass."""

    def test_creation_with_all_fields(self):
        resp = OrchestratorResponse(
            content="Hello Boss",
            context_type=ContextType.GENERAL,
            tool_calls_made=[],
            tool_results=[],
            turn_id=1,
            processing_time_ms=42.5,
            tokens_used={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert resp.content == "Hello Boss"
        assert resp.context_type == ContextType.GENERAL
        assert resp.turn_id == 1

    def test_context_type_writers_room(self):
        resp = OrchestratorResponse(
            content="Scene loaded",
            context_type=ContextType.WRITERS_ROOM,
            tool_calls_made=[{"id": "c1", "type": "function"}],
            tool_results=[{"tool_call_id": "c1", "success": True}],
            turn_id=3,
            processing_time_ms=100.0,
            tokens_used={},
        )
        assert resp.context_type == ContextType.WRITERS_ROOM
        assert len(resp.tool_calls_made) == 1

    def test_tool_calls_and_results(self):
        calls = [
            {"id": "c1", "type": "function", "function": {"name": "scene_get"}},
            {"id": "c2", "type": "function", "function": {"name": "scene_search"}},
        ]
        results = [
            {"tool_call_id": "c1", "success": True, "data": "scene data"},
            {"tool_call_id": "c2", "success": False, "error": "not found"},
        ]
        resp = OrchestratorResponse(
            content="Done",
            context_type=ContextType.GENERAL,
            tool_calls_made=calls,
            tool_results=results,
            turn_id=5,
            processing_time_ms=250.0,
            tokens_used={"prompt_tokens": 200, "completion_tokens": 100},
        )
        assert len(resp.tool_calls_made) == 2
        assert len(resp.tool_results) == 2
        assert resp.tool_results[1]["success"] is False

    def test_tokens_used_dict(self):
        resp = OrchestratorResponse(
            content="ok",
            context_type=ContextType.KITCHEN,
            tool_calls_made=[],
            tool_results=[],
            turn_id=1,
            processing_time_ms=10.0,
            tokens_used={"prompt_tokens": 50, "completion_tokens": 20},
        )
        assert resp.tokens_used["prompt_tokens"] == 50
        assert resp.tokens_used["completion_tokens"] == 20

    def test_processing_time_stored(self):
        resp = OrchestratorResponse(
            content="fast",
            context_type=ContextType.STORYBOARD,
            tool_calls_made=[],
            tool_results=[],
            turn_id=7,
            processing_time_ms=1.23,
            tokens_used={},
        )
        assert resp.processing_time_ms == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# 2. TestOrchestratorInit
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    """Tests for FridayOrchestrator.__init__."""

    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_default_config(self, mock_get_config, mock_mem_cfg):
        orch = FridayOrchestrator()
        assert orch.config is mock_get_config.return_value
        mock_get_config.assert_called_once()

    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    def test_custom_config(self, mock_mem_cfg):
        custom = _make_config_mock()
        orch = FridayOrchestrator(config=custom)
        assert orch.config is custom

    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_not_initialized_initially(self, mock_cfg, mock_mem):
        orch = FridayOrchestrator()
        assert orch.is_initialized is False

    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_no_session_initially(self, mock_cfg, mock_mem):
        orch = FridayOrchestrator()
        assert orch.current_session is None
        assert orch._current_session_id is None

    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_default_context_is_general(self, mock_cfg, mock_mem):
        orch = FridayOrchestrator()
        assert orch.current_context == ContextType.GENERAL


# ---------------------------------------------------------------------------
# 3. TestOrchestratorInitialize
# ---------------------------------------------------------------------------


class TestOrchestratorInitialize:
    """Tests for FridayOrchestrator.initialize()."""

    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="default prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.GLMRouter")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock(router_enabled=True))
    async def test_initialize_sets_up_components(
        self,
        mock_cfg,
        mock_mem_cfg,
        mock_llm,
        mock_router_cls,
        mock_registry,
        mock_detector,
        mock_prompt,
        mock_builder,
        mock_adapter,
    ):
        orch = FridayOrchestrator()
        await orch.initialize()

        assert orch.is_initialized is True
        mock_llm.assert_called_once()
        mock_router_cls.assert_called_once()
        mock_registry.assert_called_once()
        mock_detector.assert_called_once()
        mock_builder.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="default prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock(router_enabled=True))
    async def test_double_init_is_noop(
        self,
        mock_cfg,
        mock_mem_cfg,
        mock_llm,
        mock_registry,
        mock_detector,
        mock_prompt,
        mock_builder,
        mock_adapter,
    ):
        orch = FridayOrchestrator()
        await orch.initialize()
        assert mock_llm.call_count == 1

        await orch.initialize()
        # LLMClient should not be called again
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="default prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock(router_enabled=False))
    async def test_router_disabled_path(
        self,
        mock_cfg,
        mock_mem_cfg,
        mock_llm,
        mock_registry,
        mock_detector,
        mock_prompt,
        mock_builder,
        mock_adapter,
    ):
        orch = FridayOrchestrator()
        await orch.initialize()
        # Router should NOT be instantiated
        assert orch._router is None

    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="default prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock(router_enabled=False))
    async def test_creates_default_session(
        self,
        mock_cfg,
        mock_mem_cfg,
        mock_llm,
        mock_registry,
        mock_detector,
        mock_prompt,
        mock_builder,
        mock_adapter,
    ):
        orch = FridayOrchestrator()
        await orch.initialize()
        assert len(orch._sessions) == 1
        assert orch._current_session_id is not None

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="default prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    async def test_uses_custom_system_prompt_from_config(
        self,
        mock_mem_cfg,
        mock_llm,
        mock_registry,
        mock_detector,
        mock_prompt,
        mock_builder,
        mock_adapter,
    ):
        config = _make_config_mock()
        config.system_prompt_base = "Custom Friday prompt"

        orch = FridayOrchestrator(config=config)
        await orch.initialize()
        # ContextBuilder should be called with the custom prompt
        mock_builder.assert_called_once()
        assert "Custom Friday prompt" in str(mock_builder.call_args)

    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="fallback prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    async def test_uses_default_prompt_when_config_empty(
        self,
        mock_mem_cfg,
        mock_llm,
        mock_registry,
        mock_detector,
        mock_prompt,
        mock_builder,
        mock_adapter,
    ):
        config = _make_config_mock()
        config.system_prompt_base = ""  # empty -> use default
        orch = FridayOrchestrator(config=config)
        await orch.initialize()
        mock_prompt.assert_called_once()


# ---------------------------------------------------------------------------
# 4. TestOrchestratorShutdown
# ---------------------------------------------------------------------------


class TestOrchestratorShutdown:
    """Tests for FridayOrchestrator.shutdown()."""

    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock(router_enabled=False))
    async def test_shutdown_closes_llm_client(
        self,
        mock_cfg,
        mock_mem_cfg,
        mock_llm_cls,
        mock_registry,
        mock_detector,
        mock_prompt,
        mock_builder,
        mock_adapter,
    ):
        mock_llm_instance = MagicMock()
        mock_llm_instance.close = AsyncMock()
        mock_llm_cls.return_value = mock_llm_instance

        orch = FridayOrchestrator()
        await orch.initialize()
        await orch.shutdown()

        mock_llm_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.GLMRouter")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock(router_enabled=True))
    async def test_shutdown_closes_router(
        self,
        mock_cfg,
        mock_mem_cfg,
        mock_llm_cls,
        mock_router_cls,
        mock_registry,
        mock_detector,
        mock_prompt,
        mock_builder,
        mock_adapter,
    ):
        mock_router_instance = MagicMock()
        mock_router_instance.close = AsyncMock()
        mock_router_cls.return_value = mock_router_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.close = AsyncMock()
        mock_llm_cls.return_value = mock_llm_instance

        orch = FridayOrchestrator()
        await orch.initialize()
        await orch.shutdown()

        mock_router_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock(router_enabled=False))
    async def test_shutdown_sets_initialized_false(
        self,
        mock_cfg,
        mock_mem_cfg,
        mock_llm_cls,
        mock_registry,
        mock_detector,
        mock_prompt,
        mock_builder,
        mock_adapter,
    ):
        mock_llm_cls.return_value.close = AsyncMock()

        orch = FridayOrchestrator()
        await orch.initialize()
        assert orch.is_initialized is True
        await orch.shutdown()
        assert orch.is_initialized is False

    @pytest.mark.asyncio
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    async def test_shutdown_safe_when_no_clients(self, mock_cfg, mock_mem_cfg):
        """Shutdown when never initialized should not raise."""
        orch = FridayOrchestrator()
        await orch.shutdown()  # Should not raise
        assert orch.is_initialized is False


# ---------------------------------------------------------------------------
# 5. TestSessionManagement
# ---------------------------------------------------------------------------


class TestSessionManagement:
    """Tests for session creation, switching, and listing."""

    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_create_session_auto_id(self, mock_cfg, mock_mem_cfg, mock_adapter):
        orch = FridayOrchestrator()
        sid = orch._create_session()
        assert sid is not None
        assert len(sid) == 8
        assert sid in orch._sessions
        assert orch._current_session_id == sid

    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_create_session_custom_id(self, mock_cfg, mock_mem_cfg, mock_adapter):
        orch = FridayOrchestrator()
        sid = orch._create_session(session_id="my-session")
        assert sid == "my-session"
        assert "my-session" in orch._sessions

    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_switch_session_existing(self, mock_cfg, mock_mem_cfg, mock_adapter):
        orch = FridayOrchestrator()
        orch._create_session(session_id="s1")
        orch._create_session(session_id="s2")
        assert orch._current_session_id == "s2"

        result = orch.switch_session("s1")
        assert result is True
        assert orch._current_session_id == "s1"

    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_switch_session_nonexistent(self, mock_cfg, mock_mem_cfg, mock_adapter):
        orch = FridayOrchestrator()
        orch._create_session(session_id="s1")
        result = orch.switch_session("nonexistent")
        assert result is False
        assert orch._current_session_id == "s1"

    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_current_session_returns_adapter(
        self, mock_cfg, mock_mem_cfg, mock_adapter
    ):
        orch = FridayOrchestrator()
        orch._create_session(session_id="s1")
        session = orch.current_session
        assert session is not None
        assert session is orch._sessions["s1"]

    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_current_session_none_when_no_session(self, mock_cfg, mock_mem_cfg):
        orch = FridayOrchestrator()
        assert orch.current_session is None

    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_multiple_sessions_stored(self, mock_cfg, mock_mem_cfg, mock_adapter):
        orch = FridayOrchestrator()
        orch._create_session(session_id="a")
        orch._create_session(session_id="b")
        orch._create_session(session_id="c")
        assert len(orch._sessions) == 3
        assert set(orch._sessions.keys()) == {"a", "b", "c"}

    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_create_session_applies_config_overrides(
        self, mock_cfg, mock_mem_cfg, mock_adapter
    ):
        """Verify orchestrator config overrides working memory config."""
        orch = FridayOrchestrator()
        orch._create_session(session_id="cfg-test")
        # The adapter should have been created with the wm_config
        mock_adapter.assert_called()


# ---------------------------------------------------------------------------
# 6. TestChatPipeline
# ---------------------------------------------------------------------------


class TestChatPipeline:
    """Tests for FridayOrchestrator.chat() - the full pipeline."""

    def _build_initialized_orchestrator(
        self,
        router_enabled=False,
        mock_llm=None,
        mock_router=None,
        mock_detector=None,
        mock_builder=None,
        mock_registry=None,
        mock_memory=None,
    ):
        """Build an orchestrator with all components pre-set (skip initialize)."""
        config = _make_config_mock(router_enabled=router_enabled)
        mem_cfg = _make_memory_config_mock()

        with patch(f"{_P}.get_memory_config", return_value=mem_cfg), patch(
            f"{_P}.get_config", return_value=config
        ):
            orch = FridayOrchestrator(config=config)

        orch._initialized = True
        orch._llm_client = mock_llm or MagicMock()
        orch._router = mock_router if router_enabled else None
        orch._context_detector = mock_detector or MagicMock()
        orch._context_builder = mock_builder or MagicMock()
        orch._tool_registry = mock_registry or MagicMock()

        # Set up a default session with a mock memory adapter
        memory = mock_memory or MagicMock()
        memory.turn_count = 0
        memory.add_turn = MagicMock(return_value=_make_adapted_turn())
        memory.set_context = MagicMock()

        orch._sessions = {"default": memory}
        orch._current_session_id = "default"

        return orch, config

    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    async def test_chat_auto_initializes(
        self,
        mock_cfg,
        mock_mem_cfg,
        mock_llm_cls,
        mock_registry,
        mock_detector_cls,
        mock_prompt,
        mock_builder_cls,
        mock_adapter_cls,
    ):
        """If not initialized, chat() should call initialize() first."""
        # Set up mocks for the chat pipeline to work
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())
        mock_llm_cls.return_value = mock_llm

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL
        mock_detector_cls.return_value = mock_detector

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()
        mock_builder_cls.return_value = mock_builder

        mock_adapter = MagicMock()
        mock_adapter.turn_count = 0
        mock_adapter.add_turn.return_value = _make_adapted_turn()
        mock_adapter.set_context = MagicMock()
        mock_adapter_cls.return_value = mock_adapter

        orch = FridayOrchestrator()
        assert orch.is_initialized is False

        resp = await orch.chat("Hello")
        assert orch.is_initialized is True
        assert isinstance(resp, OrchestratorResponse)

    @pytest.mark.asyncio
    async def test_chat_creates_new_session_if_needed(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL

        built_ctx = _make_built_context()
        mock_builder = MagicMock()
        mock_builder.build.return_value = built_ctx

        orch, _ = self._build_initialized_orchestrator(
            mock_llm=mock_llm,
            mock_detector=mock_detector,
            mock_builder=mock_builder,
        )

        # Provide a session_id that does not exist yet
        with patch(f"{_P}.WorkingMemoryAdapter") as mock_adapter_cls:
            new_mem = MagicMock()
            new_mem.turn_count = 0
            new_mem.add_turn.return_value = _make_adapted_turn()
            new_mem.set_context = MagicMock()
            mock_adapter_cls.return_value = new_mem

            resp = await orch.chat("Hello", session_id="brand-new")

        assert "brand-new" in orch._sessions
        assert isinstance(resp, OrchestratorResponse)

    @pytest.mark.asyncio
    async def test_chat_switches_to_existing_session(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch, _ = self._build_initialized_orchestrator(
            mock_llm=mock_llm,
            mock_detector=mock_detector,
            mock_builder=mock_builder,
        )

        # Add a second session
        other_memory = MagicMock()
        other_memory.turn_count = 0
        other_memory.add_turn.return_value = _make_adapted_turn()
        other_memory.set_context = MagicMock()
        orch._sessions["other"] = other_memory

        resp = await orch.chat("Hey", session_id="other")
        assert orch._current_session_id == "other"

    @pytest.mark.asyncio
    async def test_router_analysis_called_when_enabled(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_router = MagicMock()
        decision = _make_routing_decision(confidence=0.9)
        mock_router.analyze = AsyncMock(return_value=decision)

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch, config = self._build_initialized_orchestrator(
            router_enabled=True,
            mock_llm=mock_llm,
            mock_router=mock_router,
            mock_builder=mock_builder,
        )

        resp = await orch.chat("Show me scene 5")
        mock_router.analyze.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_router_failure_falls_back_gracefully(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_router = MagicMock()
        mock_router.analyze = AsyncMock(side_effect=Exception("Router timeout"))

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch, _ = self._build_initialized_orchestrator(
            router_enabled=True,
            mock_llm=mock_llm,
            mock_router=mock_router,
            mock_detector=mock_detector,
            mock_builder=mock_builder,
        )

        # Should not raise -- falls back to keyword detection
        resp = await orch.chat("Hello Boss")
        assert isinstance(resp, OrchestratorResponse)
        mock_detector.detect.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_detection_from_router_high_confidence(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        decision = _make_routing_decision(
            confidence=0.9, primary_context="writers_room"
        )
        mock_router = MagicMock()
        mock_router.analyze = AsyncMock(return_value=decision)

        mock_detector = MagicMock()

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch, _ = self._build_initialized_orchestrator(
            router_enabled=True,
            mock_llm=mock_llm,
            mock_router=mock_router,
            mock_detector=mock_detector,
            mock_builder=mock_builder,
        )
        orch._current_context = ContextType.GENERAL

        resp = await orch.chat("Show me scene 5")
        # High confidence router should set context without calling detector
        assert orch._current_context == ContextType.WRITERS_ROOM
        mock_detector.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_detection_fallback_low_confidence(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        decision = _make_routing_decision(confidence=0.5, primary_context="kitchen")
        mock_router = MagicMock()
        mock_router.analyze = AsyncMock(return_value=decision)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.KITCHEN

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch, _ = self._build_initialized_orchestrator(
            router_enabled=True,
            mock_llm=mock_llm,
            mock_router=mock_router,
            mock_detector=mock_detector,
            mock_builder=mock_builder,
        )

        await orch.chat("Let me cook something")
        # Low confidence -> should fall back to keyword detector
        mock_detector.detect.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_switch_logged(self):
        """When context changes, memory.set_context should be called."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.KITCHEN

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        mock_memory = MagicMock()
        mock_memory.turn_count = 0
        mock_memory.add_turn.return_value = _make_adapted_turn()
        mock_memory.set_context = MagicMock()

        orch, _ = self._build_initialized_orchestrator(
            mock_llm=mock_llm,
            mock_detector=mock_detector,
            mock_builder=mock_builder,
            mock_memory=mock_memory,
        )
        orch._current_context = ContextType.GENERAL

        await orch.chat("Time to cook")
        mock_memory.set_context.assert_called_once_with("kitchen")

    @pytest.mark.asyncio
    async def test_llm_response_returned(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value=_make_chat_response(content="Boss, scene 5 loaded.")
        )

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch, _ = self._build_initialized_orchestrator(
            mock_llm=mock_llm,
            mock_detector=mock_detector,
            mock_builder=mock_builder,
        )

        resp = await orch.chat("Show scene 5")
        assert resp.content == "Boss, scene 5 loaded."

    @pytest.mark.asyncio
    async def test_tool_calls_handled(self):
        """When LLM returns tool calls, _handle_tool_calls is invoked."""
        tc = _make_tool_call()
        first_response = _make_chat_response(
            content="", has_tools=True, tool_calls=[tc]
        )
        second_response = _make_chat_response(content="Done with tool.")

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=[first_response, second_response])

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL

        built_ctx = _make_built_context()
        mock_builder = MagicMock()
        mock_builder.build.return_value = built_ctx
        mock_builder.build_for_tool_response.return_value = [MagicMock()]

        mock_registry = MagicMock()
        tool_result = MagicMock()
        tool_result.success = True
        tool_result.data = {"result": "found"}
        tool_result.error = None
        mock_registry.async_execute = AsyncMock(return_value=tool_result)

        orch, _ = self._build_initialized_orchestrator(
            mock_llm=mock_llm,
            mock_detector=mock_detector,
            mock_builder=mock_builder,
            mock_registry=mock_registry,
        )

        resp = await orch.chat("Search for scene")
        assert resp.content == "Done with tool."
        assert len(resp.tool_calls_made) == 1
        assert resp.tool_results[0]["success"] is True

    @pytest.mark.asyncio
    async def test_memory_updated_after_chat(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response(content="Reply"))

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        mock_memory = MagicMock()
        mock_memory.turn_count = 0
        mock_memory.add_turn.return_value = _make_adapted_turn(turn_id=2)
        mock_memory.set_context = MagicMock()

        orch, _ = self._build_initialized_orchestrator(
            mock_llm=mock_llm,
            mock_detector=mock_detector,
            mock_builder=mock_builder,
            mock_memory=mock_memory,
        )

        resp = await orch.chat("Hey")
        mock_memory.add_turn.assert_called_once()
        call_kwargs = mock_memory.add_turn.call_args
        assert (
            call_kwargs[1]["user_message"] == "Hey"
            or call_kwargs.kwargs.get("user_message") == "Hey"
        )

    @pytest.mark.asyncio
    async def test_stream_response_path(self):
        """When stream=True, chat should return an async iterator."""
        mock_llm = MagicMock()

        async def _mock_stream(*args, **kwargs):
            for token in ["Hello", " Boss"]:
                yield token

        mock_llm.chat = AsyncMock(return_value=_mock_stream())

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        mock_memory = MagicMock()
        mock_memory.turn_count = 0
        mock_memory.add_turn.return_value = _make_adapted_turn()
        mock_memory.set_context = MagicMock()

        orch, _ = self._build_initialized_orchestrator(
            mock_llm=mock_llm,
            mock_detector=mock_detector,
            mock_builder=mock_builder,
            mock_memory=mock_memory,
        )

        result = await orch.chat("Hello", stream=True)
        # The result should be an async iterator (the _stream_response generator)
        assert hasattr(result, "__aiter__")


# ---------------------------------------------------------------------------
# 7. TestToolExecution
# ---------------------------------------------------------------------------


class TestToolExecution:
    """Tests for FridayOrchestrator.execute_tool()."""

    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    async def test_execute_tool_auto_initializes(
        self,
        mock_cfg,
        mock_mem_cfg,
        mock_llm,
        mock_reg_fn,
        mock_detector,
        mock_prompt,
        mock_builder,
        mock_adapter,
    ):
        mock_registry = MagicMock()
        tool_result = MagicMock()
        tool_result.success = True
        tool_result.data = "executed"
        mock_registry.async_execute = AsyncMock(return_value=tool_result)
        mock_reg_fn.return_value = mock_registry

        orch = FridayOrchestrator()
        assert orch.is_initialized is False

        result = await orch.execute_tool("scene_get", {"scene_id": 5})
        assert orch.is_initialized is True
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_tool_delegates_to_registry(self):
        config = _make_config_mock()
        mem_cfg = _make_memory_config_mock()

        with patch(f"{_P}.get_memory_config", return_value=mem_cfg), patch(
            f"{_P}.get_config", return_value=config
        ):
            orch = FridayOrchestrator(config=config)

        mock_registry = MagicMock()
        tool_result = MagicMock()
        tool_result.success = True
        tool_result.data = {"scene": "data"}
        tool_result.error = None
        mock_registry.async_execute = AsyncMock(return_value=tool_result)

        orch._initialized = True
        orch._tool_registry = mock_registry

        result = await orch.execute_tool("scene_get", {"scene_id": 1})
        mock_registry.async_execute.assert_awaited_once_with(
            "scene_get", {"scene_id": 1}
        )
        assert result.data == {"scene": "data"}

    @pytest.mark.asyncio
    async def test_execute_tool_passes_arguments(self):
        config = _make_config_mock()
        mem_cfg = _make_memory_config_mock()

        with patch(f"{_P}.get_memory_config", return_value=mem_cfg), patch(
            f"{_P}.get_config", return_value=config
        ):
            orch = FridayOrchestrator(config=config)

        mock_registry = MagicMock()
        tool_result = MagicMock()
        tool_result.success = True
        mock_registry.async_execute = AsyncMock(return_value=tool_result)

        orch._initialized = True
        orch._tool_registry = mock_registry

        args = {"query": "scene with rain", "limit": 5}
        await orch.execute_tool("scene_search", args)
        mock_registry.async_execute.assert_awaited_once_with("scene_search", args)

    @pytest.mark.asyncio
    async def test_execute_tool_returns_failure(self):
        config = _make_config_mock()
        mem_cfg = _make_memory_config_mock()

        with patch(f"{_P}.get_memory_config", return_value=mem_cfg), patch(
            f"{_P}.get_config", return_value=config
        ):
            orch = FridayOrchestrator(config=config)

        mock_registry = MagicMock()
        fail_result = MagicMock()
        fail_result.success = False
        fail_result.data = None
        fail_result.error = "Tool not found"
        mock_registry.async_execute = AsyncMock(return_value=fail_result)

        orch._initialized = True
        orch._tool_registry = mock_registry

        result = await orch.execute_tool("nonexistent", {})
        assert result.success is False
        assert result.error == "Tool not found"

    @pytest.mark.asyncio
    async def test_execute_tool_with_empty_arguments(self):
        config = _make_config_mock()
        mem_cfg = _make_memory_config_mock()

        with patch(f"{_P}.get_memory_config", return_value=mem_cfg), patch(
            f"{_P}.get_config", return_value=config
        ):
            orch = FridayOrchestrator(config=config)

        mock_registry = MagicMock()
        tool_result = MagicMock()
        tool_result.success = True
        mock_registry.async_execute = AsyncMock(return_value=tool_result)

        orch._initialized = True
        orch._tool_registry = mock_registry

        await orch.execute_tool("document_list", {})
        mock_registry.async_execute.assert_awaited_once_with("document_list", {})


# ---------------------------------------------------------------------------
# 8. TestHandleToolCalls
# ---------------------------------------------------------------------------


class TestHandleToolCalls:
    """Tests for FridayOrchestrator._handle_tool_calls()."""

    def _make_orch(self, mock_llm=None, mock_builder=None, mock_registry=None):
        config = _make_config_mock()
        mem_cfg = _make_memory_config_mock()

        with patch(f"{_P}.get_memory_config", return_value=mem_cfg), patch(
            f"{_P}.get_config", return_value=config
        ):
            orch = FridayOrchestrator(config=config)

        orch._initialized = True
        orch._llm_client = mock_llm or MagicMock()
        orch._context_builder = mock_builder or MagicMock()
        orch._tool_registry = mock_registry or MagicMock()
        return orch

    @pytest.mark.asyncio
    async def test_single_tool_call(self):
        tc = _make_tool_call(call_id="c1", name="scene_get", arguments={"id": 5})
        initial_response = _make_chat_response(
            content="", has_tools=True, tool_calls=[tc]
        )
        final_response = _make_chat_response(content="Here is scene 5.")
        final_response.has_tool_calls = False

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=final_response)

        tool_result = MagicMock()
        tool_result.success = True
        tool_result.data = "scene content"
        tool_result.error = None

        mock_registry = MagicMock()
        mock_registry.async_execute = AsyncMock(return_value=tool_result)

        mock_builder = MagicMock()
        mock_builder.build_for_tool_response.return_value = [MagicMock()]

        orch = self._make_orch(
            mock_llm=mock_llm, mock_builder=mock_builder, mock_registry=mock_registry
        )

        built_ctx = _make_built_context()
        calls, results, resp = await orch._handle_tool_calls(
            initial_response, built_ctx
        )

        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "scene_get"
        assert len(results) == 1
        assert results[0]["success"] is True
        assert resp.content == "Here is scene 5."

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_round(self):
        tc1 = _make_tool_call(call_id="c1", name="scene_get")
        tc2 = _make_tool_call(call_id="c2", name="scene_search")
        initial_response = _make_chat_response(
            content="", has_tools=True, tool_calls=[tc1, tc2]
        )
        final_response = _make_chat_response(content="Found both.")
        final_response.has_tool_calls = False

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=final_response)

        tool_result = MagicMock()
        tool_result.success = True
        tool_result.data = "data"
        tool_result.error = None

        mock_registry = MagicMock()
        mock_registry.async_execute = AsyncMock(return_value=tool_result)

        mock_builder = MagicMock()
        mock_builder.build_for_tool_response.return_value = [MagicMock()]

        orch = self._make_orch(
            mock_llm=mock_llm, mock_builder=mock_builder, mock_registry=mock_registry
        )

        calls, results, resp = await orch._handle_tool_calls(
            initial_response, _make_built_context()
        )
        assert len(calls) == 2
        assert len(results) == 2
        assert mock_registry.async_execute.await_count == 2

    @pytest.mark.asyncio
    async def test_max_iterations_respected(self):
        """If tool calls keep coming, stop after max_iterations."""
        tc = _make_tool_call()

        # Every response has tool calls -> infinite loop without max_iterations
        looping_response = _make_chat_response(
            content="", has_tools=True, tool_calls=[tc]
        )

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=looping_response)

        tool_result = MagicMock()
        tool_result.success = True
        tool_result.data = "ok"
        tool_result.error = None

        mock_registry = MagicMock()
        mock_registry.async_execute = AsyncMock(return_value=tool_result)

        mock_builder = MagicMock()
        mock_builder.build_for_tool_response.return_value = [MagicMock()]

        orch = self._make_orch(
            mock_llm=mock_llm, mock_builder=mock_builder, mock_registry=mock_registry
        )

        calls, results, resp = await orch._handle_tool_calls(
            looping_response, _make_built_context(), max_iterations=3
        )
        # Initial response has 1 tool call, then 3 iterations = 4 total tool calls
        # But initial response is the first iteration, so 3 iterations total
        assert mock_registry.async_execute.await_count == 3
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_tool_results_formatted_correctly(self):
        tc = _make_tool_call(call_id="call_abc", name="scene_get")
        initial = _make_chat_response(content="", has_tools=True, tool_calls=[tc])

        final = _make_chat_response(content="Done")
        final.has_tool_calls = False

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=final)

        tr = MagicMock()
        tr.success = True
        tr.data = {"scene_id": 5, "content": "Rain scene"}
        tr.error = None

        mock_registry = MagicMock()
        mock_registry.async_execute = AsyncMock(return_value=tr)

        mock_builder = MagicMock()
        mock_builder.build_for_tool_response.return_value = [MagicMock()]

        orch = self._make_orch(
            mock_llm=mock_llm, mock_builder=mock_builder, mock_registry=mock_registry
        )

        calls, results, resp = await orch._handle_tool_calls(
            initial, _make_built_context()
        )

        result = results[0]
        assert result["tool_call_id"] == "call_abc"
        assert result["name"] == "scene_get"
        assert result["success"] is True
        assert result["data"] == {"scene_id": 5, "content": "Rain scene"}
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_tool_call_dict_format(self):
        """Verify the tool_call_dict structure matches OpenAI format."""
        tc = _make_tool_call(
            call_id="tc_1", name="send_email", arguments={"to": "x@y.com"}
        )
        initial = _make_chat_response(content="", has_tools=True, tool_calls=[tc])

        final = _make_chat_response(content="Email sent")
        final.has_tool_calls = False

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=final)

        tr = MagicMock()
        tr.success = True
        tr.data = "sent"
        tr.error = None

        mock_registry = MagicMock()
        mock_registry.async_execute = AsyncMock(return_value=tr)

        mock_builder = MagicMock()
        mock_builder.build_for_tool_response.return_value = [MagicMock()]

        orch = self._make_orch(
            mock_llm=mock_llm, mock_builder=mock_builder, mock_registry=mock_registry
        )

        calls, _, _ = await orch._handle_tool_calls(initial, _make_built_context())

        tc_dict = calls[0]
        assert tc_dict["id"] == "tc_1"
        assert tc_dict["type"] == "function"
        assert tc_dict["function"]["name"] == "send_email"
        assert tc_dict["function"]["arguments"] == {"to": "x@y.com"}

    @pytest.mark.asyncio
    async def test_multi_iteration_tool_calls(self):
        """Multiple iterations where LLM returns tool calls across rounds."""
        tc1 = _make_tool_call(call_id="c1", name="scene_search")
        tc2 = _make_tool_call(call_id="c2", name="scene_get")

        round1_resp = _make_chat_response(content="", has_tools=True, tool_calls=[tc1])
        round2_resp = _make_chat_response(content="", has_tools=True, tool_calls=[tc2])
        final_resp = _make_chat_response(content="All done")
        final_resp.has_tool_calls = False

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=[round2_resp, final_resp])

        tr = MagicMock()
        tr.success = True
        tr.data = "ok"
        tr.error = None

        mock_registry = MagicMock()
        mock_registry.async_execute = AsyncMock(return_value=tr)

        mock_builder = MagicMock()
        mock_builder.build_for_tool_response.return_value = [MagicMock()]

        orch = self._make_orch(
            mock_llm=mock_llm, mock_builder=mock_builder, mock_registry=mock_registry
        )

        calls, results, resp = await orch._handle_tool_calls(
            round1_resp, _make_built_context(), max_iterations=5
        )
        assert len(calls) == 2  # c1 from round1, c2 from round2
        assert resp.content == "All done"


# ---------------------------------------------------------------------------
# 9. TestSessionInfo
# ---------------------------------------------------------------------------


class TestSessionInfo:
    """Tests for get_session_info() and list_sessions()."""

    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_get_session_info_returns_dict(
        self, mock_cfg, mock_mem_cfg, mock_adapter_cls
    ):
        adapter = MagicMock()
        adapter.turn_count = 5
        adapter.active_turns = 3
        adapter.current_context = "writers_room"
        adapter.started_at = 1700000000.0
        adapter.total_tokens = 1500
        adapter.capacity_zone = "normal"
        adapter.capacity_percentage = 0.35
        adapter.tokens_available = 2600
        mock_adapter_cls.return_value = adapter

        orch = FridayOrchestrator()
        orch._create_session(session_id="info-test")

        info = orch.get_session_info("info-test")
        assert info["session_id"] == "info-test"
        assert info["turn_count"] == 5
        assert info["active_turns"] == 3
        assert info["current_context"] == "writers_room"
        assert info["capacity_zone"] == "normal"

    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_get_session_info_unknown_session(self, mock_cfg, mock_mem_cfg):
        orch = FridayOrchestrator()
        info = orch.get_session_info("nonexistent")
        assert "error" in info
        assert info["error"] == "Session not found"

    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_get_session_info_no_session_id(self, mock_cfg, mock_mem_cfg):
        """When no session_id is given and no current session exists."""
        orch = FridayOrchestrator()
        info = orch.get_session_info()
        assert info == {"error": "Session not found"}

    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_get_session_info_uses_current_session(
        self, mock_cfg, mock_mem_cfg, mock_adapter_cls
    ):
        adapter = MagicMock()
        adapter.turn_count = 2
        adapter.active_turns = 2
        adapter.current_context = "general"
        adapter.started_at = 1700000000.0
        adapter.total_tokens = 500
        adapter.capacity_zone = "normal"
        adapter.capacity_percentage = 0.1
        adapter.tokens_available = 3500
        mock_adapter_cls.return_value = adapter

        orch = FridayOrchestrator()
        orch._create_session(session_id="current")

        info = orch.get_session_info()  # No session_id -> uses current
        assert info["session_id"] == "current"

    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_list_sessions_returns_list(self, mock_cfg, mock_mem_cfg, mock_adapter_cls):
        adapter = MagicMock()
        adapter.turn_count = 1
        adapter.current_context = "general"
        adapter.started_at = 1700000000.0
        mock_adapter_cls.return_value = adapter

        orch = FridayOrchestrator()
        orch._create_session(session_id="s1")
        orch._create_session(session_id="s2")

        sessions = orch.list_sessions()
        assert len(sessions) == 2
        session_ids = [s["session_id"] for s in sessions]
        assert "s1" in session_ids
        assert "s2" in session_ids
        # Each entry should have expected keys
        for s in sessions:
            assert "turn_count" in s
            assert "current_context" in s
            assert "started_at" in s


# ---------------------------------------------------------------------------
# 10. TestHealthCheck
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for FridayOrchestrator.health_check()."""

    def _make_orch_for_health(
        self,
        initialized=True,
        router_enabled=False,
        llm_healthy=True,
        num_tools=5,
        session=None,
    ):
        config = _make_config_mock(router_enabled=router_enabled)
        mem_cfg = _make_memory_config_mock()

        with patch(f"{_P}.get_memory_config", return_value=mem_cfg), patch(
            f"{_P}.get_config", return_value=config
        ):
            orch = FridayOrchestrator(config=config)

        orch._initialized = initialized

        mock_llm = MagicMock()
        mock_llm.health_check = AsyncMock(return_value=llm_healthy)
        orch._llm_client = mock_llm if initialized else None

        mock_registry = MagicMock()
        mock_registry._tools = {f"tool_{i}": MagicMock() for i in range(num_tools)}
        orch._tool_registry = mock_registry if initialized else None

        if router_enabled:
            mock_router = MagicMock()
            mock_router._cache = {"key1": "val1"}
            orch._router = mock_router
        else:
            orch._router = None

        if session:
            orch._sessions = {"main": session}
            orch._current_session_id = "main"

        return orch

    @pytest.mark.asyncio
    async def test_healthy_state(self):
        orch = self._make_orch_for_health(initialized=True, llm_healthy=True)
        health = await orch.health_check()

        assert health["orchestrator"] == "healthy"
        assert health["initialized"] is True
        assert health["llm"] == "healthy"

    @pytest.mark.asyncio
    async def test_llm_unhealthy(self):
        orch = self._make_orch_for_health(llm_healthy=False)
        health = await orch.health_check()
        assert health["llm"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_llm_health_check_error(self):
        orch = self._make_orch_for_health()
        orch._llm_client.health_check = AsyncMock(
            side_effect=Exception("connection refused")
        )
        health = await orch.health_check()
        assert "error" in health["llm"]

    @pytest.mark.asyncio
    async def test_llm_not_initialized(self):
        orch = self._make_orch_for_health(initialized=False)
        health = await orch.health_check()
        assert health["llm"] == "not_initialized"

    @pytest.mark.asyncio
    async def test_router_info_included(self):
        orch = self._make_orch_for_health(router_enabled=True)
        health = await orch.health_check()

        assert health["router"]["enabled"] is True
        assert health["router"]["provider"] == "zhipu"
        assert health["router"]["model"] == "glm-4.7-flash"
        assert health["router"]["cache_size"] == 1

    @pytest.mark.asyncio
    async def test_router_disabled_info(self):
        orch = self._make_orch_for_health(router_enabled=False)
        health = await orch.health_check()
        assert health["router"] == {"enabled": False}

    @pytest.mark.asyncio
    async def test_tool_count(self):
        orch = self._make_orch_for_health(num_tools=8)
        health = await orch.health_check()
        assert health["tools"] == 8

    @pytest.mark.asyncio
    async def test_tool_count_zero_when_not_initialized(self):
        orch = self._make_orch_for_health(initialized=False, num_tools=0)
        health = await orch.health_check()
        assert health["tools"] == 0

    @pytest.mark.asyncio
    async def test_memory_health_included(self):
        session = MagicMock()
        session.get_health_status.return_value = {
            "status": "healthy",
            "capacity_zone": "normal",
            "capacity_percentage": 0.3,
        }
        orch = self._make_orch_for_health(session=session)
        health = await orch.health_check()

        assert health["memory"]["status"] == "healthy"
        assert health["memory"]["capacity_zone"] == "normal"

    @pytest.mark.asyncio
    async def test_memory_no_active_session(self):
        orch = self._make_orch_for_health()
        health = await orch.health_check()
        assert health["memory"]["status"] == "no_active_session"

    @pytest.mark.asyncio
    async def test_active_sessions_count(self):
        orch = self._make_orch_for_health()
        orch._sessions = {"a": MagicMock(), "b": MagicMock(), "c": MagicMock()}
        health = await orch.health_check()
        assert health["active_sessions"] == 3

    @pytest.mark.asyncio
    async def test_last_routing_decision_in_health(self):
        orch = self._make_orch_for_health(router_enabled=True)
        decision = MagicMock()
        decision.task_type = MagicMock()
        decision.task_type.value = "scene_query"
        decision.complexity = MagicMock()
        decision.complexity.value = "moderate"
        decision.confidence = 0.92
        orch._last_routing_decision = decision

        health = await orch.health_check()
        assert health["router"]["last_decision"]["task_type"] == "scene_query"
        assert health["router"]["last_decision"]["confidence"] == 0.92


# ---------------------------------------------------------------------------
# 11. TestSingleton
# ---------------------------------------------------------------------------


class TestSingleton:
    """Tests for get_orchestrator() and initialize_orchestrator() singletons."""

    def teardown_method(self):
        """Reset the global singleton after each test."""
        import orchestrator.core as core_mod

        core_mod._orchestrator = None

    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_get_orchestrator_returns_same_instance(self, mock_cfg, mock_mem_cfg):
        orch1 = get_orchestrator()
        orch2 = get_orchestrator()
        assert orch1 is orch2

    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_get_orchestrator_creates_instance(self, mock_cfg, mock_mem_cfg):
        orch = get_orchestrator()
        assert isinstance(orch, FridayOrchestrator)

    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    def test_reset_global_for_test_isolation(self, mock_cfg, mock_mem_cfg):
        import orchestrator.core as core_mod

        orch1 = get_orchestrator()
        core_mod._orchestrator = None
        orch2 = get_orchestrator()
        assert orch1 is not orch2

    @pytest.mark.asyncio
    @patch(f"{_P}.WorkingMemoryAdapter")
    @patch(f"{_P}.ContextBuilder")
    @patch(f"{_P}.get_default_system_prompt", return_value="prompt")
    @patch(f"{_P}.ContextDetector")
    @patch(f"{_P}.get_tool_registry")
    @patch(f"{_P}.LLMClient")
    @patch(f"{_P}.get_memory_config", return_value=_make_memory_config_mock())
    @patch(f"{_P}.get_config", return_value=_make_config_mock())
    async def test_initialize_orchestrator_returns_initialized(
        self,
        mock_cfg,
        mock_mem_cfg,
        mock_llm,
        mock_registry,
        mock_detector,
        mock_prompt,
        mock_builder,
        mock_adapter,
    ):
        orch = await initialize_orchestrator()
        assert orch.is_initialized is True


# ---------------------------------------------------------------------------
# 12. TestChatPipelineEdgeCases
# ---------------------------------------------------------------------------


class TestChatPipelineEdgeCases:
    """Additional edge case tests for the chat pipeline."""

    def _quick_orch(self, **overrides):
        """Build a minimal initialized orchestrator for edge case tests."""
        config = _make_config_mock(
            router_enabled=overrides.get("router_enabled", False)
        )
        mem_cfg = _make_memory_config_mock()

        with patch(f"{_P}.get_memory_config", return_value=mem_cfg), patch(
            f"{_P}.get_config", return_value=config
        ):
            orch = FridayOrchestrator(config=config)

        orch._initialized = True
        orch._llm_client = overrides.get("llm", MagicMock())
        orch._router = overrides.get("router", None)
        orch._context_detector = overrides.get("detector", MagicMock())
        orch._context_builder = overrides.get("builder", MagicMock())
        orch._tool_registry = overrides.get("registry", MagicMock())

        memory = overrides.get("memory", None)
        if memory is None:
            memory = MagicMock()
            memory.turn_count = 0
            memory.add_turn = MagicMock(return_value=_make_adapted_turn())
            memory.set_context = MagicMock()

        orch._sessions = {"default": memory}
        orch._current_session_id = "default"

        return orch

    @pytest.mark.asyncio
    async def test_chat_without_session_creates_one(self):
        """If no session exists at all, chat should create one."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        config = _make_config_mock()
        mem_cfg = _make_memory_config_mock()

        with patch(f"{_P}.get_memory_config", return_value=mem_cfg), patch(
            f"{_P}.get_config", return_value=config
        ), patch(f"{_P}.WorkingMemoryAdapter") as mock_adapter_cls:

            mock_memory = MagicMock()
            mock_memory.turn_count = 0
            mock_memory.add_turn.return_value = _make_adapted_turn()
            mock_memory.set_context = MagicMock()
            mock_adapter_cls.return_value = mock_memory

            orch = FridayOrchestrator(config=config)
            orch._initialized = True
            orch._llm_client = mock_llm
            orch._context_detector = mock_detector
            orch._context_builder = mock_builder
            orch._tool_registry = MagicMock()
            # No sessions at all
            orch._sessions = {}
            orch._current_session_id = None

            resp = await orch.chat("Hey")
            assert isinstance(resp, OrchestratorResponse)
            assert len(orch._sessions) >= 1

    @pytest.mark.asyncio
    async def test_router_tool_filter_passed_to_context_builder(self):
        """Router's suggested_tools should be passed as tool_filter to builder."""
        decision = _make_routing_decision(
            confidence=0.9,
            primary_context="general",
            tools=["scene_get", "scene_search"],
        )

        mock_router = MagicMock()
        mock_router.analyze = AsyncMock(return_value=decision)

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch = self._quick_orch(
            router_enabled=True,
            router=mock_router,
            llm=mock_llm,
            builder=mock_builder,
        )

        await orch.chat("Find scene 5")
        # Check that build was called with tool_filter
        build_call = mock_builder.build.call_args
        assert build_call.kwargs.get("tool_filter") == ["scene_get", "scene_search"]

    @pytest.mark.asyncio
    async def test_chat_processing_time_recorded(self):
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch = self._quick_orch(
            llm=mock_llm, detector=mock_detector, builder=mock_builder
        )

        resp = await orch.chat("Quick question")
        assert resp.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_chat_tokens_used_from_response(self):
        chat_resp = _make_chat_response()
        chat_resp.usage = {"prompt_tokens": 150, "completion_tokens": 75}

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=chat_resp)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch = self._quick_orch(
            llm=mock_llm, detector=mock_detector, builder=mock_builder
        )

        resp = await orch.chat("Check tokens")
        assert resp.tokens_used["prompt_tokens"] == 150
        assert resp.tokens_used["completion_tokens"] == 75

    @pytest.mark.asyncio
    async def test_context_not_switched_when_same(self):
        """If detected context matches current, set_context should not be called."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_detector = MagicMock()
        mock_detector.detect.return_value = ContextType.GENERAL

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        mock_memory = MagicMock()
        mock_memory.turn_count = 0
        mock_memory.add_turn.return_value = _make_adapted_turn()
        mock_memory.set_context = MagicMock()

        orch = self._quick_orch(
            llm=mock_llm,
            detector=mock_detector,
            builder=mock_builder,
            memory=mock_memory,
        )
        orch._current_context = ContextType.GENERAL

        await orch.chat("Hello")
        # Context is already GENERAL, detect returns GENERAL -> no switch
        mock_memory.set_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_router_conversation_context_built_from_history(self):
        """When memory has turns, router should receive conversation context."""
        turn_mock = MagicMock()
        turn_mock.user_message = "Previous question"
        turn_mock.assistant_response = "Previous answer"

        mock_memory = MagicMock()
        mock_memory.turn_count = 3
        mock_memory.get_last_n_turns.return_value = [turn_mock]
        mock_memory.add_turn.return_value = _make_adapted_turn()
        mock_memory.set_context = MagicMock()

        decision = _make_routing_decision(confidence=0.9)
        mock_router = MagicMock()
        mock_router.analyze = AsyncMock(return_value=decision)

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch = self._quick_orch(
            router_enabled=True,
            router=mock_router,
            llm=mock_llm,
            builder=mock_builder,
            memory=mock_memory,
        )

        await orch.chat("Follow up question")

        # Router should have been called with conversation_context
        analyze_call = mock_router.analyze.call_args
        assert analyze_call.kwargs.get("conversation_context") is not None

    @pytest.mark.asyncio
    async def test_agent_mode_increases_max_iterations(self):
        """When router returns agent_mode=True, max iterations should increase."""
        decision = _make_routing_decision(
            confidence=0.9, agent_mode=True, expected_turns=10
        )
        mock_router = MagicMock()
        mock_router.analyze = AsyncMock(return_value=decision)

        tc = _make_tool_call()
        first_resp = _make_chat_response(content="", has_tools=True, tool_calls=[tc])
        final_resp = _make_chat_response(content="Done")
        final_resp.has_tool_calls = False

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=[first_resp, final_resp])

        tr = MagicMock()
        tr.success = True
        tr.data = "ok"
        tr.error = None

        mock_registry = MagicMock()
        mock_registry.async_execute = AsyncMock(return_value=tr)

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()
        mock_builder.build_for_tool_response.return_value = [MagicMock()]

        orch = self._quick_orch(
            router_enabled=True,
            router=mock_router,
            llm=mock_llm,
            builder=mock_builder,
            registry=mock_registry,
        )

        resp = await orch.chat("Complex task requiring agent mode")
        assert resp.content == "Done"
        # The key check: it did not prematurely stop at 5 iterations
        assert len(resp.tool_calls_made) >= 1

    @pytest.mark.asyncio
    async def test_context_map_kitchen(self):
        """Router returning 'kitchen' primary_context should map to ContextType.KITCHEN."""
        decision = _make_routing_decision(confidence=0.9, primary_context="kitchen")
        mock_router = MagicMock()
        mock_router.analyze = AsyncMock(return_value=decision)

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch = self._quick_orch(
            router_enabled=True,
            router=mock_router,
            llm=mock_llm,
            builder=mock_builder,
        )
        orch._current_context = ContextType.GENERAL

        await orch.chat("Let me cook")
        assert orch._current_context == ContextType.KITCHEN

    @pytest.mark.asyncio
    async def test_context_map_storyboard(self):
        """Router returning 'storyboard' primary_context maps correctly."""
        decision = _make_routing_decision(confidence=0.85, primary_context="storyboard")
        mock_router = MagicMock()
        mock_router.analyze = AsyncMock(return_value=decision)

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch = self._quick_orch(
            router_enabled=True,
            router=mock_router,
            llm=mock_llm,
            builder=mock_builder,
        )
        orch._current_context = ContextType.GENERAL

        await orch.chat("Visualize shot")
        assert orch._current_context == ContextType.STORYBOARD

    @pytest.mark.asyncio
    async def test_context_map_unknown_keeps_current(self):
        """Router with unknown primary_context falls back to current context."""
        decision = _make_routing_decision(
            confidence=0.9, primary_context="unknown_room"
        )
        mock_router = MagicMock()
        mock_router.analyze = AsyncMock(return_value=decision)

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=_make_chat_response())

        mock_builder = MagicMock()
        mock_builder.build.return_value = _make_built_context()

        orch = self._quick_orch(
            router_enabled=True,
            router=mock_router,
            llm=mock_llm,
            builder=mock_builder,
        )
        orch._current_context = ContextType.WRITERS_ROOM

        await orch.chat("Something")
        # Unknown context -> stays at current
        assert orch._current_context == ContextType.WRITERS_ROOM
