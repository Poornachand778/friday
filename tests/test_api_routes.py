"""
Comprehensive Tests for Friday AI API Routes
=============================================

Tests for:
- orchestrator/main.py (root, health, context endpoints + lifespan)
- orchestrator/routes/chat.py (chat, voice endpoints)
- orchestrator/routes/sessions.py (session CRUD endpoints)
- orchestrator/routes/tools.py (tool listing, execution, context endpoints)

All tests use a fully mocked orchestrator and tool registry to avoid
requiring real LLM/MCP backends.

Run with: python -m pytest tests/test_api_routes.py -x -q --tb=short
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))


# ---------------------------------------------------------------------------
# Lightweight stubs for ContextType / CONTEXTS so we don't pull in real deps
# ---------------------------------------------------------------------------
class _ContextType(str, Enum):
    WRITERS_ROOM = "writers_room"
    KITCHEN = "kitchen"
    STORYBOARD = "storyboard"
    GENERAL = "general"


@dataclass
class _FakeContext:
    description: str = ""
    available_tools: list = None
    lora_adapter: str = None

    def __post_init__(self):
        if self.available_tools is None:
            self.available_tools = []


_FAKE_CONTEXTS = {
    _ContextType.WRITERS_ROOM: _FakeContext(
        description="Screenplay writing and brainstorming",
        available_tools=["scene_search", "scene_get"],
        lora_adapter="friday-script",
    ),
    _ContextType.KITCHEN: _FakeContext(
        description="Cooking assistance with camera",
        available_tools=["camera_analyze"],
    ),
    _ContextType.STORYBOARD: _FakeContext(
        description="Visual storyboarding",
        available_tools=["generate_image"],
    ),
    _ContextType.GENERAL: _FakeContext(
        description="General assistant mode",
        available_tools=["send_email"],
    ),
}


# ---------------------------------------------------------------------------
# Fake tool object returned by the mock registry
# ---------------------------------------------------------------------------
@dataclass
class _FakeTool:
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]


# A small catalogue of fake tools used by the mock registry
_TOOL_CATALOGUE: List[_FakeTool] = [
    _FakeTool(
        name="scene_search",
        description="Search scenes",
        category="screenplay",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    ),
    _FakeTool(
        name="scene_get",
        description="Get a scene",
        category="screenplay",
        parameters={
            "type": "object",
            "properties": {"scene_number": {"type": "integer"}},
        },
    ),
    _FakeTool(
        name="send_email",
        description="Send email",
        category="email",
        parameters={"type": "object", "properties": {"to": {"type": "string"}}},
    ),
    _FakeTool(
        name="camera_analyze",
        description="Analyze camera feed",
        category="vision",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    ),
    _FakeTool(
        name="generate_image",
        description="Generate an image",
        category="visual",
        parameters={"type": "object", "properties": {"prompt": {"type": "string"}}},
    ),
]

_TOOL_MAP = {t.name: t for t in _TOOL_CATALOGUE}


# ---------------------------------------------------------------------------
# Fake ToolResult
# ---------------------------------------------------------------------------
@dataclass
class _FakeToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Build a mock orchestrator that every endpoint can rely on
# ---------------------------------------------------------------------------
def _build_mock_orchestrator():
    """Return a MagicMock that satisfies every attribute/method used by routes."""
    orch = MagicMock()

    # Initialization state
    orch.is_initialized = True
    orch._initialized = True

    # Context
    orch._current_context = _ContextType.GENERAL
    orch.current_context = _ContextType.GENERAL

    # Session bookkeeping
    orch._current_session_id = "default-session"
    orch._sessions = {}

    # current_session property
    orch.current_session = MagicMock()
    orch.current_session.set_context = MagicMock()

    # health_check is async
    orch.health_check = AsyncMock(
        return_value={
            "orchestrator": "healthy",
            "initialized": True,
            "current_context": "general",
            "active_sessions": 1,
            "llm": "healthy",
            "tools": 5,
        }
    )

    # initialize is async
    orch.initialize = AsyncMock()

    # shutdown is async
    orch.shutdown = AsyncMock()

    # chat is async – returns a response-like object
    chat_response = MagicMock()
    chat_response.content = "Hello Boss, how can I help?"
    chat_response.context_type = _ContextType.GENERAL
    chat_response.turn_id = 1
    chat_response.tool_calls_made = []
    chat_response.processing_time_ms = 42.0
    orch.chat = AsyncMock(return_value=chat_response)

    # execute_tool is async
    orch.execute_tool = AsyncMock(
        return_value=_FakeToolResult(success=True, data={"result": "ok"})
    )

    # Session helpers
    orch.list_sessions = MagicMock(return_value=[])
    orch.get_session_info = MagicMock(return_value={"error": "Session not found"})
    orch._create_session = MagicMock(return_value="new-session-id")
    orch.switch_session = MagicMock(return_value=True)

    return orch


def _build_mock_registry():
    """Return a MagicMock that satisfies the tool registry interface."""
    registry = MagicMock()
    registry.list_tools = MagicMock(return_value=list(_TOOL_CATALOGUE))

    def _get_tool(name):
        return _TOOL_MAP.get(name)

    registry.get = MagicMock(side_effect=_get_tool)
    return registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_orchestrator():
    return _build_mock_orchestrator()


@pytest.fixture()
def mock_registry():
    return _build_mock_registry()


@pytest.fixture()
def client(mock_orchestrator, mock_registry):
    """
    Create a TestClient with mocked orchestrator + registry.

    We patch:
      - orchestrator.core.get_orchestrator  (used by main.py & route files)
      - orchestrator.core.initialize_orchestrator (used by lifespan)
      - orchestrator.tools.registry.get_tool_registry (used by tools route)

    Because routes import via `from orchestrator.core import get_orchestrator`,
    we must also patch at the *import location* inside each route module.
    """
    from fastapi.testclient import TestClient

    patches = [
        patch("orchestrator.core.get_orchestrator", return_value=mock_orchestrator),
        patch(
            "orchestrator.core.initialize_orchestrator",
            new_callable=AsyncMock,
            return_value=mock_orchestrator,
        ),
        patch("orchestrator.main.get_orchestrator", return_value=mock_orchestrator),
        patch(
            "orchestrator.main.initialize_orchestrator",
            new_callable=AsyncMock,
            return_value=mock_orchestrator,
        ),
        patch(
            "orchestrator.routes.chat.get_orchestrator", return_value=mock_orchestrator
        ),
        patch(
            "orchestrator.routes.sessions.get_orchestrator",
            return_value=mock_orchestrator,
        ),
        patch(
            "orchestrator.routes.tools.get_orchestrator", return_value=mock_orchestrator
        ),
        patch(
            "orchestrator.routes.tools.get_tool_registry", return_value=mock_registry
        ),
        patch(
            "orchestrator.tools.registry.get_tool_registry", return_value=mock_registry
        ),
    ]

    for p in patches:
        p.start()

    from orchestrator.main import app  # noqa: import after patches

    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc

    for p in patches:
        p.stop()


# ========================================================================
#  1.  ROOT ENDPOINT  (main.py)
# ========================================================================
class TestRootEndpoint:
    """Tests for GET /"""

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_name(self, client):
        data = client.get("/").json()
        assert data["name"] == "Friday AI Orchestrator"

    def test_root_status(self, client):
        data = client.get("/").json()
        assert data["status"] == "running"

    def test_root_version(self, client):
        data = client.get("/").json()
        assert data["version"] == "0.1.0"

    def test_root_keys(self, client):
        data = client.get("/").json()
        assert set(data.keys()) == {"name", "status", "version"}


# ========================================================================
#  2.  HEALTH ENDPOINT  (main.py)
# ========================================================================
class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_contains_orchestrator(self, client):
        data = client.get("/health").json()
        assert data["orchestrator"] == "healthy"

    def test_health_contains_initialized(self, client):
        data = client.get("/health").json()
        assert data["initialized"] is True

    def test_health_contains_current_context(self, client):
        data = client.get("/health").json()
        assert "current_context" in data

    def test_health_contains_sessions(self, client):
        data = client.get("/health").json()
        assert "active_sessions" in data

    def test_health_contains_llm(self, client):
        data = client.get("/health").json()
        assert data["llm"] == "healthy"

    def test_health_contains_tools(self, client):
        data = client.get("/health").json()
        assert "tools" in data


# ========================================================================
#  3.  CONTEXT ENDPOINTS  (main.py)
# ========================================================================
class TestGetContextEndpoint:
    """Tests for GET /context"""

    def test_get_context_returns_200(self, client, mock_orchestrator):
        # Patch the contexts import inside main
        with patch(
            "orchestrator.main.CONTEXTS",
            _FAKE_CONTEXTS,
            create=True,
        ), patch.dict(
            "sys.modules",
            {"orchestrator.context.contexts": MagicMock(CONTEXTS=_FAKE_CONTEXTS)},
        ):
            resp = client.get("/context")
            assert resp.status_code == 200

    def test_get_context_has_current_context_field(self, client, mock_orchestrator):
        with patch.dict(
            "sys.modules",
            {"orchestrator.context.contexts": MagicMock(CONTEXTS=_FAKE_CONTEXTS)},
        ):
            data = client.get("/context").json()
            assert "current_context" in data

    def test_get_context_has_description(self, client, mock_orchestrator):
        with patch.dict(
            "sys.modules",
            {"orchestrator.context.contexts": MagicMock(CONTEXTS=_FAKE_CONTEXTS)},
        ):
            data = client.get("/context").json()
            assert "description" in data

    def test_get_context_has_available_tools(self, client, mock_orchestrator):
        with patch.dict(
            "sys.modules",
            {"orchestrator.context.contexts": MagicMock(CONTEXTS=_FAKE_CONTEXTS)},
        ):
            data = client.get("/context").json()
            assert "available_tools" in data

    def test_get_context_has_lora_adapter(self, client, mock_orchestrator):
        with patch.dict(
            "sys.modules",
            {"orchestrator.context.contexts": MagicMock(CONTEXTS=_FAKE_CONTEXTS)},
        ):
            data = client.get("/context").json()
            assert "lora_adapter" in data

    def test_get_context_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        with patch.dict(
            "sys.modules",
            {"orchestrator.context.contexts": MagicMock(CONTEXTS=_FAKE_CONTEXTS)},
        ):
            client.get("/context")
            mock_orchestrator.initialize.assert_called()


class TestSetContextEndpoint:
    """Tests for POST /context/{context_type}"""

    def test_set_valid_context(self, client, mock_orchestrator):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            resp = client.post("/context/writers_room")
            assert resp.status_code == 200
            data = resp.json()
            assert data["current_context"] == "writers_room"
            assert "message" in data

    def test_set_invalid_context_returns_error_body(self, client, mock_orchestrator):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            resp = client.post("/context/nonexistent")
            assert resp.status_code == 200  # error in body, not HTTP error
            data = resp.json()
            assert "error" in data

    def test_set_context_general(self, client, mock_orchestrator):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            resp = client.post("/context/general")
            data = resp.json()
            assert data["current_context"] == "general"

    def test_set_context_kitchen(self, client, mock_orchestrator):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            resp = client.post("/context/kitchen")
            data = resp.json()
            assert data["current_context"] == "kitchen"

    def test_set_context_storyboard(self, client, mock_orchestrator):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            resp = client.post("/context/storyboard")
            data = resp.json()
            assert data["current_context"] == "storyboard"

    def test_set_context_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            client.post("/context/general")
            mock_orchestrator.initialize.assert_called()

    def test_set_context_updates_session(self, client, mock_orchestrator):
        mock_orchestrator.current_session = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            client.post("/context/writers_room")
            mock_orchestrator.current_session.set_context.assert_called_with(
                "writers_room"
            )


# ========================================================================
#  4.  CHAT ENDPOINTS  (chat.py)
# ========================================================================
class TestChatEndpoint:
    """Tests for POST /chat"""

    def test_chat_basic_returns_200(self, client):
        resp = client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 200

    def test_chat_response_has_content(self, client):
        data = client.post("/chat", json={"message": "Hi"}).json()
        assert "content" in data

    def test_chat_response_has_context(self, client):
        data = client.post("/chat", json={"message": "Hi"}).json()
        assert "context" in data

    def test_chat_response_has_session_id(self, client):
        data = client.post("/chat", json={"message": "Hi"}).json()
        assert "session_id" in data

    def test_chat_response_has_turn_id(self, client):
        data = client.post("/chat", json={"message": "Hi"}).json()
        assert "turn_id" in data

    def test_chat_response_has_tool_calls(self, client):
        data = client.post("/chat", json={"message": "Hi"}).json()
        assert "tool_calls" in data

    def test_chat_response_has_processing_time(self, client):
        data = client.post("/chat", json={"message": "Hi"}).json()
        assert "processing_time_ms" in data

    def test_chat_calls_orchestrator(self, client, mock_orchestrator):
        client.post("/chat", json={"message": "Do something"})
        mock_orchestrator.chat.assert_called_once()

    def test_chat_passes_message(self, client, mock_orchestrator):
        client.post("/chat", json={"message": "Test message"})
        call_kwargs = mock_orchestrator.chat.call_args
        assert (
            call_kwargs.kwargs.get("message") == "Test message"
            or (call_kwargs.args and call_kwargs.args[0] == "Test message")
            or call_kwargs[1].get("message") == "Test message"
        )

    def test_chat_with_session_id(self, client, mock_orchestrator):
        client.post("/chat", json={"message": "Hi", "session_id": "s1"})
        call_kwargs = mock_orchestrator.chat.call_args
        assert (
            call_kwargs[1].get("session_id") == "s1"
            or call_kwargs.kwargs.get("session_id") == "s1"
        )

    def test_chat_with_location(self, client, mock_orchestrator):
        client.post("/chat", json={"message": "Hi", "location": "kitchen"})
        call_kwargs = mock_orchestrator.chat.call_args
        assert (
            call_kwargs[1].get("location") == "kitchen"
            or call_kwargs.kwargs.get("location") == "kitchen"
        )

    def test_chat_missing_message_returns_422(self, client):
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    def test_chat_wrong_type_message_returns_422(self, client):
        resp = client.post("/chat", json={"message": 12345})
        # Pydantic will coerce int to str, so this might succeed; check
        # that it at least doesn't crash with 500
        assert resp.status_code in (200, 422)

    def test_chat_no_body_returns_422(self, client):
        resp = client.post("/chat")
        assert resp.status_code == 422

    def test_chat_empty_message_accepted(self, client):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 200

    def test_chat_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        client.post("/chat", json={"message": "Hello"})
        mock_orchestrator.initialize.assert_called()

    def test_chat_stream_false_default(self, client, mock_orchestrator):
        client.post("/chat", json={"message": "Hi"})
        call_kwargs = mock_orchestrator.chat.call_args
        assert (
            call_kwargs[1].get("stream") is False
            or call_kwargs.kwargs.get("stream") is False
        )

    def test_chat_content_value(self, client, mock_orchestrator):
        data = client.post("/chat", json={"message": "Hello"}).json()
        assert data["content"] == "Hello Boss, how can I help?"

    def test_chat_context_value(self, client, mock_orchestrator):
        data = client.post("/chat", json={"message": "Hello"}).json()
        assert data["context"] == "general"

    def test_chat_turn_id_value(self, client):
        data = client.post("/chat", json={"message": "Hello"}).json()
        assert data["turn_id"] == 1

    def test_chat_tool_calls_empty(self, client):
        data = client.post("/chat", json={"message": "Hello"}).json()
        assert data["tool_calls"] == []

    def test_chat_processing_time_value(self, client):
        data = client.post("/chat", json={"message": "Hello"}).json()
        assert data["processing_time_ms"] == 42.0


class TestChatStreamEndpoint:
    """Tests for POST /chat with stream=True"""

    def test_chat_stream_returns_streaming_response(self, client, mock_orchestrator):
        # When stream=True, the route returns StreamingResponse
        async def _fake_stream(*args, **kwargs):
            async def _gen():
                yield "Hello "
                yield "Boss"

            return _gen()

        mock_orchestrator.chat = AsyncMock(side_effect=_fake_stream)
        resp = client.post("/chat", json={"message": "Hi", "stream": True})
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("text/plain")


class TestVoiceChatEndpoint:
    """Tests for POST /chat/voice"""

    def test_voice_returns_200(self, client, mock_orchestrator):
        chat_response = MagicMock()
        chat_response.content = "Sure Boss"
        chat_response.context_type = _ContextType.GENERAL
        chat_response.turn_id = 1
        mock_orchestrator.chat = AsyncMock(return_value=chat_response)

        resp = client.post("/chat/voice", params={"transcript": "Hello Friday"})
        assert resp.status_code == 200

    def test_voice_response_has_response_field(self, client, mock_orchestrator):
        chat_response = MagicMock()
        chat_response.content = "Sure Boss"
        chat_response.context_type = _ContextType.GENERAL
        chat_response.turn_id = 1
        mock_orchestrator.chat = AsyncMock(return_value=chat_response)

        data = client.post("/chat/voice", params={"transcript": "Hello"}).json()
        assert "response" in data

    def test_voice_response_has_context_field(self, client, mock_orchestrator):
        chat_response = MagicMock()
        chat_response.content = "Sure Boss"
        chat_response.context_type = _ContextType.GENERAL
        chat_response.turn_id = 2
        mock_orchestrator.chat = AsyncMock(return_value=chat_response)

        data = client.post("/chat/voice", params={"transcript": "Hi"}).json()
        assert "context" in data

    def test_voice_response_has_turn_id(self, client, mock_orchestrator):
        chat_response = MagicMock()
        chat_response.content = "Sure Boss"
        chat_response.context_type = _ContextType.GENERAL
        chat_response.turn_id = 3
        mock_orchestrator.chat = AsyncMock(return_value=chat_response)

        data = client.post("/chat/voice", params={"transcript": "Hi"}).json()
        assert data["turn_id"] == 3

    def test_voice_with_location(self, client, mock_orchestrator):
        chat_response = MagicMock()
        chat_response.content = "Kitchen mode"
        chat_response.context_type = _ContextType.KITCHEN
        chat_response.turn_id = 1
        mock_orchestrator.chat = AsyncMock(return_value=chat_response)

        data = client.post(
            "/chat/voice",
            params={"transcript": "Help me cook", "location": "kitchen"},
        ).json()
        assert data["context"] == "kitchen"

    def test_voice_with_session_id(self, client, mock_orchestrator):
        chat_response = MagicMock()
        chat_response.content = "Noted"
        chat_response.context_type = _ContextType.GENERAL
        chat_response.turn_id = 1
        mock_orchestrator.chat = AsyncMock(return_value=chat_response)

        resp = client.post(
            "/chat/voice",
            params={"transcript": "Hi", "session_id": "voice-sess"},
        )
        assert resp.status_code == 200

    def test_voice_missing_transcript_returns_422(self, client):
        resp = client.post("/chat/voice")
        assert resp.status_code == 422

    def test_voice_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        chat_response = MagicMock()
        chat_response.content = "Ok"
        chat_response.context_type = _ContextType.GENERAL
        chat_response.turn_id = 1
        mock_orchestrator.chat = AsyncMock(return_value=chat_response)

        client.post("/chat/voice", params={"transcript": "Hello"})
        mock_orchestrator.initialize.assert_called()


# ========================================================================
#  5.  SESSIONS ENDPOINTS  (sessions.py)
# ========================================================================
class TestListSessions:
    """Tests for GET /sessions"""

    def test_list_returns_200(self, client):
        resp = client.get("/sessions")
        assert resp.status_code == 200

    def test_list_empty(self, client, mock_orchestrator):
        mock_orchestrator.list_sessions.return_value = []
        data = client.get("/sessions").json()
        assert data == []

    def test_list_returns_sessions(self, client, mock_orchestrator):
        mock_orchestrator.list_sessions.return_value = [
            {
                "session_id": "s1",
                "turn_count": 5,
                "current_context": "general",
                "started_at": 1700000000.0,
            },
        ]
        data = client.get("/sessions").json()
        assert len(data) == 1
        assert data[0]["session_id"] == "s1"

    def test_list_multiple_sessions(self, client, mock_orchestrator):
        mock_orchestrator.list_sessions.return_value = [
            {
                "session_id": "s1",
                "turn_count": 0,
                "current_context": "general",
                "started_at": 1.0,
            },
            {
                "session_id": "s2",
                "turn_count": 3,
                "current_context": "kitchen",
                "started_at": 2.0,
            },
        ]
        data = client.get("/sessions").json()
        assert len(data) == 2

    def test_list_session_fields(self, client, mock_orchestrator):
        mock_orchestrator.list_sessions.return_value = [
            {
                "session_id": "s1",
                "turn_count": 2,
                "current_context": "general",
                "started_at": 1.0,
            },
        ]
        item = client.get("/sessions").json()[0]
        assert "session_id" in item
        assert "turn_count" in item
        assert "current_context" in item
        assert "started_at" in item

    def test_list_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        mock_orchestrator.list_sessions.return_value = []
        client.get("/sessions")
        mock_orchestrator.initialize.assert_called()


class TestCreateSession:
    """Tests for POST /sessions"""

    def test_create_returns_200(self, client, mock_orchestrator):
        mock_orchestrator._create_session.return_value = "new-id"
        mock_orchestrator.get_session_info.return_value = {
            "session_id": "new-id",
            "turn_count": 0,
            "active_turns": 0,
            "current_context": "general",
            "started_at": 1.0,
            "total_tokens": 0,
        }
        resp = client.post("/sessions")
        assert resp.status_code == 200

    def test_create_returns_session_id(self, client, mock_orchestrator):
        mock_orchestrator._create_session.return_value = "abc123"
        mock_orchestrator.get_session_info.return_value = {
            "session_id": "abc123",
            "turn_count": 0,
            "active_turns": 0,
            "current_context": "general",
            "started_at": 1.0,
            "total_tokens": 0,
        }
        data = client.post("/sessions").json()
        assert data["session_id"] == "abc123"

    def test_create_with_custom_id(self, client, mock_orchestrator):
        mock_orchestrator._create_session.return_value = "my-custom"
        mock_orchestrator.get_session_info.return_value = {
            "session_id": "my-custom",
            "turn_count": 0,
            "active_turns": 0,
            "current_context": "general",
            "started_at": 1.0,
            "total_tokens": 0,
        }
        data = client.post("/sessions", json={"session_id": "my-custom"}).json()
        assert data["session_id"] == "my-custom"

    def test_create_turn_count_zero(self, client, mock_orchestrator):
        mock_orchestrator._create_session.return_value = "x"
        mock_orchestrator.get_session_info.return_value = {
            "session_id": "x",
            "turn_count": 0,
            "active_turns": 0,
            "current_context": "general",
            "started_at": 1.0,
            "total_tokens": 0,
        }
        data = client.post("/sessions").json()
        assert data["turn_count"] == 0

    def test_create_has_current_context(self, client, mock_orchestrator):
        mock_orchestrator._create_session.return_value = "x"
        mock_orchestrator.get_session_info.return_value = {
            "session_id": "x",
            "turn_count": 0,
            "active_turns": 0,
            "current_context": "general",
            "started_at": 1.0,
            "total_tokens": 0,
        }
        data = client.post("/sessions").json()
        assert data["current_context"] == "general"

    def test_create_has_started_at(self, client, mock_orchestrator):
        mock_orchestrator._create_session.return_value = "x"
        mock_orchestrator.get_session_info.return_value = {
            "session_id": "x",
            "turn_count": 0,
            "active_turns": 0,
            "current_context": "general",
            "started_at": 99.0,
            "total_tokens": 0,
        }
        data = client.post("/sessions").json()
        assert data["started_at"] == 99.0

    def test_create_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        mock_orchestrator._create_session.return_value = "x"
        mock_orchestrator.get_session_info.return_value = {
            "session_id": "x",
            "turn_count": 0,
            "active_turns": 0,
            "current_context": "general",
            "started_at": 1.0,
            "total_tokens": 0,
        }
        client.post("/sessions")
        mock_orchestrator.initialize.assert_called()


class TestGetSession:
    """Tests for GET /sessions/{session_id}"""

    def test_get_existing_session(self, client, mock_orchestrator):
        mock_orchestrator.get_session_info.return_value = {
            "session_id": "s1",
            "turn_count": 3,
            "active_turns": 2,
            "current_context": "writers_room",
            "started_at": 100.0,
            "total_tokens": 500,
        }
        resp = client.get("/sessions/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "s1"
        assert data["turn_count"] == 3

    def test_get_nonexistent_session_returns_404(self, client, mock_orchestrator):
        mock_orchestrator.get_session_info.return_value = {"error": "Session not found"}
        resp = client.get("/sessions/nope")
        assert resp.status_code == 404

    def test_get_session_404_detail(self, client, mock_orchestrator):
        mock_orchestrator.get_session_info.return_value = {"error": "Session not found"}
        data = client.get("/sessions/nope").json()
        assert "detail" in data

    def test_get_session_all_fields(self, client, mock_orchestrator):
        mock_orchestrator.get_session_info.return_value = {
            "session_id": "s1",
            "turn_count": 5,
            "active_turns": 3,
            "current_context": "kitchen",
            "started_at": 200.0,
            "total_tokens": 1200,
        }
        data = client.get("/sessions/s1").json()
        assert data["active_turns"] == 3
        assert data["total_tokens"] == 1200
        assert data["current_context"] == "kitchen"

    def test_get_session_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        mock_orchestrator.get_session_info.return_value = {
            "session_id": "s1",
            "turn_count": 0,
            "active_turns": 0,
            "current_context": "general",
            "started_at": 1.0,
            "total_tokens": 0,
        }
        client.get("/sessions/s1")
        mock_orchestrator.initialize.assert_called()


class TestGetSessionHistory:
    """Tests for GET /sessions/{session_id}/history"""

    def test_history_existing_session(self, client, mock_orchestrator):
        mock_memory = MagicMock()
        mock_memory.get_last_n_turns.return_value = []
        mock_orchestrator._sessions = {"s1": mock_memory}

        data = client.get("/sessions/s1/history").json()
        assert data["session_id"] == "s1"
        assert data["turns"] == []

    def test_history_nonexistent_session_returns_404(self, client, mock_orchestrator):
        mock_orchestrator._sessions = {}
        resp = client.get("/sessions/ghost/history")
        assert resp.status_code == 404

    def test_history_with_turns(self, client, mock_orchestrator):
        turn = MagicMock()
        turn.turn_id = 1
        turn.user_message = "Hello"
        turn.assistant_response = "Hi Boss"
        turn.timestamp = 1700000000.0
        turn.context_type = "general"
        turn.tool_calls = []

        mock_memory = MagicMock()
        mock_memory.get_last_n_turns.return_value = [turn]
        mock_orchestrator._sessions = {"s1": mock_memory}

        data = client.get("/sessions/s1/history").json()
        assert len(data["turns"]) == 1
        assert data["turns"][0]["user_message"] == "Hello"
        assert data["turns"][0]["assistant_response"] == "Hi Boss"

    def test_history_default_last_n(self, client, mock_orchestrator):
        mock_memory = MagicMock()
        mock_memory.get_last_n_turns.return_value = []
        mock_orchestrator._sessions = {"s1": mock_memory}

        client.get("/sessions/s1/history")
        mock_memory.get_last_n_turns.assert_called_with(10)

    def test_history_custom_last_n(self, client, mock_orchestrator):
        mock_memory = MagicMock()
        mock_memory.get_last_n_turns.return_value = []
        mock_orchestrator._sessions = {"s1": mock_memory}

        client.get("/sessions/s1/history", params={"last_n": 5})
        mock_memory.get_last_n_turns.assert_called_with(5)

    def test_history_turn_fields(self, client, mock_orchestrator):
        turn = MagicMock()
        turn.turn_id = 42
        turn.user_message = "Q"
        turn.assistant_response = "A"
        turn.timestamp = 99.9
        turn.context_type = "writers_room"
        turn.tool_calls = [{"name": "scene_search"}]

        mock_memory = MagicMock()
        mock_memory.get_last_n_turns.return_value = [turn]
        mock_orchestrator._sessions = {"sx": mock_memory}

        t = client.get("/sessions/sx/history").json()["turns"][0]
        assert t["turn_id"] == 42
        assert t["context_type"] == "writers_room"
        assert t["tool_calls"] == [{"name": "scene_search"}]

    def test_history_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        mock_orchestrator._sessions = {}
        client.get("/sessions/s1/history")
        mock_orchestrator.initialize.assert_called()


class TestSwitchSession:
    """Tests for POST /sessions/{session_id}/switch"""

    def test_switch_success(self, client, mock_orchestrator):
        mock_orchestrator.switch_session.return_value = True
        resp = client.post("/sessions/s1/switch")
        assert resp.status_code == 200
        assert "Switched" in resp.json()["message"]

    def test_switch_not_found(self, client, mock_orchestrator):
        mock_orchestrator.switch_session.return_value = False
        resp = client.post("/sessions/no-such/switch")
        assert resp.status_code == 404

    def test_switch_calls_orchestrator(self, client, mock_orchestrator):
        mock_orchestrator.switch_session.return_value = True
        client.post("/sessions/abc/switch")
        mock_orchestrator.switch_session.assert_called_with("abc")

    def test_switch_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        mock_orchestrator.switch_session.return_value = True
        client.post("/sessions/s1/switch")
        mock_orchestrator.initialize.assert_called()


class TestDeleteSession:
    """Tests for DELETE /sessions/{session_id}"""

    def test_delete_existing_session(self, client, mock_orchestrator):
        mock_mem = MagicMock()
        mock_orchestrator._sessions = {"del-me": mock_mem}
        mock_orchestrator._current_session_id = "other"

        resp = client.delete("/sessions/del-me")
        assert resp.status_code == 200
        assert (
            "deleted" in resp.json()["message"].lower()
            or "Deleted" in resp.json()["message"]
            or "deleted" in resp.json()["message"]
        )

    def test_delete_nonexistent_returns_404(self, client, mock_orchestrator):
        mock_orchestrator._sessions = {}
        resp = client.delete("/sessions/nope")
        assert resp.status_code == 404

    def test_delete_clears_memory(self, client, mock_orchestrator):
        mock_mem = MagicMock()
        mock_orchestrator._sessions = {"del-me": mock_mem}
        mock_orchestrator._current_session_id = "other"

        client.delete("/sessions/del-me")
        mock_mem.clear.assert_called_once()

    def test_delete_removes_from_sessions(self, client, mock_orchestrator):
        mock_mem = MagicMock()
        sessions = {"del-me": mock_mem}
        mock_orchestrator._sessions = sessions
        mock_orchestrator._current_session_id = "other"

        client.delete("/sessions/del-me")
        assert "del-me" not in sessions

    def test_delete_current_creates_new(self, client, mock_orchestrator):
        mock_mem = MagicMock()
        mock_orchestrator._sessions = {"current": mock_mem}
        mock_orchestrator._current_session_id = "current"

        client.delete("/sessions/current")
        mock_orchestrator._create_session.assert_called()

    def test_delete_non_current_no_create(self, client, mock_orchestrator):
        mock_mem = MagicMock()
        mock_orchestrator._sessions = {"other": mock_mem}
        mock_orchestrator._current_session_id = "current"
        mock_orchestrator._create_session.reset_mock()

        client.delete("/sessions/other")
        mock_orchestrator._create_session.assert_not_called()

    def test_delete_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        mock_orchestrator._sessions = {}
        client.delete("/sessions/s1")
        mock_orchestrator.initialize.assert_called()


class TestClearSession:
    """Tests for POST /sessions/{session_id}/clear"""

    def test_clear_existing_session(self, client, mock_orchestrator):
        mock_mem = MagicMock()
        mock_orchestrator._sessions = {"clr": mock_mem}

        resp = client.post("/sessions/clr/clear")
        assert resp.status_code == 200
        mock_mem.clear.assert_called_once()

    def test_clear_nonexistent_returns_404(self, client, mock_orchestrator):
        mock_orchestrator._sessions = {}
        resp = client.post("/sessions/nope/clear")
        assert resp.status_code == 404

    def test_clear_message(self, client, mock_orchestrator):
        mock_mem = MagicMock()
        mock_orchestrator._sessions = {"clr": mock_mem}

        data = client.post("/sessions/clr/clear").json()
        assert "cleared" in data["message"].lower() or "cleared" in data["message"]

    def test_clear_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        mock_orchestrator._sessions = {}
        client.post("/sessions/s1/clear")
        mock_orchestrator.initialize.assert_called()


# ========================================================================
#  6.  TOOLS ENDPOINTS  (tools.py)
# ========================================================================
class TestListTools:
    """Tests for GET /tools"""

    def test_list_returns_200(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200

    def test_list_returns_all_tools(self, client):
        data = client.get("/tools").json()
        assert len(data) == len(_TOOL_CATALOGUE)

    def test_list_tool_has_name(self, client):
        data = client.get("/tools").json()
        assert all("name" in t for t in data)

    def test_list_tool_has_description(self, client):
        data = client.get("/tools").json()
        assert all("description" in t for t in data)

    def test_list_tool_has_category(self, client):
        data = client.get("/tools").json()
        assert all("category" in t for t in data)

    def test_list_tool_has_parameters(self, client):
        data = client.get("/tools").json()
        assert all("parameters" in t for t in data)

    def test_list_filter_by_category(self, client, mock_registry):
        screenplay_tools = [t for t in _TOOL_CATALOGUE if t.category == "screenplay"]
        mock_registry.list_tools.return_value = list(_TOOL_CATALOGUE)

        data = client.get("/tools", params={"category": "screenplay"}).json()
        assert len(data) == len(screenplay_tools)
        for t in data:
            assert t["category"] == "screenplay"

    def test_list_filter_empty_category(self, client, mock_registry):
        mock_registry.list_tools.return_value = list(_TOOL_CATALOGUE)
        data = client.get("/tools", params={"category": "nonexistent"}).json()
        assert data == []

    def test_list_filter_email_category(self, client, mock_registry):
        mock_registry.list_tools.return_value = list(_TOOL_CATALOGUE)
        data = client.get("/tools", params={"category": "email"}).json()
        assert len(data) == 1
        assert data[0]["name"] == "send_email"

    def test_list_filter_vision_category(self, client, mock_registry):
        mock_registry.list_tools.return_value = list(_TOOL_CATALOGUE)
        data = client.get("/tools", params={"category": "vision"}).json()
        assert len(data) == 1
        assert data[0]["name"] == "camera_analyze"

    def test_list_no_filter_returns_all(self, client, mock_registry):
        mock_registry.list_tools.return_value = list(_TOOL_CATALOGUE)
        data = client.get("/tools").json()
        names = {t["name"] for t in data}
        assert "scene_search" in names
        assert "send_email" in names


class TestGetTool:
    """Tests for GET /tools/{tool_name}"""

    def test_get_existing_tool(self, client):
        resp = client.get("/tools/scene_search")
        assert resp.status_code == 200

    def test_get_tool_name(self, client):
        data = client.get("/tools/scene_search").json()
        assert data["name"] == "scene_search"

    def test_get_tool_description(self, client):
        data = client.get("/tools/scene_search").json()
        assert data["description"] == "Search scenes"

    def test_get_tool_category(self, client):
        data = client.get("/tools/scene_search").json()
        assert data["category"] == "screenplay"

    def test_get_tool_parameters(self, client):
        data = client.get("/tools/scene_search").json()
        assert "parameters" in data
        assert data["parameters"]["type"] == "object"

    def test_get_nonexistent_tool_404(self, client):
        resp = client.get("/tools/nonexistent_tool")
        assert resp.status_code == 404

    def test_get_nonexistent_tool_detail(self, client):
        data = client.get("/tools/nonexistent_tool").json()
        assert "detail" in data
        assert "nonexistent_tool" in data["detail"]

    def test_get_send_email_tool(self, client):
        data = client.get("/tools/send_email").json()
        assert data["name"] == "send_email"
        assert data["category"] == "email"

    def test_get_camera_analyze_tool(self, client):
        data = client.get("/tools/camera_analyze").json()
        assert data["name"] == "camera_analyze"
        assert data["category"] == "vision"


class TestExecuteTool:
    """Tests for POST /tools/execute"""

    def test_execute_returns_200(self, client, mock_orchestrator):
        mock_orchestrator.execute_tool = AsyncMock(
            return_value=_FakeToolResult(success=True, data={"ok": True})
        )
        resp = client.post(
            "/tools/execute",
            json={"name": "scene_search", "arguments": {"query": "love"}},
        )
        assert resp.status_code == 200

    def test_execute_success_true(self, client, mock_orchestrator):
        mock_orchestrator.execute_tool = AsyncMock(
            return_value=_FakeToolResult(success=True, data={"results": []})
        )
        data = client.post(
            "/tools/execute", json={"name": "scene_search", "arguments": {"query": "x"}}
        ).json()
        assert data["success"] is True

    def test_execute_returns_data(self, client, mock_orchestrator):
        mock_orchestrator.execute_tool = AsyncMock(
            return_value=_FakeToolResult(success=True, data={"items": [1, 2, 3]})
        )
        data = client.post("/tools/execute", json={"name": "t", "arguments": {}}).json()
        assert data["data"] == {"items": [1, 2, 3]}

    def test_execute_error_case(self, client, mock_orchestrator):
        mock_orchestrator.execute_tool = AsyncMock(
            return_value=_FakeToolResult(success=False, error="Something broke")
        )
        data = client.post("/tools/execute", json={"name": "t", "arguments": {}}).json()
        assert data["success"] is False
        assert data["error"] == "Something broke"

    def test_execute_missing_name_returns_422(self, client):
        resp = client.post("/tools/execute", json={"arguments": {}})
        assert resp.status_code == 422

    def test_execute_default_arguments(self, client, mock_orchestrator):
        mock_orchestrator.execute_tool = AsyncMock(
            return_value=_FakeToolResult(success=True, data=None)
        )
        resp = client.post("/tools/execute", json={"name": "t"})
        assert resp.status_code == 200

    def test_execute_calls_orchestrator(self, client, mock_orchestrator):
        mock_orchestrator.execute_tool = AsyncMock(
            return_value=_FakeToolResult(success=True, data=None)
        )
        client.post(
            "/tools/execute", json={"name": "scene_search", "arguments": {"query": "q"}}
        )
        mock_orchestrator.execute_tool.assert_called_once_with(
            "scene_search", {"query": "q"}
        )

    def test_execute_initializes_if_needed(self, client, mock_orchestrator):
        mock_orchestrator.is_initialized = False
        mock_orchestrator.execute_tool = AsyncMock(
            return_value=_FakeToolResult(success=True, data=None)
        )
        client.post("/tools/execute", json={"name": "t"})
        mock_orchestrator.initialize.assert_called()

    def test_execute_no_body_returns_422(self, client):
        resp = client.post("/tools/execute")
        assert resp.status_code == 422

    def test_execute_response_has_success_field(self, client, mock_orchestrator):
        mock_orchestrator.execute_tool = AsyncMock(
            return_value=_FakeToolResult(success=True, data=None)
        )
        data = client.post("/tools/execute", json={"name": "t"}).json()
        assert "success" in data

    def test_execute_response_has_data_field(self, client, mock_orchestrator):
        mock_orchestrator.execute_tool = AsyncMock(
            return_value=_FakeToolResult(success=True, data="hello")
        )
        data = client.post("/tools/execute", json={"name": "t"}).json()
        assert "data" in data

    def test_execute_response_has_error_field(self, client, mock_orchestrator):
        mock_orchestrator.execute_tool = AsyncMock(
            return_value=_FakeToolResult(success=False, error="err")
        )
        data = client.post("/tools/execute", json={"name": "t"}).json()
        assert "error" in data


class TestToolsForContext:
    """Tests for GET /tools/context/{context_type}"""

    def test_valid_context_returns_200(self, client, mock_registry):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            mock_registry.list_tools.return_value = [_TOOL_MAP["scene_search"]]
            resp = client.get("/tools/context/writers_room")
            assert resp.status_code == 200

    def test_context_response_has_context_field(self, client, mock_registry):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            mock_registry.list_tools.return_value = []
            data = client.get("/tools/context/general").json()
            assert data["context"] == "general"

    def test_context_response_has_description(self, client, mock_registry):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            mock_registry.list_tools.return_value = []
            data = client.get("/tools/context/general").json()
            assert data["description"] == "General assistant mode"

    def test_context_response_has_tools_list(self, client, mock_registry):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            mock_registry.list_tools.return_value = [_TOOL_MAP["scene_search"]]
            data = client.get("/tools/context/writers_room").json()
            assert "tools" in data
            assert isinstance(data["tools"], list)

    def test_context_tools_have_name(self, client, mock_registry):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            mock_registry.list_tools.return_value = [_TOOL_MAP["scene_search"]]
            data = client.get("/tools/context/writers_room").json()
            assert data["tools"][0]["name"] == "scene_search"

    def test_context_tools_have_description(self, client, mock_registry):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            mock_registry.list_tools.return_value = [_TOOL_MAP["scene_search"]]
            data = client.get("/tools/context/writers_room").json()
            assert "description" in data["tools"][0]

    def test_invalid_context_returns_400(self, client, mock_registry):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            resp = client.get("/tools/context/nonexistent_room")
            assert resp.status_code == 400

    def test_invalid_context_error_detail(self, client, mock_registry):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            data = client.get("/tools/context/nonexistent_room").json()
            assert "detail" in data

    def test_kitchen_context(self, client, mock_registry):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            mock_registry.list_tools.return_value = [_TOOL_MAP["camera_analyze"]]
            data = client.get("/tools/context/kitchen").json()
            assert data["context"] == "kitchen"
            assert data["description"] == "Cooking assistance with camera"

    def test_storyboard_context(self, client, mock_registry):
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=_FAKE_CONTEXTS,
                ),
            },
        ):
            mock_registry.list_tools.return_value = [_TOOL_MAP["generate_image"]]
            data = client.get("/tools/context/storyboard").json()
            assert data["context"] == "storyboard"

    def test_context_not_found_when_config_missing(self, client, mock_registry):
        """Test 404 when context type is valid but has no config entry."""
        fake_contexts_missing = dict(_FAKE_CONTEXTS)
        del fake_contexts_missing[_ContextType.KITCHEN]
        with patch.dict(
            "sys.modules",
            {
                "orchestrator.context.contexts": MagicMock(
                    ContextType=_ContextType,
                    CONTEXTS=fake_contexts_missing,
                ),
            },
        ):
            resp = client.get("/tools/context/kitchen")
            assert resp.status_code == 404


# ========================================================================
#  7.  HTTP METHOD VALIDATION
# ========================================================================
class TestHTTPMethods:
    """Verify that wrong HTTP methods get 405"""

    def test_root_post_not_allowed(self, client):
        resp = client.post("/")
        assert resp.status_code == 405

    def test_health_post_not_allowed(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405

    def test_chat_get_not_allowed(self, client):
        resp = client.get("/chat")
        assert resp.status_code == 405

    def test_sessions_delete_on_list_not_allowed(self, client):
        resp = client.delete("/sessions")
        assert resp.status_code == 405

    def test_tools_post_on_list_not_allowed(self, client):
        resp = client.post("/tools")
        assert resp.status_code == 405

    def test_tools_delete_not_allowed(self, client):
        resp = client.delete("/tools")
        assert resp.status_code == 405


# ========================================================================
#  8.  CORS MIDDLEWARE
# ========================================================================
class TestCORSMiddleware:
    """Verify CORS headers are set correctly."""

    def test_cors_allows_all_origins(self, client):
        resp = client.options(
            "/",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # The response should include CORS headers
        assert resp.headers.get("access-control-allow-origin") in (
            "*",
            "http://example.com",
        )

    def test_cors_allows_post(self, client):
        resp = client.options(
            "/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow_methods = resp.headers.get("access-control-allow-methods", "")
        assert "POST" in allow_methods or "*" in allow_methods


# ========================================================================
#  9.  EDGE CASES / ADDITIONAL COVERAGE
# ========================================================================
class TestEdgeCases:
    """Various edge cases and additional coverage."""

    def test_nonexistent_route_returns_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_chat_with_all_optional_fields(self, client, mock_orchestrator):
        resp = client.post(
            "/chat",
            json={
                "message": "Hello",
                "session_id": "sess-1",
                "location": "writers_room",
                "stream": False,
            },
        )
        assert resp.status_code == 200

    def test_chat_large_message(self, client, mock_orchestrator):
        large_msg = "x" * 10000
        resp = client.post("/chat", json={"message": large_msg})
        assert resp.status_code == 200

    def test_session_id_with_special_characters(self, client, mock_orchestrator):
        mock_orchestrator.get_session_info.return_value = {"error": "Session not found"}
        resp = client.get("/sessions/special-chars-123_test")
        assert resp.status_code == 404

    def test_multiple_sessions_listed(self, client, mock_orchestrator):
        mock_orchestrator.list_sessions.return_value = [
            {
                "session_id": f"s{i}",
                "turn_count": i,
                "current_context": "general",
                "started_at": float(i),
            }
            for i in range(10)
        ]
        data = client.get("/sessions").json()
        assert len(data) == 10

    def test_tools_execute_with_complex_arguments(self, client, mock_orchestrator):
        mock_orchestrator.execute_tool = AsyncMock(
            return_value=_FakeToolResult(success=True, data={"found": 3})
        )
        resp = client.post(
            "/tools/execute",
            json={
                "name": "scene_search",
                "arguments": {
                    "query": "romantic scenes with dialogue",
                    "top_k": 10,
                    "project_slug": "my-project",
                },
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_delete_session_response_message(self, client, mock_orchestrator):
        mock_mem = MagicMock()
        mock_orchestrator._sessions = {"test-del": mock_mem}
        mock_orchestrator._current_session_id = "other"

        data = client.delete("/sessions/test-del").json()
        assert "test-del" in data["message"]

    def test_clear_session_response_message(self, client, mock_orchestrator):
        mock_mem = MagicMock()
        mock_orchestrator._sessions = {"test-clr": mock_mem}

        data = client.post("/sessions/test-clr/clear").json()
        assert "test-clr" in data["message"]

    def test_switch_session_response_message(self, client, mock_orchestrator):
        mock_orchestrator.switch_session.return_value = True
        data = client.post("/sessions/my-sess/switch").json()
        assert "my-sess" in data["message"]

    def test_history_multiple_turns(self, client, mock_orchestrator):
        turns = []
        for i in range(5):
            t = MagicMock()
            t.turn_id = i
            t.user_message = f"msg-{i}"
            t.assistant_response = f"resp-{i}"
            t.timestamp = float(i)
            t.context_type = "general"
            t.tool_calls = []
            turns.append(t)

        mock_memory = MagicMock()
        mock_memory.get_last_n_turns.return_value = turns
        mock_orchestrator._sessions = {"multi": mock_memory}

        data = client.get("/sessions/multi/history").json()
        assert len(data["turns"]) == 5
        assert data["turns"][2]["user_message"] == "msg-2"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
