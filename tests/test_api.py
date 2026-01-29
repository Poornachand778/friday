"""
Tests for Friday Orchestrator API
=================================

Run with: pytest tests/test_api.py -v

Note: Chat endpoint tests are skipped by default as they require LLM.
Run with LLM: pytest tests/test_api.py -v --run-llm-tests
"""

import pytest
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from fastapi.testclient import TestClient


# Check if we should run LLM tests
def pytest_configure(config):
    config.addinivalue_line("markers", "llm: tests that require LLM backend")


@pytest.fixture(scope="module")
def app():
    """Create test app"""
    from orchestrator.main import app

    return app


@pytest.fixture(scope="module")
def client(app):
    """Create test client"""
    return TestClient(app)


# Mark for tests requiring LLM
llm_required = pytest.mark.skip(
    reason="Requires LLM backend - run with real backend to test"
)


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root(self, client):
        """Test root returns app info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Friday AI Orchestrator"
        assert data["status"] == "running"


class TestHealthEndpoint:
    """Test health endpoint"""

    def test_health(self, client):
        """Test health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "orchestrator" in data
        assert "initialized" in data


@llm_required
class TestChatEndpoint:
    """Test chat endpoint - requires LLM backend"""

    def test_chat_basic(self, client):
        """Test basic chat"""
        response = client.post(
            "/chat",
            json={
                "message": "Show me romantic scenes",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "context" in data
        assert "turn_id" in data

    def test_chat_with_location(self, client):
        """Test chat with location hint"""
        response = client.post(
            "/chat",
            json={
                "message": "Show me scene 5",
                "location": "writers_room",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["context"] in ["writers_room", "general"]

    def test_chat_with_session(self, client):
        """Test chat with session ID"""
        response = client.post(
            "/chat",
            json={
                "message": "Hello Friday",
                "session_id": "test-session-1",
            },
        )
        assert response.status_code == 200

    def test_chat_empty_message(self, client):
        """Test chat with empty message"""
        response = client.post(
            "/chat",
            json={
                "message": "",
            },
        )
        # Should still work (empty is valid)
        assert response.status_code == 200


@llm_required
class TestVoiceEndpoint:
    """Test voice chat endpoint - requires LLM backend"""

    def test_voice_chat(self, client):
        """Test voice-specific endpoint"""
        response = client.post(
            "/chat/voice",
            params={
                "transcript": "Boss, what scenes do we have?",
                "location": "writers_room",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "context" in data


class TestToolsEndpoint:
    """Test tools endpoints"""

    def test_list_tools(self, client):
        """Test listing all tools"""
        response = client.get("/tools")
        assert response.status_code == 200
        tools = response.json()
        assert len(tools) > 0
        # Check tool structure
        assert "name" in tools[0]
        assert "description" in tools[0]

    def test_list_tools_by_category(self, client):
        """Test filtering tools by category"""
        response = client.get("/tools", params={"category": "screenplay"})
        assert response.status_code == 200
        tools = response.json()
        for tool in tools:
            assert tool["category"] == "screenplay"

    def test_get_tool(self, client):
        """Test getting specific tool"""
        response = client.get("/tools/scene_search")
        assert response.status_code == 200
        tool = response.json()
        assert tool["name"] == "scene_search"

    def test_get_unknown_tool(self, client):
        """Test getting unknown tool"""
        response = client.get("/tools/nonexistent")
        assert response.status_code == 404

    def test_tools_for_context(self, client):
        """Test getting tools for context"""
        response = client.get("/tools/context/writers_room")
        assert response.status_code == 200
        data = response.json()
        assert data["context"] == "writers_room"
        assert "tools" in data

    def test_tools_invalid_context(self, client):
        """Test invalid context"""
        response = client.get("/tools/context/invalid_room")
        assert response.status_code == 400


class TestSessionsEndpoint:
    """Test sessions endpoints"""

    def test_list_sessions(self, client):
        """Test listing sessions"""
        response = client.get("/sessions")
        assert response.status_code == 200
        sessions = response.json()
        assert isinstance(sessions, list)

    def test_create_session(self, client):
        """Test creating new session"""
        response = client.post("/sessions")
        assert response.status_code == 200
        session = response.json()
        assert "session_id" in session
        assert session["turn_count"] == 0

    def test_create_session_with_id(self, client):
        """Test creating session with custom ID"""
        response = client.post(
            "/sessions",
            json={
                "session_id": "my-custom-session",
            },
        )
        assert response.status_code == 200
        session = response.json()
        assert session["session_id"] == "my-custom-session"

    def test_get_session(self, client):
        """Test getting session info"""
        # First create a session
        create_response = client.post(
            "/sessions",
            json={
                "session_id": "test-get-session",
            },
        )
        assert create_response.status_code == 200

        # Then get it
        response = client.get("/sessions/test-get-session")
        assert response.status_code == 200
        session = response.json()
        assert session["session_id"] == "test-get-session"

    def test_get_unknown_session(self, client):
        """Test getting unknown session"""
        response = client.get("/sessions/nonexistent-session-xyz")
        assert response.status_code == 404

    def test_get_session_history(self, client):
        """Test getting session history (without chat)"""
        # Create session
        client.post("/sessions", json={"session_id": "test-history-2"})

        response = client.get("/sessions/test-history-2/history")
        assert response.status_code == 200
        data = response.json()
        assert "turns" in data
        assert data["turns"] == []  # Empty since no chat

    def test_clear_session(self, client):
        """Test clearing session (without chat)"""
        client.post("/sessions", json={"session_id": "test-clear-2"})

        response = client.post("/sessions/test-clear-2/clear")
        assert response.status_code == 200

        # Verify it's cleared
        info = client.get("/sessions/test-clear-2")
        assert info.json()["turn_count"] == 0


class TestContextEndpoint:
    """Test context endpoints"""

    def test_get_context(self, client):
        """Test getting current context"""
        response = client.get("/context")
        assert response.status_code == 200
        data = response.json()
        assert "current_context" in data
        assert "available_tools" in data

    def test_set_context(self, client):
        """Test setting context manually"""
        response = client.post("/context/writers_room")
        assert response.status_code == 200
        data = response.json()
        assert data["current_context"] == "writers_room"

    def test_set_invalid_context(self, client):
        """Test setting invalid context"""
        response = client.post("/context/invalid_context")
        assert response.status_code == 200  # Returns error in body
        data = response.json()
        assert "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
