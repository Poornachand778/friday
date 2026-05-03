"""
Tests for Local LLM Client
============================

Comprehensive tests for ChatMessage, ToolCall, ChatResponse,
LLMClient, and SyncLLMClient.

Run with: pytest tests/test_local_llm.py -v
"""

import asyncio
import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from orchestrator.inference.local_llm import (
    ChatMessage,
    ChatResponse,
    LLMClient,
    SyncLLMClient,
    ToolCall,
)
from orchestrator.config import LLMConfig


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def llm_config():
    return LLMConfig(
        backend="vllm",
        model_name="test-model",
        base_url="http://localhost:8000/v1",
        api_key="test-key",
        max_tokens=512,
        temperature=0.5,
        top_p=0.95,
    )


@pytest.fixture
def openai_config():
    return LLMConfig(
        backend="openai",
        model_name="gpt-4",
        base_url="https://api.openai.com/v1",
        api_key="sk-test-key",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )


@pytest.fixture
def anthropic_config():
    return LLMConfig(
        backend="anthropic",
        model_name="claude-3-sonnet",
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-test-key",
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )


@pytest.fixture
def sagemaker_config():
    return LLMConfig(
        backend="sagemaker",
        model_name="llama-endpoint",
        base_url="friday-llama-endpoint",
        api_key="not-needed",
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )


@pytest.fixture
def llamacpp_config():
    return LLMConfig(
        backend="llamacpp",
        model_name="llama-local",
        base_url="http://localhost:8080/v1",
        api_key="not-needed",
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )


@pytest.fixture
def client(llm_config):
    return LLMClient(config=llm_config)


@pytest.fixture
def sample_messages():
    return [
        ChatMessage(role="system", content="You are Friday."),
        ChatMessage(role="user", content="Hello Boss"),
    ]


@pytest.fixture
def sample_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]


@pytest.fixture
def openai_api_response():
    """Standard OpenAI-format API response."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello Boss! How can I help?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8},
    }


@pytest.fixture
def openai_tool_call_response():
    """OpenAI-format API response with tool calls."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Hyderabad"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 20},
    }


@pytest.fixture
def anthropic_api_response():
    """Standard Anthropic API response."""
    return {
        "content": [
            {"type": "text", "text": "Hello Boss!"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }


def _mock_http_response(json_data, status_code=200):
    """Create a mock httpx response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# =========================================================================
# 1. ChatMessage Tests
# =========================================================================


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_creation_basic(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_creation_defaults(self):
        msg = ChatMessage(role="assistant", content="Hi")
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_creation_with_all_fields(self):
        msg = ChatMessage(
            role="tool",
            content="result data",
            name="get_weather",
            tool_calls=[{"id": "1", "function": {"name": "f"}}],
            tool_call_id="call_123",
        )
        assert msg.role == "tool"
        assert msg.content == "result data"
        assert msg.name == "get_weather"
        assert msg.tool_calls is not None
        assert msg.tool_call_id == "call_123"

    def test_to_dict_basic(self):
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_to_dict_excludes_none_name(self):
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_dict()
        assert "name" not in d

    def test_to_dict_excludes_none_tool_calls(self):
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_dict()
        assert "tool_calls" not in d

    def test_to_dict_excludes_none_tool_call_id(self):
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_dict()
        assert "tool_call_id" not in d

    def test_to_dict_with_name(self):
        msg = ChatMessage(role="tool", content="data", name="my_tool")
        d = msg.to_dict()
        assert d["name"] == "my_tool"
        assert d["role"] == "tool"
        assert d["content"] == "data"

    def test_to_dict_with_tool_calls(self):
        tc = [{"id": "tc1", "function": {"name": "fn", "arguments": "{}"}}]
        msg = ChatMessage(role="assistant", content="", tool_calls=tc)
        d = msg.to_dict()
        assert d["tool_calls"] == tc

    def test_to_dict_with_tool_call_id(self):
        msg = ChatMessage(role="tool", content="result", tool_call_id="call_456")
        d = msg.to_dict()
        assert d["tool_call_id"] == "call_456"

    def test_to_dict_with_all_optional_fields(self):
        tc = [{"id": "tc1"}]
        msg = ChatMessage(
            role="tool",
            content="result",
            name="fn",
            tool_calls=tc,
            tool_call_id="call_789",
        )
        d = msg.to_dict()
        assert d["name"] == "fn"
        assert d["tool_calls"] == tc
        assert d["tool_call_id"] == "call_789"

    def test_to_dict_empty_content(self):
        msg = ChatMessage(role="assistant", content="")
        d = msg.to_dict()
        assert d["content"] == ""

    def test_system_role(self):
        msg = ChatMessage(role="system", content="You are Friday.")
        assert msg.role == "system"
        d = msg.to_dict()
        assert d["role"] == "system"

    def test_empty_tool_calls_list_included(self):
        """An empty list is falsy, so tool_calls should NOT appear in dict."""
        msg = ChatMessage(role="assistant", content="hi", tool_calls=[])
        d = msg.to_dict()
        assert "tool_calls" not in d

    def test_empty_string_name_not_included(self):
        """An empty string is falsy, so name should NOT appear in dict."""
        msg = ChatMessage(role="assistant", content="hi", name="")
        d = msg.to_dict()
        assert "name" not in d


# =========================================================================
# 2. ToolCall Tests
# =========================================================================


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_creation(self):
        tc = ToolCall(id="call_1", name="get_weather", arguments={"location": "NYC"})
        assert tc.id == "call_1"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "NYC"}

    def test_fields_types(self):
        tc = ToolCall(id="id1", name="fn", arguments={})
        assert isinstance(tc.id, str)
        assert isinstance(tc.name, str)
        assert isinstance(tc.arguments, dict)

    def test_complex_arguments(self):
        args = {"query": "test", "limit": 10, "nested": {"key": "value"}}
        tc = ToolCall(id="id2", name="search", arguments=args)
        assert tc.arguments["nested"]["key"] == "value"
        assert tc.arguments["limit"] == 10

    def test_empty_arguments(self):
        tc = ToolCall(id="id3", name="no_args_tool", arguments={})
        assert tc.arguments == {}


# =========================================================================
# 3. ChatResponse Tests
# =========================================================================


class TestChatResponse:
    """Tests for ChatResponse dataclass."""

    def test_creation_basic(self):
        resp = ChatResponse(content="Hello")
        assert resp.content == "Hello"

    def test_defaults(self):
        resp = ChatResponse(content="Hi")
        assert resp.role == "assistant"
        assert resp.tool_calls == []
        assert resp.finish_reason == "stop"
        assert resp.usage == {}

    def test_custom_fields(self):
        tc = [ToolCall(id="1", name="fn", arguments={})]
        resp = ChatResponse(
            content="",
            role="assistant",
            tool_calls=tc,
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert resp.finish_reason == "tool_calls"
        assert resp.usage["prompt_tokens"] == 10

    def test_has_tool_calls_false_by_default(self):
        resp = ChatResponse(content="Hello")
        assert resp.has_tool_calls is False

    def test_has_tool_calls_true(self):
        tc = [ToolCall(id="1", name="fn", arguments={})]
        resp = ChatResponse(content="", tool_calls=tc)
        assert resp.has_tool_calls is True

    def test_has_tool_calls_multiple(self):
        tcs = [
            ToolCall(id="1", name="fn1", arguments={}),
            ToolCall(id="2", name="fn2", arguments={"x": 1}),
        ]
        resp = ChatResponse(content="", tool_calls=tcs)
        assert resp.has_tool_calls is True
        assert len(resp.tool_calls) == 2

    def test_empty_content(self):
        resp = ChatResponse(content="")
        assert resp.content == ""

    def test_none_content(self):
        resp = ChatResponse(content=None)
        assert resp.content is None


# =========================================================================
# 4. LLMClient Initialization Tests
# =========================================================================


class TestLLMClientInit:
    """Tests for LLMClient.__init__."""

    def test_init_with_custom_config(self, llm_config):
        c = LLMClient(config=llm_config)
        assert c.config is llm_config
        assert c.config.backend == "vllm"
        assert c._client is None

    def test_init_default_config_via_get_config(self, llm_config):
        with patch("orchestrator.inference.local_llm.get_config") as mock_get:
            mock_cfg = MagicMock()
            mock_cfg.llm = llm_config
            mock_get.return_value = mock_cfg
            c = LLMClient()
            assert c.config is llm_config
            mock_get.assert_called_once()

    def test_init_stores_none_client(self, llm_config):
        c = LLMClient(config=llm_config)
        assert c._client is None


# =========================================================================
# 5. _get_client Tests
# =========================================================================


class TestGetClient:
    """Tests for LLMClient._get_client."""

    @pytest.mark.asyncio
    async def test_creates_client_on_first_call(self, client):
        assert client._client is None
        http_client = await client._get_client()
        assert http_client is not None
        assert client._client is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_returns_same_client_on_subsequent_calls(self, client):
        c1 = await client._get_client()
        c2 = await client._get_client()
        assert c1 is c2
        await client.close()

    @pytest.mark.asyncio
    async def test_creates_httpx_async_client(self, client):
        import httpx

        http_client = await client._get_client()
        assert isinstance(http_client, httpx.AsyncClient)
        await client.close()


# =========================================================================
# 6. Chat Routing Tests
# =========================================================================


class TestChatRouting:
    """Tests for LLMClient.chat routing to correct backend."""

    @pytest.mark.asyncio
    async def test_routes_to_vllm(self, llm_config, sample_messages):
        c = LLMClient(config=llm_config)
        with patch.object(c, "_chat_vllm", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResponse(content="ok")
            await c.chat(sample_messages)
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_llamacpp(self, llamacpp_config, sample_messages):
        c = LLMClient(config=llamacpp_config)
        with patch.object(c, "_chat_llamacpp", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResponse(content="ok")
            await c.chat(sample_messages)
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_openai(self, openai_config, sample_messages):
        c = LLMClient(config=openai_config)
        with patch.object(c, "_chat_openai", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResponse(content="ok")
            await c.chat(sample_messages)
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_anthropic(self, anthropic_config, sample_messages):
        c = LLMClient(config=anthropic_config)
        with patch.object(c, "_chat_anthropic", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResponse(content="ok")
            await c.chat(sample_messages)
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_routes_to_sagemaker(self, sagemaker_config, sample_messages):
        c = LLMClient(config=sagemaker_config)
        with patch.object(c, "_chat_sagemaker", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResponse(content="ok")
            await c.chat(sample_messages)
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_backend_raises_value_error(self, sample_messages):
        config = LLMConfig(backend="unknown_backend")
        c = LLMClient(config=config)
        with pytest.raises(ValueError, match="Unknown backend: unknown_backend"):
            await c.chat(sample_messages)

    @pytest.mark.asyncio
    async def test_routing_passes_tools(
        self, llm_config, sample_messages, sample_tools
    ):
        c = LLMClient(config=llm_config)
        with patch.object(c, "_chat_vllm", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResponse(content="ok")
            await c.chat(sample_messages, tools=sample_tools)
            args, kwargs = mock.call_args
            assert args[1] == sample_tools

    @pytest.mark.asyncio
    async def test_routing_passes_temperature(self, llm_config, sample_messages):
        c = LLMClient(config=llm_config)
        with patch.object(c, "_chat_vllm", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResponse(content="ok")
            await c.chat(sample_messages, temperature=0.9)
            args, kwargs = mock.call_args
            assert args[2] == 0.9

    @pytest.mark.asyncio
    async def test_routing_passes_max_tokens(self, llm_config, sample_messages):
        c = LLMClient(config=llm_config)
        with patch.object(c, "_chat_vllm", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResponse(content="ok")
            await c.chat(sample_messages, max_tokens=2048)
            args, kwargs = mock.call_args
            assert args[3] == 2048

    @pytest.mark.asyncio
    async def test_routing_passes_stream(self, llm_config, sample_messages):
        c = LLMClient(config=llm_config)
        with patch.object(c, "_chat_vllm", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResponse(content="ok")
            await c.chat(sample_messages, stream=True)
            args, kwargs = mock.call_args
            assert args[4] is True


# =========================================================================
# 7. _chat_vllm Tests
# =========================================================================


class TestChatVLLM:
    """Tests for LLMClient._chat_vllm."""

    @pytest.mark.asyncio
    async def test_correct_payload_construction(
        self, llm_config, sample_messages, openai_api_response
    ):
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_vllm(sample_messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 512
        assert payload["top_p"] == 0.95
        assert payload["stream"] is False
        assert len(payload["messages"]) == 2

    @pytest.mark.asyncio
    async def test_with_tools_in_payload(
        self, llm_config, sample_messages, sample_tools, openai_api_response
    ):
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_vllm(sample_messages, sample_tools, None, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["tools"] == sample_tools
        assert payload["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_no_tools_not_in_payload(
        self, llm_config, sample_messages, openai_api_response
    ):
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_vllm(sample_messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "tools" not in payload
        assert "tool_choice" not in payload

    @pytest.mark.asyncio
    async def test_api_key_in_headers(
        self, llm_config, sample_messages, openai_api_response
    ):
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_vllm(sample_messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer test-key"

    @pytest.mark.asyncio
    async def test_api_key_not_needed_excluded(
        self, sample_messages, openai_api_response
    ):
        config = LLMConfig(
            backend="vllm",
            model_name="test-model",
            base_url="http://localhost:8000/v1",
            api_key="not-needed",
        )
        c = LLMClient(config=config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_vllm(sample_messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_empty_api_key_excluded(self, sample_messages, openai_api_response):
        config = LLMConfig(
            backend="vllm",
            model_name="test-model",
            base_url="http://localhost:8000/v1",
            api_key="",
        )
        c = LLMClient(config=config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_vllm(sample_messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_url_construction(
        self, llm_config, sample_messages, openai_api_response
    ):
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_vllm(sample_messages, None, None, None, False)

        call_args = mock_http.post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        assert url == "http://localhost:8000/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_custom_temperature_override(
        self, llm_config, sample_messages, openai_api_response
    ):
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_vllm(sample_messages, None, 0.9, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_custom_max_tokens_override(
        self, llm_config, sample_messages, openai_api_response
    ):
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_vllm(sample_messages, None, None, 2048, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["max_tokens"] == 2048

    @pytest.mark.asyncio
    async def test_returns_chat_response(
        self, llm_config, sample_messages, openai_api_response
    ):
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        result = await c._chat_vllm(sample_messages, None, None, None, False)
        assert isinstance(result, ChatResponse)
        assert result.content == "Hello Boss! How can I help?"

    @pytest.mark.asyncio
    async def test_stream_returns_generator(self, llm_config, sample_messages):
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        c._client = mock_http

        with patch.object(c, "_stream_response") as mock_stream:
            mock_stream.return_value = "stream_iter"
            result = await c._chat_vllm(sample_messages, None, None, None, True)
            assert result == "stream_iter"
            mock_stream.assert_called_once()


# =========================================================================
# 8. _chat_llamacpp Tests
# =========================================================================


class TestChatLlamaCpp:
    """Tests for LLMClient._chat_llamacpp."""

    @pytest.mark.asyncio
    async def test_delegates_to_vllm(self, llamacpp_config, sample_messages):
        c = LLMClient(config=llamacpp_config)
        with patch.object(c, "_chat_vllm", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResponse(content="llamacpp response")
            result = await c._chat_llamacpp(sample_messages, None, None, None, False)
            mock.assert_called_once_with(sample_messages, None, None, None, False)
            assert result.content == "llamacpp response"

    @pytest.mark.asyncio
    async def test_delegates_with_tools(
        self, llamacpp_config, sample_messages, sample_tools
    ):
        c = LLMClient(config=llamacpp_config)
        with patch.object(c, "_chat_vllm", new_callable=AsyncMock) as mock:
            mock.return_value = ChatResponse(content="ok")
            await c._chat_llamacpp(sample_messages, sample_tools, 0.8, 256, True)
            mock.assert_called_once_with(sample_messages, sample_tools, 0.8, 256, True)


# =========================================================================
# 9. _chat_openai Tests
# =========================================================================


class TestChatOpenAI:
    """Tests for LLMClient._chat_openai."""

    @pytest.mark.asyncio
    async def test_correct_url(
        self, openai_config, sample_messages, openai_api_response
    ):
        c = LLMClient(config=openai_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_openai(sample_messages, None, None, None, False)

        call_args = mock_http.post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        assert url == "https://api.openai.com/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_headers_with_api_key(
        self, openai_config, sample_messages, openai_api_response
    ):
        c = LLMClient(config=openai_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_openai(sample_messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_payload_model(
        self, openai_config, sample_messages, openai_api_response
    ):
        c = LLMClient(config=openai_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_openai(sample_messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_with_tools(
        self, openai_config, sample_messages, sample_tools, openai_api_response
    ):
        c = LLMClient(config=openai_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_openai(sample_messages, sample_tools, None, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["tools"] == sample_tools
        assert payload["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_stream_delegates_to_stream_response(
        self, openai_config, sample_messages
    ):
        c = LLMClient(config=openai_config)
        mock_http = AsyncMock()
        c._client = mock_http

        with patch.object(c, "_stream_response") as mock_stream:
            mock_stream.return_value = "openai_stream"
            result = await c._chat_openai(sample_messages, None, None, None, True)
            assert result == "openai_stream"

    @pytest.mark.asyncio
    async def test_no_top_p_in_openai_payload(
        self, openai_config, sample_messages, openai_api_response
    ):
        """OpenAI backend does not include top_p in payload (unlike vllm)."""
        c = LLMClient(config=openai_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_openai(sample_messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "top_p" not in payload


# =========================================================================
# 10. _chat_anthropic Tests
# =========================================================================


class TestChatAnthropic:
    """Tests for LLMClient._chat_anthropic."""

    @pytest.mark.asyncio
    async def test_system_message_extracted(
        self, anthropic_config, anthropic_api_response
    ):
        messages = [
            ChatMessage(role="system", content="You are Friday."),
            ChatMessage(role="user", content="Hello"),
        ]
        c = LLMClient(config=anthropic_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(anthropic_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_anthropic(messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["system"] == "You are Friday."
        # Only user message in messages list
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_no_system_message(self, anthropic_config, anthropic_api_response):
        messages = [
            ChatMessage(role="user", content="Hello"),
        ]
        c = LLMClient(config=anthropic_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(anthropic_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_anthropic(messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "system" not in payload

    @pytest.mark.asyncio
    async def test_tool_format_conversion(
        self, anthropic_config, sample_tools, anthropic_api_response
    ):
        messages = [ChatMessage(role="user", content="What's the weather?")]
        c = LLMClient(config=anthropic_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(anthropic_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_anthropic(messages, sample_tools, None, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert len(payload["tools"]) == 1
        tool = payload["tools"][0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get the current weather"
        assert "input_schema" in tool

    @pytest.mark.asyncio
    async def test_correct_url(self, anthropic_config, anthropic_api_response):
        messages = [ChatMessage(role="user", content="Hello")]
        c = LLMClient(config=anthropic_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(anthropic_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_anthropic(messages, None, None, None, False)

        call_args = mock_http.post.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        assert url == "https://api.anthropic.com/v1/messages"

    @pytest.mark.asyncio
    async def test_headers(self, anthropic_config, anthropic_api_response):
        messages = [ChatMessage(role="user", content="Hello")]
        c = LLMClient(config=anthropic_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(anthropic_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_anthropic(messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["x-api-key"] == "sk-ant-test-key"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_multiple_non_system_messages(
        self, anthropic_config, anthropic_api_response
    ):
        messages = [
            ChatMessage(role="system", content="System"),
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
            ChatMessage(role="user", content="How are you?"),
        ]
        c = LLMClient(config=anthropic_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(anthropic_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_anthropic(messages, None, None, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["system"] == "System"
        assert len(payload["messages"]) == 3

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens(
        self, anthropic_config, anthropic_api_response
    ):
        messages = [ChatMessage(role="user", content="Hello")]
        c = LLMClient(config=anthropic_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(anthropic_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_anthropic(messages, None, 0.3, 256, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["temperature"] == 0.3
        assert payload["max_tokens"] == 256


# =========================================================================
# 11. _chat_sagemaker Tests
# =========================================================================


class TestChatSageMaker:
    """Tests for LLMClient._chat_sagemaker."""

    def _make_mock_boto3(self, response_data):
        """Helper: create a mock boto3 module with invoke_endpoint returning response_data."""
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(response_data).encode("utf-8")

        mock_runtime = MagicMock()
        mock_runtime.invoke_endpoint.return_value = {"Body": mock_body}

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_runtime
        return mock_boto3, mock_runtime

    @pytest.mark.asyncio
    async def test_uses_boto3(self, sagemaker_config, sample_messages):
        c = LLMClient(config=sagemaker_config)
        mock_boto3, mock_runtime = self._make_mock_boto3(
            [{"generated_text": "Hello from SageMaker"}]
        )

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = await c._chat_sagemaker(sample_messages, None, None, None, False)

        mock_boto3.client.assert_called_once_with("sagemaker-runtime")
        assert result.content == "Hello from SageMaker"

    @pytest.mark.asyncio
    async def test_endpoint_name_from_base_url(self, sagemaker_config, sample_messages):
        c = LLMClient(config=sagemaker_config)
        mock_boto3, mock_runtime = self._make_mock_boto3([{"generated_text": "ok"}])

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            await c._chat_sagemaker(sample_messages, None, None, None, False)

        call_kwargs = mock_runtime.invoke_endpoint.call_args
        endpoint = call_kwargs.kwargs.get("EndpointName")
        assert endpoint == "friday-llama-endpoint"

    @pytest.mark.asyncio
    async def test_list_response_format(self, sagemaker_config, sample_messages):
        c = LLMClient(config=sagemaker_config)
        mock_boto3, _ = self._make_mock_boto3(
            [{"generated_text": "List format response"}]
        )

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = await c._chat_sagemaker(sample_messages, None, None, None, False)

        assert result.content == "List format response"

    @pytest.mark.asyncio
    async def test_dict_response_format(self, sagemaker_config, sample_messages):
        c = LLMClient(config=sagemaker_config)
        mock_boto3, _ = self._make_mock_boto3(
            {"generated_text": "Dict format response"}
        )

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = await c._chat_sagemaker(sample_messages, None, None, None, False)

        assert result.content == "Dict format response"

    @pytest.mark.asyncio
    async def test_other_response_format_str_fallback(
        self, sagemaker_config, sample_messages
    ):
        c = LLMClient(config=sagemaker_config)
        mock_boto3, _ = self._make_mock_boto3("plain string result")

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = await c._chat_sagemaker(sample_messages, None, None, None, False)

        assert result.content == "plain string result"

    @pytest.mark.asyncio
    async def test_returns_assistant_role(self, sagemaker_config, sample_messages):
        c = LLMClient(config=sagemaker_config)
        mock_boto3, _ = self._make_mock_boto3([{"generated_text": "ok"}])

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = await c._chat_sagemaker(sample_messages, None, None, None, False)

        assert result.role == "assistant"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_with_tools_in_parameters(
        self, sagemaker_config, sample_messages, sample_tools
    ):
        c = LLMClient(config=sagemaker_config)
        mock_boto3, mock_runtime = self._make_mock_boto3([{"generated_text": "ok"}])

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            await c._chat_sagemaker(sample_messages, sample_tools, None, None, False)

        call_kwargs = mock_runtime.invoke_endpoint.call_args
        body = json.loads(call_kwargs.kwargs.get("Body"))
        assert body["parameters"]["tools"] == sample_tools

    @pytest.mark.asyncio
    async def test_invocation_failure_raises(self, sagemaker_config, sample_messages):
        c = LLMClient(config=sagemaker_config)
        mock_boto3 = MagicMock()
        mock_runtime = MagicMock()
        mock_runtime.invoke_endpoint.side_effect = Exception("Endpoint error")
        mock_boto3.client.return_value = mock_runtime

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            with pytest.raises(Exception, match="Endpoint error"):
                await c._chat_sagemaker(sample_messages, None, None, None, False)

    @pytest.mark.asyncio
    async def test_payload_construction(self, sagemaker_config, sample_messages):
        c = LLMClient(config=sagemaker_config)
        mock_boto3, mock_runtime = self._make_mock_boto3([{"generated_text": "ok"}])

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            await c._chat_sagemaker(sample_messages, None, 0.8, 256, False)

        call_kwargs = mock_runtime.invoke_endpoint.call_args
        body = json.loads(call_kwargs.kwargs.get("Body"))
        assert body["parameters"]["max_new_tokens"] == 256
        assert body["parameters"]["temperature"] == 0.8
        assert body["parameters"]["do_sample"] is True
        assert body["parameters"]["return_full_text"] is False

    @pytest.mark.asyncio
    async def test_empty_list_response(self, sagemaker_config, sample_messages):
        """Empty list item should return empty generated_text via get default."""
        c = LLMClient(config=sagemaker_config)
        mock_boto3, _ = self._make_mock_boto3([{}])

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            result = await c._chat_sagemaker(sample_messages, None, None, None, False)

        assert result.content == ""


# =========================================================================
# 12. _parse_openai_response Tests
# =========================================================================


class TestParseOpenAIResponse:
    """Tests for LLMClient._parse_openai_response."""

    def test_basic_response(self, client, openai_api_response):
        result = client._parse_openai_response(openai_api_response)
        assert isinstance(result, ChatResponse)
        assert result.content == "Hello Boss! How can I help?"
        assert result.role == "assistant"
        assert result.finish_reason == "stop"
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 8}

    def test_with_tool_calls(self, client, openai_tool_call_response):
        result = client._parse_openai_response(openai_tool_call_response)
        assert result.has_tool_calls is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "call_abc123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Hyderabad"}

    def test_multiple_tool_calls(self, client):
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "fn1",
                                    "arguments": '{"a": 1}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "fn2",
                                    "arguments": '{"b": 2}',
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        result = client._parse_openai_response(data)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "fn1"
        assert result.tool_calls[1].name == "fn2"

    def test_invalid_json_in_arguments(self, client):
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {
                                    "name": "fn",
                                    "arguments": "not valid json{{{",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        result = client._parse_openai_response(data)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {}

    def test_empty_choices(self, client):
        data = {"choices": [{}], "usage": {}}
        result = client._parse_openai_response(data)
        assert result.content == ""
        assert result.role == "assistant"
        assert result.tool_calls == []

    def test_completely_empty_data(self, client):
        data = {}
        result = client._parse_openai_response(data)
        assert result.content == ""
        assert result.role == "assistant"

    def test_no_tool_calls_in_message(self, client, openai_api_response):
        result = client._parse_openai_response(openai_api_response)
        assert result.has_tool_calls is False
        assert result.tool_calls == []

    def test_usage_extraction(self, client):
        data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        result = client._parse_openai_response(data)
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["total_tokens"] == 150

    def test_none_content_in_response(self, client):
        data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
        result = client._parse_openai_response(data)
        # content defaults to "" via .get("content", "")
        # but None is returned directly since get finds it as None
        # Actually .get("content", "") returns None when content key is None
        assert result.content is None


# =========================================================================
# 13. _parse_anthropic_response Tests
# =========================================================================


class TestParseAnthropicResponse:
    """Tests for LLMClient._parse_anthropic_response."""

    def test_text_blocks(self, client):
        data = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "Boss!"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = client._parse_anthropic_response(data)
        assert result.content == "Hello Boss!"

    def test_tool_use_blocks(self, client):
        data = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"location": "Chennai"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        result = client._parse_anthropic_response(data)
        assert result.has_tool_calls is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.id == "toolu_123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "Chennai"}

    def test_mixed_text_and_tool_use(self, client):
        data = {
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "toolu_456",
                    "name": "get_weather",
                    "input": {"location": "Hyderabad"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 25},
        }
        result = client._parse_anthropic_response(data)
        assert result.content == "Let me check the weather."
        assert result.has_tool_calls is True
        assert result.tool_calls[0].name == "get_weather"

    def test_usage_parsing(self, client):
        data = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 42, "output_tokens": 17},
        }
        result = client._parse_anthropic_response(data)
        assert result.usage["prompt_tokens"] == 42
        assert result.usage["completion_tokens"] == 17

    def test_empty_content_blocks(self, client):
        data = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        result = client._parse_anthropic_response(data)
        assert result.content == ""
        assert result.tool_calls == []

    def test_finish_reason_mapping(self, client):
        data = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 5, "output_tokens": 100},
        }
        result = client._parse_anthropic_response(data)
        assert result.finish_reason == "max_tokens"

    def test_missing_usage(self, client):
        data = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
        }
        result = client._parse_anthropic_response(data)
        assert result.usage["prompt_tokens"] == 0
        assert result.usage["completion_tokens"] == 0

    def test_multiple_tool_use_blocks(self, client):
        data = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_a",
                    "name": "fn_a",
                    "input": {"x": 1},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_b",
                    "name": "fn_b",
                    "input": {"y": 2},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = client._parse_anthropic_response(data)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "toolu_a"
        assert result.tool_calls[1].id == "toolu_b"


# =========================================================================
# 14. close Tests
# =========================================================================


class TestClose:
    """Tests for LLMClient.close."""

    @pytest.mark.asyncio
    async def test_close_with_active_client(self, client):
        # Create an actual client first
        await client._get_client()
        assert client._client is not None
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self, client):
        """Should not raise even when no client exists."""
        assert client._client is None
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_calls_aclose(self, client):
        mock_http = AsyncMock()
        mock_http.aclose = AsyncMock()
        client._client = mock_http

        await client.close()

        mock_http.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_sets_client_to_none(self, client):
        mock_http = AsyncMock()
        mock_http.aclose = AsyncMock()
        client._client = mock_http

        await client.close()
        assert client._client is None


# =========================================================================
# 15. health_check Tests
# =========================================================================


class TestHealthCheck:
    """Tests for LLMClient.health_check."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_non_200(self, client):
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self, client):
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=Exception("Connection refused"))
        client._client = mock_http

        result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_url(self, client):
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        await client.health_check()

        call_args = mock_http.get.call_args
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
        assert url == "http://localhost:8000/v1/health"

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, client):
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_http.get = AsyncMock(return_value=mock_response)
        client._client = mock_http

        await client.health_check()

        call_kwargs = mock_http.get.call_args
        timeout = call_kwargs.kwargs.get("timeout") if call_kwargs.kwargs else None
        assert timeout == 5.0

    @pytest.mark.asyncio
    async def test_health_check_timeout_exception(self, client):
        """httpx.TimeoutException should return False."""
        import httpx as httpx_mod

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=httpx_mod.TimeoutException("Timed out"))
        client._client = mock_http

        result = await client.health_check()
        assert result is False


# =========================================================================
# 16. SyncLLMClient Tests
# =========================================================================


class TestSyncLLMClient:
    """Tests for SyncLLMClient."""

    @pytest.fixture(autouse=True)
    def restore_event_loop(self):
        """Restore event loop after SyncLLMClient tests that call asyncio.run()."""
        yield
        import asyncio

        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

    def test_init_with_config(self, llm_config):
        sync_client = SyncLLMClient(config=llm_config)
        assert sync_client._async_client is not None
        assert sync_client._async_client.config is llm_config

    def test_init_default_config(self, llm_config):
        with patch("orchestrator.inference.local_llm.get_config") as mock_get:
            mock_cfg = MagicMock()
            mock_cfg.llm = llm_config
            mock_get.return_value = mock_cfg
            sync_client = SyncLLMClient()
            assert sync_client._async_client.config is llm_config

    def test_chat_calls_async_client(self, llm_config, sample_messages):
        """SyncLLMClient.chat invokes the async client's chat via asyncio.run."""
        sync_client = SyncLLMClient(config=llm_config)
        expected_response = ChatResponse(content="Sync response")

        with patch.object(
            sync_client._async_client,
            "chat",
            new_callable=AsyncMock,
            return_value=expected_response,
        ) as mock_chat:
            result = sync_client.chat(sample_messages)
            assert result == expected_response
            mock_chat.assert_called_once_with(
                sample_messages, None, None, None, stream=False
            )

    def test_chat_passes_arguments(self, llm_config, sample_messages, sample_tools):
        sync_client = SyncLLMClient(config=llm_config)
        expected_response = ChatResponse(content="ok")

        with patch.object(
            sync_client._async_client,
            "chat",
            new_callable=AsyncMock,
            return_value=expected_response,
        ) as mock_chat:
            result = sync_client.chat(
                sample_messages,
                tools=sample_tools,
                temperature=0.5,
                max_tokens=256,
            )
            mock_chat.assert_called_once_with(
                sample_messages, sample_tools, 0.5, 256, stream=False
            )
            assert result == expected_response

    def test_close_calls_async_close(self, llm_config):
        sync_client = SyncLLMClient(config=llm_config)

        with patch.object(
            sync_client._async_client,
            "close",
            new_callable=AsyncMock,
        ) as mock_close:
            sync_client.close()
            mock_close.assert_called_once()

    def test_chat_stream_false(self, llm_config, sample_messages):
        """SyncLLMClient always passes stream=False."""
        sync_client = SyncLLMClient(config=llm_config)
        expected_response = ChatResponse(content="no stream")

        with patch.object(
            sync_client._async_client,
            "chat",
            new_callable=AsyncMock,
            return_value=expected_response,
        ) as mock_chat:
            sync_client.chat(sample_messages)
            call_kwargs = mock_chat.call_args
            assert call_kwargs.kwargs.get("stream") is False


# =========================================================================
# 17. Edge Cases
# =========================================================================


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_messages_list(self, llm_config, openai_api_response):
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        result = await c._chat_vllm([], None, None, None, False)
        assert isinstance(result, ChatResponse)

        # Verify empty messages list sent
        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["messages"] == []

    def test_chat_message_none_content(self):
        """ChatMessage with None content should work for to_dict."""
        msg = ChatMessage(role="assistant", content=None)
        d = msg.to_dict()
        assert d["content"] is None

    def test_parse_openai_missing_message_key(self, client):
        data = {
            "choices": [{"finish_reason": "stop"}],
            "usage": {},
        }
        result = client._parse_openai_response(data)
        assert result.content == ""
        assert result.role == "assistant"

    def test_parse_openai_empty_choices_list(self, client):
        """Empty choices list results in default values from [{}][0]."""
        data = {"choices": [{}], "usage": {}}
        result = client._parse_openai_response(data)
        assert result.content == ""

    def test_parse_anthropic_no_content_key(self, client):
        data = {"stop_reason": "end_turn"}
        result = client._parse_anthropic_response(data)
        assert result.content == ""
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_vllm_with_zero_temperature(
        self, llm_config, sample_messages, openai_api_response
    ):
        """Temperature 0 is falsy, so it falls back to config default."""
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_vllm(sample_messages, None, 0, None, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        # 0 is falsy, so `temperature or self.config.temperature` = config default
        assert payload["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_vllm_with_zero_max_tokens(
        self, llm_config, sample_messages, openai_api_response
    ):
        """max_tokens 0 is falsy, so it falls back to config default."""
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(openai_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        await c._chat_vllm(sample_messages, None, None, 0, False)

        call_kwargs = mock_http.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_chat_response_from_vllm_with_tool_calls(self, llm_config):
        c = LLMClient(config=llm_config)
        mock_http = AsyncMock()
        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_test",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "Friday AI"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        mock_resp = _mock_http_response(data)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        messages = [ChatMessage(role="user", content="Search for Friday")]
        result = await c._chat_vllm(messages, None, None, None, False)
        assert result.has_tool_calls is True
        assert result.tool_calls[0].arguments == {"query": "Friday AI"}

    def test_tool_call_equality(self):
        """Two ToolCalls with same fields should be equal (dataclass)."""
        tc1 = ToolCall(id="1", name="fn", arguments={"a": 1})
        tc2 = ToolCall(id="1", name="fn", arguments={"a": 1})
        assert tc1 == tc2

    def test_chat_message_equality(self):
        """Two ChatMessages with same fields should be equal (dataclass)."""
        m1 = ChatMessage(role="user", content="hello")
        m2 = ChatMessage(role="user", content="hello")
        assert m1 == m2

    def test_chat_response_equality(self):
        """Two ChatResponses with same fields should be equal (dataclass)."""
        r1 = ChatResponse(content="hi")
        r2 = ChatResponse(content="hi")
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_anthropic_no_stream_support(
        self, anthropic_config, anthropic_api_response
    ):
        """Anthropic backend does not check stream flag; it always does non-stream."""
        messages = [ChatMessage(role="user", content="Hello")]
        c = LLMClient(config=anthropic_config)
        mock_http = AsyncMock()
        mock_resp = _mock_http_response(anthropic_api_response)
        mock_http.post = AsyncMock(return_value=mock_resp)
        c._client = mock_http

        # Even with stream=True, Anthropic code path does not branch on it
        result = await c._chat_anthropic(messages, None, None, None, True)
        assert isinstance(result, ChatResponse)

    @pytest.mark.asyncio
    async def test_get_client_after_close(self, client):
        """After close, _get_client should create a new client."""
        c1 = await client._get_client()
        await client.close()
        assert client._client is None
        c2 = await client._get_client()
        assert c2 is not None
        assert c1 is not c2
        await client.close()

    def test_parse_openai_finish_reason_length(self, client):
        data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "truncated..."},
                    "finish_reason": "length",
                }
            ],
            "usage": {},
        }
        result = client._parse_openai_response(data)
        assert result.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_sagemaker_messages_formatting(self, sagemaker_config):
        """SageMaker should format messages as role/content dicts."""
        messages = [
            ChatMessage(role="system", content="You are Friday."),
            ChatMessage(role="user", content="Hello", name="test_user"),
        ]
        c = LLMClient(config=sagemaker_config)

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps([{"generated_text": "ok"}]).encode(
            "utf-8"
        )

        mock_runtime = MagicMock()
        mock_runtime.invoke_endpoint.return_value = {"Body": mock_body}

        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = mock_runtime

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            await c._chat_sagemaker(messages, None, None, None, False)

        call_kwargs = mock_runtime.invoke_endpoint.call_args
        body = json.loads(call_kwargs.kwargs.get("Body"))
        # SageMaker formats only role and content (not name)
        assert body["inputs"] == [
            {"role": "system", "content": "You are Friday."},
            {"role": "user", "content": "Hello"},
        ]
