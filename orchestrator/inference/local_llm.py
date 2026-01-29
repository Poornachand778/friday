"""
LLM Client for Friday AI
=========================

Unified client for LLM inference supporting:
- vLLM (OpenAI-compatible API)
- llama.cpp server
- OpenAI API (fallback)
- Anthropic API (fallback)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx

from orchestrator.config import LLMConfig, get_config


LOGGER = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A chat message"""

    role: str  # system, user, assistant, tool
    content: str
    name: Optional[str] = None  # For tool messages
    tool_calls: Optional[List[Dict]] = None  # For assistant tool calls
    tool_call_id: Optional[str] = None  # For tool responses

    def to_dict(self) -> Dict[str, Any]:
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


@dataclass
class ToolCall:
    """A tool call from the model"""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatResponse:
    """Response from LLM"""

    content: str
    role: str = "assistant"
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """
    Unified LLM client for Friday AI.

    Supports multiple backends:
    - vLLM: High-performance local inference
    - llama.cpp: Lightweight local inference
    - OpenAI: Cloud fallback
    - Anthropic: Cloud fallback

    Usage:
        client = LLMClient()

        # Simple chat
        response = await client.chat([
            ChatMessage(role="user", content="Hello Friday")
        ])

        # With tools
        response = await client.chat(messages, tools=tool_list)
        if response.has_tool_calls:
            for call in response.tool_calls:
                result = execute_tool(call.name, call.arguments)
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_config().llm
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[ChatResponse, AsyncIterator[str]]:
        """
        Send chat completion request.

        Args:
            messages: Conversation messages
            tools: Available tools (OpenAI format)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Stream response tokens

        Returns:
            ChatResponse or async iterator of tokens if streaming
        """
        if self.config.backend == "vllm":
            return await self._chat_vllm(
                messages, tools, temperature, max_tokens, stream
            )
        elif self.config.backend == "llamacpp":
            return await self._chat_llamacpp(
                messages, tools, temperature, max_tokens, stream
            )
        elif self.config.backend == "openai":
            return await self._chat_openai(
                messages, tools, temperature, max_tokens, stream
            )
        elif self.config.backend == "anthropic":
            return await self._chat_anthropic(
                messages, tools, temperature, max_tokens, stream
            )
        elif self.config.backend == "sagemaker":
            return await self._chat_sagemaker(
                messages, tools, temperature, max_tokens, stream
            )
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    async def _chat_vllm(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> Union[ChatResponse, AsyncIterator[str]]:
        """Chat via vLLM (OpenAI-compatible API)"""
        client = await self._get_client()

        payload = {
            "model": self.config.model_name,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = f"{self.config.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}

        if self.config.api_key and self.config.api_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        if stream:
            return self._stream_response(client, url, headers, payload)

        LOGGER.debug("Sending request to vLLM: %s", url)
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        return self._parse_openai_response(data)

    async def _chat_llamacpp(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> Union[ChatResponse, AsyncIterator[str]]:
        """Chat via llama.cpp server"""
        # llama.cpp server also supports OpenAI-compatible API
        return await self._chat_vllm(messages, tools, temperature, max_tokens, stream)

    async def _chat_openai(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> Union[ChatResponse, AsyncIterator[str]]:
        """Chat via OpenAI API"""
        client = await self._get_client()

        payload = {
            "model": self.config.model_name,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        if stream:
            return self._stream_response(client, url, headers, payload)

        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        return self._parse_openai_response(data)

    async def _chat_anthropic(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> Union[ChatResponse, AsyncIterator[str]]:
        """Chat via Anthropic API"""
        client = await self._get_client()

        # Convert messages to Anthropic format
        system_msg = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                anthropic_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        payload = {
            "model": self.config.model_name,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature or self.config.temperature,
        }

        if system_msg:
            payload["system"] = system_msg

        if tools:
            # Convert to Anthropic tool format
            payload["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }

        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        return self._parse_anthropic_response(data)

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict,
        payload: Dict,
    ) -> AsyncIterator[str]:
        """Stream response tokens"""
        async with client.stream(
            "POST", url, json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    def _parse_openai_response(self, data: Dict) -> ChatResponse:
        """Parse OpenAI-format response"""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=args,
                    )
                )

        return ChatResponse(
            content=message.get("content", ""),
            role=message.get("role", "assistant"),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=data.get("usage", {}),
        )

    async def _chat_sagemaker(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> Union[ChatResponse, AsyncIterator[str]]:
        """Chat via SageMaker endpoint (HuggingFace TGI format)"""
        import boto3

        # Build chat template format for Llama
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})

        payload = {
            "inputs": formatted_messages,
            "parameters": {
                "max_new_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature or self.config.temperature,
                "top_p": self.config.top_p,
                "do_sample": True,
                "return_full_text": False,
            },
        }

        # Add tools if provided (for models that support function calling)
        if tools:
            payload["parameters"]["tools"] = tools

        # Get endpoint name from config (base_url stores endpoint name for sagemaker)
        endpoint_name = self.config.base_url

        LOGGER.debug("Sending request to SageMaker endpoint: %s", endpoint_name)

        # Use boto3 for SageMaker invocation
        runtime = boto3.client("sagemaker-runtime")

        try:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload),
            )

            result = json.loads(response["Body"].read().decode("utf-8"))

            # Parse TGI response format
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "")
            else:
                generated_text = str(result)

            return ChatResponse(
                content=generated_text,
                role="assistant",
                tool_calls=[],
                finish_reason="stop",
                usage={},
            )

        except Exception as e:
            LOGGER.error("SageMaker invocation failed: %s", e)
            raise

    def _parse_anthropic_response(self, data: Dict) -> ChatResponse:
        """Parse Anthropic response"""
        content_blocks = data.get("content", [])

        text_content = ""
        tool_calls = []

        for block in content_blocks:
            if block["type"] == "text":
                text_content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        arguments=block["input"],
                    )
                )

        return ChatResponse(
            content=text_content,
            role="assistant",
            tool_calls=tool_calls,
            finish_reason=data.get("stop_reason", "stop"),
            usage={
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
            },
        )

    async def close(self) -> None:
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the LLM backend is available"""
        try:
            client = await self._get_client()
            url = f"{self.config.base_url}/health"
            response = await client.get(url, timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            LOGGER.warning("Health check failed: %s", e)
            return False


# Sync wrapper for non-async contexts
class SyncLLMClient:
    """Synchronous wrapper for LLMClient"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self._async_client = LLMClient(config)

    def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Synchronous chat"""
        import asyncio

        async def _chat():
            return await self._async_client.chat(
                messages, tools, temperature, max_tokens, stream=False
            )

        return asyncio.run(_chat())

    def close(self) -> None:
        """Close the client"""
        import asyncio

        asyncio.run(self._async_client.close())
