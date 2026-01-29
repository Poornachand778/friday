"""
Orchestrator Client for Voice Daemon
====================================

HTTP client for communicating with the Friday Orchestrator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

LOGGER = logging.getLogger(__name__)


@dataclass
class VoiceChatResponse:
    """Response from orchestrator voice endpoint"""

    response: str
    context: str
    turn_id: int


class OrchestratorClient:
    """
    Client for Friday Orchestrator API.

    Used by voice daemon to get responses from the LLM.

    Usage:
        client = OrchestratorClient()

        response = await client.chat(
            transcript="Boss, show me scene 5",
            location="writers_room",
        )

        print(response.response)  # TTS this
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._session_id: Optional[str] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def chat(
        self,
        transcript: str,
        location: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> VoiceChatResponse:
        """
        Send transcript to orchestrator and get response.

        Args:
            transcript: User's transcribed speech
            location: Physical location (writers_room, kitchen, etc.)
            session_id: Session ID to use (preserves conversation)

        Returns:
            VoiceChatResponse with text for TTS
        """
        client = await self._get_client()

        # Use stored session ID if not provided
        sid = session_id or self._session_id

        url = f"{self.base_url}/chat/voice"
        params = {
            "transcript": transcript,
        }
        if location:
            params["location"] = location
        if sid:
            params["session_id"] = sid

        try:
            LOGGER.debug("Calling orchestrator: %s", transcript[:50])
            response = await client.post(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Store session ID for continuity
            if "session_id" in data:
                self._session_id = data["session_id"]

            return VoiceChatResponse(
                response=data["response"],
                context=data.get("context", "general"),
                turn_id=data.get("turn_id", 0),
            )

        except httpx.HTTPStatusError as e:
            LOGGER.error("Orchestrator HTTP error: %s", e)
            return VoiceChatResponse(
                response="Boss, I had trouble processing that. Can you repeat?",
                context="error",
                turn_id=0,
            )
        except httpx.RequestError as e:
            LOGGER.error("Orchestrator connection error: %s", e)
            return VoiceChatResponse(
                response="Boss, I can't reach my brain right now. Is the orchestrator running?",
                context="error",
                turn_id=0,
            )
        except Exception as e:
            LOGGER.error("Orchestrator error: %s", e)
            return VoiceChatResponse(
                response="Boss, something went wrong. Let me try again.",
                context="error",
                turn_id=0,
            )

    async def health_check(self) -> bool:
        """Check if orchestrator is available"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            LOGGER.warning("Orchestrator health check failed: %s", e)
            return False

    async def get_context(self) -> str:
        """Get current orchestrator context"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/context")
            if response.status_code == 200:
                return response.json().get("current_context", "general")
        except Exception as e:
            LOGGER.debug("Failed to get context: %s", e)
        return "general"

    async def close(self) -> None:
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None


# Fallback for when orchestrator is not available
class LocalFallbackClient:
    """
    Fallback client when orchestrator is not running.

    Provides basic echo responses for testing voice pipeline.
    """

    async def chat(
        self,
        transcript: str,
        location: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> VoiceChatResponse:
        """Echo back the transcript"""
        return VoiceChatResponse(
            response=f"Boss, I heard: {transcript}",
            context=location or "general",
            turn_id=0,
        )

    async def health_check(self) -> bool:
        return True

    async def get_context(self) -> str:
        return "general"

    async def close(self) -> None:
        pass
