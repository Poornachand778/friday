"""
Chat API Routes
===============

Main chat endpoint for Friday AI interactions.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from orchestrator.core import get_orchestrator

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """Chat request body"""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID (optional)")
    location: Optional[str] = Field(None, description="Physical location hint")
    stream: bool = Field(False, description="Stream response tokens")


class ChatResponse(BaseModel):
    """Chat response body"""

    content: str
    context: str
    session_id: str
    turn_id: int
    tool_calls: list
    processing_time_ms: float


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to Friday.

    Example:
        POST /chat
        {
            "message": "Boss, show me romantic scenes",
            "location": "writers_room"
        }
    """
    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    if request.stream:
        # Return streaming response
        async def generate():
            async for token in await orchestrator.chat(
                message=request.message,
                session_id=request.session_id,
                location=request.location,
                stream=True,
            ):
                yield token

        return StreamingResponse(generate(), media_type="text/plain")

    # Non-streaming response
    response = await orchestrator.chat(
        message=request.message,
        session_id=request.session_id,
        location=request.location,
        stream=False,
    )

    return ChatResponse(
        content=response.content,
        context=response.context_type.value,
        session_id=orchestrator._current_session_id,
        turn_id=response.turn_id,
        tool_calls=response.tool_calls_made,
        processing_time_ms=response.processing_time_ms,
    )


@router.post("/voice")
async def chat_voice(
    transcript: str,
    session_id: Optional[str] = None,
    location: Optional[str] = None,
):
    """
    Voice-specific chat endpoint.

    Called by the voice daemon after STT.
    Returns response text for TTS.
    """
    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    response = await orchestrator.chat(
        message=transcript,
        session_id=session_id,
        location=location,
        stream=False,
    )

    return {
        "response": response.content,
        "context": response.context_type.value,
        "turn_id": response.turn_id,
    }
