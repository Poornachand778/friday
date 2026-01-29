"""
Sessions API Routes
===================

Conversation session management endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from orchestrator.core import get_orchestrator

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionInfo(BaseModel):
    """Session information"""

    session_id: str
    turn_count: int
    active_turns: int = 0
    current_context: str
    started_at: float
    total_tokens: int = 0


class CreateSessionRequest(BaseModel):
    """Create session request"""

    session_id: Optional[str] = Field(None, description="Custom session ID (optional)")


class SessionHistoryResponse(BaseModel):
    """Session history response"""

    session_id: str
    turns: List[Dict[str, Any]]


@router.get("", response_model=List[SessionInfo])
async def list_sessions():
    """List all active sessions."""
    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    sessions = orchestrator.list_sessions()

    return [
        SessionInfo(
            session_id=s["session_id"],
            turn_count=s["turn_count"],
            current_context=s["current_context"],
            started_at=s["started_at"],
        )
        for s in sessions
    ]


@router.post("", response_model=SessionInfo)
async def create_session(request: CreateSessionRequest = None):
    """Create a new conversation session."""
    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    session_id = request.session_id if request else None
    new_id = orchestrator._create_session(session_id)

    info = orchestrator.get_session_info(new_id)

    return SessionInfo(
        session_id=new_id,
        turn_count=info.get("turn_count", 0),
        active_turns=info.get("active_turns", 0),
        current_context=info.get("current_context", "general"),
        started_at=info.get("started_at", 0),
        total_tokens=info.get("total_tokens", 0),
    )


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get information about a specific session."""
    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    info = orchestrator.get_session_info(session_id)

    if "error" in info:
        raise HTTPException(status_code=404, detail=info["error"])

    return SessionInfo(
        session_id=info["session_id"],
        turn_count=info["turn_count"],
        active_turns=info.get("active_turns", 0),
        current_context=info["current_context"],
        started_at=info["started_at"],
        total_tokens=info.get("total_tokens", 0),
    )


@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str, last_n: int = 10):
    """Get conversation history for a session."""
    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    if session_id not in orchestrator._sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    memory = orchestrator._sessions[session_id]
    turns = memory.get_last_n_turns(last_n)

    return SessionHistoryResponse(
        session_id=session_id,
        turns=[
            {
                "turn_id": t.turn_id,
                "user_message": t.user_message,
                "assistant_response": t.assistant_response,
                "timestamp": t.timestamp,
                "context_type": t.context_type,
                "tool_calls": t.tool_calls,
            }
            for t in turns
        ],
    )


@router.post("/{session_id}/switch")
async def switch_session(session_id: str):
    """Switch to a different session."""
    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    if not orchestrator.switch_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": f"Switched to session: {session_id}"}


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its history."""
    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    if session_id not in orchestrator._sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Clear the session memory
    orchestrator._sessions[session_id].clear()
    del orchestrator._sessions[session_id]

    # If it was the current session, create a new one
    if orchestrator._current_session_id == session_id:
        orchestrator._create_session()

    return {"message": f"Session deleted: {session_id}"}


@router.post("/{session_id}/clear")
async def clear_session(session_id: str):
    """Clear conversation history for a session but keep it active."""
    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    if session_id not in orchestrator._sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator._sessions[session_id].clear()

    return {"message": f"Session history cleared: {session_id}"}
