#!/usr/bin/env python3
"""
Friday AI Orchestrator Server
=============================

FastAPI server for the Friday AI orchestrator.

Usage:
    uvicorn orchestrator.main:app --reload --port 8000

    # Or directly:
    python -m orchestrator.main

Endpoints:
    POST /chat              - Chat with Friday
    POST /chat/voice        - Voice-specific chat (for daemon)
    GET  /tools             - List available tools
    POST /tools/execute     - Execute a tool directly
    GET  /sessions          - List active sessions
    POST /sessions          - Create new session
    GET  /health            - Health check
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orchestrator.core import get_orchestrator, initialize_orchestrator
from orchestrator.routes import chat_router, tools_router, sessions_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    LOGGER.info("Starting Friday AI Orchestrator...")
    try:
        await initialize_orchestrator()
        LOGGER.info("Friday Orchestrator ready")
    except Exception as e:
        LOGGER.warning("Orchestrator initialization deferred: %s", e)

    yield

    # Shutdown
    LOGGER.info("Shutting down Friday AI Orchestrator...")
    orchestrator = get_orchestrator()
    await orchestrator.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Friday AI Orchestrator",
    description="Central orchestrator for Friday AI - JARVIS-style assistant",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(tools_router)
app.include_router(sessions_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Friday AI Orchestrator",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    orchestrator = get_orchestrator()
    health_info = await orchestrator.health_check()
    return health_info


@app.get("/context")
async def get_current_context():
    """Get current context information"""
    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    from orchestrator.context.contexts import CONTEXTS

    current = orchestrator.current_context
    config = CONTEXTS.get(current)

    return {
        "current_context": current.value,
        "description": config.description if config else "",
        "available_tools": config.available_tools if config else [],
        "lora_adapter": config.lora_adapter if config else None,
    }


@app.post("/context/{context_type}")
async def set_context(context_type: str):
    """Manually set context (override detection)"""
    from orchestrator.context.contexts import ContextType

    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    try:
        new_context = ContextType(context_type)
    except ValueError:
        return {
            "error": f"Invalid context type. Valid: {[c.value for c in ContextType]}"
        }

    orchestrator._current_context = new_context
    if orchestrator.current_session:
        orchestrator.current_session.set_context(context_type)

    return {
        "message": f"Context set to: {context_type}",
        "current_context": new_context.value,
    }


def main():
    """Run the server directly"""
    import uvicorn

    port = int(os.environ.get("FRIDAY_PORT", 8000))
    host = os.environ.get("FRIDAY_HOST", "0.0.0.0")
    reload = os.environ.get("FRIDAY_RELOAD", "false").lower() == "true"

    LOGGER.info("Starting Friday AI on %s:%d", host, port)

    uvicorn.run(
        "orchestrator.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
