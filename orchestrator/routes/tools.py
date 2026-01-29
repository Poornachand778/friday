"""
Tools API Routes
================

Direct tool execution endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from orchestrator.core import get_orchestrator
from orchestrator.tools.registry import get_tool_registry

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/tools", tags=["tools"])


class ToolExecuteRequest(BaseModel):
    """Tool execution request"""

    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )


class ToolExecuteResponse(BaseModel):
    """Tool execution response"""

    success: bool
    data: Any = None
    error: Optional[str] = None


class ToolInfo(BaseModel):
    """Tool information"""

    name: str
    description: str
    category: str
    parameters: Dict[str, Any]


@router.get("", response_model=List[ToolInfo])
async def list_tools(category: Optional[str] = None):
    """
    List all available tools.

    Optionally filter by category (screenplay, email, vision, visual).
    """
    registry = get_tool_registry()
    tools = registry.list_tools()

    if category:
        tools = [t for t in tools if t.category == category]

    return [
        ToolInfo(
            name=t.name,
            description=t.description,
            category=t.category,
            parameters=t.parameters,
        )
        for t in tools
    ]


@router.get("/{tool_name}", response_model=ToolInfo)
async def get_tool(tool_name: str):
    """Get information about a specific tool."""
    registry = get_tool_registry()
    tool = registry.get(tool_name)

    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

    return ToolInfo(
        name=tool.name,
        description=tool.description,
        category=tool.category,
        parameters=tool.parameters,
    )


@router.post("/execute", response_model=ToolExecuteResponse)
async def execute_tool(request: ToolExecuteRequest):
    """
    Execute a tool directly.

    Example:
        POST /tools/execute
        {
            "name": "scene_search",
            "arguments": {"query": "romantic scenes", "top_k": 5}
        }
    """
    orchestrator = get_orchestrator()

    if not orchestrator.is_initialized:
        await orchestrator.initialize()

    result = await orchestrator.execute_tool(request.name, request.arguments)

    return ToolExecuteResponse(
        success=result.success,
        data=result.data,
        error=result.error,
    )


@router.get("/context/{context_type}")
async def get_tools_for_context(context_type: str):
    """
    Get tools available for a specific context.

    Context types: writers_room, kitchen, storyboard, general
    """
    from orchestrator.context.contexts import CONTEXTS, ContextType

    try:
        ctx = ContextType(context_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid context type. Valid: {[c.value for c in ContextType]}",
        )

    context_config = CONTEXTS.get(ctx)
    if not context_config:
        raise HTTPException(
            status_code=404, detail=f"Context not found: {context_type}"
        )

    registry = get_tool_registry()
    tools = registry.list_tools(context_config.available_tools)

    return {
        "context": context_type,
        "description": context_config.description,
        "tools": [
            {
                "name": t.name,
                "description": t.description,
            }
            for t in tools
        ],
    }
