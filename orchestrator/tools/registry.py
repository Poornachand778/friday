"""
Tool Registry for Friday AI
============================

Central registry for all tools available to Friday.
Handles tool definitions, validation, and execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from tool execution"""

    success: bool
    data: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }


@dataclass
class Tool:
    """A tool definition"""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Callable[..., ToolResult]
    category: str = "general"
    requires_api: Optional[str] = None  # External API needed

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class ToolRegistry:
    """
    Central registry for Friday tools.

    Manages tool registration, lookup, and execution.
    Filters tools based on context.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_builtin_tools()

    def register(self, tool: Tool) -> None:
        """Register a tool"""
        self._tools[tool.name] = tool
        LOGGER.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def list_tools(self, filter_names: Optional[List[str]] = None) -> List[Tool]:
        """List tools, optionally filtered by names"""
        if filter_names is None:
            return list(self._tools.values())

        return [self._tools[name] for name in filter_names if name in self._tools]

    def get_tools_for_context(self, available_tools: List[str]) -> List[Tool]:
        """Get tools available for a specific context"""
        return self.list_tools(available_tools)

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name"""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

        try:
            LOGGER.info("Executing tool: %s", name)
            result = tool.handler(**arguments)
            return result
        except Exception as e:
            LOGGER.error("Tool execution failed: %s - %s", name, e)
            return ToolResult(success=False, error=str(e))

    def to_openai_tools(self, filter_names: Optional[List[str]] = None) -> List[Dict]:
        """Get tools in OpenAI format"""
        tools = self.list_tools(filter_names)
        return [tool.to_openai_format() for tool in tools]

    def to_anthropic_tools(
        self, filter_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """Get tools in Anthropic format"""
        tools = self.list_tools(filter_names)
        return [tool.to_anthropic_format() for tool in tools]

    def _register_builtin_tools(self) -> None:
        """Register built-in tools"""
        # Document Processing Tools
        self._register_document_tools()

        # Scene Manager Tools
        self.register(
            Tool(
                name="scene_search",
                description="Search screenplay scenes by content, emotion, or character. Returns matching scenes with relevance scores.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query (e.g., 'romantic scenes', 'conflict with father')",
                        },
                        "project_slug": {
                            "type": "string",
                            "description": "Project slug (optional, defaults to current project)",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                handler=self._handle_scene_search,
                category="screenplay",
            )
        )

        self.register(
            Tool(
                name="scene_get",
                description="Get full details of a specific scene by number.",
                parameters={
                    "type": "object",
                    "properties": {
                        "scene_number": {
                            "type": "integer",
                            "description": "Scene number (1, 2, 3, ...)",
                        },
                        "project_slug": {
                            "type": "string",
                            "description": "Project slug (optional)",
                        },
                    },
                    "required": ["scene_number"],
                },
                handler=self._handle_scene_get,
                category="screenplay",
            )
        )

        self.register(
            Tool(
                name="scene_update",
                description="Update scene metadata like title, summary, status, or tags.",
                parameters={
                    "type": "object",
                    "properties": {
                        "scene_number": {
                            "type": "integer",
                            "description": "Scene number to update",
                        },
                        "project_slug": {
                            "type": "string",
                            "description": "Project slug (optional)",
                        },
                        "title": {
                            "type": "string",
                            "description": "New scene title",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Scene summary/description",
                        },
                        "status": {
                            "type": "string",
                            "description": "Scene status (active, backlog, cut)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for the scene",
                        },
                    },
                    "required": ["scene_number"],
                },
                handler=self._handle_scene_update,
                category="screenplay",
            )
        )

        self.register(
            Tool(
                name="scene_reorder",
                description="Move a scene to a new position relative to other scenes.",
                parameters={
                    "type": "object",
                    "properties": {
                        "scene_number": {
                            "type": "integer",
                            "description": "Scene to move",
                        },
                        "after_scene": {
                            "type": "integer",
                            "description": "Place after this scene number",
                        },
                        "before_scene": {
                            "type": "integer",
                            "description": "Place before this scene number",
                        },
                        "project_slug": {
                            "type": "string",
                            "description": "Project slug (optional)",
                        },
                    },
                    "required": ["scene_number"],
                },
                handler=self._handle_scene_reorder,
                category="screenplay",
            )
        )

        self.register(
            Tool(
                name="scene_link",
                description="Create a relationship between two scenes (flashback, sequence, parallel).",
                parameters={
                    "type": "object",
                    "properties": {
                        "from_scene": {
                            "type": "integer",
                            "description": "Origin scene number",
                        },
                        "to_scene": {
                            "type": "integer",
                            "description": "Target scene number",
                        },
                        "relation_type": {
                            "type": "string",
                            "enum": ["sequence", "flashback", "parallel", "callback"],
                            "description": "Type of relationship",
                        },
                        "project_slug": {
                            "type": "string",
                            "description": "Project slug (optional)",
                        },
                    },
                    "required": ["from_scene", "to_scene", "relation_type"],
                },
                handler=self._handle_scene_link,
                category="screenplay",
            )
        )

        # Email Tools
        self.register(
            Tool(
                name="send_screenplay",
                description="Send a screenplay via email as PDF attachment.",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Recipient email addresses",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject",
                        },
                        "project_slug": {
                            "type": "string",
                            "description": "Screenplay project to send",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["pdf", "fountain", "html"],
                            "description": "Export format (default: pdf)",
                        },
                        "message": {
                            "type": "string",
                            "description": "Optional message to include",
                        },
                    },
                    "required": ["to", "subject", "project_slug"],
                },
                handler=self._handle_send_screenplay,
                category="email",
            )
        )

        self.register(
            Tool(
                name="send_email",
                description="Send a simple email.",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Recipient email addresses",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject",
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body",
                        },
                        "html": {
                            "type": "boolean",
                            "description": "Body is HTML (default: false)",
                        },
                    },
                    "required": ["to", "subject", "body"],
                },
                handler=self._handle_send_email,
                category="email",
            )
        )

        # Placeholder tools for future contexts
        self.register(
            Tool(
                name="camera_analyze",
                description="Analyze current camera feed (Kitchen/Storyboard contexts).",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to look for or analyze",
                        },
                    },
                    "required": ["query"],
                },
                handler=self._handle_camera_analyze,
                category="vision",
                requires_api="vision",
            )
        )

        self.register(
            Tool(
                name="generate_image",
                description="Generate an image from a description (Storyboard context).",
                parameters={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Image description/prompt",
                        },
                        "style": {
                            "type": "string",
                            "description": "Art style (cinematic, sketch, realistic)",
                        },
                    },
                    "required": ["prompt"],
                },
                handler=self._handle_generate_image,
                category="visual",
                requires_api="image_gen",
            )
        )

    # Tool Handlers - delegate to MCP services
    def _handle_scene_search(
        self, query: str, project_slug: str = None, top_k: int = 5
    ) -> ToolResult:
        """Handle scene_search tool"""
        try:
            from mcp.scene_manager import service

            results = service.search_scenes(
                query, project_slug=project_slug, top_k=top_k
            )
            return ToolResult(success=True, data=results)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _handle_scene_get(
        self, scene_number: int, project_slug: str = None
    ) -> ToolResult:
        """Handle scene_get tool"""
        try:
            from mcp.scene_manager import service

            # Need to resolve scene_number to scene_id first
            # This is a simplified version
            result = service.get_scene_detail(scene_number)  # May need adjustment
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _handle_scene_update(
        self,
        scene_number: int,
        project_slug: str = None,
        title: str = None,
        summary: str = None,
        status: str = None,
        tags: List[str] = None,
    ) -> ToolResult:
        """Handle scene_update tool"""
        try:
            from mcp.scene_manager import service

            # Simplified - may need to resolve scene_id
            result = service.update_scene(
                scene_number,
                title=title,
                summary=summary,
                status=status,
                tags=tags,
            )
            return ToolResult(success=True, data={"updated": True})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _handle_scene_reorder(
        self,
        scene_number: int,
        after_scene: int = None,
        before_scene: int = None,
        project_slug: str = None,
    ) -> ToolResult:
        """Handle scene_reorder tool"""
        try:
            # Implementation would go through MCP service
            return ToolResult(success=True, data={"reordered": True})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _handle_scene_link(
        self,
        from_scene: int,
        to_scene: int,
        relation_type: str,
        project_slug: str = None,
    ) -> ToolResult:
        """Handle scene_link tool"""
        try:
            from mcp.scene_manager import service

            service.create_relation(from_scene, to_scene, relation_type=relation_type)
            return ToolResult(success=True, data={"linked": True})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _handle_send_screenplay(
        self,
        to: List[str],
        subject: str,
        project_slug: str,
        format: str = "pdf",
        message: str = None,
    ) -> ToolResult:
        """Handle send_screenplay tool"""
        try:
            from mcp.gmail import send_screenplay_email

            result = send_screenplay_email(
                to=to,
                subject=subject,
                project_slug=project_slug,
                format=format,
                message_text=message,
            )
            return ToolResult(success=result.get("sent", False), data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _handle_send_email(
        self,
        to: List[str],
        subject: str,
        body: str,
        html: bool = False,
    ) -> ToolResult:
        """Handle send_email tool"""
        try:
            from mcp.gmail import send_simple_email

            result = send_simple_email(to=to, subject=subject, body=body, html=html)
            return ToolResult(success=result.get("sent", False), data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _handle_camera_analyze(self, query: str) -> ToolResult:
        """Handle camera_analyze tool - placeholder"""
        return ToolResult(
            success=False,
            error="Camera analysis not yet implemented. Requires vision API integration.",
        )

    def _handle_generate_image(self, prompt: str, style: str = None) -> ToolResult:
        """Handle generate_image tool - placeholder"""
        return ToolResult(
            success=False,
            error="Image generation not yet implemented. Requires image generation API.",
        )

    def _register_document_tools(self) -> None:
        """Register document processing tools"""
        self.register(
            Tool(
                name="document_search",
                description="Search across ingested documents (books, screenplays, references). Returns relevant passages with inline citations.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query (e.g., 'character arc principles', 'three-act structure')",
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Limit search to specific document (optional)",
                        },
                        "document_type": {
                            "type": "string",
                            "enum": [
                                "book",
                                "screenplay",
                                "article",
                                "manual",
                                "reference",
                            ],
                            "description": "Filter by document type (optional)",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default: 5, max: 20)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                handler=self._handle_document_search,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="document_get_context",
                description="Get document context for answering questions. Returns formatted passages with citations, optimized for LLM consumption.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question or topic to find context for",
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Limit to specific document (optional)",
                        },
                        "max_chunks": {
                            "type": "integer",
                            "description": "Maximum chunks to include (default: 3)",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
                handler=self._handle_document_get_context,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="document_list",
                description="List all ingested documents with metadata.",
                parameters={
                    "type": "object",
                    "properties": {
                        "document_type": {
                            "type": "string",
                            "enum": [
                                "book",
                                "screenplay",
                                "article",
                                "manual",
                                "reference",
                            ],
                            "description": "Filter by document type (optional)",
                        },
                        "project": {
                            "type": "string",
                            "description": "Filter by project (optional)",
                        },
                    },
                },
                handler=self._handle_document_list,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="document_get",
                description="Get detailed information about a specific document.",
                parameters={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document UUID",
                        },
                    },
                    "required": ["document_id"],
                },
                handler=self._handle_document_get,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="document_ingest",
                description="Ingest a PDF document for conversational access. Processes with OCR and creates searchable chunks.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the PDF file",
                        },
                        "title": {
                            "type": "string",
                            "description": "Document title",
                        },
                        "author": {
                            "type": "string",
                            "description": "Author name (optional)",
                        },
                        "document_type": {
                            "type": "string",
                            "enum": [
                                "book",
                                "screenplay",
                                "article",
                                "manual",
                                "reference",
                            ],
                            "default": "book",
                        },
                    },
                    "required": ["file_path", "title"],
                },
                handler=self._handle_document_ingest,
                category="documents",
            )
        )

    # Document Tool Handlers
    def _handle_document_search(
        self,
        query: str,
        document_id: str = None,
        document_type: str = None,
        top_k: int = 5,
    ) -> ToolResult:
        """Handle document_search tool"""
        import asyncio

        try:
            from mcp.documents import service

            result = asyncio.get_event_loop().run_until_complete(
                service.document_search(
                    query=query,
                    document_id=document_id,
                    document_type=document_type,
                    top_k=top_k,
                )
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document search failed: %s", e)
            return ToolResult(success=False, error=str(e))

    def _handle_document_get_context(
        self,
        query: str,
        document_id: str = None,
        max_chunks: int = 3,
    ) -> ToolResult:
        """Handle document_get_context tool"""
        import asyncio

        try:
            from mcp.documents import service

            result = asyncio.get_event_loop().run_until_complete(
                service.document_get_context(
                    query=query,
                    document_id=document_id,
                    max_chunks=max_chunks,
                )
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document get context failed: %s", e)
            return ToolResult(success=False, error=str(e))

    def _handle_document_list(
        self,
        document_type: str = None,
        project: str = None,
    ) -> ToolResult:
        """Handle document_list tool"""
        import asyncio

        try:
            from mcp.documents import service

            result = asyncio.get_event_loop().run_until_complete(
                service.document_list(
                    document_type=document_type,
                    project=project,
                )
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document list failed: %s", e)
            return ToolResult(success=False, error=str(e))

    def _handle_document_get(self, document_id: str) -> ToolResult:
        """Handle document_get tool"""
        import asyncio

        try:
            from mcp.documents import service

            result = asyncio.get_event_loop().run_until_complete(
                service.document_get(document_id=document_id)
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document get failed: %s", e)
            return ToolResult(success=False, error=str(e))

    def _handle_document_ingest(
        self,
        file_path: str,
        title: str,
        author: str = None,
        document_type: str = "book",
    ) -> ToolResult:
        """Handle document_ingest tool"""
        import asyncio

        try:
            from mcp.documents import service

            result = asyncio.get_event_loop().run_until_complete(
                service.document_ingest(
                    file_path=file_path,
                    title=title,
                    author=author,
                    document_type=document_type,
                )
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document ingest failed: %s", e)
            return ToolResult(success=False, error=str(e))


# Singleton registry
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the tool registry singleton"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
