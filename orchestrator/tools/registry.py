"""
Tool Registry for Friday AI
============================

Central registry for all tools available to Friday.
Handles tool definitions, validation, and execution.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from orchestrator.config import DEFAULT_PROJECT_SLUG

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
        """Execute a tool by name (sync only - use async_execute for async handlers)."""
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

    async def async_execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name, supporting both sync and async handlers."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Unknown tool: {name}")

        try:
            LOGGER.info("Executing tool: %s", name)
            result = tool.handler(**arguments)
            # If handler returns a coroutine, await it
            if inspect.isawaitable(result):
                result = await result
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

        # Book Understanding & Mentor Tools
        self._register_book_mentor_tools()

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

    def _register_book_mentor_tools(self) -> None:
        """Register book understanding and mentor tools"""
        self.register(
            Tool(
                name="book_study",
                description="Study a document and extract structured knowledge (concepts, principles, techniques, examples). Document must be ingested first.",
                parameters={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "UUID of the ingested document to study",
                        },
                        "link_to_project": {
                            "type": "string",
                            "description": "Project to link extracted knowledge to (optional)",
                        },
                        "thorough_mode": {
                            "type": "boolean",
                            "description": "Process all chapters (true) or sample (false, default)",
                        },
                        "voice_enabled": {
                            "type": "boolean",
                            "description": "Enable voice progress announcements (default: true)",
                            "default": True,
                        },
                    },
                    "required": ["document_id"],
                },
                handler=self._handle_book_study,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="book_study_status",
                description="Get live status of a book study operation. Voice-friendly progress messages.",
                parameters={
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "Job ID from book_study (optional)",
                        },
                        "document_id": {
                            "type": "string",
                            "description": "Document UUID (optional)",
                        },
                    },
                },
                handler=self._handle_book_study_status,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="book_list_studied",
                description="List all studied books with their knowledge counts.",
                parameters={"type": "object", "properties": {}},
                handler=self._handle_book_list_studied,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="book_study_jobs",
                description="List all book study jobs (active and recently completed).",
                parameters={"type": "object", "properties": {}},
                handler=self._handle_book_study_jobs,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="mentor_load_books",
                description="Load studied books for a mentoring session. Required before using mentor tools.",
                parameters={
                    "type": "object",
                    "properties": {
                        "understanding_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of book understanding UUIDs to load",
                        },
                    },
                    "required": ["understanding_ids"],
                },
                handler=self._handle_mentor_load_books,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="mentor_analyze",
                description="Analyze a scene against loaded book knowledge. Identifies strengths, missing elements, and relevant principles.",
                parameters={
                    "type": "object",
                    "properties": {
                        "scene_description": {
                            "type": "string",
                            "description": "Description of the scene to analyze",
                        },
                        "project_context": {
                            "type": "string",
                            "description": "Context about the project (optional)",
                        },
                    },
                    "required": ["scene_description"],
                },
                handler=self._handle_mentor_analyze,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="mentor_brainstorm",
                description="Brainstorm ideas using book knowledge. Generates creative ideas grounded in principles and techniques.",
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "What to brainstorm about",
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Requirements or limitations (optional)",
                        },
                    },
                    "required": ["topic"],
                },
                handler=self._handle_mentor_brainstorm,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="mentor_check_rules",
                description="Check a scene against principles from loaded books. Shows which rules are followed or violated.",
                parameters={
                    "type": "object",
                    "properties": {
                        "scene_text": {
                            "type": "string",
                            "description": "The scene text to check",
                        },
                    },
                    "required": ["scene_text"],
                },
                handler=self._handle_mentor_check_rules,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="mentor_find_inspiration",
                description="Find inspiring examples from books for a creative situation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "situation": {
                            "type": "string",
                            "description": "What you're trying to write or solve",
                        },
                    },
                    "required": ["situation"],
                },
                handler=self._handle_mentor_find_inspiration,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="mentor_ask",
                description="Ask what the loaded books say about a topic.",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Your question",
                        },
                    },
                    "required": ["question"],
                },
                handler=self._handle_mentor_ask,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="mentor_compare",
                description="Compare what different books say about a topic. Requires at least 2 books loaded.",
                parameters={
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to compare views on across books",
                        },
                    },
                    "required": ["topic"],
                },
                handler=self._handle_mentor_compare,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="book_get_understanding",
                description="Get full extracted knowledge from a studied book (concepts, principles, techniques, examples).",
                parameters={
                    "type": "object",
                    "properties": {
                        "understanding_id": {
                            "type": "string",
                            "description": "The understanding UUID (from book_list_studied)",
                        },
                    },
                    "required": ["understanding_id"],
                },
                handler=self._handle_book_get_understanding,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="knowledge_search",
                description="Search across all extracted book knowledge (concepts, principles, techniques, examples).",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "knowledge_type": {
                            "type": "string",
                            "enum": ["concept", "principle", "technique", "example"],
                            "description": "Filter by knowledge type (optional)",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum results (default: 10)",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
                handler=self._handle_knowledge_search,
                category="documents",
            )
        )

    # =========================================================================
    # Scene resolution helper
    # =========================================================================
    @staticmethod
    def _resolve_scene_id(service_module, scene_number: int, project_slug: str) -> int:
        """Resolve scene_number + project_slug to database scene_id."""
        from sqlalchemy.orm import Session as SASession

        engine = service_module.get_engine_instance()
        with SASession(engine) as session:
            scene = service_module.fetch_scene_by_number(
                session, project_slug, scene_number
            )
            return scene.id

    # =========================================================================
    # Tool Handlers - delegate to MCP services
    # =========================================================================
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
        """Handle scene_get tool - resolves scene_number to proper query."""
        try:
            from mcp.scene_manager import service

            result = service.get_scene_detail(
                scene_number=scene_number,
                project_slug=project_slug or DEFAULT_PROJECT_SLUG,
            )
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
        """Handle scene_update tool - resolves scene_number before update."""
        try:
            from mcp.scene_manager import service

            slug = project_slug or DEFAULT_PROJECT_SLUG
            scene_id = self._resolve_scene_id(service, scene_number, slug)
            changed = service.update_scene(
                scene_id,
                title=title,
                summary=summary,
                status=status,
                tags=tags,
            )
            detail = service.get_scene_detail(scene_id=scene_id)
            return ToolResult(
                success=True, data={"updated": bool(changed), "scene": detail}
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _handle_scene_reorder(
        self,
        scene_number: int,
        after_scene: int = None,
        before_scene: int = None,
        project_slug: str = None,
    ) -> ToolResult:
        """Handle scene_reorder tool - delegates to MCP server logic."""
        try:
            from mcp.scene_manager.server import SceneManagerMCPServer

            slug = project_slug or DEFAULT_PROJECT_SLUG
            server = SceneManagerMCPServer(default_project=slug)
            result = server.tool_scene_reorder(
                {
                    "scene_number": scene_number,
                    "after_scene": after_scene,
                    "before_scene": before_scene,
                    "project_slug": slug,
                }
            )
            return ToolResult(success=True, data=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _handle_scene_link(
        self,
        from_scene: int,
        to_scene: int,
        relation_type: str,
        project_slug: str = None,
    ) -> ToolResult:
        """Handle scene_link tool - resolves scene_numbers before linking."""
        try:
            from mcp.scene_manager import service

            slug = project_slug or DEFAULT_PROJECT_SLUG
            from_id = self._resolve_scene_id(service, from_scene, slug)
            to_id = self._resolve_scene_id(service, to_scene, slug)
            service.create_relation(from_id, to_id, relation_type=relation_type)
            return ToolResult(
                success=True, data={"linked": True, "relation_type": relation_type}
            )
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
                        "project": {
                            "type": "string",
                            "description": "Filter by project slug (optional)",
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
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum total characters (default: 4000)",
                            "default": 4000,
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
                        "status": {
                            "type": "string",
                            "enum": [
                                "pending",
                                "processing",
                                "completed",
                                "failed",
                            ],
                            "description": "Filter by processing status (optional)",
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
                        "language": {
                            "type": "string",
                            "enum": ["en", "te", "mixed"],
                            "description": "Primary language (default: en)",
                            "default": "en",
                        },
                        "project": {
                            "type": "string",
                            "description": "Link to Friday project slug (optional)",
                        },
                    },
                    "required": ["file_path", "title"],
                },
                handler=self._handle_document_ingest,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="document_get_chapter",
                description="Get the full text of a specific chapter from an ingested document.",
                parameters={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document UUID",
                        },
                        "chapter_title": {
                            "type": "string",
                            "description": "Chapter title to retrieve (optional if index given)",
                        },
                        "chapter_index": {
                            "type": "integer",
                            "description": "Chapter index (0-based, optional if title given)",
                        },
                    },
                    "required": ["document_id"],
                },
                handler=self._handle_document_get_chapter,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="document_status",
                description="Check the processing status of an ingested document.",
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
                handler=self._handle_document_status,
                category="documents",
            )
        )

        self.register(
            Tool(
                name="document_delete",
                description="Delete a document and all its chunks.",
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
                handler=self._handle_document_delete,
                category="documents",
            )
        )

    # =========================================================================
    # Document Tool Handlers (async - use via async_execute)
    # =========================================================================
    async def _handle_document_search(
        self,
        query: str,
        document_id: str = None,
        document_type: str = None,
        project: str = None,
        top_k: int = 5,
    ) -> ToolResult:
        """Handle document_search tool"""
        try:
            from mcp.documents import service

            result = await service.document_search(
                query=query,
                document_id=document_id,
                document_type=document_type,
                project=project,
                top_k=top_k,
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document search failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_document_get_context(
        self,
        query: str,
        document_id: str = None,
        max_chunks: int = 3,
        max_chars: int = 4000,
    ) -> ToolResult:
        """Handle document_get_context tool"""
        try:
            from mcp.documents import service

            result = await service.document_get_context(
                query=query,
                document_id=document_id,
                max_chunks=max_chunks,
                max_chars=max_chars,
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document get context failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_document_list(
        self,
        document_type: str = None,
        project: str = None,
        status: str = None,
    ) -> ToolResult:
        """Handle document_list tool"""
        try:
            from mcp.documents import service

            result = await service.document_list(
                document_type=document_type,
                project=project,
                status=status,
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document list failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_document_get(self, document_id: str) -> ToolResult:
        """Handle document_get tool"""
        try:
            from mcp.documents import service

            result = await service.document_get(document_id=document_id)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document get failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_document_ingest(
        self,
        file_path: str,
        title: str,
        author: str = None,
        document_type: str = "book",
        language: str = "en",
        project: str = None,
    ) -> ToolResult:
        """Handle document_ingest tool"""
        try:
            from mcp.documents import service

            result = await service.document_ingest(
                file_path=file_path,
                title=title,
                author=author,
                document_type=document_type,
                language=language,
                project=project,
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document ingest failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_document_get_chapter(
        self,
        document_id: str,
        chapter_title: str = None,
        chapter_index: int = None,
    ) -> ToolResult:
        """Handle document_get_chapter tool"""
        try:
            from mcp.documents import service

            result = await service.document_get_chapter(
                document_id=document_id,
                chapter_title=chapter_title,
                chapter_index=chapter_index,
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document get chapter failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_document_status(self, document_id: str) -> ToolResult:
        """Handle document_status tool"""
        try:
            from mcp.documents import service

            result = await service.document_status(document_id=document_id)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document status failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_document_delete(self, document_id: str) -> ToolResult:
        """Handle document_delete tool"""
        try:
            from mcp.documents import service

            result = await service.document_delete(document_id=document_id)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Document delete failed: %s", e)
            return ToolResult(success=False, error=str(e))

    # =========================================================================
    # Book Understanding & Mentor Tool Handlers (async)
    # =========================================================================
    async def _handle_book_study(
        self,
        document_id: str,
        link_to_project: str = None,
        thorough_mode: bool = None,
        voice_enabled: bool = True,
    ) -> ToolResult:
        """Handle book_study tool"""
        try:
            from mcp.documents import service

            result = await service.book_study(
                document_id=document_id,
                link_to_project=link_to_project,
                thorough_mode=thorough_mode,
                voice_enabled=voice_enabled,
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Book study failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_book_study_status(
        self,
        job_id: str = None,
        document_id: str = None,
    ) -> ToolResult:
        """Handle book_study_status tool"""
        try:
            from mcp.documents import service

            result = await service.book_study_status(
                job_id=job_id, document_id=document_id
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Book study status failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_book_list_studied(self) -> ToolResult:
        """Handle book_list_studied tool"""
        try:
            from mcp.documents import service

            result = await service.book_list_studied()
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Book list studied failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_book_study_jobs(self) -> ToolResult:
        """Handle book_study_jobs tool"""
        try:
            from mcp.documents import service

            result = await service.book_study_jobs()
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Book study jobs failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_mentor_load_books(
        self, understanding_ids: List[str]
    ) -> ToolResult:
        """Handle mentor_load_books tool"""
        try:
            from mcp.documents import service

            result = await service.mentor_load_books(
                understanding_ids=understanding_ids
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Mentor load books failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_mentor_analyze(
        self, scene_description: str, project_context: str = ""
    ) -> ToolResult:
        """Handle mentor_analyze tool"""
        try:
            from mcp.documents import service

            result = await service.mentor_analyze(
                scene_description=scene_description,
                project_context=project_context,
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Mentor analyze failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_mentor_brainstorm(
        self, topic: str, constraints: List[str] = None
    ) -> ToolResult:
        """Handle mentor_brainstorm tool"""
        try:
            from mcp.documents import service

            result = await service.mentor_brainstorm(
                topic=topic, constraints=constraints
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Mentor brainstorm failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_mentor_check_rules(self, scene_text: str) -> ToolResult:
        """Handle mentor_check_rules tool"""
        try:
            from mcp.documents import service

            result = await service.mentor_check_rules(scene_text=scene_text)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Mentor check rules failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_mentor_find_inspiration(self, situation: str) -> ToolResult:
        """Handle mentor_find_inspiration tool"""
        try:
            from mcp.documents import service

            result = await service.mentor_find_inspiration(situation=situation)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Mentor find inspiration failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_mentor_ask(self, question: str) -> ToolResult:
        """Handle mentor_ask tool"""
        try:
            from mcp.documents import service

            result = await service.mentor_ask(question=question)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Mentor ask failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_mentor_compare(self, topic: str) -> ToolResult:
        """Handle mentor_compare tool"""
        try:
            from mcp.documents import service

            result = await service.mentor_compare(topic=topic)
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Mentor compare failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_book_get_understanding(self, understanding_id: str) -> ToolResult:
        """Handle book_get_understanding tool"""
        try:
            from mcp.documents import service

            result = await service.book_get_understanding(
                understanding_id=understanding_id
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Book get understanding failed: %s", e)
            return ToolResult(success=False, error=str(e))

    async def _handle_knowledge_search(
        self, query: str, knowledge_type: str = None, top_k: int = 10
    ) -> ToolResult:
        """Handle knowledge_search tool"""
        try:
            from mcp.documents import service

            result = await service.knowledge_search(
                query=query, knowledge_type=knowledge_type, top_k=top_k
            )
            return ToolResult(
                success=result.get("success", False),
                data=result,
                error=result.get("error"),
            )
        except Exception as e:
            LOGGER.error("Knowledge search failed: %s", e)
            return ToolResult(success=False, error=str(e))


# Singleton registry
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the tool registry singleton"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
