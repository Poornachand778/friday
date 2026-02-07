"""
Tests for ToolRegistry - Issue #9
==================================

Comprehensive tests for:
- ToolResult dataclass and serialization
- Tool dataclass and format conversion (OpenAI, Anthropic)
- ToolRegistry: registration, lookup, listing, filtering
- execute() for sync handlers
- async_execute() for both sync and async handlers
- Error handling and edge cases
- Singleton pattern (get_tool_registry)
- All 26 built-in tools registered
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.tools.registry import (
    Tool,
    ToolRegistry,
    ToolResult,
    get_tool_registry,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# =========================================================================
# ToolResult Tests
# =========================================================================


class TestToolResult:
    """Tests for the ToolResult dataclass."""

    def test_success_result(self):
        r = ToolResult(success=True, data={"key": "value"})
        assert r.success is True
        assert r.data == {"key": "value"}
        assert r.error is None

    def test_error_result(self):
        r = ToolResult(success=False, error="Something went wrong")
        assert r.success is False
        assert r.data is None
        assert r.error == "Something went wrong"

    def test_to_dict_success(self):
        r = ToolResult(success=True, data=[1, 2, 3])
        d = r.to_dict()
        assert d == {"success": True, "data": [1, 2, 3], "error": None}

    def test_to_dict_error(self):
        r = ToolResult(success=False, error="fail")
        d = r.to_dict()
        assert d == {"success": False, "data": None, "error": "fail"}

    def test_to_dict_with_both(self):
        r = ToolResult(success=False, data={"partial": True}, error="timeout")
        d = r.to_dict()
        assert d["success"] is False
        assert d["data"] == {"partial": True}
        assert d["error"] == "timeout"


# =========================================================================
# Tool Tests
# =========================================================================


class TestTool:
    """Tests for the Tool dataclass."""

    def _make_tool(self, name="test_tool", category="general", requires_api=None):
        return Tool(
            name=name,
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query"},
                },
                "required": ["query"],
            },
            handler=lambda query: ToolResult(success=True, data=query),
            category=category,
            requires_api=requires_api,
        )

    def test_to_openai_format(self):
        tool = self._make_tool()
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "test_tool"
        assert fmt["function"]["description"] == "A test tool"
        assert fmt["function"]["parameters"]["required"] == ["query"]

    def test_to_anthropic_format(self):
        tool = self._make_tool()
        fmt = tool.to_anthropic_format()
        assert fmt["name"] == "test_tool"
        assert fmt["description"] == "A test tool"
        assert fmt["input_schema"]["type"] == "object"
        assert "query" in fmt["input_schema"]["properties"]

    def test_category_default(self):
        tool = self._make_tool()
        assert tool.category == "general"

    def test_requires_api(self):
        tool = self._make_tool(requires_api="vision")
        assert tool.requires_api == "vision"


# =========================================================================
# ToolRegistry - Registration & Lookup
# =========================================================================


class TestToolRegistryRegistration:
    """Tests for tool registration, lookup, and listing."""

    def _make_registry_no_builtins(self):
        """Create a registry without built-in tools for isolation."""
        with patch.object(ToolRegistry, "_register_builtin_tools"):
            return ToolRegistry()

    def _make_tool(self, name, category="general"):
        return Tool(
            name=name,
            description=f"Tool {name}",
            parameters={"type": "object", "properties": {}},
            handler=lambda: ToolResult(success=True),
            category=category,
        )

    def test_register_and_get(self):
        reg = self._make_registry_no_builtins()
        tool = self._make_tool("my_tool")
        reg.register(tool)
        assert reg.get("my_tool") is tool

    def test_get_unknown_returns_none(self):
        reg = self._make_registry_no_builtins()
        assert reg.get("nonexistent") is None

    def test_register_overwrites(self):
        reg = self._make_registry_no_builtins()
        tool1 = self._make_tool("dup")
        tool2 = self._make_tool("dup")
        tool2.description = "Updated"
        reg.register(tool1)
        reg.register(tool2)
        assert reg.get("dup").description == "Updated"

    def test_list_tools_all(self):
        reg = self._make_registry_no_builtins()
        reg.register(self._make_tool("a"))
        reg.register(self._make_tool("b"))
        reg.register(self._make_tool("c"))
        tools = reg.list_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"a", "b", "c"}

    def test_list_tools_filtered(self):
        reg = self._make_registry_no_builtins()
        reg.register(self._make_tool("a"))
        reg.register(self._make_tool("b"))
        reg.register(self._make_tool("c"))
        tools = reg.list_tools(filter_names=["a", "c"])
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"a", "c"}

    def test_list_tools_filter_skips_missing(self):
        reg = self._make_registry_no_builtins()
        reg.register(self._make_tool("a"))
        tools = reg.list_tools(filter_names=["a", "missing"])
        assert len(tools) == 1
        assert tools[0].name == "a"

    def test_list_tools_empty_filter(self):
        reg = self._make_registry_no_builtins()
        reg.register(self._make_tool("a"))
        tools = reg.list_tools(filter_names=[])
        assert len(tools) == 0

    def test_get_tools_for_context(self):
        reg = self._make_registry_no_builtins()
        reg.register(self._make_tool("scene_search", "screenplay"))
        reg.register(self._make_tool("scene_get", "screenplay"))
        reg.register(self._make_tool("send_email", "email"))
        tools = reg.get_tools_for_context(["scene_search", "send_email"])
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"scene_search", "send_email"}


# =========================================================================
# ToolRegistry - Format Conversion
# =========================================================================


class TestToolRegistryFormats:
    """Tests for OpenAI and Anthropic format conversion."""

    def _make_registry_no_builtins(self):
        with patch.object(ToolRegistry, "_register_builtin_tools"):
            return ToolRegistry()

    def _make_tool(self, name):
        return Tool(
            name=name,
            description=f"Tool {name}",
            parameters={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
            handler=lambda q: ToolResult(success=True),
        )

    def test_to_openai_tools_all(self):
        reg = self._make_registry_no_builtins()
        reg.register(self._make_tool("a"))
        reg.register(self._make_tool("b"))
        openai_tools = reg.to_openai_tools()
        assert len(openai_tools) == 2
        assert all(t["type"] == "function" for t in openai_tools)

    def test_to_openai_tools_filtered(self):
        reg = self._make_registry_no_builtins()
        reg.register(self._make_tool("a"))
        reg.register(self._make_tool("b"))
        openai_tools = reg.to_openai_tools(filter_names=["a"])
        assert len(openai_tools) == 1
        assert openai_tools[0]["function"]["name"] == "a"

    def test_to_anthropic_tools_all(self):
        reg = self._make_registry_no_builtins()
        reg.register(self._make_tool("x"))
        anthropic_tools = reg.to_anthropic_tools()
        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "x"
        assert "input_schema" in anthropic_tools[0]

    def test_to_anthropic_tools_filtered(self):
        reg = self._make_registry_no_builtins()
        reg.register(self._make_tool("x"))
        reg.register(self._make_tool("y"))
        anthropic_tools = reg.to_anthropic_tools(filter_names=["y"])
        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "y"


# =========================================================================
# ToolRegistry - execute() (sync)
# =========================================================================


class TestToolRegistryExecute:
    """Tests for synchronous execute() method."""

    def _make_registry_no_builtins(self):
        with patch.object(ToolRegistry, "_register_builtin_tools"):
            return ToolRegistry()

    def test_execute_success(self):
        reg = self._make_registry_no_builtins()
        handler = MagicMock(return_value=ToolResult(success=True, data="ok"))
        reg.register(
            Tool(
                name="test",
                description="test",
                parameters={"type": "object", "properties": {}},
                handler=handler,
            )
        )
        result = reg.execute("test", {})
        assert result.success is True
        assert result.data == "ok"
        handler.assert_called_once_with()

    def test_execute_with_arguments(self):
        reg = self._make_registry_no_builtins()
        handler = MagicMock(return_value=ToolResult(success=True, data="found"))
        reg.register(
            Tool(
                name="search",
                description="search",
                parameters={"type": "object", "properties": {}},
                handler=handler,
            )
        )
        result = reg.execute("search", {"query": "hello", "top_k": 3})
        handler.assert_called_once_with(query="hello", top_k=3)
        assert result.success is True

    def test_execute_unknown_tool(self):
        reg = self._make_registry_no_builtins()
        result = reg.execute("nonexistent", {})
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_execute_handler_exception(self):
        reg = self._make_registry_no_builtins()
        handler = MagicMock(side_effect=ValueError("bad input"))
        reg.register(
            Tool(
                name="fail",
                description="fail",
                parameters={"type": "object", "properties": {}},
                handler=handler,
            )
        )
        result = reg.execute("fail", {})
        assert result.success is False
        assert "bad input" in result.error

    def test_execute_handler_returns_error_result(self):
        reg = self._make_registry_no_builtins()
        handler = MagicMock(return_value=ToolResult(success=False, error="not found"))
        reg.register(
            Tool(
                name="lookup",
                description="lookup",
                parameters={"type": "object", "properties": {}},
                handler=handler,
            )
        )
        result = reg.execute("lookup", {"id": "abc"})
        assert result.success is False
        assert result.error == "not found"


# =========================================================================
# ToolRegistry - async_execute()
# =========================================================================


class TestToolRegistryAsyncExecute:
    """Tests for async_execute() supporting both sync and async handlers."""

    def _make_registry_no_builtins(self):
        with patch.object(ToolRegistry, "_register_builtin_tools"):
            return ToolRegistry()

    def test_async_execute_sync_handler(self):
        """async_execute with a sync handler should work without awaiting."""
        reg = self._make_registry_no_builtins()
        handler = MagicMock(return_value=ToolResult(success=True, data="sync_ok"))
        reg.register(
            Tool(
                name="sync_tool",
                description="sync",
                parameters={"type": "object", "properties": {}},
                handler=handler,
            )
        )
        result = _run(reg.async_execute("sync_tool", {"x": 1}))
        assert result.success is True
        assert result.data == "sync_ok"
        handler.assert_called_once_with(x=1)

    def test_async_execute_async_handler(self):
        """async_execute with an async handler should await the coroutine."""
        reg = self._make_registry_no_builtins()

        async def async_handler(query):
            return ToolResult(success=True, data=f"async:{query}")

        reg.register(
            Tool(
                name="async_tool",
                description="async",
                parameters={"type": "object", "properties": {}},
                handler=async_handler,
            )
        )
        result = _run(reg.async_execute("async_tool", {"query": "test"}))
        assert result.success is True
        assert result.data == "async:test"

    def test_async_execute_unknown_tool(self):
        """async_execute with unknown tool returns error."""
        reg = self._make_registry_no_builtins()
        result = _run(reg.async_execute("missing", {}))
        assert result.success is False
        assert "Unknown tool" in result.error

    def test_async_execute_sync_handler_exception(self):
        """async_execute with sync handler that throws."""
        reg = self._make_registry_no_builtins()
        handler = MagicMock(side_effect=RuntimeError("sync_boom"))
        reg.register(
            Tool(
                name="boom",
                description="boom",
                parameters={"type": "object", "properties": {}},
                handler=handler,
            )
        )
        result = _run(reg.async_execute("boom", {}))
        assert result.success is False
        assert "sync_boom" in result.error

    def test_async_execute_async_handler_exception(self):
        """async_execute with async handler that throws."""
        reg = self._make_registry_no_builtins()

        async def bad_handler():
            raise ConnectionError("async_boom")

        reg.register(
            Tool(
                name="aboom",
                description="aboom",
                parameters={"type": "object", "properties": {}},
                handler=bad_handler,
            )
        )
        result = _run(reg.async_execute("aboom", {}))
        assert result.success is False
        assert "async_boom" in result.error

    def test_async_execute_with_multiple_args(self):
        """async_execute passes kwargs correctly to async handler."""
        reg = self._make_registry_no_builtins()

        async def multi_handler(query, top_k=5, document_id=None):
            return ToolResult(
                success=True,
                data={"query": query, "top_k": top_k, "doc": document_id},
            )

        reg.register(
            Tool(
                name="multi",
                description="multi",
                parameters={"type": "object", "properties": {}},
                handler=multi_handler,
            )
        )
        result = _run(
            reg.async_execute(
                "multi",
                {"query": "arcs", "top_k": 10, "document_id": "abc-123"},
            )
        )
        assert result.success is True
        assert result.data["query"] == "arcs"
        assert result.data["top_k"] == 10
        assert result.data["doc"] == "abc-123"

    def test_async_execute_async_mock(self):
        """async_execute works with AsyncMock handlers."""
        reg = self._make_registry_no_builtins()
        handler = AsyncMock(return_value=ToolResult(success=True, data="mocked"))
        reg.register(
            Tool(
                name="mocked",
                description="mocked",
                parameters={"type": "object", "properties": {}},
                handler=handler,
            )
        )
        result = _run(reg.async_execute("mocked", {"q": "test"}))
        assert result.success is True
        assert result.data == "mocked"
        handler.assert_awaited_once_with(q="test")


# =========================================================================
# ToolRegistry - Built-in Tools Registration
# =========================================================================


class TestBuiltinToolRegistration:
    """Tests verifying all 30 built-in tools are registered correctly."""

    @pytest.fixture
    def registry(self):
        """Create a real registry with all builtins registered."""
        return ToolRegistry()

    def test_total_tool_count(self, registry):
        """Should have exactly 30 tools registered."""
        tools = registry.list_tools()
        assert len(tools) == 30

    def test_scene_tools_registered(self, registry):
        scene_tools = [
            "scene_search",
            "scene_get",
            "scene_update",
            "scene_reorder",
            "scene_link",
        ]
        for name in scene_tools:
            tool = registry.get(name)
            assert tool is not None, f"Missing tool: {name}"
            assert tool.category == "screenplay"

    def test_email_tools_registered(self, registry):
        email_tools = ["send_screenplay", "send_email"]
        for name in email_tools:
            tool = registry.get(name)
            assert tool is not None, f"Missing tool: {name}"
            assert tool.category == "email"

    def test_document_tools_registered(self, registry):
        doc_tools = [
            "document_search",
            "document_get_context",
            "document_list",
            "document_get",
            "document_ingest",
            "document_get_chapter",
            "document_status",
            "document_delete",
        ]
        for name in doc_tools:
            tool = registry.get(name)
            assert tool is not None, f"Missing tool: {name}"
            assert tool.category == "documents"

    def test_book_mentor_tools_registered(self, registry):
        book_tools = [
            "book_study",
            "book_study_status",
            "book_study_jobs",
            "book_list_studied",
            "book_get_understanding",
            "knowledge_search",
            "mentor_load_books",
            "mentor_analyze",
            "mentor_brainstorm",
            "mentor_check_rules",
            "mentor_find_inspiration",
            "mentor_ask",
            "mentor_compare",
        ]
        for name in book_tools:
            tool = registry.get(name)
            assert tool is not None, f"Missing tool: {name}"
            assert tool.category == "documents"

    def test_vision_tools_registered(self, registry):
        tool = registry.get("camera_analyze")
        assert tool is not None
        assert tool.category == "vision"
        assert tool.requires_api == "vision"

    def test_image_gen_tool_registered(self, registry):
        tool = registry.get("generate_image")
        assert tool is not None
        assert tool.category == "visual"
        assert tool.requires_api == "image_gen"

    def test_all_tools_have_handlers(self, registry):
        """Every registered tool must have a callable handler."""
        for tool in registry.list_tools():
            assert callable(tool.handler), f"Tool {tool.name} handler is not callable"

    def test_all_tools_have_parameters(self, registry):
        """Every tool must have a parameters dict with type=object."""
        for tool in registry.list_tools():
            assert isinstance(
                tool.parameters, dict
            ), f"Tool {tool.name} has no parameters dict"
            assert (
                tool.parameters.get("type") == "object"
            ), f"Tool {tool.name} parameters type != object"

    def test_all_tools_have_description(self, registry):
        """Every tool must have a non-empty description."""
        for tool in registry.list_tools():
            assert tool.description, f"Tool {tool.name} has empty description"
            assert len(tool.description) > 10, f"Tool {tool.name} description too short"


# =========================================================================
# ToolRegistry - Sync Handler Tests (scene tools)
# =========================================================================


class TestSceneHandlerExecution:
    """Test sync handlers via execute() with mocked MCP service."""

    def test_scene_search_success(self):
        registry = ToolRegistry()
        mock_results = [{"scene_number": 1, "title": "Opening", "score": 0.9}]
        with patch(
            "mcp.scene_manager.service.search_scenes", return_value=mock_results
        ):
            result = registry.execute("scene_search", {"query": "opening scene"})
        assert result.success is True
        assert result.data == mock_results

    def test_scene_search_exception(self):
        registry = ToolRegistry()
        with patch(
            "mcp.scene_manager.service.search_scenes", side_effect=Exception("DB down")
        ):
            result = registry.execute("scene_search", {"query": "test"})
        assert result.success is False
        assert "DB down" in result.error

    def test_scene_get_success(self):
        registry = ToolRegistry()
        mock_detail = {"scene_number": 5, "title": "Confrontation"}
        with patch(
            "mcp.scene_manager.service.get_scene_detail", return_value=mock_detail
        ):
            result = registry.execute("scene_get", {"scene_number": 5})
        assert result.success is True
        assert result.data["scene_number"] == 5

    def test_scene_get_with_project(self):
        registry = ToolRegistry()
        mock_detail = {"scene_number": 1, "project": "test-project"}
        with patch(
            "mcp.scene_manager.service.get_scene_detail", return_value=mock_detail
        ):
            result = registry.execute(
                "scene_get",
                {
                    "scene_number": 1,
                    "project_slug": "test-project",
                },
            )
        assert result.success is True

    def test_camera_analyze_placeholder(self):
        registry = ToolRegistry()
        result = registry.execute("camera_analyze", {"query": "what's on the table?"})
        assert result.success is False
        assert "not yet implemented" in result.error

    def test_generate_image_placeholder(self):
        registry = ToolRegistry()
        result = registry.execute("generate_image", {"prompt": "sunset"})
        assert result.success is False
        assert "not yet implemented" in result.error


# =========================================================================
# ToolRegistry - Async Handler Tests (document tools via async_execute)
# =========================================================================


class TestAsyncHandlerExecution:
    """Test async handlers via async_execute() with mocked MCP service."""

    def test_document_search_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "results": [{"title": "Story"}]}
        with patch(
            "mcp.documents.service.document_search",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute(
                    "document_search",
                    {"query": "character arcs"},
                )
            )
        assert result.success is True
        assert result.data["results"][0]["title"] == "Story"

    def test_document_search_exception(self):
        registry = ToolRegistry()
        with patch(
            "mcp.documents.service.document_search",
            new_callable=AsyncMock,
            side_effect=Exception("search error"),
        ):
            result = _run(
                registry.async_execute(
                    "document_search",
                    {"query": "test"},
                )
            )
        assert result.success is False
        assert "search error" in result.error

    def test_document_list_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "documents": []}
        with patch(
            "mcp.documents.service.document_list",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(registry.async_execute("document_list", {}))
        assert result.success is True

    def test_document_get_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "document": {"id": "abc"}}
        with patch(
            "mcp.documents.service.document_get",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute("document_get", {"document_id": "abc"})
            )
        assert result.success is True
        assert result.data["document"]["id"] == "abc"

    def test_document_ingest_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "document_id": "new-123"}
        with patch(
            "mcp.documents.service.document_ingest",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute(
                    "document_ingest",
                    {"file_path": "/tmp/test.pdf", "title": "Test Book"},
                )
            )
        assert result.success is True
        assert result.data["document_id"] == "new-123"

    def test_document_get_context_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "context": "Some context", "citations": []}
        with patch(
            "mcp.documents.service.document_get_context",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute(
                    "document_get_context",
                    {"query": "three-act structure"},
                )
            )
        assert result.success is True
        assert result.data["context"] == "Some context"

    def test_book_study_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "job_id": "job-001"}
        with patch(
            "mcp.documents.service.book_study",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute("book_study", {"document_id": "doc-abc"})
            )
        assert result.success is True
        assert result.data["job_id"] == "job-001"

    def test_mentor_analyze_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {
            "success": True,
            "analysis": {"strengths": [], "suggestions": []},
        }
        with patch(
            "mcp.documents.service.mentor_analyze",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute(
                    "mentor_analyze",
                    {"scene_description": "Arjun confronts his father"},
                )
            )
        assert result.success is True

    def test_mentor_brainstorm_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "ideas": ["idea1", "idea2"]}
        with patch(
            "mcp.documents.service.mentor_brainstorm",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute(
                    "mentor_brainstorm",
                    {"topic": "courtroom climax", "constraints": ["no flashback"]},
                )
            )
        assert result.success is True

    def test_mentor_check_rules_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "followed": ["rule1"], "violated": []}
        with patch(
            "mcp.documents.service.mentor_check_rules",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute(
                    "mentor_check_rules",
                    {"scene_text": "INT. COURTROOM - DAY"},
                )
            )
        assert result.success is True

    def test_mentor_ask_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "answer": "According to McKee..."}
        with patch(
            "mcp.documents.service.mentor_ask",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute(
                    "mentor_ask", {"question": "What is inciting incident?"}
                )
            )
        assert result.success is True
        assert "McKee" in result.data["answer"]

    def test_mentor_compare_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "comparison": {}}
        with patch(
            "mcp.documents.service.mentor_compare",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute("mentor_compare", {"topic": "character arc"})
            )
        assert result.success is True

    def test_mentor_find_inspiration_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "inspirations": []}
        with patch(
            "mcp.documents.service.mentor_find_inspiration",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute(
                    "mentor_find_inspiration",
                    {"situation": "writing a chase sequence"},
                )
            )
        assert result.success is True

    def test_mentor_load_books_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "loaded": 2}
        with patch(
            "mcp.documents.service.mentor_load_books",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute(
                    "mentor_load_books",
                    {"understanding_ids": ["u1", "u2"]},
                )
            )
        assert result.success is True
        assert result.data["loaded"] == 2

    def test_book_study_status_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "status": "studying", "progress": 0.5}
        with patch(
            "mcp.documents.service.book_study_status",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(registry.async_execute("book_study_status", {"job_id": "j1"}))
        assert result.success is True
        assert result.data["progress"] == 0.5

    def test_book_list_studied_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "books": []}
        with patch(
            "mcp.documents.service.book_list_studied",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(registry.async_execute("book_list_studied", {}))
        assert result.success is True

    def test_book_get_understanding_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "understanding": {"concepts": []}}
        with patch(
            "mcp.documents.service.book_get_understanding",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute(
                    "book_get_understanding", {"understanding_id": "u1"}
                )
            )
        assert result.success is True

    def test_knowledge_search_via_async_execute(self):
        registry = ToolRegistry()
        mock_result = {"success": True, "results": []}
        with patch(
            "mcp.documents.service.knowledge_search",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = _run(
                registry.async_execute(
                    "knowledge_search",
                    {"query": "inciting incident", "top_k": 5},
                )
            )
        assert result.success is True


# =========================================================================
# ToolRegistry - Sync handler via async_execute (mixed mode)
# =========================================================================


class TestMixedModeExecution:
    """Test that sync handlers work correctly through async_execute."""

    def test_sync_scene_search_via_async_execute(self):
        """Sync scene_search handler should work through async_execute."""
        registry = ToolRegistry()
        mock_results = [{"scene": 1}]
        with patch(
            "mcp.scene_manager.service.search_scenes", return_value=mock_results
        ):
            result = _run(registry.async_execute("scene_search", {"query": "love"}))
        assert result.success is True
        assert result.data == mock_results

    def test_sync_scene_get_via_async_execute(self):
        """Sync scene_get handler should work through async_execute."""
        registry = ToolRegistry()
        mock_detail = {"scene_number": 3}
        with patch(
            "mcp.scene_manager.service.get_scene_detail", return_value=mock_detail
        ):
            result = _run(registry.async_execute("scene_get", {"scene_number": 3}))
        assert result.success is True

    def test_sync_camera_placeholder_via_async_execute(self):
        """Placeholder sync handler returns error via async_execute."""
        registry = ToolRegistry()
        result = _run(
            registry.async_execute("camera_analyze", {"query": "identify object"})
        )
        assert result.success is False
        assert "not yet implemented" in result.error


# =========================================================================
# ToolRegistry - Singleton Pattern
# =========================================================================


class TestGetToolRegistrySingleton:
    """Tests for the get_tool_registry() singleton."""

    def test_singleton_returns_same_instance(self):
        import orchestrator.tools.registry as mod

        mod._registry = None  # Reset
        r1 = get_tool_registry()
        r2 = get_tool_registry()
        assert r1 is r2
        mod._registry = None  # Cleanup

    def test_singleton_creates_instance(self):
        import orchestrator.tools.registry as mod

        mod._registry = None
        r = get_tool_registry()
        assert isinstance(r, ToolRegistry)
        mod._registry = None

    def test_singleton_has_all_tools(self):
        import orchestrator.tools.registry as mod

        mod._registry = None
        r = get_tool_registry()
        assert len(r.list_tools()) == 30
        mod._registry = None


# =========================================================================
# ToolRegistry - Tool Parameter Schema Validation
# =========================================================================


class TestToolParameterSchemas:
    """Verify required fields in tool parameter schemas."""

    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    def test_scene_search_requires_query(self, registry):
        tool = registry.get("scene_search")
        assert "query" in tool.parameters["required"]

    def test_scene_get_requires_scene_number(self, registry):
        tool = registry.get("scene_get")
        assert "scene_number" in tool.parameters["required"]

    def test_scene_update_requires_scene_number(self, registry):
        tool = registry.get("scene_update")
        assert "scene_number" in tool.parameters["required"]

    def test_scene_link_requires_all_three(self, registry):
        tool = registry.get("scene_link")
        required = tool.parameters["required"]
        assert "from_scene" in required
        assert "to_scene" in required
        assert "relation_type" in required

    def test_document_ingest_requires_file_and_title(self, registry):
        tool = registry.get("document_ingest")
        required = tool.parameters["required"]
        assert "file_path" in required
        assert "title" in required

    def test_book_study_requires_document_id(self, registry):
        tool = registry.get("book_study")
        assert "document_id" in tool.parameters["required"]

    def test_mentor_analyze_requires_scene_description(self, registry):
        tool = registry.get("mentor_analyze")
        assert "scene_description" in tool.parameters["required"]

    def test_mentor_brainstorm_requires_topic(self, registry):
        tool = registry.get("mentor_brainstorm")
        assert "topic" in tool.parameters["required"]

    def test_mentor_check_rules_requires_scene_text(self, registry):
        tool = registry.get("mentor_check_rules")
        assert "scene_text" in tool.parameters["required"]

    def test_send_email_requires_to_subject_body(self, registry):
        tool = registry.get("send_email")
        required = tool.parameters["required"]
        assert "to" in required
        assert "subject" in required
        assert "body" in required

    def test_knowledge_search_requires_query(self, registry):
        tool = registry.get("knowledge_search")
        assert "query" in tool.parameters["required"]

    def test_scene_link_relation_type_enum(self, registry):
        tool = registry.get("scene_link")
        enum = tool.parameters["properties"]["relation_type"]["enum"]
        assert "sequence" in enum
        assert "flashback" in enum
        assert "parallel" in enum
        assert "callback" in enum
