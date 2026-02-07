"""
Tests for MCP Document Server (mcp/documents/server.py)
=======================================================

Comprehensive tests for the JSON-RPC MCP server layer including:
- ToolDefinition dataclass
- _tool_definitions() function
- Individual tool schema validation
- TOOL_HANDLERS mapping
- _handle_list_tools() formatting
- _handle_call_tool() success, unknown tool, and exception paths
- handle_request() routing for all JSON-RPC methods
- main_loop() stdin/stdout communication
- main() argument parsing
- Edge cases
"""

import asyncio
import json
import logging
import sys
import pytest
from dataclasses import fields
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch, call


# ============================================================
# Helpers
# ============================================================


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_mock_handlers():
    """Create a dict of AsyncMock handlers matching all 8 tool names."""
    return {
        "document_ingest": AsyncMock(
            return_value={"id": "doc-123", "status": "processing"}
        ),
        "document_search": AsyncMock(return_value=[{"chunk": "text", "score": 0.9}]),
        "document_get_context": AsyncMock(
            return_value={"context": "...", "citations": []}
        ),
        "document_list": AsyncMock(return_value=[]),
        "document_get": AsyncMock(return_value={"id": "doc-123", "title": "Test"}),
        "document_get_chapter": AsyncMock(
            return_value={"chapter": "Ch1", "text": "..."}
        ),
        "document_status": AsyncMock(return_value={"status": "completed"}),
        "document_delete": AsyncMock(return_value={"deleted": True}),
    }


@pytest.fixture
def mock_handlers():
    """Patch TOOL_HANDLERS with AsyncMock functions for each tool."""
    handlers = _make_mock_handlers()
    with patch.dict("mcp.documents.server.TOOL_HANDLERS", handlers, clear=True):
        yield handlers


# ============================================================
# 1. ToolDefinition dataclass tests
# ============================================================


class TestToolDefinition:
    """Tests for the ToolDefinition dataclass."""

    def test_creation_basic(self):
        """ToolDefinition can be created with required fields."""
        from mcp.documents.server import ToolDefinition

        td = ToolDefinition(
            name="test_tool",
            description="A test tool.",
            input_schema={"type": "object", "properties": {}},
        )
        assert td.name == "test_tool"
        assert td.description == "A test tool."
        assert td.input_schema == {"type": "object", "properties": {}}

    def test_has_three_fields(self):
        """ToolDefinition has exactly three fields: name, description, input_schema."""
        from mcp.documents.server import ToolDefinition

        field_names = [f.name for f in fields(ToolDefinition)]
        assert field_names == ["name", "description", "input_schema"]

    def test_field_types(self):
        """ToolDefinition fields have correct type annotations."""
        from mcp.documents.server import ToolDefinition

        field_map = {f.name: f.type for f in fields(ToolDefinition)}
        assert field_map["name"] == "str"
        assert field_map["description"] == "str"
        assert field_map["input_schema"] == "Dict[str, Any]"

    def test_equality(self):
        """Two ToolDefinitions with the same values are equal (dataclass)."""
        from mcp.documents.server import ToolDefinition

        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        td1 = ToolDefinition(name="t", description="d", input_schema=schema)
        td2 = ToolDefinition(name="t", description="d", input_schema=schema)
        assert td1 == td2

    def test_inequality(self):
        """Two ToolDefinitions with different values are not equal."""
        from mcp.documents.server import ToolDefinition

        td1 = ToolDefinition(name="a", description="d", input_schema={})
        td2 = ToolDefinition(name="b", description="d", input_schema={})
        assert td1 != td2


# ============================================================
# 2. _tool_definitions() tests
# ============================================================


class TestToolDefinitions:
    """Tests for the _tool_definitions() function."""

    def test_returns_list(self):
        """_tool_definitions returns a list."""
        from mcp.documents.server import _tool_definitions

        result = _tool_definitions()
        assert isinstance(result, list)

    def test_returns_eight_tools(self):
        """_tool_definitions returns exactly 8 tools."""
        from mcp.documents.server import _tool_definitions

        result = _tool_definitions()
        assert len(result) == 8

    def test_all_are_tool_definitions(self):
        """Every element in the list is a ToolDefinition."""
        from mcp.documents.server import _tool_definitions, ToolDefinition

        for tool in _tool_definitions():
            assert isinstance(tool, ToolDefinition)

    def test_each_has_non_empty_name(self):
        """Every tool has a non-empty name."""
        from mcp.documents.server import _tool_definitions

        for tool in _tool_definitions():
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0

    def test_each_has_non_empty_description(self):
        """Every tool has a non-empty description."""
        from mcp.documents.server import _tool_definitions

        for tool in _tool_definitions():
            assert isinstance(tool.description, str)
            assert len(tool.description) > 0

    def test_each_has_object_type_schema(self):
        """Every tool schema has type 'object'."""
        from mcp.documents.server import _tool_definitions

        for tool in _tool_definitions():
            assert tool.input_schema["type"] == "object"

    def test_each_has_properties(self):
        """Every tool schema has a 'properties' key."""
        from mcp.documents.server import _tool_definitions

        for tool in _tool_definitions():
            assert "properties" in tool.input_schema

    def test_each_has_additional_properties_false(self):
        """Every tool schema has additionalProperties set to False."""
        from mcp.documents.server import _tool_definitions

        for tool in _tool_definitions():
            assert tool.input_schema.get("additionalProperties") is False

    def test_tool_names_are_unique(self):
        """All tool names are unique."""
        from mcp.documents.server import _tool_definitions

        names = [t.name for t in _tool_definitions()]
        assert len(names) == len(set(names))

    def test_expected_tool_names(self):
        """All 8 expected tool names are present."""
        from mcp.documents.server import _tool_definitions

        expected = {
            "document_ingest",
            "document_search",
            "document_get_context",
            "document_list",
            "document_get",
            "document_get_chapter",
            "document_status",
            "document_delete",
        }
        actual = {t.name for t in _tool_definitions()}
        assert actual == expected


# ============================================================
# 3. Individual tool schema validation
# ============================================================


class TestToolSchemaDocumentIngest:
    """Validate the document_ingest tool schema."""

    def _get_tool(self):
        from mcp.documents.server import _tool_definitions

        return next(t for t in _tool_definitions() if t.name == "document_ingest")

    def test_required_fields(self):
        tool = self._get_tool()
        assert set(tool.input_schema["required"]) == {"file_path", "title"}

    def test_properties_keys(self):
        tool = self._get_tool()
        props = tool.input_schema["properties"]
        expected_props = {
            "file_path",
            "title",
            "author",
            "document_type",
            "language",
            "project",
        }
        assert set(props.keys()) == expected_props

    def test_document_type_enum(self):
        tool = self._get_tool()
        dt = tool.input_schema["properties"]["document_type"]
        assert set(dt["enum"]) == {
            "book",
            "screenplay",
            "article",
            "manual",
            "reference",
        }

    def test_language_enum(self):
        tool = self._get_tool()
        lang = tool.input_schema["properties"]["language"]
        assert set(lang["enum"]) == {"en", "te", "hi", "mixed"}


class TestToolSchemaDocumentSearch:
    """Validate the document_search tool schema."""

    def _get_tool(self):
        from mcp.documents.server import _tool_definitions

        return next(t for t in _tool_definitions() if t.name == "document_search")

    def test_required_fields(self):
        tool = self._get_tool()
        assert tool.input_schema["required"] == ["query"]

    def test_has_top_k_with_constraints(self):
        tool = self._get_tool()
        top_k = tool.input_schema["properties"]["top_k"]
        assert top_k["type"] == "integer"
        assert top_k["minimum"] == 1
        assert top_k["maximum"] == 20
        assert top_k["default"] == 5

    def test_has_document_type_filter(self):
        tool = self._get_tool()
        props = tool.input_schema["properties"]
        assert "document_type" in props
        assert "enum" in props["document_type"]


class TestToolSchemaDocumentGetContext:
    """Validate the document_get_context tool schema."""

    def _get_tool(self):
        from mcp.documents.server import _tool_definitions

        return next(t for t in _tool_definitions() if t.name == "document_get_context")

    def test_required_fields(self):
        tool = self._get_tool()
        assert tool.input_schema["required"] == ["query"]

    def test_max_chunks_constraints(self):
        tool = self._get_tool()
        mc = tool.input_schema["properties"]["max_chunks"]
        assert mc["minimum"] == 1
        assert mc["maximum"] == 10
        assert mc["default"] == 3

    def test_max_chars_constraints(self):
        tool = self._get_tool()
        mc = tool.input_schema["properties"]["max_chars"]
        assert mc["minimum"] == 500
        assert mc["maximum"] == 10000
        assert mc["default"] == 4000


class TestToolSchemaDocumentList:
    """Validate the document_list tool schema."""

    def _get_tool(self):
        from mcp.documents.server import _tool_definitions

        return next(t for t in _tool_definitions() if t.name == "document_list")

    def test_no_required_fields(self):
        tool = self._get_tool()
        assert "required" not in tool.input_schema

    def test_has_status_enum(self):
        tool = self._get_tool()
        status = tool.input_schema["properties"]["status"]
        assert set(status["enum"]) == {"pending", "processing", "completed", "failed"}

    def test_has_project_filter(self):
        tool = self._get_tool()
        assert "project" in tool.input_schema["properties"]


class TestToolSchemaDocumentGet:
    """Validate the document_get tool schema."""

    def _get_tool(self):
        from mcp.documents.server import _tool_definitions

        return next(t for t in _tool_definitions() if t.name == "document_get")

    def test_required_document_id(self):
        tool = self._get_tool()
        assert tool.input_schema["required"] == ["document_id"]

    def test_only_one_property(self):
        tool = self._get_tool()
        assert list(tool.input_schema["properties"].keys()) == ["document_id"]


class TestToolSchemaDocumentGetChapter:
    """Validate the document_get_chapter tool schema."""

    def _get_tool(self):
        from mcp.documents.server import _tool_definitions

        return next(t for t in _tool_definitions() if t.name == "document_get_chapter")

    def test_required_document_id(self):
        tool = self._get_tool()
        assert tool.input_schema["required"] == ["document_id"]

    def test_has_chapter_title_and_index(self):
        tool = self._get_tool()
        props = tool.input_schema["properties"]
        assert "chapter_title" in props
        assert "chapter_index" in props

    def test_chapter_index_min(self):
        tool = self._get_tool()
        ci = tool.input_schema["properties"]["chapter_index"]
        assert ci["minimum"] == 0


class TestToolSchemaDocumentStatus:
    """Validate the document_status tool schema."""

    def _get_tool(self):
        from mcp.documents.server import _tool_definitions

        return next(t for t in _tool_definitions() if t.name == "document_status")

    def test_required_document_id(self):
        tool = self._get_tool()
        assert tool.input_schema["required"] == ["document_id"]

    def test_description_mentions_status(self):
        tool = self._get_tool()
        assert "status" in tool.description.lower()


class TestToolSchemaDocumentDelete:
    """Validate the document_delete tool schema."""

    def _get_tool(self):
        from mcp.documents.server import _tool_definitions

        return next(t for t in _tool_definitions() if t.name == "document_delete")

    def test_required_document_id(self):
        tool = self._get_tool()
        assert tool.input_schema["required"] == ["document_id"]

    def test_description_mentions_delete(self):
        tool = self._get_tool()
        assert "delete" in tool.description.lower()


# ============================================================
# 4. TOOL_HANDLERS tests
# ============================================================


class TestToolHandlers:
    """Tests for the TOOL_HANDLERS dictionary."""

    def test_has_eight_entries(self):
        from mcp.documents.server import TOOL_HANDLERS

        assert len(TOOL_HANDLERS) == 8

    def test_keys_match_tool_names(self):
        from mcp.documents.server import TOOL_HANDLERS, _tool_definitions

        expected_names = {t.name for t in _tool_definitions()}
        assert set(TOOL_HANDLERS.keys()) == expected_names

    def test_document_ingest_maps_to_service_function(self):
        from mcp.documents.server import TOOL_HANDLERS
        from mcp.documents import service

        assert TOOL_HANDLERS["document_ingest"] is service.document_ingest

    def test_document_search_maps_to_service_function(self):
        from mcp.documents.server import TOOL_HANDLERS
        from mcp.documents import service

        assert TOOL_HANDLERS["document_search"] is service.document_search

    def test_document_get_context_maps_to_service_function(self):
        from mcp.documents.server import TOOL_HANDLERS
        from mcp.documents import service

        assert TOOL_HANDLERS["document_get_context"] is service.document_get_context

    def test_document_list_maps_to_service_function(self):
        from mcp.documents.server import TOOL_HANDLERS
        from mcp.documents import service

        assert TOOL_HANDLERS["document_list"] is service.document_list

    def test_document_get_maps_to_service_function(self):
        from mcp.documents.server import TOOL_HANDLERS
        from mcp.documents import service

        assert TOOL_HANDLERS["document_get"] is service.document_get

    def test_document_get_chapter_maps_to_service_function(self):
        from mcp.documents.server import TOOL_HANDLERS
        from mcp.documents import service

        assert TOOL_HANDLERS["document_get_chapter"] is service.document_get_chapter

    def test_document_status_maps_to_service_function(self):
        from mcp.documents.server import TOOL_HANDLERS
        from mcp.documents import service

        assert TOOL_HANDLERS["document_status"] is service.document_status

    def test_document_delete_maps_to_service_function(self):
        from mcp.documents.server import TOOL_HANDLERS
        from mcp.documents import service

        assert TOOL_HANDLERS["document_delete"] is service.document_delete

    def test_all_values_are_callable(self):
        from mcp.documents.server import TOOL_HANDLERS

        for name, handler in TOOL_HANDLERS.items():
            assert callable(handler), f"Handler for {name} is not callable"


# ============================================================
# 5. _handle_list_tools() tests
# ============================================================


class TestHandleListTools:
    """Tests for the _handle_list_tools() function."""

    def test_returns_dict_with_tools_key(self):
        from mcp.documents.server import _handle_list_tools

        result = _handle_list_tools()
        assert isinstance(result, dict)
        assert "tools" in result

    def test_tools_count(self):
        from mcp.documents.server import _handle_list_tools

        result = _handle_list_tools()
        assert len(result["tools"]) == 8

    def test_each_tool_has_correct_keys(self):
        from mcp.documents.server import _handle_list_tools

        for tool in _handle_list_tools()["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    def test_input_schema_key_is_camel_case(self):
        """The output uses 'inputSchema' (camelCase) not 'input_schema'."""
        from mcp.documents.server import _handle_list_tools

        for tool in _handle_list_tools()["tools"]:
            assert "inputSchema" in tool
            assert "input_schema" not in tool

    def test_tool_names_match_definitions(self):
        from mcp.documents.server import _handle_list_tools, _tool_definitions

        listed_names = {t["name"] for t in _handle_list_tools()["tools"]}
        defined_names = {t.name for t in _tool_definitions()}
        assert listed_names == defined_names

    def test_schemas_are_preserved(self):
        from mcp.documents.server import _handle_list_tools, _tool_definitions

        tools_by_name = {t["name"]: t for t in _handle_list_tools()["tools"]}
        for td in _tool_definitions():
            assert tools_by_name[td.name]["inputSchema"] == td.input_schema


# ============================================================
# 6. _handle_call_tool() - success path
# ============================================================


class TestHandleCallToolSuccess:
    """Tests for successful _handle_call_tool() invocations."""

    def test_returns_content_with_json(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        result = _run(
            _handle_call_tool("document_ingest", {"file_path": "/a.pdf", "title": "T"})
        )
        assert "content" in result
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"
        parsed = json.loads(result["content"][0]["text"])
        assert parsed["id"] == "doc-123"

    def test_calls_handler_with_arguments(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        _run(_handle_call_tool("document_search", {"query": "inciting incident"}))
        mock_handlers["document_search"].assert_awaited_once_with(
            query="inciting incident"
        )

    def test_document_get_with_id(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        result = _run(_handle_call_tool("document_get", {"document_id": "doc-123"}))
        assert "content" in result
        mock_handlers["document_get"].assert_awaited_once_with(document_id="doc-123")

    def test_document_list_no_args(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        result = _run(_handle_call_tool("document_list", {}))
        assert "content" in result
        mock_handlers["document_list"].assert_awaited_once_with()

    def test_document_delete_success(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        result = _run(_handle_call_tool("document_delete", {"document_id": "doc-456"}))
        assert "content" in result
        parsed = json.loads(result["content"][0]["text"])
        assert parsed["deleted"] is True

    def test_document_status_success(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        result = _run(_handle_call_tool("document_status", {"document_id": "doc-789"}))
        parsed = json.loads(result["content"][0]["text"])
        assert parsed["status"] == "completed"

    def test_document_get_chapter_success(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        result = _run(
            _handle_call_tool(
                "document_get_chapter",
                {"document_id": "doc-123", "chapter_title": "Ch1"},
            )
        )
        parsed = json.loads(result["content"][0]["text"])
        assert parsed["chapter"] == "Ch1"

    def test_document_get_context_success(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        result = _run(
            _handle_call_tool("document_get_context", {"query": "character arcs"})
        )
        parsed = json.loads(result["content"][0]["text"])
        assert "context" in parsed

    def test_result_is_json_formatted(self, mock_handlers):
        """The text content should be pretty-printed JSON (indent=2)."""
        from mcp.documents.server import _handle_call_tool

        result = _run(_handle_call_tool("document_list", {}))
        text = result["content"][0]["text"]
        # json.dumps with indent=2 on an empty list is "[]" (no newlines needed)
        # but for dicts it will have newlines
        assert json.loads(text) == []


# ============================================================
# 7. _handle_call_tool() - unknown tool
# ============================================================


class TestHandleCallToolUnknown:
    """Tests for _handle_call_tool() with unknown tool names."""

    def test_unknown_tool_returns_error(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        result = _run(_handle_call_tool("nonexistent_tool", {}))
        assert "error" in result
        assert "nonexistent_tool" in result["error"]

    def test_empty_tool_name_returns_error(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        result = _run(_handle_call_tool("", {}))
        assert "error" in result


# ============================================================
# 8. _handle_call_tool() - exception handling
# ============================================================


class TestHandleCallToolException:
    """Tests for _handle_call_tool() when handler raises."""

    def test_runtime_error_returns_error_dict(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        mock_handlers["document_ingest"].side_effect = RuntimeError("OCR failed")
        result = _run(
            _handle_call_tool("document_ingest", {"file_path": "/a.pdf", "title": "T"})
        )
        assert "error" in result
        assert "OCR failed" in result["error"]
        assert "content" not in result

    def test_value_error_returns_error_dict(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        mock_handlers["document_get"].side_effect = ValueError("Invalid ID format")
        result = _run(_handle_call_tool("document_get", {"document_id": "bad"}))
        assert "error" in result
        assert "Invalid ID format" in result["error"]

    def test_generic_exception_returns_error_string(self, mock_handlers):
        from mcp.documents.server import _handle_call_tool

        mock_handlers["document_search"].side_effect = Exception("Something broke")
        result = _run(_handle_call_tool("document_search", {"query": "test"}))
        assert "error" in result
        assert "Something broke" in result["error"]


# ============================================================
# 9. handle_request() - tools/list
# ============================================================


class TestHandleRequestToolsList:
    """Tests for handle_request() with tools/list method."""

    def test_returns_jsonrpc_response(self):
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        result = _run(handle_request(request))
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert "result" in result

    def test_result_contains_tools(self):
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        result = _run(handle_request(request))
        assert "tools" in result["result"]
        assert len(result["result"]["tools"]) == 8

    def test_preserves_string_id(self):
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "id": "req-abc", "method": "tools/list"}
        result = _run(handle_request(request))
        assert result["id"] == "req-abc"


# ============================================================
# 10. handle_request() - tools/call
# ============================================================


class TestHandleRequestToolsCall:
    """Tests for handle_request() with tools/call method."""

    def test_routes_to_handler(self, mock_handlers):
        from mcp.documents.server import handle_request

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "document_search",
                "arguments": {"query": "test"},
            },
        }
        result = _run(handle_request(request))
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 2
        assert "content" in result["result"]

    def test_unknown_tool_via_request(self, mock_handlers):
        from mcp.documents.server import handle_request

        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "unknown_tool", "arguments": {}},
        }
        result = _run(handle_request(request))
        assert "error" in result["result"]

    def test_missing_arguments_defaults_to_empty(self, mock_handlers):
        from mcp.documents.server import handle_request

        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "document_list"},
        }
        result = _run(handle_request(request))
        assert "content" in result["result"]
        mock_handlers["document_list"].assert_awaited_once_with()

    def test_missing_name_defaults_to_empty_string(self, mock_handlers):
        from mcp.documents.server import handle_request

        request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {},
        }
        result = _run(handle_request(request))
        assert "error" in result["result"]


# ============================================================
# 11. handle_request() - initialize
# ============================================================


class TestHandleRequestInitialize:
    """Tests for handle_request() with initialize method."""

    def test_returns_protocol_info(self):
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "id": 10, "method": "initialize"}
        result = _run(handle_request(request))
        assert result["result"]["protocolVersion"] == "2024-11-05"

    def test_returns_capabilities(self):
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "id": 10, "method": "initialize"}
        result = _run(handle_request(request))
        assert "capabilities" in result["result"]
        assert "tools" in result["result"]["capabilities"]

    def test_returns_server_info(self):
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "id": 10, "method": "initialize"}
        result = _run(handle_request(request))
        info = result["result"]["serverInfo"]
        assert info["name"] == "friday-documents"
        assert info["version"] == "1.0.0"


# ============================================================
# 12. handle_request() - notifications/initialized
# ============================================================


class TestHandleRequestNotification:
    """Tests for handle_request() with notifications/initialized."""

    def test_returns_empty_dict(self):
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        result = _run(handle_request(request))
        assert result == {}

    def test_no_jsonrpc_wrapper(self):
        """Notifications should return raw empty dict, no jsonrpc envelope."""
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        result = _run(handle_request(request))
        assert "jsonrpc" not in result
        assert "id" not in result


# ============================================================
# 13. handle_request() - unknown method
# ============================================================


class TestHandleRequestUnknownMethod:
    """Tests for handle_request() with unrecognized methods."""

    def test_returns_error_in_result(self):
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "id": 99, "method": "resources/list"}
        result = _run(handle_request(request))
        assert "error" in result["result"]
        assert "resources/list" in result["result"]["error"]

    def test_still_has_jsonrpc_envelope(self):
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "id": 99, "method": "some/unknown"}
        result = _run(handle_request(request))
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 99


# ============================================================
# 14. handle_request() - id preservation
# ============================================================


class TestHandleRequestIdPreservation:
    """Tests for handle_request() preserving the request id."""

    def test_integer_id(self):
        from mcp.documents.server import handle_request

        result = _run(
            handle_request({"jsonrpc": "2.0", "id": 42, "method": "tools/list"})
        )
        assert result["id"] == 42

    def test_string_id(self):
        from mcp.documents.server import handle_request

        result = _run(
            handle_request(
                {"jsonrpc": "2.0", "id": "req-uuid-123", "method": "tools/list"}
            )
        )
        assert result["id"] == "req-uuid-123"

    def test_none_id(self):
        from mcp.documents.server import handle_request

        result = _run(
            handle_request({"jsonrpc": "2.0", "id": None, "method": "tools/list"})
        )
        assert result["id"] is None

    def test_missing_id_defaults_to_none(self):
        from mcp.documents.server import handle_request

        result = _run(handle_request({"jsonrpc": "2.0", "method": "tools/list"}))
        assert result["id"] is None

    def test_zero_id(self):
        from mcp.documents.server import handle_request

        result = _run(
            handle_request({"jsonrpc": "2.0", "id": 0, "method": "initialize"})
        )
        assert result["id"] == 0


# ============================================================
# 15. main() argparse tests
# ============================================================


class TestMainArgparse:
    """Tests for the main() function argument parsing."""

    def test_default_log_level(self):
        """Default log-level should be INFO."""
        from mcp.documents.server import main

        with patch(
            "mcp.documents.server.argparse.ArgumentParser.parse_args"
        ) as mock_parse, patch("mcp.documents.server.asyncio.run") as mock_run, patch(
            "mcp.documents.server.logging.basicConfig"
        ) as mock_log_cfg:
            mock_args = MagicMock()
            mock_args.log_level = "INFO"
            mock_parse.return_value = mock_args
            main()
            mock_log_cfg.assert_called_once()
            call_kwargs = mock_log_cfg.call_args
            assert call_kwargs[1]["level"] == logging.INFO

    def test_debug_log_level(self):
        from mcp.documents.server import main

        with patch(
            "mcp.documents.server.argparse.ArgumentParser.parse_args"
        ) as mock_parse, patch("mcp.documents.server.asyncio.run") as mock_run, patch(
            "mcp.documents.server.logging.basicConfig"
        ) as mock_log_cfg:
            mock_args = MagicMock()
            mock_args.log_level = "DEBUG"
            mock_parse.return_value = mock_args
            main()
            call_kwargs = mock_log_cfg.call_args
            assert call_kwargs[1]["level"] == logging.DEBUG

    def test_logs_to_stderr(self):
        """Logging should go to stderr, keeping stdout clean for JSON-RPC."""
        from mcp.documents.server import main

        with patch(
            "mcp.documents.server.argparse.ArgumentParser.parse_args"
        ) as mock_parse, patch("mcp.documents.server.asyncio.run") as mock_run, patch(
            "mcp.documents.server.logging.basicConfig"
        ) as mock_log_cfg:
            mock_args = MagicMock()
            mock_args.log_level = "INFO"
            mock_parse.return_value = mock_args
            main()
            call_kwargs = mock_log_cfg.call_args
            assert call_kwargs[1]["stream"] is sys.stderr

    def test_calls_asyncio_run_with_main_loop(self):
        from mcp.documents.server import main

        with patch(
            "mcp.documents.server.argparse.ArgumentParser.parse_args"
        ) as mock_parse, patch("mcp.documents.server.asyncio.run") as mock_run, patch(
            "mcp.documents.server.logging.basicConfig"
        ):
            mock_args = MagicMock()
            mock_args.log_level = "INFO"
            mock_parse.return_value = mock_args
            main()
            mock_run.assert_called_once()


# ============================================================
# 16. Edge cases
# ============================================================


class TestEdgeCases:
    """Edge-case tests for various server functions."""

    def test_handle_request_empty_params(self, mock_handlers):
        """tools/call with empty params dict should default name to ''."""
        from mcp.documents.server import handle_request

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {},
        }
        result = _run(handle_request(request))
        assert "error" in result["result"]

    def test_handle_request_missing_method(self):
        """Request with no method key should trigger unknown method error."""
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "id": 1}
        result = _run(handle_request(request))
        assert "error" in result["result"]
        assert "Unknown method" in result["result"]["error"]

    def test_handle_request_empty_method(self):
        """Empty string method should trigger unknown method error."""
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "id": 1, "method": ""}
        result = _run(handle_request(request))
        assert "error" in result["result"]

    def test_handle_request_missing_params_defaults(self, mock_handlers):
        """tools/call without params key uses empty dict for params."""
        from mcp.documents.server import handle_request

        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call"}
        result = _run(handle_request(request))
        # name defaults to "" which is unknown tool
        assert "error" in result["result"]

    def test_call_tool_with_extra_kwargs(self, mock_handlers):
        """Handler receives all arguments as kwargs."""
        from mcp.documents.server import _handle_call_tool

        _run(
            _handle_call_tool(
                "document_search",
                {"query": "test", "document_id": "doc-1", "top_k": 3},
            )
        )
        mock_handlers["document_search"].assert_awaited_once_with(
            query="test", document_id="doc-1", top_k=3
        )

    def test_call_tool_result_is_indented_json(self, mock_handlers):
        """Result text should be json.dumps(..., indent=2)."""
        from mcp.documents.server import _handle_call_tool

        mock_handlers["document_get"].return_value = {"id": "d", "title": "T"}
        result = _run(_handle_call_tool("document_get", {"document_id": "d"}))
        text = result["content"][0]["text"]
        expected = json.dumps({"id": "d", "title": "T"}, indent=2)
        assert text == expected


# ============================================================
# 17. main_loop() tests
# ============================================================


class TestMainLoop:
    """Tests for the main_loop() stdin/stdout loop."""

    def test_processes_valid_json_line(self):
        from mcp.documents.server import main_loop

        request = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        input_line = json.dumps(request) + "\n"

        out_buffer = StringIO()
        with patch("mcp.documents.server.sys.stdin") as mock_stdin, patch(
            "mcp.documents.server.sys.stdout", out_buffer
        ):
            mock_stdin.readline = MagicMock(side_effect=[input_line, ""])
            _run(main_loop())

        output = out_buffer.getvalue()
        response = json.loads(output.strip())
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["result"]["protocolVersion"] == "2024-11-05"

    def test_skips_empty_lines(self):
        from mcp.documents.server import main_loop

        request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        input_lines = ["\n", "  \n", json.dumps(request) + "\n", ""]

        out_buffer = StringIO()
        with patch("mcp.documents.server.sys.stdin") as mock_stdin, patch(
            "mcp.documents.server.sys.stdout", out_buffer
        ):
            mock_stdin.readline = MagicMock(side_effect=input_lines)
            _run(main_loop())

        output = out_buffer.getvalue().strip()
        response = json.loads(output)
        assert "tools" in response["result"]

    def test_handles_invalid_json(self):
        from mcp.documents.server import main_loop

        input_lines = ["not valid json\n", ""]

        out_buffer = StringIO()
        with patch("mcp.documents.server.sys.stdin") as mock_stdin, patch(
            "mcp.documents.server.sys.stdout", out_buffer
        ):
            mock_stdin.readline = MagicMock(side_effect=input_lines)
            _run(main_loop())

        output = out_buffer.getvalue().strip()
        response = json.loads(output)
        assert response["error"]["code"] == -32700
        assert response["error"]["message"] == "Parse error"

    def test_stops_on_eof(self):
        """Loop should exit cleanly when stdin returns empty string."""
        from mcp.documents.server import main_loop

        with patch("mcp.documents.server.sys.stdin") as mock_stdin, patch(
            "mcp.documents.server.sys.stdout", StringIO()
        ):
            mock_stdin.readline = MagicMock(return_value="")
            _run(main_loop())
            # If we get here without hanging, the test passes.

    def test_notification_produces_no_output(self):
        from mcp.documents.server import main_loop

        request = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        input_lines = [json.dumps(request) + "\n", ""]

        out_buffer = StringIO()
        with patch("mcp.documents.server.sys.stdin") as mock_stdin, patch(
            "mcp.documents.server.sys.stdout", out_buffer
        ):
            mock_stdin.readline = MagicMock(side_effect=input_lines)
            _run(main_loop())

        assert out_buffer.getvalue() == ""

    def test_multiple_requests(self):
        from mcp.documents.server import main_loop

        req1 = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
        req2 = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        input_lines = [json.dumps(req1) + "\n", json.dumps(req2) + "\n", ""]

        out_buffer = StringIO()
        with patch("mcp.documents.server.sys.stdin") as mock_stdin, patch(
            "mcp.documents.server.sys.stdout", out_buffer
        ):
            mock_stdin.readline = MagicMock(side_effect=input_lines)
            _run(main_loop())

        lines = out_buffer.getvalue().strip().split("\n")
        assert len(lines) == 2
        resp1 = json.loads(lines[0])
        resp2 = json.loads(lines[1])
        assert resp1["id"] == 1
        assert resp2["id"] == 2

    def test_keyboard_interrupt_exits_cleanly(self):
        from mcp.documents.server import main_loop

        with patch("mcp.documents.server.sys.stdin") as mock_stdin, patch(
            "mcp.documents.server.sys.stdout", StringIO()
        ):
            mock_stdin.readline = MagicMock(side_effect=KeyboardInterrupt)
            # Should not raise
            _run(main_loop())

    def test_invalid_json_error_has_null_id(self):
        """Parse error response should have id=None."""
        from mcp.documents.server import main_loop

        input_lines = ["{invalid json\n", ""]

        out_buffer = StringIO()
        with patch("mcp.documents.server.sys.stdin") as mock_stdin, patch(
            "mcp.documents.server.sys.stdout", out_buffer
        ):
            mock_stdin.readline = MagicMock(side_effect=input_lines)
            _run(main_loop())

        response = json.loads(out_buffer.getvalue().strip())
        assert response["id"] is None
        assert response["jsonrpc"] == "2.0"
