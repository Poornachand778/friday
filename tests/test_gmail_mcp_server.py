"""
Comprehensive tests for mcp/gmail/server.py — Gmail MCP JSON-RPC server.

Covers:
- ToolDefinition dataclass
- _tool_definitions() schema correctness
- GmailMCPServer:
    handle_request (initialize, list_tools, call_tool, shutdown, unknown)
    tool_send_screenplay validation + success
    tool_send_email validation + success
    _dispatch_tool routing
    _render_tool_list structure
- _iter_stdin with various inputs
- main() function (stdin/stdout loop, argparse, JSON decode errors, shutdown)
- Error propagation and edge cases
"""

from __future__ import annotations

import io
import json
import logging
import sys
from dataclasses import fields
from typing import Any, Dict
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from mcp.gmail.server import (
    GmailMCPServer,
    ToolDefinition,
    _iter_stdin,
    _tool_definitions,
    main,
)


# ===================================================================
# Section 1: ToolDefinition dataclass
# ===================================================================


class TestToolDefinition:
    """Tests for the ToolDefinition dataclass."""

    def test_creation_with_all_fields(self):
        td = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
        )
        assert td.name == "test_tool"
        assert td.description == "A test tool"
        assert td.input_schema == {"type": "object"}

    def test_field_names(self):
        names = [f.name for f in fields(ToolDefinition)]
        assert names == ["name", "description", "input_schema"]

    def test_equality(self):
        a = ToolDefinition("n", "d", {"type": "object"})
        b = ToolDefinition("n", "d", {"type": "object"})
        assert a == b

    def test_inequality(self):
        a = ToolDefinition("n1", "d", {})
        b = ToolDefinition("n2", "d", {})
        assert a != b

    def test_repr_contains_name(self):
        td = ToolDefinition("my_tool", "desc", {})
        assert "my_tool" in repr(td)


# ===================================================================
# Section 2: _tool_definitions()
# ===================================================================


class TestToolDefinitions:
    """Tests for the _tool_definitions() factory function."""

    def test_returns_list(self):
        result = _tool_definitions()
        assert isinstance(result, list)

    def test_returns_two_tools(self):
        assert len(_tool_definitions()) == 2

    def test_all_items_are_tool_definitions(self):
        for td in _tool_definitions():
            assert isinstance(td, ToolDefinition)

    def test_first_tool_is_send_screenplay(self):
        assert _tool_definitions()[0].name == "send_screenplay"

    def test_second_tool_is_send_email(self):
        assert _tool_definitions()[1].name == "send_email"

    # -- send_screenplay schema ----------------------------------------

    def test_send_screenplay_description(self):
        td = _tool_definitions()[0]
        assert "screenplay" in td.description.lower()

    def test_send_screenplay_schema_type(self):
        schema = _tool_definitions()[0].input_schema
        assert schema["type"] == "object"

    def test_send_screenplay_required_fields(self):
        schema = _tool_definitions()[0].input_schema
        assert sorted(schema["required"]) == ["project_slug", "subject", "to"]

    def test_send_screenplay_has_to_property(self):
        props = _tool_definitions()[0].input_schema["properties"]
        assert "to" in props
        assert props["to"]["type"] == "array"

    def test_send_screenplay_has_subject_property(self):
        props = _tool_definitions()[0].input_schema["properties"]
        assert "subject" in props
        assert props["subject"]["type"] == "string"

    def test_send_screenplay_has_project_slug_property(self):
        props = _tool_definitions()[0].input_schema["properties"]
        assert "project_slug" in props
        assert props["project_slug"]["type"] == "string"

    def test_send_screenplay_has_format_property(self):
        props = _tool_definitions()[0].input_schema["properties"]
        assert "format" in props
        assert set(props["format"]["enum"]) == {"pdf", "fountain", "html"}

    def test_send_screenplay_format_default(self):
        props = _tool_definitions()[0].input_schema["properties"]
        assert props["format"]["default"] == "pdf"

    def test_send_screenplay_has_include_html_body(self):
        props = _tool_definitions()[0].input_schema["properties"]
        assert "include_html_body" in props
        assert props["include_html_body"]["type"] == "boolean"

    def test_send_screenplay_include_html_body_default(self):
        props = _tool_definitions()[0].input_schema["properties"]
        assert props["include_html_body"]["default"] is True

    def test_send_screenplay_has_message_property(self):
        props = _tool_definitions()[0].input_schema["properties"]
        assert "message" in props

    def test_send_screenplay_no_additional_properties(self):
        assert _tool_definitions()[0].input_schema["additionalProperties"] is False

    # -- send_email schema ---------------------------------------------

    def test_send_email_description(self):
        td = _tool_definitions()[1]
        assert "email" in td.description.lower()

    def test_send_email_schema_type(self):
        schema = _tool_definitions()[1].input_schema
        assert schema["type"] == "object"

    def test_send_email_required_fields(self):
        schema = _tool_definitions()[1].input_schema
        assert sorted(schema["required"]) == ["body", "subject", "to"]

    def test_send_email_has_to_property(self):
        props = _tool_definitions()[1].input_schema["properties"]
        assert props["to"]["type"] == "array"

    def test_send_email_has_body_property(self):
        props = _tool_definitions()[1].input_schema["properties"]
        assert props["body"]["type"] == "string"

    def test_send_email_has_html_property(self):
        props = _tool_definitions()[1].input_schema["properties"]
        assert "html" in props
        assert props["html"]["type"] == "boolean"
        assert props["html"]["default"] is False

    def test_send_email_no_additional_properties(self):
        assert _tool_definitions()[1].input_schema["additionalProperties"] is False

    def test_send_email_property_count(self):
        props = _tool_definitions()[1].input_schema["properties"]
        assert len(props) == 4  # to, subject, body, html


# ===================================================================
# Section 3: GmailMCPServer – _render_tool_list
# ===================================================================


class TestRenderToolList:
    """Tests for GmailMCPServer._render_tool_list()."""

    def setup_method(self):
        self.server = GmailMCPServer()

    def test_returns_dict_with_tools_key(self):
        result = self.server._render_tool_list()
        assert "tools" in result

    def test_tools_is_list(self):
        assert isinstance(self.server._render_tool_list()["tools"], list)

    def test_tools_count(self):
        assert len(self.server._render_tool_list()["tools"]) == 2

    def test_each_tool_has_name(self):
        for tool in self.server._render_tool_list()["tools"]:
            assert "name" in tool

    def test_each_tool_has_description(self):
        for tool in self.server._render_tool_list()["tools"]:
            assert "description" in tool

    def test_each_tool_has_input_schema(self):
        for tool in self.server._render_tool_list()["tools"]:
            assert "inputSchema" in tool

    def test_tool_names_match_definitions(self):
        names = [t["name"] for t in self.server._render_tool_list()["tools"]]
        assert names == ["send_screenplay", "send_email"]


# ===================================================================
# Section 4: GmailMCPServer – tool_send_screenplay
# ===================================================================


@patch("mcp.gmail.server.service")
class TestToolSendScreenplay:
    """Validation and success paths for tool_send_screenplay."""

    def _server(self):
        return GmailMCPServer()

    def test_missing_to_raises(self, mock_service):
        with pytest.raises(ValueError, match="to is required"):
            self._server().tool_send_screenplay({"subject": "s", "project_slug": "p"})

    def test_empty_to_raises(self, mock_service):
        with pytest.raises(ValueError, match="to is required"):
            self._server().tool_send_screenplay(
                {"to": [], "subject": "s", "project_slug": "p"}
            )

    def test_missing_subject_raises(self, mock_service):
        with pytest.raises(ValueError, match="subject is required"):
            self._server().tool_send_screenplay(
                {"to": ["a@b.com"], "project_slug": "p"}
            )

    def test_empty_subject_raises(self, mock_service):
        with pytest.raises(ValueError, match="subject is required"):
            self._server().tool_send_screenplay(
                {"to": ["a@b.com"], "subject": "", "project_slug": "p"}
            )

    def test_missing_project_slug_raises(self, mock_service):
        with pytest.raises(ValueError, match="project_slug is required"):
            self._server().tool_send_screenplay({"to": ["a@b.com"], "subject": "s"})

    def test_empty_project_slug_raises(self, mock_service):
        with pytest.raises(ValueError, match="project_slug is required"):
            self._server().tool_send_screenplay(
                {"to": ["a@b.com"], "subject": "s", "project_slug": ""}
            )

    def test_success_calls_service(self, mock_service):
        mock_service.send_screenplay_email.return_value = {"sent": True}
        result = self._server().tool_send_screenplay(
            {"to": ["a@b.com"], "subject": "Sub", "project_slug": "slug"}
        )
        mock_service.send_screenplay_email.assert_called_once_with(
            to=["a@b.com"],
            subject="Sub",
            project_slug="slug",
            format="pdf",
            include_html_body=True,
            message_text=None,
        )
        assert result == {"sent": True}

    def test_custom_format(self, mock_service):
        mock_service.send_screenplay_email.return_value = {}
        self._server().tool_send_screenplay(
            {
                "to": ["x@y.z"],
                "subject": "s",
                "project_slug": "p",
                "format": "html",
            }
        )
        _, kwargs = mock_service.send_screenplay_email.call_args
        assert kwargs["format"] == "html"

    def test_custom_include_html_body(self, mock_service):
        mock_service.send_screenplay_email.return_value = {}
        self._server().tool_send_screenplay(
            {
                "to": ["x@y.z"],
                "subject": "s",
                "project_slug": "p",
                "include_html_body": False,
            }
        )
        _, kwargs = mock_service.send_screenplay_email.call_args
        assert kwargs["include_html_body"] is False

    def test_custom_message(self, mock_service):
        mock_service.send_screenplay_email.return_value = {}
        self._server().tool_send_screenplay(
            {
                "to": ["x@y.z"],
                "subject": "s",
                "project_slug": "p",
                "message": "Note",
            }
        )
        _, kwargs = mock_service.send_screenplay_email.call_args
        assert kwargs["message_text"] == "Note"

    def test_multiple_recipients(self, mock_service):
        mock_service.send_screenplay_email.return_value = {}
        self._server().tool_send_screenplay(
            {
                "to": ["a@b.com", "c@d.com"],
                "subject": "s",
                "project_slug": "p",
            }
        )
        _, kwargs = mock_service.send_screenplay_email.call_args
        assert kwargs["to"] == ["a@b.com", "c@d.com"]

    def test_returns_service_result(self, mock_service):
        expected = {"sent": True, "message_id": "abc123"}
        mock_service.send_screenplay_email.return_value = expected
        result = self._server().tool_send_screenplay(
            {"to": ["a@b.com"], "subject": "s", "project_slug": "p"}
        )
        assert result is expected


# ===================================================================
# Section 5: GmailMCPServer – tool_send_email
# ===================================================================


@patch("mcp.gmail.server.service")
class TestToolSendEmail:
    """Validation and success paths for tool_send_email."""

    def _server(self):
        return GmailMCPServer()

    def test_missing_to_raises(self, mock_service):
        with pytest.raises(ValueError, match="to is required"):
            self._server().tool_send_email({"subject": "s", "body": "b"})

    def test_empty_to_raises(self, mock_service):
        with pytest.raises(ValueError, match="to is required"):
            self._server().tool_send_email({"to": [], "subject": "s", "body": "b"})

    def test_missing_subject_raises(self, mock_service):
        with pytest.raises(ValueError, match="subject is required"):
            self._server().tool_send_email({"to": ["a@b.com"], "body": "b"})

    def test_empty_subject_raises(self, mock_service):
        with pytest.raises(ValueError, match="subject is required"):
            self._server().tool_send_email(
                {"to": ["a@b.com"], "subject": "", "body": "b"}
            )

    def test_missing_body_raises(self, mock_service):
        with pytest.raises(ValueError, match="body is required"):
            self._server().tool_send_email({"to": ["a@b.com"], "subject": "s"})

    def test_empty_body_raises(self, mock_service):
        with pytest.raises(ValueError, match="body is required"):
            self._server().tool_send_email(
                {"to": ["a@b.com"], "subject": "s", "body": ""}
            )

    def test_success_calls_service(self, mock_service):
        mock_service.send_simple_email.return_value = {"sent": True}
        result = self._server().tool_send_email(
            {"to": ["a@b.com"], "subject": "Sub", "body": "Hello"}
        )
        mock_service.send_simple_email.assert_called_once_with(
            to=["a@b.com"],
            subject="Sub",
            body="Hello",
            html=False,
        )
        assert result == {"sent": True}

    def test_html_flag_true(self, mock_service):
        mock_service.send_simple_email.return_value = {}
        self._server().tool_send_email(
            {"to": ["a@b.com"], "subject": "s", "body": "<b>Hi</b>", "html": True}
        )
        _, kwargs = mock_service.send_simple_email.call_args
        assert kwargs["html"] is True

    def test_html_flag_defaults_false(self, mock_service):
        mock_service.send_simple_email.return_value = {}
        self._server().tool_send_email(
            {"to": ["a@b.com"], "subject": "s", "body": "hi"}
        )
        _, kwargs = mock_service.send_simple_email.call_args
        assert kwargs["html"] is False

    def test_multiple_recipients(self, mock_service):
        mock_service.send_simple_email.return_value = {}
        self._server().tool_send_email(
            {"to": ["a@b.com", "c@d.com"], "subject": "s", "body": "hi"}
        )
        _, kwargs = mock_service.send_simple_email.call_args
        assert kwargs["to"] == ["a@b.com", "c@d.com"]

    def test_returns_service_result(self, mock_service):
        expected = {"sent": True, "message_id": "xyz"}
        mock_service.send_simple_email.return_value = expected
        result = self._server().tool_send_email(
            {"to": ["a@b.com"], "subject": "s", "body": "b"}
        )
        assert result is expected


# ===================================================================
# Section 6: GmailMCPServer – _dispatch_tool
# ===================================================================


@patch("mcp.gmail.server.service")
class TestDispatchTool:
    """Tests for _dispatch_tool routing and error handling."""

    def _server(self):
        return GmailMCPServer()

    def test_missing_name_raises(self, mock_service):
        with pytest.raises(ValueError, match="params.name is required"):
            self._server()._dispatch_tool({})

    def test_empty_name_raises(self, mock_service):
        with pytest.raises(ValueError, match="params.name is required"):
            self._server()._dispatch_tool({"name": ""})

    def test_unknown_tool_raises(self, mock_service):
        with pytest.raises(ValueError, match="Unknown tool"):
            self._server()._dispatch_tool({"name": "nonexistent"})

    def test_dispatch_send_email(self, mock_service):
        mock_service.send_simple_email.return_value = {"sent": True}
        result = self._server()._dispatch_tool(
            {
                "name": "send_email",
                "arguments": {"to": ["a@b.com"], "subject": "s", "body": "b"},
            }
        )
        assert result == {"content": {"sent": True}}

    def test_dispatch_send_screenplay(self, mock_service):
        mock_service.send_screenplay_email.return_value = {"sent": True}
        result = self._server()._dispatch_tool(
            {
                "name": "send_screenplay",
                "arguments": {
                    "to": ["a@b.com"],
                    "subject": "s",
                    "project_slug": "p",
                },
            }
        )
        assert result == {"content": {"sent": True}}

    def test_wraps_result_in_content(self, mock_service):
        mock_service.send_simple_email.return_value = {"k": "v"}
        result = self._server()._dispatch_tool(
            {
                "name": "send_email",
                "arguments": {"to": ["a@b.com"], "subject": "s", "body": "b"},
            }
        )
        assert "content" in result
        assert result["content"] == {"k": "v"}

    def test_none_arguments_treated_as_empty(self, mock_service):
        """When arguments is None, handler receives empty dict, triggering validation."""
        with pytest.raises(ValueError, match="to is required"):
            self._server()._dispatch_tool({"name": "send_email", "arguments": None})

    def test_missing_arguments_treated_as_empty(self, mock_service):
        with pytest.raises(ValueError, match="to is required"):
            self._server()._dispatch_tool({"name": "send_email"})

    def test_name_none_raises(self, mock_service):
        with pytest.raises(ValueError, match="params.name is required"):
            self._server()._dispatch_tool({"name": None})


# ===================================================================
# Section 7: GmailMCPServer – handle_request
# ===================================================================


@patch("mcp.gmail.server.service")
class TestHandleRequest:
    """Tests for handle_request JSON-RPC routing."""

    def _server(self):
        return GmailMCPServer()

    # -- initialize ----------------------------------------------------

    def test_initialize_returns_protocol_version(self, mock_service):
        resp = self._server().handle_request({"id": 1, "method": "initialize"})
        assert resp["result"]["protocolVersion"] == "0.1"

    def test_initialize_has_capabilities(self, mock_service):
        resp = self._server().handle_request({"id": 1, "method": "initialize"})
        caps = resp["result"]["capabilities"]
        assert caps["tools"]["list"] is True
        assert caps["tools"]["call"] is True

    def test_initialize_response_shape(self, mock_service):
        resp = self._server().handle_request({"id": "abc", "method": "initialize"})
        assert resp["id"] == "abc"
        assert resp["type"] == "response"
        assert "result" in resp

    # -- list_tools ----------------------------------------------------

    def test_list_tools(self, mock_service):
        resp = self._server().handle_request({"id": 2, "method": "list_tools"})
        assert resp["type"] == "response"
        assert "tools" in resp["result"]
        assert len(resp["result"]["tools"]) == 2

    def test_list_tools_preserves_id(self, mock_service):
        resp = self._server().handle_request({"id": 42, "method": "list_tools"})
        assert resp["id"] == 42

    # -- call_tool -----------------------------------------------------

    def test_call_tool_success(self, mock_service):
        mock_service.send_simple_email.return_value = {"sent": True}
        resp = self._server().handle_request(
            {
                "id": 3,
                "method": "call_tool",
                "params": {
                    "name": "send_email",
                    "arguments": {"to": ["a@b.com"], "subject": "s", "body": "b"},
                },
            }
        )
        assert resp["type"] == "response"
        assert resp["result"]["content"] == {"sent": True}

    def test_call_tool_invalid_name_returns_error(self, mock_service):
        resp = self._server().handle_request(
            {
                "id": 4,
                "method": "call_tool",
                "params": {"name": "bogus"},
            }
        )
        assert resp["type"] == "error"
        assert "Unknown tool" in resp["error"]["message"]

    def test_call_tool_validation_error_returns_error(self, mock_service):
        resp = self._server().handle_request(
            {
                "id": 5,
                "method": "call_tool",
                "params": {
                    "name": "send_email",
                    "arguments": {"to": [], "subject": "s", "body": "b"},
                },
            }
        )
        assert resp["type"] == "error"
        assert "to is required" in resp["error"]["message"]

    # -- shutdown ------------------------------------------------------

    def test_shutdown(self, mock_service):
        resp = self._server().handle_request({"id": 6, "method": "shutdown"})
        assert resp["type"] == "response"
        assert resp["result"] == {"ok": True}

    def test_shutdown_preserves_id(self, mock_service):
        resp = self._server().handle_request({"id": "sid", "method": "shutdown"})
        assert resp["id"] == "sid"

    # -- unknown method ------------------------------------------------

    def test_unknown_method(self, mock_service):
        resp = self._server().handle_request({"id": 7, "method": "something"})
        assert resp["type"] == "error"
        assert "Unknown method" in resp["error"]["message"]

    def test_unknown_method_error_code(self, mock_service):
        resp = self._server().handle_request({"id": 8, "method": "xyz"})
        assert resp["error"]["code"] == "internal_error"

    # -- error handling ------------------------------------------------

    def test_exception_in_tool_returns_error(self, mock_service):
        mock_service.send_simple_email.side_effect = RuntimeError("boom")
        resp = self._server().handle_request(
            {
                "id": 9,
                "method": "call_tool",
                "params": {
                    "name": "send_email",
                    "arguments": {"to": ["a@b.com"], "subject": "s", "body": "b"},
                },
            }
        )
        assert resp["type"] == "error"
        assert "boom" in resp["error"]["message"]
        assert resp["id"] == 9

    def test_error_code_is_internal_error(self, mock_service):
        mock_service.send_simple_email.side_effect = Exception("fail")
        resp = self._server().handle_request(
            {
                "id": 10,
                "method": "call_tool",
                "params": {
                    "name": "send_email",
                    "arguments": {"to": ["a@b.com"], "subject": "s", "body": "b"},
                },
            }
        )
        assert resp["error"]["code"] == "internal_error"

    # -- edge cases ----------------------------------------------------

    def test_missing_id_returns_none_id(self, mock_service):
        resp = self._server().handle_request({"method": "initialize"})
        assert resp["id"] is None

    def test_missing_method_returns_error(self, mock_service):
        resp = self._server().handle_request({"id": 11})
        assert resp["type"] == "error"
        assert "Unknown method" in resp["error"]["message"]

    def test_params_default_to_empty_dict(self, mock_service):
        """When params is missing or None, handle_request should not crash."""
        resp = self._server().handle_request(
            {"id": 12, "method": "list_tools", "params": None}
        )
        assert resp["type"] == "response"

    def test_call_tool_missing_params_name(self, mock_service):
        resp = self._server().handle_request(
            {"id": 13, "method": "call_tool", "params": {}}
        )
        assert resp["type"] == "error"

    def test_integer_id_preserved(self, mock_service):
        resp = self._server().handle_request({"id": 999, "method": "shutdown"})
        assert resp["id"] == 999

    def test_string_id_preserved(self, mock_service):
        resp = self._server().handle_request({"id": "req-42", "method": "shutdown"})
        assert resp["id"] == "req-42"


# ===================================================================
# Section 8: _iter_stdin
# ===================================================================


class TestIterStdin:
    """Tests for the _iter_stdin() generator."""

    def test_empty_input(self):
        with patch("mcp.gmail.server.sys.stdin", io.StringIO("")):
            assert list(_iter_stdin()) == []

    def test_single_line(self):
        with patch("mcp.gmail.server.sys.stdin", io.StringIO("hello\n")):
            assert list(_iter_stdin()) == ["hello"]

    def test_multiple_lines(self):
        with patch("mcp.gmail.server.sys.stdin", io.StringIO("one\ntwo\nthree\n")):
            assert list(_iter_stdin()) == ["one", "two", "three"]

    def test_blank_lines_skipped(self):
        with patch("mcp.gmail.server.sys.stdin", io.StringIO("a\n\n\nb\n")):
            assert list(_iter_stdin()) == ["a", "b"]

    def test_whitespace_only_lines_skipped(self):
        with patch("mcp.gmail.server.sys.stdin", io.StringIO("a\n   \n\t\nb\n")):
            assert list(_iter_stdin()) == ["a", "b"]

    def test_strips_whitespace(self):
        with patch("mcp.gmail.server.sys.stdin", io.StringIO("  hello  \n")):
            assert list(_iter_stdin()) == ["hello"]

    def test_no_trailing_newline(self):
        """readline returns '' on EOF (no more data)."""
        with patch("mcp.gmail.server.sys.stdin", io.StringIO("hi")):
            assert list(_iter_stdin()) == ["hi"]

    def test_json_lines(self):
        lines = '{"a":1}\n{"b":2}\n'
        with patch("mcp.gmail.server.sys.stdin", io.StringIO(lines)):
            result = list(_iter_stdin())
            assert len(result) == 2
            assert json.loads(result[0]) == {"a": 1}


# ===================================================================
# Section 9: main() function
# ===================================================================


@patch("mcp.gmail.server.service")
class TestMain:
    """Tests for the main() entry point."""

    def _run_main(self, stdin_text, argv=None):
        """Helper: run main() with given stdin and capture stdout."""
        out = io.StringIO()
        with patch("mcp.gmail.server.sys.stdin", io.StringIO(stdin_text)):
            with patch("mcp.gmail.server.sys.stdout", out):
                rc = main(argv or [])
        return rc, out.getvalue()

    def test_returns_zero(self, mock_service):
        rc, _ = self._run_main("")
        assert rc == 0

    def test_initialize_request(self, mock_service):
        payload = json.dumps({"id": 1, "method": "initialize"})
        rc, out = self._run_main(payload + "\n")
        resp = json.loads(out.strip())
        assert resp["id"] == 1
        assert resp["type"] == "response"
        assert "protocolVersion" in resp["result"]

    def test_list_tools_request(self, mock_service):
        payload = json.dumps({"id": 2, "method": "list_tools"})
        rc, out = self._run_main(payload + "\n")
        resp = json.loads(out.strip())
        assert len(resp["result"]["tools"]) == 2

    def test_shutdown_breaks_loop(self, mock_service):
        lines = "\n".join(
            [
                json.dumps({"id": 1, "method": "initialize"}),
                json.dumps({"id": 2, "method": "shutdown"}),
                json.dumps({"id": 3, "method": "list_tools"}),
            ]
        )
        rc, out = self._run_main(lines + "\n")
        responses = [json.loads(l) for l in out.strip().split("\n")]
        # Should have initialize + shutdown only; list_tools never processed
        assert len(responses) == 2
        assert responses[0]["id"] == 1
        assert responses[1]["id"] == 2
        assert responses[1]["result"] == {"ok": True}

    def test_invalid_json_returns_error(self, mock_service):
        rc, out = self._run_main("NOT JSON\n")
        resp = json.loads(out.strip())
        assert resp["type"] == "error"
        assert resp["error"]["code"] == "invalid_json"

    def test_invalid_json_does_not_break_loop(self, mock_service):
        lines = "BADJSON\n" + json.dumps({"id": 1, "method": "initialize"}) + "\n"
        rc, out = self._run_main(lines)
        responses = [json.loads(l) for l in out.strip().split("\n")]
        assert len(responses) == 2
        assert responses[0]["type"] == "error"
        assert responses[1]["type"] == "response"

    def test_call_tool_through_main(self, mock_service):
        mock_service.send_simple_email.return_value = {"sent": True, "message_id": "m1"}
        payload = json.dumps(
            {
                "id": 10,
                "method": "call_tool",
                "params": {
                    "name": "send_email",
                    "arguments": {"to": ["x@y.z"], "subject": "hi", "body": "hey"},
                },
            }
        )
        rc, out = self._run_main(payload + "\n")
        resp = json.loads(out.strip())
        assert resp["type"] == "response"
        assert resp["result"]["content"]["sent"] is True

    def test_log_level_argument(self, mock_service):
        rc, _ = self._run_main("", argv=["--log-level", "DEBUG"])
        assert rc == 0

    def test_log_level_default(self, mock_service):
        """Default log level should not cause errors."""
        rc, _ = self._run_main("")
        assert rc == 0

    def test_multiple_requests(self, mock_service):
        mock_service.send_simple_email.return_value = {"sent": True}
        payloads = [
            json.dumps({"id": 1, "method": "initialize"}),
            json.dumps({"id": 2, "method": "list_tools"}),
            json.dumps(
                {
                    "id": 3,
                    "method": "call_tool",
                    "params": {
                        "name": "send_email",
                        "arguments": {"to": ["a@b.c"], "subject": "s", "body": "b"},
                    },
                }
            ),
        ]
        rc, out = self._run_main("\n".join(payloads) + "\n")
        responses = [json.loads(l) for l in out.strip().split("\n")]
        assert len(responses) == 3
        assert all(r["type"] == "response" for r in responses)

    def test_each_response_ends_with_newline(self, mock_service):
        payload = json.dumps({"id": 1, "method": "initialize"})
        rc, out = self._run_main(payload + "\n")
        assert out.endswith("\n")

    def test_empty_stdin(self, mock_service):
        rc, out = self._run_main("")
        assert rc == 0
        assert out == ""

    def test_invalid_json_error_has_no_id(self, mock_service):
        """Invalid JSON payload should result in response with no 'id' key (or missing)."""
        rc, out = self._run_main("{bad json\n")
        resp = json.loads(out.strip())
        assert resp["type"] == "error"
        # The response for bad JSON does not go through handle_request,
        # so there may be no id field at all.
        assert "id" not in resp or resp.get("id") is None


# ===================================================================
# Section 10: GmailMCPServer constructor
# ===================================================================


class TestGmailMCPServerInit:
    """Tests for GmailMCPServer construction."""

    def test_has_tool_defs(self):
        server = GmailMCPServer()
        assert hasattr(server, "_tool_defs")

    def test_tool_defs_are_tool_definitions(self):
        server = GmailMCPServer()
        for td in server._tool_defs:
            assert isinstance(td, ToolDefinition)

    def test_tool_defs_count(self):
        server = GmailMCPServer()
        assert len(server._tool_defs) == 2


# ===================================================================
# Section 11: Additional edge cases and integration-style tests
# ===================================================================


@patch("mcp.gmail.server.service")
class TestEdgeCases:
    """Additional edge-case tests."""

    def test_call_tool_screenplay_through_handle_request(self, mock_service):
        mock_service.send_screenplay_email.return_value = {
            "sent": True,
            "message_id": "sp1",
        }
        server = GmailMCPServer()
        resp = server.handle_request(
            {
                "id": 100,
                "method": "call_tool",
                "params": {
                    "name": "send_screenplay",
                    "arguments": {
                        "to": ["director@studio.com"],
                        "subject": "Final Draft",
                        "project_slug": "my-movie",
                        "format": "fountain",
                        "include_html_body": False,
                        "message": "Please review",
                    },
                },
            }
        )
        assert resp["type"] == "response"
        assert resp["result"]["content"]["sent"] is True
        mock_service.send_screenplay_email.assert_called_once_with(
            to=["director@studio.com"],
            subject="Final Draft",
            project_slug="my-movie",
            format="fountain",
            include_html_body=False,
            message_text="Please review",
        )

    def test_service_exception_propagates_as_error_response(self, mock_service):
        mock_service.send_screenplay_email.side_effect = ConnectionError("timeout")
        server = GmailMCPServer()
        resp = server.handle_request(
            {
                "id": 200,
                "method": "call_tool",
                "params": {
                    "name": "send_screenplay",
                    "arguments": {
                        "to": ["a@b.com"],
                        "subject": "s",
                        "project_slug": "p",
                    },
                },
            }
        )
        assert resp["type"] == "error"
        assert "timeout" in resp["error"]["message"]

    def test_handle_request_with_empty_params_dict(self, mock_service):
        resp = GmailMCPServer().handle_request(
            {"id": 300, "method": "call_tool", "params": {}}
        )
        assert resp["type"] == "error"
        assert "params.name is required" in resp["error"]["message"]

    def test_dispatch_tool_name_case_sensitive(self, mock_service):
        with pytest.raises(ValueError, match="Unknown tool"):
            GmailMCPServer()._dispatch_tool({"name": "Send_Email"})

    def test_dispatch_tool_name_with_spaces(self, mock_service):
        with pytest.raises(ValueError, match="Unknown tool"):
            GmailMCPServer()._dispatch_tool({"name": "send email"})

    def test_handle_request_method_none(self, mock_service):
        resp = GmailMCPServer().handle_request({"id": 400, "method": None})
        assert resp["type"] == "error"

    def test_handle_request_with_extra_fields(self, mock_service):
        """Extra fields in the payload should be ignored."""
        resp = GmailMCPServer().handle_request(
            {"id": 500, "method": "initialize", "extra": "data", "jsonrpc": "2.0"}
        )
        assert resp["type"] == "response"

    def test_call_tool_with_no_params_key(self, mock_service):
        """When params key is entirely missing, handle_request defaults to {}."""
        resp = GmailMCPServer().handle_request({"id": 600, "method": "call_tool"})
        assert resp["type"] == "error"
        assert "params.name is required" in resp["error"]["message"]
