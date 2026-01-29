#!/usr/bin/env python3
"""
Gmail MCP Server for Friday AI
==============================

Exposes Gmail operations via JSON-RPC for MCP agents.
Tools:
- send_screenplay: Send screenplay export via email
- send_email: Send simple email
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.gmail import service

LOGGER = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]


def _tool_definitions() -> List[ToolDefinition]:
    """Return the list of supported Gmail MCP tools."""
    return [
        ToolDefinition(
            name="send_screenplay",
            description="Send a screenplay via email with proper formatting.",
            input_schema={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of recipient email addresses.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line.",
                    },
                    "project_slug": {
                        "type": "string",
                        "description": "Screenplay project slug to send.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["pdf", "fountain", "html"],
                        "default": "pdf",
                        "description": "Export format for attachment.",
                    },
                    "include_html_body": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include HTML preview in email body.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Optional message to include before screenplay.",
                    },
                },
                "required": ["to", "subject", "project_slug"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="send_email",
            description="Send a simple email.",
            input_schema={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of recipient email addresses.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content.",
                    },
                    "html": {
                        "type": "boolean",
                        "default": False,
                        "description": "Whether body is HTML format.",
                    },
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        ),
    ]


class GmailMCPServer:
    """JSON-RPC handler exposing Gmail tools for MCP agents."""

    def __init__(self) -> None:
        self._tool_defs = _tool_definitions()

    def tool_send_screenplay(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send screenplay via email."""
        to = params.get("to", [])
        if not to:
            raise ValueError("to is required and must be non-empty")

        subject = params.get("subject")
        if not subject:
            raise ValueError("subject is required")

        project_slug = params.get("project_slug")
        if not project_slug:
            raise ValueError("project_slug is required")

        return service.send_screenplay_email(
            to=to,
            subject=subject,
            project_slug=project_slug,
            format=params.get("format", "pdf"),
            include_html_body=params.get("include_html_body", True),
            message_text=params.get("message"),
        )

    def tool_send_email(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send simple email."""
        to = params.get("to", [])
        if not to:
            raise ValueError("to is required and must be non-empty")

        subject = params.get("subject")
        if not subject:
            raise ValueError("subject is required")

        body = params.get("body")
        if not body:
            raise ValueError("body is required")

        return service.send_simple_email(
            to=to,
            subject=subject,
            body=body,
            html=params.get("html", False),
        )

    def handle_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        req_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params") or {}

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "0.1",
                    "capabilities": {"tools": {"list": True, "call": True}},
                }
            elif method == "list_tools":
                result = self._render_tool_list()
            elif method == "call_tool":
                result = self._dispatch_tool(params)
            elif method == "shutdown":
                result = {"ok": True}
            else:
                raise ValueError(f"Unknown method {method!r}")
            return {"id": req_id, "type": "response", "result": result}
        except Exception as exc:
            LOGGER.exception("MCP call failed: method=%s params=%s", method, params)
            return {
                "id": req_id,
                "type": "error",
                "error": {
                    "code": "internal_error",
                    "message": str(exc),
                },
            }

    def _render_tool_list(self) -> Dict[str, Any]:
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in self._tool_defs
            ]
        }

    def _dispatch_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not name:
            raise ValueError("params.name is required")

        handlers = {
            "send_screenplay": self.tool_send_screenplay,
            "send_email": self.tool_send_email,
        }
        handler = handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool {name!r}")

        result = handler(arguments)
        return {"content": result}


def _iter_stdin() -> Iterable[str]:
    """Yield non-empty lines from stdin."""
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        yield line


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Gmail MCP server")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))

    server = GmailMCPServer()

    for raw in _iter_stdin():
        payload: Optional[Dict[str, Any]] = None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            LOGGER.error("Invalid JSON payload: %s", exc)
            response = {
                "type": "error",
                "error": {
                    "code": "invalid_json",
                    "message": str(exc),
                },
            }
        else:
            response = server.handle_request(payload)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
        if payload and payload.get("method") == "shutdown":
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
