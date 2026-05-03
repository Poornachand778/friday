#!/usr/bin/env python3
"""MCP-compatible server for document processing.

This implementation exposes the document manager operations from
`mcp.documents.service` via a JSON-RPC style protocol over stdin/stdout,
matching the core concepts of the Model Context Protocol.

Usage:
    python -m mcp.documents.server
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.documents import service

LOGGER = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]


def _tool_definitions() -> List[ToolDefinition]:
    """Return the list of supported MCP tools."""

    return [
        ToolDefinition(
            name="document_ingest",
            description="Ingest a PDF document for conversational access. Processes pages with OCR, creates searchable chunks with embeddings.",
            input_schema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PDF file to ingest.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the document (e.g., 'Story by Robert McKee').",
                    },
                    "author": {
                        "type": "string",
                        "description": "Author name (optional).",
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
                        "description": "Type of document.",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["en", "te", "hi", "mixed"],
                        "default": "en",
                        "description": "Primary language of the document.",
                    },
                    "project": {
                        "type": "string",
                        "description": "Link to Friday project slug (optional).",
                    },
                },
                "required": ["file_path", "title"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="document_search",
            description="Search across ingested documents. Returns relevant passages with inline citations.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Limit search to specific document (optional).",
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
                        "description": "Filter by document type (optional).",
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project slug (optional).",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                        "description": "Number of results to return.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="document_get_context",
            description="Get document context for LLM generation. Returns formatted context with citations for answering questions about documents.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to find context for.",
                    },
                    "document_id": {
                        "type": "string",
                        "description": "Limit to specific document (optional).",
                    },
                    "max_chunks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 3,
                        "description": "Maximum number of chunks to include.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "minimum": 500,
                        "maximum": 10000,
                        "default": 4000,
                        "description": "Maximum total characters.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="document_list",
            description="List all ingested documents with their metadata.",
            input_schema={
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
                        "description": "Filter by document type (optional).",
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project slug (optional).",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "processing", "completed", "failed"],
                        "description": "Filter by processing status (optional).",
                    },
                },
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="document_get",
            description="Get detailed information about a specific document including chapters.",
            input_schema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The document UUID.",
                    },
                },
                "required": ["document_id"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="document_get_chapter",
            description="Get the full text of a specific chapter from a document.",
            input_schema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The document UUID.",
                    },
                    "chapter_title": {
                        "type": "string",
                        "description": "Title of the chapter to retrieve.",
                    },
                    "chapter_index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Chapter index (0-based) if title not provided.",
                    },
                },
                "required": ["document_id"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="document_status",
            description="Check the processing status of a document being ingested.",
            input_schema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The document UUID.",
                    },
                },
                "required": ["document_id"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="document_delete",
            description="Delete a document and all its associated chunks and embeddings.",
            input_schema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "The document UUID.",
                    },
                },
                "required": ["document_id"],
                "additionalProperties": False,
            },
        ),
    ]


TOOL_HANDLERS = {
    "document_ingest": service.document_ingest,
    "document_search": service.document_search,
    "document_get_context": service.document_get_context,
    "document_list": service.document_list,
    "document_get": service.document_get,
    "document_get_chapter": service.document_get_chapter,
    "document_status": service.document_status,
    "document_delete": service.document_delete,
}


def _handle_list_tools() -> Dict[str, Any]:
    """Handle tools/list request."""
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in _tool_definitions()
        ]
    }


async def _handle_call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tools/call request."""
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}

    try:
        result = await handler(**arguments)
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}
    except Exception as e:
        LOGGER.exception("Tool execution failed: %s", name)
        return {"error": str(e)}


async def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Route an incoming JSON-RPC request to the appropriate handler."""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")

    if method == "tools/list":
        result = _handle_list_tools()
    elif method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments", {})
        result = await _handle_call_tool(name, arguments)
    elif method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "friday-documents", "version": "1.0.0"},
        }
    elif method == "notifications/initialized":
        return {}  # No response for notifications
    else:
        result = {"error": f"Unknown method: {method}"}

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


async def main_loop():
    """Main stdin/stdout loop for MCP communication."""
    LOGGER.info("Friday Document MCP Server starting...")

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            line = line.strip()
            if not line:
                continue

            request = json.loads(line)
            response = await handle_request(request)

            if response:  # Don't send empty responses (notifications)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

        except json.JSONDecodeError as e:
            LOGGER.error("Invalid JSON: %s", e)
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            sys.stdout.write(json.dumps(error_response) + "\n")
            sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            LOGGER.exception("Unexpected error")


def main():
    parser = argparse.ArgumentParser(description="Friday Document MCP Server")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Log to stderr, keep stdout for JSON-RPC
    )

    asyncio.run(main_loop())


if __name__ == "__main__":
    main()
