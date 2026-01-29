#!/usr/bin/env python3
"""
Voice MCP Server for Friday AI
===============================

Exposes voice operations via JSON-RPC for MCP agents.
Tools:
- voice_start_listening: Start daemon listening mode
- voice_stop_listening: Stop daemon
- voice_speak: Synthesize and speak text
- voice_get_status: Get daemon status
- voice_get_session: Get session details
- voice_export_training: Export for training
- voice_list_profiles: List TTS voice profiles
- voice_transcribe: Transcribe audio file
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

from mcp.voice import service

LOGGER = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]


def _tool_definitions() -> List[ToolDefinition]:
    """Return the list of supported Voice MCP tools."""
    return [
        ToolDefinition(
            name="voice_start_listening",
            description="Start the voice daemon listening for wake words.",
            input_schema={
                "type": "object",
                "properties": {
                    "config_path": {
                        "type": "string",
                        "description": "Path to voice config YAML (optional).",
                    },
                    "background": {
                        "type": "boolean",
                        "default": True,
                        "description": "Run in background process.",
                    },
                },
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="voice_stop_listening",
            description="Stop the voice daemon.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="voice_speak",
            description="Synthesize and speak text using TTS.",
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to speak.",
                    },
                    "language": {
                        "type": "string",
                        "default": "te",
                        "description": "Language code (te=Telugu, en=English).",
                    },
                    "profile": {
                        "type": "string",
                        "description": "Voice profile name (optional).",
                    },
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="voice_get_status",
            description="Get voice daemon status.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="voice_get_session",
            description="Get voice session details.",
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (optional, defaults to recent).",
                    },
                },
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="voice_export_training",
            description="Export voice data for training.",
            input_schema={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Output file path (optional).",
                    },
                    "min_length": {
                        "type": "integer",
                        "default": 5,
                        "description": "Minimum transcript length.",
                    },
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by languages (optional).",
                    },
                },
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="voice_list_profiles",
            description="List available TTS voice profiles.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="voice_transcribe",
            description="Transcribe an audio file.",
            input_schema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to audio file.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Force language (optional, auto-detect if not set).",
                    },
                },
                "required": ["audio_path"],
                "additionalProperties": False,
            },
        ),
    ]


class VoiceMCPServer:
    """JSON-RPC handler exposing voice tools for MCP agents."""

    def __init__(self) -> None:
        self._tool_defs = _tool_definitions()

    def tool_voice_start_listening(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start voice daemon."""
        return service.start_listening(
            config_path=params.get("config_path"),
            background=params.get("background", True),
        )

    def tool_voice_stop_listening(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop voice daemon."""
        return service.stop_listening()

    def tool_voice_speak(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Speak text via TTS."""
        text = params.get("text")
        if not text:
            raise ValueError("text is required")

        return service.speak_text(
            text=text,
            language=params.get("language", "te"),
            profile=params.get("profile"),
        )

    def tool_voice_get_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get daemon status."""
        return service.get_daemon_status()

    def tool_voice_get_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get session info."""
        return service.get_session_info(
            session_id=params.get("session_id"),
        )

    def tool_voice_export_training(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Export training data."""
        return service.export_training_data(
            output_path=params.get("output_path"),
            min_length=params.get("min_length", 5),
            languages=params.get("languages"),
        )

    def tool_voice_list_profiles(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List voice profiles."""
        return service.list_voice_profiles()

    def tool_voice_transcribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio file."""
        audio_path = params.get("audio_path")
        if not audio_path:
            raise ValueError("audio_path is required")

        return service.transcribe_audio(
            audio_path=audio_path,
            language=params.get("language"),
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
            "voice_start_listening": self.tool_voice_start_listening,
            "voice_stop_listening": self.tool_voice_stop_listening,
            "voice_speak": self.tool_voice_speak,
            "voice_get_status": self.tool_voice_get_status,
            "voice_get_session": self.tool_voice_get_session,
            "voice_export_training": self.tool_voice_export_training,
            "voice_list_profiles": self.tool_voice_list_profiles,
            "voice_transcribe": self.tool_voice_transcribe,
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
    parser = argparse.ArgumentParser(description="Voice MCP server")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))

    server = VoiceMCPServer()

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
