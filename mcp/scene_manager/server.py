#!/usr/bin/env python3
"""Minimal MCP-compatible server for screenplay scene management.

This implementation exposes the scene manager operations from
`mcp.scene_manager.service` via a JSON-RPC style protocol over stdin/stdout,
matching the core concepts of the Model Context Protocol (tools that can be
listed and invoked by name with structured arguments).

It keeps the FastAPI service available for HTTP clients while letting MCP-based
agents talk to the same business logic without re-implementing database or
embedding concerns.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db.screenplay_schema import ScreenplayProject, ScreenplayScene
from mcp.scene_manager import service


LOGGER = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]


def _tool_definitions(default_project: str) -> List[ToolDefinition]:
    """Return the list of supported MCP tools."""

    def project_property(required: bool = False) -> Dict[str, Any]:
        prop = {
            "type": "string",
            "description": "Slug of the script project (default: ``%s``)."
            % default_project,
        }
        if not required:
            prop["default"] = default_project
        return prop

    return [
        ToolDefinition(
            name="scene_search",
            description="Semantic search across screenplay scenes.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    },
                    "project_slug": project_property(False),
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="scene_get",
            description="Retrieve canonical details for a scene.",
            input_schema={
                "type": "object",
                "properties": {
                    "scene_number": {
                        "type": "integer",
                        "description": "Scene number (1, 2, 3, ...).",
                    },
                    "scene_id": {
                        "type": "integer",
                        "description": "Numeric database ID (alternative to scene_number).",
                    },
                    "project_slug": project_property(False),
                },
                "anyOf": [
                    {"required": ["scene_number"]},
                    {"required": ["scene_id"]},
                ],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="scene_update",
            description="Update scene metadata (status, title, summary, tags, order).",
            input_schema={
                "type": "object",
                "properties": {
                    "scene_number": {
                        "type": "integer",
                        "description": "Scene number to update.",
                    },
                    "project_slug": project_property(False),
                    "title": {
                        "type": "string",
                        "description": "Scene title/name.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Scene summary or description.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorizing the scene.",
                    },
                    "narrative_order": {
                        "type": "number",
                        "description": "Explicit float order to assign (optional).",
                    },
                    "status": {
                        "type": "string",
                        "description": "Scene status e.g. draft/revision/locked.",
                    },
                },
                "required": ["scene_number"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="scene_reorder",
            description="Move a scene before/after another scene.",
            input_schema={
                "type": "object",
                "properties": {
                    "scene_number": {
                        "type": "integer",
                        "description": "Scene number to reposition.",
                    },
                    "project_slug": project_property(False),
                    "after_scene": {
                        "type": "integer",
                        "description": "Scene number to place after (optional).",
                    },
                    "before_scene": {
                        "type": "integer",
                        "description": "Scene number to place before (optional).",
                    },
                },
                "required": ["scene_number"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="scene_link",
            description="Create a semantic or sequential link between two scenes.",
            input_schema={
                "type": "object",
                "properties": {
                    "from_scene": {
                        "type": "integer",
                        "description": "Origin scene number.",
                    },
                    "to_scene": {
                        "type": "integer",
                        "description": "Destination scene number.",
                    },
                    "relation_type": {
                        "type": "string",
                        "default": "sequence",
                        "description": "Relationship label (sequence/flashback/alternate).",
                    },
                    "project_slug": project_property(False),
                },
                "required": ["from_scene", "to_scene"],
                "additionalProperties": False,
            },
        ),
    ]


class SceneResolutionError(RuntimeError):
    """Raised when a project or scene code cannot be resolved."""


class SceneManagerMCPServer:
    """JSON-RPC handler exposing screenplay tools for MCP agents."""

    def __init__(self, default_project: str) -> None:
        self._default_project = default_project
        self._engine = service.get_engine_instance()
        self._tool_defs = _tool_definitions(default_project)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _resolve_project_id(self, session: Session, slug: Optional[str]) -> int:
        query = select(ScreenplayProject.id)
        query = query.where(ScreenplayProject.slug == (slug or self._default_project))
        project_id = session.execute(query).scalar_one_or_none()
        if project_id is None:
            raise SceneResolutionError(f"Unknown project slug: {slug!r}")
        return project_id

    def _resolve_scene_id(
        self,
        session: Session,
        project_id: int,
        *,
        scene_number: Optional[int] = None,
        scene_id: Optional[int] = None,
    ) -> int:
        if scene_id is not None:
            exists = session.execute(
                select(ScreenplayScene.id).where(
                    ScreenplayScene.id == scene_id,
                    ScreenplayScene.project_id == project_id,
                )
            ).scalar_one_or_none()
            if exists is None:
                raise SceneResolutionError(
                    f"Scene id {scene_id} does not belong to project {project_id}"
                )
            return scene_id

        if not scene_number:
            raise SceneResolutionError("A scene_number or scene_id must be supplied")

        result = session.execute(
            select(ScreenplayScene.id).where(
                ScreenplayScene.project_id == project_id,
                ScreenplayScene.scene_number == scene_number,
            )
        ).scalar_one_or_none()
        if result is None:
            raise SceneResolutionError(
                f"Unknown scene number {scene_number!r} for project {project_id}"
            )
        return result

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------
    def tool_scene_search(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = params["query"].strip()
        if not query:
            raise ValueError("query must not be empty")
        top_k = int(params.get("top_k", 5))
        project_slug = params.get("project_slug") or self._default_project
        return service.search_scenes(query, project_slug=project_slug, top_k=top_k)

    def tool_scene_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        project_slug = params.get("project_slug") or self._default_project
        scene_number = params.get("scene_number")
        scene_id = params.get("scene_id")
        with Session(self._engine) as session:
            project_id = self._resolve_project_id(session, project_slug)
            resolved_id = self._resolve_scene_id(
                session, project_id, scene_number=scene_number, scene_id=scene_id
            )
        return service.get_scene_detail(resolved_id)

    def tool_scene_update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        project_slug = params.get("project_slug") or self._default_project
        scene_number = params.get("scene_number")
        if not scene_number:
            raise ValueError("scene_number is required")

        with Session(self._engine) as session:
            project_id = self._resolve_project_id(session, project_slug)
            resolved_id = self._resolve_scene_id(
                session, project_id, scene_number=scene_number
            )

        changed = service.update_scene(
            resolved_id,
            title=params.get("title"),
            summary=params.get("summary"),
            tags=params.get("tags"),
            narrative_order=params.get("narrative_order"),
            status=params.get("status"),
        )
        detail = service.get_scene_detail(resolved_id)
        return {"updated": bool(changed), "scene": detail}

    def tool_scene_reorder(self, params: Dict[str, Any]) -> Dict[str, Any]:
        project_slug = params.get("project_slug") or self._default_project
        scene_number = params.get("scene_number")
        if not scene_number:
            raise ValueError("scene_number is required")

        after_scene = params.get("after_scene")
        before_scene = params.get("before_scene")
        if not any([after_scene, before_scene]):
            raise ValueError("Provide at least one of after_scene or before_scene")

        with Session(self._engine) as session:
            project_id = self._resolve_project_id(session, project_slug)
            target_id = self._resolve_scene_id(
                session, project_id, scene_number=scene_number
            )

            new_order = self._calculate_new_order(
                session,
                project_id,
                target_id,
                after_scene=after_scene,
                before_scene=before_scene,
            )

        service.update_scene(target_id, narrative_order=new_order)
        detail = service.get_scene_detail(target_id)
        return {"scene": detail, "narrative_order": new_order}

    def tool_scene_link(self, params: Dict[str, Any]) -> Dict[str, Any]:
        project_slug = params.get("project_slug") or self._default_project
        from_scene = params.get("from_scene")
        to_scene = params.get("to_scene")
        if not from_scene or not to_scene:
            raise ValueError("from_scene and to_scene are required")

        relation_type = params.get("relation_type", "sequence")

        with Session(self._engine) as session:
            project_id = self._resolve_project_id(session, project_slug)
            from_id = self._resolve_scene_id(
                session, project_id, scene_number=from_scene
            )
            to_id = self._resolve_scene_id(session, project_id, scene_number=to_scene)

        service.create_relation(from_id, to_id, relation_type=relation_type)
        return {"linked": True, "relation_type": relation_type}

    # ------------------------------------------------------------------
    # Ordering helpers
    # ------------------------------------------------------------------
    def _calculate_new_order(
        self,
        session: Session,
        project_id: int,
        target_id: int,
        *,
        after_scene: Optional[str],
        before_scene: Optional[str],
    ) -> float:
        """Calculate a narrative_order value that respects requested neighbours."""

        current_order = session.execute(
            select(ScreenplayScene.narrative_order).where(
                ScreenplayScene.id == target_id
            )
        ).scalar_one_or_none()

        def order_for(code: str) -> Optional[float]:
            if not code:
                return None
            return session.execute(
                select(ScreenplayScene.narrative_order).where(
                    ScreenplayScene.project_id == project_id,
                    ScreenplayScene.scene_number == code,
                )
            ).scalar_one_or_none()

        after_order = order_for(after_scene) if after_scene else None
        before_order = order_for(before_scene) if before_scene else None

        if after_scene and after_order is None:
            raise SceneResolutionError(f"after_scene {after_scene!r} not found")
        if before_scene and before_order is None:
            raise SceneResolutionError(f"before_scene {before_scene!r} not found")

        if after_order is not None and before_order is not None:
            if before_order <= after_order:
                raise ValueError(
                    "before_scene must appear after after_scene in the current order"
                )
            return (after_order + before_order) / 2.0

        if after_order is not None:
            next_order = session.execute(
                select(ScreenplayScene.narrative_order)
                .where(
                    ScreenplayScene.project_id == project_id,
                    ScreenplayScene.narrative_order > after_order,
                    ScreenplayScene.id != target_id,
                )
                .order_by(ScreenplayScene.narrative_order)
                .limit(1)
            ).scalar_one_or_none()
            if next_order is None:
                return after_order + 1.0
            return (after_order + next_order) / 2.0

        assert before_order is not None  # due to earlier guard
        previous_order = session.execute(
            select(ScreenplayScene.narrative_order)
            .where(
                ScreenplayScene.project_id == project_id,
                ScreenplayScene.narrative_order < before_order,
                ScreenplayScene.id != target_id,
            )
            .order_by(ScreenplayScene.narrative_order.desc())
            .limit(1)
        ).scalar_one_or_none()
        if previous_order is None:
            return before_order - 1.0
        return (previous_order + before_order) / 2.0

    # ------------------------------------------------------------------
    # JSON-RPC plumbing
    # ------------------------------------------------------------------
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
        except Exception as exc:  # pylint: disable=broad-except
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
            "scene_search": self.tool_scene_search,
            "scene_get": self.tool_scene_get,
            "scene_update": self.tool_scene_update,
            "scene_reorder": self.tool_scene_reorder,
            "scene_link": self.tool_scene_link,
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
    parser = argparse.ArgumentParser(description="Scene Manager MCP server")
    parser.add_argument(
        "--default-project",
        default="aa-janta-naduma",
        help="Fallback project slug when requests omit project_slug.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))

    server = SceneManagerMCPServer(default_project=args.default_project)

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
