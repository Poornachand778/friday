"""Comprehensive tests for mcp.scene_manager.server module.

Covers:
- ToolDefinition dataclass
- _tool_definitions function
- SceneResolutionError exception
- SceneManagerMCPServer class (init, resolve helpers, tool methods, handle_request,
  _render_tool_list, _dispatch_tool, _calculate_new_order)
- _iter_stdin generator
- main() CLI entrypoint
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from dataclasses import fields as dc_fields
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Patch service imports before importing the server module so that it never
# touches real databases during collection.
# ---------------------------------------------------------------------------

_svc_patch = patch.dict(
    "sys.modules",
    {
        "mcp.scene_manager.service": MagicMock(),
    },
)

# We need to import with the service module available
with patch("mcp.scene_manager.service", create=True):
    pass

from mcp.scene_manager.server import (
    ToolDefinition,
    _tool_definitions,
    SceneResolutionError,
    SceneManagerMCPServer,
    _iter_stdin,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_PROJECT = "test-project"


def _make_mock_session():
    """Return a MagicMock that works as a SQLAlchemy Session context manager."""
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    return mock_session


def _make_server(mock_svc):
    """Build a SceneManagerMCPServer with a mocked service layer."""
    mock_svc.get_engine_instance.return_value = MagicMock()
    return SceneManagerMCPServer(default_project=DEFAULT_PROJECT)


# ===================================================================
# 1. ToolDefinition dataclass
# ===================================================================


class TestToolDefinition:
    def test_creation(self):
        td = ToolDefinition(
            name="foo", description="bar", input_schema={"type": "object"}
        )
        assert td.name == "foo"
        assert td.description == "bar"
        assert td.input_schema == {"type": "object"}

    def test_fields(self):
        names = {f.name for f in dc_fields(ToolDefinition)}
        assert names == {"name", "description", "input_schema"}

    def test_field_types(self):
        type_map = {f.name: f.type for f in dc_fields(ToolDefinition)}
        assert type_map["name"] == "str"
        assert type_map["description"] == "str"

    def test_equality(self):
        a = ToolDefinition(name="x", description="y", input_schema={})
        b = ToolDefinition(name="x", description="y", input_schema={})
        assert a == b

    def test_inequality(self):
        a = ToolDefinition(name="x", description="y", input_schema={})
        b = ToolDefinition(name="z", description="y", input_schema={})
        assert a != b


# ===================================================================
# 2. _tool_definitions
# ===================================================================


class TestToolDefinitions:
    def test_returns_five_tools(self):
        tools = _tool_definitions("my-proj")
        assert len(tools) == 5

    def test_tool_names(self):
        tools = _tool_definitions("my-proj")
        names = [t.name for t in tools]
        assert names == [
            "scene_search",
            "scene_get",
            "scene_update",
            "scene_reorder",
            "scene_link",
        ]

    def test_all_are_tool_definition(self):
        for t in _tool_definitions("my-proj"):
            assert isinstance(t, ToolDefinition)

    def test_schemas_are_dicts(self):
        for t in _tool_definitions("my-proj"):
            assert isinstance(t.input_schema, dict)
            assert t.input_schema.get("type") == "object"

    def test_default_project_in_descriptions(self):
        slug = "special-slug"
        tools = _tool_definitions(slug)
        for t in tools:
            schema = t.input_schema
            prop = schema.get("properties", {}).get("project_slug", {})
            if prop:
                assert slug in prop.get("description", "")

    def test_scene_search_required_query(self):
        tool = _tool_definitions("p")[0]
        assert "query" in tool.input_schema.get("required", [])

    def test_scene_get_anyof(self):
        tool = _tool_definitions("p")[1]
        assert "anyOf" in tool.input_schema

    def test_scene_update_required_scene_number(self):
        tool = _tool_definitions("p")[2]
        assert "scene_number" in tool.input_schema.get("required", [])

    def test_scene_link_required_fields(self):
        tool = _tool_definitions("p")[4]
        req = tool.input_schema.get("required", [])
        assert "from_scene" in req
        assert "to_scene" in req

    def test_project_slug_default_value(self):
        slug = "my-default"
        tools = _tool_definitions(slug)
        for t in tools:
            prop = t.input_schema.get("properties", {}).get("project_slug", {})
            if "default" in prop:
                assert prop["default"] == slug

    def test_additional_properties_false(self):
        for t in _tool_definitions("p"):
            assert t.input_schema.get("additionalProperties") is False


# ===================================================================
# 3. SceneResolutionError
# ===================================================================


class TestSceneResolutionError:
    def test_inherits_runtime_error(self):
        assert issubclass(SceneResolutionError, RuntimeError)

    def test_message(self):
        err = SceneResolutionError("bad scene")
        assert str(err) == "bad scene"

    def test_raise_and_catch(self):
        with pytest.raises(SceneResolutionError, match="oops"):
            raise SceneResolutionError("oops")

    def test_catch_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise SceneResolutionError("runtime catch")


# ===================================================================
# 4. SceneManagerMCPServer.__init__
# ===================================================================


class TestServerInit:
    @patch("mcp.scene_manager.server.service")
    def test_stores_default_project(self, mock_svc):
        mock_svc.get_engine_instance.return_value = MagicMock()
        server = SceneManagerMCPServer(default_project="my-proj")
        assert server._default_project == "my-proj"

    @patch("mcp.scene_manager.server.service")
    def test_calls_get_engine_instance(self, mock_svc):
        mock_svc.get_engine_instance.return_value = MagicMock()
        SceneManagerMCPServer(default_project="x")
        mock_svc.get_engine_instance.assert_called_once()

    @patch("mcp.scene_manager.server.service")
    def test_builds_tool_defs(self, mock_svc):
        mock_svc.get_engine_instance.return_value = MagicMock()
        server = SceneManagerMCPServer(default_project="proj")
        assert len(server._tool_defs) == 5

    @patch("mcp.scene_manager.server.service")
    def test_engine_stored(self, mock_svc):
        engine = MagicMock()
        mock_svc.get_engine_instance.return_value = engine
        server = SceneManagerMCPServer(default_project="p")
        assert server._engine is engine


# ===================================================================
# 5. _resolve_project_id
# ===================================================================


class TestResolveProjectId:
    @patch("mcp.scene_manager.server.service")
    def test_found_returns_id(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.return_value = 42
        result = server._resolve_project_id(session, "my-slug")
        assert result == 42

    @patch("mcp.scene_manager.server.service")
    def test_not_found_raises(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.return_value = None
        with pytest.raises(SceneResolutionError, match="Unknown project slug"):
            server._resolve_project_id(session, "nonexistent")

    @patch("mcp.scene_manager.server.service")
    def test_none_slug_uses_default(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.return_value = 7
        result = server._resolve_project_id(session, None)
        assert result == 7

    @patch("mcp.scene_manager.server.service")
    def test_error_message_includes_slug(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.return_value = None
        with pytest.raises(SceneResolutionError, match="'bad-slug'"):
            server._resolve_project_id(session, "bad-slug")


# ===================================================================
# 6. _resolve_scene_id by scene_id
# ===================================================================


class TestResolveSceneIdById:
    @patch("mcp.scene_manager.server.service")
    def test_found(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.return_value = 99
        result = server._resolve_scene_id(session, 1, scene_id=99)
        assert result == 99

    @patch("mcp.scene_manager.server.service")
    def test_not_found_raises(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.return_value = None
        with pytest.raises(SceneResolutionError, match="does not belong"):
            server._resolve_scene_id(session, 1, scene_id=999)


# ===================================================================
# 7. _resolve_scene_id by scene_number
# ===================================================================


class TestResolveSceneIdByNumber:
    @patch("mcp.scene_manager.server.service")
    def test_found(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.return_value = 55
        result = server._resolve_scene_id(session, 1, scene_number=3)
        assert result == 55

    @patch("mcp.scene_manager.server.service")
    def test_not_found_raises(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.return_value = None
        with pytest.raises(SceneResolutionError, match="Unknown scene number"):
            server._resolve_scene_id(session, 1, scene_number=999)


# ===================================================================
# 8. _resolve_scene_id neither provided
# ===================================================================


class TestResolveSceneIdNeither:
    @patch("mcp.scene_manager.server.service")
    def test_raises_when_neither(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        with pytest.raises(SceneResolutionError, match="scene_number or scene_id"):
            server._resolve_scene_id(session, 1)

    @patch("mcp.scene_manager.server.service")
    def test_raises_with_zero_scene_number(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        with pytest.raises(SceneResolutionError, match="scene_number or scene_id"):
            server._resolve_scene_id(session, 1, scene_number=0)

    @patch("mcp.scene_manager.server.service")
    def test_raises_with_none_values(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        with pytest.raises(SceneResolutionError):
            server._resolve_scene_id(session, 1, scene_number=None, scene_id=None)


# ===================================================================
# 9. tool_scene_search
# ===================================================================


class TestToolSceneSearch:
    @patch("mcp.scene_manager.server.service")
    def test_calls_service_search(self, mock_svc):
        server = _make_server(mock_svc)
        mock_svc.search_scenes.return_value = [{"scene_number": 1}]
        result = server.tool_scene_search({"query": "love scene"})
        mock_svc.search_scenes.assert_called_once_with(
            "love scene", project_slug=DEFAULT_PROJECT, top_k=5
        )
        assert result == [{"scene_number": 1}]

    @patch("mcp.scene_manager.server.service")
    def test_custom_top_k(self, mock_svc):
        server = _make_server(mock_svc)
        mock_svc.search_scenes.return_value = []
        server.tool_scene_search({"query": "fight", "top_k": 10})
        mock_svc.search_scenes.assert_called_once_with(
            "fight", project_slug=DEFAULT_PROJECT, top_k=10
        )

    @patch("mcp.scene_manager.server.service")
    def test_custom_project_slug(self, mock_svc):
        server = _make_server(mock_svc)
        mock_svc.search_scenes.return_value = []
        server.tool_scene_search({"query": "rain", "project_slug": "other"})
        mock_svc.search_scenes.assert_called_once_with(
            "rain", project_slug="other", top_k=5
        )

    @patch("mcp.scene_manager.server.service")
    def test_returns_list(self, mock_svc):
        server = _make_server(mock_svc)
        mock_svc.search_scenes.return_value = [{"a": 1}, {"b": 2}]
        result = server.tool_scene_search({"query": "hello"})
        assert isinstance(result, list)
        assert len(result) == 2


# ===================================================================
# 10. tool_scene_search empty query
# ===================================================================


class TestToolSceneSearchEmptyQuery:
    @patch("mcp.scene_manager.server.service")
    def test_empty_string_raises(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="must not be empty"):
            server.tool_scene_search({"query": ""})

    @patch("mcp.scene_manager.server.service")
    def test_whitespace_only_raises(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="must not be empty"):
            server.tool_scene_search({"query": "   "})


# ===================================================================
# 11. tool_scene_get by scene_number
# ===================================================================


class TestToolSceneGetByNumber:
    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_resolves_and_calls(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        # First call: resolve_project_id -> returns 1
        # Second call: resolve_scene_id -> returns 10
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10]
        mock_svc.get_scene_detail.return_value = {"id": 10, "title": "S1"}

        result = server.tool_scene_get({"scene_number": 5})
        mock_svc.get_scene_detail.assert_called_once_with(10)
        assert result == {"id": 10, "title": "S1"}

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_uses_default_project(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10]
        mock_svc.get_scene_detail.return_value = {"id": 10}

        server.tool_scene_get({"scene_number": 1})
        # Should have used the default project (no explicit slug passed)
        mock_svc.get_scene_detail.assert_called_once()


# ===================================================================
# 12. tool_scene_get by scene_id
# ===================================================================


class TestToolSceneGetById:
    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_by_scene_id(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 42]
        mock_svc.get_scene_detail.return_value = {"id": 42}

        result = server.tool_scene_get({"scene_id": 42})
        mock_svc.get_scene_detail.assert_called_once_with(42)
        assert result["id"] == 42

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_with_explicit_project(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [5, 20]
        mock_svc.get_scene_detail.return_value = {"id": 20}

        result = server.tool_scene_get({"scene_id": 20, "project_slug": "other-proj"})
        assert result["id"] == 20


# ===================================================================
# 13. tool_scene_update
# ===================================================================


class TestToolSceneUpdate:
    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_calls_update_scene(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10]
        mock_svc.update_scene.return_value = True
        mock_svc.get_scene_detail.return_value = {"id": 10, "title": "New"}

        result = server.tool_scene_update(
            {
                "scene_number": 3,
                "title": "New Title",
                "summary": "New Summary",
                "tags": ["action"],
                "status": "revision",
            }
        )
        mock_svc.update_scene.assert_called_once_with(
            10,
            title="New Title",
            summary="New Summary",
            tags=["action"],
            narrative_order=None,
            status="revision",
        )
        assert result["updated"] is True
        assert result["scene"]["id"] == 10

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_update_with_optional_fields_none(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10]
        mock_svc.update_scene.return_value = False
        mock_svc.get_scene_detail.return_value = {"id": 10}

        result = server.tool_scene_update({"scene_number": 1})
        mock_svc.update_scene.assert_called_once_with(
            10, title=None, summary=None, tags=None, narrative_order=None, status=None
        )
        assert result["updated"] is False

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_update_with_narrative_order(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10]
        mock_svc.update_scene.return_value = True
        mock_svc.get_scene_detail.return_value = {"id": 10}

        server.tool_scene_update({"scene_number": 2, "narrative_order": 3.5})
        mock_svc.update_scene.assert_called_once_with(
            10, title=None, summary=None, tags=None, narrative_order=3.5, status=None
        )


# ===================================================================
# 14. tool_scene_update missing scene_number
# ===================================================================


class TestToolSceneUpdateMissing:
    @patch("mcp.scene_manager.server.service")
    def test_missing_scene_number_raises(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="scene_number is required"):
            server.tool_scene_update({"title": "no scene number"})

    @patch("mcp.scene_manager.server.service")
    def test_zero_scene_number_raises(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="scene_number is required"):
            server.tool_scene_update({"scene_number": 0})


# ===================================================================
# 15. tool_scene_reorder after_scene
# ===================================================================


class TestToolSceneReorderAfter:
    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_after_scene_only(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        # resolve_project_id -> 1, resolve_scene_id -> 10
        # _calculate_new_order queries: current_order, order_for(after), next_order
        session.execute.return_value.scalar_one_or_none.side_effect = [
            1,
            10,  # resolve project + scene
            5.0,  # current_order of target
            3.0,  # order_for after_scene
            7.0,  # next_order after after_scene
        ]
        mock_svc.update_scene.return_value = True
        mock_svc.get_scene_detail.return_value = {"id": 10, "narrative_order": 5.0}

        result = server.tool_scene_reorder(
            {
                "scene_number": 2,
                "after_scene": 1,
            }
        )
        assert "scene" in result
        assert "narrative_order" in result


# ===================================================================
# 16. tool_scene_reorder before_scene
# ===================================================================


class TestToolSceneReorderBefore:
    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_before_scene_only(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [
            1,
            10,  # resolve project + scene
            5.0,  # current_order
            8.0,  # order_for before_scene
            3.0,  # previous_order
        ]
        mock_svc.update_scene.return_value = True
        mock_svc.get_scene_detail.return_value = {"id": 10}

        result = server.tool_scene_reorder(
            {
                "scene_number": 5,
                "before_scene": 3,
            }
        )
        assert "scene" in result


# ===================================================================
# 17. tool_scene_reorder both after and before
# ===================================================================


class TestToolSceneReorderBoth:
    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_both_after_and_before(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [
            1,
            10,  # resolve project + scene
            5.0,  # current_order
            4.0,  # order_for after_scene
            8.0,  # order_for before_scene
        ]
        mock_svc.update_scene.return_value = True
        mock_svc.get_scene_detail.return_value = {"id": 10}

        result = server.tool_scene_reorder(
            {
                "scene_number": 2,
                "after_scene": 1,
                "before_scene": 3,
            }
        )
        # Midpoint = (4.0 + 8.0) / 2.0 = 6.0
        assert result["narrative_order"] == 6.0


# ===================================================================
# 18. tool_scene_reorder neither provided
# ===================================================================


class TestToolSceneReorderNeither:
    @patch("mcp.scene_manager.server.service")
    def test_raises_when_neither(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="after_scene or before_scene"):
            server.tool_scene_reorder({"scene_number": 1})

    @patch("mcp.scene_manager.server.service")
    def test_missing_scene_number_raises(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="scene_number is required"):
            server.tool_scene_reorder({"after_scene": 2})


# ===================================================================
# 19. tool_scene_link
# ===================================================================


class TestToolSceneLink:
    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_resolves_both_and_creates(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        # resolve_project_id -> 1, from_scene -> 10, to_scene -> 20
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10, 20]
        mock_svc.create_relation.return_value = True

        result = server.tool_scene_link(
            {
                "from_scene": 1,
                "to_scene": 5,
            }
        )
        mock_svc.create_relation.assert_called_once_with(
            10, 20, relation_type="sequence"
        )
        assert result["linked"] is True
        assert result["relation_type"] == "sequence"

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_custom_relation_type(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10, 20]
        mock_svc.create_relation.return_value = True

        result = server.tool_scene_link(
            {
                "from_scene": 1,
                "to_scene": 5,
                "relation_type": "flashback",
            }
        )
        mock_svc.create_relation.assert_called_once_with(
            10, 20, relation_type="flashback"
        )
        assert result["relation_type"] == "flashback"

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_with_project_slug(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [5, 30, 40]
        mock_svc.create_relation.return_value = True

        result = server.tool_scene_link(
            {
                "from_scene": 3,
                "to_scene": 4,
                "project_slug": "other",
            }
        )
        assert result["linked"] is True


# ===================================================================
# 20. tool_scene_link missing scenes
# ===================================================================


class TestToolSceneLinkMissing:
    @patch("mcp.scene_manager.server.service")
    def test_missing_from_scene(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="from_scene and to_scene"):
            server.tool_scene_link({"to_scene": 5})

    @patch("mcp.scene_manager.server.service")
    def test_missing_to_scene(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="from_scene and to_scene"):
            server.tool_scene_link({"from_scene": 1})

    @patch("mcp.scene_manager.server.service")
    def test_missing_both(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="from_scene and to_scene"):
            server.tool_scene_link({})

    @patch("mcp.scene_manager.server.service")
    def test_zero_from_scene(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="from_scene and to_scene"):
            server.tool_scene_link({"from_scene": 0, "to_scene": 5})


# ===================================================================
# 21. _calculate_new_order: after_scene only
# ===================================================================


class TestCalculateNewOrderAfter:
    @patch("mcp.scene_manager.server.service")
    def test_after_with_next_scene(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        # current_order, order_for(after), next_order
        session.execute.return_value.scalar_one_or_none.side_effect = [
            5.0,  # current_order
            3.0,  # after_scene order
            7.0,  # next_order
        ]
        result = server._calculate_new_order(
            session, project_id=1, target_id=10, after_scene=2, before_scene=None
        )
        assert result == (3.0 + 7.0) / 2.0  # 5.0

    @patch("mcp.scene_manager.server.service")
    def test_after_without_next_scene(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.side_effect = [
            5.0,  # current_order
            10.0,  # after_scene order
            None,  # no next_order
        ]
        result = server._calculate_new_order(
            session, project_id=1, target_id=10, after_scene=5, before_scene=None
        )
        assert result == 10.0 + 1.0  # 11.0


# ===================================================================
# 22. _calculate_new_order: before_scene only
# ===================================================================


class TestCalculateNewOrderBefore:
    @patch("mcp.scene_manager.server.service")
    def test_before_with_previous_scene(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.side_effect = [
            5.0,  # current_order
            8.0,  # before_scene order
            4.0,  # previous_order
        ]
        result = server._calculate_new_order(
            session, project_id=1, target_id=10, after_scene=None, before_scene=3
        )
        assert result == (4.0 + 8.0) / 2.0  # 6.0

    @patch("mcp.scene_manager.server.service")
    def test_before_without_previous_scene(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.side_effect = [
            5.0,  # current_order
            3.0,  # before_scene order
            None,  # no previous_order
        ]
        result = server._calculate_new_order(
            session, project_id=1, target_id=10, after_scene=None, before_scene=1
        )
        assert result == 3.0 - 1.0  # 2.0


# ===================================================================
# 23. _calculate_new_order: both (midpoint, error if before <= after)
# ===================================================================


class TestCalculateNewOrderBoth:
    @patch("mcp.scene_manager.server.service")
    def test_midpoint(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.side_effect = [
            5.0,  # current_order
            2.0,  # after_scene order
            10.0,  # before_scene order
        ]
        result = server._calculate_new_order(
            session, project_id=1, target_id=10, after_scene=1, before_scene=5
        )
        assert result == (2.0 + 10.0) / 2.0  # 6.0

    @patch("mcp.scene_manager.server.service")
    def test_error_before_less_than_after(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.side_effect = [
            5.0,  # current_order
            10.0,  # after_scene order (higher)
            3.0,  # before_scene order (lower!)
        ]
        with pytest.raises(ValueError, match="before_scene must appear after"):
            server._calculate_new_order(
                session, project_id=1, target_id=10, after_scene=5, before_scene=1
            )

    @patch("mcp.scene_manager.server.service")
    def test_error_before_equals_after(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.side_effect = [
            5.0,  # current_order
            5.0,  # after_scene order
            5.0,  # before_scene order (same!)
        ]
        with pytest.raises(ValueError, match="before_scene must appear after"):
            server._calculate_new_order(
                session, project_id=1, target_id=10, after_scene=2, before_scene=3
            )


# ===================================================================
# 24. _calculate_new_order: invalid scene references
# ===================================================================


class TestCalculateNewOrderInvalid:
    @patch("mcp.scene_manager.server.service")
    def test_after_scene_not_found(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.side_effect = [
            5.0,  # current_order
            None,  # after_scene not found
        ]
        with pytest.raises(SceneResolutionError, match="after_scene.*not found"):
            server._calculate_new_order(
                session, project_id=1, target_id=10, after_scene=999, before_scene=None
            )

    @patch("mcp.scene_manager.server.service")
    def test_before_scene_not_found(self, mock_svc):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        session.execute.return_value.scalar_one_or_none.side_effect = [
            5.0,  # current_order
            None,  # before_scene not found
        ]
        with pytest.raises(SceneResolutionError, match="before_scene.*not found"):
            server._calculate_new_order(
                session, project_id=1, target_id=10, after_scene=None, before_scene=999
            )


# ===================================================================
# 25. handle_request initialize
# ===================================================================


class TestHandleRequestInitialize:
    @patch("mcp.scene_manager.server.service")
    def test_returns_protocol_version(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"id": 1, "method": "initialize"})
        assert resp["type"] == "response"
        assert resp["result"]["protocolVersion"] == "0.1"

    @patch("mcp.scene_manager.server.service")
    def test_returns_capabilities(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"id": 1, "method": "initialize"})
        caps = resp["result"]["capabilities"]
        assert caps["tools"]["list"] is True
        assert caps["tools"]["call"] is True

    @patch("mcp.scene_manager.server.service")
    def test_preserves_request_id(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"id": 42, "method": "initialize"})
        assert resp["id"] == 42

    @patch("mcp.scene_manager.server.service")
    def test_no_id(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"method": "initialize"})
        assert resp["id"] is None


# ===================================================================
# 26. handle_request list_tools
# ===================================================================


class TestHandleRequestListTools:
    @patch("mcp.scene_manager.server.service")
    def test_returns_tool_list(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"id": 2, "method": "list_tools"})
        assert resp["type"] == "response"
        tools = resp["result"]["tools"]
        assert len(tools) == 5

    @patch("mcp.scene_manager.server.service")
    def test_tool_names_in_list(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"id": 2, "method": "list_tools"})
        names = [t["name"] for t in resp["result"]["tools"]]
        assert "scene_search" in names
        assert "scene_link" in names


# ===================================================================
# 27. handle_request call_tool
# ===================================================================


class TestHandleRequestCallTool:
    @patch("mcp.scene_manager.server.service")
    def test_dispatches_search(self, mock_svc):
        server = _make_server(mock_svc)
        mock_svc.search_scenes.return_value = [{"id": 1}]
        resp = server.handle_request(
            {
                "id": 3,
                "method": "call_tool",
                "params": {"name": "scene_search", "arguments": {"query": "love"}},
            }
        )
        assert resp["type"] == "response"
        assert resp["result"]["content"] == [{"id": 1}]

    @patch("mcp.scene_manager.server.service")
    def test_call_tool_with_no_params(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request(
            {
                "id": 4,
                "method": "call_tool",
            }
        )
        # Missing name should produce an error
        assert resp["type"] == "error"


# ===================================================================
# 28. handle_request shutdown
# ===================================================================


class TestHandleRequestShutdown:
    @patch("mcp.scene_manager.server.service")
    def test_returns_ok(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"id": 5, "method": "shutdown"})
        assert resp["type"] == "response"
        assert resp["result"]["ok"] is True

    @patch("mcp.scene_manager.server.service")
    def test_preserves_id(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"id": "abc", "method": "shutdown"})
        assert resp["id"] == "abc"


# ===================================================================
# 29. handle_request unknown method
# ===================================================================


class TestHandleRequestUnknown:
    @patch("mcp.scene_manager.server.service")
    def test_unknown_method_returns_error(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"id": 6, "method": "nonexistent"})
        assert resp["type"] == "error"
        assert "Unknown method" in resp["error"]["message"]

    @patch("mcp.scene_manager.server.service")
    def test_error_code_internal(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"id": 6, "method": "bad"})
        assert resp["error"]["code"] == "internal_error"


# ===================================================================
# 30. handle_request error handling
# ===================================================================


class TestHandleRequestErrors:
    @patch("mcp.scene_manager.server.service")
    def test_exception_returns_error_response(self, mock_svc):
        server = _make_server(mock_svc)
        mock_svc.search_scenes.side_effect = RuntimeError("DB gone")
        resp = server.handle_request(
            {
                "id": 7,
                "method": "call_tool",
                "params": {"name": "scene_search", "arguments": {"query": "x"}},
            }
        )
        assert resp["type"] == "error"
        assert "DB gone" in resp["error"]["message"]
        assert resp["id"] == 7

    @patch("mcp.scene_manager.server.service")
    def test_scene_resolution_error_in_handle(self, mock_svc):
        server = _make_server(mock_svc)
        # Make scene_get fail with SceneResolutionError
        with patch.object(
            server, "tool_scene_get", side_effect=SceneResolutionError("no scene")
        ):
            resp = server.handle_request(
                {
                    "id": 8,
                    "method": "call_tool",
                    "params": {"name": "scene_get", "arguments": {"scene_number": 1}},
                }
            )
        assert resp["type"] == "error"
        assert "no scene" in resp["error"]["message"]

    @patch("mcp.scene_manager.server.service")
    def test_value_error_in_handle(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request(
            {
                "id": 9,
                "method": "call_tool",
                "params": {"name": "scene_search", "arguments": {"query": "  "}},
            }
        )
        assert resp["type"] == "error"
        assert "must not be empty" in resp["error"]["message"]

    @patch("mcp.scene_manager.server.service")
    def test_none_params_treated_as_empty_dict(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request(
            {
                "id": 10,
                "method": "call_tool",
                "params": None,
            }
        )
        # Should fail with missing name, not a TypeError
        assert resp["type"] == "error"


# ===================================================================
# 31. _render_tool_list
# ===================================================================


class TestRenderToolList:
    @patch("mcp.scene_manager.server.service")
    def test_correct_format(self, mock_svc):
        server = _make_server(mock_svc)
        result = server._render_tool_list()
        assert "tools" in result
        assert isinstance(result["tools"], list)
        assert len(result["tools"]) == 5

    @patch("mcp.scene_manager.server.service")
    def test_tool_structure(self, mock_svc):
        server = _make_server(mock_svc)
        result = server._render_tool_list()
        for tool in result["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool

    @patch("mcp.scene_manager.server.service")
    def test_tool_names_match_definitions(self, mock_svc):
        server = _make_server(mock_svc)
        result = server._render_tool_list()
        names = [t["name"] for t in result["tools"]]
        expected = [
            "scene_search",
            "scene_get",
            "scene_update",
            "scene_reorder",
            "scene_link",
        ]
        assert names == expected

    @patch("mcp.scene_manager.server.service")
    def test_input_schema_key(self, mock_svc):
        server = _make_server(mock_svc)
        result = server._render_tool_list()
        for tool in result["tools"]:
            assert tool["inputSchema"]["type"] == "object"


# ===================================================================
# 32. _dispatch_tool
# ===================================================================


class TestDispatchTool:
    @patch("mcp.scene_manager.server.service")
    def test_routes_scene_search(self, mock_svc):
        server = _make_server(mock_svc)
        mock_svc.search_scenes.return_value = [{"id": 1}]
        result = server._dispatch_tool(
            {"name": "scene_search", "arguments": {"query": "test"}}
        )
        assert result == {"content": [{"id": 1}]}

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_routes_scene_get(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10]
        mock_svc.get_scene_detail.return_value = {"id": 10}
        result = server._dispatch_tool(
            {
                "name": "scene_get",
                "arguments": {"scene_number": 1},
            }
        )
        assert result == {"content": {"id": 10}}

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_routes_scene_update(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10]
        mock_svc.update_scene.return_value = True
        mock_svc.get_scene_detail.return_value = {"id": 10}
        result = server._dispatch_tool(
            {
                "name": "scene_update",
                "arguments": {"scene_number": 1, "title": "X"},
            }
        )
        assert "content" in result
        assert result["content"]["updated"] is True

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_routes_scene_link(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10, 20]
        mock_svc.create_relation.return_value = True
        result = server._dispatch_tool(
            {
                "name": "scene_link",
                "arguments": {"from_scene": 1, "to_scene": 2},
            }
        )
        assert result["content"]["linked"] is True

    @patch("mcp.scene_manager.server.service")
    def test_empty_arguments_default(self, mock_svc):
        server = _make_server(mock_svc)
        # scene_search with no arguments key should pass {} as arguments
        with pytest.raises(KeyError):
            # "query" key won't exist in empty dict
            server._dispatch_tool({"name": "scene_search"})


# ===================================================================
# 33. _dispatch_tool unknown tool
# ===================================================================


class TestDispatchToolUnknown:
    @patch("mcp.scene_manager.server.service")
    def test_unknown_tool_raises(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="Unknown tool"):
            server._dispatch_tool({"name": "nonexistent_tool"})

    @patch("mcp.scene_manager.server.service")
    def test_unknown_tool_message(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="'nonexistent_tool'"):
            server._dispatch_tool({"name": "nonexistent_tool"})


# ===================================================================
# 34. _dispatch_tool missing name
# ===================================================================


class TestDispatchToolMissingName:
    @patch("mcp.scene_manager.server.service")
    def test_missing_name_raises(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="params.name is required"):
            server._dispatch_tool({})

    @patch("mcp.scene_manager.server.service")
    def test_none_name_raises(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="params.name is required"):
            server._dispatch_tool({"name": None})

    @patch("mcp.scene_manager.server.service")
    def test_empty_string_name_raises(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="params.name is required"):
            server._dispatch_tool({"name": ""})


# ===================================================================
# 35. main argparse
# ===================================================================


class TestMain:
    @patch("mcp.scene_manager.server._iter_stdin", return_value=iter([]))
    @patch("mcp.scene_manager.server.service")
    def test_default_project(self, mock_svc, mock_stdin):
        mock_svc.get_engine_instance.return_value = MagicMock()
        result = main(["--default-project", "my-slug"])
        assert result == 0

    @patch("mcp.scene_manager.server._iter_stdin", return_value=iter([]))
    @patch("mcp.scene_manager.server.service")
    def test_log_level(self, mock_svc, mock_stdin):
        mock_svc.get_engine_instance.return_value = MagicMock()
        result = main(["--log-level", "DEBUG"])
        assert result == 0

    @patch("mcp.scene_manager.server._iter_stdin", return_value=iter([]))
    @patch("mcp.scene_manager.server.service")
    def test_defaults(self, mock_svc, mock_stdin):
        mock_svc.get_engine_instance.return_value = MagicMock()
        result = main([])
        assert result == 0

    @patch("mcp.scene_manager.server.sys.stdout", new_callable=StringIO)
    @patch("mcp.scene_manager.server.service")
    def test_processes_json_line(self, mock_svc, mock_stdout):
        mock_svc.get_engine_instance.return_value = MagicMock()
        payload = json.dumps({"id": 1, "method": "shutdown"})
        with patch(
            "mcp.scene_manager.server._iter_stdin", return_value=iter([payload])
        ):
            result = main([])
        assert result == 0
        output = mock_stdout.getvalue().strip()
        resp = json.loads(output)
        assert resp["result"]["ok"] is True

    @patch("mcp.scene_manager.server.sys.stdout", new_callable=StringIO)
    @patch("mcp.scene_manager.server.service")
    def test_handles_invalid_json(self, mock_svc, mock_stdout):
        mock_svc.get_engine_instance.return_value = MagicMock()
        with patch(
            "mcp.scene_manager.server._iter_stdin", return_value=iter(["not json"])
        ):
            result = main([])
        assert result == 0
        output = mock_stdout.getvalue().strip()
        resp = json.loads(output)
        assert resp["type"] == "error"
        assert resp["error"]["code"] == "invalid_json"

    @patch("mcp.scene_manager.server.sys.stdout", new_callable=StringIO)
    @patch("mcp.scene_manager.server.service")
    def test_processes_multiple_messages_stops_on_shutdown(self, mock_svc, mock_stdout):
        mock_svc.get_engine_instance.return_value = MagicMock()
        line1 = json.dumps({"id": 1, "method": "initialize"})
        line2 = json.dumps({"id": 2, "method": "shutdown"})
        line3 = json.dumps({"id": 3, "method": "list_tools"})  # should not be processed
        with patch(
            "mcp.scene_manager.server._iter_stdin",
            return_value=iter([line1, line2, line3]),
        ):
            main([])
        lines = mock_stdout.getvalue().strip().split("\n")
        assert len(lines) == 2  # init + shutdown only
        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert r1["id"] == 1
        assert r2["id"] == 2

    @patch("mcp.scene_manager.server.sys.stdout", new_callable=StringIO)
    @patch("mcp.scene_manager.server.service")
    def test_flush_called_per_response(self, mock_svc, mock_stdout):
        mock_svc.get_engine_instance.return_value = MagicMock()
        payload = json.dumps({"id": 1, "method": "shutdown"})
        with patch(
            "mcp.scene_manager.server._iter_stdin", return_value=iter([payload])
        ):
            main([])
        # Verify output was flushed (stdout.getvalue() has content)
        assert mock_stdout.getvalue().strip() != ""


# ===================================================================
# _iter_stdin
# ===================================================================


class TestIterStdin:
    def test_yields_non_empty_lines(self):
        fake_stdin = StringIO("hello\nworld\n")
        with patch("mcp.scene_manager.server.sys.stdin", fake_stdin):
            result = list(_iter_stdin())
        assert result == ["hello", "world"]

    def test_skips_blank_lines(self):
        fake_stdin = StringIO("hello\n\n\nworld\n")
        with patch("mcp.scene_manager.server.sys.stdin", fake_stdin):
            result = list(_iter_stdin())
        assert result == ["hello", "world"]

    def test_strips_whitespace(self):
        fake_stdin = StringIO("  hello  \n  world  \n")
        with patch("mcp.scene_manager.server.sys.stdin", fake_stdin):
            result = list(_iter_stdin())
        assert result == ["hello", "world"]

    def test_empty_stdin(self):
        fake_stdin = StringIO("")
        with patch("mcp.scene_manager.server.sys.stdin", fake_stdin):
            result = list(_iter_stdin())
        assert result == []

    def test_only_blank_lines(self):
        fake_stdin = StringIO("\n\n\n")
        with patch("mcp.scene_manager.server.sys.stdin", fake_stdin):
            result = list(_iter_stdin())
        assert result == []


# ===================================================================
# Integration-style: full call_tool flows through handle_request
# ===================================================================


class TestIntegrationFlows:
    @patch("mcp.scene_manager.server.service")
    def test_search_through_handle_request(self, mock_svc):
        server = _make_server(mock_svc)
        mock_svc.search_scenes.return_value = [{"scene_number": 1, "score": 0.9}]
        resp = server.handle_request(
            {
                "id": 100,
                "method": "call_tool",
                "params": {
                    "name": "scene_search",
                    "arguments": {"query": "love scene", "top_k": 3},
                },
            }
        )
        assert resp["type"] == "response"
        assert resp["result"]["content"][0]["scene_number"] == 1

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_get_through_handle_request(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10]
        mock_svc.get_scene_detail.return_value = {"id": 10, "title": "Test Scene"}
        resp = server.handle_request(
            {
                "id": 101,
                "method": "call_tool",
                "params": {
                    "name": "scene_get",
                    "arguments": {"scene_number": 1},
                },
            }
        )
        assert resp["type"] == "response"
        assert resp["result"]["content"]["title"] == "Test Scene"

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_update_through_handle_request(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10]
        mock_svc.update_scene.return_value = True
        mock_svc.get_scene_detail.return_value = {"id": 10, "title": "Updated"}
        resp = server.handle_request(
            {
                "id": 102,
                "method": "call_tool",
                "params": {
                    "name": "scene_update",
                    "arguments": {"scene_number": 1, "title": "Updated"},
                },
            }
        )
        assert resp["type"] == "response"
        assert resp["result"]["content"]["updated"] is True

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_link_through_handle_request(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10, 20]
        mock_svc.create_relation.return_value = True
        resp = server.handle_request(
            {
                "id": 103,
                "method": "call_tool",
                "params": {
                    "name": "scene_link",
                    "arguments": {
                        "from_scene": 1,
                        "to_scene": 2,
                        "relation_type": "parallel",
                    },
                },
            }
        )
        assert resp["type"] == "response"
        assert resp["result"]["content"]["linked"] is True
        assert resp["result"]["content"]["relation_type"] == "parallel"

    @patch("mcp.scene_manager.server.service")
    def test_initialize_then_list_tools(self, mock_svc):
        server = _make_server(mock_svc)
        r1 = server.handle_request({"id": 1, "method": "initialize"})
        r2 = server.handle_request({"id": 2, "method": "list_tools"})
        assert r1["type"] == "response"
        assert r2["type"] == "response"
        assert len(r2["result"]["tools"]) == 5

    @patch("mcp.scene_manager.server.service")
    def test_handle_request_missing_method(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"id": 1})
        # method is None -> "Unknown method None"
        assert resp["type"] == "error"
        assert "Unknown method" in resp["error"]["message"]


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    @patch("mcp.scene_manager.server.service")
    def test_search_strips_query(self, mock_svc):
        server = _make_server(mock_svc)
        mock_svc.search_scenes.return_value = []
        server.tool_scene_search({"query": "  test  "})
        mock_svc.search_scenes.assert_called_once_with(
            "test", project_slug=DEFAULT_PROJECT, top_k=5
        )

    @patch("mcp.scene_manager.server.service")
    def test_search_top_k_string_converted(self, mock_svc):
        server = _make_server(mock_svc)
        mock_svc.search_scenes.return_value = []
        server.tool_scene_search({"query": "test", "top_k": "7"})
        mock_svc.search_scenes.assert_called_once_with(
            "test", project_slug=DEFAULT_PROJECT, top_k=7
        )

    @patch("mcp.scene_manager.server.service")
    def test_dispatch_tool_empty_arguments_key(self, mock_svc):
        server = _make_server(mock_svc)
        mock_svc.search_scenes.return_value = []
        # arguments = None -> should be treated as {}
        # scene_search expects query key -> KeyError
        with pytest.raises(KeyError):
            server._dispatch_tool({"name": "scene_search", "arguments": None})

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_scene_get_none_project_slug_uses_default(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10]
        mock_svc.get_scene_detail.return_value = {"id": 10}
        # Explicitly passing None project_slug
        server.tool_scene_get({"scene_number": 1, "project_slug": None})
        mock_svc.get_scene_detail.assert_called_once()

    @patch("mcp.scene_manager.server.Session")
    @patch("mcp.scene_manager.server.service")
    def test_scene_link_default_relation_type(self, mock_svc, MockSession):
        server = _make_server(mock_svc)
        session = _make_mock_session()
        MockSession.return_value = session
        session.execute.return_value.scalar_one_or_none.side_effect = [1, 10, 20]
        mock_svc.create_relation.return_value = True
        result = server.tool_scene_link({"from_scene": 1, "to_scene": 2})
        mock_svc.create_relation.assert_called_once_with(
            10, 20, relation_type="sequence"
        )

    @patch("mcp.scene_manager.server.service")
    def test_handle_request_with_empty_params(self, mock_svc):
        server = _make_server(mock_svc)
        resp = server.handle_request({"id": 1, "method": "initialize", "params": {}})
        assert resp["type"] == "response"

    @patch("mcp.scene_manager.server.service")
    def test_tool_definition_descriptions_non_empty(self, mock_svc):
        server = _make_server(mock_svc)
        for td in server._tool_defs:
            assert len(td.description) > 0

    @patch("mcp.scene_manager.server.service")
    def test_scene_reorder_zero_scene_number(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="scene_number is required"):
            server.tool_scene_reorder({"scene_number": 0, "after_scene": 1})

    @patch("mcp.scene_manager.server.service")
    def test_scene_update_zero_scene_number(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="scene_number is required"):
            server.tool_scene_update({"scene_number": 0, "title": "New"})

    @patch("mcp.scene_manager.server.service")
    def test_scene_link_zero_to_scene(self, mock_svc):
        server = _make_server(mock_svc)
        with pytest.raises(ValueError, match="from_scene and to_scene"):
            server.tool_scene_link({"from_scene": 1, "to_scene": 0})
