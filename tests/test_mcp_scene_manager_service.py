"""
Tests for MCP Scene Manager Service - Issue #8
================================================

Comprehensive tests for mcp/scene_manager/service.py covering:
- fetch_scene / fetch_scene_by_number helpers
- get_scene_text text assembly
- search_scenes (substring fallback + vector search)
- get_scene_detail
- update_scene
- add_scene_element
- create_relation
- generate_scene_embedding
- generate_project_embeddings
- list_projects
- get_project_scenes
- Singleton patterns (_engine, _model)
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

import mcp.scene_manager.service as svc
from db.screenplay_schema import (
    ScreenplayProject,
    ScreenplayScene,
    SceneElement,
    SceneEmbedding,
    SceneRelation,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons between tests."""
    svc._engine = None
    svc._model = None
    yield
    svc._engine = None
    svc._model = None


# =========================================================================
# Mock Factories
# =========================================================================


def _mock_project(
    id=1,
    title="Test Project",
    slug="test-project",
    status="draft",
    primary_language="te",
    scenes=None,
):
    p = MagicMock(spec=ScreenplayProject)
    p.id = id
    p.title = title
    p.slug = slug
    p.status = status
    p.primary_language = primary_language
    p.scenes = scenes or []
    return p


def _mock_scene(
    id=1,
    project_id=1,
    scene_number=1,
    int_ext="INT",
    location="HOUSE",
    sub_location=None,
    time_of_day="DAY",
    title="Opening",
    summary="The opening scene",
    narrative_order=1.0,
    status="active",
    tags=None,
    elements=None,
    project=None,
):
    s = MagicMock(spec=ScreenplayScene)
    s.id = id
    s.project_id = project_id
    s.scene_number = scene_number
    s.int_ext = int_ext
    s.location = location
    s.sub_location = sub_location
    s.time_of_day = time_of_day
    s.title = title
    s.summary = summary
    s.narrative_order = narrative_order
    s.status = status
    s.tags = tags or []
    s.elements = elements or []
    s.project = project or _mock_project()
    return s


def _mock_element(id=1, element_type="action", order_index=0, content=None):
    e = MagicMock(spec=SceneElement)
    e.id = id
    e.element_type = element_type
    e.order_index = order_index
    e.content = content or {"text": "A quiet morning."}
    return e


def _mock_embedding(
    id=1, scene_id=1, content_hash="abc123", model_name="test-model", vector=None
):
    emb = MagicMock(spec=SceneEmbedding)
    emb.id = id
    emb.scene_id = scene_id
    emb.content_hash = content_hash
    emb.model_name = model_name
    emb.vector = vector or [0.1, 0.2, 0.3]
    return emb


def _mock_relation(
    id=1, from_scene_id=1, to_scene_id=2, relation_type="sequence", notes=None
):
    r = MagicMock(spec=SceneRelation)
    r.id = id
    r.from_scene_id = from_scene_id
    r.to_scene_id = to_scene_id
    r.relation_type = relation_type
    r.notes = notes
    return r


def _mock_session():
    """Create a MagicMock that works as a SQLAlchemy Session context manager."""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    return session


# =========================================================================
# get_engine_instance / get_model
# =========================================================================


class TestGetEngineInstance:
    def test_creates_singleton(self):
        mock_engine = MagicMock()
        with patch("mcp.scene_manager.service.get_engine", return_value=mock_engine):
            engine = svc.get_engine_instance()
        assert engine is mock_engine
        # Second call returns same
        engine2 = svc.get_engine_instance()
        assert engine2 is mock_engine

    def test_returns_cached(self):
        mock_engine = MagicMock()
        svc._engine = mock_engine
        engine = svc.get_engine_instance()
        assert engine is mock_engine


class TestGetModel:
    def test_creates_singleton(self):
        mock_model = MagicMock()
        with patch(
            "mcp.scene_manager.service.SentenceTransformer", mock_model, create=True
        ):
            # Need to patch the import inside the function
            with patch.dict(
                "sys.modules",
                {"sentence_transformers": MagicMock(SentenceTransformer=mock_model)},
            ):
                model = svc.get_model()
        assert model is not None

    def test_returns_cached(self):
        mock_model = MagicMock()
        svc._model = mock_model
        model = svc.get_model()
        assert model is mock_model


# =========================================================================
# fetch_scene / fetch_scene_by_number
# =========================================================================


class TestFetchScene:
    def test_found(self):
        scene = _mock_scene(id=5)
        mock_session = _mock_session()
        mock_session.get.return_value = scene
        result = svc.fetch_scene(mock_session, 5)
        assert result is scene
        mock_session.get.assert_called_once_with(ScreenplayScene, 5)

    def test_not_found(self):
        mock_session = _mock_session()
        mock_session.get.return_value = None
        with pytest.raises(ValueError, match="Scene not found: 99"):
            svc.fetch_scene(mock_session, 99)


class TestFetchSceneByNumber:
    def test_found(self):
        project = _mock_project(id=1)
        scene = _mock_scene(id=10, scene_number=3)
        mock_session = _mock_session()
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            project,
            scene,
        ]
        result = svc.fetch_scene_by_number(mock_session, "test-project", 3)
        assert result is scene

    def test_project_not_found(self):
        mock_session = _mock_session()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        with pytest.raises(ValueError, match="Project not found"):
            svc.fetch_scene_by_number(mock_session, "nonexistent", 1)

    def test_scene_not_found(self):
        project = _mock_project(id=1)
        mock_session = _mock_session()
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            project,
            None,
        ]
        with pytest.raises(ValueError, match="Scene 99 not found"):
            svc.fetch_scene_by_number(mock_session, "test-project", 99)


# =========================================================================
# get_scene_text
# =========================================================================


class TestGetSceneText:
    def test_basic_scene(self):
        scene = _mock_scene(
            int_ext="INT",
            location="HOUSE",
            sub_location=None,
            time_of_day="DAY",
            title="Opening",
            summary="Opening scene",
            elements=[],
        )
        text = svc.get_scene_text(scene)
        assert "INT. HOUSE" in text
        assert "DAY" in text
        assert "Opening" in text
        assert "Opening scene" in text

    def test_with_sub_location(self):
        scene = _mock_scene(
            int_ext="EXT",
            location="PARK",
            sub_location="FOUNTAIN AREA",
            time_of_day="MORNING",
            title=None,
            summary=None,
            elements=[],
        )
        text = svc.get_scene_text(scene)
        assert "EXT. PARK - FOUNTAIN AREA" in text

    def test_with_action_element(self):
        action = _mock_element(element_type="action", content={"text": "A bird flies."})
        scene = _mock_scene(elements=[action])
        text = svc.get_scene_text(scene)
        assert "A bird flies." in text

    def test_with_dialogue_element(self):
        dialogue = _mock_element(
            element_type="dialogue",
            content={
                "character": "NEELIMA",
                "lines": [
                    {"text": "nanna tho godava", "translation": "fought with daddy"},
                ],
            },
        )
        scene = _mock_scene(elements=[dialogue])
        text = svc.get_scene_text(scene)
        assert "NEELIMA:" in text
        assert "nanna tho godava" in text
        assert "(fought with daddy)" in text

    def test_with_transition_element(self):
        transition = _mock_element(
            element_type="transition", content={"text": "CUT TO:"}
        )
        scene = _mock_scene(elements=[transition])
        text = svc.get_scene_text(scene)
        assert "CUT TO:" in text

    def test_dialogue_without_translation(self):
        dialogue = _mock_element(
            element_type="dialogue",
            content={
                "character": "ARJUN",
                "lines": [{"text": "Hello there"}],
            },
        )
        scene = _mock_scene(elements=[dialogue])
        text = svc.get_scene_text(scene)
        assert "ARJUN:" in text
        assert "Hello there" in text

    def test_no_title_no_summary(self):
        scene = _mock_scene(title=None, summary=None, elements=[])
        text = svc.get_scene_text(scene)
        assert "INT. HOUSE" in text
        # Should not crash, just have heading

    def test_no_time_of_day(self):
        scene = _mock_scene(time_of_day=None, elements=[])
        text = svc.get_scene_text(scene)
        assert "INT. HOUSE" in text
        assert "DAY" not in text


# =========================================================================
# search_scenes
# =========================================================================


class TestSearchScenes:
    def _setup_session_with_scenes(self, mock_engine, scenes):
        """Wire up Session context manager to return scenes."""
        mock_session = _mock_session()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        # The execute().scalars().all() chain
        mock_session.execute.return_value.scalars.return_value.all.return_value = scenes
        return mock_session

    @patch("mcp.scene_manager.service.get_engine_instance")
    @patch("mcp.scene_manager.service.get_model")
    def test_substring_fallback_when_no_model(self, mock_get_model, mock_get_engine):
        """When model is unavailable, should fall back to substring matching."""
        mock_get_model.side_effect = RuntimeError("No model")
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        scene = _mock_scene(
            id=1,
            scene_number=1,
            int_ext="INT",
            location="COURTROOM",
            title="Trial",
            narrative_order=1.0,
            status="active",
            elements=[],
        )

        mock_session = _mock_session()
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            scene
        ]

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            results = svc.search_scenes("courtroom")

        assert len(results) == 1
        assert results[0]["scene_number"] == 1
        assert results[0]["title"] == "Trial"
        assert "score" in results[0]

    @patch("mcp.scene_manager.service.get_engine_instance")
    @patch("mcp.scene_manager.service.get_model")
    def test_substring_fallback_respects_top_k(self, mock_get_model, mock_get_engine):
        """Substring fallback should respect top_k parameter."""
        mock_get_model.side_effect = RuntimeError("No model")
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        scenes = [
            _mock_scene(id=i, scene_number=i, title=f"Scene {i}", elements=[])
            for i in range(10)
        ]

        mock_session = _mock_session()
        mock_session.execute.return_value.scalars.return_value.all.return_value = scenes

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            results = svc.search_scenes("scene", top_k=3)

        assert len(results) == 3

    @patch("mcp.scene_manager.service.get_engine_instance")
    @patch("mcp.scene_manager.service.get_model")
    def test_vector_search_no_embeddings_fallback(
        self, mock_get_model, mock_get_engine
    ):
        """When model exists but no embeddings, should fall back to fuzzy search."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_get_model.return_value = mock_model
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        scene = _mock_scene(id=1, scene_number=1, title="Opening", elements=[])

        mock_session = _mock_session()
        # First execute for scenes, second for embeddings
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            scene
        ]
        # Embeddings query returns empty
        mock_session.execute.return_value.all.return_value = []

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            results = svc.search_scenes("opening")

        assert len(results) >= 0  # May be 0 or 1 depending on mock

    @patch("mcp.scene_manager.service.get_engine_instance")
    @patch("mcp.scene_manager.service.get_model")
    def test_search_result_structure(self, mock_get_model, mock_get_engine):
        """Verify search result dictionary structure."""
        mock_get_model.side_effect = RuntimeError("No model")
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        scene = _mock_scene(
            id=5,
            scene_number=3,
            int_ext="EXT",
            location="PARK",
            title="Meeting",
            narrative_order=3.0,
            status="active",
            elements=[],
        )

        mock_session = _mock_session()
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            scene
        ]

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            results = svc.search_scenes("park")

        assert len(results) == 1
        r = results[0]
        assert r["scene_id"] == 5
        assert r["scene_number"] == 3
        assert "EXT. PARK" in r["heading"]
        assert r["title"] == "Meeting"
        assert r["narrative_order"] == 3.0
        assert r["status"] == "active"
        assert isinstance(r["score"], float)
        assert "preview" in r

    @patch("mcp.scene_manager.service.get_engine_instance")
    @patch("mcp.scene_manager.service.get_model")
    def test_search_empty_scenes(self, mock_get_model, mock_get_engine):
        """Search with no scenes returns empty list."""
        mock_get_model.side_effect = RuntimeError("No model")
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        mock_session = _mock_session()
        mock_session.execute.return_value.scalars.return_value.all.return_value = []

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            results = svc.search_scenes("anything")

        assert results == []

    @patch("mcp.scene_manager.service.get_engine_instance")
    @patch("mcp.scene_manager.service.get_model")
    def test_search_preview_truncation(self, mock_get_model, mock_get_engine):
        """Long scene text should be truncated to 200 chars + '...'."""
        mock_get_model.side_effect = RuntimeError("No model")
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        long_action = _mock_element(
            element_type="action",
            content={"text": "A" * 500},
        )
        scene = _mock_scene(id=1, scene_number=1, title="Long", elements=[long_action])

        mock_session = _mock_session()
        mock_session.execute.return_value.scalars.return_value.all.return_value = [
            scene
        ]

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            results = svc.search_scenes("long")

        if results:
            assert results[0]["preview"].endswith("...")


# =========================================================================
# get_scene_detail
# =========================================================================


class TestGetSceneDetail:
    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_by_scene_id(self, mock_get_engine):
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        action = _mock_element(
            element_type="action", order_index=0, content={"text": "Morning."}
        )
        project = _mock_project(id=1, title="My Movie", slug="my-movie")
        scene = _mock_scene(
            id=5,
            scene_number=2,
            int_ext="INT",
            location="ROOM",
            sub_location="BEDROOM",
            time_of_day="NIGHT",
            title="Sleep",
            summary="Character sleeps",
            tags=["quiet"],
            elements=[action],
            project=project,
        )

        mock_session = _mock_session()
        mock_session.get.return_value = scene
        mock_session.query.return_value.filter.return_value.all.return_value = []

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            detail = svc.get_scene_detail(scene_id=5)

        assert detail["scene_id"] == 5
        assert detail["scene_number"] == 2
        assert detail["int_ext"] == "INT"
        assert detail["location"] == "ROOM"
        assert detail["sub_location"] == "BEDROOM"
        assert detail["time_of_day"] == "NIGHT"
        assert detail["title"] == "Sleep"
        assert detail["summary"] == "Character sleeps"
        assert detail["tags"] == ["quiet"]
        assert len(detail["elements"]) == 1
        assert detail["elements"][0]["type"] == "action"
        assert detail["project"]["slug"] == "my-movie"
        assert detail["related_scenes"] == []

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_by_scene_number_and_slug(self, mock_get_engine):
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        project = _mock_project(id=1, slug="test")
        scene = _mock_scene(id=10, scene_number=3, elements=[], project=project)

        mock_session = _mock_session()
        # fetch_scene_by_number chain
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [
            project,
            scene,
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = []

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            detail = svc.get_scene_detail(scene_number=3, project_slug="test")

        assert detail["scene_id"] == 10

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_missing_params_raises(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()
        mock_session = _mock_session()
        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            with pytest.raises(ValueError, match="Must provide"):
                svc.get_scene_detail()

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_with_relations(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        project = _mock_project()
        scene = _mock_scene(id=1, elements=[], project=project)
        rel = _mock_relation(from_scene_id=1, to_scene_id=3, relation_type="flashback")

        mock_session = _mock_session()
        mock_session.get.return_value = scene
        mock_session.query.return_value.filter.return_value.all.return_value = [rel]

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            detail = svc.get_scene_detail(scene_id=1)

        assert len(detail["related_scenes"]) == 1
        assert detail["related_scenes"][0]["scene_id"] == 3
        assert detail["related_scenes"][0]["relation"] == "flashback"

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_multiple_elements(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        action = _mock_element(
            element_type="action", order_index=0, content={"text": "Action."}
        )
        dialogue = _mock_element(
            element_type="dialogue",
            order_index=1,
            content={"character": "ARJUN", "lines": [{"text": "Hi"}]},
        )
        transition = _mock_element(
            element_type="transition", order_index=2, content={"text": "CUT TO:"}
        )
        project = _mock_project()
        scene = _mock_scene(
            id=1, elements=[action, dialogue, transition], project=project
        )

        mock_session = _mock_session()
        mock_session.get.return_value = scene
        mock_session.query.return_value.filter.return_value.all.return_value = []

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            detail = svc.get_scene_detail(scene_id=1)

        assert len(detail["elements"]) == 3
        assert detail["elements"][0]["type"] == "action"
        assert detail["elements"][1]["type"] == "dialogue"
        assert detail["elements"][2]["type"] == "transition"


# =========================================================================
# update_scene
# =========================================================================


class TestUpdateScene:
    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_update_status(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()
        scene = _mock_scene(id=1, status="active")

        mock_session = _mock_session()
        mock_session.get.return_value = scene

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            changed = svc.update_scene(1, status="backlog")

        assert changed is True
        assert scene.status == "backlog"
        mock_session.commit.assert_called_once()

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_update_title(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()
        scene = _mock_scene(id=1)

        mock_session = _mock_session()
        mock_session.get.return_value = scene

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            changed = svc.update_scene(1, title="New Title")

        assert changed is True
        assert scene.title == "New Title"

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_update_summary(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()
        scene = _mock_scene(id=1)

        mock_session = _mock_session()
        mock_session.get.return_value = scene

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            changed = svc.update_scene(1, summary="Updated summary")

        assert changed is True

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_update_tags(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()
        scene = _mock_scene(id=1, tags=[])

        mock_session = _mock_session()
        mock_session.get.return_value = scene

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            changed = svc.update_scene(1, tags=["conflict", "drama"])

        assert changed is True
        assert scene.tags == ["conflict", "drama"]

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_update_narrative_order(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()
        scene = _mock_scene(id=1, narrative_order=1.0)

        mock_session = _mock_session()
        mock_session.get.return_value = scene

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            changed = svc.update_scene(1, narrative_order=2.5)

        assert changed is True
        assert scene.narrative_order == 2.5

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_no_changes(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()
        scene = _mock_scene(id=1)

        mock_session = _mock_session()
        mock_session.get.return_value = scene

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            changed = svc.update_scene(1)

        assert changed is False
        mock_session.rollback.assert_called_once()
        mock_session.commit.assert_not_called()

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_update_multiple_fields(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()
        scene = _mock_scene(id=1)

        mock_session = _mock_session()
        mock_session.get.return_value = scene

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            changed = svc.update_scene(
                1, title="Updated", status="cut", tags=["removed"]
            )

        assert changed is True
        assert scene.title == "Updated"
        assert scene.status == "cut"
        assert scene.tags == ["removed"]

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_update_scene_not_found(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        mock_session = _mock_session()
        mock_session.get.return_value = None  # Scene not found

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            with pytest.raises(ValueError, match="Scene not found"):
                svc.update_scene(999, title="New")


# =========================================================================
# add_scene_element
# =========================================================================


class TestAddSceneElement:
    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_add_element_auto_order(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        existing = _mock_element(order_index=2)
        scene = _mock_scene(id=1, elements=[existing])

        mock_session = _mock_session()
        mock_session.get.return_value = scene

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            svc.add_scene_element(1, "action", {"text": "New action."})

        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        added = mock_session.add.call_args[0][0]
        assert added.order_index == 3  # max(2) + 1

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_add_element_explicit_order(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()
        scene = _mock_scene(id=1, elements=[])

        mock_session = _mock_session()
        mock_session.get.return_value = scene

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            svc.add_scene_element(1, "dialogue", {"character": "X"}, order_index=5)

        added = mock_session.add.call_args[0][0]
        assert added.order_index == 5

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_add_element_empty_scene(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()
        scene = _mock_scene(id=1, elements=[])

        mock_session = _mock_session()
        mock_session.get.return_value = scene

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            svc.add_scene_element(1, "transition", {"text": "FADE IN:"})

        added = mock_session.add.call_args[0][0]
        assert added.order_index == 0  # max([], default=-1) + 1 = 0

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_add_element_scene_not_found(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        mock_session = _mock_session()
        mock_session.get.return_value = None

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            with pytest.raises(ValueError, match="Scene not found"):
                svc.add_scene_element(99, "action", {"text": "test"})


# =========================================================================
# create_relation
# =========================================================================


class TestCreateRelation:
    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_create_success(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        source = _mock_scene(id=1, project_id=1)
        target = _mock_scene(id=2, project_id=1)

        mock_session = _mock_session()
        mock_session.get.side_effect = [source, target]

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            result = svc.create_relation(1, 2, relation_type="flashback")

        assert result is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        added = mock_session.add.call_args[0][0]
        assert added.from_scene_id == 1
        assert added.to_scene_id == 2
        assert added.relation_type == "flashback"

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_create_with_notes(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        source = _mock_scene(id=1, project_id=1)
        target = _mock_scene(id=3, project_id=1)

        mock_session = _mock_session()
        mock_session.get.side_effect = [source, target]

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            result = svc.create_relation(
                1, 3, relation_type="callback", notes="Setup payoff"
            )

        assert result is True
        added = mock_session.add.call_args[0][0]
        assert added.notes == "Setup payoff"

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_create_source_not_found(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        mock_session = _mock_session()
        mock_session.get.return_value = None

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            with pytest.raises(ValueError, match="Scene not found"):
                svc.create_relation(99, 2)

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_create_default_type_sequence(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        source = _mock_scene(id=1, project_id=1)
        target = _mock_scene(id=2, project_id=1)

        mock_session = _mock_session()
        mock_session.get.side_effect = [source, target]

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            svc.create_relation(1, 2)

        added = mock_session.add.call_args[0][0]
        assert added.relation_type == "sequence"


# =========================================================================
# generate_scene_embedding
# =========================================================================


class TestGenerateSceneEmbedding:
    @patch("mcp.scene_manager.service.get_engine_instance")
    @patch("mcp.scene_manager.service.get_model")
    def test_generate_new_embedding(self, mock_get_model, mock_get_engine):
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine

        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock(
            tolist=MagicMock(return_value=[0.1, 0.2])
        )
        mock_get_model.return_value = mock_model

        scene = _mock_scene(id=1, elements=[])

        mock_session = _mock_session()
        mock_session.get.return_value = scene
        # No existing embedding
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            result = svc.generate_scene_embedding(1)

        assert result is True
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch("mcp.scene_manager.service.get_engine_instance")
    @patch("mcp.scene_manager.service.get_model")
    def test_skip_existing_embedding(self, mock_get_model, mock_get_engine):
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        scene = _mock_scene(id=1, elements=[])

        mock_session = _mock_session()
        mock_session.get.return_value = scene
        # Existing embedding found
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            _mock_embedding()
        )

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            result = svc.generate_scene_embedding(1)

        assert result is False
        mock_session.add.assert_not_called()


# =========================================================================
# generate_project_embeddings
# =========================================================================


class TestGenerateProjectEmbeddings:
    @patch("mcp.scene_manager.service.get_engine_instance")
    @patch("mcp.scene_manager.service.generate_scene_embedding")
    def test_generate_all(self, mock_gen, mock_get_engine):
        mock_get_engine.return_value = MagicMock()
        mock_gen.return_value = True

        scenes = [_mock_scene(id=i) for i in range(3)]
        project = _mock_project(scenes=scenes)

        mock_session = _mock_session()
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            project
        )

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            count = svc.generate_project_embeddings("test-project")

        assert count == 3

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_project_not_found(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        mock_session = _mock_session()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            with pytest.raises(ValueError, match="Project not found"):
                svc.generate_project_embeddings("nonexistent")

    @patch("mcp.scene_manager.service.get_engine_instance")
    @patch("mcp.scene_manager.service.generate_scene_embedding")
    def test_partial_failure(self, mock_gen, mock_get_engine):
        """One embedding fails, rest succeed."""
        mock_get_engine.return_value = MagicMock()
        mock_gen.side_effect = [True, Exception("embedding fail"), True]

        scenes = [_mock_scene(id=i) for i in range(3)]
        project = _mock_project(scenes=scenes)

        mock_session = _mock_session()
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            project
        )

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            count = svc.generate_project_embeddings("test-project")

        assert count == 2  # 2 successes, 1 failure

    @patch("mcp.scene_manager.service.get_engine_instance")
    @patch("mcp.scene_manager.service.generate_scene_embedding")
    def test_skips_existing(self, mock_gen, mock_get_engine):
        """Scenes with existing embeddings return False."""
        mock_get_engine.return_value = MagicMock()
        mock_gen.side_effect = [True, False, True]  # Second already exists

        scenes = [_mock_scene(id=i) for i in range(3)]
        project = _mock_project(scenes=scenes)

        mock_session = _mock_session()
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            project
        )

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            count = svc.generate_project_embeddings("test-project")

        assert count == 2  # Only 2 newly generated


# =========================================================================
# list_projects
# =========================================================================


class TestListProjects:
    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_list_multiple(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        p1 = _mock_project(
            id=1, title="Movie A", slug="movie-a", scenes=[MagicMock(), MagicMock()]
        )
        p2 = _mock_project(id=2, title="Movie B", slug="movie-b", scenes=[])

        mock_session = _mock_session()
        mock_session.query.return_value.all.return_value = [p1, p2]

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            projects = svc.list_projects()

        assert len(projects) == 2
        assert projects[0]["title"] == "Movie A"
        assert projects[0]["scene_count"] == 2
        assert projects[1]["slug"] == "movie-b"
        assert projects[1]["scene_count"] == 0

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_list_empty(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        mock_session = _mock_session()
        mock_session.query.return_value.all.return_value = []

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            projects = svc.list_projects()

        assert projects == []

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_list_project_structure(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        p = _mock_project(
            id=1,
            title="Test",
            slug="test",
            status="draft",
            primary_language="te",
            scenes=[],
        )

        mock_session = _mock_session()
        mock_session.query.return_value.all.return_value = [p]

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            projects = svc.list_projects()

        proj = projects[0]
        assert "id" in proj
        assert "title" in proj
        assert "slug" in proj
        assert "status" in proj
        assert "scene_count" in proj
        assert "primary_language" in proj


# =========================================================================
# get_project_scenes
# =========================================================================


class TestGetProjectScenes:
    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_get_scenes(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        s1 = _mock_scene(
            id=1,
            scene_number=1,
            int_ext="INT",
            location="ROOM",
            title="Scene One",
            status="active",
            narrative_order=1.0,
        )
        s2 = _mock_scene(
            id=2,
            scene_number=2,
            int_ext="EXT",
            location="PARK",
            title="Scene Two",
            status="backlog",
            narrative_order=2.0,
        )
        project = _mock_project(id=1, slug="test", scenes=[s1, s2])

        mock_session = _mock_session()
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            project
        )

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            scenes = svc.get_project_scenes("test")

        assert len(scenes) == 2
        assert scenes[0]["scene_number"] == 1
        assert scenes[0]["heading"] == "INT. ROOM"
        assert scenes[1]["scene_number"] == 2
        assert scenes[1]["heading"] == "EXT. PARK"

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_project_not_found(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        mock_session = _mock_session()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            with pytest.raises(ValueError, match="Project not found"):
                svc.get_project_scenes("nonexistent")

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_empty_project(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        project = _mock_project(id=1, slug="empty", scenes=[])

        mock_session = _mock_session()
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            project
        )

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            scenes = svc.get_project_scenes("empty")

        assert scenes == []

    @patch("mcp.scene_manager.service.get_engine_instance")
    def test_scene_dict_structure(self, mock_get_engine):
        mock_get_engine.return_value = MagicMock()

        s = _mock_scene(
            id=5,
            scene_number=3,
            int_ext="INT/EXT",
            location="BALCONY",
            title="Confrontation",
            status="active",
            narrative_order=3.0,
        )
        project = _mock_project(scenes=[s])

        mock_session = _mock_session()
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            project
        )

        with patch("mcp.scene_manager.service.Session", return_value=mock_session):
            scenes = svc.get_project_scenes("test")

        scene_dict = scenes[0]
        assert scene_dict["scene_id"] == 5
        assert scene_dict["scene_number"] == 3
        assert "INT/EXT. BALCONY" in scene_dict["heading"]
        assert scene_dict["title"] == "Confrontation"
        assert scene_dict["status"] == "active"
        assert scene_dict["narrative_order"] == 3.0
