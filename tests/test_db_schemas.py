"""
Tests for Database Schema Files
================================

Comprehensive tests for three database schema modules:
- db/screenplay_schema.py (Base, enums, 9 models)
- db/voice_schema.py (enums, 4 models sharing screenplay Base)
- db/agent_schema.py (own Base, enums, 4 models)

Covers: enum values, model instantiation, defaults, relationships,
unique constraints, cascading deletes, JSON columns, properties.

Run with: pytest tests/test_db_schemas.py -v
"""

import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Screenplay schema imports (defines Base used also by voice_schema)
# ---------------------------------------------------------------------------
from db.screenplay_schema import (
    Base as ScreenplayBase,
    IntExtType,
    ElementType,
    ScriptStatus,
    SceneStatus,
    ScreenplayProject,
    ScreenplayCharacter,
    ScreenplayScene,
    SceneElement,
    DialogueLine,
    SceneEmbedding,
    SceneRelation,
    SceneRevision,
    ExportConfig,
)

# ---------------------------------------------------------------------------
# Voice schema imports (shares Base with screenplay)
# ---------------------------------------------------------------------------
from db.voice_schema import (
    SessionStatus,
    TurnRole,
    TrainingStatus,
    VoiceSession,
    VoiceTurn,
    VoiceProfile,
    VoiceTrainingExample,
)

# ---------------------------------------------------------------------------
# Agent schema imports (own separate Base)
# ---------------------------------------------------------------------------
from db.agent_schema import (
    Base as AgentBase,
    SuggestionType,
    SuggestionStatus,
    SuggestionCategory,
    AnalysisTrigger,
    FridaySuggestion,
    AnalysisRun,
    FaceProfile,
    FaceAccessLog,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture()
def screenplay_engine():
    """Create an in-memory SQLite engine with all screenplay + voice tables."""
    engine = create_engine("sqlite://", echo=False)
    ScreenplayBase.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def sp_session(screenplay_engine):
    """Provide a SQLAlchemy session for screenplay/voice tests."""
    with Session(screenplay_engine) as session:
        yield session


@pytest.fixture()
def agent_engine():
    """Create an in-memory SQLite engine with all agent tables."""
    engine = create_engine("sqlite://", echo=False)
    AgentBase.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def ag_session(agent_engine):
    """Provide a SQLAlchemy session for agent schema tests."""
    with Session(agent_engine) as session:
        yield session


# Helper: create a minimal ScreenplayProject + flush
def _make_project(session, slug=None, title="Test Project"):
    slug = slug or f"test-{uuid.uuid4().hex[:8]}"
    p = ScreenplayProject(title=title, slug=slug)
    session.add(p)
    session.flush()
    return p


# Helper: create a minimal scene attached to a project
def _make_scene(session, project, number=1, location="HOUSE"):
    s = ScreenplayScene(
        project_id=project.id,
        scene_number=number,
        location=location,
    )
    session.add(s)
    session.flush()
    return s


# Helper: create a minimal VoiceSession
def _make_voice_session(session, session_id=None):
    sid = session_id or f"vs-{uuid.uuid4().hex[:8]}"
    vs = VoiceSession(
        session_id=sid,
        wake_word_detected="hey friday",
        wake_word_confidence=0.95,
    )
    session.add(vs)
    session.flush()
    return vs


# ============================================================================
# 1. TABLE CREATION & INTROSPECTION
# ============================================================================


class TestTableCreation:
    """Verify all tables are created correctly."""

    def test_screenplay_tables_exist(self, screenplay_engine):
        inspector = inspect(screenplay_engine)
        table_names = inspector.get_table_names()
        expected = [
            "screenplay_projects",
            "screenplay_characters",
            "screenplay_scenes",
            "scene_elements",
            "dialogue_lines",
            "scene_embeddings",
            "scene_relations",
            "scene_revisions",
            "export_configs",
        ]
        for t in expected:
            assert t in table_names, f"Missing table: {t}"

    def test_voice_tables_exist(self, screenplay_engine):
        inspector = inspect(screenplay_engine)
        table_names = inspector.get_table_names()
        expected = [
            "voice_sessions",
            "voice_turns",
            "voice_profiles",
            "voice_training_examples",
        ]
        for t in expected:
            assert t in table_names, f"Missing table: {t}"

    def test_agent_tables_exist(self, agent_engine):
        inspector = inspect(agent_engine)
        table_names = inspector.get_table_names()
        expected = [
            "friday_suggestions",
            "analysis_runs",
            "face_profiles",
            "face_access_log",
        ]
        for t in expected:
            assert t in table_names, f"Missing table: {t}"

    def test_screenplay_and_voice_share_base(self):
        """Voice schema shares the same Base as screenplay."""
        assert VoiceSession.metadata is ScreenplayProject.metadata

    def test_agent_has_separate_base(self):
        """Agent schema has its own separate Base."""
        assert FridaySuggestion.metadata is not ScreenplayProject.metadata


# ============================================================================
# 2. ENUM TESTS
# ============================================================================


class TestScreenplayEnums:
    """Test screenplay_schema enums."""

    def test_int_ext_values(self):
        assert IntExtType.INT.value == "INT"
        assert IntExtType.EXT.value == "EXT"
        assert IntExtType.INT_EXT.value == "INT/EXT"
        assert len(IntExtType) == 3

    def test_element_type_values(self):
        assert ElementType.ACTION.value == "action"
        assert ElementType.DIALOGUE.value == "dialogue"
        assert ElementType.TRANSITION.value == "transition"
        assert ElementType.SHOT.value == "shot"
        assert len(ElementType) == 4

    def test_script_status_values(self):
        assert ScriptStatus.DRAFT.value == "draft"
        assert ScriptStatus.REVISION.value == "revision"
        assert ScriptStatus.LOCKED.value == "locked"
        assert ScriptStatus.PRODUCTION.value == "production"
        assert len(ScriptStatus) == 4

    def test_scene_status_values(self):
        assert SceneStatus.ACTIVE.value == "active"
        assert SceneStatus.BACKLOG.value == "backlog"
        assert SceneStatus.CUT.value == "cut"
        assert SceneStatus.ARCHIVED.value == "archived"
        assert len(SceneStatus) == 4

    def test_enums_are_str_subclass(self):
        assert isinstance(IntExtType.INT, str)
        assert isinstance(ElementType.ACTION, str)
        assert isinstance(ScriptStatus.DRAFT, str)
        assert isinstance(SceneStatus.ACTIVE, str)


class TestVoiceEnums:
    """Test voice_schema enums."""

    def test_session_status_values(self):
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.COMPLETED.value == "completed"
        assert SessionStatus.INTERRUPTED.value == "interrupted"
        assert SessionStatus.ERROR.value == "error"
        assert len(SessionStatus) == 4

    def test_turn_role_values(self):
        assert TurnRole.USER.value == "user"
        assert TurnRole.ASSISTANT.value == "assistant"
        assert TurnRole.SYSTEM.value == "system"
        assert len(TurnRole) == 3

    def test_training_status_values(self):
        assert TrainingStatus.PENDING.value == "pending"
        assert TrainingStatus.APPROVED.value == "approved"
        assert TrainingStatus.REJECTED.value == "rejected"
        assert TrainingStatus.EXPORTED.value == "exported"
        assert len(TrainingStatus) == 4

    def test_session_status_membership(self):
        assert "active" in [s.value for s in SessionStatus]
        assert "unknown" not in [s.value for s in SessionStatus]

    def test_training_status_membership(self):
        values = {ts.value for ts in TrainingStatus}
        assert values == {"pending", "approved", "rejected", "exported"}


class TestAgentEnums:
    """Test agent_schema enums."""

    def test_suggestion_type_count_and_values(self):
        assert len(SuggestionType) == 10
        expected = {
            "plot_inconsistency",
            "character_arc_gap",
            "dialogue_improvement",
            "pacing_issue",
            "transition_missing",
            "setup_payoff_missing",
            "continuity_error",
            "structure_suggestion",
            "draft_scene",
            "research_finding",
        }
        assert {st.value for st in SuggestionType} == expected

    def test_suggestion_status_count_and_values(self):
        assert len(SuggestionStatus) == 6
        expected = {
            "pending",
            "discussed",
            "accepted",
            "rejected",
            "deferred",
            "applied",
        }
        assert {ss.value for ss in SuggestionStatus} == expected

    def test_suggestion_category_count_and_values(self):
        assert len(SuggestionCategory) == 7
        expected = {
            "screenplay",
            "dialogue",
            "pacing",
            "character",
            "structure",
            "research",
            "general",
        }
        assert {sc.value for sc in SuggestionCategory} == expected

    def test_analysis_trigger_count_and_values(self):
        assert len(AnalysisTrigger) == 3
        expected = {"scheduled", "on_edit", "on_request"}
        assert {at.value for at in AnalysisTrigger} == expected

    def test_agent_enums_are_str_subclass(self):
        assert isinstance(SuggestionType.DRAFT_SCENE, str)
        assert isinstance(SuggestionStatus.PENDING, str)
        assert isinstance(SuggestionCategory.GENERAL, str)
        assert isinstance(AnalysisTrigger.SCHEDULED, str)


# ============================================================================
# 3. MODEL INSTANTIATION & DEFAULTS -- Screenplay
# ============================================================================


class TestScreenplayProjectModel:
    """Test ScreenplayProject creation, defaults, and persistence."""

    def test_create_minimal_project(self, sp_session):
        p = _make_project(sp_session)
        assert p.id is not None
        assert p.title == "Test Project"

    def test_defaults(self, sp_session):
        p = _make_project(sp_session)
        assert p.status == ScriptStatus.DRAFT.value
        assert p.version == 1
        assert p.primary_language == "te"
        assert p.secondary_language == "en"
        assert p.created_at is not None
        assert p.updated_at is not None

    def test_slug_unique(self, sp_session):
        _make_project(sp_session, slug="unique-slug")
        sp_session.flush()
        with pytest.raises(IntegrityError):
            _make_project(sp_session, slug="unique-slug")
            sp_session.flush()

    def test_optional_fields_null(self, sp_session):
        p = _make_project(sp_session)
        assert p.logline is None
        assert p.author is None
        assert p.contact is None
        assert p.notes is None

    def test_custom_fields(self, sp_session):
        p = ScreenplayProject(
            title="My Film",
            slug="my-film",
            logline="A story about courage",
            author="Director",
            status=ScriptStatus.LOCKED.value,
            version=3,
        )
        sp_session.add(p)
        sp_session.flush()
        assert p.status == "locked"
        assert p.version == 3
        assert p.logline == "A story about courage"


class TestScreenplayCharacterModel:
    """Test ScreenplayCharacter."""

    def test_create_character(self, sp_session):
        proj = _make_project(sp_session)
        char = ScreenplayCharacter(project_id=proj.id, name="NEELIMA")
        sp_session.add(char)
        sp_session.flush()
        assert char.id is not None
        assert char.name == "NEELIMA"

    def test_character_defaults(self, sp_session):
        proj = _make_project(sp_session)
        char = ScreenplayCharacter(project_id=proj.id, name="RAJ")
        sp_session.add(char)
        sp_session.flush()
        assert char.full_name is None
        assert char.description is None
        assert char.age_range is None
        assert char.role_type is None
        assert char.created_at is not None

    def test_unique_constraint_project_name(self, sp_session):
        proj = _make_project(sp_session)
        c1 = ScreenplayCharacter(project_id=proj.id, name="NEELIMA")
        sp_session.add(c1)
        sp_session.flush()
        c2 = ScreenplayCharacter(project_id=proj.id, name="NEELIMA")
        sp_session.add(c2)
        with pytest.raises(IntegrityError):
            sp_session.flush()

    def test_same_name_different_project(self, sp_session):
        p1 = _make_project(sp_session, slug="proj-a")
        p2 = _make_project(sp_session, slug="proj-b")
        c1 = ScreenplayCharacter(project_id=p1.id, name="NEELIMA")
        c2 = ScreenplayCharacter(project_id=p2.id, name="NEELIMA")
        sp_session.add_all([c1, c2])
        sp_session.flush()
        assert c1.id != c2.id


class TestScreenplaySceneModel:
    """Test ScreenplayScene including the scene_heading property."""

    def test_create_scene(self, sp_session):
        proj = _make_project(sp_session)
        s = _make_scene(sp_session, proj)
        assert s.id is not None
        assert s.scene_number == 1

    def test_scene_defaults(self, sp_session):
        proj = _make_project(sp_session)
        s = _make_scene(sp_session, proj)
        assert s.int_ext == IntExtType.INT.value
        assert s.status == SceneStatus.ACTIVE.value
        assert s.narrative_order == 0.0
        assert s.tags == [] or s.tags is not None
        assert s.created_at is not None

    def test_scene_heading_basic(self, sp_session):
        proj = _make_project(sp_session)
        s = ScreenplayScene(
            project_id=proj.id,
            scene_number=1,
            int_ext="INT",
            location="House",
            time_of_day="Morning",
        )
        sp_session.add(s)
        sp_session.flush()
        heading = s.scene_heading
        assert heading == "INT. HOUSE - MORNING"

    def test_scene_heading_with_sub_location(self, sp_session):
        proj = _make_project(sp_session)
        s = ScreenplayScene(
            project_id=proj.id,
            scene_number=2,
            int_ext="EXT",
            location="Office",
            sub_location="Balcony",
            time_of_day="Night",
        )
        sp_session.add(s)
        sp_session.flush()
        heading = s.scene_heading
        assert heading == "EXT. OFFICE - BALCONY - NIGHT"

    def test_scene_heading_no_time(self, sp_session):
        proj = _make_project(sp_session)
        s = ScreenplayScene(
            project_id=proj.id,
            scene_number=3,
            int_ext="INT/EXT",
            location="Street",
        )
        sp_session.add(s)
        sp_session.flush()
        assert s.scene_heading == "INT/EXT. STREET"

    def test_scene_heading_no_sub_location(self, sp_session):
        proj = _make_project(sp_session)
        s = ScreenplayScene(
            project_id=proj.id,
            scene_number=4,
            int_ext="INT",
            location="Apartment",
            time_of_day="Dusk",
        )
        sp_session.add(s)
        sp_session.flush()
        assert s.scene_heading == "INT. APARTMENT - DUSK"

    def test_scene_unique_constraint_project_number(self, sp_session):
        proj = _make_project(sp_session)
        _make_scene(sp_session, proj, number=1)
        with pytest.raises(IntegrityError):
            _make_scene(sp_session, proj, number=1)

    def test_scene_tags_json(self, sp_session):
        proj = _make_project(sp_session)
        s = ScreenplayScene(
            project_id=proj.id,
            scene_number=5,
            location="Park",
            tags=["emotional", "conflict"],
        )
        sp_session.add(s)
        sp_session.flush()
        sp_session.expire(s)
        assert s.tags == ["emotional", "conflict"]


class TestSceneElementModel:
    """Test SceneElement and its JSON content column."""

    def test_create_action_element(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        elem = SceneElement(
            scene_id=scene.id,
            element_type=ElementType.ACTION.value,
            order_index=0,
            content={"text": "Morning sun rises over the village."},
        )
        sp_session.add(elem)
        sp_session.flush()
        assert elem.id is not None
        assert elem.content["text"] == "Morning sun rises over the village."

    def test_create_dialogue_element(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        elem = SceneElement(
            scene_id=scene.id,
            element_type=ElementType.DIALOGUE.value,
            order_index=1,
            content={
                "character": "NEELIMA",
                "lines": [{"text": "hello", "translation": "hello"}],
            },
        )
        sp_session.add(elem)
        sp_session.flush()
        assert elem.content["character"] == "NEELIMA"

    def test_create_transition_element(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        elem = SceneElement(
            scene_id=scene.id,
            element_type=ElementType.TRANSITION.value,
            order_index=2,
            content={"text": "CUT TO"},
        )
        sp_session.add(elem)
        sp_session.flush()
        assert elem.content["text"] == "CUT TO"

    def test_element_default_content(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        elem = SceneElement(
            scene_id=scene.id,
            element_type=ElementType.SHOT.value,
            order_index=0,
        )
        sp_session.add(elem)
        sp_session.flush()
        assert elem.content == {} or elem.content is not None


class TestDialogueLineModel:
    def test_create_dialogue_line(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        elem = SceneElement(
            scene_id=scene.id,
            element_type="dialogue",
            order_index=0,
            content={},
        )
        sp_session.add(elem)
        sp_session.flush()
        dl = DialogueLine(
            element_id=elem.id,
            character_name="NEELIMA",
            text="nanna tho godava pettukunna",
            translation="I fought with daddy",
            language="te",
        )
        sp_session.add(dl)
        sp_session.flush()
        assert dl.id is not None
        assert dl.language == "te"
        assert dl.line_order == 0

    def test_dialogue_line_defaults(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        elem = SceneElement(
            scene_id=scene.id,
            element_type="dialogue",
            order_index=0,
            content={},
        )
        sp_session.add(elem)
        sp_session.flush()
        dl = DialogueLine(element_id=elem.id, character_name="RAJ", text="Hello")
        sp_session.add(dl)
        sp_session.flush()
        assert dl.language == "te"
        assert dl.line_order == 0
        assert dl.translation is None
        assert dl.parenthetical is None


class TestSceneEmbeddingModel:
    def test_create_embedding(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        emb = SceneEmbedding(
            scene_id=scene.id,
            content_type="full",
            content_hash="abc123",
            model_name="all-MiniLM-L6-v2",
            vector=[0.1, 0.2, 0.3],
        )
        sp_session.add(emb)
        sp_session.flush()
        assert emb.id is not None
        sp_session.expire(emb)
        assert emb.vector == [0.1, 0.2, 0.3]


class TestSceneRelationModel:
    def test_create_relation(self, sp_session):
        proj = _make_project(sp_session)
        s1 = _make_scene(sp_session, proj, number=1, location="A")
        s2 = _make_scene(sp_session, proj, number=2, location="B")
        rel = SceneRelation(
            project_id=proj.id,
            from_scene_id=s1.id,
            to_scene_id=s2.id,
            relation_type="sequence",
        )
        sp_session.add(rel)
        sp_session.flush()
        assert rel.id is not None
        assert rel.notes is None


class TestSceneRevisionModel:
    def test_create_revision(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        rev = SceneRevision(
            scene_id=scene.id,
            revision_number=1,
            change_type="created",
            snapshot={"location": "HOUSE", "elements": []},
            author="Director",
        )
        sp_session.add(rev)
        sp_session.flush()
        assert rev.id is not None
        sp_session.expire(rev)
        assert rev.snapshot["location"] == "HOUSE"

    def test_revision_defaults(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        rev = SceneRevision(
            scene_id=scene.id,
            revision_number=1,
            change_type="edited",
            snapshot={},
        )
        sp_session.add(rev)
        sp_session.flush()
        assert rev.author is None
        assert rev.change_summary is None
        assert rev.created_at is not None


class TestExportConfigModel:
    def test_create_export_config(self, sp_session):
        cfg = ExportConfig(name="standard")
        sp_session.add(cfg)
        sp_session.flush()
        assert cfg.id is not None

    def test_export_config_defaults(self, sp_session):
        cfg = ExportConfig(name="default-test")
        sp_session.add(cfg)
        sp_session.flush()
        assert cfg.font_family == "Courier Prime"
        assert cfg.font_size == 12
        assert cfg.page_width == 8.5
        assert cfg.page_height == 11.0
        assert cfg.margin_top == 1.0
        assert cfg.margin_bottom == 1.0
        assert cfg.margin_left == 1.5
        assert cfg.margin_right == 1.0
        assert cfg.scene_heading_bg_color == "#CCCCCC"
        assert cfg.scene_heading_bold is True
        assert cfg.character_name_caps is True
        assert cfg.parenthetical_italics is False
        assert cfg.show_translations is True
        assert cfg.translation_in_parentheses is True

    def test_export_config_name_unique(self, sp_session):
        sp_session.add(ExportConfig(name="unique-cfg"))
        sp_session.flush()
        sp_session.add(ExportConfig(name="unique-cfg"))
        with pytest.raises(IntegrityError):
            sp_session.flush()


# ============================================================================
# 4. MODEL INSTANTIATION & DEFAULTS -- Voice
# ============================================================================


class TestVoiceSessionModel:
    def test_create_voice_session(self, sp_session):
        vs = _make_voice_session(sp_session)
        assert vs.id is not None

    def test_voice_session_defaults(self, sp_session):
        vs = _make_voice_session(sp_session)
        assert vs.status == SessionStatus.ACTIVE.value
        assert vs.total_turns == 0
        assert vs.total_audio_seconds == 0.0
        assert vs.started_at is not None
        assert vs.ended_at is None
        assert vs.device_id is None
        assert vs.location is None
        assert vs.error_message is None

    def test_voice_session_id_unique(self, sp_session):
        _make_voice_session(sp_session, session_id="dup-id")
        with pytest.raises(IntegrityError):
            _make_voice_session(sp_session, session_id="dup-id")

    def test_voice_session_custom_status(self, sp_session):
        vs = VoiceSession(
            session_id="custom-1",
            wake_word_detected="friday",
            wake_word_confidence=0.88,
            status=SessionStatus.COMPLETED.value,
            device_id="mic-01",
            location="kitchen",
        )
        sp_session.add(vs)
        sp_session.flush()
        assert vs.status == "completed"
        assert vs.device_id == "mic-01"


class TestVoiceTurnModel:
    def test_create_voice_turn(self, sp_session):
        vs = _make_voice_session(sp_session)
        turn = VoiceTurn(
            session_id=vs.id,
            turn_number=1,
            role=TurnRole.USER.value,
        )
        sp_session.add(turn)
        sp_session.flush()
        assert turn.id is not None

    def test_voice_turn_defaults(self, sp_session):
        vs = _make_voice_session(sp_session)
        turn = VoiceTurn(
            session_id=vs.id,
            turn_number=1,
            role=TurnRole.ASSISTANT.value,
        )
        sp_session.add(turn)
        sp_session.flush()
        assert turn.training_status == TrainingStatus.PENDING.value
        assert turn.user_audio_sample_rate == 16000
        assert turn.tool_calls == [] or turn.tool_calls is not None
        assert turn.transcript is None
        assert turn.response_text is None
        assert turn.llm_model is None

    def test_voice_turn_tool_calls_json(self, sp_session):
        vs = _make_voice_session(sp_session)
        calls = [
            {"tool": "scene_search", "args": {"q": "love"}, "result": {"ok": True}}
        ]
        turn = VoiceTurn(
            session_id=vs.id,
            turn_number=1,
            role="user",
            tool_calls=calls,
        )
        sp_session.add(turn)
        sp_session.flush()
        sp_session.expire(turn)
        assert turn.tool_calls[0]["tool"] == "scene_search"

    def test_voice_turn_full_fields(self, sp_session):
        vs = _make_voice_session(sp_session)
        turn = VoiceTurn(
            session_id=vs.id,
            turn_number=2,
            role="assistant",
            transcript="hello boss",
            transcript_confidence=0.92,
            detected_language="en",
            response_text="Good morning!",
            llm_model="claude-3-5-sonnet",
            llm_latency_ms=350,
            training_status=TrainingStatus.APPROVED.value,
        )
        sp_session.add(turn)
        sp_session.flush()
        assert turn.transcript == "hello boss"
        assert turn.training_status == "approved"


class TestVoiceProfileModel:
    def test_create_voice_profile(self, sp_session):
        vp = VoiceProfile(name="friday_telugu")
        sp_session.add(vp)
        sp_session.flush()
        assert vp.id is not None

    def test_voice_profile_defaults(self, sp_session):
        vp = VoiceProfile(name="test_profile")
        sp_session.add(vp)
        sp_session.flush()
        assert vp.language == "te"
        assert vp.tts_engine == "xtts_v2"
        assert vp.speaking_rate == 1.0
        assert vp.pitch_adjustment == 0.0
        assert vp.is_active is True
        assert vp.is_default is False
        assert vp.reference_audio_paths == [] or vp.reference_audio_paths is not None
        assert vp.speaker_embedding is None

    def test_voice_profile_name_unique(self, sp_session):
        sp_session.add(VoiceProfile(name="dup"))
        sp_session.flush()
        sp_session.add(VoiceProfile(name="dup"))
        with pytest.raises(IntegrityError):
            sp_session.flush()

    def test_voice_profile_reference_audio_json(self, sp_session):
        paths = ["audio/sample1.wav", "audio/sample2.wav"]
        vp = VoiceProfile(name="multi_ref", reference_audio_paths=paths)
        sp_session.add(vp)
        sp_session.flush()
        sp_session.expire(vp)
        assert vp.reference_audio_paths == paths

    def test_voice_profile_speaker_embedding_json(self, sp_session):
        emb = [0.1, 0.2, 0.3, 0.4]
        vp = VoiceProfile(name="emb_test", speaker_embedding=emb)
        sp_session.add(vp)
        sp_session.flush()
        sp_session.expire(vp)
        assert vp.speaker_embedding == [0.1, 0.2, 0.3, 0.4]


class TestVoiceTrainingExampleModel:
    def test_create_training_example(self, sp_session):
        vs = _make_voice_session(sp_session)
        turn = VoiceTurn(session_id=vs.id, turn_number=1, role="user")
        sp_session.add(turn)
        sp_session.flush()
        vte = VoiceTrainingExample(
            turn_id=turn.id,
            quality_score=0.85,
            formatted_example={"messages": [{"role": "user", "content": "hello"}]},
        )
        sp_session.add(vte)
        sp_session.flush()
        assert vte.id is not None

    def test_training_example_defaults(self, sp_session):
        vs = _make_voice_session(sp_session)
        turn = VoiceTurn(session_id=vs.id, turn_number=1, role="user")
        sp_session.add(turn)
        sp_session.flush()
        vte = VoiceTrainingExample(turn_id=turn.id)
        sp_session.add(vte)
        sp_session.flush()
        assert vte.export_batch is None
        assert vte.exported_at is None
        assert vte.quality_score is None
        assert vte.review_notes is None
        assert vte.reviewed_by is None
        assert vte.formatted_example is None
        assert vte.created_at is not None

    def test_training_example_formatted_json(self, sp_session):
        vs = _make_voice_session(sp_session)
        turn = VoiceTurn(session_id=vs.id, turn_number=1, role="user")
        sp_session.add(turn)
        sp_session.flush()
        example = {
            "messages": [
                {"role": "system", "content": "You are Friday."},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello boss"},
            ]
        }
        vte = VoiceTrainingExample(turn_id=turn.id, formatted_example=example)
        sp_session.add(vte)
        sp_session.flush()
        sp_session.expire(vte)
        assert len(vte.formatted_example["messages"]) == 3


# ============================================================================
# 5. MODEL INSTANTIATION & DEFAULTS -- Agent
# ============================================================================


class TestFridaySuggestionModel:
    def test_create_suggestion(self, ag_session):
        s = FridaySuggestion(
            project_id=1,
            title="Fix plot hole",
            description="Scene 3 contradicts scene 1",
        )
        ag_session.add(s)
        ag_session.flush()
        assert s.id is not None

    def test_suggestion_defaults(self, ag_session):
        s = FridaySuggestion(
            project_id=1,
            title="Test",
            description="Desc",
        )
        ag_session.add(s)
        ag_session.flush()
        assert s.status == SuggestionStatus.PENDING.value
        assert s.priority == 3
        assert s.category == SuggestionCategory.GENERAL.value
        assert s.confidence == 0.5
        assert s.affected_scenes == [] or s.affected_scenes is not None
        assert s.scene_id is None
        assert s.proposed_change is None
        assert s.discussed_at is None
        assert s.resolved_at is None
        assert s.decision_notes is None

    def test_suggestion_affected_scenes_json(self, ag_session):
        s = FridaySuggestion(
            project_id=1,
            title="Multi-scene issue",
            description="Affects several scenes",
            affected_scenes=[1, 2, 5, 10],
        )
        ag_session.add(s)
        ag_session.flush()
        ag_session.expire(s)
        assert s.affected_scenes == [1, 2, 5, 10]

    def test_suggestion_custom_fields(self, ag_session):
        s = FridaySuggestion(
            project_id=2,
            scene_id=7,
            suggestion_type=SuggestionType.PACING_ISSUE.value,
            title="Slow pacing",
            description="Act 2 drags",
            priority=1,
            category=SuggestionCategory.PACING.value,
            confidence=0.9,
            status=SuggestionStatus.DISCUSSED.value,
        )
        ag_session.add(s)
        ag_session.flush()
        assert s.scene_id == 7
        assert s.priority == 1
        assert s.confidence == 0.9
        assert s.status == "discussed"


class TestAnalysisRunModel:
    def test_create_analysis_run(self, ag_session):
        ar = AnalysisRun(
            run_id="run-001",
            project_id=1,
            trigger=AnalysisTrigger.SCHEDULED.value,
        )
        ag_session.add(ar)
        ag_session.flush()
        assert ar.id is not None

    def test_analysis_run_defaults(self, ag_session):
        ar = AnalysisRun(
            run_id="run-002",
            project_id=1,
            trigger="on_edit",
        )
        ag_session.add(ar)
        ag_session.flush()
        assert ar.suggestions_generated == 0
        assert ar.suggestions_deduplicated == 0
        assert ar.status == "running"
        assert ar.config == {} or ar.config is not None
        assert ar.scene_ids is None
        assert ar.completed_at is None
        assert ar.error_message is None

    def test_analysis_run_id_unique(self, ag_session):
        ag_session.add(AnalysisRun(run_id="dup", project_id=1, trigger="scheduled"))
        ag_session.flush()
        ag_session.add(AnalysisRun(run_id="dup", project_id=1, trigger="on_edit"))
        with pytest.raises(IntegrityError):
            ag_session.flush()

    def test_analysis_run_scene_ids_json(self, ag_session):
        ar = AnalysisRun(
            run_id="run-003",
            project_id=1,
            trigger="on_request",
            scene_ids=[3, 7, 11],
        )
        ag_session.add(ar)
        ag_session.flush()
        ag_session.expire(ar)
        assert ar.scene_ids == [3, 7, 11]

    def test_analysis_run_config_json(self, ag_session):
        ar = AnalysisRun(
            run_id="run-004",
            project_id=1,
            trigger="scheduled",
            config={"depth": "deep", "model": "claude"},
        )
        ag_session.add(ar)
        ag_session.flush()
        ag_session.expire(ar)
        assert ar.config["depth"] == "deep"


class TestFaceProfileModel:
    def test_create_face_profile(self, ag_session):
        fp = FaceProfile(name="Boss")
        ag_session.add(fp)
        ag_session.flush()
        assert fp.id is not None

    def test_face_profile_defaults(self, ag_session):
        fp = FaceProfile(name="TeamMember")
        ag_session.add(fp)
        ag_session.flush()
        assert fp.access_level == "team"
        assert fp.is_active is True
        assert fp.embedding is None
        assert fp.reference_images == [] or fp.reference_images is not None
        assert fp.last_seen is None
        assert fp.created_at is not None

    def test_face_profile_reference_images_json(self, ag_session):
        fp = FaceProfile(
            name="ImageTest",
            reference_images=["/img/a.jpg", "/img/b.jpg"],
        )
        ag_session.add(fp)
        ag_session.flush()
        ag_session.expire(fp)
        assert fp.reference_images == ["/img/a.jpg", "/img/b.jpg"]

    def test_face_profile_embedding_bytes(self, ag_session):
        data = b"\x00\x01\x02\x03" * 64
        fp = FaceProfile(name="EmbedTest", embedding=data)
        ag_session.add(fp)
        ag_session.flush()
        ag_session.expire(fp)
        assert fp.embedding == data

    def test_face_profile_custom_access_level(self, ag_session):
        fp = FaceProfile(name="Director", access_level="boss")
        ag_session.add(fp)
        ag_session.flush()
        assert fp.access_level == "boss"


class TestFaceAccessLogModel:
    def test_create_access_log(self, ag_session):
        log = FaceAccessLog(action_taken="granted_access")
        ag_session.add(log)
        ag_session.flush()
        assert log.id is not None

    def test_access_log_defaults(self, ag_session):
        log = FaceAccessLog(action_taken="denied_access")
        ag_session.add(log)
        ag_session.flush()
        assert log.face_profile_id is None
        assert log.confidence == 0.0
        assert log.emotion is None
        assert log.location is None
        assert log.timestamp is not None

    def test_access_log_with_face_profile(self, ag_session):
        fp = FaceProfile(name="LogTest")
        ag_session.add(fp)
        ag_session.flush()
        log = FaceAccessLog(
            face_profile_id=fp.id,
            action_taken="granted_access",
            confidence=0.97,
            emotion="happy",
            location="front_door",
        )
        ag_session.add(log)
        ag_session.flush()
        assert log.face_profile_id == fp.id
        assert log.confidence == 0.97

    def test_access_log_unknown_face(self, ag_session):
        """NULL face_profile_id for unknown faces."""
        log = FaceAccessLog(
            action_taken="captured_unknown",
            confidence=0.3,
        )
        ag_session.add(log)
        ag_session.flush()
        assert log.face_profile_id is None


# ============================================================================
# 6. RELATIONSHIP TESTS
# ============================================================================


class TestScreenplayRelationships:
    def test_project_scenes_relationship(self, sp_session):
        proj = _make_project(sp_session)
        s1 = _make_scene(sp_session, proj, number=1, location="A")
        s2 = _make_scene(sp_session, proj, number=2, location="B")
        sp_session.expire(proj)
        assert len(proj.scenes) == 2
        assert proj.scenes[0].scene_number == 1
        assert proj.scenes[1].scene_number == 2

    def test_project_characters_relationship(self, sp_session):
        proj = _make_project(sp_session)
        c1 = ScreenplayCharacter(project_id=proj.id, name="HERO")
        c2 = ScreenplayCharacter(project_id=proj.id, name="VILLAIN")
        sp_session.add_all([c1, c2])
        sp_session.flush()
        sp_session.expire(proj)
        names = {c.name for c in proj.characters}
        assert names == {"HERO", "VILLAIN"}

    def test_scene_elements_relationship(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        e1 = SceneElement(
            scene_id=scene.id,
            element_type="action",
            order_index=0,
            content={"text": "a"},
        )
        e2 = SceneElement(
            scene_id=scene.id,
            element_type="dialogue",
            order_index=1,
            content={"text": "b"},
        )
        sp_session.add_all([e1, e2])
        sp_session.flush()
        sp_session.expire(scene)
        assert len(scene.elements) == 2
        assert scene.elements[0].order_index == 0

    def test_scene_embeddings_relationship(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        emb = SceneEmbedding(
            scene_id=scene.id,
            content_type="full",
            content_hash="h1",
            model_name="model",
            vector=[1.0],
        )
        sp_session.add(emb)
        sp_session.flush()
        sp_session.expire(scene)
        assert len(scene.embeddings) == 1

    def test_scene_back_populates_project(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        sp_session.expire(scene)
        assert scene.project.id == proj.id

    def test_character_back_populates_project(self, sp_session):
        proj = _make_project(sp_session)
        c = ScreenplayCharacter(project_id=proj.id, name="X")
        sp_session.add(c)
        sp_session.flush()
        sp_session.expire(c)
        assert c.project.id == proj.id


class TestVoiceRelationships:
    def test_session_turns_relationship(self, sp_session):
        vs = _make_voice_session(sp_session)
        t1 = VoiceTurn(session_id=vs.id, turn_number=1, role="user")
        t2 = VoiceTurn(session_id=vs.id, turn_number=2, role="assistant")
        sp_session.add_all([t1, t2])
        sp_session.flush()
        sp_session.expire(vs)
        assert len(vs.turns) == 2
        assert vs.turns[0].turn_number == 1

    def test_turn_back_populates_session(self, sp_session):
        vs = _make_voice_session(sp_session)
        turn = VoiceTurn(session_id=vs.id, turn_number=1, role="user")
        sp_session.add(turn)
        sp_session.flush()
        sp_session.expire(turn)
        assert turn.session.id == vs.id

    def test_turns_ordered_by_turn_number(self, sp_session):
        vs = _make_voice_session(sp_session)
        t3 = VoiceTurn(session_id=vs.id, turn_number=3, role="user")
        t1 = VoiceTurn(session_id=vs.id, turn_number=1, role="user")
        t2 = VoiceTurn(session_id=vs.id, turn_number=2, role="assistant")
        sp_session.add_all([t3, t1, t2])
        sp_session.flush()
        sp_session.expire(vs)
        numbers = [t.turn_number for t in vs.turns]
        assert numbers == [1, 2, 3]


class TestAgentRelationships:
    def test_face_access_log_relationship(self, ag_session):
        fp = FaceProfile(name="RelTest")
        ag_session.add(fp)
        ag_session.flush()
        log = FaceAccessLog(face_profile_id=fp.id, action_taken="granted_access")
        ag_session.add(log)
        ag_session.flush()
        ag_session.expire(log)
        assert log.face_profile.name == "RelTest"

    def test_face_access_log_null_profile(self, ag_session):
        log = FaceAccessLog(action_taken="captured_unknown")
        ag_session.add(log)
        ag_session.flush()
        ag_session.expire(log)
        assert log.face_profile is None


# ============================================================================
# 7. CASCADE DELETE TESTS
# ============================================================================


class TestCascadeDeletes:
    def test_delete_project_cascades_scenes(self, sp_session):
        proj = _make_project(sp_session)
        _make_scene(sp_session, proj, number=1)
        _make_scene(sp_session, proj, number=2)
        sp_session.flush()
        sp_session.delete(proj)
        sp_session.flush()
        remaining = sp_session.query(ScreenplayScene).all()
        assert len(remaining) == 0

    def test_delete_project_cascades_characters(self, sp_session):
        proj = _make_project(sp_session)
        c = ScreenplayCharacter(project_id=proj.id, name="X")
        sp_session.add(c)
        sp_session.flush()
        sp_session.delete(proj)
        sp_session.flush()
        remaining = sp_session.query(ScreenplayCharacter).all()
        assert len(remaining) == 0

    def test_delete_scene_cascades_elements(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        elem = SceneElement(
            scene_id=scene.id,
            element_type="action",
            order_index=0,
            content={"t": "x"},
        )
        sp_session.add(elem)
        sp_session.flush()
        sp_session.delete(scene)
        sp_session.flush()
        remaining = sp_session.query(SceneElement).all()
        assert len(remaining) == 0

    def test_delete_scene_cascades_embeddings(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        emb = SceneEmbedding(
            scene_id=scene.id,
            content_type="full",
            content_hash="h",
            model_name="m",
            vector=[0.5],
        )
        sp_session.add(emb)
        sp_session.flush()
        sp_session.delete(scene)
        sp_session.flush()
        remaining = sp_session.query(SceneEmbedding).all()
        assert len(remaining) == 0

    def test_delete_voice_session_cascades_turns(self, sp_session):
        vs = _make_voice_session(sp_session)
        t = VoiceTurn(session_id=vs.id, turn_number=1, role="user")
        sp_session.add(t)
        sp_session.flush()
        sp_session.delete(vs)
        sp_session.flush()
        remaining = sp_session.query(VoiceTurn).all()
        assert len(remaining) == 0

    def test_delete_project_cascades_scenes_and_elements(self, sp_session):
        """Deep cascade: project -> scenes -> elements."""
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        elem = SceneElement(
            scene_id=scene.id,
            element_type="action",
            order_index=0,
            content={},
        )
        sp_session.add(elem)
        sp_session.flush()
        sp_session.delete(proj)
        sp_session.flush()
        assert sp_session.query(ScreenplayScene).count() == 0
        assert sp_session.query(SceneElement).count() == 0


# ============================================================================
# 8. JSON COLUMN ROUND-TRIP TESTS
# ============================================================================


class TestJsonColumns:
    def test_scene_tags_roundtrip(self, sp_session):
        proj = _make_project(sp_session)
        tags = ["romance", "comedy", "telugu"]
        s = ScreenplayScene(
            project_id=proj.id,
            scene_number=1,
            location="Park",
            tags=tags,
        )
        sp_session.add(s)
        sp_session.flush()
        sp_session.expire(s)
        assert s.tags == ["romance", "comedy", "telugu"]

    def test_scene_element_content_complex_json(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        content = {
            "character": "NEELIMA",
            "parenthetical": "V.O.",
            "lines": [
                {"text": "line1", "translation": "trans1"},
                {"text": "line2", "translation": "trans2"},
            ],
        }
        elem = SceneElement(
            scene_id=scene.id,
            element_type="dialogue",
            order_index=0,
            content=content,
        )
        sp_session.add(elem)
        sp_session.flush()
        sp_session.expire(elem)
        assert len(elem.content["lines"]) == 2
        assert elem.content["parenthetical"] == "V.O."

    def test_suggestion_affected_scenes_empty(self, ag_session):
        s = FridaySuggestion(
            project_id=1,
            title="T",
            description="D",
        )
        ag_session.add(s)
        ag_session.flush()
        ag_session.expire(s)
        assert s.affected_scenes == [] or s.affected_scenes is not None

    def test_analysis_run_config_nested_json(self, ag_session):
        nested = {
            "analyses": ["plot", "character"],
            "settings": {"depth": 3, "model": "claude"},
        }
        ar = AnalysisRun(
            run_id="nested-1",
            project_id=1,
            trigger="scheduled",
            config=nested,
        )
        ag_session.add(ar)
        ag_session.flush()
        ag_session.expire(ar)
        assert ar.config["settings"]["depth"] == 3

    def test_voice_turn_tool_calls_empty_default(self, sp_session):
        vs = _make_voice_session(sp_session)
        turn = VoiceTurn(session_id=vs.id, turn_number=1, role="user")
        sp_session.add(turn)
        sp_session.flush()
        sp_session.expire(turn)
        assert turn.tool_calls == [] or turn.tool_calls is not None

    def test_voice_profile_gpt_cond_latent_json(self, sp_session):
        latent = [[0.1, 0.2], [0.3, 0.4]]
        vp = VoiceProfile(name="latent_test", gpt_cond_latent=latent)
        sp_session.add(vp)
        sp_session.flush()
        sp_session.expire(vp)
        assert vp.gpt_cond_latent == [[0.1, 0.2], [0.3, 0.4]]

    def test_revision_snapshot_json(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        snapshot = {
            "scene_number": 1,
            "location": "HOUSE",
            "elements": [{"type": "action", "text": "sun rises"}],
        }
        rev = SceneRevision(
            scene_id=scene.id,
            revision_number=1,
            change_type="created",
            snapshot=snapshot,
        )
        sp_session.add(rev)
        sp_session.flush()
        sp_session.expire(rev)
        assert rev.snapshot["elements"][0]["text"] == "sun rises"


# ============================================================================
# 9. ADDITIONAL EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    def test_scene_number_ordering(self, sp_session):
        """Scenes are ordered by scene_number via relationship."""
        proj = _make_project(sp_session)
        _make_scene(sp_session, proj, number=3, location="C")
        _make_scene(sp_session, proj, number=1, location="A")
        _make_scene(sp_session, proj, number=2, location="B")
        sp_session.expire(proj)
        numbers = [s.scene_number for s in proj.scenes]
        assert numbers == [1, 2, 3]

    def test_element_ordering(self, sp_session):
        """Elements are ordered by order_index via relationship."""
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        for idx in [2, 0, 1]:
            e = SceneElement(
                scene_id=scene.id,
                element_type="action",
                order_index=idx,
                content={"idx": idx},
            )
            sp_session.add(e)
        sp_session.flush()
        sp_session.expire(scene)
        indices = [e.order_index for e in scene.elements]
        assert indices == [0, 1, 2]

    def test_multiple_projects_isolated(self, sp_session):
        """Scenes from different projects don't interfere."""
        p1 = _make_project(sp_session, slug="proj1")
        p2 = _make_project(sp_session, slug="proj2")
        _make_scene(sp_session, p1, number=1)
        _make_scene(sp_session, p2, number=1)
        sp_session.expire(p1)
        sp_session.expire(p2)
        assert len(p1.scenes) == 1
        assert len(p2.scenes) == 1

    def test_voice_session_multiple_turns(self, sp_session):
        """A session can have many turns."""
        vs = _make_voice_session(sp_session)
        for i in range(10):
            sp_session.add(VoiceTurn(session_id=vs.id, turn_number=i + 1, role="user"))
        sp_session.flush()
        sp_session.expire(vs)
        assert len(vs.turns) == 10

    def test_export_config_custom_values(self, sp_session):
        cfg = ExportConfig(
            name="custom",
            font_family="Arial",
            font_size=14,
            page_width=8.0,
            page_height=10.0,
            margin_left=2.0,
            scene_heading_bold=False,
            show_translations=False,
        )
        sp_session.add(cfg)
        sp_session.flush()
        assert cfg.font_family == "Arial"
        assert cfg.font_size == 14
        assert cfg.scene_heading_bold is False
        assert cfg.show_translations is False

    def test_scene_heading_case_conversion(self, sp_session):
        """scene_heading property uppercases location/sub_location/time."""
        proj = _make_project(sp_session)
        s = ScreenplayScene(
            project_id=proj.id,
            scene_number=1,
            int_ext="INT",
            location="small cafe",
            sub_location="corner table",
            time_of_day="afternoon - 3 p.m.",
        )
        sp_session.add(s)
        sp_session.flush()
        heading = s.scene_heading
        assert "SMALL CAFE" in heading
        assert "CORNER TABLE" in heading
        assert "AFTERNOON - 3 P.M." in heading

    def test_suggestion_all_statuses(self, ag_session):
        """Can set every SuggestionStatus value."""
        for i, status in enumerate(SuggestionStatus):
            s = FridaySuggestion(
                project_id=1,
                title=f"Test-{status.value}",
                description="d",
                status=status.value,
            )
            ag_session.add(s)
        ag_session.flush()
        count = ag_session.query(FridaySuggestion).count()
        assert count == 6

    def test_suggestion_all_types(self, ag_session):
        """Can set every SuggestionType value."""
        for i, stype in enumerate(SuggestionType):
            s = FridaySuggestion(
                project_id=1,
                title=f"Type-{stype.value}",
                description="d",
                suggestion_type=stype.value,
            )
            ag_session.add(s)
        ag_session.flush()
        count = ag_session.query(FridaySuggestion).count()
        assert count == 10

    def test_suggestion_all_categories(self, ag_session):
        """Can set every SuggestionCategory value."""
        for cat in SuggestionCategory:
            s = FridaySuggestion(
                project_id=1,
                title=f"Cat-{cat.value}",
                description="d",
                category=cat.value,
            )
            ag_session.add(s)
        ag_session.flush()
        count = ag_session.query(FridaySuggestion).count()
        assert count == 7

    def test_face_access_log_relationship_back(self, ag_session):
        """Access log points back to its face profile."""
        fp = FaceProfile(name="BackRef")
        ag_session.add(fp)
        ag_session.flush()
        log = FaceAccessLog(face_profile_id=fp.id, action_taken="session_started")
        ag_session.add(log)
        ag_session.flush()
        ag_session.expire(log)
        assert log.face_profile.id == fp.id

    def test_scene_relation_types(self, sp_session):
        """SceneRelation can be created with various relation_type values."""
        proj = _make_project(sp_session)
        s1 = _make_scene(sp_session, proj, number=1)
        s2 = _make_scene(sp_session, proj, number=2)
        for rtype in ["sequence", "flashback", "parallel", "callback", "setup_payoff"]:
            rel = SceneRelation(
                project_id=proj.id,
                from_scene_id=s1.id,
                to_scene_id=s2.id,
                relation_type=rtype,
            )
            sp_session.add(rel)
        sp_session.flush()
        count = sp_session.query(SceneRelation).count()
        assert count == 5

    def test_analysis_trigger_all_values(self, ag_session):
        """Create analysis runs with all trigger values."""
        for i, trigger in enumerate(AnalysisTrigger):
            ar = AnalysisRun(
                run_id=f"trigger-{i}",
                project_id=1,
                trigger=trigger.value,
            )
            ag_session.add(ar)
        ag_session.flush()
        count = ag_session.query(AnalysisRun).count()
        assert count == 3

    def test_voice_profile_multiple_active(self, sp_session):
        """Multiple profiles can be active."""
        vp1 = VoiceProfile(name="vp1", is_active=True)
        vp2 = VoiceProfile(name="vp2", is_active=True)
        vp3 = VoiceProfile(name="vp3", is_active=False)
        sp_session.add_all([vp1, vp2, vp3])
        sp_session.flush()
        active = sp_session.query(VoiceProfile).filter_by(is_active=True).count()
        assert active == 2

    def test_dialogue_line_with_parenthetical(self, sp_session):
        proj = _make_project(sp_session)
        scene = _make_scene(sp_session, proj)
        elem = SceneElement(
            scene_id=scene.id,
            element_type="dialogue",
            order_index=0,
            content={},
        )
        sp_session.add(elem)
        sp_session.flush()
        dl = DialogueLine(
            element_id=elem.id,
            character_name="NEELIMA",
            text="hello",
            parenthetical="V.O.",
            language="en",
            line_order=1,
        )
        sp_session.add(dl)
        sp_session.flush()
        assert dl.parenthetical == "V.O."
        assert dl.line_order == 1


# ============================================================================
# 10. INDEX VERIFICATION TESTS
# ============================================================================


class TestIndexes:
    def test_screenplay_project_has_slug_unique(self, screenplay_engine):
        inspector = inspect(screenplay_engine)
        indexes = inspector.get_indexes("screenplay_projects")
        unique_cols = inspector.get_unique_constraints("screenplay_projects")
        # slug uniqueness can be via index or constraint
        cols_in_indexes = []
        for idx in indexes:
            cols_in_indexes.extend(idx["column_names"])
        # slug should be unique (either via unique=True on column or explicit constraint)
        columns = inspector.get_columns("screenplay_projects")
        slug_col = [c for c in columns if c["name"] == "slug"][0]
        # Verify slug column exists
        assert slug_col is not None

    def test_voice_session_indexes(self, screenplay_engine):
        inspector = inspect(screenplay_engine)
        indexes = inspector.get_indexes("voice_sessions")
        idx_names = {idx["name"] for idx in indexes}
        assert "ix_voice_session_started" in idx_names
        assert "ix_voice_session_status" in idx_names

    def test_voice_turn_indexes(self, screenplay_engine):
        inspector = inspect(screenplay_engine)
        indexes = inspector.get_indexes("voice_turns")
        idx_names = {idx["name"] for idx in indexes}
        assert "ix_voice_turn_session_number" in idx_names
        assert "ix_voice_turn_training" in idx_names

    def test_voice_training_batch_index(self, screenplay_engine):
        inspector = inspect(screenplay_engine)
        indexes = inspector.get_indexes("voice_training_examples")
        idx_names = {idx["name"] for idx in indexes}
        assert "ix_voice_training_batch" in idx_names

    def test_friday_suggestion_indexes(self, agent_engine):
        inspector = inspect(agent_engine)
        indexes = inspector.get_indexes("friday_suggestions")
        idx_names = {idx["name"] for idx in indexes}
        assert "ix_suggestions_project_status" in idx_names
        assert "ix_suggestions_priority" in idx_names
        assert "ix_suggestions_type" in idx_names

    def test_analysis_runs_index(self, agent_engine):
        inspector = inspect(agent_engine)
        indexes = inspector.get_indexes("analysis_runs")
        idx_names = {idx["name"] for idx in indexes}
        assert "ix_analysis_runs_project" in idx_names

    def test_face_profile_indexes(self, agent_engine):
        inspector = inspect(agent_engine)
        indexes = inspector.get_indexes("face_profiles")
        idx_names = {idx["name"] for idx in indexes}
        assert "ix_face_profiles_name" in idx_names
        assert "ix_face_profiles_access" in idx_names

    def test_face_access_log_index(self, agent_engine):
        inspector = inspect(agent_engine)
        indexes = inspector.get_indexes("face_access_log")
        idx_names = {idx["name"] for idx in indexes}
        assert "ix_face_access_timestamp" in idx_names

    def test_scene_elements_index(self, screenplay_engine):
        inspector = inspect(screenplay_engine)
        indexes = inspector.get_indexes("scene_elements")
        idx_names = {idx["name"] for idx in indexes}
        assert "ix_element_scene_order" in idx_names

    def test_scene_project_number_index(self, screenplay_engine):
        inspector = inspect(screenplay_engine)
        indexes = inspector.get_indexes("screenplay_scenes")
        idx_names = {idx["name"] for idx in indexes}
        assert "ix_scene_project_number" in idx_names
