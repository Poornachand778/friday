"""
Tests for memory/layers/profile.py
====================================

Comprehensive tests for ProfileStore, UserProfile, Relationship, Project.
Covers JSON persistence, version history, voice confirmation, preference
learning, relationships, projects, rollback, and edge cases.

Tests: 80+
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from memory.layers.profile import (
    ProfileStore,
    Relationship,
    Project,
    UserProfile,
    VoiceConfirmationRequired,
)


# ── Helpers ───────────────────────────────────────────────────────────────


@pytest.fixture
def profile_dir(tmp_path):
    """Temporary directory structure for profiles."""
    profile_path = tmp_path / "profile.json"
    history_path = tmp_path / "history"
    return profile_path, history_path


@pytest.fixture
def mock_config(profile_dir):
    """Mock ProfileConfig."""
    profile_path, history_path = profile_dir
    config = MagicMock()
    config.profile_path = str(profile_path)
    config.history_path = str(history_path)
    config.max_history_versions = 10
    return config


@pytest.fixture
def store(mock_config):
    """Initialized ProfileStore."""
    with patch("memory.layers.profile.get_memory_config") as mock_get:
        mock_mem_config = MagicMock()
        mock_mem_config.profile = mock_config
        mock_get.return_value = mock_mem_config
        s = ProfileStore(config=mock_config)
    return s


@pytest.fixture
async def initialized_store(store):
    """Fully initialized store."""
    await store.initialize()
    return store


# ── VoiceConfirmationRequired ─────────────────────────────────────────────


class TestVoiceConfirmationRequired:
    def test_exception(self):
        with pytest.raises(VoiceConfirmationRequired):
            raise VoiceConfirmationRequired("test")

    def test_exception_message(self):
        try:
            raise VoiceConfirmationRequired("change name")
        except VoiceConfirmationRequired as e:
            assert "change name" in str(e)


# ── Relationship Dataclass ────────────────────────────────────────────────


class TestRelationship:
    def test_create(self):
        r = Relationship(name="Ravi", relation="character", project="gusagusalu")
        assert r.name == "Ravi"
        assert r.relation == "character"
        assert r.project == "gusagusalu"

    def test_to_dict(self):
        r = Relationship(
            name="Ravi", relation="character", project="proj1", notes="hero"
        )
        d = r.to_dict()
        assert d["name"] == "Ravi"
        assert d["relation"] == "character"
        assert d["notes"] == "hero"
        assert "added_at" in d

    def test_from_dict(self):
        data = {
            "name": "Ravi",
            "relation": "character",
            "project": "proj1",
            "notes": "hero",
            "added_at": "2025-01-15T10:00:00",
        }
        r = Relationship.from_dict(data)
        assert r.name == "Ravi"
        assert r.relation == "character"
        assert r.project == "proj1"

    def test_from_dict_defaults(self):
        data = {"name": "X", "relation": "friend"}
        r = Relationship.from_dict(data)
        assert r.notes == ""
        assert r.project is None

    def test_roundtrip(self):
        r = Relationship(name="Ravi", relation="character", project="proj1")
        d = r.to_dict()
        r2 = Relationship.from_dict(d)
        assert r2.name == r.name
        assert r2.relation == r.relation


# ── Project Dataclass ─────────────────────────────────────────────────────


class TestProject:
    def test_create(self):
        p = Project(name="Gusagusalu", slug="gusagusalu", status="active")
        assert p.name == "Gusagusalu"
        assert p.slug == "gusagusalu"
        assert p.status == "active"

    def test_to_dict(self):
        p = Project(
            name="Kitchen",
            slug="kitchen",
            status="active",
            description="Kitchen drama",
            deadline=datetime(2025, 3, 1),
        )
        d = p.to_dict()
        assert d["name"] == "Kitchen"
        assert d["slug"] == "kitchen"
        assert d["deadline"] is not None

    def test_to_dict_no_deadline(self):
        p = Project(name="X", slug="x", status="active")
        d = p.to_dict()
        assert d["deadline"] is None

    def test_from_dict(self):
        data = {
            "name": "Gusagusalu",
            "slug": "gusagusalu",
            "status": "active",
            "description": "Film",
            "deadline": "2025-03-01T00:00:00",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-15T00:00:00",
        }
        p = Project.from_dict(data)
        assert p.name == "Gusagusalu"
        assert p.deadline == datetime(2025, 3, 1)

    def test_from_dict_no_deadline(self):
        data = {"name": "X", "slug": "x", "status": "active"}
        p = Project.from_dict(data)
        assert p.deadline is None

    def test_roundtrip(self):
        p = Project(name="Test", slug="test", status="active")
        d = p.to_dict()
        p2 = Project.from_dict(d)
        assert p2.name == p.name
        assert p2.slug == p.slug


# ── UserProfile ───────────────────────────────────────────────────────────


class TestUserProfile:
    def test_defaults(self):
        p = UserProfile()
        assert p.static["name"] == "Poorna"
        assert p.static["address_as"] == "Boss"
        assert p.dynamic["current_project"] is None
        assert p.version == 1

    def test_to_dict(self):
        p = UserProfile()
        d = p.to_dict()
        assert "static" in d
        assert "dynamic" in d
        assert "preferences" in d
        assert "relationships" in d
        assert "projects" in d
        assert d["version"] == 1

    def test_from_dict(self):
        data = {
            "static": {"name": "Test"},
            "dynamic": {"current_project": "proj1"},
            "preferences": {"lang": "en"},
            "relationships": {},
            "projects": {},
            "version": 5,
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-15T00:00:00",
        }
        p = UserProfile.from_dict(data)
        assert p.static["name"] == "Test"
        assert p.version == 5

    def test_from_dict_with_relationships(self):
        data = {
            "relationships": {
                "ravi": {
                    "name": "Ravi",
                    "relation": "character",
                    "project": "proj1",
                    "added_at": "2025-01-01T00:00:00",
                }
            },
        }
        p = UserProfile.from_dict(data)
        assert "ravi" in p.relationships
        assert p.relationships["ravi"].name == "Ravi"

    def test_from_dict_with_projects(self):
        data = {
            "projects": {
                "gusagusalu": {
                    "name": "Gusagusalu",
                    "slug": "gusagusalu",
                    "status": "active",
                    "created_at": "2025-01-01T00:00:00",
                    "updated_at": "2025-01-15T00:00:00",
                }
            },
        }
        p = UserProfile.from_dict(data)
        assert "gusagusalu" in p.projects

    def test_roundtrip(self):
        p = UserProfile()
        p.static["name"] = "TestUser"
        d = p.to_dict()
        p2 = UserProfile.from_dict(d)
        assert p2.static["name"] == "TestUser"


# ── ProfileStore Init & Load ─────────────────────────────────────────────


class TestProfileStoreInit:
    @pytest.mark.asyncio
    async def test_initialize_creates_profile(self, store):
        await store.initialize()
        assert store.profile is not None
        assert store.profile.static["name"] == "Poorna"

    @pytest.mark.asyncio
    async def test_initialize_creates_file(self, store, profile_dir):
        profile_path, _ = profile_dir
        await store.initialize()
        assert Path(profile_path).exists()

    @pytest.mark.asyncio
    async def test_initialize_creates_history_dir(self, store, profile_dir):
        _, history_path = profile_dir
        await store.initialize()
        assert Path(history_path).exists()

    @pytest.mark.asyncio
    async def test_initialize_loads_existing(self, store, profile_dir):
        profile_path, history_path = profile_dir
        Path(history_path).mkdir(parents=True, exist_ok=True)

        # Write an existing profile
        data = UserProfile()
        data.static["name"] = "ExistingUser"
        data.version = 42
        Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w") as f:
            json.dump(data.to_dict(), f)

        await store.initialize()
        assert store.profile.static["name"] == "ExistingUser"
        assert store.profile.version == 42

    @pytest.mark.asyncio
    async def test_initialize_corrupt_file_recovers(self, store, profile_dir):
        profile_path, history_path = profile_dir
        Path(history_path).mkdir(parents=True, exist_ok=True)
        Path(profile_path).parent.mkdir(parents=True, exist_ok=True)

        # Write corrupt JSON
        with open(profile_path, "w") as f:
            f.write("not valid json {{{}}")

        await store.initialize()
        # Should recover with new profile
        assert store.profile is not None


# ── Static Updates ────────────────────────────────────────────────────────


class TestStaticUpdates:
    @pytest.mark.asyncio
    async def test_update_static_with_voice(self, store):
        await store.initialize()
        store.update_static("name", "NewName", voice_confirmed=True)
        assert store.get_static("name") == "NewName"

    @pytest.mark.asyncio
    async def test_update_static_without_voice_raises(self, store):
        await store.initialize()
        with pytest.raises(VoiceConfirmationRequired):
            store.update_static("name", "NewName", voice_confirmed=False)

    @pytest.mark.asyncio
    async def test_update_static_increments_version(self, store):
        await store.initialize()
        v1 = store.profile.version
        store.update_static("name", "NewName", voice_confirmed=True)
        assert store.profile.version == v1 + 1

    @pytest.mark.asyncio
    async def test_update_static_same_value_no_change(self, store):
        await store.initialize()
        v1 = store.profile.version
        store.update_static("name", "Poorna", voice_confirmed=True)  # Same as default
        assert store.profile.version == v1  # No increment

    @pytest.mark.asyncio
    async def test_update_static_creates_audit(self, store):
        await store.initialize()
        store.update_static("role", "Director", voice_confirmed=True)
        log = store.get_audit_log()
        assert len(log) > 0
        assert log[-1]["operation"] == "static_update"
        assert log[-1]["key"] == "role"

    @pytest.mark.asyncio
    async def test_update_static_persists(self, store, profile_dir):
        profile_path, _ = profile_dir
        await store.initialize()
        store.update_static("name", "Persistent", voice_confirmed=True)

        # Reload
        with open(profile_path) as f:
            data = json.load(f)
        assert data["static"]["name"] == "Persistent"


# ── Dynamic Updates ───────────────────────────────────────────────────────


class TestDynamicUpdates:
    @pytest.mark.asyncio
    async def test_update_dynamic(self, store):
        await store.initialize()
        store.update_dynamic("current_project", "gusagusalu")
        assert store.get_dynamic("current_project") == "gusagusalu"

    @pytest.mark.asyncio
    async def test_update_dynamic_no_voice_needed(self, store):
        await store.initialize()
        store.update_dynamic("recent_mood", "happy")
        assert store.get_dynamic("recent_mood") == "happy"

    @pytest.mark.asyncio
    async def test_update_dynamic_same_value(self, store):
        await store.initialize()
        store.update_dynamic("current_room", "general")  # Same as default
        # Should not create audit entry for same value
        log = store.get_audit_log()
        assert not any(e["key"] == "current_room" for e in log)

    @pytest.mark.asyncio
    async def test_set_current_project(self, store):
        await store.initialize()
        store.set_current_project("kitchen")
        assert store.get_dynamic("current_project") == "kitchen"

    @pytest.mark.asyncio
    async def test_set_current_room(self, store):
        await store.initialize()
        store.set_current_room("writers_room")
        assert store.get_dynamic("current_room") == "writers_room"

    @pytest.mark.asyncio
    async def test_set_mood(self, store):
        await store.initialize()
        store.set_mood("excited")
        assert store.get_dynamic("recent_mood") == "excited"

    @pytest.mark.asyncio
    async def test_record_interaction(self, store):
        await store.initialize()
        store.record_interaction()
        assert store.get_dynamic("last_interaction") is not None


# ── Preference Learning ──────────────────────────────────────────────────


class TestPreferenceLearning:
    @pytest.mark.asyncio
    async def test_learn_preference_high_confidence(self, store):
        await store.initialize()
        result = store.learn_preference("response_length", "verbose", confidence=0.9)
        assert result is True
        assert store.get_preference("response_length") == "verbose"

    @pytest.mark.asyncio
    async def test_learn_preference_low_confidence(self, store):
        await store.initialize()
        result = store.learn_preference("response_length", "verbose", confidence=0.5)
        assert result is False
        assert store.get_preference("response_length") == "concise"  # Unchanged

    @pytest.mark.asyncio
    async def test_learn_preference_custom_threshold(self, store):
        await store.initialize()
        result = store.learn_preference(
            "lang", "te", confidence=0.6, min_confidence=0.5
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_learn_preference_same_value(self, store):
        await store.initialize()
        result = store.learn_preference("response_length", "concise", confidence=0.9)
        assert result is False  # Same value, no change

    @pytest.mark.asyncio
    async def test_learn_preference_creates_audit(self, store):
        await store.initialize()
        store.learn_preference("new_pref", "value", confidence=0.9)
        log = store.get_audit_log()
        assert any(e["operation"] == "preference_learned" for e in log)


# ── Relationship Management ──────────────────────────────────────────────


class TestRelationships:
    @pytest.mark.asyncio
    async def test_add_relationship(self, store):
        await store.initialize()
        rel = store.add_relationship("Ravi", "character", project="gusagusalu")
        assert rel.name == "Ravi"
        assert rel.relation == "character"

    @pytest.mark.asyncio
    async def test_get_relationship(self, store):
        await store.initialize()
        store.add_relationship("Ravi", "character")
        r = store.get_relationship("Ravi")
        assert r is not None
        assert r.name == "Ravi"

    @pytest.mark.asyncio
    async def test_get_relationship_case_insensitive(self, store):
        await store.initialize()
        store.add_relationship("Ravi", "character")
        assert store.get_relationship("ravi") is not None
        assert store.get_relationship("RAVI") is not None

    @pytest.mark.asyncio
    async def test_update_relationship(self, store):
        await store.initialize()
        store.add_relationship("Ravi", "character")
        updated = store.update_relationship("Ravi", notes="protagonist")
        assert updated is not None
        assert updated.notes == "protagonist"

    @pytest.mark.asyncio
    async def test_update_nonexistent_relationship(self, store):
        await store.initialize()
        result = store.update_relationship("Nobody", notes="test")
        assert result is None

    @pytest.mark.asyncio
    async def test_remove_relationship(self, store):
        await store.initialize()
        store.add_relationship("Ravi", "character")
        result = store.remove_relationship("Ravi")
        assert result is True
        assert store.get_relationship("Ravi") is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_relationship(self, store):
        await store.initialize()
        result = store.remove_relationship("Nobody")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_relationships(self, store):
        await store.initialize()
        store.add_relationship("Ravi", "character", project="proj1")
        store.add_relationship("Father", "character", project="proj1")
        store.add_relationship("Friend", "friend")

        all_rels = store.list_relationships()
        assert len(all_rels) == 3

    @pytest.mark.asyncio
    async def test_list_relationships_by_relation(self, store):
        await store.initialize()
        store.add_relationship("Ravi", "character")
        store.add_relationship("Friend", "friend")

        chars = store.list_relationships(relation="character")
        assert len(chars) == 1
        assert chars[0].name == "Ravi"

    @pytest.mark.asyncio
    async def test_list_relationships_by_project(self, store):
        await store.initialize()
        store.add_relationship("Ravi", "character", project="proj1")
        store.add_relationship("Other", "character", project="proj2")

        proj1 = store.list_relationships(project="proj1")
        assert len(proj1) == 1


# ── Project Management ────────────────────────────────────────────────────


class TestProjects:
    @pytest.mark.asyncio
    async def test_add_project(self, store):
        await store.initialize()
        p = store.add_project("Gusagusalu", "gusagusalu", description="Film")
        assert p.name == "Gusagusalu"
        assert p.status == "active"

    @pytest.mark.asyncio
    async def test_get_project(self, store):
        await store.initialize()
        store.add_project("Gusagusalu", "gusagusalu")
        p = store.get_project("gusagusalu")
        assert p is not None
        assert p.name == "Gusagusalu"

    @pytest.mark.asyncio
    async def test_get_project_case_insensitive(self, store):
        await store.initialize()
        store.add_project("Test", "test-proj")
        assert store.get_project("TEST-PROJ") is not None

    @pytest.mark.asyncio
    async def test_update_project(self, store):
        await store.initialize()
        store.add_project("Gusagusalu", "gusagusalu")
        updated = store.update_project("gusagusalu", description="Updated")
        assert updated is not None
        assert updated.description == "Updated"

    @pytest.mark.asyncio
    async def test_update_nonexistent_project(self, store):
        await store.initialize()
        result = store.update_project("nonexistent", status="paused")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_project_status(self, store):
        await store.initialize()
        store.add_project("Test", "test")
        result = store.set_project_status("test", "paused")
        assert result.status == "paused"

    @pytest.mark.asyncio
    async def test_list_projects(self, store):
        await store.initialize()
        store.add_project("P1", "p1", status="active")
        store.add_project("P2", "p2", status="paused")
        store.add_project("P3", "p3", status="active")

        all_projs = store.list_projects()
        assert len(all_projs) == 3

    @pytest.mark.asyncio
    async def test_list_projects_by_status(self, store):
        await store.initialize()
        store.add_project("P1", "p1", status="active")
        store.add_project("P2", "p2", status="paused")

        active = store.list_projects(status="active")
        assert len(active) == 1

    @pytest.mark.asyncio
    async def test_get_active_projects(self, store):
        await store.initialize()
        store.add_project("Active", "active-proj", status="active")
        store.add_project("Paused", "paused-proj", status="paused")

        active = store.get_active_projects()
        assert len(active) == 1
        assert active[0].name == "Active"


# ── History & Rollback ────────────────────────────────────────────────────


class TestHistory:
    @pytest.mark.asyncio
    async def test_history_created_on_save(self, store, profile_dir):
        _, history_path = profile_dir
        await store.initialize()
        store.update_dynamic("current_project", "test")

        history_files = list(Path(history_path).glob("*.json"))
        assert len(history_files) >= 1

    @pytest.mark.asyncio
    async def test_list_history(self, store):
        await store.initialize()
        # Make several changes to create history
        store.update_dynamic("current_project", "p1")
        store.update_dynamic("current_project", "p2")

        history = store.list_history()
        assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_rollback(self, store):
        await store.initialize()
        store.update_static("name", "Original", voice_confirmed=True)
        store.update_static("name", "Changed", voice_confirmed=True)

        history = store.list_history()
        assert len(history) >= 1

        # Rollback to first version
        result = store.rollback_to_version(history[-1])
        assert result is True

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_file(self, store):
        await store.initialize()
        result = store.rollback_to_version(Path("/nonexistent/file.json"))
        assert result is False

    @pytest.mark.asyncio
    async def test_history_cleanup(self, store, mock_config, profile_dir):
        _, history_path = profile_dir
        mock_config.max_history_versions = 3
        await store.initialize()

        # Make many changes
        for i in range(10):
            store.update_dynamic("current_project", f"p{i}")

        history = list(Path(history_path).glob("*.json"))
        assert len(history) <= 4  # max_history_versions + 1 buffer


# ── Audit Log ─────────────────────────────────────────────────────────────


class TestAuditLog:
    @pytest.mark.asyncio
    async def test_audit_log_recorded(self, store):
        await store.initialize()
        store.update_dynamic("current_project", "test")
        log = store.get_audit_log()
        assert len(log) > 0

    @pytest.mark.asyncio
    async def test_audit_log_limit(self, store):
        await store.initialize()
        store.update_dynamic("current_project", "test")
        log = store.get_audit_log(limit=1)
        assert len(log) <= 1

    @pytest.mark.asyncio
    async def test_audit_log_truncation(self, store):
        await store.initialize()
        # Add > 1000 entries
        for i in range(1050):
            store._audit("test", f"key_{i}", None, i)
        log = store.get_audit_log(limit=2000)
        assert len(log) <= 1000


# ── Export & Summary ──────────────────────────────────────────────────────


class TestExportSummary:
    @pytest.mark.asyncio
    async def test_get_summary(self, store):
        await store.initialize()
        summary = store.get_summary()
        assert summary["name"] == "Poorna"
        assert summary["address_as"] == "Boss"
        assert "preferences" in summary

    @pytest.mark.asyncio
    async def test_export_for_prompt(self, store):
        await store.initialize()
        prompt = store.export_for_prompt()
        assert "Poorna" in prompt
        assert "Boss" in prompt
        assert "Telugu" in prompt

    @pytest.mark.asyncio
    async def test_export_with_current_project(self, store):
        await store.initialize()
        store.add_project("Gusagusalu", "gusagusalu")
        store.set_current_project("gusagusalu")
        prompt = store.export_for_prompt()
        assert "Gusagusalu" in prompt

    @pytest.mark.asyncio
    async def test_repr(self, store):
        await store.initialize()
        r = repr(store)
        assert "ProfileStore" in r
        assert "v" in r


# ── Edge Cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_get_static_default(self, store):
        await store.initialize()
        val = store.get_static("nonexistent", "default_val")
        assert val == "default_val"

    @pytest.mark.asyncio
    async def test_get_dynamic_default(self, store):
        await store.initialize()
        val = store.get_dynamic("nonexistent", "default_val")
        assert val == "default_val"

    @pytest.mark.asyncio
    async def test_get_preference_default(self, store):
        await store.initialize()
        val = store.get_preference("nonexistent", "default_val")
        assert val == "default_val"

    @pytest.mark.asyncio
    async def test_add_project_with_deadline(self, store):
        await store.initialize()
        deadline = datetime(2025, 6, 1)
        p = store.add_project("P", "p", deadline=deadline)
        assert p.deadline == deadline

    @pytest.mark.asyncio
    async def test_overwrite_relationship(self, store):
        await store.initialize()
        store.add_relationship("Ravi", "character")
        store.add_relationship("Ravi", "protagonist")  # Overwrite
        r = store.get_relationship("Ravi")
        assert r.relation == "protagonist"

    @pytest.mark.asyncio
    async def test_overwrite_project(self, store):
        await store.initialize()
        store.add_project("P", "p", status="active")
        store.add_project("P", "p", status="paused")  # Overwrite
        p = store.get_project("p")
        assert p.status == "paused"

    def test_repr_no_profile(self, store):
        r = repr(store)
        assert "v0" in r
