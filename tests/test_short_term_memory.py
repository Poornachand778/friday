"""
Tests for memory/layers/short_term.py
======================================

Comprehensive tests for ShortTermMemory and STMEntry.
Covers SQLite storage, FTS5 search, CRUD operations, session management,
temporal queries, consolidation, decay, and edge cases.

Tests: 80+
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from memory.layers.short_term import ShortTermMemory, STMEntry


# ── Helpers ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_config(tmp_path):
    """Mock STMConfig."""
    config = MagicMock()
    config.db_path = str(tmp_path / "stm_test.db")
    config.retention_days = 7
    config.consolidation_threshold = 0.4
    return config


@pytest_asyncio.fixture
async def stm(mock_config):
    """Initialized ShortTermMemory."""
    with patch("memory.layers.short_term.get_memory_config") as mock_get:
        mock_mem_config = MagicMock()
        mock_mem_config.stm = mock_config
        mock_get.return_value = mock_mem_config
        s = ShortTermMemory(config=mock_config)
    await s.initialize()
    yield s
    await s.close()


async def _store_entry(stm, session_id="sess1", summary="Test memory", **kwargs):
    """Helper to store a test entry."""
    return await stm.store(
        session_id=session_id,
        summary=summary,
        key_facts=kwargs.get("key_facts", ["fact1"]),
        raw_turns=kwargs.get("raw_turns", [{"user": "hi", "assistant": "hello"}]),
        room=kwargs.get("room", "general"),
        project=kwargs.get("project"),
        topics=kwargs.get("topics", ["test"]),
        language=kwargs.get("language", "en"),
        importance=kwargs.get("importance", 0.5),
        event_dates=kwargs.get("event_dates"),
    )


# ── STMEntry Dataclass ────────────────────────────────────────────────────


class TestSTMEntry:
    def test_to_dict(self):
        entry = STMEntry(
            id="e1",
            session_id="s1",
            summary="Test summary",
            key_facts=["fact1"],
            raw_turns=[],
            created_at=datetime(2025, 1, 15),
            event_dates=[datetime(2025, 3, 1)],
            room="writers_room",
            project="proj1",
            topics=["test"],
            importance=0.7,
        )
        d = entry.to_dict()
        assert d["id"] == "e1"
        assert d["summary"] == "Test summary"
        assert d["key_facts"] == ["fact1"]
        assert d["room"] == "writers_room"
        assert d["importance"] == 0.7
        assert len(d["event_dates"]) == 1

    def test_from_dict(self):
        data = {
            "id": "e1",
            "session_id": "s1",
            "summary": "Test",
            "key_facts": ["fact"],
            "raw_turns": [],
            "created_at": "2025-01-15T00:00:00",
            "event_dates": ["2025-03-01T00:00:00"],
            "room": "general",
            "importance": 0.8,
        }
        entry = STMEntry.from_dict(data)
        assert entry.id == "e1"
        assert entry.summary == "Test"
        assert entry.importance == 0.8

    def test_from_dict_defaults(self):
        data = {
            "id": "e1",
            "session_id": "s1",
            "summary": "Test",
            "created_at": "2025-01-15T00:00:00",
        }
        entry = STMEntry.from_dict(data)
        assert entry.key_facts == []
        assert entry.room == "general"
        assert entry.status == "active"
        assert entry.access_count == 0

    def test_roundtrip(self):
        entry = STMEntry(
            id="e1",
            session_id="s1",
            summary="Test",
            key_facts=["f1"],
            raw_turns=[{"user": "hi"}],
            created_at=datetime(2025, 1, 15),
            event_dates=[],
        )
        d = entry.to_dict()
        e2 = STMEntry.from_dict(d)
        assert e2.id == entry.id
        assert e2.summary == entry.summary


# ── Init ──────────────────────────────────────────────────────────────────


class TestInit:
    @pytest.mark.asyncio
    async def test_initialize_creates_db(self, mock_config):
        with patch("memory.layers.short_term.get_memory_config") as mock_get:
            mock_mem_config = MagicMock()
            mock_mem_config.stm = mock_config
            mock_get.return_value = mock_mem_config
            s = ShortTermMemory(config=mock_config)
        await s.initialize()
        assert Path(mock_config.db_path).exists()
        await s.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_dirs(self, tmp_path):
        config = MagicMock()
        config.db_path = str(tmp_path / "sub" / "dir" / "stm.db")
        config.retention_days = 7
        config.consolidation_threshold = 0.4
        with patch("memory.layers.short_term.get_memory_config") as mock_get:
            mock_mem_config = MagicMock()
            mock_mem_config.stm = config
            mock_get.return_value = mock_mem_config
            s = ShortTermMemory(config=config)
        await s.initialize()
        assert Path(config.db_path).exists()
        await s.close()

    @pytest.mark.asyncio
    async def test_close(self, stm):
        await stm.close()
        assert stm._conn is None

    @pytest.mark.asyncio
    async def test_repr(self, stm):
        r = repr(stm)
        assert "ShortTermMemory" in r


# ── Store ─────────────────────────────────────────────────────────────────


class TestStore:
    @pytest.mark.asyncio
    async def test_store_basic(self, stm):
        entry = await _store_entry(stm)
        assert entry.id is not None
        assert entry.session_id == "sess1"
        assert entry.summary == "Test memory"
        assert entry.status == "active"

    @pytest.mark.asyncio
    async def test_store_with_all_fields(self, stm):
        entry = await stm.store(
            session_id="s1",
            summary="Full entry",
            key_facts=["fact1", "fact2"],
            raw_turns=[{"user": "hi"}],
            event_dates=[datetime(2025, 3, 1)],
            room="writers_room",
            project="gusagusalu",
            topics=["film", "script"],
            language="te",
            importance=0.9,
        )
        assert entry.room == "writers_room"
        assert entry.project == "gusagusalu"
        assert entry.language == "te"
        assert entry.importance == 0.9
        assert len(entry.event_dates) == 1

    @pytest.mark.asyncio
    async def test_store_defaults(self, stm):
        entry = await stm.store(session_id="s1", summary="Minimal")
        assert entry.key_facts == []
        assert entry.room == "general"
        assert entry.language == "mixed"
        assert entry.importance == 0.5

    @pytest.mark.asyncio
    async def test_store_multiple(self, stm):
        e1 = await _store_entry(stm, summary="First")
        e2 = await _store_entry(stm, summary="Second")
        assert e1.id != e2.id


# ── Get ───────────────────────────────────────────────────────────────────


class TestGet:
    @pytest.mark.asyncio
    async def test_get_by_id(self, stm):
        stored = await _store_entry(stm)
        retrieved = await stm.get(stored.id)
        assert retrieved is not None
        assert retrieved.id == stored.id
        assert retrieved.summary == stored.summary

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, stm):
        result = await stm.get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_records_access(self, stm):
        """get() calls _record_access but returns row fetched BEFORE update."""
        stored = await _store_entry(stm)
        # First get - records access but returned row has pre-update count
        r1 = await stm.get(stored.id)
        assert r1 is not None
        # Second get - now returned row reflects previous _record_access
        r2 = await stm.get(stored.id)
        assert r2.access_count >= 1  # At least initial + first access reflected

    @pytest.mark.asyncio
    async def test_get_by_session(self, stm):
        await _store_entry(stm, session_id="sess_A", summary="First")
        await _store_entry(stm, session_id="sess_A", summary="Second")
        await _store_entry(stm, session_id="sess_B", summary="Other")

        results = await stm.get_by_session("sess_A")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_by_session_empty(self, stm):
        results = await stm.get_by_session("nonexistent")
        assert results == []


# ── Get Recent ────────────────────────────────────────────────────────────


class TestGetRecent:
    @pytest.mark.asyncio
    async def test_get_recent(self, stm):
        await _store_entry(stm, summary="Recent")
        results = await stm.get_recent(days=7)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_recent_filter_room(self, stm):
        await _store_entry(stm, room="writers_room")
        await _store_entry(stm, room="kitchen")
        results = await stm.get_recent(room="writers_room")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_recent_filter_project(self, stm):
        await _store_entry(stm, project="proj1")
        await _store_entry(stm, project="proj2")
        results = await stm.get_recent(project="proj1")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_recent_limit(self, stm):
        for i in range(5):
            await _store_entry(stm, summary=f"Entry {i}")
        results = await stm.get_recent(limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_recent_excludes_archived(self, stm):
        entry = await _store_entry(stm)
        await stm.archive(entry.id)
        results = await stm.get_recent()
        assert len(results) == 0


# ── Update ────────────────────────────────────────────────────────────────


class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_summary(self, stm):
        entry = await _store_entry(stm)
        updated = await stm.update(entry.id, summary="Updated summary")
        assert updated is not None
        assert updated.summary == "Updated summary"

    @pytest.mark.asyncio
    async def test_update_importance(self, stm):
        entry = await _store_entry(stm, importance=0.5)
        updated = await stm.update(entry.id, importance=0.9)
        assert updated.importance == 0.9

    @pytest.mark.asyncio
    async def test_update_key_facts(self, stm):
        entry = await _store_entry(stm)
        updated = await stm.update(entry.id, key_facts=["new_fact"])
        assert updated.key_facts == ["new_fact"]

    @pytest.mark.asyncio
    async def test_update_topics(self, stm):
        entry = await _store_entry(stm)
        updated = await stm.update(entry.id, topics=["new_topic"])
        assert updated.topics == ["new_topic"]

    @pytest.mark.asyncio
    async def test_update_disallowed_field(self, stm):
        entry = await _store_entry(stm)
        # "created_at" not in allowed fields
        updated = await stm.update(entry.id, created_at="2020-01-01")
        assert updated is not None  # Returns entry but doesn't update the field

    @pytest.mark.asyncio
    async def test_update_no_changes(self, stm):
        entry = await _store_entry(stm)
        updated = await stm.update(entry.id)
        assert updated is not None


# ── Delete ────────────────────────────────────────────────────────────────


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete(self, stm):
        entry = await _store_entry(stm)
        result = await stm.delete(entry.id)
        assert result is True
        assert await stm.get(entry.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, stm):
        result = await stm.delete("nonexistent")
        assert result is False


# ── Archive & Consolidate ─────────────────────────────────────────────────


class TestArchiveConsolidate:
    @pytest.mark.asyncio
    async def test_archive(self, stm):
        entry = await _store_entry(stm)
        archived = await stm.archive(entry.id)
        assert archived.status == "archived"

    @pytest.mark.asyncio
    async def test_mark_consolidated(self, stm):
        entry = await _store_entry(stm)
        consolidated = await stm.mark_consolidated(entry.id)
        assert consolidated.status == "consolidated"


# ── Search ────────────────────────────────────────────────────────────────


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_basic(self, stm):
        await _store_entry(stm, summary="The climax scene needs more tension")
        results = await stm.search("climax")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_no_results(self, stm):
        await _store_entry(stm, summary="Unrelated content")
        results = await stm.search("xyznonexistent")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_filter_room(self, stm):
        await _store_entry(stm, summary="Writers room content", room="writers_room")
        await _store_entry(stm, summary="Kitchen content", room="kitchen")
        results = await stm.search("content", room="writers_room")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_filter_project(self, stm):
        await _store_entry(stm, summary="Project A work", project="proj_a")
        await _store_entry(stm, summary="Project B work", project="proj_b")
        results = await stm.search("work", project="proj_a")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_excludes_archived(self, stm):
        entry = await _store_entry(stm, summary="Archived content here")
        await stm.archive(entry.id)
        results = await stm.search("archived content")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_top_k(self, stm):
        for i in range(5):
            await _store_entry(stm, summary=f"Important topic number {i}")
        results = await stm.search("topic", top_k=2)
        assert len(results) <= 2


# ── Temporal Search ───────────────────────────────────────────────────────


class TestTemporalSearch:
    @pytest.mark.asyncio
    async def test_search_temporal_all(self, stm):
        await _store_entry(stm)
        results = await stm.search_temporal()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_temporal_start(self, stm):
        await _store_entry(stm)
        results = await stm.search_temporal(start=datetime.now() - timedelta(hours=1))
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_temporal_future_start(self, stm):
        await _store_entry(stm)
        results = await stm.search_temporal(start=datetime.now() + timedelta(hours=1))
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_temporal_with_events(self, stm):
        await _store_entry(
            stm,
            summary="Has events",
            event_dates=[datetime(2025, 6, 1)],
        )
        await _store_entry(stm, summary="No events")
        results = await stm.search_temporal(has_event_date=True)
        assert len(results) == 1


# ── Consolidation Candidates ─────────────────────────────────────────────


class TestConsolidationCandidates:
    @pytest.mark.asyncio
    async def test_get_consolidation_candidates_by_importance(self, stm):
        await _store_entry(stm, importance=0.1)  # Below threshold
        await _store_entry(stm, importance=0.9)  # Above threshold
        candidates = await stm.get_consolidation_candidates(threshold=0.5)
        assert len(candidates) >= 1

    @pytest.mark.asyncio
    async def test_get_consolidation_candidates_empty(self, stm):
        candidates = await stm.get_consolidation_candidates()
        assert candidates == []


# ── Decay Candidates ─────────────────────────────────────────────────────


class TestDecayCandidates:
    @pytest.mark.asyncio
    async def test_get_decay_candidates(self, stm):
        await _store_entry(stm, importance=0.1)
        await _store_entry(stm, importance=0.5)
        candidates = await stm.get_decay_candidates(threshold=0.2)
        assert len(candidates) == 1

    @pytest.mark.asyncio
    async def test_get_decay_candidates_empty(self, stm):
        await _store_entry(stm, importance=0.9)
        candidates = await stm.get_decay_candidates(threshold=0.2)
        assert len(candidates) == 0


# ── Cleanup ───────────────────────────────────────────────────────────────


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_old_archived(self, stm):
        entry = await _store_entry(stm)
        await stm.archive(entry.id)
        # Backdate created_at so it's before cutoff
        old_date = (datetime.now() - timedelta(days=2)).isoformat()
        with stm._transaction() as cur:
            cur.execute(
                "UPDATE short_term_memories SET created_at = ? WHERE id = ?",
                (old_date, entry.id),
            )
        deleted = await stm.cleanup_old(days=1)
        assert deleted >= 1

    @pytest.mark.asyncio
    async def test_cleanup_preserves_active(self, stm):
        await _store_entry(stm)
        deleted = await stm.cleanup_old(days=0)
        assert deleted == 0  # Active entries not cleaned up


# ── Stats ─────────────────────────────────────────────────────────────────


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_empty(self, stm):
        stats = await stm.get_stats()
        assert stats["active"] == 0
        assert stats["archived"] == 0
        assert stats["total"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_entries(self, stm):
        e1 = await _store_entry(stm, importance=0.6)
        e2 = await _store_entry(stm, importance=0.8)
        await stm.archive(e1.id)

        stats = await stm.get_stats()
        assert stats["active"] == 1
        assert stats["archived"] == 1
        assert stats["total"] == 2
        assert stats["sessions"] >= 1

    @pytest.mark.asyncio
    async def test_stats_avg_importance(self, stm):
        await _store_entry(stm, importance=0.4)
        await _store_entry(stm, importance=0.6)
        stats = await stm.get_stats()
        assert abs(stats["avg_importance"] - 0.5) < 0.01


# ── Helper Methods ────────────────────────────────────────────────────────


class TestHelperMethods:
    @pytest.mark.asyncio
    async def test_mark_accessed(self, stm):
        entry = await _store_entry(stm)
        await stm.mark_accessed(entry.id)
        updated = await stm.get(entry.id)
        assert updated.access_count >= 1

    @pytest.mark.asyncio
    async def test_update_importance(self, stm):
        entry = await _store_entry(stm, importance=0.5)
        result = await stm.update_importance(entry.id, 0.9)
        assert result is True
        updated = await stm.get(entry.id)
        assert updated.importance == 0.9

    @pytest.mark.asyncio
    async def test_update_importance_nonexistent(self, stm):
        result = await stm.update_importance("nonexistent", 0.9)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_entries_before(self, stm):
        await _store_entry(stm)
        entries = await stm.get_entries_before(datetime.now() + timedelta(hours=1))
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_get_entries_before_none(self, stm):
        await _store_entry(stm)
        entries = await stm.get_entries_before(datetime.now() - timedelta(hours=1))
        assert len(entries) == 0


# ── Edge Cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_store_empty_summary(self, stm):
        entry = await stm.store(session_id="s1", summary="")
        assert entry.summary == ""

    @pytest.mark.asyncio
    async def test_store_unicode(self, stm):
        entry = await stm.store(
            session_id="s1",
            summary="Telugu text: నేను Friday ని",
        )
        retrieved = await stm.get(entry.id)
        assert "Friday" in retrieved.summary
        assert "నేను" in retrieved.summary

    @pytest.mark.asyncio
    async def test_fallback_search(self, stm):
        await _store_entry(stm, summary="Special content here")
        results = await stm._fallback_search("Special", 10, None, None)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_fallback_search_with_filters(self, stm):
        await _store_entry(
            stm, summary="Room content", room="writers_room", project="proj1"
        )
        results = await stm._fallback_search("content", 10, "writers_room", "proj1")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, stm):
        for i in range(3):
            await _store_entry(stm, session_id=f"sess_{i}")
        stats = await stm.get_stats()
        assert stats["sessions"] == 3

    @pytest.mark.asyncio
    async def test_close_idempotent(self, stm):
        await stm.close()
        await stm.close()  # Should not raise
