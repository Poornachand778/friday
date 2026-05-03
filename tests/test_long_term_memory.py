"""
Tests for memory/layers/long_term.py
=====================================

Comprehensive tests for LongTermMemory, LTMEntry, EmbeddingModel, and MemoryType.
Covers SQLite storage, vector search, FTS fallback, embedding serialization,
CRUD operations, maintenance, Qdrant sync, and edge cases.

Tests: 90+
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

from memory.layers.long_term import (
    EmbeddingModel,
    LongTermMemory,
    LTMEntry,
    MemoryType,
)


# ── Helpers ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_config(tmp_path):
    """Mock LTMConfig."""
    config = MagicMock()
    config.sqlite_path = str(tmp_path / "ltm_test.db")
    config.embedding_model = "test-model"
    return config


def _make_mock_embedder(available=True, dim=4):
    """Create mock EmbeddingModel."""
    embedder = MagicMock(spec=EmbeddingModel)
    embedder.is_available = available
    if available:
        embedder.encode.side_effect = lambda text: [0.1, 0.2, 0.3, 0.4]
        embedder.encode_batch.side_effect = lambda texts: [
            [0.1, 0.2, 0.3, 0.4] for _ in texts
        ]
    else:
        embedder.encode.return_value = None
        embedder.encode_batch.return_value = [None]
    return embedder


@pytest_asyncio.fixture
async def ltm(mock_config):
    """Initialized LongTermMemory with mocked embedder and no Qdrant."""
    with patch("memory.layers.long_term.get_memory_config") as mock_get:
        mock_mem_config = MagicMock()
        mock_mem_config.ltm = mock_config
        mock_get.return_value = mock_mem_config
        mem = LongTermMemory(config=mock_config)

    mock_emb = _make_mock_embedder()
    with patch("memory.layers.long_term.EmbeddingModel", return_value=mock_emb):
        with patch.dict(
            "sys.modules",
            {
                "db.vector_store": MagicMock(
                    get_vector_store=AsyncMock(return_value=None)
                )
            },
        ):
            await mem.initialize()

    # Ensure embedder is our mock (initialize sets it from EmbeddingModel())
    mem._embedder = mock_emb

    yield mem
    await mem.close()


@pytest_asyncio.fixture
async def ltm_no_embeddings(mock_config):
    """LTM with embeddings unavailable (keyword fallback)."""
    with patch("memory.layers.long_term.get_memory_config") as mock_get:
        mock_mem_config = MagicMock()
        mock_mem_config.ltm = mock_config
        mock_get.return_value = mock_mem_config
        mem = LongTermMemory(config=mock_config)

    with patch("memory.layers.long_term.EmbeddingModel") as MockEmb:
        mock_emb = _make_mock_embedder(available=False)
        MockEmb.return_value = mock_emb
        with patch.dict(
            "sys.modules",
            {
                "db.vector_store": MagicMock(
                    get_vector_store=AsyncMock(return_value=None)
                )
            },
        ):
            await mem.initialize()
            mem._embedder = mock_emb

    yield mem
    await mem.close()


async def _store_entry(ltm, content="Test memory", **kwargs):
    """Helper to store a test entry."""
    return await ltm.store(
        content=content,
        memory_type=kwargs.get("memory_type", MemoryType.FACT),
        source_summary=kwargs.get("source_summary", "test source"),
        domain=kwargs.get("domain", "general"),
        event_date=kwargs.get("event_date"),
        valid_until=kwargs.get("valid_until"),
        project=kwargs.get("project"),
        entities=kwargs.get("entities"),
        confidence=kwargs.get("confidence", 0.8),
        trust_level=kwargs.get("trust_level", 3),
        importance=kwargs.get("importance", 0.5),
        language=kwargs.get("language", "en"),
        telugu_keywords=kwargs.get("telugu_keywords"),
        source_stm_ids=kwargs.get("source_stm_ids"),
    )


# ── MemoryType ────────────────────────────────────────────────────────────


class TestMemoryType:
    def test_values(self):
        assert MemoryType.FACT == "fact"
        assert MemoryType.PREFERENCE == "preference"
        assert MemoryType.EVENT == "event"
        assert MemoryType.PATTERN == "pattern"
        assert MemoryType.DECISION == "decision"
        assert MemoryType.RELATIONSHIP == "relationship"

    def test_from_value(self):
        assert MemoryType("fact") == MemoryType.FACT

    def test_all_types(self):
        assert len(MemoryType) == 6


# ── LTMEntry ──────────────────────────────────────────────────────────────


class TestLTMEntry:
    def test_to_dict(self):
        entry = LTMEntry(
            id="e1",
            content="Test fact",
            source_summary="source",
            memory_type=MemoryType.FACT,
            domain="personal",
            created_at=datetime(2025, 1, 15),
            project="proj1",
            entities=["Ravi"],
            confidence=0.9,
            trust_level=4,
            importance=0.7,
        )
        d = entry.to_dict()
        assert d["id"] == "e1"
        assert d["content"] == "Test fact"
        assert d["memory_type"] == "fact"
        assert d["domain"] == "personal"
        assert d["project"] == "proj1"
        assert d["entities"] == ["Ravi"]
        assert d["confidence"] == 0.9
        assert d["trust_level"] == 4
        assert d["importance"] == 0.7

    def test_to_dict_with_dates(self):
        entry = LTMEntry(
            id="e2",
            content="Event",
            source_summary="src",
            memory_type=MemoryType.EVENT,
            event_date=datetime(2025, 3, 1),
            valid_until=datetime(2025, 6, 1),
        )
        d = entry.to_dict()
        assert d["event_date"] == "2025-03-01T00:00:00"
        assert d["valid_until"] == "2025-06-01T00:00:00"

    def test_to_dict_none_dates(self):
        entry = LTMEntry(
            id="e3",
            content="Basic",
            source_summary="src",
            memory_type=MemoryType.FACT,
        )
        d = entry.to_dict()
        assert d["event_date"] is None
        assert d["valid_until"] is None
        assert d["last_accessed"] is None

    def test_from_dict(self):
        data = {
            "id": "e1",
            "content": "Test",
            "source_summary": "src",
            "memory_type": "preference",
            "domain": "personal",
            "created_at": "2025-01-15T00:00:00",
            "confidence": 0.9,
            "trust_level": 4,
            "importance": 0.7,
            "language": "te",
        }
        entry = LTMEntry.from_dict(data)
        assert entry.id == "e1"
        assert entry.memory_type == MemoryType.PREFERENCE
        assert entry.domain == "personal"
        assert entry.language == "te"

    def test_from_dict_defaults(self):
        data = {
            "id": "e2",
            "content": "Min",
            "memory_type": "fact",
            "created_at": "2025-01-01T00:00:00",
        }
        entry = LTMEntry.from_dict(data)
        assert entry.source_summary == ""
        assert entry.domain == "general"
        assert entry.confidence == 0.8
        assert entry.trust_level == 3
        assert entry.importance == 0.5
        assert entry.language == "en"

    def test_from_dict_with_dates(self):
        data = {
            "id": "e3",
            "content": "Event",
            "memory_type": "event",
            "created_at": "2025-01-15T00:00:00",
            "event_date": "2025-03-01T00:00:00",
            "valid_until": "2025-06-01T00:00:00",
        }
        entry = LTMEntry.from_dict(data)
        assert entry.event_date == datetime(2025, 3, 1)
        assert entry.valid_until == datetime(2025, 6, 1)

    def test_roundtrip(self):
        original = LTMEntry(
            id="rt1",
            content="Roundtrip test",
            source_summary="source",
            memory_type=MemoryType.DECISION,
            domain="film",
            created_at=datetime(2025, 2, 1),
            entities=["char1", "char2"],
            telugu_keywords=["paata"],
            source_stm_ids=["stm1", "stm2"],
        )
        restored = LTMEntry.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.memory_type == original.memory_type
        assert restored.entities == original.entities
        assert restored.telugu_keywords == original.telugu_keywords
        assert restored.source_stm_ids == original.source_stm_ids


# ── EmbeddingModel ────────────────────────────────────────────────────────


class TestEmbeddingModel:
    def test_init(self):
        em = EmbeddingModel("test-model")
        assert em.model_name == "test-model"
        assert em._model is None

    def test_encode_no_sentence_transformers(self):
        em = EmbeddingModel()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            em._model = None  # Reset
            with patch(
                "memory.layers.long_term.EmbeddingModel._load_model",
                side_effect=lambda: setattr(em, "_model", False),
            ):
                result = em.encode("test")
                assert result is None

    def test_encode_model_unavailable(self):
        em = EmbeddingModel()
        em._model = False
        result = em.encode("test")
        assert result is None

    def test_encode_batch_model_unavailable(self):
        em = EmbeddingModel()
        em._model = False
        result = em.encode_batch(["a", "b", "c"])
        assert result == [None, None, None]

    def test_is_available_false(self):
        em = EmbeddingModel()
        em._model = False
        assert em.is_available is False

    def test_is_available_true(self):
        em = EmbeddingModel()
        em._model = MagicMock()  # Pretend loaded
        assert em.is_available is True

    def test_lazy_load(self):
        em = EmbeddingModel()
        assert em._model is None
        em._model = False  # Simulate failed load
        assert em.is_available is False


# ── Init & Close ──────────────────────────────────────────────────────────


class TestInit:
    @pytest.mark.asyncio
    async def test_initialize(self, ltm):
        assert ltm._conn is not None
        assert ltm._embedder is not None

    @pytest.mark.asyncio
    async def test_tables_created(self, ltm):
        with ltm._transaction() as cur:
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='long_term_memories'"
            )
            assert cur.fetchone() is not None

    @pytest.mark.asyncio
    async def test_fts_table_created(self, ltm):
        with ltm._transaction() as cur:
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ltm_fts'"
            )
            assert cur.fetchone() is not None

    @pytest.mark.asyncio
    async def test_close(self, ltm):
        await ltm.close()
        assert ltm._conn is None

    @pytest.mark.asyncio
    async def test_repr(self, ltm):
        r = repr(ltm)
        assert "LongTermMemory" in r

    @pytest.mark.asyncio
    async def test_double_init_no_error(self, mock_config):
        """Re-initialization is safe."""
        with patch("memory.layers.long_term.get_memory_config") as mock_get:
            mock_mem_config = MagicMock()
            mock_mem_config.ltm = mock_config
            mock_get.return_value = mock_mem_config
            mem = LongTermMemory(config=mock_config)

        with patch("memory.layers.long_term.EmbeddingModel") as MockEmb:
            MockEmb.return_value = _make_mock_embedder()
            with patch.dict(
                "sys.modules",
                {
                    "db.vector_store": MagicMock(
                        get_vector_store=AsyncMock(return_value=None)
                    )
                },
            ):
                await mem.initialize()
                # Tables already exist - should not raise
                mem._create_tables()
        await mem.close()


# ── Store ─────────────────────────────────────────────────────────────────


class TestStore:
    @pytest.mark.asyncio
    async def test_store_basic(self, ltm):
        entry = await _store_entry(ltm)
        assert entry.id is not None
        assert entry.content == "Test memory"
        assert entry.memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_store_with_embedding(self, ltm):
        entry = await _store_entry(ltm)
        assert entry.embedding == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_store_all_types(self, ltm):
        for mt in MemoryType:
            entry = await _store_entry(ltm, content=f"Type {mt.value}", memory_type=mt)
            assert entry.memory_type == mt

    @pytest.mark.asyncio
    async def test_store_with_entities(self, ltm):
        entry = await _store_entry(ltm, entities=["Ravi", "Priya"])
        assert entry.entities == ["Ravi", "Priya"]

    @pytest.mark.asyncio
    async def test_store_with_telugu_keywords(self, ltm):
        entry = await _store_entry(ltm, telugu_keywords=["paata", "cinemaa"])
        assert entry.telugu_keywords == ["paata", "cinemaa"]

    @pytest.mark.asyncio
    async def test_store_with_event_date(self, ltm):
        dt = datetime(2025, 6, 15)
        entry = await _store_entry(ltm, event_date=dt)
        assert entry.event_date == dt

    @pytest.mark.asyncio
    async def test_store_with_valid_until(self, ltm):
        dt = datetime(2025, 12, 31)
        entry = await _store_entry(ltm, valid_until=dt)
        assert entry.valid_until == dt

    @pytest.mark.asyncio
    async def test_store_with_project(self, ltm):
        entry = await _store_entry(ltm, project="gusagusalu")
        assert entry.project == "gusagusalu"

    @pytest.mark.asyncio
    async def test_store_with_source_stm_ids(self, ltm):
        entry = await _store_entry(ltm, source_stm_ids=["stm1", "stm2"])
        assert entry.source_stm_ids == ["stm1", "stm2"]

    @pytest.mark.asyncio
    async def test_store_no_embedding_when_unavailable(self, ltm_no_embeddings):
        entry = await _store_entry(ltm_no_embeddings)
        assert entry.embedding is None

    @pytest.mark.asyncio
    async def test_store_custom_confidence(self, ltm):
        entry = await _store_entry(ltm, confidence=0.95)
        assert entry.confidence == 0.95

    @pytest.mark.asyncio
    async def test_store_custom_importance(self, ltm):
        entry = await _store_entry(ltm, importance=0.9)
        assert entry.importance == 0.9


# ── Get ───────────────────────────────────────────────────────────────────


class TestGet:
    @pytest.mark.asyncio
    async def test_get(self, ltm):
        stored = await _store_entry(ltm)
        retrieved = await ltm.get(stored.id)
        assert retrieved is not None
        assert retrieved.content == "Test memory"
        assert retrieved.memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, ltm):
        assert await ltm.get("nonexistent-id") is None

    @pytest.mark.asyncio
    async def test_get_preserves_entities(self, ltm):
        stored = await _store_entry(ltm, entities=["A", "B"])
        retrieved = await ltm.get(stored.id)
        assert retrieved.entities == ["A", "B"]

    @pytest.mark.asyncio
    async def test_get_preserves_embedding(self, ltm):
        stored = await _store_entry(ltm)
        retrieved = await ltm.get(stored.id)
        assert retrieved.embedding is not None
        assert len(retrieved.embedding) == 4

    @pytest.mark.asyncio
    async def test_get_records_access(self, ltm):
        stored = await _store_entry(ltm)
        # get() fetches row BEFORE _record_access, so first call returns count=0
        await ltm.get(stored.id)
        r2 = await ltm.get(stored.id)
        # Second get returns row with at least 1 access
        assert r2.access_count >= 1


# ── Update ────────────────────────────────────────────────────────────────


class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_content(self, ltm):
        stored = await _store_entry(ltm)
        updated = await ltm.update(stored.id, content="Updated content")
        assert updated.content == "Updated content"

    @pytest.mark.asyncio
    async def test_update_importance(self, ltm):
        stored = await _store_entry(ltm, importance=0.3)
        updated = await ltm.update(stored.id, importance=0.8)
        assert updated.importance == 0.8

    @pytest.mark.asyncio
    async def test_update_confidence(self, ltm):
        stored = await _store_entry(ltm, confidence=0.5)
        updated = await ltm.update(stored.id, confidence=0.95)
        assert updated.confidence == 0.95

    @pytest.mark.asyncio
    async def test_update_trust_level(self, ltm):
        stored = await _store_entry(ltm, trust_level=2)
        updated = await ltm.update(stored.id, trust_level=5)
        assert updated.trust_level == 5

    @pytest.mark.asyncio
    async def test_update_entities(self, ltm):
        stored = await _store_entry(ltm, entities=["old"])
        updated = await ltm.update(stored.id, entities=["new1", "new2"])
        assert updated.entities == ["new1", "new2"]

    @pytest.mark.asyncio
    async def test_update_related_memories(self, ltm):
        stored = await _store_entry(ltm)
        updated = await ltm.update(stored.id, related_memories=["mem1", "mem2"])
        assert updated.related_memories == ["mem1", "mem2"]

    @pytest.mark.asyncio
    async def test_update_telugu_keywords(self, ltm):
        stored = await _store_entry(ltm)
        updated = await ltm.update(stored.id, telugu_keywords=["paata"])
        assert updated.telugu_keywords == ["paata"]

    @pytest.mark.asyncio
    async def test_update_project(self, ltm):
        stored = await _store_entry(ltm)
        updated = await ltm.update(stored.id, project="new_project")
        assert updated.project == "new_project"

    @pytest.mark.asyncio
    async def test_update_event_date(self, ltm):
        stored = await _store_entry(ltm)
        new_date = datetime(2025, 7, 1)
        updated = await ltm.update(stored.id, event_date=new_date)
        assert updated.event_date == new_date

    @pytest.mark.asyncio
    async def test_update_valid_until(self, ltm):
        stored = await _store_entry(ltm)
        dt = datetime(2025, 12, 31)
        updated = await ltm.update(stored.id, valid_until=dt)
        assert updated.valid_until == dt

    @pytest.mark.asyncio
    async def test_update_disallowed_field_ignored(self, ltm):
        stored = await _store_entry(ltm)
        # "id" is not in allowed set
        updated = await ltm.update(stored.id, id="new_id")
        assert updated.id == stored.id  # unchanged

    @pytest.mark.asyncio
    async def test_update_no_changes(self, ltm):
        stored = await _store_entry(ltm)
        # No allowed kwargs → returns get()
        updated = await ltm.update(stored.id, unknown_field="value")
        assert updated is not None

    @pytest.mark.asyncio
    async def test_update_regenerates_embedding(self, ltm):
        stored = await _store_entry(ltm)
        ltm._embedder.encode.reset_mock()
        await ltm.update(stored.id, content="New content with new embedding")
        ltm._embedder.encode.assert_called_with("New content with new embedding")


# ── Delete ────────────────────────────────────────────────────────────────


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete(self, ltm):
        stored = await _store_entry(ltm)
        result = await ltm.delete(stored.id)
        assert result is True
        assert await ltm.get(stored.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, ltm):
        result = await ltm.delete("nonexistent")
        assert result is False


# ── Boost Importance ──────────────────────────────────────────────────────


class TestBoostImportance:
    @pytest.mark.asyncio
    async def test_boost(self, ltm):
        stored = await _store_entry(ltm, importance=0.5)
        boosted = await ltm.boost_importance(stored.id, boost=0.2)
        assert boosted.importance == pytest.approx(0.7, abs=0.01)

    @pytest.mark.asyncio
    async def test_boost_caps_at_1(self, ltm):
        stored = await _store_entry(ltm, importance=0.95)
        boosted = await ltm.boost_importance(stored.id, boost=0.2)
        assert boosted.importance == 1.0

    @pytest.mark.asyncio
    async def test_boost_default(self, ltm):
        stored = await _store_entry(ltm, importance=0.5)
        boosted = await ltm.boost_importance(stored.id)  # default boost=0.1
        assert boosted.importance == pytest.approx(0.6, abs=0.01)

    @pytest.mark.asyncio
    async def test_boost_nonexistent(self, ltm):
        result = await ltm.boost_importance("nonexistent")
        assert result is None


# ── Search (Vector) ───────────────────────────────────────────────────────


class TestVectorSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, ltm):
        await _store_entry(ltm, content="Boss prefers morning meetings")
        await _store_entry(ltm, content="Script deadline is March")
        results = await ltm.search("morning meetings")
        assert len(results) >= 1
        # Results are (LTMEntry, score) tuples
        entry, score = results[0]
        assert isinstance(entry, LTMEntry)
        assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_search_empty_db(self, ltm):
        results = await ltm.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_filter_by_type(self, ltm):
        await _store_entry(ltm, content="Fact", memory_type=MemoryType.FACT)
        await _store_entry(ltm, content="Pref", memory_type=MemoryType.PREFERENCE)
        results = await ltm.search("test", memory_type=MemoryType.FACT)
        for entry, _ in results:
            assert entry.memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_search_filter_by_project(self, ltm):
        await _store_entry(ltm, content="Project A", project="proj_a")
        await _store_entry(ltm, content="Project B", project="proj_b")
        results = await ltm.search("test", project="proj_a")
        for entry, _ in results:
            assert entry.project == "proj_a"

    @pytest.mark.asyncio
    async def test_search_min_importance(self, ltm):
        await _store_entry(ltm, content="Low", importance=0.1)
        await _store_entry(ltm, content="High", importance=0.9)
        results = await ltm.search("test", min_importance=0.5)
        for entry, _ in results:
            assert entry.importance >= 0.5

    @pytest.mark.asyncio
    async def test_search_top_k(self, ltm):
        for i in range(5):
            await _store_entry(ltm, content=f"Memory {i}")
        results = await ltm.search("test", top_k=2)
        assert len(results) <= 2


# ── Search (Keyword Fallback) ────────────────────────────────────────────


class TestKeywordSearch:
    @pytest.mark.asyncio
    async def test_keyword_search_fallback(self, ltm_no_embeddings):
        await _store_entry(ltm_no_embeddings, content="Boss likes morning coffee")
        results = await ltm_no_embeddings.search("coffee")
        assert len(results) >= 1
        entry, score = results[0]
        assert "coffee" in entry.content

    @pytest.mark.asyncio
    async def test_keyword_search_no_results(self, ltm_no_embeddings):
        await _store_entry(ltm_no_embeddings, content="Something unrelated")
        results = await ltm_no_embeddings.search("zzzznonexistentzzzz")
        assert results == []

    @pytest.mark.asyncio
    async def test_keyword_search_filter_by_type(self, ltm_no_embeddings):
        await _store_entry(
            ltm_no_embeddings,
            content="Boss prefers tea",
            memory_type=MemoryType.PREFERENCE,
        )
        await _store_entry(
            ltm_no_embeddings,
            content="Meeting facts about tea",
            memory_type=MemoryType.FACT,
        )
        results = await ltm_no_embeddings.search(
            "tea", memory_type=MemoryType.PREFERENCE
        )
        for entry, _ in results:
            assert entry.memory_type == MemoryType.PREFERENCE


# ── Search by Entity ─────────────────────────────────────────────────────


class TestSearchByEntity:
    @pytest.mark.asyncio
    async def test_search_by_entity(self, ltm):
        await _store_entry(ltm, content="About Ravi", entities=["Ravi"])
        await _store_entry(ltm, content="About Priya", entities=["Priya"])
        results = await ltm.search_by_entity("Ravi")
        assert len(results) >= 1
        assert any("Ravi" in e.entities for e in results)

    @pytest.mark.asyncio
    async def test_search_by_entity_no_results(self, ltm):
        await _store_entry(ltm, content="No entities", entities=[])
        results = await ltm.search_by_entity("Unknown")
        assert results == []


# ── Search Upcoming Events ───────────────────────────────────────────────


class TestSearchUpcomingEvents:
    @pytest.mark.asyncio
    async def test_upcoming_events(self, ltm):
        future = datetime.now() + timedelta(days=3)
        await _store_entry(ltm, content="Deadline", event_date=future)
        results = await ltm.search_upcoming_events(days=7)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_upcoming_events_excludes_past(self, ltm):
        past = datetime.now() - timedelta(days=3)
        await _store_entry(ltm, content="Past event", event_date=past)
        results = await ltm.search_upcoming_events(days=7)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_upcoming_events_filter_project(self, ltm):
        future = datetime.now() + timedelta(days=2)
        await _store_entry(ltm, content="E1", event_date=future, project="p1")
        await _store_entry(ltm, content="E2", event_date=future, project="p2")
        results = await ltm.search_upcoming_events(days=7, project="p1")
        assert all(e.project == "p1" for e in results)


# ── Search by Type ────────────────────────────────────────────────────────


class TestSearchByType:
    @pytest.mark.asyncio
    async def test_search_by_type(self, ltm):
        await _store_entry(ltm, content="Fact 1", memory_type=MemoryType.FACT)
        await _store_entry(ltm, content="Pref 1", memory_type=MemoryType.PREFERENCE)
        results = await ltm.search_by_type(MemoryType.FACT)
        assert len(results) >= 1
        assert all(e.memory_type == MemoryType.FACT for e in results)

    @pytest.mark.asyncio
    async def test_search_by_type_with_project(self, ltm):
        await _store_entry(ltm, content="F1", memory_type=MemoryType.FACT, project="p1")
        await _store_entry(ltm, content="F2", memory_type=MemoryType.FACT, project="p2")
        results = await ltm.search_by_type(MemoryType.FACT, project="p1")
        assert all(e.project == "p1" for e in results)

    @pytest.mark.asyncio
    async def test_search_by_type_respects_top_k(self, ltm):
        for i in range(5):
            await _store_entry(ltm, content=f"Fact {i}", memory_type=MemoryType.FACT)
        results = await ltm.search_by_type(MemoryType.FACT, top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_by_type_ordered_by_importance(self, ltm):
        await _store_entry(
            ltm, content="Low", memory_type=MemoryType.FACT, importance=0.1
        )
        await _store_entry(
            ltm, content="High", memory_type=MemoryType.FACT, importance=0.9
        )
        results = await ltm.search_by_type(MemoryType.FACT)
        assert results[0].importance >= results[-1].importance


# ── Maintenance ───────────────────────────────────────────────────────────


class TestMaintenance:
    @pytest.mark.asyncio
    async def test_get_decay_candidates(self, ltm):
        await _store_entry(ltm, content="Low imp", importance=0.1)
        await _store_entry(ltm, content="High imp", importance=0.9)
        candidates = await ltm.get_decay_candidates(threshold=0.2)
        assert len(candidates) >= 1
        assert all(c.importance < 0.2 for c in candidates)

    @pytest.mark.asyncio
    async def test_get_decay_candidates_exclude_types(self, ltm):
        await _store_entry(
            ltm, content="Pref", memory_type=MemoryType.PREFERENCE, importance=0.1
        )
        await _store_entry(
            ltm, content="Fact", memory_type=MemoryType.FACT, importance=0.1
        )
        candidates = await ltm.get_decay_candidates(
            threshold=0.5, exclude_types=[MemoryType.PREFERENCE]
        )
        assert all(c.memory_type != MemoryType.PREFERENCE for c in candidates)

    @pytest.mark.asyncio
    async def test_get_expired_memories(self, ltm):
        past = datetime.now() - timedelta(days=5)
        await _store_entry(ltm, content="Expired", valid_until=past)
        await _store_entry(ltm, content="Not expired")
        expired = await ltm.get_expired_memories()
        assert len(expired) >= 1
        assert all(e.valid_until is not None for e in expired)

    @pytest.mark.asyncio
    async def test_get_expired_none_when_all_valid(self, ltm):
        future = datetime.now() + timedelta(days=30)
        await _store_entry(ltm, content="Valid", valid_until=future)
        expired = await ltm.get_expired_memories()
        assert len(expired) == 0


# ── Stats ─────────────────────────────────────────────────────────────────


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_empty(self, ltm):
        stats = await ltm.get_stats()
        assert stats["total"] == 0
        assert stats["with_embeddings"] == 0
        assert stats["embedding_coverage"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_data(self, ltm):
        await _store_entry(ltm, content="Fact", memory_type=MemoryType.FACT)
        await _store_entry(ltm, content="Pref", memory_type=MemoryType.PREFERENCE)
        stats = await ltm.get_stats()
        assert stats["total"] == 2
        assert stats["with_embeddings"] == 2
        assert stats["embedding_coverage"] == 1.0
        assert "fact" in stats["by_type"]
        assert "preference" in stats["by_type"]

    @pytest.mark.asyncio
    async def test_stats_projects(self, ltm):
        await _store_entry(ltm, project="p1")
        await _store_entry(ltm, project="p2")
        stats = await ltm.get_stats()
        assert stats["projects"] == 2

    @pytest.mark.asyncio
    async def test_stats_vector_backend(self, ltm):
        stats = await ltm.get_stats()
        assert stats["vector_backend"] == "sqlite"

    @pytest.mark.asyncio
    async def test_stats_avg_importance(self, ltm):
        await _store_entry(ltm, importance=0.4)
        await _store_entry(ltm, importance=0.6)
        stats = await ltm.get_stats()
        assert stats["avg_importance"] == pytest.approx(0.5, abs=0.01)


# ── Embedding Serialization ──────────────────────────────────────────────


class TestEmbeddingSerialization:
    def test_serialize_none(self, ltm):
        assert ltm._serialize_embedding(None) is None

    def test_deserialize_none(self, ltm):
        assert ltm._deserialize_embedding(None) is None

    def test_roundtrip(self, ltm):
        original = [0.1, 0.2, 0.3, 0.4]
        serialized = ltm._serialize_embedding(original)
        deserialized = ltm._deserialize_embedding(serialized)
        np.testing.assert_array_almost_equal(deserialized, original, decimal=5)

    def test_serialized_is_bytes(self, ltm):
        serialized = ltm._serialize_embedding([1.0, 2.0])
        assert isinstance(serialized, bytes)


# ── Sync to Vector Store ─────────────────────────────────────────────────


class TestSyncToVectorStore:
    @pytest.mark.asyncio
    async def test_sync_no_vector_store(self, ltm):
        ltm._vector_store = None
        result = await ltm.sync_to_vector_store()
        assert result == 0

    @pytest.mark.asyncio
    async def test_sync_empty_db(self, ltm):
        mock_vs = AsyncMock()
        mock_vs.ensure_collection = AsyncMock()
        ltm._vector_store = mock_vs
        with patch.dict(
            "sys.modules",
            {"db.vector_store": MagicMock(COLLECTION_MEMORIES="friday_memories")},
        ):
            result = await ltm.sync_to_vector_store()
        assert result == 0

    @pytest.mark.asyncio
    async def test_sync_with_data(self, ltm):
        await _store_entry(ltm, content="Memory 1")
        await _store_entry(ltm, content="Memory 2")

        mock_vs = AsyncMock()
        mock_vs.ensure_collection = AsyncMock()
        mock_vs.upsert_batch = AsyncMock(return_value=2)
        ltm._vector_store = mock_vs

        with patch.dict(
            "sys.modules",
            {"db.vector_store": MagicMock(COLLECTION_MEMORIES="friday_memories")},
        ):
            result = await ltm.sync_to_vector_store()
        assert result == 2


# ── Row to Entry ──────────────────────────────────────────────────────────


class TestRowToEntry:
    @pytest.mark.asyncio
    async def test_full_roundtrip_via_db(self, ltm):
        """Store → DB → get → verify all fields preserved."""
        stored = await _store_entry(
            ltm,
            content="Full test",
            memory_type=MemoryType.DECISION,
            source_summary="from test",
            domain="film",
            project="proj1",
            entities=["char1"],
            confidence=0.95,
            trust_level=5,
            importance=0.8,
            language="te",
            telugu_keywords=["paata"],
            source_stm_ids=["stm1"],
        )
        retrieved = await ltm.get(stored.id)
        assert retrieved.content == "Full test"
        assert retrieved.memory_type == MemoryType.DECISION
        assert retrieved.source_summary == "from test"
        assert retrieved.domain == "film"
        assert retrieved.project == "proj1"
        assert retrieved.entities == ["char1"]
        assert retrieved.confidence == 0.95
        assert retrieved.trust_level == 5
        assert retrieved.language == "te"
        assert retrieved.telugu_keywords == ["paata"]
        assert retrieved.source_stm_ids == ["stm1"]


# ── Edge Cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_store_empty_content(self, ltm):
        entry = await _store_entry(ltm, content="")
        assert entry.content == ""

    @pytest.mark.asyncio
    async def test_store_unicode_telugu(self, ltm):
        entry = await _store_entry(ltm, content="నేను Friday", language="te")
        retrieved = await ltm.get(entry.id)
        assert "Friday" in retrieved.content

    @pytest.mark.asyncio
    async def test_store_very_long_content(self, ltm):
        long_content = "A" * 10000
        entry = await _store_entry(ltm, content=long_content)
        retrieved = await ltm.get(entry.id)
        assert len(retrieved.content) == 10000

    @pytest.mark.asyncio
    async def test_multiple_stores_unique_ids(self, ltm):
        e1 = await _store_entry(ltm, content="First")
        e2 = await _store_entry(ltm, content="Second")
        assert e1.id != e2.id

    @pytest.mark.asyncio
    async def test_delete_then_get(self, ltm):
        stored = await _store_entry(ltm)
        await ltm.delete(stored.id)
        assert await ltm.get(stored.id) is None

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, ltm):
        """Update on nonexistent returns None from get()."""
        result = await ltm.update("nonexistent", content="new")
        # update calls get() internally, returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, ltm):
        """Transaction rolls back on error."""
        initial_count = (await ltm.get_stats())["total"]
        try:
            with ltm._transaction() as cur:
                cur.execute(
                    "INSERT INTO long_term_memories (id, content, memory_type, created_at) VALUES (?, ?, ?, ?)",
                    ("test_id", "test", "fact", datetime.now().isoformat()),
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass
        final_count = (await ltm.get_stats())["total"]
        assert final_count == initial_count

    @pytest.mark.asyncio
    async def test_close_idempotent(self, ltm):
        """Closing twice is safe."""
        await ltm.close()
        await ltm.close()  # Should not raise
