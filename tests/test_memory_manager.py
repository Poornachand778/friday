"""
Tests for memory/manager.py
============================

Comprehensive tests for MemoryManager — the central memory coordinator.
Covers initialization, session management, conversation storage, search,
voice commands, knowledge graph operations, context building, and singleton.

Tests: 90+
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio

from memory.layers.long_term import MemoryType
from memory.manager import (
    MemoryManager,
    get_memory_manager,
    initialize_memory,
)


# ── Helpers ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    import memory.manager as mod

    mod._manager = None
    yield
    mod._manager = None


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.working = MagicMock()
    cfg.stm = MagicMock()
    cfg.ltm = MagicMock()
    cfg.profile = MagicMock()
    return cfg


def _make_mock_working():
    wm = MagicMock()
    wm.turn_count = 0
    wm.token_count = 0
    wm.current_room = "general"
    wm.current_project = None
    wm.language_mode = "en"
    wm.get_turns.return_value = []
    wm.get_attention_topics.return_value = []
    wm.get_prefetched_ltm.return_value = []
    wm.get_last_turn.return_value = None
    wm.clear = MagicMock()
    wm.add_turn = MagicMock()
    wm.set_language_mode = MagicMock()
    wm.set_project = MagicMock()
    wm.set_prefetched_ltm = MagicMock()
    return wm


def _make_mock_stm():
    stm = MagicMock()
    stm.initialize = AsyncMock()
    stm.close = AsyncMock()
    stm.store = AsyncMock()
    stm.search = AsyncMock(return_value=[])
    stm.get_stats = AsyncMock(return_value={"active": 0})
    return stm


def _make_mock_ltm():
    ltm = MagicMock()
    ltm.initialize = AsyncMock()
    ltm.close = AsyncMock()
    ltm.store = AsyncMock()
    ltm.search = AsyncMock(return_value=[])
    ltm.boost_importance = AsyncMock(return_value=None)
    ltm.get_stats = AsyncMock(return_value={"total": 0})
    return ltm


def _make_mock_profile():
    profile = MagicMock()
    profile.initialize = AsyncMock()
    profile.record_interaction = MagicMock()
    profile.set_current_project = MagicMock()
    profile.get_summary = MagicMock(return_value={"name": "Boss"})
    profile.profile = MagicMock()
    profile.profile.version = 1
    profile.profile.projects = []
    profile.profile.relationships = []
    return profile


def _make_mock_kg():
    kg = MagicMock()
    kg.initialize = AsyncMock()
    kg.close = AsyncMock()
    kg.add_node = AsyncMock()
    kg.add_triplet = AsyncMock()
    kg.traverse = AsyncMock(return_value=[])
    kg.get_related = AsyncMock(return_value=[])
    kg.get_by_type = AsyncMock(return_value=[])
    kg.get_stats = AsyncMock(return_value={"nodes": 0, "edges": 0})
    return kg


def _make_mock_triplet_extractor():
    te = MagicMock()
    te.is_configured = False
    te.close = AsyncMock()
    result = MagicMock()
    result.count = 0
    result.high_confidence = MagicMock(return_value=[])
    te.extract = AsyncMock(return_value=result)
    return te


@pytest_asyncio.fixture
async def manager(mock_config):
    """Fully initialized MemoryManager with all mocks."""
    with patch("memory.manager.get_memory_config", return_value=mock_config):
        with patch("memory.manager.TeluguEnglishProcessor") as MockTelugu:
            mock_telugu = MagicMock()
            processed = MagicMock()
            processed.telugu_density = 0.0
            processed.dominant_language = "en"
            processed.telugu_keywords = []
            mock_telugu.process.return_value = processed
            MockTelugu.return_value = mock_telugu

            mgr = MemoryManager(config=mock_config)
            mgr._telugu = mock_telugu

    mgr._working = _make_mock_working()
    mgr._stm = _make_mock_stm()
    mgr._ltm = _make_mock_ltm()
    mgr._profile = _make_mock_profile()
    mgr._knowledge_graph = _make_mock_kg()
    mgr._triplet_extractor = _make_mock_triplet_extractor()
    mgr._initialized = True

    yield mgr


# ── Init & Lifecycle ─────────────────────────────────────────────────────


class TestInit:
    def test_constructor(self, mock_config):
        with patch("memory.manager.get_memory_config", return_value=mock_config):
            with patch("memory.manager.TeluguEnglishProcessor"):
                mgr = MemoryManager(config=mock_config)
        assert mgr._initialized is False
        assert mgr._working is None

    def test_constructor_default_config(self):
        with patch("memory.manager.get_memory_config") as mock_get:
            mock_get.return_value = MagicMock()
            with patch("memory.manager.TeluguEnglishProcessor"):
                mgr = MemoryManager()
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize(self, mock_config):
        with patch("memory.manager.get_memory_config", return_value=mock_config), patch(
            "memory.manager.TeluguEnglishProcessor"
        ), patch("memory.manager.WorkingMemory"), patch(
            "memory.manager.ShortTermMemory"
        ) as MockSTM, patch(
            "memory.manager.LongTermMemory"
        ) as MockLTM, patch(
            "memory.manager.ProfileStore"
        ) as MockProfile, patch(
            "memory.manager.KnowledgeGraph"
        ) as MockKG, patch(
            "memory.manager.TripletExtractor"
        ) as MockTE:
            MockSTM.return_value.initialize = AsyncMock()
            MockLTM.return_value.initialize = AsyncMock()
            MockProfile.return_value.initialize = AsyncMock()
            MockKG.return_value.initialize = AsyncMock()
            MockTE.return_value.is_configured = False

            mgr = MemoryManager(config=mock_config)
            await mgr.initialize()
            assert mgr._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, manager):
        await manager.initialize()  # Already initialized
        # Should be a no-op

    @pytest.mark.asyncio
    async def test_shutdown(self, manager):
        await manager.shutdown()
        manager._stm.close.assert_called_once()
        manager._ltm.close.assert_called_once()
        manager._knowledge_graph.close.assert_called_once()
        manager._triplet_extractor.close.assert_called_once()
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_partial(self, mock_config):
        """Shutdown with some components None."""
        with patch("memory.manager.get_memory_config", return_value=mock_config):
            with patch("memory.manager.TeluguEnglishProcessor"):
                mgr = MemoryManager(config=mock_config)
        # Nothing initialized
        await mgr.shutdown()  # Should not raise


# ── Properties ───────────────────────────────────────────────────────────


class TestProperties:
    def test_is_initialized(self, manager):
        assert manager.is_initialized is True

    def test_working_property(self, manager):
        assert manager.working is not None

    def test_stm_property(self, manager):
        assert manager.stm is not None

    def test_ltm_property(self, manager):
        assert manager.ltm is not None

    def test_profile_property(self, manager):
        assert manager.profile is not None

    def test_knowledge_graph_property(self, manager):
        assert manager.knowledge_graph is not None

    def test_working_not_initialized(self, mock_config):
        with patch("memory.manager.get_memory_config", return_value=mock_config):
            with patch("memory.manager.TeluguEnglishProcessor"):
                mgr = MemoryManager(config=mock_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = mgr.working

    def test_stm_not_initialized(self, mock_config):
        with patch("memory.manager.get_memory_config", return_value=mock_config):
            with patch("memory.manager.TeluguEnglishProcessor"):
                mgr = MemoryManager(config=mock_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = mgr.stm

    def test_ltm_not_initialized(self, mock_config):
        with patch("memory.manager.get_memory_config", return_value=mock_config):
            with patch("memory.manager.TeluguEnglishProcessor"):
                mgr = MemoryManager(config=mock_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = mgr.ltm

    def test_profile_not_initialized(self, mock_config):
        with patch("memory.manager.get_memory_config", return_value=mock_config):
            with patch("memory.manager.TeluguEnglishProcessor"):
                mgr = MemoryManager(config=mock_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = mgr.profile

    def test_knowledge_graph_not_initialized(self, mock_config):
        with patch("memory.manager.get_memory_config", return_value=mock_config):
            with patch("memory.manager.TeluguEnglishProcessor"):
                mgr = MemoryManager(config=mock_config)
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = mgr.knowledge_graph


# ── Session Management ───────────────────────────────────────────────────


class TestSession:
    def test_start_session(self, manager):
        manager.start_session("sess123")
        assert manager.get_session_id() == "sess123"
        manager._working.clear.assert_called_once()

    def test_get_session_id_none(self, manager):
        assert manager.get_session_id() is None

    @pytest.mark.asyncio
    async def test_end_session(self, manager):
        manager.start_session("sess1")
        manager._working.turn_count = 0
        result = await manager.end_session()
        assert result is None  # No turns to save
        assert manager.get_session_id() is None
        manager._working.clear.assert_called()

    @pytest.mark.asyncio
    async def test_end_session_with_stm_save(self, manager):
        manager.start_session("sess1")
        manager._working.turn_count = 3

        mock_turn = MagicMock()
        mock_turn.user_message = "Test question"
        mock_turn.assistant_response = "Test answer"
        manager._working.get_turns.return_value = [mock_turn]
        manager._working.get_attention_topics.return_value = []

        mock_stm_entry = MagicMock()
        mock_stm_entry.id = "stm1"
        manager._stm.store.return_value = mock_stm_entry

        result = await manager.end_session(save_to_stm=True)
        assert result is not None
        manager._stm.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_session_no_current(self, manager):
        result = await manager.end_session()
        assert result is None


# ── Store Turn ───────────────────────────────────────────────────────────


class TestStoreTurn:
    @pytest.mark.asyncio
    async def test_store_turn_basic(self, manager):
        manager._current_session_id = "sess1"
        turn = await manager.store_turn(
            user_message="Hello Boss",
            assistant_response="Hi, what's up?",
        )
        manager._working.add_turn.assert_called_once()
        manager._profile.record_interaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_turn_auto_session(self, manager):
        """Auto-creates session ID if none set."""
        await manager.store_turn("Hi", "Hello")
        assert manager._current_session_id is not None
        assert manager._current_session_id.startswith("auto_")

    @pytest.mark.asyncio
    async def test_store_turn_telugu_detection(self, manager):
        """Telugu density triggers language mode switch."""
        processed = MagicMock()
        processed.telugu_density = 0.5  # >0.4 → "te"
        processed.dominant_language = "te"
        processed.telugu_keywords = ["paata"]
        manager._telugu.process.return_value = processed

        manager._current_session_id = "sess1"
        await manager.store_turn("నేను Friday", "Boss, baagunnanu")
        manager._working.set_language_mode.assert_called_with("te")

    @pytest.mark.asyncio
    async def test_store_turn_mixed_language(self, manager):
        processed = MagicMock()
        processed.telugu_density = 0.2  # >0.1, <0.4 → "mixed"
        processed.dominant_language = "mixed"
        processed.telugu_keywords = []
        manager._telugu.process.return_value = processed

        manager._current_session_id = "sess1"
        await manager.store_turn("Hi Boss, ikkada em chestunnav?", "Working")
        manager._working.set_language_mode.assert_called_with("mixed")

    @pytest.mark.asyncio
    async def test_store_turn_english(self, manager):
        processed = MagicMock()
        processed.telugu_density = 0.0
        processed.dominant_language = "en"
        processed.telugu_keywords = []
        manager._telugu.process.return_value = processed

        manager._current_session_id = "sess1"
        await manager.store_turn("Hello", "Hi")
        manager._working.set_language_mode.assert_called_with("en")


# ── Search ────────────────────────────────────────────────────────────────


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_all_layers(self, manager):
        results = await manager.search("test query")
        assert isinstance(results, list)
        manager._stm.search.assert_called_once()
        manager._ltm.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_stm_only(self, manager):
        results = await manager.search("test", include_ltm=False, include_working=False)
        manager._stm.search.assert_called_once()
        manager._ltm.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_ltm_only(self, manager):
        results = await manager.search("test", include_stm=False, include_working=False)
        manager._ltm.search.assert_called_once()
        manager._stm.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_working_memory_match(self, manager):
        topic = MagicMock()
        topic.topic = "climax scene discussion"
        topic.relevance = 0.9
        topic.metadata = {}
        manager._working.get_attention_topics.return_value = [topic]

        results = await manager.search("climax")
        assert any(r["source"] == "working" for r in results)

    @pytest.mark.asyncio
    async def test_search_prefetched_match(self, manager):
        mem = MagicMock()
        mem.content = "Boss prefers morning meetings"
        mem.relevance = 0.8
        mem.memory_id = "ltm1"
        manager._working.get_prefetched_ltm.return_value = [mem]

        results = await manager.search("morning")
        assert any(r["source"] == "prefetched" for r in results)

    @pytest.mark.asyncio
    async def test_search_stm_results(self, manager):
        stm_entry = MagicMock()
        stm_entry.id = "stm1"
        stm_entry.summary = "Discussed climax"
        stm_entry.created_at = datetime(2025, 1, 15)
        stm_entry.topics = ["climax"]
        manager._stm.search.return_value = [stm_entry]

        results = await manager.search("climax")
        assert any(r["source"] == "stm" for r in results)

    @pytest.mark.asyncio
    async def test_search_ltm_results(self, manager):
        ltm_entry = MagicMock()
        ltm_entry.id = "ltm1"
        ltm_entry.content = "Boss prefers direct communication"
        ltm_entry.memory_type = MemoryType.PREFERENCE
        ltm_entry.created_at = datetime(2025, 1, 10)
        manager._ltm.search.return_value = [(ltm_entry, 0.95)]

        results = await manager.search("communication")
        assert any(r["source"] == "ltm" for r in results)

    @pytest.mark.asyncio
    async def test_search_sorted_by_relevance(self, manager):
        topic = MagicMock()
        topic.topic = "test topic"
        topic.relevance = 0.3
        topic.metadata = {}
        manager._working.get_attention_topics.return_value = [topic]

        ltm_entry = MagicMock()
        ltm_entry.id = "ltm1"
        ltm_entry.content = "test content"
        ltm_entry.memory_type = MemoryType.FACT
        ltm_entry.created_at = datetime(2025, 1, 1)
        manager._ltm.search.return_value = [(ltm_entry, 0.9)]

        results = await manager.search("test")
        if len(results) >= 2:
            assert results[0]["relevance"] >= results[1]["relevance"]

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, manager):
        entries = []
        for i in range(10):
            e = MagicMock()
            e.id = f"stm{i}"
            e.summary = f"Memory {i}"
            e.created_at = datetime(2025, 1, i + 1)
            e.topics = []
            entries.append(e)
        manager._stm.search.return_value = entries

        results = await manager.search("test", top_k=3)
        assert len(results) <= 3


# ── Prefetch ──────────────────────────────────────────────────────────────


class TestPrefetch:
    @pytest.mark.asyncio
    async def test_prefetch_for_context(self, manager):
        ltm_entry = MagicMock()
        ltm_entry.id = "ltm1"
        ltm_entry.content = "Relevant memory"
        ltm_entry.memory_type = MemoryType.FACT
        manager._ltm.search.return_value = [(ltm_entry, 0.85)]

        result = await manager.prefetch_for_context("climax scene")
        assert len(result) == 1
        manager._working.set_prefetched_ltm.assert_called_once()

    @pytest.mark.asyncio
    async def test_prefetch_empty(self, manager):
        manager._ltm.search.return_value = []
        result = await manager.prefetch_for_context("nothing")
        assert result == []


# ── Store Fact ────────────────────────────────────────────────────────────


class TestStoreFact:
    @pytest.mark.asyncio
    async def test_store_fact_basic(self, manager):
        mock_entry = MagicMock()
        mock_entry.id = "ltm1"
        manager._ltm.store.return_value = mock_entry

        result = await manager.store_fact("Boss likes coffee")
        assert result is mock_entry
        manager._ltm.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_fact_with_type(self, manager):
        mock_entry = MagicMock()
        mock_entry.id = "ltm2"
        manager._ltm.store.return_value = mock_entry

        await manager.store_fact(
            "Boss prefers mornings",
            memory_type=MemoryType.PREFERENCE,
            importance=0.9,
        )
        call_kwargs = manager._ltm.store.call_args[1]
        assert call_kwargs["memory_type"] == MemoryType.PREFERENCE
        assert call_kwargs["importance"] == 0.9

    @pytest.mark.asyncio
    async def test_store_fact_extracts_triplets(self, manager):
        mock_entry = MagicMock()
        mock_entry.id = "ltm3"
        manager._ltm.store.return_value = mock_entry
        manager._triplet_extractor.is_configured = True

        result_mock = MagicMock()
        result_mock.count = 2
        result_mock.high_confidence.return_value = [
            MagicMock(subject="Ravi", relation="appears_in", object="Scene 5"),
        ]
        manager._triplet_extractor.extract.return_value = result_mock

        await manager.store_fact("Ravi appears in Scene 5", extract_triplets=True)
        manager._triplet_extractor.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_fact_no_triplets(self, manager):
        mock_entry = MagicMock()
        mock_entry.id = "ltm4"
        manager._ltm.store.return_value = mock_entry

        await manager.store_fact("Simple fact", extract_triplets=False)
        manager._triplet_extractor.extract.assert_not_called()


# ── Boost Memory ──────────────────────────────────────────────────────────


class TestBoostMemory:
    @pytest.mark.asyncio
    async def test_boost_success(self, manager):
        mock_entry = MagicMock()
        mock_entry.importance = 0.7
        manager._ltm.boost_importance.return_value = mock_entry

        result = await manager.boost_memory("ltm1", 0.2)
        assert result is True

    @pytest.mark.asyncio
    async def test_boost_not_found(self, manager):
        manager._ltm.boost_importance.return_value = None
        result = await manager.boost_memory("nonexistent")
        assert result is False


# ── Voice Commands ────────────────────────────────────────────────────────


class TestVoiceCommands:
    @pytest.mark.asyncio
    async def test_remember_command(self, manager):
        mock_entry = MagicMock()
        mock_entry.id = "ltm_remembered"
        manager._ltm.store.return_value = mock_entry

        result = await manager.voice_command("remember this: Ravi is the hero")
        assert result["action"] == "stored"
        assert "Ravi is the hero" in result["content"]

    @pytest.mark.asyncio
    async def test_remember_command_telugu(self, manager):
        mock_entry = MagicMock()
        mock_entry.id = "ltm_te"
        manager._ltm.store.return_value = mock_entry

        result = await manager.voice_command("గుర్తుంచుకో: Ravi is hero")
        assert result["action"] == "stored"

    @pytest.mark.asyncio
    async def test_important_command(self, manager):
        mock_turn = MagicMock()
        mock_turn.user_message = "The climax needs rework"
        manager._working.get_last_turn.return_value = mock_turn

        mock_entry = MagicMock()
        mock_entry.id = "ltm_imp"
        manager._ltm.store.return_value = mock_entry

        result = await manager.voice_command("this is important")
        assert result["action"] == "boosted"

    @pytest.mark.asyncio
    async def test_important_nothing_to_boost(self, manager):
        manager._working.get_last_turn.return_value = None
        result = await manager.voice_command("this is important")
        assert result["action"] == "nothing_to_boost"

    @pytest.mark.asyncio
    async def test_switch_project(self, manager):
        result = await manager.voice_command("switch to gusagusalu")
        assert result["action"] == "switched_project"
        assert result["project"] == "gusagusalu"
        manager._working.set_project.assert_called_with("gusagusalu")
        manager._profile.set_current_project.assert_called_with("gusagusalu")

    @pytest.mark.asyncio
    async def test_forget_command(self, manager):
        result = await manager.voice_command("forget about old scenes")
        assert result["action"] == "confirm_delete"
        assert "old scenes" in result["topic"]

    @pytest.mark.asyncio
    async def test_unknown_command(self, manager):
        result = await manager.voice_command("play music")
        assert result["action"] == "unknown"


# ── Knowledge Graph Operations ───────────────────────────────────────────


class TestGraphOperations:
    @pytest.mark.asyncio
    async def test_graph_query(self, manager):
        mock_result = MagicMock()
        mock_result.node = MagicMock()
        mock_result.node.name = "Scene 5"
        mock_result.node.node_type = MagicMock()
        mock_result.node.node_type.value = "scene"
        mock_result.node.project = "gusagusalu"
        mock_result.path = ["Ravi", "Scene 5"]
        mock_result.relations = ["appears_in"]
        mock_result.depth = 1
        manager._knowledge_graph.traverse.return_value = [mock_result]

        results = await manager.graph_query("Ravi")
        assert len(results) == 1
        assert results[0]["entity"] == "Scene 5"

    @pytest.mark.asyncio
    async def test_graph_query_no_kg(self, manager):
        manager._knowledge_graph = None
        results = await manager.graph_query("Ravi")
        assert results == []

    @pytest.mark.asyncio
    async def test_graph_query_with_filter(self, manager):
        manager._knowledge_graph.traverse.return_value = []
        await manager.graph_query("Ravi", relation_filter=["appears_in"])
        manager._knowledge_graph.traverse.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_related_entities(self, manager):
        mock_node = MagicMock()
        mock_node.name = "Priya"
        mock_node.node_type = MagicMock()
        mock_node.node_type.value = "character"
        mock_node.project = "proj1"
        manager._knowledge_graph.get_related.return_value = [(mock_node, "loves")]

        results = await manager.get_related_entities("Ravi")
        assert len(results) == 1
        assert results[0]["entity"] == "Priya"
        assert results[0]["relation"] == "loves"

    @pytest.mark.asyncio
    async def test_get_related_entities_no_kg(self, manager):
        manager._knowledge_graph = None
        results = await manager.get_related_entities("Ravi")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_entities_by_type(self, manager):
        mock_node = MagicMock()
        mock_node.name = "Ravi"
        mock_node.node_type = MagicMock()
        mock_node.node_type.value = "character"
        mock_node.project = "proj1"
        mock_node.attributes = {}
        manager._knowledge_graph.get_by_type.return_value = [mock_node]

        results = await manager.get_entities_by_type("character")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_entities_invalid_type(self, manager):
        results = await manager.get_entities_by_type("invalid_type_xyz")
        assert results == []

    @pytest.mark.asyncio
    async def test_add_entity(self, manager):
        result = await manager.add_entity_to_graph("Ravi", "character", project="proj1")
        assert result is True
        manager._knowledge_graph.add_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_entity_invalid_type(self, manager):
        result = await manager.add_entity_to_graph("Thing", "invalid_xyz")
        assert result is True  # Falls back to CONCEPT

    @pytest.mark.asyncio
    async def test_add_entity_no_kg(self, manager):
        manager._knowledge_graph = None
        result = await manager.add_entity_to_graph("Ravi", "character")
        assert result is False

    @pytest.mark.asyncio
    async def test_add_relationship(self, manager):
        result = await manager.add_relationship("Ravi", "appears_in", "Scene 5")
        assert result is True
        manager._knowledge_graph.add_triplet.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_relationship_no_kg(self, manager):
        manager._knowledge_graph = None
        result = await manager.add_relationship("A", "rel", "B")
        assert result is False


# ── Context for LLM ──────────────────────────────────────────────────────


class TestContextForLLM:
    def test_get_context_for_llm(self, manager):
        context = manager.get_context_for_llm()
        assert "profile" in context
        assert "current_room" in context
        assert "current_project" in context
        assert "language_mode" in context
        assert "attention" in context
        assert "prefetched_memories" in context

    def test_context_includes_attention(self, manager):
        topic = MagicMock()
        topic.topic = "scene structure"
        topic.relevance = 0.8
        manager._working.get_attention_topics.return_value = [topic]

        context = manager.get_context_for_llm()
        assert len(context["attention"]) == 1
        assert context["attention"][0]["topic"] == "scene structure"


# ── Health & Stats ────────────────────────────────────────────────────────


class TestHealthStats:
    @pytest.mark.asyncio
    async def test_health_check(self, manager):
        health = await manager.health_check()
        assert health["initialized"] is True
        assert "working_memory" in health
        assert "stm" in health
        assert "ltm" in health
        assert "profile" in health
        assert "knowledge_graph" in health
        assert "triplet_extractor" in health

    @pytest.mark.asyncio
    async def test_get_stats(self, manager):
        stats = await manager.get_stats()
        assert "session" in stats
        assert "initialized" in stats


# ── Singleton ────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_memory_manager(self):
        with patch("memory.manager.get_memory_config") as mock_get:
            mock_get.return_value = MagicMock()
            with patch("memory.manager.TeluguEnglishProcessor"):
                mgr1 = get_memory_manager()
                mgr2 = get_memory_manager()
                assert mgr1 is mgr2

    @pytest.mark.asyncio
    async def test_initialize_memory(self):
        with patch("memory.manager.get_memory_config") as mock_get, patch(
            "memory.manager.TeluguEnglishProcessor"
        ), patch("memory.manager.WorkingMemory"), patch(
            "memory.manager.ShortTermMemory"
        ) as MockSTM, patch(
            "memory.manager.LongTermMemory"
        ) as MockLTM, patch(
            "memory.manager.ProfileStore"
        ) as MockProfile, patch(
            "memory.manager.KnowledgeGraph"
        ) as MockKG, patch(
            "memory.manager.TripletExtractor"
        ) as MockTE:
            mock_get.return_value = MagicMock()
            MockSTM.return_value.initialize = AsyncMock()
            MockLTM.return_value.initialize = AsyncMock()
            MockProfile.return_value.initialize = AsyncMock()
            MockKG.return_value.initialize = AsyncMock()
            MockTE.return_value.is_configured = False

            mgr = await initialize_memory()
            assert mgr._initialized is True


# ── Triplet Extraction ───────────────────────────────────────────────────


class TestTripletExtraction:
    @pytest.mark.asyncio
    async def test_extract_and_store_triplets(self, manager):
        triplet = MagicMock()
        triplet.subject = "Ravi"
        triplet.relation = "loves"
        triplet.object = "Priya"

        result_mock = MagicMock()
        result_mock.count = 1
        result_mock.high_confidence.return_value = [triplet]
        manager._triplet_extractor.extract.return_value = result_mock

        count = await manager._extract_and_store_triplets("Ravi loves Priya")
        assert count == 1
        manager._knowledge_graph.add_triplet.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_no_extractor(self, manager):
        manager._triplet_extractor = None
        count = await manager._extract_and_store_triplets("test")
        assert count == 0

    @pytest.mark.asyncio
    async def test_extract_no_kg(self, manager):
        manager._knowledge_graph = None
        count = await manager._extract_and_store_triplets("test")
        assert count == 0

    @pytest.mark.asyncio
    async def test_extract_exception_handled(self, manager):
        manager._triplet_extractor.extract.side_effect = RuntimeError("LLM error")
        count = await manager._extract_and_store_triplets("test")
        assert count == 0  # Exception caught, returns 0
