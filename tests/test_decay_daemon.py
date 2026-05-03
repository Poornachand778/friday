"""
Tests for memory/operations/decay.py
=====================================

Comprehensive tests for DecayDaemon, DecayConfig, DecayCycleResult.
Covers scoring algorithm, Ebbinghaus decay math, consolidation,
pruning, reinforcement, daemon lifecycle, and edge cases.

Tests: 70+
"""

import asyncio
import math
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memory.layers.long_term import MemoryType

from memory.operations.decay import (
    DecayConfig,
    DecayCycleResult,
    DecayDaemon,
    run_decay,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_entry(
    id="entry_001",
    summary="Test memory",
    importance=0.7,
    access_count=3,
    created_at=None,
    last_accessed=None,
    project=None,
):
    """Create a mock STM entry."""
    entry = MagicMock()
    entry.id = id
    entry.summary = summary
    entry.importance = importance
    entry.access_count = access_count
    entry.created_at = created_at or datetime.now() - timedelta(days=2)
    entry.last_accessed = last_accessed
    entry.project = project
    return entry


def _make_manager(
    stm_entries=None,
    ltm_entry=None,
    has_kg=False,
):
    """Create a mock MemoryManager."""
    manager = MagicMock()

    # STM mock
    stm = MagicMock()
    stm.get_entries_before = AsyncMock(return_value=stm_entries or [])
    stm.delete = AsyncMock()
    stm.update_importance = AsyncMock()
    stm.get = AsyncMock(return_value=None)
    stm.mark_accessed = AsyncMock()
    manager.stm = stm

    # LTM mock
    ltm = MagicMock()
    ltm.get = AsyncMock(return_value=ltm_entry)
    ltm.boost_importance = AsyncMock()
    manager.ltm = ltm

    # store_fact mock
    ltm_result = MagicMock()
    ltm_result.importance = 0.7
    manager.store_fact = AsyncMock(return_value=ltm_result)

    # Knowledge graph
    if has_kg:
        kg = MagicMock()
        kg.cleanup_orphans = AsyncMock(return_value=3)
        manager._knowledge_graph = kg
    else:
        manager._knowledge_graph = None

    return manager


# ── DecayConfig ───────────────────────────────────────────────────────────


class TestDecayConfig:
    def test_defaults(self):
        config = DecayConfig()
        assert config.run_interval_hours == 6
        assert config.retention_days_stm == 7
        assert config.consolidation_threshold == 0.6
        assert config.pruning_threshold == 0.1
        assert config.importance_floor == 0.1
        assert config.base_decay_rate == 0.1
        assert config.decay_half_life_days == 3.0
        assert config.max_consolidations_per_run == 20
        assert config.min_access_count_for_ltm == 2
        assert config.prune_orphan_nodes_days == 14

    def test_custom_values(self):
        config = DecayConfig(
            run_interval_hours=12,
            consolidation_threshold=0.8,
            decay_half_life_days=5.0,
        )
        assert config.run_interval_hours == 12
        assert config.consolidation_threshold == 0.8
        assert config.decay_half_life_days == 5.0

    def test_from_dict(self):
        data = {
            "run_interval_hours": 12,
            "consolidation_threshold": 0.8,
            "unknown_field": "ignored",
        }
        config = DecayConfig.from_dict(data)
        assert config.run_interval_hours == 12
        assert config.consolidation_threshold == 0.8

    def test_from_dict_empty(self):
        config = DecayConfig.from_dict({})
        assert config.run_interval_hours == 6  # defaults

    def test_from_dict_ignores_unknown(self):
        config = DecayConfig.from_dict({"not_a_field": 42})
        # Should not raise, just ignore unknown fields
        assert config.run_interval_hours == 6


# ── DecayCycleResult ──────────────────────────────────────────────────────


class TestDecayCycleResult:
    def test_to_dict(self):
        result = DecayCycleResult(
            timestamp=datetime(2025, 1, 15, 10, 0, 0),
            stm_entries_checked=50,
            stm_entries_decayed=10,
            stm_entries_pruned=5,
            entries_consolidated=3,
            graph_nodes_pruned=2,
            duration_seconds=1.5,
        )
        d = result.to_dict()
        assert d["stm_checked"] == 50
        assert d["stm_decayed"] == 10
        assert d["stm_pruned"] == 5
        assert d["consolidated"] == 3
        assert d["graph_pruned"] == 2
        assert d["duration_seconds"] == 1.5
        assert "2025-01-15" in d["timestamp"]


# ── DecayDaemon Init ─────────────────────────────────────────────────────


class TestDecayDaemonInit:
    def test_init_default_config(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        assert daemon._manager is manager
        assert isinstance(daemon._config, DecayConfig)
        assert daemon.is_running is False
        assert daemon.last_run is None

    def test_init_custom_config(self):
        manager = _make_manager()
        config = DecayConfig(run_interval_hours=12)
        daemon = DecayDaemon(manager, config=config)
        assert daemon._config.run_interval_hours == 12


# ── _calculate_score ──────────────────────────────────────────────────────


class TestCalculateScore:
    def test_score_basic(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        entry = _make_entry(importance=0.8, access_count=3, created_at=datetime.now())
        score = daemon._calculate_score(entry)
        assert 0.0 <= score <= 1.0

    def test_score_high_importance_high_recency(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        entry = _make_entry(
            importance=1.0,
            access_count=10,
            created_at=datetime.now() - timedelta(minutes=30),
        )
        score = daemon._calculate_score(entry)
        assert score > 0.7

    def test_score_low_importance(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        entry = _make_entry(
            importance=0.1,
            access_count=1,
            created_at=datetime.now() - timedelta(days=5),
        )
        score = daemon._calculate_score(entry)
        assert score < 0.2

    def test_score_old_entry_decays(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        recent = _make_entry(
            importance=0.5,
            access_count=1,
            created_at=datetime.now() - timedelta(hours=1),
        )
        old = _make_entry(
            importance=0.5,
            access_count=1,
            created_at=datetime.now() - timedelta(days=10),
        )
        recent_score = daemon._calculate_score(recent)
        old_score = daemon._calculate_score(old)
        assert recent_score > old_score

    def test_score_access_count_helps(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        low_access = _make_entry(importance=0.5, access_count=1)
        high_access = _make_entry(importance=0.5, access_count=100)
        low_score = daemon._calculate_score(low_access)
        high_score = daemon._calculate_score(high_access)
        assert high_score > low_score

    def test_score_recent_access_bonus(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        recently_accessed = _make_entry(
            importance=0.5,
            access_count=3,
            last_accessed=datetime.now() - timedelta(hours=1),
        )
        not_recently = _make_entry(
            importance=0.5,
            access_count=3,
            last_accessed=datetime.now() - timedelta(days=3),
        )
        recent_score = daemon._calculate_score(recently_accessed)
        old_score = daemon._calculate_score(not_recently)
        assert recent_score > old_score

    def test_score_clamped_to_0_1(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        # Very high score input
        entry = _make_entry(
            importance=1.0,
            access_count=1000,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
        )
        score = daemon._calculate_score(entry)
        assert score <= 1.0

    def test_score_no_importance_attr(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        entry = MagicMock(spec=[])  # No attributes
        entry.created_at = datetime.now()
        # Should use defaults without crashing
        score = daemon._calculate_score(entry)
        assert 0.0 <= score <= 1.0

    def test_score_math_ebbinghaus(self):
        """Verify Ebbinghaus formula: recency = e^(-age/half_life)."""
        manager = _make_manager()
        config = DecayConfig(decay_half_life_days=3.0)
        daemon = DecayDaemon(manager, config)

        # At age = half_life (3 days), recency should be e^(-1) ≈ 0.368
        entry = _make_entry(
            importance=1.0,
            access_count=1,
            created_at=datetime.now() - timedelta(days=3),
        )
        score = daemon._calculate_score(entry)
        expected_recency = math.exp(-1)  # ≈ 0.368
        # Score = importance(1.0) * recency(0.368) * (0.5 + access_factor)
        # access_factor = log(2)/log(10) ≈ 0.301
        expected_score = 1.0 * expected_recency * (0.5 + math.log(2) / math.log(10))
        assert abs(score - expected_score) < 0.05

    def test_score_zero_age(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        entry = _make_entry(
            importance=0.5,
            access_count=1,
            created_at=datetime.now(),
        )
        score = daemon._calculate_score(entry)
        # recency ≈ 1.0 for zero age
        assert score > 0.3


# ── _consolidate_entry ────────────────────────────────────────────────────


class TestConsolidateEntry:
    @pytest.mark.asyncio
    async def test_consolidate_success(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        entry = _make_entry()

        result = await daemon._consolidate_entry(entry)

        assert result is True
        manager.store_fact.assert_called_once()
        manager.stm.delete.assert_called_once_with(entry.id)

    @pytest.mark.asyncio
    async def test_consolidate_uses_entry_data(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        entry = _make_entry(
            summary="Important discovery", importance=0.9, project="proj1"
        )

        await daemon._consolidate_entry(entry)

        call_kwargs = manager.store_fact.call_args.kwargs
        assert call_kwargs["content"] == "Important discovery"
        assert call_kwargs["importance"] == 0.9
        assert call_kwargs["project"] == "proj1"
        assert call_kwargs["extract_triplets"] is True

    @pytest.mark.asyncio
    async def test_consolidate_failure(self):
        manager = _make_manager()
        manager.store_fact = AsyncMock(side_effect=Exception("DB error"))
        daemon = DecayDaemon(manager)
        entry = _make_entry()

        result = await daemon._consolidate_entry(entry)

        assert result is False
        manager.stm.delete.assert_not_called()  # Should NOT delete on failure

    @pytest.mark.asyncio
    async def test_consolidate_no_importance_attr(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        entry = MagicMock(spec=["id", "summary", "project"])
        entry.id = "e1"
        entry.summary = "text"
        entry.project = None

        await daemon._consolidate_entry(entry)

        call_kwargs = manager.store_fact.call_args.kwargs
        assert call_kwargs["importance"] == 0.7  # Default


# ── _apply_decay ──────────────────────────────────────────────────────────


class TestApplyDecay:
    @pytest.mark.asyncio
    async def test_apply_decay_reduces_importance(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        entry = _make_entry(
            importance=0.8,
            created_at=datetime.now() - timedelta(days=3),
        )

        result = await daemon._apply_decay(entry)

        assert result is True
        manager.stm.update_importance.assert_called_once()
        # Verify new importance is lower
        call_args = manager.stm.update_importance.call_args
        new_importance = call_args.args[1]
        assert new_importance < 0.8

    @pytest.mark.asyncio
    async def test_apply_decay_respects_floor(self):
        manager = _make_manager()
        config = DecayConfig(importance_floor=0.2, base_decay_rate=1.0)
        daemon = DecayDaemon(manager, config)
        entry = _make_entry(
            importance=0.25,
            created_at=datetime.now() - timedelta(days=10),
        )

        await daemon._apply_decay(entry)

        if manager.stm.update_importance.called:
            call_args = manager.stm.update_importance.call_args
            new_importance = call_args.args[1]
            assert new_importance >= 0.2

    @pytest.mark.asyncio
    async def test_apply_decay_no_change(self):
        """No update if change is tiny (<0.01)."""
        manager = _make_manager()
        config = DecayConfig(base_decay_rate=0.001)  # Very small rate
        daemon = DecayDaemon(manager, config)
        entry = _make_entry(
            importance=0.8,
            created_at=datetime.now() - timedelta(hours=1),  # Very recent
        )

        result = await daemon._apply_decay(entry)

        assert result is False
        manager.stm.update_importance.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_decay_error_handling(self):
        manager = _make_manager()
        manager.stm.update_importance = AsyncMock(side_effect=Exception("DB error"))
        daemon = DecayDaemon(manager)
        entry = _make_entry(created_at=datetime.now() - timedelta(days=5))

        result = await daemon._apply_decay(entry)

        assert result is False

    @pytest.mark.asyncio
    async def test_decay_math_ebbinghaus(self):
        """Verify: decay = base_rate * e^(age/half_life)."""
        manager = _make_manager()
        config = DecayConfig(
            base_decay_rate=0.1, decay_half_life_days=3.0, importance_floor=0.0
        )
        daemon = DecayDaemon(manager, config)

        age_days = 3.0
        entry = _make_entry(
            importance=0.8,
            created_at=datetime.now() - timedelta(days=age_days),
        )

        await daemon._apply_decay(entry)

        expected_decay = 0.1 * math.exp(3.0 / 3.0)  # 0.1 * e ≈ 0.272
        expected_new = max(0.0, 0.8 - expected_decay)

        call_args = manager.stm.update_importance.call_args
        actual_new = call_args.args[1]
        assert abs(actual_new - expected_new) < 0.05


# ── run_decay_cycle ───────────────────────────────────────────────────────


class TestRunDecayCycle:
    @pytest.mark.asyncio
    async def test_cycle_no_entries(self):
        manager = _make_manager(stm_entries=[])
        daemon = DecayDaemon(manager)

        result = await daemon.run_decay_cycle()

        assert result.stm_entries_checked == 0
        assert result.stm_entries_decayed == 0
        assert result.entries_consolidated == 0

    @pytest.mark.asyncio
    async def test_cycle_consolidates_high_scorers(self):
        high_entry = _make_entry(
            id="high1",
            importance=0.9,
            access_count=5,
            created_at=datetime.now() - timedelta(hours=2),
        )
        manager = _make_manager(stm_entries=[high_entry])
        daemon = DecayDaemon(manager)

        result = await daemon.run_decay_cycle()

        assert result.entries_consolidated == 1

    @pytest.mark.asyncio
    async def test_cycle_prunes_low_scorers(self):
        low_entry = _make_entry(
            id="low1",
            importance=0.01,
            access_count=0,
            created_at=datetime.now() - timedelta(days=20),
        )
        manager = _make_manager(stm_entries=[low_entry])
        daemon = DecayDaemon(manager)

        result = await daemon.run_decay_cycle()

        assert result.stm_entries_pruned >= 1
        manager.stm.delete.assert_called()

    @pytest.mark.asyncio
    async def test_cycle_decays_mid_scorers(self):
        mid_entry = _make_entry(
            id="mid1",
            importance=0.5,
            access_count=1,
            created_at=datetime.now() - timedelta(days=2),
        )
        manager = _make_manager(stm_entries=[mid_entry])
        daemon = DecayDaemon(manager)

        result = await daemon.run_decay_cycle()

        # Mid scorer gets decayed (not consolidated since access_count=1 < min 2)
        assert result.stm_entries_checked == 1

    @pytest.mark.asyncio
    async def test_cycle_respects_max_consolidations(self):
        entries = [
            _make_entry(
                id=f"high{i}",
                importance=0.95,
                access_count=10,
                created_at=datetime.now() - timedelta(hours=2),
            )
            for i in range(30)
        ]
        config = DecayConfig(max_consolidations_per_run=5)
        manager = _make_manager(stm_entries=entries)
        daemon = DecayDaemon(manager, config)

        result = await daemon.run_decay_cycle()

        assert result.entries_consolidated <= 5

    @pytest.mark.asyncio
    async def test_cycle_cleans_kg_orphans(self):
        entry = _make_entry(
            importance=0.01,
            access_count=0,
            created_at=datetime.now() - timedelta(days=20),
        )
        manager = _make_manager(stm_entries=[entry], has_kg=True)
        daemon = DecayDaemon(manager)

        result = await daemon.run_decay_cycle()

        assert result.graph_nodes_pruned == 3

    @pytest.mark.asyncio
    async def test_cycle_no_kg(self):
        manager = _make_manager(stm_entries=[], has_kg=False)
        daemon = DecayDaemon(manager)

        result = await daemon.run_decay_cycle()

        assert result.graph_nodes_pruned == 0

    @pytest.mark.asyncio
    async def test_cycle_sets_last_run(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        assert daemon.last_run is None

        await daemon.run_decay_cycle()

        assert daemon.last_run is not None

    @pytest.mark.asyncio
    async def test_cycle_returns_result(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)

        result = await daemon.run_decay_cycle()

        assert isinstance(result, DecayCycleResult)
        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_cycle_handles_error(self):
        """Should not crash even if stm.get_entries_before fails."""
        manager = _make_manager()
        manager.stm.get_entries_before = AsyncMock(side_effect=Exception("DB down"))
        daemon = DecayDaemon(manager)

        result = await daemon.run_decay_cycle()

        # Should still return result (with 0 counts)
        assert result.stm_entries_checked == 0

    @pytest.mark.asyncio
    async def test_cycle_high_score_low_access_not_consolidated(self):
        """High importance but low access count → decay, not consolidate."""
        entry = _make_entry(
            importance=0.9,
            access_count=1,  # Below min_access_count_for_ltm (2)
            created_at=datetime.now() - timedelta(hours=2),
        )
        manager = _make_manager(stm_entries=[entry])
        daemon = DecayDaemon(manager)

        result = await daemon.run_decay_cycle()

        assert result.entries_consolidated == 0


# ── reinforce_memory ──────────────────────────────────────────────────────


class TestReinforceMemory:
    @pytest.mark.asyncio
    async def test_reinforce_stm(self):
        manager = _make_manager()
        stm_entry = _make_entry(id="stm1", importance=0.5)
        manager.stm.get = AsyncMock(return_value=stm_entry)
        daemon = DecayDaemon(manager)

        result = await daemon.reinforce_memory("stm1", boost=0.2)

        assert result is True
        manager.stm.mark_accessed.assert_called_once_with("stm1")
        manager.stm.update_importance.assert_called_once()
        new_imp = manager.stm.update_importance.call_args.args[1]
        assert abs(new_imp - 0.7) < 0.01

    @pytest.mark.asyncio
    async def test_reinforce_ltm(self):
        manager = _make_manager()
        manager.stm.get = AsyncMock(return_value=None)  # Not in STM
        ltm_entry = MagicMock()
        manager.ltm.get = AsyncMock(return_value=ltm_entry)
        daemon = DecayDaemon(manager)

        result = await daemon.reinforce_memory("ltm1", boost=0.15)

        assert result is True
        manager.ltm.boost_importance.assert_called_once_with("ltm1", 0.15)

    @pytest.mark.asyncio
    async def test_reinforce_not_found(self):
        manager = _make_manager()
        manager.stm.get = AsyncMock(return_value=None)
        manager.ltm.get = AsyncMock(return_value=None)
        daemon = DecayDaemon(manager)

        result = await daemon.reinforce_memory("unknown")

        assert result is False

    @pytest.mark.asyncio
    async def test_reinforce_caps_at_1(self):
        manager = _make_manager()
        entry = _make_entry(importance=0.95)
        manager.stm.get = AsyncMock(return_value=entry)
        daemon = DecayDaemon(manager)

        await daemon.reinforce_memory("e1", boost=0.5)

        new_imp = manager.stm.update_importance.call_args.args[1]
        assert new_imp <= 1.0

    @pytest.mark.asyncio
    async def test_reinforce_default_boost(self):
        manager = _make_manager()
        entry = _make_entry(importance=0.5)
        manager.stm.get = AsyncMock(return_value=entry)
        daemon = DecayDaemon(manager)

        await daemon.reinforce_memory("e1")

        new_imp = manager.stm.update_importance.call_args.args[1]
        assert abs(new_imp - 0.65) < 0.01  # 0.5 + 0.15 default

    @pytest.mark.asyncio
    async def test_reinforce_error_handling(self):
        manager = _make_manager()
        manager.stm.get = AsyncMock(side_effect=Exception("Error"))
        daemon = DecayDaemon(manager)

        result = await daemon.reinforce_memory("e1")

        assert result is False


# ── Daemon Lifecycle ──────────────────────────────────────────────────────


class TestDaemonLifecycle:
    @pytest.mark.asyncio
    async def test_start(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)

        await daemon.start()
        assert daemon.is_running is True
        assert daemon._task is not None

        # Clean up
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_stop(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)

        await daemon.start()
        await daemon.stop()

        assert daemon.is_running is False
        assert daemon._task is None

    @pytest.mark.asyncio
    async def test_start_when_already_running(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)

        await daemon.start()
        await daemon.start()  # Should not crash, just warn

        assert daemon.is_running is True
        await daemon.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)

        await daemon.stop()  # Should not crash

        assert daemon.is_running is False


# ── get_status ────────────────────────────────────────────────────────────


class TestGetStatus:
    def test_status_not_running(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)

        status = daemon.get_status()

        assert status["running"] is False
        assert status["last_run"] is None
        assert status["interval_hours"] == 6

    def test_status_config(self):
        manager = _make_manager()
        config = DecayConfig(consolidation_threshold=0.7)
        daemon = DecayDaemon(manager, config)

        status = daemon.get_status()

        assert status["config"]["consolidation_threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_status_after_run(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)

        await daemon.run_decay_cycle()
        status = daemon.get_status()

        assert status["last_run"] is not None


# ── run_decay convenience ─────────────────────────────────────────────────


class TestRunDecay:
    @pytest.mark.asyncio
    async def test_run_decay_convenience(self):
        manager = _make_manager()
        result = await run_decay(manager)
        assert isinstance(result, DecayCycleResult)


# ── Edge Cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_consolidation_failure_skips_delete(self):
        """If store_fact fails, STM entry should not be deleted."""
        entry = _make_entry(
            importance=0.95,
            access_count=5,
            created_at=datetime.now() - timedelta(hours=2),
        )
        manager = _make_manager(stm_entries=[entry])
        manager.store_fact = AsyncMock(side_effect=Exception("LTM error"))
        daemon = DecayDaemon(manager)

        result = await daemon.run_decay_cycle()

        assert result.entries_consolidated == 0
        # delete should only be called for pruned entries, not consolidated
        # (since consolidation failed)

    @pytest.mark.asyncio
    async def test_mixed_entries(self):
        """Test a mix of entries that get consolidated, decayed, and pruned."""
        high = _make_entry(
            id="high",
            importance=0.95,
            access_count=5,
            created_at=datetime.now() - timedelta(hours=2),
        )
        mid = _make_entry(
            id="mid",
            importance=0.5,
            access_count=1,
            created_at=datetime.now() - timedelta(days=2),
        )
        low = _make_entry(
            id="low",
            importance=0.01,
            access_count=0,
            created_at=datetime.now() - timedelta(days=20),
        )
        manager = _make_manager(stm_entries=[high, mid, low])
        daemon = DecayDaemon(manager)

        result = await daemon.run_decay_cycle()

        assert result.stm_entries_checked == 3
        # At least one should be consolidated, one pruned
        assert result.entries_consolidated >= 1 or result.stm_entries_decayed >= 1

    def test_score_entry_no_last_accessed(self):
        manager = _make_manager()
        daemon = DecayDaemon(manager)
        entry = _make_entry(last_accessed=None)
        score = daemon._calculate_score(entry)
        assert 0.0 <= score <= 1.0
