"""
Memory Decay Algorithm
======================

Brain-inspired forgetting and consolidation.

Key Concepts:
    - Ebbinghaus forgetting curve: Memories decay exponentially without reinforcement
    - Consolidation: Important STM entries graduate to LTM during "sleep"
    - Profile immunity: Identity facts never decay
    - Reinforcement: Access/mention strengthens memories

Architecture:
    DecayDaemon (background) → runs daily/hourly
        ↓
    ┌──────────────────────────────────────────────────┐
    │ 1. Score STM entries (importance + recency)       │
    │ 2. Consolidate high-scorers to LTM               │
    │ 3. Decay low-scorers (reduce importance)         │
    │ 4. Prune entries below threshold                 │
    │ 5. Update knowledge graph connections            │
    └──────────────────────────────────────────────────┘

Usage:
    daemon = DecayDaemon(memory_manager)
    await daemon.run_decay_cycle()  # Manual run
    await daemon.start()            # Background mode
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memory.manager import MemoryManager

from memory.config import get_memory_config
from memory.layers.long_term import MemoryType

LOGGER = logging.getLogger(__name__)


@dataclass
class DecayConfig:
    """Configuration for decay algorithm"""

    # Decay timing
    run_interval_hours: int = 6  # How often to run decay
    retention_days_stm: int = 7  # STM retention window

    # Scoring thresholds
    consolidation_threshold: float = 0.6  # Min score to move to LTM
    pruning_threshold: float = 0.1  # Below this, delete entry
    importance_floor: float = 0.1  # Minimum importance value

    # Decay rates (Ebbinghaus-inspired)
    base_decay_rate: float = 0.1  # Base daily decay
    decay_half_life_days: float = 3.0  # Days for 50% decay

    # Consolidation
    max_consolidations_per_run: int = 20  # Limit batch size
    min_access_count_for_ltm: int = 2  # Must be accessed 2+ times

    # Knowledge graph
    prune_orphan_nodes_days: int = 14  # Remove orphan nodes after N days

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecayConfig":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class DecayCycleResult:
    """Result from a decay cycle"""

    timestamp: datetime
    stm_entries_checked: int
    stm_entries_decayed: int
    stm_entries_pruned: int
    entries_consolidated: int
    graph_nodes_pruned: int
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "stm_checked": self.stm_entries_checked,
            "stm_decayed": self.stm_entries_decayed,
            "stm_pruned": self.stm_entries_pruned,
            "consolidated": self.entries_consolidated,
            "graph_pruned": self.graph_nodes_pruned,
            "duration_seconds": self.duration_seconds,
        }


class DecayDaemon:
    """
    Background daemon for memory decay and consolidation.

    Implements Ebbinghaus-inspired forgetting curve:
        retention = e^(-t/S)
        where t = time since creation, S = memory strength

    STM entries with high importance and frequent access
    are consolidated to LTM. Others decay and eventually
    get pruned.

    Usage:
        daemon = DecayDaemon(memory_manager)

        # Run single cycle
        result = await daemon.run_decay_cycle()

        # Start background daemon
        await daemon.start()

        # Stop daemon
        await daemon.stop()
    """

    def __init__(
        self,
        manager: "MemoryManager",
        config: Optional[DecayConfig] = None,
    ):
        self._manager = manager
        self._config = config or DecayConfig()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[datetime] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_run(self) -> Optional[datetime]:
        return self._last_run

    async def start(self) -> None:
        """Start background decay daemon"""
        if self._running:
            LOGGER.warning("Decay daemon already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._daemon_loop())
        LOGGER.info(
            "Decay daemon started (interval: %d hours)", self._config.run_interval_hours
        )

    async def stop(self) -> None:
        """Stop background daemon"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        LOGGER.info("Decay daemon stopped")

    async def _daemon_loop(self) -> None:
        """Main daemon loop"""
        while self._running:
            try:
                await self.run_decay_cycle()
            except Exception as e:
                LOGGER.error("Decay cycle failed: %s", e)

            # Sleep until next run
            await asyncio.sleep(self._config.run_interval_hours * 3600)

    async def run_decay_cycle(self) -> DecayCycleResult:
        """
        Run a complete decay cycle.

        Steps:
            1. Score all STM entries
            2. Consolidate high-scorers to LTM
            3. Apply decay to remaining entries
            4. Prune entries below threshold
            5. Clean up knowledge graph orphans

        Returns:
            DecayCycleResult with statistics
        """
        start_time = datetime.now()
        LOGGER.info("Starting decay cycle at %s", start_time.isoformat())

        stm_checked = 0
        stm_decayed = 0
        stm_pruned = 0
        consolidated = 0
        graph_pruned = 0

        try:
            # Step 1: Get all STM entries older than 1 hour (skip very recent)
            stm = self._manager.stm
            cutoff = datetime.now() - timedelta(hours=1)
            entries = await stm.get_entries_before(cutoff)
            stm_checked = len(entries)

            # Step 2: Score and categorize
            to_consolidate = []
            to_decay = []
            to_prune = []

            for entry in entries:
                score = self._calculate_score(entry)

                if score >= self._config.consolidation_threshold:
                    if entry.access_count >= self._config.min_access_count_for_ltm:
                        to_consolidate.append((entry, score))
                    else:
                        to_decay.append(entry)
                elif score <= self._config.pruning_threshold:
                    to_prune.append(entry)
                else:
                    to_decay.append(entry)

            # Step 3: Consolidate top entries to LTM
            to_consolidate.sort(key=lambda x: x[1], reverse=True)
            for entry, score in to_consolidate[
                : self._config.max_consolidations_per_run
            ]:
                success = await self._consolidate_entry(entry)
                if success:
                    consolidated += 1

            # Step 4: Apply decay
            for entry in to_decay:
                decayed = await self._apply_decay(entry)
                if decayed:
                    stm_decayed += 1

            # Step 5: Prune low-importance entries
            for entry in to_prune:
                await stm.delete(entry.id)
                stm_pruned += 1
                LOGGER.debug("Pruned STM entry: %s", entry.id[:8])

            # Step 6: Clean up knowledge graph orphans
            if self._manager._knowledge_graph:
                graph_pruned = await self._manager._knowledge_graph.cleanup_orphans(
                    min_age_days=self._config.prune_orphan_nodes_days
                )

        except Exception as e:
            LOGGER.error("Error in decay cycle: %s", e)

        duration = (datetime.now() - start_time).total_seconds()
        self._last_run = datetime.now()

        result = DecayCycleResult(
            timestamp=start_time,
            stm_entries_checked=stm_checked,
            stm_entries_decayed=stm_decayed,
            stm_entries_pruned=stm_pruned,
            entries_consolidated=consolidated,
            graph_nodes_pruned=graph_pruned,
            duration_seconds=duration,
        )

        LOGGER.info(
            "Decay cycle complete: checked=%d, decayed=%d, pruned=%d, consolidated=%d (%.2fs)",
            stm_checked,
            stm_decayed,
            stm_pruned,
            consolidated,
            duration,
        )

        return result

    def _calculate_score(self, entry) -> float:
        """
        Calculate consolidation score for STM entry.

        Score = importance * recency_factor * access_factor

        Higher score = more likely to be consolidated to LTM
        """
        now = datetime.now()

        # Base importance (0-1)
        importance = entry.importance if hasattr(entry, "importance") else 0.5

        # Recency factor (exponential decay)
        age_days = (now - entry.created_at).total_seconds() / 86400
        recency = math.exp(-age_days / self._config.decay_half_life_days)

        # Access factor (log scale)
        access_count = entry.access_count if hasattr(entry, "access_count") else 1
        access_factor = math.log(1 + access_count) / math.log(10)  # Normalize

        # Recent access bonus
        if hasattr(entry, "last_accessed") and entry.last_accessed:
            days_since_access = (now - entry.last_accessed).total_seconds() / 86400
            if days_since_access < 1:
                access_factor *= 1.5  # Bonus for very recent access

        # Combined score
        score = importance * recency * (0.5 + access_factor)

        return min(1.0, max(0.0, score))

    async def _consolidate_entry(self, entry) -> bool:
        """
        Consolidate STM entry to LTM.

        Also extracts triplets and adds to knowledge graph.
        """
        try:
            # Store to LTM
            ltm_entry = await self._manager.store_fact(
                content=entry.summary,
                memory_type=MemoryType.CONVERSATION,
                importance=entry.importance if hasattr(entry, "importance") else 0.7,
                project=entry.project,
                extract_triplets=True,  # Extract knowledge triplets
            )

            # Delete from STM after successful consolidation
            await self._manager.stm.delete(entry.id)

            LOGGER.info(
                "Consolidated STM→LTM: %s (importance: %.2f)",
                entry.id[:8],
                ltm_entry.importance,
            )
            return True

        except Exception as e:
            LOGGER.warning("Failed to consolidate entry %s: %s", entry.id[:8], e)
            return False

    async def _apply_decay(self, entry) -> bool:
        """
        Apply Ebbinghaus decay to entry importance.

        decay = base_rate * e^(age/half_life)
        new_importance = max(floor, old_importance - decay)
        """
        try:
            age_days = (datetime.now() - entry.created_at).total_seconds() / 86400

            # Ebbinghaus decay formula
            decay = self._config.base_decay_rate * math.exp(
                age_days / self._config.decay_half_life_days
            )

            old_importance = entry.importance if hasattr(entry, "importance") else 0.5
            new_importance = max(self._config.importance_floor, old_importance - decay)

            # Only update if importance changed significantly
            if abs(new_importance - old_importance) > 0.01:
                await self._manager.stm.update_importance(entry.id, new_importance)
                LOGGER.debug(
                    "Decayed entry %s: %.2f → %.2f",
                    entry.id[:8],
                    old_importance,
                    new_importance,
                )
                return True

            return False

        except Exception as e:
            LOGGER.warning("Failed to apply decay to %s: %s", entry.id[:8], e)
            return False

    async def reinforce_memory(
        self,
        memory_id: str,
        boost: float = 0.15,
    ) -> bool:
        """
        Reinforce a memory (opposite of decay).

        Called when user mentions or accesses a memory.
        This resets decay and boosts importance.
        """
        try:
            # Check STM first
            stm = self._manager.stm
            entry = await stm.get(memory_id)
            if entry:
                await stm.mark_accessed(memory_id)
                old_importance = (
                    entry.importance if hasattr(entry, "importance") else 0.5
                )
                new_importance = min(1.0, old_importance + boost)
                await stm.update_importance(memory_id, new_importance)
                LOGGER.debug(
                    "Reinforced STM: %s (%.2f → %.2f)",
                    memory_id[:8],
                    old_importance,
                    new_importance,
                )
                return True

            # Check LTM
            ltm = self._manager.ltm
            ltm_entry = await ltm.get(memory_id)
            if ltm_entry:
                await ltm.boost_importance(memory_id, boost)
                LOGGER.debug("Reinforced LTM: %s", memory_id[:8])
                return True

            return False

        except Exception as e:
            LOGGER.warning("Failed to reinforce memory %s: %s", memory_id[:8], e)
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status"""
        return {
            "running": self._running,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "interval_hours": self._config.run_interval_hours,
            "config": {
                "consolidation_threshold": self._config.consolidation_threshold,
                "pruning_threshold": self._config.pruning_threshold,
                "decay_half_life_days": self._config.decay_half_life_days,
            },
        }


# Convenience function for one-off decay runs
async def run_decay(manager: "MemoryManager") -> DecayCycleResult:
    """Run a single decay cycle"""
    daemon = DecayDaemon(manager)
    return await daemon.run_decay_cycle()
