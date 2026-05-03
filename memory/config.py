"""
Memory System Configuration
===========================

Centralized configuration for all memory layers and operations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "memory_config.yaml"
DATA_DIR = Path(__file__).parent / "data"


@dataclass
class SensoryConfig:
    """Sensory buffer configuration"""

    buffer_size_ms: int = 2000
    sample_rate: int = 16000
    vad_threshold: float = 0.5


@dataclass
class WorkingMemoryConfig:
    """Working memory (active context) configuration.

    Context Window Management:
        The working memory uses a sliding window with proactive compression
        to prevent context overflow during 24/7 operation.

        Capacity Zones:
            - Normal (<70%): No compression needed
            - Proactive (70-85%): Gentle summarization begins
            - Aggressive (85-95%): Strong compression
            - Emergency (>95%): Drop oldest content

        Buffer Allocation:
            - 20% for compressed history (summaries)
            - 60% for recent verbatim turns
            - 20% reserved for attention + LTM prefetch
    """

    max_turns: int = 10
    max_tokens: int = 4000
    attention_decay_rate: float = 0.1
    prefetch_top_k: int = 5
    max_attention_items: int = 7  # Human working memory limit

    # Context window capacity thresholds
    proactive_threshold: float = 0.70  # Start proactive summarization
    aggressive_threshold: float = 0.85  # Aggressive compression
    emergency_threshold: float = 0.95  # Emergency pruning

    # Minimum verbatim turns to keep (never compress these)
    min_verbatim_turns: int = 3

    # Context poisoning detection
    repetition_threshold: int = 3  # Flag if same content appears N times
    low_confidence_threshold: float = 0.5  # Quarantine below this


@dataclass
class STMConfig:
    """Short-term memory configuration"""

    retention_days: int = 7
    max_entries: int = 500
    consolidation_threshold: float = 0.3
    db_path: str = str(DATA_DIR / "stm.db")


@dataclass
class LTMConfig:
    """Long-term memory configuration"""

    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    embedding_dim: int = 768
    vector_search_top_k: int = 10
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "friday_memory"
    db_user: str = "friday"
    db_password: str = ""
    use_sqlite_fallback: bool = True  # Use SQLite if Postgres unavailable
    sqlite_path: str = str(DATA_DIR / "ltm.db")


@dataclass
class ProfileConfig:
    """Profile store configuration"""

    profile_path: str = str(DATA_DIR / "profile" / "current.json")
    history_path: str = str(DATA_DIR / "profile" / "history")
    max_history_versions: int = 100


@dataclass
class DecayConfig:
    """Decay algorithm configuration"""

    run_interval_hours: int = 1
    decay_threshold: float = 0.4
    archive_threshold: float = 0.2
    delete_threshold: float = 0.05
    recency_decay_rate: float = 0.1  # Per day

    # Scoring weights (should sum to ~1.0)
    weight_recency: float = 0.30
    weight_frequency: float = 0.15
    weight_importance: float = 0.30
    weight_event: float = 0.15
    weight_profile: float = 0.10

    # Type-specific bonuses
    type_bonus_preference: float = 0.2
    type_bonus_decision: float = 0.15
    type_bonus_fact: float = 0.1
    type_bonus_pattern: float = 0.1


@dataclass
class ConsolidationConfig:
    """Consolidation daemon configuration"""

    run_time: str = "03:00"  # 3 AM daily
    similarity_threshold: float = 0.8
    min_memories_to_merge: int = 2
    max_memories_to_merge: int = 10


@dataclass
class BackupConfig:
    """Backup configuration"""

    interval_hours: int = 6
    retention_days: int = 30
    local_path: str = str(DATA_DIR / "backups")
    s3_bucket: Optional[str] = None
    s3_prefix: str = "friday-memory/"


@dataclass
class HealthConfig:
    """Health monitoring configuration"""

    check_interval_seconds: int = 60
    storage_alert_threshold: float = 0.90
    query_latency_alert_ms: int = 2000
    daemon_restart_threshold: int = 3


@dataclass
class TeluguConfig:
    """Telugu-English processing configuration"""

    stopwords_file: str = str(Path(__file__).parent / "telugu" / "stopwords.txt")
    keyword_min_length: int = 2
    high_density_threshold: float = 0.4
    medium_density_threshold: float = 0.15


@dataclass
class VoiceConfig:
    """Voice command configuration"""

    confirmation_required: list = field(
        default_factory=lambda: ["memory_delete", "profile_static_update"]
    )
    languages: list = field(default_factory=lambda: ["en", "te"])


@dataclass
class MemorySystemConfig:
    """Complete memory system configuration"""

    sensory: SensoryConfig = field(default_factory=SensoryConfig)
    working: WorkingMemoryConfig = field(default_factory=WorkingMemoryConfig)
    stm: STMConfig = field(default_factory=STMConfig)
    ltm: LTMConfig = field(default_factory=LTMConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    telugu: TeluguConfig = field(default_factory=TeluguConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)

    # GLM router for fact extraction
    glm_api_key: str = ""
    glm_base_url: str = "https://api.z.ai/api/paas/v4"

    # Logging
    log_level: str = "INFO"
    log_path: str = str(DATA_DIR / "logs" / "memory.log")

    @classmethod
    def from_yaml(cls, path: Path) -> "MemorySystemConfig":
        """Load configuration from YAML file"""
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "MemorySystemConfig":
        """Create config from dictionary"""
        config = cls()

        # Sensory
        if "sensory" in data:
            s = data["sensory"]
            config.sensory = SensoryConfig(
                buffer_size_ms=s.get("buffer_size_ms", 2000),
                sample_rate=s.get("sample_rate", 16000),
                vad_threshold=s.get("vad_threshold", 0.5),
            )

        # Working memory
        if "working" in data:
            w = data["working"]
            config.working = WorkingMemoryConfig(
                max_turns=w.get("max_turns", 10),
                max_tokens=w.get("max_tokens", 4000),
                attention_decay_rate=w.get("attention_decay_rate", 0.1),
                prefetch_top_k=w.get("prefetch_top_k", 5),
            )

        # STM
        if "short_term" in data:
            s = data["short_term"]
            config.stm = STMConfig(
                retention_days=s.get("retention_days", 7),
                max_entries=s.get("max_entries", 500),
                consolidation_threshold=s.get("consolidation_threshold", 0.3),
            )

        # LTM
        if "long_term" in data:
            l = data["long_term"]
            config.ltm = LTMConfig(
                embedding_model=l.get("embedding_model", config.ltm.embedding_model),
                embedding_dim=l.get("embedding_dim", 768),
                vector_search_top_k=l.get("vector_search_top_k", 10),
                db_host=_env_or_default(
                    "MEMORY_DB_HOST", l.get("db_host", "localhost")
                ),
                db_port=int(_env_or_default("MEMORY_DB_PORT", l.get("db_port", 5432))),
                db_name=l.get("db_name", "friday_memory"),
                db_user=l.get("db_user", "friday"),
                db_password=_env_or_default(
                    "MEMORY_DB_PASSWORD", l.get("db_password", "")
                ),
            )

        # Decay
        if "decay" in data:
            d = data["decay"]
            config.decay = DecayConfig(
                run_interval_hours=d.get("run_interval_hours", 1),
                decay_threshold=d.get("thresholds", {}).get("decay", 0.4),
                archive_threshold=d.get("thresholds", {}).get("archive", 0.2),
                delete_threshold=d.get("thresholds", {}).get("delete", 0.05),
                recency_decay_rate=d.get("recency_decay_rate", 0.1),
                weight_recency=d.get("weights", {}).get("recency", 0.30),
                weight_frequency=d.get("weights", {}).get("frequency", 0.15),
                weight_importance=d.get("weights", {}).get("importance", 0.30),
                weight_event=d.get("weights", {}).get("event", 0.15),
                weight_profile=d.get("weights", {}).get("profile", 0.10),
            )

        # Consolidation
        if "consolidation" in data:
            c = data["consolidation"]
            config.consolidation = ConsolidationConfig(
                run_time=c.get("run_time", "03:00"),
                similarity_threshold=c.get("similarity_threshold", 0.8),
                min_memories_to_merge=c.get("min_memories_to_merge", 2),
                max_memories_to_merge=c.get("max_memories_to_merge", 10),
            )

        # Backup
        if "backup" in data:
            b = data["backup"]
            config.backup = BackupConfig(
                interval_hours=b.get("interval_hours", 6),
                retention_days=b.get("retention_days", 30),
                local_path=b.get("local_path", str(DATA_DIR / "backups")),
                s3_bucket=b.get("s3_bucket"),
                s3_prefix=b.get("s3_prefix", "friday-memory/"),
            )

        # GLM settings
        config.glm_api_key = _env_or_default(
            "ZHIPU_API_KEY", data.get("glm_api_key", "")
        )
        config.glm_base_url = data.get("glm_base_url", "https://api.z.ai/api/paas/v4")

        # Logging
        config.log_level = data.get("log_level", "INFO")

        return config


def _env_or_default(env_key: str, default: Any) -> Any:
    """Get from environment or return default"""
    return os.environ.get(env_key, default)


# Singleton instance
_config: Optional[MemorySystemConfig] = None


def get_memory_config(config_path: Optional[Path] = None) -> MemorySystemConfig:
    """Get memory configuration singleton"""
    global _config

    if _config is None:
        path = config_path or DEFAULT_CONFIG_PATH
        _config = MemorySystemConfig.from_yaml(path)

    return _config


def reload_memory_config(config_path: Optional[Path] = None) -> MemorySystemConfig:
    """Reload memory configuration"""
    global _config
    path = config_path or DEFAULT_CONFIG_PATH
    _config = MemorySystemConfig.from_yaml(path)
    return _config


# Alias for simpler import
MemoryConfig = MemorySystemConfig
