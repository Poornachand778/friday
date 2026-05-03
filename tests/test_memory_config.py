"""
Tests for Memory System Configuration
======================================

Comprehensive tests for memory/config.py covering all 12 dataclasses,
YAML loading, environment variable overrides, singleton behavior,
reload, and alias.

Run with: pytest tests/test_memory_config.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory.config as cfg_mod
from memory.config import (
    DATA_DIR,
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    BackupConfig,
    ConsolidationConfig,
    DecayConfig,
    HealthConfig,
    LTMConfig,
    MemoryConfig,
    MemorySystemConfig,
    ProfileConfig,
    STMConfig,
    SensoryConfig,
    TeluguConfig,
    VoiceConfig,
    WorkingMemoryConfig,
    _env_or_default,
    get_memory_config,
    reload_memory_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before and after every test."""
    cfg_mod._config = None
    yield
    cfg_mod._config = None


@pytest.fixture()
def clean_env():
    """Remove memory-related env vars that may leak between tests."""
    keys = [
        "MEMORY_DB_HOST",
        "MEMORY_DB_PORT",
        "MEMORY_DB_PASSWORD",
        "ZHIPU_API_KEY",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ===========================================================================
# 1. Module-level constants
# ===========================================================================


class TestModuleLevelConstants:
    """Tests for REPO_ROOT, DEFAULT_CONFIG_PATH, DATA_DIR."""

    def test_repo_root_is_absolute(self):
        assert REPO_ROOT.is_absolute()

    def test_repo_root_is_parent_of_memory(self):
        memory_dir = REPO_ROOT / "memory"
        assert memory_dir.is_dir()

    def test_default_config_path_under_repo_root(self):
        assert DEFAULT_CONFIG_PATH == REPO_ROOT / "config" / "memory_config.yaml"

    def test_data_dir_is_inside_memory(self):
        assert "memory" in str(DATA_DIR)
        assert DATA_DIR.name == "data"


# ===========================================================================
# 2. SensoryConfig defaults
# ===========================================================================


class TestSensoryConfigDefaults:
    """Every field of SensoryConfig must have the documented default."""

    def test_buffer_size_ms_default(self):
        assert SensoryConfig().buffer_size_ms == 2000

    def test_sample_rate_default(self):
        assert SensoryConfig().sample_rate == 16000

    def test_vad_threshold_default(self):
        assert SensoryConfig().vad_threshold == 0.5

    def test_custom_values(self):
        cfg = SensoryConfig(buffer_size_ms=5000, sample_rate=44100, vad_threshold=0.8)
        assert cfg.buffer_size_ms == 5000
        assert cfg.sample_rate == 44100
        assert cfg.vad_threshold == 0.8

    def test_partial_custom_values(self):
        cfg = SensoryConfig(buffer_size_ms=3000)
        assert cfg.buffer_size_ms == 3000
        assert cfg.sample_rate == 16000
        assert cfg.vad_threshold == 0.5

    def test_is_dataclass(self):
        from dataclasses import fields

        f = fields(SensoryConfig)
        names = {fld.name for fld in f}
        assert names == {"buffer_size_ms", "sample_rate", "vad_threshold"}


# ===========================================================================
# 3. WorkingMemoryConfig defaults
# ===========================================================================


class TestWorkingMemoryConfigDefaults:
    """Every field of WorkingMemoryConfig must have the documented default."""

    def test_max_turns_default(self):
        assert WorkingMemoryConfig().max_turns == 10

    def test_max_tokens_default(self):
        assert WorkingMemoryConfig().max_tokens == 4000

    def test_attention_decay_rate_default(self):
        assert WorkingMemoryConfig().attention_decay_rate == 0.1

    def test_prefetch_top_k_default(self):
        assert WorkingMemoryConfig().prefetch_top_k == 5

    def test_max_attention_items_default(self):
        assert WorkingMemoryConfig().max_attention_items == 7

    def test_proactive_threshold_default(self):
        assert WorkingMemoryConfig().proactive_threshold == 0.70

    def test_aggressive_threshold_default(self):
        assert WorkingMemoryConfig().aggressive_threshold == 0.85

    def test_emergency_threshold_default(self):
        assert WorkingMemoryConfig().emergency_threshold == 0.95

    def test_min_verbatim_turns_default(self):
        assert WorkingMemoryConfig().min_verbatim_turns == 3

    def test_repetition_threshold_default(self):
        assert WorkingMemoryConfig().repetition_threshold == 3

    def test_low_confidence_threshold_default(self):
        assert WorkingMemoryConfig().low_confidence_threshold == 0.5

    def test_custom_values(self):
        cfg = WorkingMemoryConfig(max_turns=20, max_tokens=8000)
        assert cfg.max_turns == 20
        assert cfg.max_tokens == 8000
        assert cfg.attention_decay_rate == 0.1  # untouched


# ===========================================================================
# 4. STMConfig defaults
# ===========================================================================


class TestSTMConfigDefaults:
    """Every field of STMConfig must have the documented default."""

    def test_retention_days_default(self):
        assert STMConfig().retention_days == 7

    def test_max_entries_default(self):
        assert STMConfig().max_entries == 500

    def test_consolidation_threshold_default(self):
        assert STMConfig().consolidation_threshold == 0.3

    def test_db_path_default(self):
        assert STMConfig().db_path == str(DATA_DIR / "stm.db")

    def test_db_path_is_string(self):
        assert isinstance(STMConfig().db_path, str)

    def test_custom_values(self):
        cfg = STMConfig(retention_days=14, max_entries=1000, db_path="/tmp/test.db")
        assert cfg.retention_days == 14
        assert cfg.max_entries == 1000
        assert cfg.db_path == "/tmp/test.db"


# ===========================================================================
# 5. LTMConfig defaults
# ===========================================================================


class TestLTMConfigDefaults:
    """Every field of LTMConfig must have the documented default."""

    def test_embedding_model_default(self):
        assert LTMConfig().embedding_model == "paraphrase-multilingual-mpnet-base-v2"

    def test_embedding_dim_default(self):
        assert LTMConfig().embedding_dim == 768

    def test_vector_search_top_k_default(self):
        assert LTMConfig().vector_search_top_k == 10

    def test_db_host_default(self):
        assert LTMConfig().db_host == "localhost"

    def test_db_port_default(self):
        assert LTMConfig().db_port == 5432

    def test_db_name_default(self):
        assert LTMConfig().db_name == "friday_memory"

    def test_db_user_default(self):
        assert LTMConfig().db_user == "friday"

    def test_db_password_default(self):
        assert LTMConfig().db_password == ""

    def test_use_sqlite_fallback_default(self):
        assert LTMConfig().use_sqlite_fallback is True

    def test_sqlite_path_default(self):
        assert LTMConfig().sqlite_path == str(DATA_DIR / "ltm.db")

    def test_custom_values(self):
        cfg = LTMConfig(db_host="db.example.com", db_port=5433, db_password="secret")
        assert cfg.db_host == "db.example.com"
        assert cfg.db_port == 5433
        assert cfg.db_password == "secret"
        assert cfg.embedding_dim == 768  # untouched


# ===========================================================================
# 6. ProfileConfig defaults
# ===========================================================================


class TestProfileConfigDefaults:
    """Every field of ProfileConfig must have the documented default."""

    def test_profile_path_default(self):
        assert ProfileConfig().profile_path == str(
            DATA_DIR / "profile" / "current.json"
        )

    def test_history_path_default(self):
        assert ProfileConfig().history_path == str(DATA_DIR / "profile" / "history")

    def test_max_history_versions_default(self):
        assert ProfileConfig().max_history_versions == 100

    def test_custom_values(self):
        cfg = ProfileConfig(profile_path="/tmp/p.json", max_history_versions=50)
        assert cfg.profile_path == "/tmp/p.json"
        assert cfg.max_history_versions == 50


# ===========================================================================
# 7. DecayConfig defaults
# ===========================================================================


class TestDecayConfigDefaults:
    """Every field of DecayConfig must have the documented default."""

    def test_run_interval_hours_default(self):
        assert DecayConfig().run_interval_hours == 1

    def test_decay_threshold_default(self):
        assert DecayConfig().decay_threshold == 0.4

    def test_archive_threshold_default(self):
        assert DecayConfig().archive_threshold == 0.2

    def test_delete_threshold_default(self):
        assert DecayConfig().delete_threshold == 0.05

    def test_recency_decay_rate_default(self):
        assert DecayConfig().recency_decay_rate == 0.1

    def test_weight_recency_default(self):
        assert DecayConfig().weight_recency == 0.30

    def test_weight_frequency_default(self):
        assert DecayConfig().weight_frequency == 0.15

    def test_weight_importance_default(self):
        assert DecayConfig().weight_importance == 0.30

    def test_weight_event_default(self):
        assert DecayConfig().weight_event == 0.15

    def test_weight_profile_default(self):
        assert DecayConfig().weight_profile == 0.10

    def test_weights_sum_to_one(self):
        d = DecayConfig()
        total = (
            d.weight_recency
            + d.weight_frequency
            + d.weight_importance
            + d.weight_event
            + d.weight_profile
        )
        assert abs(total - 1.0) < 1e-9

    def test_type_bonus_preference_default(self):
        assert DecayConfig().type_bonus_preference == 0.2

    def test_type_bonus_decision_default(self):
        assert DecayConfig().type_bonus_decision == 0.15

    def test_type_bonus_fact_default(self):
        assert DecayConfig().type_bonus_fact == 0.1

    def test_type_bonus_pattern_default(self):
        assert DecayConfig().type_bonus_pattern == 0.1

    def test_threshold_ordering(self):
        d = DecayConfig()
        assert d.delete_threshold < d.archive_threshold < d.decay_threshold


# ===========================================================================
# 8. ConsolidationConfig defaults
# ===========================================================================


class TestConsolidationConfigDefaults:
    """Every field of ConsolidationConfig must have the documented default."""

    def test_run_time_default(self):
        assert ConsolidationConfig().run_time == "03:00"

    def test_similarity_threshold_default(self):
        assert ConsolidationConfig().similarity_threshold == 0.8

    def test_min_memories_to_merge_default(self):
        assert ConsolidationConfig().min_memories_to_merge == 2

    def test_max_memories_to_merge_default(self):
        assert ConsolidationConfig().max_memories_to_merge == 10

    def test_merge_range_valid(self):
        c = ConsolidationConfig()
        assert c.min_memories_to_merge <= c.max_memories_to_merge

    def test_custom_values(self):
        cfg = ConsolidationConfig(run_time="05:30", similarity_threshold=0.9)
        assert cfg.run_time == "05:30"
        assert cfg.similarity_threshold == 0.9


# ===========================================================================
# 9. BackupConfig defaults
# ===========================================================================


class TestBackupConfigDefaults:
    """Every field of BackupConfig must have the documented default."""

    def test_interval_hours_default(self):
        assert BackupConfig().interval_hours == 6

    def test_retention_days_default(self):
        assert BackupConfig().retention_days == 30

    def test_local_path_default(self):
        assert BackupConfig().local_path == str(DATA_DIR / "backups")

    def test_s3_bucket_default(self):
        assert BackupConfig().s3_bucket is None

    def test_s3_prefix_default(self):
        assert BackupConfig().s3_prefix == "friday-memory/"

    def test_custom_s3_values(self):
        cfg = BackupConfig(s3_bucket="my-bucket", s3_prefix="custom/")
        assert cfg.s3_bucket == "my-bucket"
        assert cfg.s3_prefix == "custom/"


# ===========================================================================
# 10. HealthConfig defaults
# ===========================================================================


class TestHealthConfigDefaults:
    """Every field of HealthConfig must have the documented default."""

    def test_check_interval_seconds_default(self):
        assert HealthConfig().check_interval_seconds == 60

    def test_storage_alert_threshold_default(self):
        assert HealthConfig().storage_alert_threshold == 0.90

    def test_query_latency_alert_ms_default(self):
        assert HealthConfig().query_latency_alert_ms == 2000

    def test_daemon_restart_threshold_default(self):
        assert HealthConfig().daemon_restart_threshold == 3

    def test_custom_values(self):
        cfg = HealthConfig(check_interval_seconds=30, daemon_restart_threshold=5)
        assert cfg.check_interval_seconds == 30
        assert cfg.daemon_restart_threshold == 5


# ===========================================================================
# 11. TeluguConfig defaults
# ===========================================================================


class TestTeluguConfigDefaults:
    """Every field of TeluguConfig must have the documented default."""

    def test_stopwords_file_default(self):
        expected = str(Path(cfg_mod.__file__).parent / "telugu" / "stopwords.txt")
        assert TeluguConfig().stopwords_file == expected

    def test_keyword_min_length_default(self):
        assert TeluguConfig().keyword_min_length == 2

    def test_high_density_threshold_default(self):
        assert TeluguConfig().high_density_threshold == 0.4

    def test_medium_density_threshold_default(self):
        assert TeluguConfig().medium_density_threshold == 0.15

    def test_density_threshold_ordering(self):
        t = TeluguConfig()
        assert t.medium_density_threshold < t.high_density_threshold


# ===========================================================================
# 12. VoiceConfig defaults
# ===========================================================================


class TestVoiceConfigDefaults:
    """Every field of VoiceConfig must have the documented default."""

    def test_confirmation_required_default(self):
        assert VoiceConfig().confirmation_required == [
            "memory_delete",
            "profile_static_update",
        ]

    def test_languages_default(self):
        assert VoiceConfig().languages == ["en", "te"]

    def test_confirmation_required_is_list(self):
        assert isinstance(VoiceConfig().confirmation_required, list)

    def test_languages_is_list(self):
        assert isinstance(VoiceConfig().languages, list)

    def test_default_factory_creates_separate_lists(self):
        """Ensure each instance gets its own list (not shared)."""
        a = VoiceConfig()
        b = VoiceConfig()
        a.languages.append("hi")
        assert "hi" not in b.languages

    def test_confirmation_required_factory_creates_separate_lists(self):
        a = VoiceConfig()
        b = VoiceConfig()
        a.confirmation_required.append("test_action")
        assert "test_action" not in b.confirmation_required


# ===========================================================================
# 13. MemorySystemConfig defaults
# ===========================================================================


class TestMemorySystemConfigDefaults:

    def test_sensory_sub_config_default(self):
        cfg = MemorySystemConfig()
        assert isinstance(cfg.sensory, SensoryConfig)
        assert cfg.sensory.buffer_size_ms == 2000

    def test_working_sub_config_default(self):
        cfg = MemorySystemConfig()
        assert isinstance(cfg.working, WorkingMemoryConfig)
        assert cfg.working.max_turns == 10

    def test_stm_sub_config_default(self):
        cfg = MemorySystemConfig()
        assert isinstance(cfg.stm, STMConfig)
        assert cfg.stm.retention_days == 7

    def test_ltm_sub_config_default(self):
        cfg = MemorySystemConfig()
        assert isinstance(cfg.ltm, LTMConfig)
        assert cfg.ltm.db_host == "localhost"

    def test_profile_sub_config_default(self):
        cfg = MemorySystemConfig()
        assert isinstance(cfg.profile, ProfileConfig)

    def test_decay_sub_config_default(self):
        cfg = MemorySystemConfig()
        assert isinstance(cfg.decay, DecayConfig)
        assert cfg.decay.decay_threshold == 0.4

    def test_consolidation_sub_config_default(self):
        cfg = MemorySystemConfig()
        assert isinstance(cfg.consolidation, ConsolidationConfig)
        assert cfg.consolidation.run_time == "03:00"

    def test_backup_sub_config_default(self):
        cfg = MemorySystemConfig()
        assert isinstance(cfg.backup, BackupConfig)
        assert cfg.backup.interval_hours == 6

    def test_health_sub_config_default(self):
        cfg = MemorySystemConfig()
        assert isinstance(cfg.health, HealthConfig)
        assert cfg.health.check_interval_seconds == 60

    def test_telugu_sub_config_default(self):
        cfg = MemorySystemConfig()
        assert isinstance(cfg.telugu, TeluguConfig)

    def test_voice_sub_config_default(self):
        cfg = MemorySystemConfig()
        assert isinstance(cfg.voice, VoiceConfig)
        assert cfg.voice.languages == ["en", "te"]

    def test_glm_api_key_default(self):
        assert MemorySystemConfig().glm_api_key == ""

    def test_glm_base_url_default(self):
        assert MemorySystemConfig().glm_base_url == "https://api.z.ai/api/paas/v4"

    def test_log_level_default(self):
        assert MemorySystemConfig().log_level == "INFO"

    def test_log_path_default(self):
        assert MemorySystemConfig().log_path == str(DATA_DIR / "logs" / "memory.log")

    def test_sub_configs_are_independent_instances(self):
        """Two MemorySystemConfig instances should not share mutable sub-config state."""
        a = MemorySystemConfig()
        b = MemorySystemConfig()
        a.voice.languages.append("hi")
        assert "hi" not in b.voice.languages


# ===========================================================================
# 14. from_yaml
# ===========================================================================


class TestFromYaml:

    def test_nonexistent_file_returns_defaults(self, tmp_path):
        cfg = MemorySystemConfig.from_yaml(tmp_path / "does_not_exist.yaml")
        assert cfg.log_level == "INFO"
        assert cfg.sensory.buffer_size_ms == 2000
        assert cfg.working.max_turns == 10

    def test_empty_yaml_returns_defaults(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.log_level == "INFO"
        assert cfg.ltm.db_host == "localhost"

    def test_yaml_with_only_comments_returns_defaults(self, tmp_path):
        yaml_file = tmp_path / "comments.yaml"
        yaml_file.write_text("# just a comment\n# nothing else\n")
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.working.max_tokens == 4000

    def test_yaml_null_document_returns_defaults(self, tmp_path):
        yaml_file = tmp_path / "null.yaml"
        yaml_file.write_text("---\n~\n")
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.log_level == "INFO"

    def test_valid_yaml_sensory_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "sensory": {
                        "buffer_size_ms": 4000,
                        "sample_rate": 44100,
                        "vad_threshold": 0.7,
                    }
                }
            )
        )
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.sensory.buffer_size_ms == 4000
        assert cfg.sensory.sample_rate == 44100
        assert cfg.sensory.vad_threshold == 0.7

    def test_valid_yaml_working_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "working": {
                        "max_turns": 20,
                        "max_tokens": 8000,
                        "attention_decay_rate": 0.2,
                        "prefetch_top_k": 10,
                    }
                }
            )
        )
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.working.max_turns == 20
        assert cfg.working.max_tokens == 8000
        assert cfg.working.attention_decay_rate == 0.2
        assert cfg.working.prefetch_top_k == 10

    def test_valid_yaml_short_term_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "short_term": {
                        "retention_days": 14,
                        "max_entries": 1000,
                        "consolidation_threshold": 0.5,
                    }
                }
            )
        )
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.stm.retention_days == 14
        assert cfg.stm.max_entries == 1000
        assert cfg.stm.consolidation_threshold == 0.5

    def test_valid_yaml_long_term_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "long_term": {
                        "embedding_model": "custom-model",
                        "embedding_dim": 512,
                        "vector_search_top_k": 20,
                        "db_host": "db.example.com",
                        "db_port": 5433,
                        "db_name": "test_db",
                        "db_user": "test_user",
                        "db_password": "test_pass",
                    }
                }
            )
        )
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.ltm.embedding_model == "custom-model"
        assert cfg.ltm.embedding_dim == 512
        assert cfg.ltm.vector_search_top_k == 20
        assert cfg.ltm.db_host == "db.example.com"
        assert cfg.ltm.db_port == 5433
        assert cfg.ltm.db_name == "test_db"
        assert cfg.ltm.db_user == "test_user"
        assert cfg.ltm.db_password == "test_pass"

    def test_valid_yaml_decay_section_with_nested(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "decay": {
                        "run_interval_hours": 2,
                        "recency_decay_rate": 0.2,
                        "thresholds": {
                            "decay": 0.5,
                            "archive": 0.3,
                            "delete": 0.1,
                        },
                        "weights": {
                            "recency": 0.25,
                            "frequency": 0.20,
                            "importance": 0.25,
                            "event": 0.20,
                            "profile": 0.10,
                        },
                    }
                }
            )
        )
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.decay.run_interval_hours == 2
        assert cfg.decay.recency_decay_rate == 0.2
        assert cfg.decay.decay_threshold == 0.5
        assert cfg.decay.archive_threshold == 0.3
        assert cfg.decay.delete_threshold == 0.1
        assert cfg.decay.weight_recency == 0.25
        assert cfg.decay.weight_frequency == 0.20
        assert cfg.decay.weight_importance == 0.25
        assert cfg.decay.weight_event == 0.20
        assert cfg.decay.weight_profile == 0.10

    def test_valid_yaml_consolidation_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "consolidation": {
                        "run_time": "05:30",
                        "similarity_threshold": 0.9,
                        "min_memories_to_merge": 3,
                        "max_memories_to_merge": 15,
                    }
                }
            )
        )
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.consolidation.run_time == "05:30"
        assert cfg.consolidation.similarity_threshold == 0.9
        assert cfg.consolidation.min_memories_to_merge == 3
        assert cfg.consolidation.max_memories_to_merge == 15

    def test_valid_yaml_backup_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "backup": {
                        "interval_hours": 12,
                        "retention_days": 60,
                        "local_path": "/tmp/backups",
                        "s3_bucket": "my-bucket",
                        "s3_prefix": "custom-prefix/",
                    }
                }
            )
        )
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.backup.interval_hours == 12
        assert cfg.backup.retention_days == 60
        assert cfg.backup.local_path == "/tmp/backups"
        assert cfg.backup.s3_bucket == "my-bucket"
        assert cfg.backup.s3_prefix == "custom-prefix/"

    def test_valid_yaml_glm_and_logging(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "glm_api_key": "test-key-123",
                    "glm_base_url": "https://custom.api/v1",
                    "log_level": "DEBUG",
                }
            )
        )
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.glm_api_key == "test-key-123"
        assert cfg.glm_base_url == "https://custom.api/v1"
        assert cfg.log_level == "DEBUG"

    def test_valid_yaml_full(self, tmp_path, clean_env):
        data = {
            "sensory": {"buffer_size_ms": 3000},
            "working": {"max_turns": 15},
            "short_term": {"retention_days": 10},
            "long_term": {"db_host": "remote.host", "db_port": 5433},
            "decay": {
                "run_interval_hours": 3,
                "thresholds": {"decay": 0.6, "archive": 0.3, "delete": 0.08},
                "weights": {
                    "recency": 0.20,
                    "frequency": 0.20,
                    "importance": 0.20,
                    "event": 0.20,
                    "profile": 0.20,
                },
            },
            "consolidation": {"run_time": "04:00"},
            "backup": {"interval_hours": 24},
            "glm_api_key": "full-key",
            "log_level": "WARNING",
        }
        yaml_file = tmp_path / "full.yaml"
        yaml_file.write_text(yaml.dump(data))
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.sensory.buffer_size_ms == 3000
        assert cfg.working.max_turns == 15
        assert cfg.stm.retention_days == 10
        assert cfg.ltm.db_host == "remote.host"
        assert cfg.ltm.db_port == 5433
        assert cfg.decay.run_interval_hours == 3
        assert cfg.decay.decay_threshold == 0.6
        assert cfg.decay.weight_recency == 0.20
        assert cfg.consolidation.run_time == "04:00"
        assert cfg.backup.interval_hours == 24
        assert cfg.glm_api_key == "full-key"
        assert cfg.log_level == "WARNING"

    def test_yaml_reads_utf8(self, tmp_path, clean_env):
        yaml_file = tmp_path / "utf8.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "glm_base_url": "https://api.example.com/\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41"
                }
            ),
            encoding="utf-8",
        )
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41" in cfg.glm_base_url


# ===========================================================================
# 15. _from_dict
# ===========================================================================


class TestFromDict:

    def test_empty_dict_returns_defaults(self, clean_env):
        cfg = MemorySystemConfig._from_dict({})
        assert cfg.sensory.buffer_size_ms == 2000
        assert cfg.working.max_turns == 10
        assert cfg.log_level == "INFO"

    def test_partial_sensory_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"sensory": {"buffer_size_ms": 5000}})
        assert cfg.sensory.buffer_size_ms == 5000
        assert cfg.sensory.sample_rate == 16000
        assert cfg.sensory.vad_threshold == 0.5

    def test_partial_working_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"working": {"max_turns": 25}})
        assert cfg.working.max_turns == 25
        assert cfg.working.max_tokens == 4000

    def test_partial_short_term_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"short_term": {"retention_days": 30}})
        assert cfg.stm.retention_days == 30
        assert cfg.stm.max_entries == 500

    def test_partial_long_term_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"long_term": {"db_name": "custom_db"}})
        assert cfg.ltm.db_name == "custom_db"
        assert cfg.ltm.db_host == "localhost"

    def test_partial_decay_section_with_empty_thresholds(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"decay": {"run_interval_hours": 4}})
        assert cfg.decay.run_interval_hours == 4
        assert cfg.decay.decay_threshold == 0.4  # default from .get fallback
        assert cfg.decay.archive_threshold == 0.2
        assert cfg.decay.delete_threshold == 0.05

    def test_partial_decay_section_with_thresholds_only(self, clean_env):
        cfg = MemorySystemConfig._from_dict(
            {
                "decay": {
                    "thresholds": {"decay": 0.6, "archive": 0.35},
                }
            }
        )
        assert cfg.decay.decay_threshold == 0.6
        assert cfg.decay.archive_threshold == 0.35
        assert cfg.decay.delete_threshold == 0.05  # default fallback

    def test_partial_decay_section_with_weights_only(self, clean_env):
        cfg = MemorySystemConfig._from_dict(
            {
                "decay": {
                    "weights": {"recency": 0.40, "importance": 0.40},
                }
            }
        )
        assert cfg.decay.weight_recency == 0.40
        assert cfg.decay.weight_importance == 0.40
        assert cfg.decay.weight_frequency == 0.15  # default fallback
        assert cfg.decay.weight_event == 0.15
        assert cfg.decay.weight_profile == 0.10

    def test_partial_consolidation_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"consolidation": {"run_time": "06:00"}})
        assert cfg.consolidation.run_time == "06:00"
        assert cfg.consolidation.similarity_threshold == 0.8

    def test_partial_backup_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"backup": {"s3_bucket": "test-bucket"}})
        assert cfg.backup.s3_bucket == "test-bucket"
        assert cfg.backup.interval_hours == 6

    def test_backup_section_no_s3_bucket(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"backup": {"interval_hours": 12}})
        assert cfg.backup.s3_bucket is None

    def test_glm_api_key_from_dict(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"glm_api_key": "dict-key"})
        assert cfg.glm_api_key == "dict-key"

    def test_glm_base_url_from_dict(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"glm_base_url": "https://custom.api/v2"})
        assert cfg.glm_base_url == "https://custom.api/v2"

    def test_log_level_from_dict(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"log_level": "ERROR"})
        assert cfg.log_level == "ERROR"

    def test_unknown_keys_are_ignored(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"unknown_key": "value", "another": 123})
        assert cfg.log_level == "INFO"

    def test_unknown_keys_in_subsection_are_ignored(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"sensory": {"unknown_sub_key": True}})
        assert cfg.sensory.buffer_size_ms == 2000

    def test_all_sections_populated(self, clean_env):
        data = {
            "sensory": {"buffer_size_ms": 1000},
            "working": {"max_turns": 5},
            "short_term": {"retention_days": 3},
            "long_term": {"db_host": "10.0.0.1"},
            "decay": {
                "run_interval_hours": 2,
                "thresholds": {"decay": 0.5, "archive": 0.25, "delete": 0.08},
                "weights": {
                    "recency": 0.20,
                    "frequency": 0.20,
                    "importance": 0.20,
                    "event": 0.20,
                    "profile": 0.20,
                },
            },
            "consolidation": {"run_time": "02:00"},
            "backup": {"interval_hours": 1, "s3_bucket": "bucket"},
            "glm_api_key": "all-sections-key",
            "glm_base_url": "https://all.api/v1",
            "log_level": "DEBUG",
        }
        cfg = MemorySystemConfig._from_dict(data)
        assert cfg.sensory.buffer_size_ms == 1000
        assert cfg.working.max_turns == 5
        assert cfg.stm.retention_days == 3
        assert cfg.ltm.db_host == "10.0.0.1"
        assert cfg.decay.run_interval_hours == 2
        assert cfg.decay.decay_threshold == 0.5
        assert cfg.decay.weight_recency == 0.20
        assert cfg.consolidation.run_time == "02:00"
        assert cfg.backup.interval_hours == 1
        assert cfg.backup.s3_bucket == "bucket"
        assert cfg.glm_api_key == "all-sections-key"
        assert cfg.glm_base_url == "https://all.api/v1"
        assert cfg.log_level == "DEBUG"

    def test_long_term_embedding_model_uses_default_from_config(self, clean_env):
        """When embedding_model is absent from dict, should use the config's own default."""
        cfg = MemorySystemConfig._from_dict({"long_term": {"db_host": "other"}})
        assert cfg.ltm.embedding_model == "paraphrase-multilingual-mpnet-base-v2"

    def test_sensory_missing_does_not_override_defaults(self, clean_env):
        """When sensory section is absent, sensory sub-config stays at defaults."""
        cfg = MemorySystemConfig._from_dict({"log_level": "DEBUG"})
        assert cfg.sensory.buffer_size_ms == 2000
        assert cfg.sensory.sample_rate == 16000

    def test_working_missing_does_not_override_defaults(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"log_level": "DEBUG"})
        assert cfg.working.max_turns == 10
        assert cfg.working.max_tokens == 4000

    def test_decay_missing_does_not_override_defaults(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"log_level": "DEBUG"})
        assert cfg.decay.decay_threshold == 0.4
        assert cfg.decay.weight_recency == 0.30

    def test_consolidation_missing_does_not_override_defaults(self, clean_env):
        cfg = MemorySystemConfig._from_dict({})
        assert cfg.consolidation.run_time == "03:00"

    def test_backup_missing_does_not_override_defaults(self, clean_env):
        cfg = MemorySystemConfig._from_dict({})
        assert cfg.backup.interval_hours == 6
        assert cfg.backup.s3_bucket is None


# ===========================================================================
# 16. Environment variable overrides
# ===========================================================================


class TestEnvVarOverrides:

    def test_memory_db_host_override(self):
        with patch.dict(os.environ, {"MEMORY_DB_HOST": "env-host.example.com"}):
            cfg = MemorySystemConfig._from_dict({"long_term": {"db_host": "yaml-host"}})
            assert cfg.ltm.db_host == "env-host.example.com"

    def test_memory_db_port_override(self):
        with patch.dict(os.environ, {"MEMORY_DB_PORT": "9999"}):
            cfg = MemorySystemConfig._from_dict({"long_term": {"db_port": 5432}})
            assert cfg.ltm.db_port == 9999

    def test_memory_db_port_override_is_int(self):
        with patch.dict(os.environ, {"MEMORY_DB_PORT": "7777"}):
            cfg = MemorySystemConfig._from_dict({"long_term": {}})
            assert isinstance(cfg.ltm.db_port, int)

    def test_memory_db_password_override(self):
        with patch.dict(os.environ, {"MEMORY_DB_PASSWORD": "env-secret"}):
            cfg = MemorySystemConfig._from_dict(
                {"long_term": {"db_password": "yaml-pass"}}
            )
            assert cfg.ltm.db_password == "env-secret"

    def test_zhipu_api_key_override(self):
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "zhipu-env-key"}):
            cfg = MemorySystemConfig._from_dict({"glm_api_key": "yaml-key"})
            assert cfg.glm_api_key == "zhipu-env-key"

    def test_zhipu_api_key_override_without_yaml(self):
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "zhipu-env-key-no-yaml"}):
            cfg = MemorySystemConfig._from_dict({})
            assert cfg.glm_api_key == "zhipu-env-key-no-yaml"

    def test_env_does_not_affect_when_absent(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"long_term": {"db_host": "yaml-host"}})
        assert cfg.ltm.db_host == "yaml-host"

    def test_env_db_host_absent_falls_back_to_yaml(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"long_term": {"db_host": "yaml-host"}})
        assert cfg.ltm.db_host == "yaml-host"

    def test_env_db_password_absent_falls_back_to_yaml(self, clean_env):
        cfg = MemorySystemConfig._from_dict(
            {"long_term": {"db_password": "yaml-password"}}
        )
        assert cfg.ltm.db_password == "yaml-password"

    def test_env_zhipu_absent_falls_back_to_yaml(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"glm_api_key": "yaml-glm-key"})
        assert cfg.glm_api_key == "yaml-glm-key"

    def test_multiple_env_vars_simultaneously(self):
        env = {
            "MEMORY_DB_HOST": "env-host",
            "MEMORY_DB_PORT": "6543",
            "MEMORY_DB_PASSWORD": "env-pass",
            "ZHIPU_API_KEY": "env-zhipu",
        }
        with patch.dict(os.environ, env):
            cfg = MemorySystemConfig._from_dict(
                {
                    "long_term": {
                        "db_host": "yaml-host",
                        "db_port": 5432,
                        "db_password": "yaml-pass",
                    },
                    "glm_api_key": "yaml-glm",
                }
            )
            assert cfg.ltm.db_host == "env-host"
            assert cfg.ltm.db_port == 6543
            assert cfg.ltm.db_password == "env-pass"
            assert cfg.glm_api_key == "env-zhipu"

    def test_env_override_takes_priority_over_yaml_file(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "long_term": {"db_host": "yaml-host", "db_password": "yaml-pass"},
                    "glm_api_key": "yaml-key",
                }
            )
        )
        with patch.dict(
            os.environ,
            {
                "MEMORY_DB_HOST": "env-host",
                "MEMORY_DB_PASSWORD": "env-pass",
                "ZHIPU_API_KEY": "env-key",
            },
        ):
            cfg = MemorySystemConfig.from_yaml(yaml_file)
            assert cfg.ltm.db_host == "env-host"
            assert cfg.ltm.db_password == "env-pass"
            assert cfg.glm_api_key == "env-key"

    def test_env_empty_string_is_still_returned_for_password(self):
        with patch.dict(os.environ, {"MEMORY_DB_PASSWORD": ""}):
            cfg = MemorySystemConfig._from_dict(
                {"long_term": {"db_password": "yaml-pass"}}
            )
            assert cfg.ltm.db_password == ""

    def test_env_empty_string_is_still_returned_for_zhipu(self):
        with patch.dict(os.environ, {"ZHIPU_API_KEY": ""}):
            cfg = MemorySystemConfig._from_dict({"glm_api_key": "yaml-key"})
            assert cfg.glm_api_key == ""


# ===========================================================================
# 17. _env_or_default helper
# ===========================================================================


class TestEnvOrDefault:

    def test_returns_default_when_env_not_set(self, clean_env):
        result = _env_or_default("NONEXISTENT_TEST_KEY_XYZ_12345", "fallback")
        assert result == "fallback"

    def test_returns_env_value_when_set(self):
        with patch.dict(os.environ, {"MY_TEST_KEY_MEM": "from_env"}):
            result = _env_or_default("MY_TEST_KEY_MEM", "fallback")
            assert result == "from_env"

    def test_env_overrides_none_default(self):
        with patch.dict(os.environ, {"MY_TEST_KEY_MEM": "val"}):
            result = _env_or_default("MY_TEST_KEY_MEM", None)
            assert result == "val"

    def test_returns_none_default_when_env_not_set(self, clean_env):
        result = _env_or_default("NONEXISTENT_TEST_KEY_XYZ_12345", None)
        assert result is None

    def test_env_empty_string_is_still_returned(self):
        with patch.dict(os.environ, {"MY_TEST_KEY_MEM": ""}):
            result = _env_or_default("MY_TEST_KEY_MEM", "fallback")
            assert result == ""

    def test_returns_integer_default_when_env_not_set(self, clean_env):
        result = _env_or_default("NONEXISTENT_TEST_KEY_XYZ_12345", 42)
        assert result == 42

    def test_returns_string_even_when_default_is_int(self):
        """os.environ.get always returns a string, so env value replaces int default."""
        with patch.dict(os.environ, {"MY_TEST_KEY_MEM": "123"}):
            result = _env_or_default("MY_TEST_KEY_MEM", 42)
            assert result == "123"
            assert isinstance(result, str)


# ===========================================================================
# 18. get_memory_config singleton
# ===========================================================================


class TestGetMemoryConfig:

    def test_returns_memory_system_config_instance(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        cfg = get_memory_config(yaml_file)
        assert isinstance(cfg, MemorySystemConfig)

    def test_singleton_returns_same_object(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        cfg1 = get_memory_config(yaml_file)
        cfg2 = get_memory_config(yaml_file)
        assert cfg1 is cfg2

    def test_singleton_ignores_second_path(self, tmp_path, clean_env):
        f1 = tmp_path / "a.yaml"
        f1.write_text(yaml.dump({"log_level": "DEBUG"}))
        f2 = tmp_path / "b.yaml"
        f2.write_text(yaml.dump({"log_level": "ERROR"}))
        cfg1 = get_memory_config(f1)
        cfg2 = get_memory_config(f2)
        assert cfg1 is cfg2
        assert cfg1.log_level == "DEBUG"

    def test_get_config_with_nonexistent_path(self, tmp_path):
        cfg = get_memory_config(tmp_path / "missing.yaml")
        assert cfg.log_level == "INFO"

    def test_get_config_default_path_used_when_none(self):
        """When config_path is None, DEFAULT_CONFIG_PATH is used."""
        cfg = get_memory_config(None)
        assert isinstance(cfg, MemorySystemConfig)

    def test_singleton_not_none_after_first_call(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        get_memory_config(yaml_file)
        assert cfg_mod._config is not None

    def test_singleton_none_before_first_call(self):
        assert cfg_mod._config is None


# ===========================================================================
# 19. reload_memory_config
# ===========================================================================


class TestReloadMemoryConfig:

    def test_reload_creates_new_instance(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(yaml.dump({"log_level": "DEBUG"}))
        cfg1 = get_memory_config(yaml_file)
        assert cfg1.log_level == "DEBUG"

        yaml_file.write_text(yaml.dump({"log_level": "ERROR"}))
        cfg2 = reload_memory_config(yaml_file)
        assert cfg2.log_level == "ERROR"
        assert cfg1 is not cfg2

    def test_reload_updates_singleton(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(yaml.dump({"log_level": "INFO"}))
        get_memory_config(yaml_file)

        yaml_file.write_text(yaml.dump({"log_level": "WARNING"}))
        reload_memory_config(yaml_file)
        cfg = get_memory_config()
        assert cfg.log_level == "WARNING"

    def test_reload_with_nonexistent_file_returns_defaults(self, tmp_path):
        cfg = reload_memory_config(tmp_path / "gone.yaml")
        assert cfg.log_level == "INFO"
        assert cfg.sensory.buffer_size_ms == 2000

    def test_reload_with_none_uses_default_path(self):
        cfg = reload_memory_config(None)
        assert isinstance(cfg, MemorySystemConfig)

    def test_reload_replaces_previous_singleton(self, tmp_path, clean_env):
        f1 = tmp_path / "first.yaml"
        f1.write_text(yaml.dump({"log_level": "DEBUG"}))
        cfg1 = get_memory_config(f1)

        f2 = tmp_path / "second.yaml"
        f2.write_text(yaml.dump({"log_level": "CRITICAL"}))
        cfg2 = reload_memory_config(f2)

        assert cfg_mod._config is cfg2
        assert cfg_mod._config is not cfg1
        assert cfg2.log_level == "CRITICAL"

    def test_reload_with_changed_sub_configs(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "sensory": {"buffer_size_ms": 1000},
                    "working": {"max_turns": 5},
                }
            )
        )
        cfg1 = get_memory_config(yaml_file)
        assert cfg1.sensory.buffer_size_ms == 1000
        assert cfg1.working.max_turns == 5

        yaml_file.write_text(
            yaml.dump(
                {
                    "sensory": {"buffer_size_ms": 9000},
                    "working": {"max_turns": 50},
                }
            )
        )
        cfg2 = reload_memory_config(yaml_file)
        assert cfg2.sensory.buffer_size_ms == 9000
        assert cfg2.working.max_turns == 50


# ===========================================================================
# 20. MemoryConfig alias
# ===========================================================================


class TestMemoryConfigAlias:

    def test_alias_is_same_class(self):
        assert MemoryConfig is MemorySystemConfig

    def test_alias_creates_valid_instance(self):
        cfg = MemoryConfig()
        assert isinstance(cfg, MemorySystemConfig)
        assert cfg.log_level == "INFO"

    def test_alias_from_yaml_works(self, tmp_path):
        yaml_file = tmp_path / "alias.yaml"
        yaml_file.write_text(yaml.dump({"log_level": "DEBUG"}))
        cfg = MemoryConfig.from_yaml(yaml_file)
        assert cfg.log_level == "DEBUG"

    def test_alias_from_dict_works(self, clean_env):
        cfg = MemoryConfig._from_dict({"log_level": "ERROR"})
        assert cfg.log_level == "ERROR"


# ===========================================================================
# 21. Edge cases and integration
# ===========================================================================


class TestEdgeCases:

    def test_from_dict_with_empty_sensory_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"sensory": {}})
        assert cfg.sensory.buffer_size_ms == 2000
        assert cfg.sensory.sample_rate == 16000

    def test_from_dict_with_empty_working_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"working": {}})
        assert cfg.working.max_turns == 10
        assert cfg.working.max_tokens == 4000

    def test_from_dict_with_empty_short_term_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"short_term": {}})
        assert cfg.stm.retention_days == 7
        assert cfg.stm.max_entries == 500

    def test_from_dict_with_empty_long_term_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"long_term": {}})
        assert cfg.ltm.db_host == "localhost"
        assert cfg.ltm.db_port == 5432
        assert cfg.ltm.embedding_model == "paraphrase-multilingual-mpnet-base-v2"

    def test_from_dict_with_empty_decay_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"decay": {}})
        assert cfg.decay.run_interval_hours == 1
        assert cfg.decay.decay_threshold == 0.4
        assert cfg.decay.weight_recency == 0.30

    def test_from_dict_with_empty_consolidation_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"consolidation": {}})
        assert cfg.consolidation.run_time == "03:00"

    def test_from_dict_with_empty_backup_section(self, clean_env):
        cfg = MemorySystemConfig._from_dict({"backup": {}})
        assert cfg.backup.interval_hours == 6
        assert cfg.backup.s3_bucket is None

    def test_yaml_with_extra_sections(self, tmp_path, clean_env):
        yaml_file = tmp_path / "extra.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "sensory": {"buffer_size_ms": 999},
                    "some_unknown_section": {"a": 1},
                }
            )
        )
        cfg = MemorySystemConfig.from_yaml(yaml_file)
        assert cfg.sensory.buffer_size_ms == 999

    def test_multiple_configs_are_independent(self):
        a = MemorySystemConfig()
        b = MemorySystemConfig()
        a.voice.languages.append("hi")
        assert "hi" not in b.voice.languages

    def test_dataclass_equality(self):
        assert SensoryConfig() == SensoryConfig()
        assert WorkingMemoryConfig() == WorkingMemoryConfig()
        assert STMConfig() == STMConfig()
        assert LTMConfig() == LTMConfig()
        assert DecayConfig() == DecayConfig()
        assert ConsolidationConfig() == ConsolidationConfig()
        assert BackupConfig() == BackupConfig()
        assert HealthConfig() == HealthConfig()
        assert TeluguConfig() == TeluguConfig()

    def test_dataclass_inequality(self):
        assert SensoryConfig(buffer_size_ms=1) != SensoryConfig(buffer_size_ms=2)
        assert LTMConfig(db_host="a") != LTMConfig(db_host="b")

    def test_decay_empty_thresholds_dict_uses_defaults(self, clean_env):
        cfg = MemorySystemConfig._from_dict(
            {"decay": {"thresholds": {}, "weights": {}}}
        )
        assert cfg.decay.decay_threshold == 0.4
        assert cfg.decay.archive_threshold == 0.2
        assert cfg.decay.delete_threshold == 0.05
        assert cfg.decay.weight_recency == 0.30
        assert cfg.decay.weight_frequency == 0.15

    def test_long_term_db_port_int_cast_from_env(self):
        """Port from env is a string; _from_dict casts it to int."""
        with patch.dict(os.environ, {"MEMORY_DB_PORT": "3333"}):
            cfg = MemorySystemConfig._from_dict({"long_term": {}})
            assert cfg.ltm.db_port == 3333
            assert isinstance(cfg.ltm.db_port, int)

    def test_long_term_use_sqlite_fallback_not_overridden_by_from_dict(self, clean_env):
        """The _from_dict for long_term doesn't set use_sqlite_fallback, so it stays True."""
        cfg = MemorySystemConfig._from_dict({"long_term": {"db_host": "other"}})
        # use_sqlite_fallback is not parsed in _from_dict, so the LTMConfig default is overridden
        # with a new LTMConfig that uses default True for use_sqlite_fallback
        assert cfg.ltm.use_sqlite_fallback is True

    def test_from_dict_preserves_health_defaults(self, clean_env):
        """Health config is never parsed by _from_dict so it stays at defaults."""
        cfg = MemorySystemConfig._from_dict({"log_level": "DEBUG"})
        assert cfg.health.check_interval_seconds == 60
        assert cfg.health.storage_alert_threshold == 0.90
        assert cfg.health.query_latency_alert_ms == 2000
        assert cfg.health.daemon_restart_threshold == 3

    def test_from_dict_preserves_telugu_defaults(self, clean_env):
        """Telugu config is never parsed by _from_dict so it stays at defaults."""
        cfg = MemorySystemConfig._from_dict({})
        assert cfg.telugu.keyword_min_length == 2
        assert cfg.telugu.high_density_threshold == 0.4

    def test_from_dict_preserves_voice_defaults(self, clean_env):
        """Voice config is never parsed by _from_dict so it stays at defaults."""
        cfg = MemorySystemConfig._from_dict({})
        assert cfg.voice.confirmation_required == [
            "memory_delete",
            "profile_static_update",
        ]
        assert cfg.voice.languages == ["en", "te"]

    def test_from_dict_preserves_profile_defaults(self, clean_env):
        """Profile config is never parsed by _from_dict so it stays at defaults."""
        cfg = MemorySystemConfig._from_dict({})
        assert cfg.profile.max_history_versions == 100

    def test_sensory_section_replaces_entire_sub_config(self, clean_env):
        """When a section is present in data, the entire sub-config is replaced."""
        cfg = MemorySystemConfig._from_dict({"sensory": {"buffer_size_ms": 9999}})
        assert cfg.sensory.buffer_size_ms == 9999
        # Other fields are from the SensoryConfig constructor defaults, not from the old instance
        assert cfg.sensory.sample_rate == 16000

    def test_working_section_only_sets_four_fields(self, clean_env):
        """_from_dict only passes max_turns, max_tokens, attention_decay_rate, prefetch_top_k."""
        cfg = MemorySystemConfig._from_dict(
            {
                "working": {
                    "max_turns": 100,
                    "max_tokens": 100,
                    "attention_decay_rate": 0.9,
                    "prefetch_top_k": 99,
                }
            }
        )
        assert cfg.working.max_turns == 100
        assert cfg.working.max_tokens == 100
        assert cfg.working.attention_decay_rate == 0.9
        assert cfg.working.prefetch_top_k == 99
        # These revert to WorkingMemoryConfig defaults since not passed
        assert cfg.working.max_attention_items == 7
        assert cfg.working.proactive_threshold == 0.70
        assert cfg.working.aggressive_threshold == 0.85
        assert cfg.working.emergency_threshold == 0.95

    def test_glm_base_url_defaults_when_absent(self, clean_env):
        cfg = MemorySystemConfig._from_dict({})
        assert cfg.glm_base_url == "https://api.z.ai/api/paas/v4"

    def test_glm_api_key_defaults_to_empty_when_absent(self, clean_env):
        cfg = MemorySystemConfig._from_dict({})
        assert cfg.glm_api_key == ""
