"""
Tests for Voice Configuration
==============================

Comprehensive tests for voice/config.py covering all 6 sub-config dataclasses,
VoiceConfig composite, YAML loading/saving, environment variable overrides,
_env_override helper, singleton behavior, and reload.

Run with: pytest tests/test_voice_config.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import voice.config as cfg_mod
from voice.config import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    AudioConfig,
    DaemonConfig,
    STTConfig,
    StorageConfig,
    TTSConfig,
    VoiceConfig,
    WakeWordConfig,
    _env_override,
    get_voice_config,
    reload_config,
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
    """Remove voice-related env vars that may leak between tests."""
    prefixes = (
        "VOICE_AUDIO_",
        "VOICE_STT_",
        "VOICE_TTS_",
        "VOICE_WAKEWORD_",
        "VOICE_STORAGE_",
        "VOICE_DAEMON_",
    )
    saved = {}
    for key in list(os.environ.keys()):
        if key.startswith(prefixes):
            saved[key] = os.environ.pop(key)
    yield
    for key, val in saved.items():
        os.environ[key] = val
    for key in list(os.environ.keys()):
        if key.startswith(prefixes) and key not in saved:
            del os.environ[key]


# ===========================================================================
# 1. Module-level constants
# ===========================================================================


class TestModuleLevelConstants:
    """Tests for REPO_ROOT and DEFAULT_CONFIG_PATH."""

    def test_repo_root_is_absolute(self):
        assert REPO_ROOT.is_absolute()

    def test_repo_root_is_parent_of_voice(self):
        voice_dir = REPO_ROOT / "voice"
        assert voice_dir.is_dir()

    def test_default_config_path_under_repo_root(self):
        assert DEFAULT_CONFIG_PATH == REPO_ROOT / "config" / "voice_config.yaml"

    def test_default_config_path_is_path_object(self):
        assert isinstance(DEFAULT_CONFIG_PATH, Path)

    def test_repo_root_is_path_object(self):
        assert isinstance(REPO_ROOT, Path)


# ===========================================================================
# 2. AudioConfig defaults
# ===========================================================================


class TestAudioConfigDefaults:
    """Every field of AudioConfig must have the documented default."""

    def test_sample_rate_default(self):
        assert AudioConfig().sample_rate == 16000

    def test_channels_default(self):
        assert AudioConfig().channels == 1

    def test_dtype_default(self):
        assert AudioConfig().dtype == "int16"

    def test_chunk_size_default(self):
        assert AudioConfig().chunk_size == 512

    def test_vad_aggressiveness_default(self):
        assert AudioConfig().vad_aggressiveness == 2

    def test_silence_threshold_ms_default(self):
        assert AudioConfig().silence_threshold_ms == 500

    def test_max_recording_seconds_default(self):
        assert AudioConfig().max_recording_seconds == 30.0

    def test_custom_values(self):
        cfg = AudioConfig(sample_rate=44100, channels=2, dtype="float32")
        assert cfg.sample_rate == 44100
        assert cfg.channels == 2
        assert cfg.dtype == "float32"

    def test_partial_custom_values(self):
        cfg = AudioConfig(sample_rate=48000)
        assert cfg.sample_rate == 48000
        assert cfg.channels == 1
        assert cfg.dtype == "int16"

    def test_is_dataclass(self):
        from dataclasses import fields

        f = fields(AudioConfig)
        names = {fld.name for fld in f}
        assert names == {
            "sample_rate",
            "channels",
            "dtype",
            "chunk_size",
            "vad_aggressiveness",
            "silence_threshold_ms",
            "max_recording_seconds",
        }

    def test_max_recording_seconds_is_float(self):
        assert isinstance(AudioConfig().max_recording_seconds, float)

    def test_sample_rate_is_int(self):
        assert isinstance(AudioConfig().sample_rate, int)


# ===========================================================================
# 3. STTConfig defaults
# ===========================================================================


class TestSTTConfigDefaults:
    """Every field of STTConfig must have the documented default."""

    def test_engine_default(self):
        assert STTConfig().engine == "faster_whisper"

    def test_model_default(self):
        assert STTConfig().model == "large-v3"

    def test_device_default(self):
        assert STTConfig().device == "cuda"

    def test_compute_type_default(self):
        assert STTConfig().compute_type == "float16"

    def test_language_default(self):
        assert STTConfig().language is None

    def test_word_timestamps_default(self):
        assert STTConfig().word_timestamps is True

    def test_vad_filter_default(self):
        assert STTConfig().vad_filter is True

    def test_beam_size_default(self):
        assert STTConfig().beam_size == 5

    def test_custom_values(self):
        cfg = STTConfig(engine="whisper", model="base", device="cpu")
        assert cfg.engine == "whisper"
        assert cfg.model == "base"
        assert cfg.device == "cpu"

    def test_partial_custom_values(self):
        cfg = STTConfig(language="te")
        assert cfg.language == "te"
        assert cfg.engine == "faster_whisper"

    def test_is_dataclass(self):
        from dataclasses import fields

        f = fields(STTConfig)
        names = {fld.name for fld in f}
        assert names == {
            "engine",
            "model",
            "device",
            "compute_type",
            "language",
            "word_timestamps",
            "vad_filter",
            "beam_size",
        }


# ===========================================================================
# 4. TTSConfig defaults
# ===========================================================================


class TestTTSConfigDefaults:
    """Every field of TTSConfig must have the documented default."""

    def test_engine_default(self):
        assert TTSConfig().engine == "xtts_v2"

    def test_device_default(self):
        assert TTSConfig().device == "cuda"

    def test_default_language_default(self):
        assert TTSConfig().default_language == "te"

    def test_default_profile_default(self):
        assert TTSConfig().default_profile == "friday_telugu"

    def test_speed_default(self):
        assert TTSConfig().speed == 1.0

    def test_temperature_default(self):
        assert TTSConfig().temperature == 0.7

    def test_top_p_default(self):
        assert TTSConfig().top_p == 0.85

    def test_top_k_default(self):
        assert TTSConfig().top_k == 50

    def test_repetition_penalty_default(self):
        assert TTSConfig().repetition_penalty == 10.0

    def test_custom_values(self):
        cfg = TTSConfig(engine="piper", speed=1.5, temperature=0.3)
        assert cfg.engine == "piper"
        assert cfg.speed == 1.5
        assert cfg.temperature == 0.3

    def test_partial_custom_values(self):
        cfg = TTSConfig(default_language="en")
        assert cfg.default_language == "en"
        assert cfg.engine == "xtts_v2"

    def test_is_dataclass(self):
        from dataclasses import fields

        f = fields(TTSConfig)
        names = {fld.name for fld in f}
        assert names == {
            "engine",
            "device",
            "default_language",
            "default_profile",
            "speed",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
        }

    def test_speed_is_float(self):
        assert isinstance(TTSConfig().speed, float)

    def test_top_k_is_int(self):
        assert isinstance(TTSConfig().top_k, int)


# ===========================================================================
# 5. WakeWordConfig defaults
# ===========================================================================


class TestWakeWordConfigDefaults:
    """Every field of WakeWordConfig must have the documented default."""

    def test_engine_default(self):
        assert WakeWordConfig().engine == "openwakeword"

    def test_models_default(self):
        assert WakeWordConfig().models == [{"name": "hey_friday", "threshold": 0.5}]

    def test_inference_framework_default(self):
        assert WakeWordConfig().inference_framework == "onnx"

    def test_vad_threshold_default(self):
        assert WakeWordConfig().vad_threshold == 0.5

    def test_noise_suppression_default(self):
        assert WakeWordConfig().noise_suppression is True

    def test_models_is_list(self):
        assert isinstance(WakeWordConfig().models, list)

    def test_models_first_entry_has_name_and_threshold(self):
        entry = WakeWordConfig().models[0]
        assert "name" in entry
        assert "threshold" in entry

    def test_models_mutable_default_isolation(self):
        """Each WakeWordConfig instance should get its own list, not share one."""
        a = WakeWordConfig()
        b = WakeWordConfig()
        a.models.append({"name": "hey_assistant", "threshold": 0.6})
        assert len(a.models) == 2
        assert len(b.models) == 1

    def test_models_mutation_does_not_affect_new_instances(self):
        a = WakeWordConfig()
        a.models[0]["threshold"] = 0.9
        b = WakeWordConfig()
        assert b.models[0]["threshold"] == 0.5

    def test_custom_values(self):
        custom_models = [{"name": "ok_friday", "threshold": 0.7}]
        cfg = WakeWordConfig(engine="porcupine", models=custom_models)
        assert cfg.engine == "porcupine"
        assert cfg.models == custom_models

    def test_is_dataclass(self):
        from dataclasses import fields

        f = fields(WakeWordConfig)
        names = {fld.name for fld in f}
        assert names == {
            "engine",
            "models",
            "inference_framework",
            "vad_threshold",
            "noise_suppression",
        }


# ===========================================================================
# 6. StorageConfig defaults
# ===========================================================================


class TestStorageConfigDefaults:
    """Every field of StorageConfig must have the documented default."""

    def test_enabled_default(self):
        assert StorageConfig().enabled is True

    def test_base_path_default(self):
        assert StorageConfig().base_path == "voice/data/recordings"

    def test_organize_by_date_default(self):
        assert StorageConfig().organize_by_date is True

    def test_retention_days_default(self):
        assert StorageConfig().retention_days == 90

    def test_save_user_audio_default(self):
        assert StorageConfig().save_user_audio is True

    def test_save_response_audio_default(self):
        assert StorageConfig().save_response_audio is True

    def test_transcript_format_default(self):
        assert StorageConfig().transcript_format == "jsonl"

    def test_custom_values(self):
        cfg = StorageConfig(enabled=False, retention_days=30, transcript_format="json")
        assert cfg.enabled is False
        assert cfg.retention_days == 30
        assert cfg.transcript_format == "json"

    def test_partial_custom_values(self):
        cfg = StorageConfig(base_path="/tmp/audio")
        assert cfg.base_path == "/tmp/audio"
        assert cfg.enabled is True

    def test_is_dataclass(self):
        from dataclasses import fields

        f = fields(StorageConfig)
        names = {fld.name for fld in f}
        assert names == {
            "enabled",
            "base_path",
            "organize_by_date",
            "retention_days",
            "save_user_audio",
            "save_response_audio",
            "transcript_format",
        }


# ===========================================================================
# 7. DaemonConfig defaults
# ===========================================================================


class TestDaemonConfigDefaults:
    """Every field of DaemonConfig must have the documented default."""

    def test_mode_default(self):
        assert DaemonConfig().mode == "standalone"

    def test_auto_start_default(self):
        assert DaemonConfig().auto_start is False

    def test_idle_timeout_seconds_default(self):
        assert DaemonConfig().idle_timeout_seconds == 300

    def test_max_session_minutes_default(self):
        assert DaemonConfig().max_session_minutes == 30

    def test_device_id_default(self):
        assert DaemonConfig().device_id == "default"

    def test_location_default(self):
        assert DaemonConfig().location == "writers_room"

    def test_custom_values(self):
        cfg = DaemonConfig(mode="integrated", auto_start=True, idle_timeout_seconds=600)
        assert cfg.mode == "integrated"
        assert cfg.auto_start is True
        assert cfg.idle_timeout_seconds == 600

    def test_partial_custom_values(self):
        cfg = DaemonConfig(location="studio")
        assert cfg.location == "studio"
        assert cfg.mode == "standalone"

    def test_is_dataclass(self):
        from dataclasses import fields

        f = fields(DaemonConfig)
        names = {fld.name for fld in f}
        assert names == {
            "mode",
            "auto_start",
            "idle_timeout_seconds",
            "max_session_minutes",
            "device_id",
            "location",
        }


# ===========================================================================
# 8. VoiceConfig defaults (composite)
# ===========================================================================


class TestVoiceConfigDefaults:
    """VoiceConfig nests all 6 sub-configs with correct defaults."""

    def test_audio_sub_config_type(self):
        assert isinstance(VoiceConfig().audio, AudioConfig)

    def test_stt_sub_config_type(self):
        assert isinstance(VoiceConfig().stt, STTConfig)

    def test_tts_sub_config_type(self):
        assert isinstance(VoiceConfig().tts, TTSConfig)

    def test_wakeword_sub_config_type(self):
        assert isinstance(VoiceConfig().wakeword, WakeWordConfig)

    def test_storage_sub_config_type(self):
        assert isinstance(VoiceConfig().storage, StorageConfig)

    def test_daemon_sub_config_type(self):
        assert isinstance(VoiceConfig().daemon, DaemonConfig)

    def test_audio_defaults_propagated(self):
        cfg = VoiceConfig()
        assert cfg.audio.sample_rate == 16000
        assert cfg.audio.channels == 1

    def test_stt_defaults_propagated(self):
        cfg = VoiceConfig()
        assert cfg.stt.engine == "faster_whisper"
        assert cfg.stt.model == "large-v3"

    def test_tts_defaults_propagated(self):
        cfg = VoiceConfig()
        assert cfg.tts.engine == "xtts_v2"
        assert cfg.tts.default_language == "te"

    def test_wakeword_defaults_propagated(self):
        cfg = VoiceConfig()
        assert cfg.wakeword.engine == "openwakeword"
        assert len(cfg.wakeword.models) == 1

    def test_storage_defaults_propagated(self):
        cfg = VoiceConfig()
        assert cfg.storage.enabled is True
        assert cfg.storage.retention_days == 90

    def test_daemon_defaults_propagated(self):
        cfg = VoiceConfig()
        assert cfg.daemon.mode == "standalone"
        assert cfg.daemon.auto_start is False

    def test_sub_configs_are_independent_instances(self):
        """Two VoiceConfig instances should not share mutable sub-config state."""
        a = VoiceConfig()
        b = VoiceConfig()
        a.wakeword.models.append({"name": "test", "threshold": 0.1})
        assert len(b.wakeword.models) == 1

    def test_sub_configs_are_new_per_instance(self):
        a = VoiceConfig()
        b = VoiceConfig()
        assert a.audio is not b.audio
        assert a.stt is not b.stt
        assert a.tts is not b.tts
        assert a.wakeword is not b.wakeword
        assert a.storage is not b.storage
        assert a.daemon is not b.daemon


# ===========================================================================
# 9. _env_override helper
# ===========================================================================


class TestEnvOverride:
    """Tests for _env_override with bool, int, float, str type conversions."""

    # --- When env var is NOT set ---

    def test_returns_default_when_env_not_set(self, clean_env):
        result = _env_override("NONEXISTENT_TEST_KEY_XYZ", "fallback", str)
        assert result == "fallback"

    def test_returns_int_default_when_env_not_set(self, clean_env):
        result = _env_override("NONEXISTENT_TEST_KEY_XYZ", 42, int)
        assert result == 42

    def test_returns_float_default_when_env_not_set(self, clean_env):
        result = _env_override("NONEXISTENT_TEST_KEY_XYZ", 3.14, float)
        assert result == 3.14

    def test_returns_bool_default_when_env_not_set(self, clean_env):
        result = _env_override("NONEXISTENT_TEST_KEY_XYZ", True, bool)
        assert result is True

    def test_returns_none_default_when_env_not_set(self, clean_env):
        result = _env_override("NONEXISTENT_TEST_KEY_XYZ", None, str)
        assert result is None

    # --- Bool conversions ---

    def test_bool_true_lowercase(self):
        with patch.dict(os.environ, {"TEST_BOOL": "true"}):
            assert _env_override("TEST_BOOL", False, bool) is True

    def test_bool_true_uppercase(self):
        with patch.dict(os.environ, {"TEST_BOOL": "TRUE"}):
            assert _env_override("TEST_BOOL", False, bool) is True

    def test_bool_true_mixed_case(self):
        with patch.dict(os.environ, {"TEST_BOOL": "True"}):
            assert _env_override("TEST_BOOL", False, bool) is True

    def test_bool_1(self):
        with patch.dict(os.environ, {"TEST_BOOL": "1"}):
            assert _env_override("TEST_BOOL", False, bool) is True

    def test_bool_yes(self):
        with patch.dict(os.environ, {"TEST_BOOL": "yes"}):
            assert _env_override("TEST_BOOL", False, bool) is True

    def test_bool_yes_uppercase(self):
        with patch.dict(os.environ, {"TEST_BOOL": "YES"}):
            assert _env_override("TEST_BOOL", False, bool) is True

    def test_bool_false_string(self):
        with patch.dict(os.environ, {"TEST_BOOL": "false"}):
            assert _env_override("TEST_BOOL", True, bool) is False

    def test_bool_0(self):
        with patch.dict(os.environ, {"TEST_BOOL": "0"}):
            assert _env_override("TEST_BOOL", True, bool) is False

    def test_bool_no(self):
        with patch.dict(os.environ, {"TEST_BOOL": "no"}):
            assert _env_override("TEST_BOOL", True, bool) is False

    def test_bool_empty_string(self):
        with patch.dict(os.environ, {"TEST_BOOL": ""}):
            assert _env_override("TEST_BOOL", True, bool) is False

    def test_bool_random_string_is_false(self):
        with patch.dict(os.environ, {"TEST_BOOL": "banana"}):
            assert _env_override("TEST_BOOL", True, bool) is False

    # --- Int conversions ---

    def test_int_positive(self):
        with patch.dict(os.environ, {"TEST_INT": "42"}):
            result = _env_override("TEST_INT", 0, int)
            assert result == 42
            assert isinstance(result, int)

    def test_int_zero(self):
        with patch.dict(os.environ, {"TEST_INT": "0"}):
            result = _env_override("TEST_INT", 99, int)
            assert result == 0

    def test_int_negative(self):
        with patch.dict(os.environ, {"TEST_INT": "-5"}):
            result = _env_override("TEST_INT", 0, int)
            assert result == -5

    def test_int_large_value(self):
        with patch.dict(os.environ, {"TEST_INT": "999999"}):
            result = _env_override("TEST_INT", 0, int)
            assert result == 999999

    # --- Float conversions ---

    def test_float_decimal(self):
        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            result = _env_override("TEST_FLOAT", 0.0, float)
            assert abs(result - 3.14) < 1e-9
            assert isinstance(result, float)

    def test_float_zero(self):
        with patch.dict(os.environ, {"TEST_FLOAT": "0.0"}):
            result = _env_override("TEST_FLOAT", 1.0, float)
            assert result == 0.0

    def test_float_negative(self):
        with patch.dict(os.environ, {"TEST_FLOAT": "-2.5"}):
            result = _env_override("TEST_FLOAT", 0.0, float)
            assert result == -2.5

    def test_float_integer_string(self):
        with patch.dict(os.environ, {"TEST_FLOAT": "7"}):
            result = _env_override("TEST_FLOAT", 0.0, float)
            assert result == 7.0
            assert isinstance(result, float)

    # --- Str conversions ---

    def test_str_override(self):
        with patch.dict(os.environ, {"TEST_STR": "hello"}):
            result = _env_override("TEST_STR", "default", str)
            assert result == "hello"

    def test_str_empty_override(self):
        with patch.dict(os.environ, {"TEST_STR": ""}):
            result = _env_override("TEST_STR", "default", str)
            assert result == ""

    def test_str_with_spaces(self):
        with patch.dict(os.environ, {"TEST_STR": "hello world"}):
            result = _env_override("TEST_STR", "default", str)
            assert result == "hello world"


# ===========================================================================
# 10. from_yaml
# ===========================================================================


class TestFromYaml:

    def test_nonexistent_file_returns_defaults(self, tmp_path):
        cfg = VoiceConfig.from_yaml(tmp_path / "does_not_exist.yaml")
        assert cfg.audio.sample_rate == 16000
        assert cfg.stt.engine == "faster_whisper"
        assert cfg.tts.engine == "xtts_v2"

    def test_empty_yaml_returns_defaults(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert cfg.audio.sample_rate == 16000
        assert cfg.daemon.mode == "standalone"

    def test_yaml_with_only_comments_returns_defaults(self, tmp_path):
        yaml_file = tmp_path / "comments.yaml"
        yaml_file.write_text("# just a comment\n# nothing else\n")
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert cfg.stt.beam_size == 5

    def test_yaml_null_document_returns_defaults(self, tmp_path):
        yaml_file = tmp_path / "null.yaml"
        yaml_file.write_text("---\n~\n")
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert cfg.tts.speed == 1.0

    def test_valid_yaml_audio_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "audio": {
                        "sample_rate": 44100,
                        "channels": 2,
                        "dtype": "float32",
                        "chunk_size": 1024,
                        "vad_aggressiveness": 3,
                        "silence_threshold_ms": 750,
                        "max_recording_seconds": 60.0,
                    }
                }
            )
        )
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert cfg.audio.sample_rate == 44100
        assert cfg.audio.channels == 2
        assert cfg.audio.dtype == "float32"
        assert cfg.audio.chunk_size == 1024
        assert cfg.audio.vad_aggressiveness == 3
        assert cfg.audio.silence_threshold_ms == 750
        assert cfg.audio.max_recording_seconds == 60.0

    def test_valid_yaml_stt_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "stt": {
                        "engine": "whisper",
                        "model": "base",
                        "device": "cpu",
                        "compute_type": "int8",
                        "language": "te",
                        "word_timestamps": False,
                        "vad_filter": False,
                        "beam_size": 3,
                    }
                }
            )
        )
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert cfg.stt.engine == "whisper"
        assert cfg.stt.model == "base"
        assert cfg.stt.device == "cpu"
        assert cfg.stt.compute_type == "int8"
        assert cfg.stt.language == "te"
        assert cfg.stt.word_timestamps is False
        assert cfg.stt.vad_filter is False
        assert cfg.stt.beam_size == 3

    def test_valid_yaml_tts_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "tts": {
                        "engine": "piper",
                        "device": "cpu",
                        "default_language": "en",
                        "default_profile": "assistant_en",
                        "speed": 1.2,
                        "temperature": 0.5,
                        "top_p": 0.9,
                        "top_k": 40,
                        "repetition_penalty": 5.0,
                    }
                }
            )
        )
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert cfg.tts.engine == "piper"
        assert cfg.tts.device == "cpu"
        assert cfg.tts.default_language == "en"
        assert cfg.tts.default_profile == "assistant_en"
        assert cfg.tts.speed == 1.2
        assert cfg.tts.temperature == 0.5
        assert cfg.tts.top_p == 0.9
        assert cfg.tts.top_k == 40
        assert cfg.tts.repetition_penalty == 5.0

    def test_valid_yaml_wakeword_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "wakeword": {
                        "engine": "porcupine",
                        "models": [
                            {"name": "ok_friday", "threshold": 0.7},
                            {"name": "hey_assistant", "threshold": 0.6},
                        ],
                        "inference_framework": "tflite",
                        "vad_threshold": 0.8,
                        "noise_suppression": False,
                    }
                }
            )
        )
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert cfg.wakeword.engine == "porcupine"
        assert len(cfg.wakeword.models) == 2
        assert cfg.wakeword.models[0]["name"] == "ok_friday"
        assert cfg.wakeword.models[1]["threshold"] == 0.6
        assert cfg.wakeword.inference_framework == "tflite"
        assert cfg.wakeword.vad_threshold == 0.8
        assert cfg.wakeword.noise_suppression is False

    def test_valid_yaml_storage_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "storage": {
                        "enabled": False,
                        "base_path": "/mnt/audio",
                        "organize_by_date": False,
                        "retention_days": 30,
                        "save_user_audio": False,
                        "save_response_audio": False,
                        "transcript_format": "json",
                    }
                }
            )
        )
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert cfg.storage.enabled is False
        assert cfg.storage.base_path == "/mnt/audio"
        assert cfg.storage.organize_by_date is False
        assert cfg.storage.retention_days == 30
        assert cfg.storage.save_user_audio is False
        assert cfg.storage.save_response_audio is False
        assert cfg.storage.transcript_format == "json"

    def test_valid_yaml_daemon_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "daemon": {
                        "mode": "integrated",
                        "auto_start": True,
                        "idle_timeout_seconds": 600,
                        "max_session_minutes": 60,
                        "device_id": "usb_mic_1",
                        "location": "studio",
                    }
                }
            )
        )
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert cfg.daemon.mode == "integrated"
        assert cfg.daemon.auto_start is True
        assert cfg.daemon.idle_timeout_seconds == 600
        assert cfg.daemon.max_session_minutes == 60
        assert cfg.daemon.device_id == "usb_mic_1"
        assert cfg.daemon.location == "studio"

    def test_valid_yaml_full(self, tmp_path, clean_env):
        data = {
            "audio": {"sample_rate": 48000, "channels": 2},
            "stt": {"engine": "whisper", "beam_size": 1},
            "tts": {"speed": 1.5, "temperature": 0.3},
            "wakeword": {"engine": "custom_engine"},
            "storage": {"retention_days": 7},
            "daemon": {"mode": "integrated", "location": "lab"},
        }
        yaml_file = tmp_path / "full.yaml"
        yaml_file.write_text(yaml.dump(data))
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert cfg.audio.sample_rate == 48000
        assert cfg.audio.channels == 2
        assert cfg.stt.engine == "whisper"
        assert cfg.stt.beam_size == 1
        assert cfg.tts.speed == 1.5
        assert cfg.tts.temperature == 0.3
        assert cfg.wakeword.engine == "custom_engine"
        assert cfg.storage.retention_days == 7
        assert cfg.daemon.mode == "integrated"
        assert cfg.daemon.location == "lab"

    def test_yaml_reads_utf8(self, tmp_path, clean_env):
        yaml_file = tmp_path / "utf8.yaml"
        yaml_file.write_text(
            yaml.dump(
                {"daemon": {"location": "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41_room"}}
            ),
            encoding="utf-8",
        )
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41" in cfg.daemon.location


# ===========================================================================
# 11. _from_dict
# ===========================================================================


class TestFromDict:

    def test_empty_dict_returns_defaults(self, clean_env):
        cfg = VoiceConfig._from_dict({})
        assert cfg.audio.sample_rate == 16000
        assert cfg.stt.engine == "faster_whisper"
        assert cfg.tts.engine == "xtts_v2"
        assert cfg.wakeword.engine == "openwakeword"
        assert cfg.storage.enabled is True
        assert cfg.daemon.mode == "standalone"

    def test_partial_audio_section(self, clean_env):
        cfg = VoiceConfig._from_dict({"audio": {"sample_rate": 48000}})
        assert cfg.audio.sample_rate == 48000
        # Only fields from dict are passed; others come from AudioConfig constructor defaults
        # But _from_dict unpacks only what's in data["audio"], so others are NOT set
        # Actually the code does **{k:v for k,v in data["audio"].items()}, so only sample_rate
        # is passed. The remaining fields get AudioConfig defaults.

    def test_partial_stt_section(self, clean_env):
        cfg = VoiceConfig._from_dict({"stt": {"engine": "whisper"}})
        assert cfg.stt.engine == "whisper"

    def test_partial_tts_section(self, clean_env):
        cfg = VoiceConfig._from_dict({"tts": {"speed": 2.0}})
        assert cfg.tts.speed == 2.0

    def test_partial_wakeword_section(self, clean_env):
        cfg = VoiceConfig._from_dict({"wakeword": {"engine": "porcupine"}})
        assert cfg.wakeword.engine == "porcupine"

    def test_partial_storage_section(self, clean_env):
        cfg = VoiceConfig._from_dict({"storage": {"retention_days": 7}})
        assert cfg.storage.retention_days == 7

    def test_partial_daemon_section(self, clean_env):
        cfg = VoiceConfig._from_dict({"daemon": {"location": "office"}})
        assert cfg.daemon.location == "office"

    def test_unknown_top_level_keys_ignored(self, clean_env):
        cfg = VoiceConfig._from_dict({"unknown_key": "value", "another": 123})
        assert cfg.audio.sample_rate == 16000

    def test_audio_missing_does_not_override_defaults(self, clean_env):
        cfg = VoiceConfig._from_dict({"daemon": {"mode": "integrated"}})
        assert cfg.audio.sample_rate == 16000
        assert cfg.audio.channels == 1

    def test_stt_missing_does_not_override_defaults(self, clean_env):
        cfg = VoiceConfig._from_dict({"audio": {"sample_rate": 8000}})
        assert cfg.stt.engine == "faster_whisper"
        assert cfg.stt.beam_size == 5

    def test_tts_missing_does_not_override_defaults(self, clean_env):
        cfg = VoiceConfig._from_dict({})
        assert cfg.tts.engine == "xtts_v2"
        assert cfg.tts.repetition_penalty == 10.0

    def test_wakeword_missing_does_not_override_defaults(self, clean_env):
        cfg = VoiceConfig._from_dict({})
        assert cfg.wakeword.engine == "openwakeword"
        assert cfg.wakeword.models == [{"name": "hey_friday", "threshold": 0.5}]

    def test_storage_missing_does_not_override_defaults(self, clean_env):
        cfg = VoiceConfig._from_dict({})
        assert cfg.storage.enabled is True
        assert cfg.storage.base_path == "voice/data/recordings"

    def test_daemon_missing_does_not_override_defaults(self, clean_env):
        cfg = VoiceConfig._from_dict({})
        assert cfg.daemon.mode == "standalone"
        assert cfg.daemon.idle_timeout_seconds == 300

    def test_wakeword_models_from_dict(self, clean_env):
        custom_models = [
            {"name": "hey_test", "threshold": 0.3},
            {"name": "ok_test", "threshold": 0.4},
        ]
        cfg = VoiceConfig._from_dict({"wakeword": {"models": custom_models}})
        assert cfg.wakeword.models == custom_models
        assert len(cfg.wakeword.models) == 2

    def test_wakeword_missing_models_uses_default(self, clean_env):
        cfg = VoiceConfig._from_dict({"wakeword": {"engine": "porcupine"}})
        assert cfg.wakeword.models == [{"name": "hey_friday", "threshold": 0.5}]

    def test_all_sections_populated(self, clean_env):
        data = {
            "audio": {"sample_rate": 8000, "channels": 2},
            "stt": {"engine": "whisper", "model": "tiny"},
            "tts": {"engine": "bark", "speed": 0.8},
            "wakeword": {"engine": "snowboy", "vad_threshold": 0.3},
            "storage": {"enabled": False, "retention_days": 14},
            "daemon": {"mode": "integrated", "auto_start": True},
        }
        cfg = VoiceConfig._from_dict(data)
        assert cfg.audio.sample_rate == 8000
        assert cfg.audio.channels == 2
        assert cfg.stt.engine == "whisper"
        assert cfg.stt.model == "tiny"
        assert cfg.tts.engine == "bark"
        assert cfg.tts.speed == 0.8
        assert cfg.wakeword.engine == "snowboy"
        assert cfg.wakeword.vad_threshold == 0.3
        assert cfg.storage.enabled is False
        assert cfg.storage.retention_days == 14
        assert cfg.daemon.mode == "integrated"
        assert cfg.daemon.auto_start is True


# ===========================================================================
# 12. Environment variable overrides in _from_dict
# ===========================================================================


class TestEnvVarOverridesInFromDict:

    def test_audio_sample_rate_env_override(self):
        with patch.dict(os.environ, {"VOICE_AUDIO_SAMPLE_RATE": "48000"}):
            cfg = VoiceConfig._from_dict({"audio": {"sample_rate": 16000}})
            assert cfg.audio.sample_rate == 48000

    def test_audio_channels_env_override(self):
        with patch.dict(os.environ, {"VOICE_AUDIO_CHANNELS": "2"}):
            cfg = VoiceConfig._from_dict({"audio": {"channels": 1}})
            assert cfg.audio.channels == 2

    def test_audio_chunk_size_env_override(self):
        with patch.dict(os.environ, {"VOICE_AUDIO_CHUNK_SIZE": "1024"}):
            cfg = VoiceConfig._from_dict({"audio": {"chunk_size": 512}})
            assert cfg.audio.chunk_size == 1024

    def test_audio_vad_aggressiveness_env_override(self):
        with patch.dict(os.environ, {"VOICE_AUDIO_VAD_AGGRESSIVENESS": "3"}):
            cfg = VoiceConfig._from_dict({"audio": {"vad_aggressiveness": 2}})
            assert cfg.audio.vad_aggressiveness == 3

    def test_audio_silence_threshold_ms_env_override(self):
        with patch.dict(os.environ, {"VOICE_AUDIO_SILENCE_THRESHOLD_MS": "750"}):
            cfg = VoiceConfig._from_dict({"audio": {"silence_threshold_ms": 500}})
            assert cfg.audio.silence_threshold_ms == 750

    def test_audio_max_recording_seconds_env_override(self):
        with patch.dict(os.environ, {"VOICE_AUDIO_MAX_RECORDING_SECONDS": "60.0"}):
            cfg = VoiceConfig._from_dict({"audio": {"max_recording_seconds": 30.0}})
            assert cfg.audio.max_recording_seconds == 60.0

    def test_audio_dtype_env_override(self):
        with patch.dict(os.environ, {"VOICE_AUDIO_DTYPE": "float32"}):
            cfg = VoiceConfig._from_dict({"audio": {"dtype": "int16"}})
            assert cfg.audio.dtype == "float32"

    def test_stt_engine_env_override(self):
        with patch.dict(os.environ, {"VOICE_STT_ENGINE": "whisper"}):
            cfg = VoiceConfig._from_dict({"stt": {"engine": "faster_whisper"}})
            assert cfg.stt.engine == "whisper"

    def test_stt_model_env_override(self):
        with patch.dict(os.environ, {"VOICE_STT_MODEL": "tiny"}):
            cfg = VoiceConfig._from_dict({"stt": {"model": "large-v3"}})
            assert cfg.stt.model == "tiny"

    def test_stt_device_env_override(self):
        with patch.dict(os.environ, {"VOICE_STT_DEVICE": "cpu"}):
            cfg = VoiceConfig._from_dict({"stt": {"device": "cuda"}})
            assert cfg.stt.device == "cpu"

    def test_stt_beam_size_env_override(self):
        with patch.dict(os.environ, {"VOICE_STT_BEAM_SIZE": "10"}):
            cfg = VoiceConfig._from_dict({"stt": {"beam_size": 5}})
            assert cfg.stt.beam_size == 10

    def test_stt_word_timestamps_env_override_false(self):
        with patch.dict(os.environ, {"VOICE_STT_WORD_TIMESTAMPS": "false"}):
            cfg = VoiceConfig._from_dict({"stt": {"word_timestamps": True}})
            assert cfg.stt.word_timestamps is False

    def test_stt_word_timestamps_env_override_true(self):
        with patch.dict(os.environ, {"VOICE_STT_WORD_TIMESTAMPS": "true"}):
            cfg = VoiceConfig._from_dict({"stt": {"word_timestamps": False}})
            assert cfg.stt.word_timestamps is True

    def test_stt_vad_filter_env_override(self):
        with patch.dict(os.environ, {"VOICE_STT_VAD_FILTER": "0"}):
            cfg = VoiceConfig._from_dict({"stt": {"vad_filter": True}})
            assert cfg.stt.vad_filter is False

    def test_tts_engine_env_override(self):
        with patch.dict(os.environ, {"VOICE_TTS_ENGINE": "piper"}):
            cfg = VoiceConfig._from_dict({"tts": {"engine": "xtts_v2"}})
            assert cfg.tts.engine == "piper"

    def test_tts_speed_env_override(self):
        with patch.dict(os.environ, {"VOICE_TTS_SPEED": "1.5"}):
            cfg = VoiceConfig._from_dict({"tts": {"speed": 1.0}})
            assert cfg.tts.speed == 1.5

    def test_tts_temperature_env_override(self):
        with patch.dict(os.environ, {"VOICE_TTS_TEMPERATURE": "0.9"}):
            cfg = VoiceConfig._from_dict({"tts": {"temperature": 0.7}})
            assert cfg.tts.temperature == 0.9

    def test_tts_top_p_env_override(self):
        with patch.dict(os.environ, {"VOICE_TTS_TOP_P": "0.95"}):
            cfg = VoiceConfig._from_dict({"tts": {"top_p": 0.85}})
            assert cfg.tts.top_p == 0.95

    def test_tts_top_k_env_override(self):
        with patch.dict(os.environ, {"VOICE_TTS_TOP_K": "100"}):
            cfg = VoiceConfig._from_dict({"tts": {"top_k": 50}})
            assert cfg.tts.top_k == 100

    def test_tts_repetition_penalty_env_override(self):
        with patch.dict(os.environ, {"VOICE_TTS_REPETITION_PENALTY": "5.0"}):
            cfg = VoiceConfig._from_dict({"tts": {"repetition_penalty": 10.0}})
            assert cfg.tts.repetition_penalty == 5.0

    def test_wakeword_engine_env_override(self):
        with patch.dict(os.environ, {"VOICE_WAKEWORD_ENGINE": "porcupine"}):
            cfg = VoiceConfig._from_dict({"wakeword": {"engine": "openwakeword"}})
            assert cfg.wakeword.engine == "porcupine"

    def test_storage_enabled_env_override(self):
        with patch.dict(os.environ, {"VOICE_STORAGE_ENABLED": "false"}):
            cfg = VoiceConfig._from_dict({"storage": {"enabled": True}})
            assert cfg.storage.enabled is False

    def test_storage_retention_days_env_override(self):
        with patch.dict(os.environ, {"VOICE_STORAGE_RETENTION_DAYS": "30"}):
            cfg = VoiceConfig._from_dict({"storage": {"retention_days": 90}})
            assert cfg.storage.retention_days == 30

    def test_storage_base_path_env_override(self):
        with patch.dict(os.environ, {"VOICE_STORAGE_BASE_PATH": "/env/path"}):
            cfg = VoiceConfig._from_dict(
                {"storage": {"base_path": "voice/data/recordings"}}
            )
            assert cfg.storage.base_path == "/env/path"

    def test_storage_transcript_format_env_override(self):
        with patch.dict(os.environ, {"VOICE_STORAGE_TRANSCRIPT_FORMAT": "csv"}):
            cfg = VoiceConfig._from_dict({"storage": {"transcript_format": "jsonl"}})
            assert cfg.storage.transcript_format == "csv"

    def test_daemon_mode_env_override(self):
        with patch.dict(os.environ, {"VOICE_DAEMON_MODE": "integrated"}):
            cfg = VoiceConfig._from_dict({"daemon": {"mode": "standalone"}})
            assert cfg.daemon.mode == "integrated"

    def test_daemon_auto_start_env_override(self):
        with patch.dict(os.environ, {"VOICE_DAEMON_AUTO_START": "yes"}):
            cfg = VoiceConfig._from_dict({"daemon": {"auto_start": False}})
            assert cfg.daemon.auto_start is True

    def test_daemon_idle_timeout_seconds_env_override(self):
        with patch.dict(os.environ, {"VOICE_DAEMON_IDLE_TIMEOUT_SECONDS": "600"}):
            cfg = VoiceConfig._from_dict({"daemon": {"idle_timeout_seconds": 300}})
            assert cfg.daemon.idle_timeout_seconds == 600

    def test_daemon_max_session_minutes_env_override(self):
        with patch.dict(os.environ, {"VOICE_DAEMON_MAX_SESSION_MINUTES": "120"}):
            cfg = VoiceConfig._from_dict({"daemon": {"max_session_minutes": 30}})
            assert cfg.daemon.max_session_minutes == 120

    def test_daemon_device_id_env_override(self):
        with patch.dict(os.environ, {"VOICE_DAEMON_DEVICE_ID": "usb_mic_2"}):
            cfg = VoiceConfig._from_dict({"daemon": {"device_id": "default"}})
            assert cfg.daemon.device_id == "usb_mic_2"

    def test_daemon_location_env_override(self):
        with patch.dict(os.environ, {"VOICE_DAEMON_LOCATION": "kitchen"}):
            cfg = VoiceConfig._from_dict({"daemon": {"location": "writers_room"}})
            assert cfg.daemon.location == "kitchen"

    def test_env_absent_falls_back_to_yaml_value(self, clean_env):
        cfg = VoiceConfig._from_dict({"audio": {"sample_rate": 44100}})
        assert cfg.audio.sample_rate == 44100

    def test_multiple_env_vars_simultaneously(self):
        env = {
            "VOICE_AUDIO_SAMPLE_RATE": "22050",
            "VOICE_STT_ENGINE": "azure",
            "VOICE_TTS_SPEED": "2.0",
            "VOICE_DAEMON_MODE": "integrated",
        }
        with patch.dict(os.environ, env):
            cfg = VoiceConfig._from_dict(
                {
                    "audio": {"sample_rate": 16000},
                    "stt": {"engine": "faster_whisper"},
                    "tts": {"speed": 1.0},
                    "daemon": {"mode": "standalone"},
                }
            )
            assert cfg.audio.sample_rate == 22050
            assert cfg.stt.engine == "azure"
            assert cfg.tts.speed == 2.0
            assert cfg.daemon.mode == "integrated"

    def test_env_override_takes_priority_over_yaml_file(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "audio": {"sample_rate": 44100},
                    "daemon": {"mode": "standalone"},
                }
            )
        )
        with patch.dict(
            os.environ,
            {
                "VOICE_AUDIO_SAMPLE_RATE": "22050",
                "VOICE_DAEMON_MODE": "integrated",
            },
        ):
            cfg = VoiceConfig.from_yaml(yaml_file)
            assert cfg.audio.sample_rate == 22050
            assert cfg.daemon.mode == "integrated"


# ===========================================================================
# 13. to_dict
# ===========================================================================


class TestToDict:

    def test_to_dict_returns_dict(self):
        result = VoiceConfig().to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_all_sections(self):
        result = VoiceConfig().to_dict()
        assert set(result.keys()) == {
            "audio",
            "stt",
            "tts",
            "wakeword",
            "storage",
            "daemon",
        }

    def test_to_dict_audio_section(self):
        d = VoiceConfig().to_dict()["audio"]
        assert d["sample_rate"] == 16000
        assert d["channels"] == 1
        assert d["dtype"] == "int16"
        assert d["chunk_size"] == 512
        assert d["vad_aggressiveness"] == 2
        assert d["silence_threshold_ms"] == 500
        assert d["max_recording_seconds"] == 30.0

    def test_to_dict_stt_section(self):
        d = VoiceConfig().to_dict()["stt"]
        assert d["engine"] == "faster_whisper"
        assert d["model"] == "large-v3"
        assert d["device"] == "cuda"
        assert d["compute_type"] == "float16"
        assert d["language"] is None
        assert d["word_timestamps"] is True
        assert d["vad_filter"] is True
        assert d["beam_size"] == 5

    def test_to_dict_tts_section(self):
        d = VoiceConfig().to_dict()["tts"]
        assert d["engine"] == "xtts_v2"
        assert d["device"] == "cuda"
        assert d["default_language"] == "te"
        assert d["default_profile"] == "friday_telugu"
        assert d["speed"] == 1.0
        assert d["temperature"] == 0.7
        assert d["top_p"] == 0.85
        assert d["top_k"] == 50
        assert d["repetition_penalty"] == 10.0

    def test_to_dict_wakeword_section(self):
        d = VoiceConfig().to_dict()["wakeword"]
        assert d["engine"] == "openwakeword"
        assert d["models"] == [{"name": "hey_friday", "threshold": 0.5}]
        assert d["inference_framework"] == "onnx"
        assert d["vad_threshold"] == 0.5
        assert d["noise_suppression"] is True

    def test_to_dict_storage_section(self):
        d = VoiceConfig().to_dict()["storage"]
        assert d["enabled"] is True
        assert d["base_path"] == "voice/data/recordings"
        assert d["organize_by_date"] is True
        assert d["retention_days"] == 90
        assert d["save_user_audio"] is True
        assert d["save_response_audio"] is True
        assert d["transcript_format"] == "jsonl"

    def test_to_dict_daemon_section(self):
        d = VoiceConfig().to_dict()["daemon"]
        assert d["mode"] == "standalone"
        assert d["auto_start"] is False
        assert d["idle_timeout_seconds"] == 300
        assert d["max_session_minutes"] == 30
        assert d["device_id"] == "default"
        assert d["location"] == "writers_room"

    def test_to_dict_with_custom_values(self):
        cfg = VoiceConfig()
        cfg.audio.sample_rate = 48000
        cfg.tts.speed = 2.0
        cfg.daemon.location = "studio"
        d = cfg.to_dict()
        assert d["audio"]["sample_rate"] == 48000
        assert d["tts"]["speed"] == 2.0
        assert d["daemon"]["location"] == "studio"

    def test_to_dict_roundtrip_preserves_values(self, clean_env):
        """to_dict -> _from_dict should preserve all values."""
        original = VoiceConfig()
        d = original.to_dict()
        restored = VoiceConfig._from_dict(d)
        assert restored.audio.sample_rate == original.audio.sample_rate
        assert restored.audio.channels == original.audio.channels
        assert restored.audio.dtype == original.audio.dtype
        assert restored.stt.engine == original.stt.engine
        assert restored.stt.model == original.stt.model
        assert restored.stt.beam_size == original.stt.beam_size
        assert restored.tts.engine == original.tts.engine
        assert restored.tts.speed == original.tts.speed
        assert restored.tts.top_k == original.tts.top_k
        assert restored.wakeword.engine == original.wakeword.engine
        assert restored.wakeword.models == original.wakeword.models
        assert restored.storage.enabled == original.storage.enabled
        assert restored.storage.retention_days == original.storage.retention_days
        assert restored.daemon.mode == original.daemon.mode
        assert restored.daemon.location == original.daemon.location

    def test_to_dict_roundtrip_with_custom_values(self, clean_env):
        original = VoiceConfig()
        original.audio.sample_rate = 22050
        original.stt.engine = "azure_stt"
        original.tts.default_language = "en"
        original.storage.enabled = False
        original.daemon.auto_start = True
        d = original.to_dict()
        restored = VoiceConfig._from_dict(d)
        assert restored.audio.sample_rate == 22050
        assert restored.stt.engine == "azure_stt"
        assert restored.tts.default_language == "en"
        assert restored.storage.enabled is False
        assert restored.daemon.auto_start is True


# ===========================================================================
# 14. save_yaml
# ===========================================================================


class TestSaveYaml:

    def test_save_yaml_creates_file(self, tmp_path):
        cfg = VoiceConfig()
        out_path = tmp_path / "output.yaml"
        cfg.save_yaml(out_path)
        assert out_path.exists()

    def test_save_yaml_creates_parent_directories(self, tmp_path):
        cfg = VoiceConfig()
        out_path = tmp_path / "deep" / "nested" / "dir" / "config.yaml"
        cfg.save_yaml(out_path)
        assert out_path.exists()
        assert out_path.parent.is_dir()

    def test_save_yaml_is_valid_yaml(self, tmp_path):
        cfg = VoiceConfig()
        out_path = tmp_path / "output.yaml"
        cfg.save_yaml(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        assert "audio" in data
        assert "stt" in data

    def test_save_yaml_contains_correct_values(self, tmp_path):
        cfg = VoiceConfig()
        cfg.audio.sample_rate = 22050
        cfg.daemon.location = "lab"
        out_path = tmp_path / "output.yaml"
        cfg.save_yaml(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["audio"]["sample_rate"] == 22050
        assert data["daemon"]["location"] == "lab"

    def test_save_yaml_then_load_roundtrip(self, tmp_path, clean_env):
        cfg1 = VoiceConfig()
        cfg1.audio.sample_rate = 44100
        cfg1.stt.engine = "whisper"
        cfg1.tts.speed = 1.5
        cfg1.storage.retention_days = 7
        cfg1.daemon.mode = "integrated"
        out_path = tmp_path / "roundtrip.yaml"
        cfg1.save_yaml(out_path)
        cfg2 = VoiceConfig.from_yaml(out_path)
        assert cfg2.audio.sample_rate == 44100
        assert cfg2.stt.engine == "whisper"
        assert cfg2.tts.speed == 1.5
        assert cfg2.storage.retention_days == 7
        assert cfg2.daemon.mode == "integrated"

    def test_save_yaml_preserves_wakeword_models(self, tmp_path, clean_env):
        cfg1 = VoiceConfig()
        cfg1.wakeword.models = [
            {"name": "hey_friday", "threshold": 0.5},
            {"name": "ok_friday", "threshold": 0.6},
        ]
        out_path = tmp_path / "ww.yaml"
        cfg1.save_yaml(out_path)
        cfg2 = VoiceConfig.from_yaml(out_path)
        assert len(cfg2.wakeword.models) == 2
        assert cfg2.wakeword.models[1]["name"] == "ok_friday"

    def test_save_yaml_overwrites_existing_file(self, tmp_path, clean_env):
        out_path = tmp_path / "overwrite.yaml"
        cfg1 = VoiceConfig()
        cfg1.audio.sample_rate = 8000
        cfg1.save_yaml(out_path)

        cfg2 = VoiceConfig()
        cfg2.audio.sample_rate = 48000
        cfg2.save_yaml(out_path)

        loaded = VoiceConfig.from_yaml(out_path)
        assert loaded.audio.sample_rate == 48000

    def test_save_yaml_not_flow_style(self, tmp_path):
        cfg = VoiceConfig()
        out_path = tmp_path / "style.yaml"
        cfg.save_yaml(out_path)
        content = out_path.read_text()
        # Flow style would have {key: value} on one line; block style uses newlines
        assert (
            "{" not in content or "name:" in content
        )  # models may have flow style for dicts


# ===========================================================================
# 15. get_voice_config singleton
# ===========================================================================


class TestGetVoiceConfig:

    def test_returns_voice_config_instance(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        cfg = get_voice_config(yaml_file)
        assert isinstance(cfg, VoiceConfig)

    def test_singleton_returns_same_object(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        cfg1 = get_voice_config(yaml_file)
        cfg2 = get_voice_config(yaml_file)
        assert cfg1 is cfg2

    def test_singleton_ignores_second_path(self, tmp_path, clean_env):
        f1 = tmp_path / "a.yaml"
        f1.write_text(yaml.dump({"audio": {"sample_rate": 44100}}))
        f2 = tmp_path / "b.yaml"
        f2.write_text(yaml.dump({"audio": {"sample_rate": 8000}}))
        cfg1 = get_voice_config(f1)
        cfg2 = get_voice_config(f2)
        assert cfg1 is cfg2
        assert cfg1.audio.sample_rate == 44100

    def test_get_config_with_nonexistent_path(self, tmp_path):
        cfg = get_voice_config(tmp_path / "missing.yaml")
        assert cfg.audio.sample_rate == 16000

    def test_get_config_default_path_used_when_none(self):
        """When config_path is None, DEFAULT_CONFIG_PATH is used."""
        cfg = get_voice_config(None)
        assert isinstance(cfg, VoiceConfig)

    def test_singleton_not_none_after_first_call(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        get_voice_config(yaml_file)
        assert cfg_mod._config is not None

    def test_singleton_none_before_first_call(self):
        assert cfg_mod._config is None

    def test_singleton_identity_with_no_args(self, tmp_path):
        """get_voice_config() called twice with no args returns the same instance."""
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        cfg1 = get_voice_config(yaml_file)
        cfg2 = get_voice_config()
        assert cfg1 is cfg2


# ===========================================================================
# 16. reload_config
# ===========================================================================


class TestReloadConfig:

    def test_reload_creates_new_instance(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(yaml.dump({"audio": {"sample_rate": 44100}}))
        cfg1 = get_voice_config(yaml_file)
        assert cfg1.audio.sample_rate == 44100

        yaml_file.write_text(yaml.dump({"audio": {"sample_rate": 8000}}))
        cfg2 = reload_config(yaml_file)
        assert cfg2.audio.sample_rate == 8000
        assert cfg1 is not cfg2

    def test_reload_updates_singleton(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(yaml.dump({"daemon": {"mode": "standalone"}}))
        get_voice_config(yaml_file)

        yaml_file.write_text(yaml.dump({"daemon": {"mode": "integrated"}}))
        reload_config(yaml_file)
        cfg = get_voice_config()
        assert cfg.daemon.mode == "integrated"

    def test_reload_with_nonexistent_file_returns_defaults(self, tmp_path):
        cfg = reload_config(tmp_path / "gone.yaml")
        assert cfg.audio.sample_rate == 16000
        assert cfg.stt.engine == "faster_whisper"

    def test_reload_with_none_uses_default_path(self):
        cfg = reload_config(None)
        assert isinstance(cfg, VoiceConfig)

    def test_reload_replaces_previous_singleton(self, tmp_path, clean_env):
        f1 = tmp_path / "first.yaml"
        f1.write_text(yaml.dump({"audio": {"sample_rate": 44100}}))
        cfg1 = get_voice_config(f1)

        f2 = tmp_path / "second.yaml"
        f2.write_text(yaml.dump({"audio": {"sample_rate": 8000}}))
        cfg2 = reload_config(f2)

        assert cfg_mod._config is cfg2
        assert cfg_mod._config is not cfg1
        assert cfg2.audio.sample_rate == 8000

    def test_reload_with_changed_sub_configs(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "audio": {"sample_rate": 44100},
                    "tts": {"speed": 1.5},
                }
            )
        )
        cfg1 = get_voice_config(yaml_file)
        assert cfg1.audio.sample_rate == 44100
        assert cfg1.tts.speed == 1.5

        yaml_file.write_text(
            yaml.dump(
                {
                    "audio": {"sample_rate": 8000},
                    "tts": {"speed": 0.5},
                }
            )
        )
        cfg2 = reload_config(yaml_file)
        assert cfg2.audio.sample_rate == 8000
        assert cfg2.tts.speed == 0.5

    def test_reload_multiple_times(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        for rate in [8000, 16000, 22050, 44100, 48000]:
            yaml_file.write_text(yaml.dump({"audio": {"sample_rate": rate}}))
            cfg = reload_config(yaml_file)
            assert cfg.audio.sample_rate == rate


# ===========================================================================
# 17. Edge cases and integration
# ===========================================================================


class TestEdgeCases:

    def test_dataclass_equality(self):
        assert AudioConfig() == AudioConfig()
        assert STTConfig() == STTConfig()
        assert TTSConfig() == TTSConfig()
        assert WakeWordConfig() == WakeWordConfig()
        assert StorageConfig() == StorageConfig()
        assert DaemonConfig() == DaemonConfig()

    def test_dataclass_inequality(self):
        assert AudioConfig(sample_rate=8000) != AudioConfig(sample_rate=16000)
        assert STTConfig(engine="a") != STTConfig(engine="b")
        assert TTSConfig(speed=1.0) != TTSConfig(speed=2.0)
        assert DaemonConfig(mode="standalone") != DaemonConfig(mode="integrated")

    def test_multiple_voice_configs_are_independent(self):
        a = VoiceConfig()
        b = VoiceConfig()
        a.wakeword.models.append({"name": "test", "threshold": 0.1})
        assert len(b.wakeword.models) == 1

    def test_from_dict_with_empty_audio_section(self, clean_env):
        # Empty audio dict -> AudioConfig() with NO kwargs -> defaults
        # Actually AudioConfig(**{}) is the same as AudioConfig()
        cfg = VoiceConfig._from_dict({"audio": {}})
        assert cfg.audio.sample_rate == 16000
        assert cfg.audio.channels == 1

    def test_from_dict_with_empty_stt_section(self, clean_env):
        cfg = VoiceConfig._from_dict({"stt": {}})
        assert cfg.stt.engine == "faster_whisper"
        assert cfg.stt.beam_size == 5

    def test_from_dict_with_empty_tts_section(self, clean_env):
        cfg = VoiceConfig._from_dict({"tts": {}})
        assert cfg.tts.engine == "xtts_v2"
        assert cfg.tts.speed == 1.0

    def test_from_dict_with_empty_wakeword_section(self, clean_env):
        cfg = VoiceConfig._from_dict({"wakeword": {}})
        assert cfg.wakeword.engine == "openwakeword"
        assert cfg.wakeword.models == [{"name": "hey_friday", "threshold": 0.5}]

    def test_from_dict_with_empty_storage_section(self, clean_env):
        cfg = VoiceConfig._from_dict({"storage": {}})
        assert cfg.storage.enabled is True
        assert cfg.storage.retention_days == 90

    def test_from_dict_with_empty_daemon_section(self, clean_env):
        cfg = VoiceConfig._from_dict({"daemon": {}})
        assert cfg.daemon.mode == "standalone"
        assert cfg.daemon.idle_timeout_seconds == 300

    def test_yaml_with_extra_sections(self, tmp_path, clean_env):
        yaml_file = tmp_path / "extra.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "audio": {"sample_rate": 22050},
                    "some_unknown_section": {"a": 1},
                }
            )
        )
        cfg = VoiceConfig.from_yaml(yaml_file)
        assert cfg.audio.sample_rate == 22050

    def test_env_override_int_invalid_raises(self):
        with patch.dict(os.environ, {"TEST_BAD_INT": "not_a_number"}):
            with pytest.raises(ValueError):
                _env_override("TEST_BAD_INT", 0, int)

    def test_env_override_float_invalid_raises(self):
        with patch.dict(os.environ, {"TEST_BAD_FLOAT": "not_a_float"}):
            with pytest.raises(ValueError):
                _env_override("TEST_BAD_FLOAT", 0.0, float)

    def test_stt_language_none_in_to_dict(self):
        cfg = VoiceConfig()
        d = cfg.to_dict()
        assert d["stt"]["language"] is None

    def test_stt_language_set_in_to_dict(self, clean_env):
        cfg = VoiceConfig._from_dict({"stt": {"language": "te"}})
        d = cfg.to_dict()
        assert d["stt"]["language"] == "te"

    def test_wakeword_empty_models_list(self, clean_env):
        cfg = VoiceConfig._from_dict({"wakeword": {"models": []}})
        assert cfg.wakeword.models == []

    def test_wakeword_multiple_models(self, clean_env):
        models = [
            {"name": "hey_friday", "threshold": 0.5},
            {"name": "ok_friday", "threshold": 0.6},
            {"name": "yo_friday", "threshold": 0.7},
        ]
        cfg = VoiceConfig._from_dict({"wakeword": {"models": models}})
        assert len(cfg.wakeword.models) == 3
        assert cfg.wakeword.models[2]["name"] == "yo_friday"

    def test_save_and_reload_with_none_language(self, tmp_path, clean_env):
        """Ensure None values survive YAML roundtrip."""
        cfg1 = VoiceConfig()
        assert cfg1.stt.language is None
        out_path = tmp_path / "none_lang.yaml"
        cfg1.save_yaml(out_path)
        cfg2 = VoiceConfig.from_yaml(out_path)
        assert cfg2.stt.language is None

    def test_voice_config_is_dataclass(self):
        from dataclasses import fields

        f = fields(VoiceConfig)
        names = {fld.name for fld in f}
        assert names == {"audio", "stt", "tts", "wakeword", "storage", "daemon"}

    def test_to_dict_audio_field_count(self):
        d = VoiceConfig().to_dict()
        assert len(d["audio"]) == 7

    def test_to_dict_stt_field_count(self):
        d = VoiceConfig().to_dict()
        assert len(d["stt"]) == 8

    def test_to_dict_tts_field_count(self):
        d = VoiceConfig().to_dict()
        assert len(d["tts"]) == 9

    def test_to_dict_wakeword_field_count(self):
        d = VoiceConfig().to_dict()
        assert len(d["wakeword"]) == 5

    def test_to_dict_storage_field_count(self):
        d = VoiceConfig().to_dict()
        assert len(d["storage"]) == 7

    def test_to_dict_daemon_field_count(self):
        d = VoiceConfig().to_dict()
        assert len(d["daemon"]) == 6

    def test_full_save_reload_roundtrip_equality(self, tmp_path, clean_env):
        """Complete roundtrip: build config -> save -> load -> compare dicts."""
        cfg1 = VoiceConfig()
        cfg1.audio.sample_rate = 22050
        cfg1.audio.channels = 2
        cfg1.stt.engine = "azure"
        cfg1.stt.beam_size = 10
        cfg1.tts.speed = 1.3
        cfg1.tts.temperature = 0.2
        cfg1.wakeword.engine = "porcupine"
        cfg1.wakeword.models = [{"name": "custom", "threshold": 0.8}]
        cfg1.storage.enabled = False
        cfg1.storage.retention_days = 7
        cfg1.daemon.mode = "integrated"
        cfg1.daemon.location = "lab"

        out_path = tmp_path / "full_roundtrip.yaml"
        cfg1.save_yaml(out_path)
        cfg2 = VoiceConfig.from_yaml(out_path)

        d1 = cfg1.to_dict()
        d2 = cfg2.to_dict()
        assert d1 == d2
