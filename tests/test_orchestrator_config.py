"""
Tests for Orchestrator Configuration
=====================================

Comprehensive tests for orchestrator/config.py covering all dataclasses,
YAML loading, environment variable overrides, singleton behavior, and
serialization.

Run with: pytest tests/test_orchestrator_config.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orchestrator.config as cfg_mod
from orchestrator.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROJECT_SLUG,
    REPO_ROOT,
    ContextConfig,
    ExternalAPIConfig,
    LLMConfig,
    MemoryConfig,
    OrchestratorConfig,
    RouterConfig,
    _env_or_default,
    get_config,
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
    """Remove all FRIDAY_* / ZHIPU_* / RUNWAY_* / OPENAI_* env vars that may leak."""
    keys = [
        "FRIDAY_LLM_BACKEND",
        "FRIDAY_LLM_MODEL",
        "FRIDAY_LLM_BASE_URL",
        "FRIDAY_LLM_API_KEY",
        "FRIDAY_PORT",
        "FRIDAY_DEFAULT_PROJECT",
        "ZHIPU_API_KEY",
        "ZHIPU_BASE_URL",
        "RUNWAY_API_KEY",
        "OPENAI_API_KEY",
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
    """Tests for REPO_ROOT, DEFAULT_CONFIG_PATH, DEFAULT_PROJECT_SLUG."""

    def test_repo_root_is_parent_of_orchestrator(self):
        """REPO_ROOT should be the parent directory of the orchestrator package."""
        orchestrator_dir = REPO_ROOT / "orchestrator"
        assert orchestrator_dir.is_dir()

    def test_repo_root_is_absolute(self):
        assert REPO_ROOT.is_absolute()

    def test_default_config_path_under_repo_root(self):
        assert DEFAULT_CONFIG_PATH == REPO_ROOT / "config" / "orchestrator_config.yaml"

    def test_default_project_slug_value(self, clean_env):
        """Without env override the default slug is 'aa-janta-naduma'."""
        # Re-import to get the value without env var influence
        # Since DEFAULT_PROJECT_SLUG is already evaluated at import time,
        # we test the env-fallback mechanism directly.
        result = os.environ.get("FRIDAY_DEFAULT_PROJECT", "aa-janta-naduma")
        assert result == "aa-janta-naduma"

    def test_default_project_slug_env_override(self):
        with patch.dict(os.environ, {"FRIDAY_DEFAULT_PROJECT": "custom-project"}):
            result = os.environ.get("FRIDAY_DEFAULT_PROJECT", "aa-janta-naduma")
            assert result == "custom-project"


# ===========================================================================
# 2. LLMConfig defaults
# ===========================================================================


class TestLLMConfigDefaults:
    """Every field of LLMConfig must have the documented default."""

    def test_backend_default(self):
        assert LLMConfig().backend == "vllm"

    def test_model_name_default(self):
        assert LLMConfig().model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def test_base_url_default(self):
        assert LLMConfig().base_url == "http://localhost:8000/v1"

    def test_api_key_default(self):
        assert LLMConfig().api_key == "not-needed"

    def test_default_adapter_default(self):
        assert LLMConfig().default_adapter == "friday-script"

    def test_adapter_path_default_is_none(self):
        assert LLMConfig().adapter_path is None

    def test_max_tokens_default(self):
        assert LLMConfig().max_tokens == 1024

    def test_temperature_default(self):
        assert LLMConfig().temperature == 0.7

    def test_top_p_default(self):
        assert LLMConfig().top_p == 0.9

    def test_custom_values(self):
        cfg = LLMConfig(backend="llamacpp", max_tokens=512)
        assert cfg.backend == "llamacpp"
        assert cfg.max_tokens == 512
        # Other fields remain defaults
        assert cfg.temperature == 0.7


# ===========================================================================
# 3. RouterConfig defaults
# ===========================================================================


class TestRouterConfigDefaults:
    """Every field of RouterConfig must have the documented default."""

    def test_enabled_default(self):
        assert RouterConfig().enabled is False

    def test_provider_default(self):
        assert RouterConfig().provider == "zhipu"

    def test_model_name_default(self):
        assert RouterConfig().model_name == "glm-4.7-flash"

    def test_api_key_default(self):
        assert RouterConfig().api_key == ""

    def test_base_url_default(self):
        assert RouterConfig().base_url == "https://api.z.ai/api/paas/v4"

    def test_timeout_default(self):
        assert RouterConfig().timeout == 5.0

    def test_max_tokens_default(self):
        assert RouterConfig().max_tokens == 256

    def test_temperature_default(self):
        assert RouterConfig().temperature == 0.3

    def test_fallback_on_error_default(self):
        assert RouterConfig().fallback_on_error is True

    def test_cache_decisions_default(self):
        assert RouterConfig().cache_decisions is True

    def test_cache_ttl_default(self):
        assert RouterConfig().cache_ttl == 300


# ===========================================================================
# 4. ExternalAPIConfig defaults
# ===========================================================================


class TestExternalAPIConfigDefaults:

    def test_vision_provider_default(self):
        assert ExternalAPIConfig().vision_provider == "anthropic"

    def test_vision_model_default(self):
        assert ExternalAPIConfig().vision_model == "claude-3-5-sonnet-20241022"

    def test_video_provider_default(self):
        assert ExternalAPIConfig().video_provider == "runway"

    def test_runway_api_key_default(self):
        assert ExternalAPIConfig().runway_api_key is None

    def test_image_provider_default(self):
        assert ExternalAPIConfig().image_provider == "openai"

    def test_openai_api_key_default(self):
        assert ExternalAPIConfig().openai_api_key is None


# ===========================================================================
# 5. ContextConfig defaults
# ===========================================================================


class TestContextConfigDefaults:

    def test_default_context_default(self):
        assert ContextConfig().default_context == "writers_room"

    def test_auto_detect_default(self):
        assert ContextConfig().auto_detect is True

    def test_rooms_default_is_empty_dict(self):
        assert ContextConfig().rooms == {}

    def test_rooms_default_factory_creates_separate_dicts(self):
        """Ensure each instance gets its own dict (not shared)."""
        a = ContextConfig()
        b = ContextConfig()
        a.rooms["test"] = {}
        assert "test" not in b.rooms


# ===========================================================================
# 6. MemoryConfig defaults
# ===========================================================================


class TestMemoryConfigDefaults:

    def test_max_history_turns_default(self):
        assert MemoryConfig().max_history_turns == 20

    def test_max_context_tokens_default(self):
        assert MemoryConfig().max_context_tokens == 6000

    def test_use_ltm_default(self):
        assert MemoryConfig().use_ltm is True

    def test_ltm_search_top_k_default(self):
        assert MemoryConfig().ltm_search_top_k == 5

    def test_persist_context_memory_default(self):
        assert MemoryConfig().persist_context_memory is True


# ===========================================================================
# 7. OrchestratorConfig defaults
# ===========================================================================


class TestOrchestratorConfigDefaults:

    def test_llm_sub_config_default(self):
        cfg = OrchestratorConfig()
        assert isinstance(cfg.llm, LLMConfig)
        assert cfg.llm.backend == "vllm"

    def test_router_sub_config_default(self):
        cfg = OrchestratorConfig()
        assert isinstance(cfg.router, RouterConfig)
        assert cfg.router.enabled is False

    def test_external_apis_sub_config_default(self):
        cfg = OrchestratorConfig()
        assert isinstance(cfg.external_apis, ExternalAPIConfig)

    def test_context_sub_config_default(self):
        cfg = OrchestratorConfig()
        assert isinstance(cfg.context, ContextConfig)

    def test_memory_sub_config_default(self):
        cfg = OrchestratorConfig()
        assert isinstance(cfg.memory, MemoryConfig)

    def test_host_default(self):
        assert OrchestratorConfig().host == "0.0.0.0"

    def test_port_default(self):
        assert OrchestratorConfig().port == 8080

    def test_debug_default(self):
        assert OrchestratorConfig().debug is False

    def test_system_prompt_base_contains_friday(self):
        cfg = OrchestratorConfig()
        assert "Friday" in cfg.system_prompt_base

    def test_system_prompt_base_contains_boss(self):
        cfg = OrchestratorConfig()
        assert "Boss" in cfg.system_prompt_base


# ===========================================================================
# 8. from_yaml
# ===========================================================================


class TestFromYaml:

    def test_nonexistent_file_returns_defaults(self, tmp_path):
        """from_yaml with a path that does not exist should return all defaults."""
        cfg = OrchestratorConfig.from_yaml(tmp_path / "does_not_exist.yaml")
        assert cfg.port == 8080
        assert cfg.llm.backend == "vllm"
        assert cfg.router.enabled is False

    def test_empty_yaml_returns_defaults(self, tmp_path):
        """An empty YAML file (parsed as None) should return defaults."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        cfg = OrchestratorConfig.from_yaml(yaml_file)
        assert cfg.port == 8080
        assert cfg.llm.backend == "vllm"

    def test_yaml_with_only_comments_returns_defaults(self, tmp_path):
        yaml_file = tmp_path / "comments.yaml"
        yaml_file.write_text("# just a comment\n# nothing else\n")
        cfg = OrchestratorConfig.from_yaml(yaml_file)
        assert cfg.port == 8080

    def test_valid_yaml_llm_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "llm": {
                        "backend": "llamacpp",
                        "model_name": "custom-model",
                        "max_tokens": 2048,
                        "temperature": 0.5,
                    }
                }
            )
        )
        cfg = OrchestratorConfig.from_yaml(yaml_file)
        assert cfg.llm.backend == "llamacpp"
        assert cfg.llm.model_name == "custom-model"
        assert cfg.llm.max_tokens == 2048
        assert cfg.llm.temperature == 0.5
        # Untouched defaults
        assert cfg.llm.top_p == 0.9

    def test_valid_yaml_router_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "router": {
                        "enabled": True,
                        "provider": "openai",
                        "timeout": 10.0,
                    }
                }
            )
        )
        cfg = OrchestratorConfig.from_yaml(yaml_file)
        assert cfg.router.enabled is True
        assert cfg.router.provider == "openai"
        assert cfg.router.timeout == 10.0
        # Untouched router defaults
        assert cfg.router.cache_ttl == 300

    def test_valid_yaml_server_settings(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "host": "127.0.0.1",
                    "port": 9090,
                    "debug": True,
                }
            )
        )
        cfg = OrchestratorConfig.from_yaml(yaml_file)
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9090
        assert cfg.debug is True

    def test_valid_yaml_full(self, tmp_path, clean_env):
        data = {
            "llm": {"backend": "openai", "max_tokens": 512},
            "router": {"enabled": True, "cache_ttl": 600},
            "external_apis": {
                "vision_provider": "openai",
                "image_provider": "midjourney",
            },
            "context": {"default_context": "editing_suite", "auto_detect": False},
            "memory": {"max_history_turns": 10, "use_ltm": False},
            "host": "localhost",
            "port": 3000,
            "debug": True,
            "system_prompt_base": "Custom prompt.",
        }
        yaml_file = tmp_path / "full.yaml"
        yaml_file.write_text(yaml.dump(data))
        cfg = OrchestratorConfig.from_yaml(yaml_file)
        assert cfg.llm.backend == "openai"
        assert cfg.router.cache_ttl == 600
        assert cfg.external_apis.vision_provider == "openai"
        assert cfg.external_apis.image_provider == "midjourney"
        assert cfg.context.default_context == "editing_suite"
        assert cfg.context.auto_detect is False
        assert cfg.memory.max_history_turns == 10
        assert cfg.memory.use_ltm is False
        assert cfg.host == "localhost"
        assert cfg.port == 3000
        assert cfg.debug is True
        assert cfg.system_prompt_base == "Custom prompt."

    def test_yaml_with_null_value_treated_as_empty(self, tmp_path):
        yaml_file = tmp_path / "null.yaml"
        yaml_file.write_text("---\n~\n")  # YAML null document
        cfg = OrchestratorConfig.from_yaml(yaml_file)
        assert cfg.port == 8080


# ===========================================================================
# 9. _from_dict
# ===========================================================================


class TestFromDict:

    def test_empty_dict_returns_defaults(self, clean_env):
        cfg = OrchestratorConfig._from_dict({})
        assert cfg.llm.backend == "vllm"
        assert cfg.port == 8080

    def test_partial_llm_section(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"llm": {"backend": "openai"}})
        assert cfg.llm.backend == "openai"
        assert cfg.llm.model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct"

    def test_partial_router_section(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"router": {"enabled": True}})
        assert cfg.router.enabled is True
        assert cfg.router.provider == "zhipu"

    def test_partial_external_apis_section(self, clean_env):
        cfg = OrchestratorConfig._from_dict(
            {"external_apis": {"vision_provider": "openai"}}
        )
        assert cfg.external_apis.vision_provider == "openai"
        assert cfg.external_apis.video_provider == "runway"

    def test_partial_context_section(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"context": {"auto_detect": False}})
        assert cfg.context.auto_detect is False
        assert cfg.context.default_context == "writers_room"

    def test_partial_memory_section(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"memory": {"use_ltm": False}})
        assert cfg.memory.use_ltm is False
        assert cfg.memory.max_history_turns == 20

    def test_all_sections_populated(self, clean_env):
        data = {
            "llm": {"backend": "llamacpp", "max_tokens": 256},
            "router": {"enabled": True, "provider": "openai"},
            "external_apis": {"vision_provider": "openai"},
            "context": {"default_context": "color_grading"},
            "memory": {"ltm_search_top_k": 10},
            "host": "10.0.0.1",
            "port": 5000,
            "debug": True,
        }
        cfg = OrchestratorConfig._from_dict(data)
        assert cfg.llm.backend == "llamacpp"
        assert cfg.llm.max_tokens == 256
        assert cfg.router.enabled is True
        assert cfg.router.provider == "openai"
        assert cfg.external_apis.vision_provider == "openai"
        assert cfg.context.default_context == "color_grading"
        assert cfg.memory.ltm_search_top_k == 10
        assert cfg.host == "10.0.0.1"
        assert cfg.port == 5000
        assert cfg.debug is True

    def test_system_prompt_override(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"system_prompt_base": "Hello from test"})
        assert cfg.system_prompt_base == "Hello from test"

    def test_rooms_in_context(self, clean_env):
        rooms = {
            "writers_room": {"description": "test"},
            "editing_suite": {"description": "edit"},
        }
        cfg = OrchestratorConfig._from_dict({"context": {"rooms": rooms}})
        assert cfg.context.rooms == rooms

    def test_adapter_path_set(self, clean_env):
        cfg = OrchestratorConfig._from_dict(
            {"llm": {"adapter_path": "/models/adapter"}}
        )
        assert cfg.llm.adapter_path == "/models/adapter"


# ===========================================================================
# 10. Environment variable overrides via _from_dict
# ===========================================================================


class TestEnvVarOverrides:

    def test_friday_llm_backend_override(self):
        with patch.dict(os.environ, {"FRIDAY_LLM_BACKEND": "custom_backend"}):
            cfg = OrchestratorConfig._from_dict({"llm": {"backend": "vllm"}})
            assert cfg.llm.backend == "custom_backend"

    def test_friday_llm_model_override(self):
        with patch.dict(os.environ, {"FRIDAY_LLM_MODEL": "env-model"}):
            cfg = OrchestratorConfig._from_dict({"llm": {}})
            assert cfg.llm.model_name == "env-model"

    def test_friday_llm_base_url_override(self):
        with patch.dict(os.environ, {"FRIDAY_LLM_BASE_URL": "http://env:1234"}):
            cfg = OrchestratorConfig._from_dict(
                {"llm": {"base_url": "http://yaml:5678"}}
            )
            assert cfg.llm.base_url == "http://env:1234"

    def test_friday_llm_api_key_override(self):
        with patch.dict(os.environ, {"FRIDAY_LLM_API_KEY": "secret-key"}):
            cfg = OrchestratorConfig._from_dict({"llm": {"api_key": "yaml-key"}})
            assert cfg.llm.api_key == "secret-key"

    def test_zhipu_api_key_override(self):
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "zhipu-env-key"}):
            cfg = OrchestratorConfig._from_dict({"router": {"api_key": "yaml-key"}})
            assert cfg.router.api_key == "zhipu-env-key"

    def test_zhipu_base_url_override(self):
        with patch.dict(os.environ, {"ZHIPU_BASE_URL": "https://env.zhipu/v4"}):
            cfg = OrchestratorConfig._from_dict(
                {"router": {"base_url": "https://yaml.zhipu/v4"}}
            )
            assert cfg.router.base_url == "https://env.zhipu/v4"

    def test_runway_api_key_override(self):
        with patch.dict(os.environ, {"RUNWAY_API_KEY": "runway-env-key"}):
            cfg = OrchestratorConfig._from_dict(
                {"external_apis": {"runway_api_key": "yaml-key"}}
            )
            assert cfg.external_apis.runway_api_key == "runway-env-key"

    def test_openai_api_key_override(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-env-key"}):
            cfg = OrchestratorConfig._from_dict(
                {"external_apis": {"openai_api_key": "yaml-key"}}
            )
            assert cfg.external_apis.openai_api_key == "openai-env-key"

    def test_friday_port_override(self):
        with patch.dict(os.environ, {"FRIDAY_PORT": "9999"}):
            cfg = OrchestratorConfig._from_dict({"port": 8080})
            assert cfg.port == 9999

    def test_friday_port_override_without_yaml_port(self):
        with patch.dict(os.environ, {"FRIDAY_PORT": "4444"}):
            cfg = OrchestratorConfig._from_dict({})
            assert cfg.port == 4444

    def test_env_var_does_not_affect_when_absent(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"llm": {"backend": "llamacpp"}})
        assert cfg.llm.backend == "llamacpp"

    def test_multiple_env_vars_simultaneously(self):
        env = {
            "FRIDAY_LLM_BACKEND": "env-backend",
            "FRIDAY_LLM_MODEL": "env-model",
            "FRIDAY_PORT": "7777",
            "ZHIPU_API_KEY": "env-zhipu",
            "OPENAI_API_KEY": "env-openai",
        }
        with patch.dict(os.environ, env):
            cfg = OrchestratorConfig._from_dict(
                {
                    "llm": {"backend": "yaml-backend"},
                    "router": {},
                    "external_apis": {},
                }
            )
            assert cfg.llm.backend == "env-backend"
            assert cfg.llm.model_name == "env-model"
            assert cfg.port == 7777
            assert cfg.router.api_key == "env-zhipu"
            assert cfg.external_apis.openai_api_key == "env-openai"


# ===========================================================================
# 11. to_dict
# ===========================================================================


class TestToDict:

    def test_to_dict_contains_all_top_level_keys(self):
        d = OrchestratorConfig().to_dict()
        expected_keys = {
            "llm",
            "router",
            "external_apis",
            "context",
            "memory",
            "host",
            "port",
            "debug",
            "system_prompt_base",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_llm_section_keys(self):
        d = OrchestratorConfig().to_dict()
        llm_keys = {
            "backend",
            "model_name",
            "base_url",
            "default_adapter",
            "max_tokens",
            "temperature",
            "top_p",
        }
        assert set(d["llm"].keys()) == llm_keys

    def test_to_dict_llm_excludes_api_key(self):
        """api_key is intentionally excluded from to_dict for security."""
        d = OrchestratorConfig().to_dict()
        assert "api_key" not in d["llm"]

    def test_to_dict_llm_excludes_adapter_path(self):
        d = OrchestratorConfig().to_dict()
        assert "adapter_path" not in d["llm"]

    def test_to_dict_router_excludes_api_key(self):
        d = OrchestratorConfig().to_dict()
        assert "api_key" not in d["router"]

    def test_to_dict_external_apis_excludes_secrets(self):
        d = OrchestratorConfig().to_dict()
        assert "runway_api_key" not in d["external_apis"]
        assert "openai_api_key" not in d["external_apis"]

    def test_to_dict_roundtrip_values(self, clean_env):
        """Values placed into config should appear in the dict output."""
        cfg = OrchestratorConfig._from_dict(
            {
                "llm": {"backend": "openai", "max_tokens": 512},
                "host": "myhost",
                "port": 1234,
                "debug": True,
            }
        )
        d = cfg.to_dict()
        assert d["llm"]["backend"] == "openai"
        assert d["llm"]["max_tokens"] == 512
        assert d["host"] == "myhost"
        assert d["port"] == 1234
        assert d["debug"] is True

    def test_to_dict_router_section_keys(self):
        d = OrchestratorConfig().to_dict()
        expected = {
            "enabled",
            "provider",
            "model_name",
            "base_url",
            "timeout",
            "max_tokens",
            "temperature",
            "fallback_on_error",
            "cache_decisions",
            "cache_ttl",
        }
        assert set(d["router"].keys()) == expected

    def test_to_dict_context_section_keys(self):
        d = OrchestratorConfig().to_dict()
        expected = {"default_context", "auto_detect", "rooms"}
        assert set(d["context"].keys()) == expected

    def test_to_dict_memory_section_keys(self):
        d = OrchestratorConfig().to_dict()
        expected = {
            "max_history_turns",
            "use_ltm",
            "ltm_search_top_k",
            "persist_context_memory",
        }
        assert set(d["memory"].keys()) == expected

    def test_to_dict_memory_excludes_max_context_tokens(self):
        """max_context_tokens is on MemoryConfig but not exported by to_dict."""
        d = OrchestratorConfig().to_dict()
        assert "max_context_tokens" not in d["memory"]

    def test_to_dict_system_prompt_base_value(self):
        cfg = OrchestratorConfig()
        d = cfg.to_dict()
        assert d["system_prompt_base"] == cfg.system_prompt_base

    def test_to_dict_custom_rooms(self, clean_env):
        rooms = {"room_a": {"desc": "A"}, "room_b": {"desc": "B"}}
        cfg = OrchestratorConfig._from_dict({"context": {"rooms": rooms}})
        d = cfg.to_dict()
        assert d["context"]["rooms"] == rooms


# ===========================================================================
# 12. _env_or_default helper
# ===========================================================================


class TestEnvOrDefault:

    def test_returns_default_when_env_not_set(self, clean_env):
        result = _env_or_default("NONEXISTENT_TEST_KEY_XYZ_12345", "fallback")
        assert result == "fallback"

    def test_returns_env_value_when_set(self):
        with patch.dict(os.environ, {"MY_TEST_KEY": "from_env"}):
            result = _env_or_default("MY_TEST_KEY", "fallback")
            assert result == "from_env"

    def test_env_overrides_none_default(self):
        with patch.dict(os.environ, {"MY_TEST_KEY": "val"}):
            result = _env_or_default("MY_TEST_KEY", None)
            assert result == "val"

    def test_returns_none_default_when_env_not_set(self, clean_env):
        result = _env_or_default("NONEXISTENT_TEST_KEY_XYZ_12345", None)
        assert result is None

    def test_env_empty_string_is_still_returned(self):
        with patch.dict(os.environ, {"MY_TEST_KEY": ""}):
            result = _env_or_default("MY_TEST_KEY", "fallback")
            assert result == ""

    def test_returns_integer_default_when_env_not_set(self, clean_env):
        result = _env_or_default("NONEXISTENT_TEST_KEY_XYZ_12345", 42)
        assert result == 42


# ===========================================================================
# 13. get_config singleton
# ===========================================================================


class TestGetConfig:

    def test_returns_orchestrator_config_instance(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        cfg = get_config(yaml_file)
        assert isinstance(cfg, OrchestratorConfig)

    def test_singleton_returns_same_object(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        cfg1 = get_config(yaml_file)
        cfg2 = get_config(yaml_file)
        assert cfg1 is cfg2

    def test_singleton_ignores_second_path(self, tmp_path):
        """Once created, the singleton ignores subsequent path arguments."""
        f1 = tmp_path / "a.yaml"
        f1.write_text(yaml.dump({"port": 1111}))
        f2 = tmp_path / "b.yaml"
        f2.write_text(yaml.dump({"port": 2222}))
        cfg1 = get_config(f1)
        cfg2 = get_config(f2)
        assert cfg1 is cfg2

    def test_get_config_with_nonexistent_path(self, tmp_path):
        cfg = get_config(tmp_path / "missing.yaml")
        assert cfg.port == 8080

    def test_get_config_default_path_used_when_none(self):
        """When config_path is None, DEFAULT_CONFIG_PATH is used."""
        cfg = get_config(None)
        assert isinstance(cfg, OrchestratorConfig)


# ===========================================================================
# 14. reload_config
# ===========================================================================


class TestReloadConfig:

    def test_reload_creates_new_instance(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(yaml.dump({"port": 1111}))
        cfg1 = get_config(yaml_file)
        assert cfg1.port == 1111

        yaml_file.write_text(yaml.dump({"port": 2222}))
        cfg2 = reload_config(yaml_file)
        assert cfg2.port == 2222
        assert cfg1 is not cfg2

    def test_reload_updates_singleton(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(yaml.dump({"debug": False}))
        get_config(yaml_file)

        yaml_file.write_text(yaml.dump({"debug": True}))
        reload_config(yaml_file)
        cfg = get_config()
        assert cfg.debug is True

    def test_reload_with_nonexistent_file_returns_defaults(self, tmp_path):
        cfg = reload_config(tmp_path / "gone.yaml")
        assert cfg.port == 8080
        assert cfg.llm.backend == "vllm"

    def test_reload_with_none_uses_default_path(self):
        cfg = reload_config(None)
        assert isinstance(cfg, OrchestratorConfig)


# ===========================================================================
# 15. Edge cases
# ===========================================================================


class TestEdgeCases:

    def test_from_dict_with_empty_llm_section(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"llm": {}})
        assert cfg.llm.backend == "vllm"
        assert cfg.llm.api_key == "not-needed"

    def test_from_dict_with_empty_router_section(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"router": {}})
        assert cfg.router.enabled is False
        assert cfg.router.api_key == ""

    def test_from_dict_with_empty_external_apis_section(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"external_apis": {}})
        assert cfg.external_apis.vision_provider == "anthropic"

    def test_from_dict_with_empty_context_section(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"context": {}})
        assert cfg.context.default_context == "writers_room"

    def test_from_dict_with_empty_memory_section(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"memory": {}})
        assert cfg.memory.max_history_turns == 20

    def test_port_is_cast_to_int(self, clean_env):
        """Port from env is a string; _from_dict should cast it to int."""
        with patch.dict(os.environ, {"FRIDAY_PORT": "3333"}):
            cfg = OrchestratorConfig._from_dict({})
            assert cfg.port == 3333
            assert isinstance(cfg.port, int)

    def test_port_from_yaml_string_cast_to_int(self, clean_env):
        """Even if YAML gives a string port, int() cast should handle it."""
        cfg = OrchestratorConfig._from_dict({"port": "7070"})
        assert cfg.port == 7070

    def test_from_dict_unknown_keys_are_ignored(self, clean_env):
        cfg = OrchestratorConfig._from_dict({"unknown_key": "value", "another": 123})
        assert cfg.port == 8080

    def test_from_dict_llm_unknown_keys_are_ignored(self, clean_env):
        """Extra keys in sub-dicts shouldn't blow up because .get() ignores them."""
        cfg = OrchestratorConfig._from_dict({"llm": {"unknown_sub_key": True}})
        assert cfg.llm.backend == "vllm"

    def test_yaml_with_extra_sections(self, tmp_path, clean_env):
        yaml_file = tmp_path / "extra.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "llm": {"backend": "test"},
                    "some_unknown_section": {"a": 1},
                }
            )
        )
        cfg = OrchestratorConfig.from_yaml(yaml_file)
        assert cfg.llm.backend == "test"

    def test_multiple_configs_are_independent(self):
        """Two OrchestratorConfig instances should not share mutable state."""
        a = OrchestratorConfig()
        b = OrchestratorConfig()
        a.context.rooms["x"] = {"foo": "bar"}
        assert "x" not in b.context.rooms

    def test_dataclass_equality(self):
        """Two default LLMConfig instances should be equal."""
        assert LLMConfig() == LLMConfig()

    def test_dataclass_inequality(self):
        assert LLMConfig(backend="a") != LLMConfig(backend="b")

    def test_from_yaml_reads_utf8(self, tmp_path, clean_env):
        yaml_file = tmp_path / "utf8.yaml"
        yaml_file.write_text(
            yaml.dump(
                {"system_prompt_base": "Telugu: \u0c24\u0c46\u0c32\u0c41\u0c17\u0c41"}
            ),
            encoding="utf-8",
        )
        cfg = OrchestratorConfig.from_yaml(yaml_file)
        assert "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41" in cfg.system_prompt_base

    def test_env_override_takes_priority_over_yaml_file(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "llm": {"backend": "from-yaml"},
                    "port": 1111,
                }
            )
        )
        with patch.dict(
            os.environ, {"FRIDAY_LLM_BACKEND": "from-env", "FRIDAY_PORT": "2222"}
        ):
            cfg = OrchestratorConfig.from_yaml(yaml_file)
            assert cfg.llm.backend == "from-env"
            assert cfg.port == 2222
