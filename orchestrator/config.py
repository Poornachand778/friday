"""
Orchestrator Configuration
==========================

Configuration for Friday AI Orchestrator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "orchestrator_config.yaml"


@dataclass
class LLMConfig:
    """Local LLM inference configuration"""

    backend: str = "vllm"  # vllm, llamacpp, openai
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    base_url: str = "http://localhost:8000/v1"  # vLLM OpenAI-compatible endpoint
    api_key: str = "not-needed"  # For local vLLM

    # LoRA adapter settings
    default_adapter: str = "friday-script"
    adapter_path: Optional[str] = None

    # Generation settings
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class ExternalAPIConfig:
    """External API configuration"""

    # Vision API (for camera)
    vision_provider: str = "anthropic"  # anthropic, openai
    vision_model: str = "claude-3-5-sonnet-20241022"

    # Video generation (for storyboard)
    video_provider: str = "runway"  # runway, pika, kling
    runway_api_key: Optional[str] = None

    # Image generation
    image_provider: str = "openai"  # openai, midjourney
    openai_api_key: Optional[str] = None


@dataclass
class ContextConfig:
    """Context/room configuration"""

    default_context: str = "writers_room"
    auto_detect: bool = True

    # Room definitions loaded from config
    rooms: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Memory configuration"""

    # Conversation history
    max_history_turns: int = 20
    max_context_tokens: int = 6000

    # Long-term memory
    use_ltm: bool = True
    ltm_search_top_k: int = 5

    # Context-specific memory
    persist_context_memory: bool = True


@dataclass
class OrchestratorConfig:
    """Complete orchestrator configuration"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    external_apis: ExternalAPIConfig = field(default_factory=ExternalAPIConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False

    # System prompt base
    system_prompt_base: str = (
        "You are Friday, Poorna's AI assistant. "
        "You blend Telugu and English naturally, addressing him as 'Boss'. "
        "Be concise, helpful, and direct. No flattery or excessive formality. "
        "You have access to tools - use them when needed."
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "OrchestratorConfig":
        """Load configuration from YAML file"""
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "OrchestratorConfig":
        """Create config from dictionary"""
        config = cls()

        # LLM config
        if "llm" in data:
            llm_data = data["llm"]
            config.llm = LLMConfig(
                backend=_env_or_default(
                    "FRIDAY_LLM_BACKEND", llm_data.get("backend", "vllm")
                ),
                model_name=_env_or_default(
                    "FRIDAY_LLM_MODEL",
                    llm_data.get("model_name", config.llm.model_name),
                ),
                base_url=_env_or_default(
                    "FRIDAY_LLM_BASE_URL", llm_data.get("base_url", config.llm.base_url)
                ),
                api_key=_env_or_default(
                    "FRIDAY_LLM_API_KEY", llm_data.get("api_key", "not-needed")
                ),
                default_adapter=llm_data.get("default_adapter", "friday-script"),
                adapter_path=llm_data.get("adapter_path"),
                max_tokens=llm_data.get("max_tokens", 1024),
                temperature=llm_data.get("temperature", 0.7),
                top_p=llm_data.get("top_p", 0.9),
            )

        # External APIs
        if "external_apis" in data:
            api_data = data["external_apis"]
            config.external_apis = ExternalAPIConfig(
                vision_provider=api_data.get("vision_provider", "anthropic"),
                vision_model=api_data.get("vision_model", "claude-3-5-sonnet-20241022"),
                video_provider=api_data.get("video_provider", "runway"),
                runway_api_key=_env_or_default(
                    "RUNWAY_API_KEY", api_data.get("runway_api_key")
                ),
                image_provider=api_data.get("image_provider", "openai"),
                openai_api_key=_env_or_default(
                    "OPENAI_API_KEY", api_data.get("openai_api_key")
                ),
            )

        # Context config
        if "context" in data:
            ctx_data = data["context"]
            config.context = ContextConfig(
                default_context=ctx_data.get("default_context", "writers_room"),
                auto_detect=ctx_data.get("auto_detect", True),
                rooms=ctx_data.get("rooms", {}),
            )

        # Memory config
        if "memory" in data:
            mem_data = data["memory"]
            config.memory = MemoryConfig(
                max_history_turns=mem_data.get("max_history_turns", 20),
                use_ltm=mem_data.get("use_ltm", True),
                ltm_search_top_k=mem_data.get("ltm_search_top_k", 5),
                persist_context_memory=mem_data.get("persist_context_memory", True),
            )

        # Server settings
        config.host = data.get("host", "0.0.0.0")
        config.port = int(_env_or_default("FRIDAY_PORT", str(data.get("port", 8080))))
        config.debug = data.get("debug", False)

        # System prompt
        if "system_prompt_base" in data:
            config.system_prompt_base = data["system_prompt_base"]

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            "llm": {
                "backend": self.llm.backend,
                "model_name": self.llm.model_name,
                "base_url": self.llm.base_url,
                "default_adapter": self.llm.default_adapter,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
            },
            "external_apis": {
                "vision_provider": self.external_apis.vision_provider,
                "vision_model": self.external_apis.vision_model,
                "video_provider": self.external_apis.video_provider,
                "image_provider": self.external_apis.image_provider,
            },
            "context": {
                "default_context": self.context.default_context,
                "auto_detect": self.context.auto_detect,
                "rooms": self.context.rooms,
            },
            "memory": {
                "max_history_turns": self.memory.max_history_turns,
                "use_ltm": self.memory.use_ltm,
                "ltm_search_top_k": self.memory.ltm_search_top_k,
                "persist_context_memory": self.memory.persist_context_memory,
            },
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "system_prompt_base": self.system_prompt_base,
        }


def _env_or_default(env_key: str, default: Any) -> Any:
    """Get from environment or return default"""
    return os.environ.get(env_key, default)


# Singleton instance
_config: Optional[OrchestratorConfig] = None


def get_config(config_path: Optional[Path] = None) -> OrchestratorConfig:
    """Get orchestrator configuration singleton"""
    global _config

    if _config is None:
        path = config_path or DEFAULT_CONFIG_PATH
        _config = OrchestratorConfig.from_yaml(path)

    return _config


def reload_config(config_path: Optional[Path] = None) -> OrchestratorConfig:
    """Reload configuration"""
    global _config
    path = config_path or DEFAULT_CONFIG_PATH
    _config = OrchestratorConfig.from_yaml(path)
    return _config
