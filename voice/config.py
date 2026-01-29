"""
Voice Configuration for Friday AI
==================================

YAML-based configuration with environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "voice_config.yaml"


@dataclass
class AudioConfig:
    """Audio I/O settings"""

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    chunk_size: int = 512  # samples per chunk
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    silence_threshold_ms: int = 500  # Silence duration to end utterance
    max_recording_seconds: float = 30.0  # Max single utterance


@dataclass
class STTConfig:
    """Speech-to-Text settings"""

    engine: str = "faster_whisper"
    model: str = "large-v3"
    device: str = "cuda"  # cuda, cpu
    compute_type: str = "float16"  # float16, int8, float32
    language: Optional[str] = None  # None = auto-detect
    word_timestamps: bool = True
    vad_filter: bool = True
    beam_size: int = 5


@dataclass
class TTSConfig:
    """Text-to-Speech settings"""

    engine: str = "xtts_v2"
    device: str = "cuda"
    default_language: str = "te"  # Telugu
    default_profile: str = "friday_telugu"
    speed: float = 1.0
    temperature: float = 0.7
    top_p: float = 0.85
    top_k: int = 50
    repetition_penalty: float = 10.0


@dataclass
class WakeWordConfig:
    """Wake word detection settings"""

    engine: str = "openwakeword"
    models: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "hey_friday", "threshold": 0.5},
        ]
    )
    inference_framework: str = "onnx"  # onnx, tflite
    vad_threshold: float = 0.5
    noise_suppression: bool = True


@dataclass
class StorageConfig:
    """Audio storage settings"""

    enabled: bool = True
    base_path: str = "voice/data/recordings"
    organize_by_date: bool = True  # Creates YYYY/MM/DD subfolders
    retention_days: int = 90
    save_user_audio: bool = True
    save_response_audio: bool = True
    transcript_format: str = "jsonl"


@dataclass
class DaemonConfig:
    """Voice daemon settings"""

    mode: str = "standalone"  # standalone, integrated
    auto_start: bool = False
    idle_timeout_seconds: int = 300  # 5 minutes idle = stop listening
    max_session_minutes: int = 30
    device_id: str = "default"
    location: str = "writers_room"


@dataclass
class VoiceConfig:
    """Complete voice configuration"""

    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "VoiceConfig":
        """Load configuration from YAML file"""
        if not path.exists():
            return cls()  # Return defaults if no config file

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "VoiceConfig":
        """Create config from dictionary with env var overrides"""
        config = cls()

        # Audio config
        if "audio" in data:
            config.audio = AudioConfig(
                **{
                    k: _env_override(
                        f"VOICE_AUDIO_{k.upper()}", v, type(getattr(AudioConfig, k, v))
                    )
                    for k, v in data["audio"].items()
                }
            )

        # STT config
        if "stt" in data:
            config.stt = STTConfig(
                **{
                    k: _env_override(
                        f"VOICE_STT_{k.upper()}",
                        v,
                        (
                            type(getattr(STTConfig, k, v))
                            if hasattr(STTConfig, k)
                            else type(v)
                        ),
                    )
                    for k, v in data["stt"].items()
                }
            )

        # TTS config
        if "tts" in data:
            config.tts = TTSConfig(
                **{
                    k: _env_override(
                        f"VOICE_TTS_{k.upper()}",
                        v,
                        (
                            type(getattr(TTSConfig, k, v))
                            if hasattr(TTSConfig, k)
                            else type(v)
                        ),
                    )
                    for k, v in data["tts"].items()
                }
            )

        # Wake word config
        if "wakeword" in data:
            ww_data = data["wakeword"]
            config.wakeword = WakeWordConfig(
                engine=_env_override(
                    "VOICE_WAKEWORD_ENGINE", ww_data.get("engine", "openwakeword"), str
                ),
                models=ww_data.get(
                    "models", [{"name": "hey_friday", "threshold": 0.5}]
                ),
                inference_framework=ww_data.get("inference_framework", "onnx"),
                vad_threshold=ww_data.get("vad_threshold", 0.5),
                noise_suppression=ww_data.get("noise_suppression", True),
            )

        # Storage config
        if "storage" in data:
            config.storage = StorageConfig(
                **{
                    k: _env_override(
                        f"VOICE_STORAGE_{k.upper()}",
                        v,
                        (
                            type(getattr(StorageConfig, k, v))
                            if hasattr(StorageConfig, k)
                            else type(v)
                        ),
                    )
                    for k, v in data["storage"].items()
                }
            )

        # Daemon config
        if "daemon" in data:
            config.daemon = DaemonConfig(
                **{
                    k: _env_override(
                        f"VOICE_DAEMON_{k.upper()}",
                        v,
                        (
                            type(getattr(DaemonConfig, k, v))
                            if hasattr(DaemonConfig, k)
                            else type(v)
                        ),
                    )
                    for k, v in data["daemon"].items()
                }
            )

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "channels": self.audio.channels,
                "dtype": self.audio.dtype,
                "chunk_size": self.audio.chunk_size,
                "vad_aggressiveness": self.audio.vad_aggressiveness,
                "silence_threshold_ms": self.audio.silence_threshold_ms,
                "max_recording_seconds": self.audio.max_recording_seconds,
            },
            "stt": {
                "engine": self.stt.engine,
                "model": self.stt.model,
                "device": self.stt.device,
                "compute_type": self.stt.compute_type,
                "language": self.stt.language,
                "word_timestamps": self.stt.word_timestamps,
                "vad_filter": self.stt.vad_filter,
                "beam_size": self.stt.beam_size,
            },
            "tts": {
                "engine": self.tts.engine,
                "device": self.tts.device,
                "default_language": self.tts.default_language,
                "default_profile": self.tts.default_profile,
                "speed": self.tts.speed,
                "temperature": self.tts.temperature,
                "top_p": self.tts.top_p,
                "top_k": self.tts.top_k,
                "repetition_penalty": self.tts.repetition_penalty,
            },
            "wakeword": {
                "engine": self.wakeword.engine,
                "models": self.wakeword.models,
                "inference_framework": self.wakeword.inference_framework,
                "vad_threshold": self.wakeword.vad_threshold,
                "noise_suppression": self.wakeword.noise_suppression,
            },
            "storage": {
                "enabled": self.storage.enabled,
                "base_path": self.storage.base_path,
                "organize_by_date": self.storage.organize_by_date,
                "retention_days": self.storage.retention_days,
                "save_user_audio": self.storage.save_user_audio,
                "save_response_audio": self.storage.save_response_audio,
                "transcript_format": self.storage.transcript_format,
            },
            "daemon": {
                "mode": self.daemon.mode,
                "auto_start": self.daemon.auto_start,
                "idle_timeout_seconds": self.daemon.idle_timeout_seconds,
                "max_session_minutes": self.daemon.max_session_minutes,
                "device_id": self.daemon.device_id,
                "location": self.daemon.location,
            },
        }

    def save_yaml(self, path: Path) -> None:
        """Save configuration to YAML file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def _env_override(env_key: str, default: Any, dtype: type) -> Any:
    """Override value from environment variable if set"""
    value = os.environ.get(env_key)
    if value is None:
        return default

    # Type conversion
    if dtype == bool:
        return value.lower() in ("true", "1", "yes")
    elif dtype == int:
        return int(value)
    elif dtype == float:
        return float(value)
    return value


# Singleton config instance
_config: Optional[VoiceConfig] = None


def get_voice_config(config_path: Optional[Path] = None) -> VoiceConfig:
    """Get the voice configuration singleton"""
    global _config

    if _config is None:
        path = config_path or DEFAULT_CONFIG_PATH
        _config = VoiceConfig.from_yaml(path)

    return _config


def reload_config(config_path: Optional[Path] = None) -> VoiceConfig:
    """Reload configuration from file"""
    global _config
    path = config_path or DEFAULT_CONFIG_PATH
    _config = VoiceConfig.from_yaml(path)
    return _config
