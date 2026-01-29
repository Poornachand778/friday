"""
Friday AI Voice Pipeline
========================

Voice conversation pipeline for JARVIS-style interaction.
Includes STT (Faster-Whisper), TTS (XTTS v2), and wake word detection.
"""

from .config import VoiceConfig, get_voice_config

__all__ = [
    "VoiceConfig",
    "get_voice_config",
]
