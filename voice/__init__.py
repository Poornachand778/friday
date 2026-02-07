"""
Friday AI Voice Pipeline
========================

Voice conversation pipeline for JARVIS-style interaction.
Includes STT (Faster-Whisper), TTS (XTTS v2), and wake word detection.

Status:
    TODO: Complete voice daemon integration with orchestrator
    TODO: Test Faster-Whisper STT with Telugu audio
    TODO: Configure XTTS v2 with Boss's voice samples
    TODO: Train custom wake word ("Hey Friday" or "Wake up Daddy's home")
    TODO: Implement audio storage for training data generation
    BLOCKED: Needs voice samples from Boss for XTTS cloning
"""

from .config import VoiceConfig, get_voice_config

__all__ = [
    "VoiceConfig",
    "get_voice_config",
]
