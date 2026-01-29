"""
Text-to-Speech module for Friday Voice Pipeline
==============================================

Provides TTS using XTTS v2 (Coqui) with Telugu voice cloning.
"""

from .xtts_service import XTTSService, TTSResult, synthesize_speech
from .voice_profiles import VoiceProfileManager, VoiceProfileInfo

__all__ = [
    "XTTSService",
    "TTSResult",
    "synthesize_speech",
    "VoiceProfileManager",
    "VoiceProfileInfo",
]
