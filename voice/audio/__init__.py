"""
Audio I/O module for Friday Voice Pipeline
==========================================

Handles microphone capture, speaker playback, and voice activity detection.
"""

from .capture import AudioCapture, AudioChunk
from .playback import AudioPlayback
from .vad import VoiceActivityDetector

__all__ = [
    "AudioCapture",
    "AudioChunk",
    "AudioPlayback",
    "VoiceActivityDetector",
]
