"""
Wake Word Detection module for Friday Voice Pipeline
====================================================

Provides wake word detection using OpenWakeWord.
Supports custom wake phrases like "Hey Friday" or "Wake up Daddy's home".
"""

from .openwakeword_service import (
    OpenWakeWordService,
    WakeWordDetection,
    detect_wake_word,
)
from .trainer import WakeWordTrainer, TrainingConfig

__all__ = [
    "OpenWakeWordService",
    "WakeWordDetection",
    "detect_wake_word",
    "WakeWordTrainer",
    "TrainingConfig",
]
