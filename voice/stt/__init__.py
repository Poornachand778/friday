"""
Speech-to-Text module for Friday Voice Pipeline
==============================================

Provides STT using Faster-Whisper with Telugu/English language detection.
"""

from .faster_whisper_service import (
    FasterWhisperSTT,
    TranscriptionResult,
    WordTiming,
    transcribe_audio,
)
from .language_detector import detect_language, LanguageInfo

__all__ = [
    "FasterWhisperSTT",
    "TranscriptionResult",
    "WordTiming",
    "transcribe_audio",
    "detect_language",
    "LanguageInfo",
]
