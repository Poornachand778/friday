"""
Telugu-English Processing
=========================

Handles Telugu-English code-switching in memory operations.

Features:
    - Language detection (Telugu, English, mixed)
    - Telugu density calculation
    - Telugu keyword extraction
    - Code-switch point detection
"""

from memory.telugu.processor import (
    TeluguEnglishProcessor,
    ProcessedText,
    TextSegment,
    detect_telugu,
    calculate_telugu_density,
)

__all__ = [
    "TeluguEnglishProcessor",
    "ProcessedText",
    "TextSegment",
    "detect_telugu",
    "calculate_telugu_density",
]
