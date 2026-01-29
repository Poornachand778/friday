"""
Language Detection for Friday Voice Pipeline
=============================================

Detects Telugu, English, or mixed language in transcribed text.
Uses character ranges and common patterns for accurate detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


# Telugu Unicode range
TELUGU_RANGE = (0x0C00, 0x0C7F)

# Common Telugu words/patterns that appear in Romanized Telugu
TELUGU_ROMANIZED_PATTERNS = [
    r"\b(nenu|meeru|vaadu|adi|idi|emi|enduku|ela|cheppandi|cheyyi)\b",
    r"\b(naaku|neeku|vaadiki|daniki|ee|aa|oka|anni|konni)\b",
    r"\b(vellu|ra|po|chestha|chesthanu|chesthunnav|chesthunna)\b",
    r"\b(antha|antha|entha|enta|chala|koncham|koddiga)\b",
    r"\b(anna|akka|amma|nanna|thammudu|chelli)\b",
    r"\b(ledu|undi|undhi|ledhu|ayindi|ayyindi)\b",
    r"\b(ante|kani|kuda|aithe|aithey|matram)\b",
    r"\b(ippudu|appudu|yeppudu|yepudu|mundu|taruvatha)\b",
]

# English stop words (common function words)
ENGLISH_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "up",
    "about",
    "into",
    "over",
    "after",
    "and",
    "but",
    "or",
    "nor",
    "so",
    "yet",
    "both",
    "either",
    "neither",
    "not",
    "only",
    "own",
    "same",
    "than",
    "too",
    "very",
    "just",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
}


@dataclass
class LanguageInfo:
    """Language detection result"""

    primary_language: str  # 'te', 'en', or 'mixed'
    telugu_ratio: float  # 0.0 - 1.0
    english_ratio: float  # 0.0 - 1.0
    mixed_ratio: float  # 0.0 - 1.0
    has_telugu_script: bool
    has_romanized_telugu: bool
    confidence: float

    @property
    def is_telugu(self) -> bool:
        return self.primary_language == "te"

    @property
    def is_english(self) -> bool:
        return self.primary_language == "en"

    @property
    def is_mixed(self) -> bool:
        return self.primary_language == "mixed"


def detect_language(text: str) -> LanguageInfo:
    """
    Detect language of text.

    Handles:
    - Pure Telugu (Telugu script)
    - Pure English
    - Romanized Telugu (Telugu written in Latin script)
    - Code-mixed Telugu-English

    Args:
        text: Input text

    Returns:
        LanguageInfo with detection results
    """
    if not text or not text.strip():
        return LanguageInfo(
            primary_language="en",
            telugu_ratio=0.0,
            english_ratio=0.0,
            mixed_ratio=0.0,
            has_telugu_script=False,
            has_romanized_telugu=False,
            confidence=0.0,
        )

    # Count characters
    telugu_chars = 0
    english_chars = 0
    total_chars = 0

    for char in text:
        if char.isalpha():
            total_chars += 1
            code_point = ord(char)
            if TELUGU_RANGE[0] <= code_point <= TELUGU_RANGE[1]:
                telugu_chars += 1
            elif char.isascii():
                english_chars += 1

    has_telugu_script = telugu_chars > 0

    # Check for Romanized Telugu patterns
    text_lower = text.lower()
    romanized_matches = 0
    for pattern in TELUGU_ROMANIZED_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        romanized_matches += len(matches)

    has_romanized_telugu = romanized_matches >= 2

    # Count English words
    words = re.findall(r"\b\w+\b", text_lower)
    english_word_count = sum(1 for w in words if w in ENGLISH_STOP_WORDS)
    english_word_ratio = english_word_count / len(words) if words else 0.0

    # Calculate ratios
    if total_chars > 0:
        telugu_ratio = telugu_chars / total_chars
        english_ratio = english_chars / total_chars
    else:
        telugu_ratio = 0.0
        english_ratio = 0.0

    # Adjust for Romanized Telugu
    if has_romanized_telugu and not has_telugu_script:
        # Romanized Telugu appears as English chars but is actually Telugu
        romanized_adjustment = min(0.5, romanized_matches * 0.1)
        telugu_ratio += romanized_adjustment
        english_ratio = max(0, english_ratio - romanized_adjustment)

    # Determine primary language
    if has_telugu_script:
        if telugu_ratio > 0.7:
            primary_language = "te"
            mixed_ratio = english_ratio
        elif english_ratio > 0.7:
            primary_language = "en"
            mixed_ratio = telugu_ratio
        else:
            primary_language = "mixed"
            mixed_ratio = 1.0 - abs(telugu_ratio - english_ratio)
    else:
        # No Telugu script - check for Romanized Telugu
        if has_romanized_telugu:
            if english_word_ratio > 0.4:
                primary_language = "mixed"
                mixed_ratio = 0.5
            else:
                primary_language = "te"
                mixed_ratio = english_word_ratio
                telugu_ratio = 1.0 - english_word_ratio
                english_ratio = english_word_ratio
        else:
            primary_language = "en"
            mixed_ratio = 0.0

    # Calculate confidence
    if primary_language == "te":
        confidence = telugu_ratio
    elif primary_language == "en":
        confidence = (
            english_ratio
            if not has_romanized_telugu
            else 1.0 - (romanized_matches * 0.1)
        )
    else:
        confidence = mixed_ratio

    return LanguageInfo(
        primary_language=primary_language,
        telugu_ratio=round(telugu_ratio, 3),
        english_ratio=round(english_ratio, 3),
        mixed_ratio=round(mixed_ratio, 3),
        has_telugu_script=has_telugu_script,
        has_romanized_telugu=has_romanized_telugu,
        confidence=round(max(0.0, min(1.0, confidence)), 3),
    )


def split_by_language(text: str) -> List[Tuple[str, str]]:
    """
    Split text into segments by language.

    Args:
        text: Input text

    Returns:
        List of (segment, language) tuples
    """
    if not text:
        return []

    segments = []
    current_segment = []
    current_lang = None

    for char in text:
        if char.isalpha():
            code_point = ord(char)
            if TELUGU_RANGE[0] <= code_point <= TELUGU_RANGE[1]:
                char_lang = "te"
            elif char.isascii():
                char_lang = "en"
            else:
                char_lang = "other"

            if current_lang is None:
                current_lang = char_lang
            elif char_lang != current_lang and char_lang != "other":
                # Language changed - save segment
                segment_text = "".join(current_segment).strip()
                if segment_text:
                    segments.append((segment_text, current_lang))
                current_segment = []
                current_lang = char_lang

        current_segment.append(char)

    # Add final segment
    segment_text = "".join(current_segment).strip()
    if segment_text and current_lang:
        segments.append((segment_text, current_lang))

    return segments


def is_code_switched(text: str) -> bool:
    """
    Check if text contains code-switching (mixing Telugu and English).

    Args:
        text: Input text

    Returns:
        True if code-switching detected
    """
    info = detect_language(text)
    return info.is_mixed or (info.has_telugu_script and info.english_ratio > 0.1)
