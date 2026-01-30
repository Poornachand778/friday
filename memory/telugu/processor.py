"""
Telugu-English Processor
========================

Handles language detection, segmentation, and keyword extraction
for Telugu-English code-switched text.

Features:
    - Character-level language detection
    - Segment-based processing
    - Telugu density calculation
    - Code-switch point detection
    - Telugu keyword extraction for search
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

LOGGER = logging.getLogger(__name__)

# Telugu Unicode range: U+0C00 to U+0C7F
TELUGU_START = 0x0C00
TELUGU_END = 0x0C7F

# Common Telugu stopwords (add more as needed)
DEFAULT_TELUGU_STOPWORDS = {
    "మరియు",
    "ఒక",
    "ఈ",
    "ఆ",
    "కు",
    "లో",
    "తో",
    "యొక్క",
    "కి",
    "ని",
    "ను",
    "తన",
    "ఇది",
    "అది",
    "ఎందుకు",
    "ఏమి",
    "ఎలా",
    "ఏం",
    "అవును",
    "కాదు",
    "లేదు",
    "నేను",
    "నీవు",
    "అతను",
    "ఆమె",
    "మేము",
    "వారు",
    "ఉంది",
    "ఉన్నారు",
    "ఉన్నాను",
    "ఉన్నావు",
}


def detect_telugu(char: str) -> bool:
    """Check if a character is Telugu"""
    if len(char) != 1:
        return False
    code = ord(char)
    return TELUGU_START <= code <= TELUGU_END


def calculate_telugu_density(text: str) -> float:
    """
    Calculate Telugu character density in text.

    Returns:
        Float between 0.0 (no Telugu) and 1.0 (all Telugu)
    """
    if not text:
        return 0.0

    telugu_chars = sum(1 for c in text if detect_telugu(c))
    return telugu_chars / len(text)


def get_density_category(density: float) -> str:
    """
    Get density category from ratio.

    Categories:
        - "high": > 40% Telugu
        - "medium": 15-40% Telugu
        - "low": 1-15% Telugu
        - "none": 0% Telugu
    """
    if density > 0.4:
        return "high"
    elif density > 0.15:
        return "medium"
    elif density > 0:
        return "low"
    return "none"


@dataclass
class TextSegment:
    """A segment of text in a single language"""

    text: str
    language: str  # "te", "en", "neutral"
    start: int  # Start position in original text
    end: int  # End position in original text

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class ProcessedText:
    """Result of processing Telugu-English mixed text"""

    original: str
    segments: List[TextSegment]
    dominant_language: str
    telugu_density: float
    density_category: str
    code_switch_count: int
    telugu_keywords: List[str] = field(default_factory=list)
    english_keywords: List[str] = field(default_factory=list)

    @property
    def is_mixed(self) -> bool:
        """Check if text has both Telugu and English"""
        langs = {s.language for s in self.segments if s.language != "neutral"}
        return len(langs) > 1

    @property
    def telugu_text(self) -> str:
        """Get only Telugu segments"""
        return " ".join(s.text for s in self.segments if s.language == "te")

    @property
    def english_text(self) -> str:
        """Get only English segments"""
        return " ".join(s.text for s in self.segments if s.language == "en")

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "dominant_language": self.dominant_language,
            "telugu_density": self.telugu_density,
            "density_category": self.density_category,
            "code_switch_count": self.code_switch_count,
            "is_mixed": self.is_mixed,
            "telugu_keywords": self.telugu_keywords,
            "english_keywords": self.english_keywords,
        }


class TeluguEnglishProcessor:
    """
    Processor for Telugu-English code-switched text.

    Handles:
        - Language segmentation
        - Density calculation
        - Keyword extraction
        - Code-switch detection

    Usage:
        processor = TeluguEnglishProcessor()
        result = processor.process("Boss, climax scene బాగుంది")

        print(result.telugu_density)  # 0.27
        print(result.telugu_keywords)  # ["బాగుంది"]
    """

    def __init__(
        self,
        stopwords_file: Optional[Path] = None,
        min_keyword_length: int = 2,
    ):
        self.min_keyword_length = min_keyword_length

        # Load stopwords
        self.telugu_stopwords: Set[str] = DEFAULT_TELUGU_STOPWORDS.copy()
        if stopwords_file and stopwords_file.exists():
            self._load_stopwords(stopwords_file)

        # English stopwords (basic)
        self.english_stopwords: Set[str] = {
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
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "although",
            "since",
            "unless",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "its",
            "they",
            "them",
            "their",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "whom",
        }

    def _load_stopwords(self, path: Path) -> None:
        """Load Telugu stopwords from file"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.telugu_stopwords.add(word)
            LOGGER.debug("Loaded %d Telugu stopwords", len(self.telugu_stopwords))
        except Exception as e:
            LOGGER.warning("Failed to load stopwords: %s", e)

    def process(self, text: str) -> ProcessedText:
        """
        Process text and extract language information.

        Args:
            text: Input text (Telugu, English, or mixed)

        Returns:
            ProcessedText with segments and metadata
        """
        if not text:
            return ProcessedText(
                original="",
                segments=[],
                dominant_language="none",
                telugu_density=0.0,
                density_category="none",
                code_switch_count=0,
            )

        # Segment text by language
        segments = self._segment_text(text)

        # Calculate density
        density = calculate_telugu_density(text)
        density_category = get_density_category(density)

        # Determine dominant language
        telugu_len = sum(s.length for s in segments if s.language == "te")
        english_len = sum(s.length for s in segments if s.language == "en")

        if telugu_len > english_len * 1.5:
            dominant = "te"
        elif english_len > telugu_len * 1.5:
            dominant = "en"
        else:
            dominant = "mixed"

        # Count code switches
        switch_count = self._count_code_switches(segments)

        # Extract keywords
        telugu_keywords = self._extract_telugu_keywords(segments)
        english_keywords = self._extract_english_keywords(segments)

        return ProcessedText(
            original=text,
            segments=segments,
            dominant_language=dominant,
            telugu_density=density,
            density_category=density_category,
            code_switch_count=switch_count,
            telugu_keywords=telugu_keywords,
            english_keywords=english_keywords,
        )

    def _segment_text(self, text: str) -> List[TextSegment]:
        """Segment text into language chunks"""
        segments: List[TextSegment] = []
        current_segment = ""
        current_lang: Optional[str] = None
        start_pos = 0

        for i, char in enumerate(text):
            char_lang = self._detect_char_language(char)

            if current_lang is None:
                current_lang = char_lang
                current_segment = char
                start_pos = i
            elif char_lang == current_lang or char_lang == "neutral":
                current_segment += char
            else:
                # Language changed - save current segment
                if current_segment.strip():
                    segments.append(
                        TextSegment(
                            text=current_segment,
                            language=current_lang,
                            start=start_pos,
                            end=i,
                        )
                    )
                current_segment = char
                current_lang = char_lang
                start_pos = i

        # Save final segment
        if current_segment.strip():
            segments.append(
                TextSegment(
                    text=current_segment,
                    language=current_lang or "neutral",
                    start=start_pos,
                    end=len(text),
                )
            )

        # Merge adjacent segments of same language
        segments = self._merge_adjacent_segments(segments)

        return segments

    def _detect_char_language(self, char: str) -> str:
        """Detect language of a single character"""
        if detect_telugu(char):
            return "te"
        elif char.isalpha():
            return "en"
        else:
            return "neutral"  # Punctuation, numbers, spaces

    def _merge_adjacent_segments(
        self, segments: List[TextSegment]
    ) -> List[TextSegment]:
        """Merge adjacent segments of the same language"""
        if not segments:
            return segments

        merged: List[TextSegment] = []

        for segment in segments:
            if merged and merged[-1].language == segment.language:
                # Merge with previous
                prev = merged[-1]
                merged[-1] = TextSegment(
                    text=prev.text + segment.text,
                    language=prev.language,
                    start=prev.start,
                    end=segment.end,
                )
            elif (
                merged
                and segment.language == "neutral"
                and merged[-1].language != "neutral"
            ):
                # Append neutral to previous non-neutral
                prev = merged[-1]
                merged[-1] = TextSegment(
                    text=prev.text + segment.text,
                    language=prev.language,
                    start=prev.start,
                    end=segment.end,
                )
            else:
                merged.append(segment)

        return merged

    def _count_code_switches(self, segments: List[TextSegment]) -> int:
        """Count number of language switches"""
        switches = 0
        prev_lang = None

        for segment in segments:
            if segment.language in ("te", "en"):
                if prev_lang and prev_lang != segment.language:
                    switches += 1
                prev_lang = segment.language

        return switches

    def _extract_telugu_keywords(self, segments: List[TextSegment]) -> List[str]:
        """Extract Telugu keywords from segments"""
        keywords: List[str] = []

        for segment in segments:
            if segment.language == "te":
                # Split by whitespace and punctuation
                words = re.split(r"[\s\.,!?;:]+", segment.text)
                for word in words:
                    word = word.strip()
                    if (
                        len(word) >= self.min_keyword_length
                        and word not in self.telugu_stopwords
                        and any(detect_telugu(c) for c in word)
                    ):
                        keywords.append(word)

        return list(set(keywords))  # Deduplicate

    def _extract_english_keywords(self, segments: List[TextSegment]) -> List[str]:
        """Extract English keywords from segments"""
        keywords: List[str] = []

        for segment in segments:
            if segment.language == "en":
                # Split by whitespace and punctuation
                words = re.split(r"[\s\.,!?;:]+", segment.text)
                for word in words:
                    word = word.strip().lower()
                    if (
                        len(word) >= self.min_keyword_length
                        and word not in self.english_stopwords
                        and word.isalpha()
                    ):
                        keywords.append(word)

        return list(set(keywords))  # Deduplicate

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract all keywords (Telugu and English) from text.

        Convenience method that combines both language keywords.
        """
        result = self.process(text)
        return result.telugu_keywords + result.english_keywords

    def is_telugu_dominant(self, text: str, threshold: float = 0.4) -> bool:
        """Check if text is predominantly Telugu"""
        density = calculate_telugu_density(text)
        return density >= threshold

    def normalize_telugu(self, text: str) -> str:
        """
        Normalize Telugu text for better embedding.

        Standardizes vowel signs and common variations.
        """
        # Telugu-specific normalizations
        replacements = {
            "\u0c48": "\u0c46\u0c56",  # AI vowel sign
            "\u0c4c": "\u0c4a\u0c55",  # AU vowel sign
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text


# Convenience functions for simple usage
def has_telugu(text: str) -> bool:
    """Check if text contains any Telugu characters"""
    return any(detect_telugu(c) for c in text)


def get_dominant_language(text: str) -> str:
    """Get dominant language in text"""
    processor = TeluguEnglishProcessor()
    result = processor.process(text)
    return result.dominant_language
