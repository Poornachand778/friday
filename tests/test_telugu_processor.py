"""
Tests for Telugu-English Processor
====================================

Tests language detection, segmentation, density calculation,
keyword extraction, and code-switch detection.

Run with: pytest tests/test_telugu_processor.py -v
"""

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from memory.telugu.processor import (
    TeluguEnglishProcessor,
    TextSegment,
    ProcessedText,
    calculate_telugu_density,
    detect_telugu,
    get_density_category,
    has_telugu,
    get_dominant_language,
    DEFAULT_TELUGU_STOPWORDS,
)


# =========================================================================
# detect_telugu
# =========================================================================


class TestDetectTelugu:
    """Test single character Telugu detection"""

    def test_telugu_char(self):
        assert detect_telugu("అ") is True
        assert detect_telugu("క") is True
        assert detect_telugu("ం") is True

    def test_english_char(self):
        assert detect_telugu("A") is False
        assert detect_telugu("z") is False

    def test_digit(self):
        assert detect_telugu("5") is False

    def test_punctuation(self):
        assert detect_telugu("!") is False

    def test_space(self):
        assert detect_telugu(" ") is False

    def test_empty_string(self):
        assert detect_telugu("") is False

    def test_multi_char_string(self):
        assert detect_telugu("అక") is False  # len != 1


# =========================================================================
# calculate_telugu_density
# =========================================================================


class TestTeluguDensity:
    """Test Telugu density calculation"""

    def test_all_telugu(self):
        density = calculate_telugu_density("అక్కడ")
        assert density > 0.5

    def test_all_english(self):
        density = calculate_telugu_density("Hello world")
        assert density == 0.0

    def test_mixed(self):
        density = calculate_telugu_density("Boss, బాగుంది")
        assert 0.0 < density < 1.0

    def test_empty(self):
        assert calculate_telugu_density("") == 0.0

    def test_only_spaces(self):
        assert calculate_telugu_density("   ") == 0.0


# =========================================================================
# get_density_category
# =========================================================================


class TestDensityCategory:
    """Test density categorization"""

    def test_high(self):
        assert get_density_category(0.5) == "high"
        assert get_density_category(0.41) == "high"

    def test_medium(self):
        assert get_density_category(0.3) == "medium"
        assert get_density_category(0.16) == "medium"

    def test_low(self):
        assert get_density_category(0.1) == "low"
        assert get_density_category(0.01) == "low"

    def test_none(self):
        assert get_density_category(0.0) == "none"


# =========================================================================
# has_telugu
# =========================================================================


class TestHasTelugu:
    """Test has_telugu convenience function"""

    def test_telugu_text(self):
        assert has_telugu("నేను") is True

    def test_mixed_text(self):
        assert has_telugu("Boss, నేను ready") is True

    def test_english_only(self):
        assert has_telugu("Hello world") is False

    def test_empty(self):
        assert has_telugu("") is False


# =========================================================================
# TextSegment
# =========================================================================


class TestTextSegment:
    """Test TextSegment dataclass"""

    def test_length_property(self):
        seg = TextSegment(text="Hello", language="en", start=0, end=5)
        assert seg.length == 5

    def test_telugu_segment(self):
        seg = TextSegment(text="బాగుంది", language="te", start=0, end=7)
        assert seg.language == "te"

    def test_neutral_segment(self):
        seg = TextSegment(text=", ", language="neutral", start=5, end=7)
        assert seg.language == "neutral"


# =========================================================================
# ProcessedText
# =========================================================================


class TestProcessedText:
    """Test ProcessedText dataclass"""

    def test_is_mixed_true(self):
        segments = [
            TextSegment(text="Hello", language="en", start=0, end=5),
            TextSegment(text="బాగుంది", language="te", start=6, end=13),
        ]
        result = ProcessedText(
            original="Hello బాగుంది",
            segments=segments,
            dominant_language="mixed",
            telugu_density=0.4,
            density_category="medium",
            code_switch_count=1,
        )
        assert result.is_mixed is True

    def test_is_mixed_false(self):
        segments = [
            TextSegment(text="Hello world", language="en", start=0, end=11),
        ]
        result = ProcessedText(
            original="Hello world",
            segments=segments,
            dominant_language="en",
            telugu_density=0.0,
            density_category="none",
            code_switch_count=0,
        )
        assert result.is_mixed is False

    def test_telugu_text_property(self):
        segments = [
            TextSegment(text="Hello ", language="en", start=0, end=6),
            TextSegment(text="బాగుంది", language="te", start=6, end=13),
        ]
        result = ProcessedText(
            original="Hello బాగుంది",
            segments=segments,
            dominant_language="mixed",
            telugu_density=0.4,
            density_category="medium",
            code_switch_count=1,
        )
        assert "బాగుంది" in result.telugu_text

    def test_english_text_property(self):
        segments = [
            TextSegment(text="Hello", language="en", start=0, end=5),
            TextSegment(text="బాగుంది", language="te", start=6, end=13),
        ]
        result = ProcessedText(
            original="Hello బాగుంది",
            segments=segments,
            dominant_language="mixed",
            telugu_density=0.4,
            density_category="medium",
            code_switch_count=1,
        )
        assert "Hello" in result.english_text

    def test_to_dict(self):
        result = ProcessedText(
            original="test",
            segments=[],
            dominant_language="en",
            telugu_density=0.0,
            density_category="none",
            code_switch_count=0,
            telugu_keywords=["word"],
            english_keywords=["test"],
        )
        d = result.to_dict()
        assert d["dominant_language"] == "en"
        assert d["telugu_density"] == 0.0
        assert d["telugu_keywords"] == ["word"]
        assert d["english_keywords"] == ["test"]
        assert "is_mixed" in d


# =========================================================================
# TeluguEnglishProcessor - Basic
# =========================================================================


class TestProcessorBasic:
    """Test processor initialization and basic processing"""

    def test_init_default(self):
        processor = TeluguEnglishProcessor()
        assert processor.min_keyword_length == 2
        assert len(processor.telugu_stopwords) > 0
        assert len(processor.english_stopwords) > 0

    def test_init_custom_keyword_length(self):
        processor = TeluguEnglishProcessor(min_keyword_length=5)
        assert processor.min_keyword_length == 5

    def test_process_empty(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("")
        assert result.dominant_language == "none"
        assert result.telugu_density == 0.0
        assert result.code_switch_count == 0
        assert result.segments == []

    def test_process_english_only(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("Hello, how are you doing today?")
        assert result.dominant_language == "en"
        assert result.telugu_density == 0.0
        assert result.density_category == "none"

    def test_process_telugu_only(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("నేను బాగున్నాను")
        assert result.dominant_language == "te"
        assert result.telugu_density > 0.5
        assert result.density_category == "high"


# =========================================================================
# TeluguEnglishProcessor - Code Switching
# =========================================================================


class TestCodeSwitching:
    """Test code-switch detection"""

    def test_mixed_text(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("Boss, climax scene బాగుంది")
        assert result.code_switch_count >= 1
        assert result.is_mixed

    def test_no_switches(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("Just pure English text here.")
        assert result.code_switch_count == 0

    def test_multiple_switches(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("Hello నేను good అని feel అవుతున్నాను")
        assert result.code_switch_count >= 2


# =========================================================================
# TeluguEnglishProcessor - Segmentation
# =========================================================================


class TestSegmentation:
    """Test text segmentation"""

    def test_single_language_segment(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("Hello world")
        # Should have at least one English segment
        en_segments = [s for s in result.segments if s.language == "en"]
        assert len(en_segments) >= 1

    def test_segments_have_positions(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("Hello world")
        for seg in result.segments:
            assert seg.start >= 0
            assert seg.end > seg.start

    def test_merge_adjacent_same_language(self):
        processor = TeluguEnglishProcessor()
        # Adjacent English words should be merged
        result = processor.process("Hello world test")
        en_segments = [s for s in result.segments if s.language == "en"]
        # Should be merged into fewer segments
        assert len(en_segments) <= 2


# =========================================================================
# TeluguEnglishProcessor - Keyword Extraction
# =========================================================================


class TestKeywordExtraction:
    """Test keyword extraction"""

    def test_english_keywords_extracted(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("The climax scene needs better dialogue and emotion")
        assert "climax" in result.english_keywords
        assert "scene" in result.english_keywords

    def test_english_stopwords_filtered(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("The quick brown fox is very fast")
        assert "the" not in result.english_keywords
        assert "is" not in result.english_keywords
        assert "very" not in result.english_keywords

    def test_telugu_keywords_extracted(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("ఈ సినిమా చాలా బాగుంది")
        # Should have keywords (non-stopwords)
        assert len(result.telugu_keywords) > 0

    def test_telugu_stopwords_filtered(self):
        processor = TeluguEnglishProcessor()
        # "ఈ" and "ఆ" are stopwords
        result = processor.process("ఈ ఆ మరియు")
        for kw in result.telugu_keywords:
            assert kw not in DEFAULT_TELUGU_STOPWORDS

    def test_extract_keywords_convenience(self):
        processor = TeluguEnglishProcessor()
        keywords = processor.extract_keywords("The climax scene is powerful")
        assert "climax" in keywords
        assert "powerful" in keywords

    def test_short_words_filtered(self):
        processor = TeluguEnglishProcessor(min_keyword_length=4)
        result = processor.process("Go run and test it now please")
        for kw in result.english_keywords:
            assert len(kw) >= 4


# =========================================================================
# TeluguEnglishProcessor - Dominance
# =========================================================================


class TestDominance:
    """Test dominant language detection"""

    def test_english_dominant(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("This is a long English sentence with no Telugu.")
        assert result.dominant_language == "en"

    def test_telugu_dominant(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("నేను చాలా బాగున్నాను ఇప్పుడు")
        assert result.dominant_language == "te"

    def test_is_telugu_dominant_method(self):
        processor = TeluguEnglishProcessor()
        assert processor.is_telugu_dominant("నేను చాలా బాగున్నాను") is True
        assert processor.is_telugu_dominant("Hello world") is False


# =========================================================================
# TeluguEnglishProcessor - Normalization
# =========================================================================


class TestNormalization:
    """Test Telugu text normalization"""

    def test_normalize_returns_string(self):
        processor = TeluguEnglishProcessor()
        result = processor.normalize_telugu("తెలుగు")
        assert isinstance(result, str)

    def test_normalize_english_unchanged(self):
        processor = TeluguEnglishProcessor()
        result = processor.normalize_telugu("Hello world")
        assert result == "Hello world"

    def test_normalize_empty(self):
        processor = TeluguEnglishProcessor()
        assert processor.normalize_telugu("") == ""


# =========================================================================
# TeluguEnglishProcessor - Convenience Functions
# =========================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions"""

    def test_get_dominant_language_english(self):
        assert get_dominant_language("Hello world, how are you?") == "en"

    def test_get_dominant_language_telugu(self):
        assert get_dominant_language("నేను చాలా బాగున్నాను") == "te"


# =========================================================================
# TeluguEnglishProcessor - Stopwords File
# =========================================================================


class TestStopwordsFile:
    """Test custom stopwords loading"""

    def test_load_from_file(self, tmp_path):
        stopwords_file = tmp_path / "stopwords.txt"
        stopwords_file.write_text("custom_stop\nanother_stop\n", encoding="utf-8")
        processor = TeluguEnglishProcessor(stopwords_file=stopwords_file)
        assert "custom_stop" in processor.telugu_stopwords
        assert "another_stop" in processor.telugu_stopwords

    def test_nonexistent_file_ok(self):
        processor = TeluguEnglishProcessor(stopwords_file=Path("/nonexistent"))
        assert len(processor.telugu_stopwords) > 0  # Still has defaults


# =========================================================================
# Edge Cases
# =========================================================================


class TestEdgeCases:
    """Test edge cases"""

    def test_only_numbers(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("12345 67890")
        assert result.telugu_density == 0.0

    def test_only_punctuation(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("!@#$%^&*()")
        assert result.telugu_density == 0.0

    def test_single_telugu_char(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("అ")
        assert result.telugu_density > 0

    def test_very_long_text(self):
        processor = TeluguEnglishProcessor()
        text = "Hello world. " * 1000
        result = processor.process(text)
        assert result.dominant_language == "en"

    def test_unicode_emoji(self):
        processor = TeluguEnglishProcessor()
        result = processor.process("Hello 🎬 world")
        assert result.dominant_language == "en"
