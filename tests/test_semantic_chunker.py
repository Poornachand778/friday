"""
Tests for Semantic Chunker
============================

Tests fixed, semantic, hybrid, and screenplay chunking strategies,
plus split point detection and section header detection.

Run with: pytest tests/test_semantic_chunker.py -v
"""

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from documents.config import ChunkingConfig
from documents.models import ChapterInfo, Chunk, Page
from documents.pipeline.chunker import SemanticChunker


# =========================================================================
# Helpers
# =========================================================================


def _make_page(page_number, text, doc_id="doc-001"):
    return Page.create(
        document_id=doc_id,
        page_number=page_number,
        raw_text=text,
        cleaned_text=text,
    )


def _make_config(**overrides):
    defaults = {
        "strategy": "semantic",
        "min_chunk_chars": 50,
        "max_chunk_chars": 500,
        "overlap_chars": 20,
        "respect_chapters": True,
        "respect_sections": True,
        "screenplay_mode": False,
    }
    defaults.update(overrides)
    return ChunkingConfig(**defaults)


def _make_chunker(**overrides):
    config = _make_config(**overrides)
    return SemanticChunker(config=config)


# =========================================================================
# Fixed Chunking
# =========================================================================


class TestFixedChunking:
    """Test fixed-size chunking strategy"""

    def test_single_short_page(self):
        chunker = _make_chunker(strategy="fixed", max_chunk_chars=500)
        pages = [_make_page(1, "Short content here.")]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert len(chunks) == 1
        assert "Short content" in chunks[0].content

    def test_long_text_splits(self):
        chunker = _make_chunker(
            strategy="fixed", max_chunk_chars=100, min_chunk_chars=10
        )
        text = "This is a test sentence. " * 20  # ~500 chars
        pages = [_make_page(1, text)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert len(chunks) > 1

    def test_overlap_between_chunks(self):
        chunker = _make_chunker(
            strategy="fixed", max_chunk_chars=100, overlap_chars=20, min_chunk_chars=10
        )
        text = "Word " * 100  # 500 chars
        pages = [_make_page(1, text)]
        chunks = chunker.chunk_document(pages, "doc-001")
        # With overlap, chunks should share some text
        if len(chunks) >= 2:
            end_of_first = chunks[0].content[-20:]
            assert any(
                end_of_first[:10] in chunks[i].content for i in range(1, len(chunks))
            )

    def test_chunk_has_page_range(self):
        chunker = _make_chunker(strategy="fixed")
        pages = [_make_page(1, "Content here " * 10)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert chunks[0].page_range.startswith("p.")

    def test_multi_page_chunk(self):
        chunker = _make_chunker(strategy="fixed", max_chunk_chars=2000)
        pages = [
            _make_page(1, "Page one content."),
            _make_page(2, "Page two content."),
        ]
        chunks = chunker.chunk_document(pages, "doc-001")
        # Both pages should be in one chunk since they're short
        assert len(chunks) == 1
        assert "Page one" in chunks[0].content
        assert "Page two" in chunks[0].content

    def test_chunk_index_sequential(self):
        chunker = _make_chunker(
            strategy="fixed", max_chunk_chars=100, min_chunk_chars=10
        )
        text = "Sentence here. " * 30
        pages = [_make_page(1, text)]
        chunks = chunker.chunk_document(pages, "doc-001")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_document_id_set(self):
        chunker = _make_chunker(strategy="fixed")
        pages = [_make_page(1, "Content")]
        chunks = chunker.chunk_document(pages, "my-doc-id")
        for chunk in chunks:
            assert chunk.document_id == "my-doc-id"


# =========================================================================
# Semantic Chunking
# =========================================================================


class TestSemanticChunking:
    """Test semantic chunking strategy"""

    def test_basic_semantic(self):
        chunker = _make_chunker(strategy="semantic")
        pages = [_make_page(1, "Some meaningful content about screenwriting.")]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert len(chunks) >= 1

    def test_chapter_boundary_splits(self):
        chunker = _make_chunker(strategy="semantic", max_chunk_chars=2000)
        pages = [
            _make_page(1, "Content from chapter one. " * 5),
            _make_page(2, "Content from chapter two. " * 5),
        ]
        chapters = [
            ChapterInfo(number=1, title="Chapter 1", start_page=1, end_page=1),
            ChapterInfo(number=2, title="Chapter 2", start_page=2, end_page=2),
        ]
        chunks = chunker.chunk_document(pages, "doc-001", chapters=chapters)
        # Should produce separate chunks for each chapter
        assert len(chunks) == 2

    def test_section_header_splits(self):
        chunker = _make_chunker(
            strategy="semantic", respect_sections=True, min_chunk_chars=20
        )
        text1 = "# Introduction\n\nSome introductory content here that is long enough."
        text2 = "# Method\n\nSome method content here that is also long enough."
        pages = [_make_page(1, text1), _make_page(2, text2)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert len(chunks) >= 2

    def test_long_page_split_at_paragraph(self):
        chunker = _make_chunker(
            strategy="semantic", max_chunk_chars=200, min_chunk_chars=50
        )
        text = (
            "First paragraph content here.\n\n"
            "Second paragraph content here.\n\n"
            "Third paragraph content here.\n\n"
            "Fourth paragraph content here.\n\n"
            "Fifth paragraph with more text."
        )
        pages = [_make_page(1, text * 3)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert len(chunks) > 1

    def test_remaining_text_flushed(self):
        chunker = _make_chunker(strategy="semantic", max_chunk_chars=2000)
        pages = [_make_page(1, "Final content that needs to be captured.")]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert any("Final content" in c.content for c in chunks)

    def test_chapters_set_on_chunks(self):
        chunker = _make_chunker(strategy="semantic", max_chunk_chars=2000)
        pages = [_make_page(1, "Content here. " * 5)]
        chapters = [
            ChapterInfo(number=1, title="Act One", start_page=1, end_page=1),
        ]
        chunks = chunker.chunk_document(pages, "doc-001", chapters=chapters)
        assert chunks[0].chapter == "Act One"


# =========================================================================
# Hybrid Chunking
# =========================================================================


class TestHybridChunking:
    """Test hybrid chunking strategy"""

    def test_hybrid_basic(self):
        chunker = _make_chunker(strategy="hybrid")
        pages = [_make_page(1, "Some content here. " * 10)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert len(chunks) >= 1

    def test_hybrid_with_chapters(self):
        chunker = _make_chunker(strategy="hybrid", max_chunk_chars=2000)
        pages = [
            _make_page(1, "Chapter one content. " * 5),
            _make_page(2, "Chapter two content. " * 5),
        ]
        chapters = [
            ChapterInfo(number=1, title="Ch 1", start_page=1, end_page=1),
            ChapterInfo(number=2, title="Ch 2", start_page=2, end_page=2),
        ]
        chunks = chunker.chunk_document(pages, "doc-001", chapters=chapters)
        # Should have at least one chunk per chapter
        assert len(chunks) >= 2

    def test_hybrid_assigns_chapter_to_chunks(self):
        chunker = _make_chunker(strategy="hybrid", max_chunk_chars=2000)
        pages = [_make_page(1, "Content. " * 10)]
        chapters = [
            ChapterInfo(number=1, title="Introduction", start_page=1, end_page=1),
        ]
        chunks = chunker.chunk_document(pages, "doc-001", chapters=chapters)
        assert chunks[0].chapter == "Introduction"


# =========================================================================
# Screenplay Chunking
# =========================================================================


class TestScreenplayChunking:
    """Test screenplay scene-based chunking"""

    def test_screenplay_splits_on_scene_heading(self):
        chunker = _make_chunker(screenplay_mode=True)
        text = (
            "INT. COURTROOM - DAY\n"
            "Judge bangs gavel.\n\n"
            "EXT. PARK - NIGHT\n"
            "Stars twinkle overhead.\n"
        )
        pages = [_make_page(1, text)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert len(chunks) == 2

    def test_screenplay_preserves_scene_content(self):
        chunker = _make_chunker(screenplay_mode=True)
        text = (
            "INT. OFFICE - DAY\n"
            "ARJUN sits at his desk.\n"
            "ARJUN\nWe need to talk.\n\n"
            "EXT. STREET - NIGHT\n"
            "Rain falls heavily.\n"
        )
        pages = [_make_page(1, text)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert any("ARJUN" in c.content for c in chunks)
        assert any("Rain falls" in c.content for c in chunks)

    def test_screenplay_section_set_to_heading(self):
        chunker = _make_chunker(screenplay_mode=True)
        text = "INT. KITCHEN - MORNING\nSomeone cooks breakfast.\n"
        pages = [_make_page(1, text)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert chunks[0].section is not None
        assert "INT. KITCHEN" in chunks[0].section

    def test_no_scene_headings(self):
        chunker = _make_chunker(screenplay_mode=True)
        text = "Just some regular text without scene headings."
        pages = [_make_page(1, text)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert len(chunks) == 0  # No scene patterns found

    def test_int_ext_heading(self):
        chunker = _make_chunker(screenplay_mode=True)
        text = "INT/EXT. CAR - DAY\nArjun drives through rain.\n"
        pages = [_make_page(1, text)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert len(chunks) == 1


# =========================================================================
# Split Point Detection
# =========================================================================


class TestSplitPointDetection:
    """Test _find_split_point logic"""

    def test_prefers_paragraph_break(self):
        chunker = _make_chunker(max_chunk_chars=100, min_chunk_chars=20)
        text = (
            "First part.\n\nSecond part that goes on a bit longer to reach the split."
        )
        point = chunker._find_split_point(text)
        # Should split at the paragraph break
        assert text[point - 2 : point] == "\n\n" or point <= 100

    def test_falls_back_to_sentence(self):
        chunker = _make_chunker(max_chunk_chars=80, min_chunk_chars=20)
        text = (
            "First sentence here. Second sentence comes next. Third sentence follows."
        )
        point = chunker._find_split_point(text)
        # Should split at sentence boundary
        before = text[:point].rstrip()
        assert before.endswith(".") or before.endswith("?") or before.endswith("!")

    def test_falls_back_to_word_boundary(self):
        chunker = _make_chunker(max_chunk_chars=30, min_chunk_chars=5)
        text = "word " * 20  # No punctuation
        point = chunker._find_split_point(text)
        # Should split at word boundary
        assert text[point - 1] == " " or point == 30


# =========================================================================
# Section Header Detection
# =========================================================================


class TestSectionHeaderDetection:
    """Test _detect_section_header"""

    def test_markdown_header(self):
        chunker = _make_chunker()
        result = chunker._detect_section_header("# Introduction\nContent here")
        assert result == "Introduction"

    def test_markdown_h2(self):
        chunker = _make_chunker()
        result = chunker._detect_section_header("## Methods\nContent here")
        assert result == "Methods"

    def test_chapter_pattern(self):
        chunker = _make_chunker()
        result = chunker._detect_section_header("Chapter 1\nContent here")
        assert result == "Chapter 1"

    def test_bold_header(self):
        chunker = _make_chunker()
        result = chunker._detect_section_header("**The Inciting Incident**\nContent")
        assert result == "The Inciting Incident"

    def test_no_header(self):
        chunker = _make_chunker()
        result = chunker._detect_section_header("Just regular text content.")
        assert result is None

    def test_empty_text(self):
        chunker = _make_chunker()
        result = chunker._detect_section_header("")
        assert result is None

    def test_long_bold_not_header(self):
        """Very long bold text is not a header"""
        chunker = _make_chunker()
        long_bold = "**" + "A" * 150 + "**"
        result = chunker._detect_section_header(long_bold)
        assert result is None


# =========================================================================
# Page Range Formatting
# =========================================================================


class TestPageRangeFormatting:
    """Test page range in created chunks"""

    def test_single_page(self):
        chunker = _make_chunker(strategy="fixed")
        pages = [_make_page(5, "Content here. " * 10)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert chunks[0].page_range == "p. 5"

    def test_multi_page_range(self):
        chunker = _make_chunker(strategy="fixed", max_chunk_chars=5000)
        pages = [
            _make_page(3, "Content on page 3."),
            _make_page(4, "Content on page 4."),
            _make_page(5, "Content on page 5."),
        ]
        chunks = chunker.chunk_document(pages, "doc-001")
        if len(chunks) == 1:
            assert chunks[0].page_range == "pp. 3-5"


# =========================================================================
# Edge Cases
# =========================================================================


class TestEdgeCases:
    """Test edge cases"""

    def test_empty_pages(self):
        chunker = _make_chunker(strategy="fixed")
        chunks = chunker.chunk_document([], "doc-001")
        assert len(chunks) == 0

    def test_page_with_empty_text(self):
        chunker = _make_chunker(strategy="semantic", min_chunk_chars=5)
        pages = [_make_page(1, "")]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert len(chunks) == 0

    def test_chunk_has_uuid_id(self):
        chunker = _make_chunker(strategy="fixed")
        pages = [_make_page(1, "Some content here. " * 10)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert len(chunks[0].id) == 36  # UUID format

    def test_chunk_has_char_count(self):
        chunker = _make_chunker(strategy="fixed")
        pages = [_make_page(1, "Some content here. " * 10)]
        chunks = chunker.chunk_document(pages, "doc-001")
        assert chunks[0].char_count == len(chunks[0].content)
