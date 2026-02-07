"""
Tests for Citation Tracker
============================

Tests citation management, formatting, bibliography generation,
and context building with citations.

Run with: pytest tests/test_citation_tracker.py -v
"""

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from documents.models import (
    Chunk,
    Citation,
    Document,
    DocumentMetadata,
    DocumentSearchResult,
    DocumentType,
    DocumentLanguage,
)
from documents.retrieval.citation import CitationTracker


# =========================================================================
# Helpers
# =========================================================================


def _make_document(title="Story", author="Robert McKee", doc_id="doc-001"):
    return Document(
        id=doc_id,
        file_path="/test/story.pdf",
        file_hash="abc123",
        file_size=1000000,
        document_type=DocumentType.BOOK,
        metadata=DocumentMetadata(title=title, author=author),
        language=DocumentLanguage.ENGLISH,
        total_pages=300,
    )


def _make_chunk(
    content="Every scene must have a turning point.",
    page_range="pp. 45-47",
    chapter="Chapter 3",
    section=None,
    chunk_id=None,
    doc_id="doc-001",
):
    chunk = Chunk.create(
        document_id=doc_id,
        page_ids=["p1", "p2"],
        content=content,
        page_range=page_range,
        chunk_index=0,
        chapter=chapter,
        section=section,
    )
    if chunk_id:
        chunk.id = chunk_id
    return chunk


def _make_search_result(
    doc=None, chunk=None, similarity=0.9, highlight="turning point"
):
    doc = doc or _make_document()
    chunk = chunk or _make_chunk()
    citation = Citation(
        document_id=doc.id,
        document_title=doc.metadata.title,
        chunk_id=chunk.id,
        page_range=chunk.page_range,
        chapter=chunk.chapter,
        section=chunk.section,
        quote=highlight,
        relevance=similarity,
    )
    return DocumentSearchResult(
        chunk=chunk,
        document=doc,
        similarity=similarity,
        highlight=highlight,
        citation=citation,
    )


# =========================================================================
# Citation Model
# =========================================================================


class TestCitationModel:
    """Test Citation dataclass formatting"""

    def test_format_inline(self):
        c = Citation(
            document_id="d1",
            document_title="Story",
            chunk_id="c1",
            page_range="pp. 45-47",
            chapter="Ch 3",
            section=None,
            quote="test",
            relevance=0.9,
        )
        assert c.format_inline() == "[Story, pp. 45-47]"

    def test_format_footnote(self):
        c = Citation(
            document_id="d1",
            document_title="Story",
            chunk_id="c1",
            page_range="pp. 45-47",
            chapter="Chapter 3",
            section=None,
            quote="test",
            relevance=0.9,
        )
        result = c.format_footnote()
        assert "Story" in result
        assert "Chapter 3" in result
        assert "pp. 45-47" in result

    def test_format_footnote_no_chapter(self):
        c = Citation(
            document_id="d1",
            document_title="Story",
            chunk_id="c1",
            page_range="p. 10",
            chapter=None,
            section=None,
            quote="test",
            relevance=0.9,
        )
        result = c.format_footnote()
        assert "Story" in result
        assert "p. 10" in result

    def test_format_default(self):
        c = Citation(
            document_id="d1",
            document_title="Story",
            chunk_id="c1",
            page_range="p. 1",
            chapter=None,
            section=None,
            quote="test",
            relevance=0.9,
        )
        result = c.format("unknown_style")
        assert "Source:" in result


# =========================================================================
# CitationTracker - Basic Operations
# =========================================================================


class TestCitationTrackerBasic:
    """Test basic citation operations"""

    def test_init_default_style(self):
        tracker = CitationTracker()
        assert tracker.style == "inline"
        assert tracker.count == 0

    def test_init_custom_style(self):
        tracker = CitationTracker(style="footnote")
        assert tracker.style == "footnote"

    def test_add_citation(self):
        tracker = CitationTracker()
        doc = _make_document()
        chunk = _make_chunk()
        num = tracker.add_citation(chunk, doc, 0.9)
        assert num == 1
        assert tracker.count == 1

    def test_add_multiple_citations(self):
        tracker = CitationTracker()
        doc = _make_document()
        c1 = _make_chunk(content="First chunk", chunk_id="c1")
        c2 = _make_chunk(content="Second chunk", chunk_id="c2")
        num1 = tracker.add_citation(c1, doc, 0.9)
        num2 = tracker.add_citation(c2, doc, 0.8)
        assert num1 == 1
        assert num2 == 2
        assert tracker.count == 2

    def test_deduplication(self):
        """Same chunk cited twice returns same number"""
        tracker = CitationTracker()
        doc = _make_document()
        chunk = _make_chunk(chunk_id="c1")
        num1 = tracker.add_citation(chunk, doc, 0.9)
        num2 = tracker.add_citation(chunk, doc, 0.8)
        assert num1 == num2
        assert tracker.count == 1

    def test_clear(self):
        tracker = CitationTracker()
        doc = _make_document()
        chunk = _make_chunk()
        tracker.add_citation(chunk, doc, 0.9)
        tracker.clear()
        assert tracker.count == 0

    def test_get_citation_number(self):
        tracker = CitationTracker()
        doc = _make_document()
        chunk = _make_chunk(chunk_id="c1")
        tracker.add_citation(chunk, doc, 0.9)
        assert tracker.get_citation_number("c1") == 1

    def test_get_citation_number_unknown(self):
        tracker = CitationTracker()
        assert tracker.get_citation_number("nonexistent") is None

    def test_citations_property_returns_copy(self):
        tracker = CitationTracker()
        doc = _make_document()
        chunk = _make_chunk()
        tracker.add_citation(chunk, doc, 0.9)
        citations = tracker.citations
        citations.clear()
        assert tracker.count == 1  # Original unaffected

    def test_add_citation_with_custom_quote(self):
        tracker = CitationTracker()
        doc = _make_document()
        chunk = _make_chunk()
        tracker.add_citation(chunk, doc, 0.9, quote="Custom quote text")
        assert tracker.citations[0].quote == "Custom quote text"


# =========================================================================
# CitationTracker - Formatting
# =========================================================================


class TestCitationTrackerFormatting:
    """Test citation reference formatting"""

    def test_format_inline_reference(self):
        tracker = CitationTracker()
        doc = _make_document(title="Story")
        chunk = _make_chunk(chunk_id="c1", page_range="pp. 45-47")
        tracker.add_citation(chunk, doc, 0.9)
        ref = tracker.format_inline_reference("c1")
        assert ref == "[Story, pp. 45-47]"

    def test_format_inline_reference_unknown(self):
        tracker = CitationTracker()
        assert tracker.format_inline_reference("unknown") == ""

    def test_format_numbered_reference(self):
        tracker = CitationTracker()
        doc = _make_document()
        chunk = _make_chunk(chunk_id="c1")
        tracker.add_citation(chunk, doc, 0.9)
        ref = tracker.format_numbered_reference("c1")
        assert ref == "[1]"

    def test_format_numbered_reference_unknown(self):
        tracker = CitationTracker()
        assert tracker.format_numbered_reference("unknown") == ""

    def test_format_numbered_multiple(self):
        tracker = CitationTracker()
        doc = _make_document()
        c1 = _make_chunk(chunk_id="c1")
        c2 = _make_chunk(chunk_id="c2")
        tracker.add_citation(c1, doc, 0.9)
        tracker.add_citation(c2, doc, 0.8)
        assert tracker.format_numbered_reference("c1") == "[1]"
        assert tracker.format_numbered_reference("c2") == "[2]"


# =========================================================================
# CitationTracker - Bibliography
# =========================================================================


class TestBibliography:
    """Test bibliography formatting"""

    def test_empty_bibliography(self):
        tracker = CitationTracker()
        assert tracker.format_bibliography() == ""

    def test_bibliography_contains_title(self):
        tracker = CitationTracker()
        doc = _make_document(title="Story by McKee")
        chunk = _make_chunk(chunk_id="c1")
        tracker.add_citation(chunk, doc, 0.9)
        bib = tracker.format_bibliography()
        assert "Story by McKee" in bib

    def test_bibliography_contains_page_range(self):
        tracker = CitationTracker()
        doc = _make_document()
        chunk = _make_chunk(chunk_id="c1", page_range="pp. 100-102")
        tracker.add_citation(chunk, doc, 0.9)
        bib = tracker.format_bibliography()
        assert "pp. 100-102" in bib

    def test_bibliography_contains_chapter(self):
        tracker = CitationTracker()
        doc = _make_document()
        chunk = _make_chunk(chunk_id="c1", chapter="Chapter 5")
        tracker.add_citation(chunk, doc, 0.9)
        bib = tracker.format_bibliography()
        assert "Chapter 5" in bib

    def test_bibliography_has_sources_header(self):
        tracker = CitationTracker()
        doc = _make_document()
        tracker.add_citation(_make_chunk(chunk_id="c1"), doc, 0.9)
        bib = tracker.format_bibliography()
        assert "Sources" in bib

    def test_bibliography_numbered(self):
        tracker = CitationTracker()
        doc = _make_document()
        tracker.add_citation(_make_chunk(chunk_id="c1"), doc, 0.9)
        tracker.add_citation(_make_chunk(chunk_id="c2"), doc, 0.8)
        bib = tracker.format_bibliography()
        assert "1." in bib
        assert "2." in bib

    def test_footnote_style_includes_quote(self):
        tracker = CitationTracker(style="footnote")
        doc = _make_document()
        chunk = _make_chunk(
            chunk_id="c1",
            content="Every scene must turn. That is the fundamental principle.",
        )
        tracker.add_citation(chunk, doc, 0.9)
        bib = tracker.format_bibliography()
        assert ">" in bib  # Quote marker


# =========================================================================
# CitationTracker - Context Building
# =========================================================================


class TestContextWithCitations:
    """Test get_context_with_citations"""

    def test_basic_context(self):
        tracker = CitationTracker()
        doc = _make_document(title="Story")
        chunk = _make_chunk(content="Scenes need turning points.")
        result = _make_search_result(doc=doc, chunk=chunk)
        context, citations = tracker.get_context_with_citations([result])
        assert "Story" in context
        assert "Scenes need turning points." in context
        assert len(citations) == 1

    def test_context_clears_previous(self):
        tracker = CitationTracker()
        doc = _make_document()
        # First call
        tracker.get_context_with_citations([_make_search_result(doc=doc)])
        # Second call should start fresh
        context, citations = tracker.get_context_with_citations(
            [_make_search_result(doc=doc)]
        )
        assert len(citations) == 1  # Not 2

    def test_max_chars_limits_context(self):
        tracker = CitationTracker()
        doc = _make_document()
        results = []
        for i in range(10):
            chunk = _make_chunk(
                content=f"Long content block {i}. " * 50,
                chunk_id=f"c{i}",
            )
            results.append(_make_search_result(doc=doc, chunk=chunk))
        context, citations = tracker.get_context_with_citations(results, max_chars=500)
        assert len(context) <= 600  # Allow some margin for truncation

    def test_without_quotes(self):
        tracker = CitationTracker()
        doc = _make_document(title="Story")
        result = _make_search_result(doc=doc)
        context, _ = tracker.get_context_with_citations([result], include_quotes=False)
        assert "Story" in context

    def test_numbered_references_in_context(self):
        tracker = CitationTracker()
        doc = _make_document()
        result = _make_search_result(doc=doc)
        context, _ = tracker.get_context_with_citations([result])
        assert "[1]" in context

    def test_empty_results(self):
        tracker = CitationTracker()
        context, citations = tracker.get_context_with_citations([])
        assert context == ""
        assert citations == []


# =========================================================================
# CitationTracker - Response Formatting
# =========================================================================


class TestResponseFormatting:
    """Test format_response_with_citations"""

    def test_appends_bibliography(self):
        tracker = CitationTracker()
        doc = _make_document(title="Story")
        tracker.add_citation(_make_chunk(chunk_id="c1"), doc, 0.9)
        response = tracker.format_response_with_citations("Here is the analysis.")
        assert "Here is the analysis." in response
        assert "Sources" in response

    def test_without_bibliography(self):
        tracker = CitationTracker()
        doc = _make_document()
        tracker.add_citation(_make_chunk(chunk_id="c1"), doc, 0.9)
        response = tracker.format_response_with_citations(
            "Analysis here.", add_bibliography=False
        )
        assert response == "Analysis here."

    def test_no_citations_returns_unchanged(self):
        tracker = CitationTracker()
        response = tracker.format_response_with_citations("Just a response.")
        assert response == "Just a response."


# =========================================================================
# CitationTracker - Quote Extraction
# =========================================================================


class TestQuoteExtraction:
    """Test _extract_quote helper"""

    def test_short_content(self):
        tracker = CitationTracker()
        quote = tracker._extract_quote("A short sentence.")
        assert quote == "A short sentence."

    def test_long_content_truncated(self):
        tracker = CitationTracker()
        long_text = "This is a sentence. " * 30  # ~600 chars
        quote = tracker._extract_quote(long_text, max_length=100)
        assert len(quote) <= 110  # Allow some margin

    def test_strips_markdown(self):
        tracker = CitationTracker()
        quote = tracker._extract_quote(
            "**Bold** and *italic* text here with enough length to be selected."
        )
        assert "**" not in quote
        assert "*" not in quote

    def test_skips_short_lines(self):
        tracker = CitationTracker()
        text = "Hi\nOk\nThis is a much longer meaningful line that should be selected."
        quote = tracker._extract_quote(text)
        assert "much longer" in quote

    def test_sentence_boundary_truncation(self):
        tracker = CitationTracker()
        text = "First sentence here. Second sentence that is also here. Third one comes after."
        quote = tracker._extract_quote(text, max_length=50)
        # Should end at sentence boundary
        assert quote.endswith(".") or quote.endswith("...")
