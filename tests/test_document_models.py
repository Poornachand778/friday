"""
Tests for Document Processing Data Models
==========================================

Comprehensive tests for all enums, dataclasses, factory methods,
serialization (to_dict/from_dict), citation formatting, and edge cases
in documents/models.py.

Run with: pytest tests/test_document_models.py -v
"""

import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from documents.models import (
    Chunk,
    ChapterInfo,
    Citation,
    Document,
    DocumentLanguage,
    DocumentMetadata,
    DocumentSearchResult,
    DocumentType,
    OCRResult,
    Page,
    ProcessingResult,
    ProcessingStatus,
)


# =========================================================================
# Enum Tests
# =========================================================================


class TestDocumentType:
    """Tests for DocumentType enum."""

    def test_all_members_exist(self):
        assert DocumentType.BOOK.value == "book"
        assert DocumentType.SCREENPLAY.value == "screenplay"
        assert DocumentType.ARTICLE.value == "article"
        assert DocumentType.MANUAL.value == "manual"
        assert DocumentType.REFERENCE.value == "reference"
        assert DocumentType.CUSTOM.value == "custom"

    def test_member_count(self):
        assert len(DocumentType) == 6

    def test_is_str_enum(self):
        assert isinstance(DocumentType.BOOK, str)
        assert DocumentType.BOOK == "book"

    def test_lookup_by_value(self):
        assert DocumentType("book") is DocumentType.BOOK
        assert DocumentType("screenplay") is DocumentType.SCREENPLAY

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DocumentType("nonexistent")


class TestDocumentLanguage:
    """Tests for DocumentLanguage enum."""

    def test_all_members_exist(self):
        assert DocumentLanguage.ENGLISH.value == "en"
        assert DocumentLanguage.TELUGU.value == "te"
        assert DocumentLanguage.MIXED.value == "mixed"

    def test_member_count(self):
        assert len(DocumentLanguage) == 3

    def test_is_str_enum(self):
        assert isinstance(DocumentLanguage.ENGLISH, str)
        assert DocumentLanguage.ENGLISH == "en"

    def test_lookup_by_value(self):
        assert DocumentLanguage("en") is DocumentLanguage.ENGLISH
        assert DocumentLanguage("te") is DocumentLanguage.TELUGU
        assert DocumentLanguage("mixed") is DocumentLanguage.MIXED


class TestProcessingStatus:
    """Tests for ProcessingStatus enum."""

    def test_all_members_exist(self):
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.PROCESSING.value == "processing"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"
        assert ProcessingStatus.PARTIAL.value == "partial"

    def test_member_count(self):
        assert len(ProcessingStatus) == 5

    def test_is_str_enum(self):
        assert isinstance(ProcessingStatus.PENDING, str)
        assert ProcessingStatus.COMPLETED == "completed"


# =========================================================================
# DocumentMetadata Tests
# =========================================================================


class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""

    def test_create_with_required_only(self):
        meta = DocumentMetadata(title="Test Book")
        assert meta.title == "Test Book"
        assert meta.author is None
        assert meta.publication_date is None
        assert meta.isbn is None
        assert meta.description is None
        assert meta.tags == []
        assert meta.custom_fields == {}

    def test_create_with_all_fields(self):
        pub_date = datetime(2024, 6, 15, 12, 0, 0)
        meta = DocumentMetadata(
            title="Advanced Python",
            author="Jane Smith",
            publication_date=pub_date,
            isbn="978-1234567890",
            description="A deep dive into Python.",
            tags=["python", "programming"],
            custom_fields={"edition": 3, "publisher": "TechPress"},
        )
        assert meta.title == "Advanced Python"
        assert meta.author == "Jane Smith"
        assert meta.publication_date == pub_date
        assert meta.isbn == "978-1234567890"
        assert meta.description == "A deep dive into Python."
        assert meta.tags == ["python", "programming"]
        assert meta.custom_fields == {"edition": 3, "publisher": "TechPress"}

    def test_to_dict_with_all_fields(self):
        pub_date = datetime(2024, 1, 15, 10, 30, 0)
        meta = DocumentMetadata(
            title="Test",
            author="Author",
            publication_date=pub_date,
            isbn="123",
            description="Desc",
            tags=["a", "b"],
            custom_fields={"key": "value"},
        )
        d = meta.to_dict()
        assert d["title"] == "Test"
        assert d["author"] == "Author"
        assert d["publication_date"] == pub_date.isoformat()
        assert d["isbn"] == "123"
        assert d["description"] == "Desc"
        assert d["tags"] == ["a", "b"]
        assert d["custom_fields"] == {"key": "value"}

    def test_to_dict_with_no_publication_date(self):
        meta = DocumentMetadata(title="No Date")
        d = meta.to_dict()
        assert d["publication_date"] is None

    def test_from_dict_full(self):
        pub_date = datetime(2024, 3, 20, 8, 0, 0)
        data = {
            "title": "From Dict",
            "author": "Author X",
            "publication_date": pub_date.isoformat(),
            "isbn": "978-0000000000",
            "description": "A description",
            "tags": ["tag1"],
            "custom_fields": {"lang": "en"},
        }
        meta = DocumentMetadata.from_dict(data)
        assert meta.title == "From Dict"
        assert meta.author == "Author X"
        assert meta.publication_date == pub_date
        assert meta.isbn == "978-0000000000"
        assert meta.description == "A description"
        assert meta.tags == ["tag1"]
        assert meta.custom_fields == {"lang": "en"}

    def test_from_dict_minimal(self):
        meta = DocumentMetadata.from_dict({})
        assert meta.title == "Untitled"
        assert meta.author is None
        assert meta.publication_date is None
        assert meta.isbn is None
        assert meta.description is None
        assert meta.tags == []
        assert meta.custom_fields == {}

    def test_from_dict_with_none_publication_date(self):
        data = {"title": "No Date", "publication_date": None}
        meta = DocumentMetadata.from_dict(data)
        assert meta.publication_date is None

    def test_from_dict_with_datetime_object_publication_date(self):
        """When publication_date is already a datetime object (not a string)."""
        dt = datetime(2025, 5, 10)
        data = {"title": "DT Test", "publication_date": dt}
        meta = DocumentMetadata.from_dict(data)
        assert meta.publication_date is dt

    def test_to_dict_from_dict_roundtrip(self):
        pub_date = datetime(2024, 12, 25, 0, 0, 0)
        original = DocumentMetadata(
            title="Roundtrip",
            author="RT Author",
            publication_date=pub_date,
            isbn="111",
            description="Roundtrip test",
            tags=["round", "trip"],
            custom_fields={"x": 1},
        )
        restored = DocumentMetadata.from_dict(original.to_dict())
        assert restored.title == original.title
        assert restored.author == original.author
        assert restored.publication_date == original.publication_date
        assert restored.isbn == original.isbn
        assert restored.tags == original.tags
        assert restored.custom_fields == original.custom_fields

    def test_tags_default_not_shared(self):
        """Ensure default mutable list is not shared across instances."""
        m1 = DocumentMetadata(title="A")
        m2 = DocumentMetadata(title="B")
        m1.tags.append("only_m1")
        assert "only_m1" not in m2.tags

    def test_custom_fields_default_not_shared(self):
        """Ensure default mutable dict is not shared across instances."""
        m1 = DocumentMetadata(title="A")
        m2 = DocumentMetadata(title="B")
        m1.custom_fields["key"] = "val"
        assert "key" not in m2.custom_fields


# =========================================================================
# ChapterInfo Tests
# =========================================================================


class TestChapterInfo:
    """Tests for ChapterInfo dataclass."""

    def test_create_with_required_fields(self):
        ch = ChapterInfo(number=1, title="Introduction", start_page=1, end_page=20)
        assert ch.number == 1
        assert ch.title == "Introduction"
        assert ch.start_page == 1
        assert ch.end_page == 20
        assert ch.summary is None

    def test_create_with_all_fields(self):
        ch = ChapterInfo(
            number=3,
            title="Climax",
            start_page=100,
            end_page=150,
            summary="The pivotal scene.",
        )
        assert ch.summary == "The pivotal scene."

    def test_to_dict(self):
        ch = ChapterInfo(
            number=2,
            title="Rising Action",
            start_page=21,
            end_page=60,
            summary="Tension builds.",
        )
        d = ch.to_dict()
        assert d == {
            "number": 2,
            "title": "Rising Action",
            "start_page": 21,
            "end_page": 60,
            "summary": "Tension builds.",
        }

    def test_to_dict_no_summary(self):
        ch = ChapterInfo(number=1, title="Ch1", start_page=1, end_page=10)
        d = ch.to_dict()
        assert d["summary"] is None

    def test_from_dict_full(self):
        data = {
            "number": 5,
            "title": "Resolution",
            "start_page": 200,
            "end_page": 230,
            "summary": "All resolved.",
        }
        ch = ChapterInfo.from_dict(data)
        assert ch.number == 5
        assert ch.title == "Resolution"
        assert ch.start_page == 200
        assert ch.end_page == 230
        assert ch.summary == "All resolved."

    def test_from_dict_without_summary(self):
        data = {"number": 1, "title": "Intro", "start_page": 1, "end_page": 15}
        ch = ChapterInfo.from_dict(data)
        assert ch.summary is None

    def test_to_dict_from_dict_roundtrip(self):
        original = ChapterInfo(
            number=7, title="Epilogue", start_page=300, end_page=310, summary="End."
        )
        restored = ChapterInfo.from_dict(original.to_dict())
        assert restored.number == original.number
        assert restored.title == original.title
        assert restored.start_page == original.start_page
        assert restored.end_page == original.end_page
        assert restored.summary == original.summary


# =========================================================================
# Document Tests
# =========================================================================


@pytest.fixture
def sample_metadata():
    return DocumentMetadata(
        title="Story",
        author="Robert McKee",
        isbn="978-0060391683",
        description="A screenwriting classic",
        tags=["screenwriting", "film"],
    )


@pytest.fixture
def sample_chapters():
    return [
        ChapterInfo(number=1, title="Story Structure", start_page=1, end_page=50),
        ChapterInfo(number=2, title="Character Design", start_page=51, end_page=100),
    ]


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_factory_defaults(self, sample_metadata):
        doc = Document.create(
            file_path="/books/story.pdf",
            file_hash="abc123hash",
            file_size=1024000,
            metadata=sample_metadata,
            total_pages=350,
        )
        # UUID is generated
        assert len(doc.id) == 36  # UUID format with hyphens
        uuid.UUID(doc.id)  # validates it is a real UUID
        assert doc.file_path == "/books/story.pdf"
        assert doc.file_hash == "abc123hash"
        assert doc.file_size == 1024000
        assert doc.document_type is DocumentType.BOOK
        assert doc.language is DocumentLanguage.ENGLISH
        assert doc.total_pages == 350
        assert doc.project is None
        assert doc.status is ProcessingStatus.PENDING
        assert doc.processed_pages == 0
        assert doc.chapters == []
        assert doc.access_count == 0
        assert doc.processed_at is None
        assert doc.last_accessed is None
        assert isinstance(doc.created_at, datetime)

    def test_create_factory_custom_params(self, sample_metadata):
        doc = Document.create(
            file_path="/scripts/film.fdx",
            file_hash="xyz789",
            file_size=500,
            metadata=sample_metadata,
            total_pages=120,
            document_type=DocumentType.SCREENPLAY,
            language=DocumentLanguage.TELUGU,
            project="my_film",
        )
        assert doc.document_type is DocumentType.SCREENPLAY
        assert doc.language is DocumentLanguage.TELUGU
        assert doc.project == "my_film"

    def test_create_factory_generates_unique_ids(self, sample_metadata):
        ids = set()
        for _ in range(50):
            doc = Document.create(
                file_path="/f.pdf",
                file_hash="h",
                file_size=1,
                metadata=sample_metadata,
                total_pages=1,
            )
            ids.add(doc.id)
        assert len(ids) == 50

    def test_direct_construction(self, sample_metadata, sample_chapters):
        now = datetime.now()
        doc = Document(
            id="fixed-id-001",
            file_path="/test.pdf",
            file_hash="hashvalue",
            file_size=2048,
            document_type=DocumentType.MANUAL,
            metadata=sample_metadata,
            language=DocumentLanguage.MIXED,
            total_pages=50,
            chapters=sample_chapters,
            status=ProcessingStatus.COMPLETED,
            processed_pages=50,
            created_at=now,
            processed_at=now,
            last_accessed=now,
            project="proj_alpha",
            access_count=10,
        )
        assert doc.id == "fixed-id-001"
        assert doc.document_type is DocumentType.MANUAL
        assert doc.language is DocumentLanguage.MIXED
        assert len(doc.chapters) == 2
        assert doc.status is ProcessingStatus.COMPLETED
        assert doc.processed_pages == 50
        assert doc.access_count == 10
        assert doc.project == "proj_alpha"

    def test_to_dict_all_fields(self, sample_metadata, sample_chapters):
        now = datetime(2025, 1, 1, 12, 0, 0)
        processed = datetime(2025, 1, 1, 13, 0, 0)
        accessed = datetime(2025, 1, 2, 8, 0, 0)
        doc = Document(
            id="dict-test-id",
            file_path="/books/test.pdf",
            file_hash="sha256hash",
            file_size=4096,
            document_type=DocumentType.ARTICLE,
            metadata=sample_metadata,
            language=DocumentLanguage.ENGLISH,
            total_pages=10,
            chapters=sample_chapters,
            status=ProcessingStatus.COMPLETED,
            processed_pages=10,
            created_at=now,
            processed_at=processed,
            last_accessed=accessed,
            project="proj_beta",
            access_count=5,
        )
        d = doc.to_dict()
        assert d["id"] == "dict-test-id"
        assert d["file_path"] == "/books/test.pdf"
        assert d["file_hash"] == "sha256hash"
        assert d["file_size"] == 4096
        assert d["document_type"] == "article"
        assert d["language"] == "en"
        assert d["total_pages"] == 10
        assert d["status"] == "completed"
        assert d["processed_pages"] == 10
        assert d["created_at"] == now.isoformat()
        assert d["processed_at"] == processed.isoformat()
        assert d["last_accessed"] == accessed.isoformat()
        assert d["project"] == "proj_beta"
        assert d["access_count"] == 5
        # Nested metadata
        assert d["metadata"]["title"] == "Story"
        assert d["metadata"]["author"] == "Robert McKee"
        # Nested chapters
        assert len(d["chapters"]) == 2
        assert d["chapters"][0]["title"] == "Story Structure"
        assert d["chapters"][1]["number"] == 2

    def test_to_dict_none_datetimes(self, sample_metadata):
        doc = Document.create(
            file_path="/f.pdf",
            file_hash="h",
            file_size=1,
            metadata=sample_metadata,
            total_pages=1,
        )
        d = doc.to_dict()
        assert d["processed_at"] is None
        assert d["last_accessed"] is None
        assert d["project"] is None

    def test_to_dict_empty_chapters(self, sample_metadata):
        doc = Document.create(
            file_path="/f.pdf",
            file_hash="h",
            file_size=1,
            metadata=sample_metadata,
            total_pages=1,
        )
        d = doc.to_dict()
        assert d["chapters"] == []


# =========================================================================
# Page Tests
# =========================================================================


class TestPage:
    """Tests for Page dataclass."""

    def test_create_factory_defaults(self):
        page = Page.create(
            document_id="doc-001",
            page_number=1,
            raw_text="Hello world",
        )
        uuid.UUID(page.id)
        assert page.document_id == "doc-001"
        assert page.page_number == 1
        assert page.raw_text == "Hello world"
        # cleaned_text defaults to raw_text when not provided
        assert page.cleaned_text == "Hello world"
        assert page.ocr_confidence == 0.0
        assert page.ocr_model == "deepseek-ocr-2"
        assert page.has_images is False
        assert page.has_tables is False
        assert page.detected_headers == []
        assert isinstance(page.processed_at, datetime)

    def test_create_factory_with_cleaned_text(self):
        page = Page.create(
            document_id="doc-002",
            page_number=5,
            raw_text="Raw OCR output with noise ##",
            cleaned_text="Clean text without noise",
            ocr_confidence=0.95,
            ocr_model="custom-ocr-v3",
        )
        assert page.raw_text == "Raw OCR output with noise ##"
        assert page.cleaned_text == "Clean text without noise"
        assert page.ocr_confidence == 0.95
        assert page.ocr_model == "custom-ocr-v3"

    def test_create_factory_cleaned_text_none_defaults_to_raw(self):
        """Explicit None for cleaned_text should fall back to raw_text."""
        page = Page.create(
            document_id="doc-003",
            page_number=1,
            raw_text="Some text",
            cleaned_text=None,
        )
        assert page.cleaned_text == "Some text"

    def test_create_factory_cleaned_text_empty_string_uses_raw(self):
        """Empty string is falsy, so it should fall back to raw_text."""
        page = Page.create(
            document_id="doc-004",
            page_number=1,
            raw_text="Fallback text",
            cleaned_text="",
        )
        assert page.cleaned_text == "Fallback text"

    def test_create_generates_unique_ids(self):
        ids = set()
        for i in range(30):
            p = Page.create(document_id="d", page_number=i, raw_text="t")
            ids.add(p.id)
        assert len(ids) == 30

    def test_direct_construction(self):
        now = datetime(2025, 6, 1, 10, 0, 0)
        page = Page(
            id="page-fixed-id",
            document_id="doc-abc",
            page_number=42,
            raw_text="raw",
            cleaned_text="clean",
            has_images=True,
            has_tables=True,
            detected_headers=["Header One", "Header Two"],
            ocr_confidence=0.98,
            ocr_model="tesseract",
            processed_at=now,
        )
        assert page.id == "page-fixed-id"
        assert page.has_images is True
        assert page.has_tables is True
        assert page.detected_headers == ["Header One", "Header Two"]
        assert page.ocr_confidence == 0.98
        assert page.ocr_model == "tesseract"
        assert page.processed_at == now

    def test_to_dict(self):
        now = datetime(2025, 7, 4, 14, 30, 0)
        page = Page(
            id="p-id",
            document_id="d-id",
            page_number=3,
            raw_text="raw text",
            cleaned_text="cleaned text",
            has_images=True,
            has_tables=False,
            detected_headers=["H1"],
            ocr_confidence=0.88,
            ocr_model="model-x",
            processed_at=now,
        )
        d = page.to_dict()
        assert d["id"] == "p-id"
        assert d["document_id"] == "d-id"
        assert d["page_number"] == 3
        assert d["raw_text"] == "raw text"
        assert d["cleaned_text"] == "cleaned text"
        assert d["has_images"] is True
        assert d["has_tables"] is False
        assert d["detected_headers"] == ["H1"]
        assert d["ocr_confidence"] == 0.88
        assert d["ocr_model"] == "model-x"
        assert d["processed_at"] == now.isoformat()


# =========================================================================
# Chunk Tests
# =========================================================================


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_create_factory_basic(self):
        content = "This is a chunk of text from the document for retrieval."
        chunk = Chunk.create(
            document_id="doc-100",
            page_ids=["p1", "p2"],
            content=content,
            page_range="pp. 10-11",
            chunk_index=0,
        )
        uuid.UUID(chunk.id)
        assert chunk.document_id == "doc-100"
        assert chunk.page_ids == ["p1", "p2"]
        assert chunk.content == content
        assert chunk.page_range == "pp. 10-11"
        assert chunk.chunk_index == 0
        assert chunk.chapter is None
        assert chunk.section is None
        assert chunk.char_count == len(content)
        assert chunk.token_count_approx == len(content) // 4
        assert chunk.embedding is None
        assert chunk.ltm_entry_id is None
        assert chunk.entities == []
        assert isinstance(chunk.created_at, datetime)

    def test_create_factory_with_chapter_and_section(self):
        chunk = Chunk.create(
            document_id="doc-200",
            page_ids=["p5"],
            content="Chapter content here.",
            page_range="pp. 50-52",
            chunk_index=3,
            chapter="Chapter 3",
            section="Section 3.1",
        )
        assert chunk.chapter == "Chapter 3"
        assert chunk.section == "Section 3.1"

    def test_create_factory_char_count_calculation(self):
        content = "A" * 100
        chunk = Chunk.create(
            document_id="d",
            page_ids=["p"],
            content=content,
            page_range="pp. 1",
            chunk_index=0,
        )
        assert chunk.char_count == 100
        assert chunk.token_count_approx == 25  # 100 // 4

    def test_create_factory_empty_content(self):
        chunk = Chunk.create(
            document_id="d",
            page_ids=[],
            content="",
            page_range="pp. 0",
            chunk_index=0,
        )
        assert chunk.char_count == 0
        assert chunk.token_count_approx == 0

    def test_create_factory_token_count_integer_division(self):
        """Token count uses integer division by 4."""
        content = "A" * 7  # 7 // 4 = 1
        chunk = Chunk.create(
            document_id="d",
            page_ids=["p"],
            content=content,
            page_range="pp. 1",
            chunk_index=0,
        )
        assert chunk.token_count_approx == 1

    def test_create_generates_unique_ids(self):
        ids = set()
        for i in range(30):
            c = Chunk.create(
                document_id="d",
                page_ids=["p"],
                content="c",
                page_range="pp. 1",
                chunk_index=i,
            )
            ids.add(c.id)
        assert len(ids) == 30

    def test_direct_construction(self):
        now = datetime(2025, 3, 15, 9, 0, 0)
        chunk = Chunk(
            id="chunk-fixed",
            document_id="doc-x",
            page_ids=["p1", "p2", "p3"],
            content="Some content",
            page_range="pp. 1-3",
            chapter="Ch1",
            section="S1",
            embedding=[0.1, 0.2, 0.3],
            chunk_index=5,
            char_count=12,
            token_count_approx=3,
            ltm_entry_id="ltm-001",
            entities=["entity_a", "entity_b"],
            created_at=now,
        )
        assert chunk.id == "chunk-fixed"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.ltm_entry_id == "ltm-001"
        assert chunk.entities == ["entity_a", "entity_b"]
        assert chunk.created_at == now

    def test_to_dict(self):
        now = datetime(2025, 8, 10, 16, 45, 0)
        chunk = Chunk(
            id="c-id",
            document_id="d-id",
            page_ids=["p1"],
            content="text",
            page_range="pp. 5",
            chapter="Ch2",
            section="S2.1",
            embedding=[0.5, 0.6],
            chunk_index=1,
            char_count=4,
            token_count_approx=1,
            ltm_entry_id="ltm-x",
            entities=["e1"],
            created_at=now,
        )
        d = chunk.to_dict()
        assert d["id"] == "c-id"
        assert d["document_id"] == "d-id"
        assert d["page_ids"] == ["p1"]
        assert d["content"] == "text"
        assert d["page_range"] == "pp. 5"
        assert d["chapter"] == "Ch2"
        assert d["section"] == "S2.1"
        assert d["embedding"] == [0.5, 0.6]
        assert d["chunk_index"] == 1
        assert d["char_count"] == 4
        assert d["token_count_approx"] == 1
        assert d["ltm_entry_id"] == "ltm-x"
        assert d["entities"] == ["e1"]
        assert d["created_at"] == now.isoformat()

    def test_to_dict_none_embedding(self):
        chunk = Chunk.create(
            document_id="d",
            page_ids=["p"],
            content="x",
            page_range="pp. 1",
            chunk_index=0,
        )
        d = chunk.to_dict()
        assert d["embedding"] is None
        assert d["ltm_entry_id"] is None

    def test_entities_default_not_shared(self):
        c1 = Chunk.create(
            document_id="d",
            page_ids=["p"],
            content="a",
            page_range="pp.1",
            chunk_index=0,
        )
        c2 = Chunk.create(
            document_id="d",
            page_ids=["p"],
            content="b",
            page_range="pp.2",
            chunk_index=1,
        )
        c1.entities.append("only_c1")
        assert "only_c1" not in c2.entities


# =========================================================================
# Citation Tests
# =========================================================================


class TestCitation:
    """Tests for Citation dataclass and formatting methods."""

    @pytest.fixture
    def citation_with_chapter(self):
        return Citation(
            document_id="doc-001",
            document_title="Story",
            chunk_id="chunk-001",
            page_range="pp. 45-47",
            chapter="Chapter 5",
            section="Scene Design",
            quote="The scene must turn.",
            relevance=0.92,
        )

    @pytest.fixture
    def citation_without_chapter(self):
        return Citation(
            document_id="doc-002",
            document_title="Screenplay Basics",
            chunk_id="chunk-002",
            page_range="pp. 10",
            chapter=None,
            section=None,
            quote="Dialogue drives drama.",
            relevance=0.85,
        )

    def test_creation(self, citation_with_chapter):
        c = citation_with_chapter
        assert c.document_id == "doc-001"
        assert c.document_title == "Story"
        assert c.chunk_id == "chunk-001"
        assert c.page_range == "pp. 45-47"
        assert c.chapter == "Chapter 5"
        assert c.section == "Scene Design"
        assert c.quote == "The scene must turn."
        assert c.relevance == 0.92

    def test_format_inline(self, citation_with_chapter):
        result = citation_with_chapter.format_inline()
        assert result == "[Story, pp. 45-47]"

    def test_format_inline_without_chapter(self, citation_without_chapter):
        result = citation_without_chapter.format_inline()
        assert result == "[Screenplay Basics, pp. 10]"

    def test_format_footnote_with_chapter(self, citation_with_chapter):
        result = citation_with_chapter.format_footnote()
        assert result == "Story. Chapter 5. pp. 45-47"

    def test_format_footnote_without_chapter(self, citation_without_chapter):
        result = citation_without_chapter.format_footnote()
        assert result == "Screenplay Basics. pp. 10"

    def test_format_style_inline(self, citation_with_chapter):
        result = citation_with_chapter.format(style="inline")
        assert result == "[Story, pp. 45-47]"

    def test_format_style_footnote(self, citation_with_chapter):
        result = citation_with_chapter.format(style="footnote")
        assert result == "Story. Chapter 5. pp. 45-47"

    def test_format_style_unknown_fallback(self, citation_with_chapter):
        result = citation_with_chapter.format(style="chicago")
        assert result == "Source: Story, pp. 45-47"

    def test_format_default_style_is_inline(self, citation_with_chapter):
        """Calling format() with no argument should default to inline."""
        assert citation_with_chapter.format() == citation_with_chapter.format_inline()


# =========================================================================
# DocumentSearchResult Tests
# =========================================================================


class TestDocumentSearchResult:
    """Tests for DocumentSearchResult dataclass and to_dict."""

    @pytest.fixture
    def search_result(self, sample_metadata):
        chunk = Chunk(
            id="chunk-sr",
            document_id="doc-sr",
            page_ids=["p10", "p11"],
            content="Found this relevant passage in the text.",
            page_range="pp. 10-11",
            chapter="Chapter 2",
            section="Subtext",
            embedding=None,
            chunk_index=4,
            char_count=41,
            token_count_approx=10,
            ltm_entry_id=None,
            entities=[],
            created_at=datetime(2025, 1, 1),
        )
        doc = Document(
            id="doc-sr",
            file_path="/books/story.pdf",
            file_hash="hash123",
            file_size=2048,
            document_type=DocumentType.BOOK,
            metadata=sample_metadata,
            language=DocumentLanguage.ENGLISH,
            total_pages=200,
            created_at=datetime(2025, 1, 1),
        )
        citation = Citation(
            document_id="doc-sr",
            document_title="Story",
            chunk_id="chunk-sr",
            page_range="pp. 10-11",
            chapter="Chapter 2",
            section="Subtext",
            quote="Found this relevant passage",
            relevance=0.91,
        )
        return DocumentSearchResult(
            chunk=chunk,
            document=doc,
            similarity=0.91,
            highlight="Found this **relevant** passage",
            citation=citation,
        )

    def test_search_result_creation(self, search_result):
        assert search_result.similarity == 0.91
        assert search_result.highlight == "Found this **relevant** passage"
        assert search_result.chunk.id == "chunk-sr"
        assert search_result.document.id == "doc-sr"

    def test_search_result_to_dict(self, search_result):
        d = search_result.to_dict()
        assert d["chunk_id"] == "chunk-sr"
        assert d["document_id"] == "doc-sr"
        assert d["document_title"] == "Story"
        assert d["content"] == "Found this relevant passage in the text."
        assert d["page_range"] == "pp. 10-11"
        assert d["chapter"] == "Chapter 2"
        assert d["similarity"] == 0.91
        assert d["highlight"] == "Found this **relevant** passage"
        assert d["citation"] == "[Story, pp. 10-11]"

    def test_search_result_to_dict_keys(self, search_result):
        d = search_result.to_dict()
        expected_keys = {
            "chunk_id",
            "document_id",
            "document_title",
            "content",
            "page_range",
            "chapter",
            "similarity",
            "highlight",
            "citation",
        }
        assert set(d.keys()) == expected_keys


# =========================================================================
# OCRResult Tests
# =========================================================================


class TestOCRResult:
    """Tests for OCRResult dataclass."""

    def test_create_with_required_only(self):
        ocr = OCRResult(text="Recognized text", confidence=0.95)
        assert ocr.text == "Recognized text"
        assert ocr.confidence == 0.95
        assert ocr.has_images is False
        assert ocr.has_tables is False
        assert ocr.detected_headers == []
        assert ocr.model_used == "deepseek-ocr-2"
        assert ocr.processing_time_ms == 0

    def test_create_with_all_fields(self):
        ocr = OCRResult(
            text="Complex page",
            confidence=0.88,
            has_images=True,
            has_tables=True,
            detected_headers=["Title", "Subtitle"],
            model_used="tesseract-v5",
            processing_time_ms=1500,
        )
        assert ocr.has_images is True
        assert ocr.has_tables is True
        assert ocr.detected_headers == ["Title", "Subtitle"]
        assert ocr.model_used == "tesseract-v5"
        assert ocr.processing_time_ms == 1500

    def test_detected_headers_default_not_shared(self):
        o1 = OCRResult(text="a", confidence=0.9)
        o2 = OCRResult(text="b", confidence=0.8)
        o1.detected_headers.append("Header")
        assert "Header" not in o2.detected_headers


# =========================================================================
# ProcessingResult Tests
# =========================================================================


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_create_with_required_only(self):
        pr = ProcessingResult(
            document_id="doc-pr",
            status=ProcessingStatus.COMPLETED,
            pages_processed=100,
            chunks_created=50,
        )
        assert pr.document_id == "doc-pr"
        assert pr.status is ProcessingStatus.COMPLETED
        assert pr.pages_processed == 100
        assert pr.chunks_created == 50
        assert pr.errors == []
        assert pr.processing_time_seconds == 0.0

    def test_create_with_all_fields(self):
        pr = ProcessingResult(
            document_id="doc-fail",
            status=ProcessingStatus.FAILED,
            pages_processed=10,
            chunks_created=3,
            errors=["OCR failed on page 11", "Timeout on page 12"],
            processing_time_seconds=45.2,
        )
        assert pr.status is ProcessingStatus.FAILED
        assert len(pr.errors) == 2
        assert "OCR failed on page 11" in pr.errors
        assert pr.processing_time_seconds == 45.2

    def test_errors_default_not_shared(self):
        pr1 = ProcessingResult(
            document_id="d1",
            status=ProcessingStatus.COMPLETED,
            pages_processed=1,
            chunks_created=1,
        )
        pr2 = ProcessingResult(
            document_id="d2",
            status=ProcessingStatus.COMPLETED,
            pages_processed=2,
            chunks_created=2,
        )
        pr1.errors.append("err")
        assert "err" not in pr2.errors

    def test_partial_status(self):
        pr = ProcessingResult(
            document_id="doc-partial",
            status=ProcessingStatus.PARTIAL,
            pages_processed=5,
            chunks_created=2,
            errors=["Skipped page 6 due to corruption"],
        )
        assert pr.status is ProcessingStatus.PARTIAL
        assert pr.pages_processed == 5


# =========================================================================
# Edge Case / Cross-cutting Tests
# =========================================================================


class TestEdgeCases:
    """Edge case and cross-cutting tests."""

    def test_document_metadata_from_dict_iso_string_with_timezone_info(self):
        """datetime.fromisoformat should handle ISO strings with timezone offset (Python 3.11+)."""
        data = {"title": "TZ Test", "publication_date": "2024-06-15T12:00:00+05:30"}
        meta = DocumentMetadata.from_dict(data)
        assert meta.publication_date is not None
        assert meta.publication_date.year == 2024

    def test_chunk_create_large_content(self):
        """Chunk with large content should correctly compute char_count and token_count."""
        large_content = "x" * 100000
        chunk = Chunk.create(
            document_id="d",
            page_ids=["p"],
            content=large_content,
            page_range="pp. 1-500",
            chunk_index=0,
        )
        assert chunk.char_count == 100000
        assert chunk.token_count_approx == 25000

    def test_page_create_with_unicode_text(self):
        """Page should handle Unicode text (e.g., Telugu script) correctly."""
        telugu_text = "\u0c28\u0c2e\u0c38\u0c4d\u0c15\u0c3e\u0c30\u0c02"
        page = Page.create(
            document_id="doc-te",
            page_number=1,
            raw_text=telugu_text,
        )
        assert page.raw_text == telugu_text
        assert page.cleaned_text == telugu_text

    def test_chunk_create_with_unicode_content(self):
        telugu_content = "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 \u0c2d\u0c3e\u0c37"
        chunk = Chunk.create(
            document_id="d",
            page_ids=["p"],
            content=telugu_content,
            page_range="pp. 1",
            chunk_index=0,
        )
        assert chunk.char_count == len(telugu_content)
        assert chunk.token_count_approx == len(telugu_content) // 4

    def test_document_to_dict_chapters_preserves_order(self, sample_metadata):
        chapters = [
            ChapterInfo(
                number=i, title=f"Ch {i}", start_page=i * 10, end_page=(i + 1) * 10
            )
            for i in range(1, 6)
        ]
        doc = Document(
            id="order-test",
            file_path="/f.pdf",
            file_hash="h",
            file_size=1,
            document_type=DocumentType.BOOK,
            metadata=sample_metadata,
            language=DocumentLanguage.ENGLISH,
            total_pages=60,
            chapters=chapters,
            created_at=datetime(2025, 1, 1),
        )
        d = doc.to_dict()
        for i, ch_dict in enumerate(d["chapters"]):
            assert ch_dict["number"] == i + 1

    def test_citation_format_footnote_joins_with_dot_space(self):
        """Verify the separator is '. ' between parts."""
        c = Citation(
            document_id="d",
            document_title="Title",
            chunk_id="c",
            page_range="p 1",
            chapter="Ch1",
            section="S1",
            quote="q",
            relevance=0.5,
        )
        result = c.format_footnote()
        # format_footnote joins [document_title, chapter, page_range] with ". "
        assert result == "Title. Ch1. p 1"

    def test_document_search_result_uses_inline_citation(self, sample_metadata):
        """DocumentSearchResult.to_dict() should use format_inline for citation."""
        chunk = Chunk.create(
            document_id="d",
            page_ids=["p"],
            content="content",
            page_range="pp. 99",
            chunk_index=0,
        )
        doc = Document.create(
            file_path="/f.pdf",
            file_hash="h",
            file_size=1,
            metadata=sample_metadata,
            total_pages=100,
        )
        citation = Citation(
            document_id=doc.id,
            document_title="Story",
            chunk_id=chunk.id,
            page_range="pp. 99",
            chapter=None,
            section=None,
            quote="content",
            relevance=0.7,
        )
        sr = DocumentSearchResult(
            chunk=chunk,
            document=doc,
            similarity=0.7,
            highlight="content",
            citation=citation,
        )
        d = sr.to_dict()
        assert d["citation"] == "[Story, pp. 99]"

    def test_chapter_info_from_dict_missing_required_raises(self):
        """from_dict should raise KeyError when required fields are missing."""
        with pytest.raises(KeyError):
            ChapterInfo.from_dict({"number": 1})

    def test_enum_values_are_lowercase(self):
        """All enum values should be lowercase strings."""
        for member in DocumentType:
            assert member.value == member.value.lower()
        for member in ProcessingStatus:
            assert member.value == member.value.lower()

    def test_document_default_status_is_pending(self, sample_metadata):
        doc = Document.create(
            file_path="/f.pdf",
            file_hash="h",
            file_size=1,
            metadata=sample_metadata,
            total_pages=1,
        )
        assert doc.status is ProcessingStatus.PENDING

    def test_page_to_dict_processed_at_is_isoformat_string(self):
        page = Page.create(document_id="d", page_number=1, raw_text="t")
        d = page.to_dict()
        # Verify it is a parseable ISO format string
        parsed = datetime.fromisoformat(d["processed_at"])
        assert isinstance(parsed, datetime)

    def test_chunk_to_dict_created_at_is_isoformat_string(self):
        chunk = Chunk.create(
            document_id="d",
            page_ids=["p"],
            content="c",
            page_range="pp.1",
            chunk_index=0,
        )
        d = chunk.to_dict()
        parsed = datetime.fromisoformat(d["created_at"])
        assert isinstance(parsed, datetime)
