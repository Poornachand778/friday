"""
Tests for DocumentStore
========================

Tests SQLite-based document storage including CRUD operations
for documents, pages, and chunks, plus search and stats.

Run with: pytest tests/test_document_store.py -v
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from documents.config import StorageConfig
from documents.models import (
    Chunk,
    ChapterInfo,
    Document,
    DocumentLanguage,
    DocumentMetadata,
    DocumentType,
    Page,
    ProcessingStatus,
)
from documents.storage.document_store import DocumentStore


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def store(tmp_path):
    """Create a DocumentStore with temp database"""
    config = StorageConfig(
        db_path=str(tmp_path / "test_docs.db"),
        documents_dir=str(tmp_path / "raw"),
        images_dir=str(tmp_path / "images"),
    )
    ds = DocumentStore(config=config)
    ds.initialize()
    yield ds
    ds.close()


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
def sample_document(sample_metadata):
    return Document(
        id="doc-001",
        file_path="/books/story.pdf",
        file_hash="abc123hash",
        file_size=1024000,
        document_type=DocumentType.BOOK,
        metadata=sample_metadata,
        language=DocumentLanguage.ENGLISH,
        total_pages=400,
        chapters=[
            ChapterInfo(number=1, title="The Story Problem", start_page=1, end_page=30),
            ChapterInfo(
                number=2, title="Structure Spectrum", start_page=31, end_page=60
            ),
        ],
        status=ProcessingStatus.PENDING,
        processed_pages=0,
        created_at=datetime(2025, 1, 15, 10, 0, 0),
        project="aa-janta-naduma",
        access_count=0,
    )


@pytest.fixture
def sample_page():
    return Page(
        id="page-001",
        document_id="doc-001",
        page_number=1,
        raw_text="# Chapter 1: The Story Problem\n\nStory is about...",
        cleaned_text="Chapter 1: The Story Problem\nStory is about...",
        has_images=False,
        has_tables=False,
        detected_headers=["Chapter 1: The Story Problem"],
        ocr_confidence=0.95,
        ocr_model="deepseek-ocr-2",
        processed_at=datetime(2025, 1, 15, 10, 1, 0),
    )


@pytest.fixture
def sample_chunk():
    return Chunk(
        id="chunk-001",
        document_id="doc-001",
        page_ids=["page-001"],
        content="Story is about principles, not rules.",
        page_range="p. 1",
        chapter="The Story Problem",
        section="Introduction",
        embedding=None,
        chunk_index=0,
        char_count=37,
        token_count_approx=9,
        entities=["story", "principles"],
        created_at=datetime(2025, 1, 15, 10, 2, 0),
    )


@pytest.fixture
def populated_store(store, sample_document, sample_page, sample_chunk):
    """Store with one document, page, and chunk already inserted"""
    store.store_document(sample_document)
    store.store_page(sample_page)
    store.store_chunk(sample_chunk)
    return store


# =========================================================================
# Initialization
# =========================================================================


class TestStoreInit:
    """Test DocumentStore initialization"""

    def test_initialize(self, tmp_path):
        config = StorageConfig(db_path=str(tmp_path / "test.db"))
        ds = DocumentStore(config=config)
        ds.initialize()
        assert ds._initialized is True
        ds.close()

    def test_initialize_creates_db(self, tmp_path):
        db_path = tmp_path / "new.db"
        config = StorageConfig(db_path=str(db_path))
        ds = DocumentStore(config=config)
        ds.initialize()
        assert db_path.exists()
        ds.close()

    def test_double_initialize_safe(self, store):
        store.initialize()  # Already initialized in fixture
        assert store._initialized is True

    def test_close(self, tmp_path):
        config = StorageConfig(db_path=str(tmp_path / "test.db"))
        ds = DocumentStore(config=config)
        ds.initialize()
        ds.close()
        assert ds._conn is None
        assert ds._initialized is False

    def test_auto_initialize_on_transaction(self, tmp_path):
        config = StorageConfig(db_path=str(tmp_path / "auto.db"))
        ds = DocumentStore(config=config)
        # Don't call initialize(), let transaction auto-init
        stats = ds.get_stats()
        assert stats["total_documents"] == 0
        ds.close()


# =========================================================================
# Document CRUD
# =========================================================================


class TestDocumentCRUD:
    """Test document create/read/update/delete"""

    def test_store_document(self, store, sample_document):
        store.store_document(sample_document)
        doc = store.get_document("doc-001")
        assert doc is not None
        assert doc.id == "doc-001"

    def test_get_document_fields(self, store, sample_document):
        store.store_document(sample_document)
        doc = store.get_document("doc-001")
        assert doc.file_path == "/books/story.pdf"
        assert doc.file_hash == "abc123hash"
        assert doc.file_size == 1024000
        assert doc.document_type == DocumentType.BOOK
        assert doc.language == DocumentLanguage.ENGLISH
        assert doc.total_pages == 400
        assert doc.project == "aa-janta-naduma"

    def test_get_document_metadata(self, store, sample_document):
        store.store_document(sample_document)
        doc = store.get_document("doc-001")
        assert doc.metadata.title == "Story"
        assert doc.metadata.author == "Robert McKee"
        assert doc.metadata.isbn == "978-0060391683"
        assert "screenwriting" in doc.metadata.tags

    def test_get_document_chapters(self, store, sample_document):
        store.store_document(sample_document)
        doc = store.get_document("doc-001")
        assert len(doc.chapters) == 2
        assert doc.chapters[0].title == "The Story Problem"
        assert doc.chapters[1].start_page == 31

    def test_get_document_not_found(self, store):
        assert store.get_document("nonexistent") is None

    def test_get_document_by_hash(self, store, sample_document):
        store.store_document(sample_document)
        doc = store.get_document_by_hash("abc123hash")
        assert doc is not None
        assert doc.id == "doc-001"

    def test_get_document_by_hash_not_found(self, store):
        assert store.get_document_by_hash("nonexistent") is None

    def test_delete_document(self, populated_store):
        result = populated_store.delete_document("doc-001")
        assert result is True
        assert populated_store.get_document("doc-001") is None

    def test_delete_document_not_found(self, store):
        result = store.delete_document("nonexistent")
        assert result is False

    def test_delete_cascades_pages(self, populated_store):
        populated_store.delete_document("doc-001")
        page = populated_store.get_page("doc-001", 1)
        assert page is None

    def test_delete_cascades_chunks(self, populated_store):
        populated_store.delete_document("doc-001")
        chunks = populated_store.get_chunks_for_document("doc-001")
        assert len(chunks) == 0

    def test_update_document_status(self, populated_store):
        populated_store.update_document_status("doc-001", ProcessingStatus.PROCESSING)
        doc = populated_store.get_document("doc-001")
        assert doc.status == ProcessingStatus.PROCESSING

    def test_update_document_status_with_pages(self, populated_store):
        populated_store.update_document_status(
            "doc-001", ProcessingStatus.COMPLETED, processed_pages=400
        )
        doc = populated_store.get_document("doc-001")
        assert doc.status == ProcessingStatus.COMPLETED
        assert doc.processed_pages == 400
        assert doc.processed_at is not None

    def test_store_document_replace(self, store, sample_document):
        store.store_document(sample_document)
        sample_document.status = ProcessingStatus.COMPLETED
        store.store_document(sample_document)  # INSERT OR REPLACE
        doc = store.get_document("doc-001")
        assert doc.status == ProcessingStatus.COMPLETED


# =========================================================================
# Document Listing
# =========================================================================


class TestDocumentListing:
    """Test list_documents with filters"""

    def _make_doc(
        self,
        doc_id,
        doc_type=DocumentType.BOOK,
        project=None,
        status=ProcessingStatus.PENDING,
    ):
        return Document(
            id=doc_id,
            file_path=f"/books/{doc_id}.pdf",
            file_hash=f"hash-{doc_id}",
            file_size=1000,
            document_type=doc_type,
            metadata=DocumentMetadata(title=f"Doc {doc_id}"),
            language=DocumentLanguage.ENGLISH,
            total_pages=10,
            status=status,
            project=project,
            created_at=datetime.now(),
        )

    def test_list_all(self, store):
        store.store_document(self._make_doc("d1"))
        store.store_document(self._make_doc("d2"))
        docs = store.list_documents()
        assert len(docs) == 2

    def test_filter_by_type(self, store):
        store.store_document(self._make_doc("d1", doc_type=DocumentType.BOOK))
        store.store_document(self._make_doc("d2", doc_type=DocumentType.SCREENPLAY))
        docs = store.list_documents(document_type=DocumentType.BOOK)
        assert len(docs) == 1
        assert docs[0].id == "d1"

    def test_filter_by_project(self, store):
        store.store_document(self._make_doc("d1", project="proj-a"))
        store.store_document(self._make_doc("d2", project="proj-b"))
        docs = store.list_documents(project="proj-a")
        assert len(docs) == 1
        assert docs[0].id == "d1"

    def test_filter_by_status(self, store):
        store.store_document(self._make_doc("d1", status=ProcessingStatus.PENDING))
        store.store_document(self._make_doc("d2", status=ProcessingStatus.COMPLETED))
        docs = store.list_documents(status=ProcessingStatus.COMPLETED)
        assert len(docs) == 1
        assert docs[0].id == "d2"

    def test_limit(self, store):
        for i in range(10):
            store.store_document(self._make_doc(f"d{i}"))
        docs = store.list_documents(limit=3)
        assert len(docs) == 3

    def test_empty_list(self, store):
        docs = store.list_documents()
        assert docs == []


# =========================================================================
# Page CRUD
# =========================================================================


class TestPageCRUD:
    """Test page create/read operations"""

    def test_store_page(self, populated_store):
        page = populated_store.get_page("doc-001", 1)
        assert page is not None
        assert page.page_number == 1

    def test_get_page_fields(self, populated_store):
        page = populated_store.get_page("doc-001", 1)
        assert "Story Problem" in page.raw_text
        assert page.ocr_confidence == 0.95
        assert page.ocr_model == "deepseek-ocr-2"
        assert page.has_images is False

    def test_get_page_not_found(self, populated_store):
        assert populated_store.get_page("doc-001", 999) is None

    def test_store_pages_batch(self, store, sample_document):
        store.store_document(sample_document)
        pages = [
            Page.create(
                document_id="doc-001", page_number=i, raw_text=f"Page {i} content"
            )
            for i in range(1, 6)
        ]
        store.store_pages(pages)
        all_pages = store.get_pages_for_document("doc-001")
        assert len(all_pages) == 5

    def test_get_pages_ordered(self, store, sample_document):
        store.store_document(sample_document)
        for i in [3, 1, 2]:
            store.store_page(Page.create("doc-001", i, f"Page {i}"))
        pages = store.get_pages_for_document("doc-001")
        page_numbers = [p.page_number for p in pages]
        assert page_numbers == [1, 2, 3]

    def test_get_pages_range(self, store, sample_document):
        store.store_document(sample_document)
        for i in range(1, 11):
            store.store_page(Page.create("doc-001", i, f"Page {i}"))
        pages = store.get_pages("doc-001", start_page=3, end_page=7)
        assert len(pages) == 5
        assert pages[0].page_number == 3
        assert pages[-1].page_number == 7

    def test_get_pages_start_only(self, store, sample_document):
        store.store_document(sample_document)
        for i in range(1, 6):
            store.store_page(Page.create("doc-001", i, f"Page {i}"))
        pages = store.get_pages("doc-001", start_page=3)
        assert len(pages) == 3
        assert pages[0].page_number == 3

    def test_page_detected_headers(self, populated_store):
        page = populated_store.get_page("doc-001", 1)
        assert "Chapter 1: The Story Problem" in page.detected_headers


# =========================================================================
# Chunk CRUD
# =========================================================================


class TestChunkCRUD:
    """Test chunk create/read operations"""

    def test_store_chunk(self, populated_store):
        chunk = populated_store.get_chunk("chunk-001")
        assert chunk is not None
        assert chunk.id == "chunk-001"

    def test_get_chunk_fields(self, populated_store):
        chunk = populated_store.get_chunk("chunk-001")
        assert chunk.document_id == "doc-001"
        assert "principles" in chunk.content
        assert chunk.page_range == "p. 1"
        assert chunk.chapter == "The Story Problem"
        assert chunk.section == "Introduction"
        assert chunk.chunk_index == 0
        assert chunk.char_count == 37

    def test_get_chunk_not_found(self, store):
        assert store.get_chunk("nonexistent") is None

    def test_get_chunks_for_document(self, populated_store):
        chunks = populated_store.get_chunks_for_document("doc-001")
        assert len(chunks) == 1
        assert chunks[0].id == "chunk-001"

    def test_get_chunks_empty_document(self, store):
        chunks = store.get_chunks_for_document("nonexistent")
        assert chunks == []

    def test_get_chunks_filter_by_chapter(self, store, sample_document):
        store.store_document(sample_document)
        c1 = Chunk.create("doc-001", ["p1"], "Content ch1", "p. 1", 0, chapter="Ch 1")
        c2 = Chunk.create("doc-001", ["p2"], "Content ch2", "p. 2", 1, chapter="Ch 2")
        store.store_chunk(c1)
        store.store_chunk(c2)
        chunks = store.get_chunks_for_document("doc-001", chapter="Ch 1")
        assert len(chunks) == 1
        assert chunks[0].chapter == "Ch 1"

    def test_store_chunks_batch(self, store, sample_document):
        store.store_document(sample_document)
        chunks = [
            Chunk.create("doc-001", [f"p{i}"], f"Content {i}", f"p. {i}", i)
            for i in range(5)
        ]
        store.store_chunks(chunks)
        all_chunks = store.get_chunks_for_document("doc-001")
        assert len(all_chunks) == 5

    def test_chunks_ordered_by_index(self, store, sample_document):
        store.store_document(sample_document)
        for i in [3, 1, 0, 2]:
            store.store_chunk(
                Chunk.create("doc-001", [f"p{i}"], f"Content {i}", f"p. {i}", i)
            )
        chunks = store.get_chunks_for_document("doc-001")
        indices = [c.chunk_index for c in chunks]
        assert indices == [0, 1, 2, 3]

    def test_update_chunk_ltm_link(self, populated_store):
        populated_store.update_chunk_ltm_link("chunk-001", "ltm-entry-42")
        chunk = populated_store.get_chunk("chunk-001")
        assert chunk.ltm_entry_id == "ltm-entry-42"

    def test_chunk_entities_stored(self, populated_store):
        chunk = populated_store.get_chunk("chunk-001")
        assert "story" in chunk.entities
        assert "principles" in chunk.entities


# =========================================================================
# Chunk with Embeddings
# =========================================================================


class TestChunkEmbeddings:
    """Test chunk embedding storage and retrieval"""

    def test_store_chunk_with_embedding(self, store, sample_document):
        store.store_document(sample_document)
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        chunk = Chunk.create("doc-001", ["p1"], "Content", "p. 1", 0)
        chunk.embedding = embedding
        store.store_chunk(chunk)
        retrieved = store.get_chunk(chunk.id)
        assert retrieved.embedding is not None
        assert len(retrieved.embedding) == 5
        assert abs(retrieved.embedding[0] - 0.1) < 1e-5

    def test_store_chunk_without_embedding(self, populated_store):
        chunk = populated_store.get_chunk("chunk-001")
        assert chunk.embedding is None

    def test_embedding_round_trip_precision(self, store, sample_document):
        store.store_document(sample_document)
        embedding = np.random.randn(384).tolist()
        chunk = Chunk.create("doc-001", ["p1"], "Content", "p. 1", 0)
        chunk.embedding = embedding
        store.store_chunk(chunk)
        retrieved = store.get_chunk(chunk.id)
        # float32 precision
        np.testing.assert_array_almost_equal(retrieved.embedding, embedding, decimal=5)


# =========================================================================
# Vector Search
# =========================================================================


class TestVectorSearch:
    """Test vector similarity search"""

    def _store_chunks_with_embeddings(self, store, sample_document, embeddings_map):
        """Helper to store chunks with known embeddings"""
        store.store_document(sample_document)
        for idx, (content, emb) in enumerate(embeddings_map.items()):
            chunk = Chunk.create("doc-001", [f"p{idx}"], content, f"p. {idx}", idx)
            chunk.embedding = emb
            store.store_chunk(chunk)

    def test_vector_search_basic(self, store, sample_document):
        self._store_chunks_with_embeddings(
            store,
            sample_document,
            {
                "screenplay writing": [1.0, 0.0, 0.0],
                "cooking recipes": [0.0, 1.0, 0.0],
                "math formulas": [0.0, 0.0, 1.0],
            },
        )
        results = store.vector_search([1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3
        # Most similar first
        assert results[0][0].content == "screenplay writing"
        assert results[0][1] > 0.9

    def test_vector_search_top_k(self, store, sample_document):
        self._store_chunks_with_embeddings(
            store,
            sample_document,
            {
                "content A": [1.0, 0.0, 0.0],
                "content B": [0.9, 0.1, 0.0],
                "content C": [0.0, 1.0, 0.0],
            },
        )
        results = store.vector_search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1

    def test_vector_search_min_similarity(self, store, sample_document):
        self._store_chunks_with_embeddings(
            store,
            sample_document,
            {
                "close match": [1.0, 0.0, 0.0],
                "far match": [0.0, 1.0, 0.0],
            },
        )
        results = store.vector_search([1.0, 0.0, 0.0], min_similarity=0.9)
        assert len(results) == 1
        assert results[0][0].content == "close match"

    def test_vector_search_by_document(self, store):
        # Two documents
        meta = DocumentMetadata(title="Doc")
        doc1 = Document(
            id="d1",
            file_path="/d1.pdf",
            file_hash="h1",
            file_size=100,
            document_type=DocumentType.BOOK,
            metadata=meta,
            language=DocumentLanguage.ENGLISH,
            total_pages=1,
            created_at=datetime.now(),
        )
        doc2 = Document(
            id="d2",
            file_path="/d2.pdf",
            file_hash="h2",
            file_size=100,
            document_type=DocumentType.BOOK,
            metadata=meta,
            language=DocumentLanguage.ENGLISH,
            total_pages=1,
            created_at=datetime.now(),
        )
        store.store_document(doc1)
        store.store_document(doc2)

        c1 = Chunk.create("d1", ["p1"], "doc1 content", "p. 1", 0)
        c1.embedding = [1.0, 0.0, 0.0]
        c2 = Chunk.create("d2", ["p1"], "doc2 content", "p. 1", 0)
        c2.embedding = [0.9, 0.1, 0.0]
        store.store_chunk(c1)
        store.store_chunk(c2)

        results = store.vector_search([1.0, 0.0, 0.0], document_id="d1")
        assert len(results) == 1
        assert results[0][0].document_id == "d1"

    def test_vector_search_zero_vector(self, store, sample_document):
        self._store_chunks_with_embeddings(
            store,
            sample_document,
            {
                "content": [1.0, 0.0, 0.0],
            },
        )
        results = store.vector_search([0.0, 0.0, 0.0])
        assert results == []

    def test_vector_search_no_embeddings(self, populated_store):
        # chunk-001 has no embedding
        results = populated_store.vector_search([1.0, 0.0, 0.0])
        assert results == []


# =========================================================================
# Keyword Search (FTS5)
# =========================================================================


class TestKeywordSearch:
    """Test FTS5 keyword search"""

    def test_keyword_search_basic(self, store, sample_document):
        store.store_document(sample_document)
        store.store_chunk(
            Chunk.create("doc-001", ["p1"], "Screenplay writing principles", "p. 1", 0)
        )
        results = store.keyword_search("screenplay")
        assert len(results) >= 1
        assert "Screenplay" in results[0][0].content

    def test_keyword_search_no_match(self, populated_store):
        results = populated_store.keyword_search("xyznonexistent")
        assert results == []

    def test_keyword_search_by_document(self, store):
        meta = DocumentMetadata(title="Doc")
        d1 = Document(
            id="d1",
            file_path="/d1.pdf",
            file_hash="h1",
            file_size=100,
            document_type=DocumentType.BOOK,
            metadata=meta,
            language=DocumentLanguage.ENGLISH,
            total_pages=1,
            created_at=datetime.now(),
        )
        d2 = Document(
            id="d2",
            file_path="/d2.pdf",
            file_hash="h2",
            file_size=100,
            document_type=DocumentType.BOOK,
            metadata=meta,
            language=DocumentLanguage.ENGLISH,
            total_pages=1,
            created_at=datetime.now(),
        )
        store.store_document(d1)
        store.store_document(d2)
        store.store_chunk(
            Chunk.create("d1", ["p1"], "screenplay writing guide", "p. 1", 0)
        )
        store.store_chunk(
            Chunk.create("d2", ["p1"], "screenplay directing tips", "p. 1", 0)
        )
        results = store.keyword_search("screenplay", document_id="d1")
        assert len(results) == 1
        assert results[0][0].document_id == "d1"

    def test_keyword_search_top_k(self, store, sample_document):
        store.store_document(sample_document)
        for i in range(5):
            store.store_chunk(
                Chunk.create(
                    "doc-001", [f"p{i}"], f"Screenplay content {i}", f"p. {i}", i
                )
            )
        results = store.keyword_search("screenplay", top_k=2)
        assert len(results) <= 2

    def test_keyword_search_score_range(self, store, sample_document):
        store.store_document(sample_document)
        store.store_chunk(
            Chunk.create("doc-001", ["p1"], "character arc development", "p. 1", 0)
        )
        results = store.keyword_search("character")
        if results:
            assert 0.0 <= results[0][1] <= 1.0


# =========================================================================
# Statistics
# =========================================================================


class TestStats:
    """Test get_stats"""

    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats["total_documents"] == 0
        assert stats["total_pages"] == 0
        assert stats["total_chunks"] == 0
        assert stats["chunks_with_embeddings"] == 0

    def test_populated_stats(self, populated_store):
        stats = populated_store.get_stats()
        assert stats["total_documents"] == 1
        assert stats["total_pages"] == 1
        assert stats["total_chunks"] == 1
        assert stats["chunks_with_embeddings"] == 0

    def test_stats_type_distribution(self, store):
        meta = DocumentMetadata(title="Doc")
        for i, doc_type in enumerate(
            [DocumentType.BOOK, DocumentType.BOOK, DocumentType.SCREENPLAY]
        ):
            store.store_document(
                Document(
                    id=f"d{i}",
                    file_path=f"/d{i}.pdf",
                    file_hash=f"h{i}",
                    file_size=100,
                    document_type=doc_type,
                    metadata=meta,
                    language=DocumentLanguage.ENGLISH,
                    total_pages=1,
                    created_at=datetime.now(),
                )
            )
        stats = store.get_stats()
        assert stats["document_types"]["book"] == 2
        assert stats["document_types"]["screenplay"] == 1

    def test_stats_status_distribution(self, store):
        meta = DocumentMetadata(title="Doc")
        store.store_document(
            Document(
                id="d1",
                file_path="/d1.pdf",
                file_hash="h1",
                file_size=100,
                document_type=DocumentType.BOOK,
                metadata=meta,
                language=DocumentLanguage.ENGLISH,
                total_pages=1,
                status=ProcessingStatus.COMPLETED,
                created_at=datetime.now(),
            )
        )
        store.store_document(
            Document(
                id="d2",
                file_path="/d2.pdf",
                file_hash="h2",
                file_size=100,
                document_type=DocumentType.BOOK,
                metadata=meta,
                language=DocumentLanguage.ENGLISH,
                total_pages=1,
                status=ProcessingStatus.PENDING,
                created_at=datetime.now(),
            )
        )
        stats = store.get_stats()
        assert stats["processing_status"]["completed"] == 1
        assert stats["processing_status"]["pending"] == 1

    def test_stats_db_path(self, store):
        stats = store.get_stats()
        assert "test_docs.db" in stats["db_path"]

    def test_stats_embedding_count(self, store, sample_document):
        store.store_document(sample_document)
        c1 = Chunk.create("doc-001", ["p1"], "No embedding", "p. 1", 0)
        c2 = Chunk.create("doc-001", ["p2"], "Has embedding", "p. 2", 1)
        c2.embedding = [0.1, 0.2, 0.3]
        store.store_chunk(c1)
        store.store_chunk(c2)
        stats = store.get_stats()
        assert stats["total_chunks"] == 2
        assert stats["chunks_with_embeddings"] == 1


# =========================================================================
# Edge Cases
# =========================================================================


class TestEdgeCases:
    """Test edge cases"""

    def test_unicode_content(self, store, sample_document):
        store.store_document(sample_document)
        chunk = Chunk.create(
            "doc-001",
            ["p1"],
            "Boss, ఈ screenplay చాలా బాగుంది",
            "p. 1",
            0,
        )
        store.store_chunk(chunk)
        retrieved = store.get_chunk(chunk.id)
        assert "బాగుంది" in retrieved.content

    def test_empty_content(self, store, sample_document):
        store.store_document(sample_document)
        chunk = Chunk.create("doc-001", ["p1"], "", "p. 1", 0)
        store.store_chunk(chunk)
        retrieved = store.get_chunk(chunk.id)
        assert retrieved.content == ""

    def test_very_large_content(self, store, sample_document):
        store.store_document(sample_document)
        big_content = "word " * 10000
        chunk = Chunk.create("doc-001", ["p1"], big_content, "p. 1", 0)
        store.store_chunk(chunk)
        retrieved = store.get_chunk(chunk.id)
        assert len(retrieved.content) == len(big_content)

    def test_document_no_chapters(self, store):
        doc = Document(
            id="d-no-ch",
            file_path="/empty.pdf",
            file_hash="empty-hash",
            file_size=100,
            document_type=DocumentType.ARTICLE,
            metadata=DocumentMetadata(title="Article"),
            language=DocumentLanguage.ENGLISH,
            total_pages=5,
            chapters=[],
            created_at=datetime.now(),
        )
        store.store_document(doc)
        retrieved = store.get_document("d-no-ch")
        assert retrieved.chapters == []

    def test_document_optional_fields_none(self, store):
        doc = Document(
            id="d-min",
            file_path="/min.pdf",
            file_hash="min-hash",
            file_size=50,
            document_type=DocumentType.CUSTOM,
            metadata=DocumentMetadata(title="Minimal"),
            language=DocumentLanguage.ENGLISH,
            total_pages=1,
            created_at=datetime.now(),
        )
        store.store_document(doc)
        retrieved = store.get_document("d-min")
        assert retrieved.project is None
        assert retrieved.processed_at is None
        assert retrieved.last_accessed is None

    def test_multiple_page_ids_in_chunk(self, store, sample_document):
        store.store_document(sample_document)
        chunk = Chunk.create(
            "doc-001",
            ["p1", "p2", "p3"],
            "Multi-page content",
            "pp. 1-3",
            0,
        )
        store.store_chunk(chunk)
        retrieved = store.get_chunk(chunk.id)
        assert retrieved.page_ids == ["p1", "p2", "p3"]

    def test_special_chars_in_search(self, store, sample_document):
        store.store_document(sample_document)
        store.store_chunk(
            Chunk.create("doc-001", ["p1"], 'He said "hello" to everyone', "p. 1", 0)
        )
        # Should not crash with quotes in query
        results = store.keyword_search('"hello"')
        # May or may not find results depending on FTS handling, but should not error
        assert isinstance(results, list)
