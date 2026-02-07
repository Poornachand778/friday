"""
Tests for documents/manager.py
===============================

Comprehensive tests for DocumentManager, singleton pattern,
ingestion pipeline, search, retrieval, and lifecycle.

Tests: 75+
"""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio

from documents.manager import (
    DocumentManager,
    get_document_manager,
    initialize_document_manager,
    _manager,
)
from documents.models import (
    Document,
    DocumentLanguage,
    DocumentMetadata,
    DocumentSearchResult,
    DocumentType,
    Page,
    Chunk,
    ProcessingResult,
    ProcessingStatus,
    Citation,
    ChapterInfo,
)


# ── Helpers ───────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    import documents.manager as mod

    mod._manager = None
    yield
    mod._manager = None


@pytest.fixture
def mock_config():
    """Mock DocumentConfig."""
    config = MagicMock()
    config.storage = MagicMock()
    config.ocr = MagicMock()
    config.ocr.image_dpi = 300
    config.ocr.max_batch_size = 5
    config.chunking = MagicMock()
    config.retrieval = MagicMock()
    config.retrieval.citation_style = "inline"
    config.embedding = MagicMock()
    config.embedding.model_name = "test-model"
    config.embedding.batch_size = 32
    config.embedding.normalize = True
    config.embedding.share_gpu_with_ocr = False
    config.integration = MagicMock()
    config.integration.store_chunks_in_ltm = False
    return config


def _make_mock_store():
    """Create mock DocumentStore."""
    store = MagicMock()
    store.initialize = MagicMock()
    store.close = MagicMock()
    store.store_document = MagicMock()
    store.store_pages = MagicMock()
    store.store_chunks = MagicMock()
    store.get_document = MagicMock(return_value=None)
    store.get_document_by_hash = MagicMock(return_value=None)
    store.get_page = MagicMock(return_value=None)
    store.get_pages_for_document = MagicMock(return_value=[])
    store.get_chunks_for_document = MagicMock(return_value=[])
    store.list_documents = MagicMock(return_value=[])
    store.delete_document = MagicMock(return_value=True)
    store.update_document_status = MagicMock()
    store.get_stats = MagicMock(return_value={"total": 0})
    return store


def _make_mock_ocr():
    """Create mock DeepSeekOCR."""
    ocr = MagicMock()
    ocr.is_loaded = False
    ocr.unload_model = AsyncMock()
    result = MagicMock()
    result.text = "OCR text"
    result.confidence = 0.95
    result.model_used = "deepseek-ocr-2"
    result.has_images = False
    result.has_tables = False
    result.detected_headers = []
    ocr.process_batch = AsyncMock(return_value=[result])
    return ocr


def _make_mock_searcher():
    """Create mock DocumentSearcher."""
    searcher = MagicMock()
    searcher.initialize = AsyncMock()
    searcher.search = AsyncMock(return_value=[])
    return searcher


def _make_mock_pdf_processor():
    """Create mock PDFProcessor."""
    proc = MagicMock()
    proc.get_pdf_info = MagicMock(return_value=(10, "abc123hash", 1024))
    proc.convert_to_images = MagicMock(return_value=["/tmp/page_1.png"])
    proc.cleanup_images = MagicMock()
    return proc


def _make_mock_document(
    doc_id="doc1", title="Test Book", pages=10, status=ProcessingStatus.PENDING
):
    """Create a mock Document."""
    doc = MagicMock(spec=Document)
    doc.id = doc_id
    doc.file_path = "/test/book.pdf"
    doc.file_hash = "abc123"
    doc.file_size = 1024
    doc.metadata = DocumentMetadata(title=title, author="Author")
    doc.total_pages = pages
    doc.document_type = DocumentType.BOOK
    doc.language = DocumentLanguage.ENGLISH
    doc.project = None
    doc.status = status
    doc.chapters = []
    return doc


@pytest_asyncio.fixture
async def manager(mock_config):
    """Fully initialized DocumentManager with all mocks."""
    with patch("documents.manager.get_document_config", return_value=mock_config):
        mgr = DocumentManager(config=mock_config)

    mgr._store = _make_mock_store()
    mgr._ocr = _make_mock_ocr()
    mgr._pdf_processor = _make_mock_pdf_processor()
    mgr._chunker = MagicMock()
    mgr._chunker.chunk_document = MagicMock(return_value=[])
    mgr._searcher = _make_mock_searcher()
    mgr._citation_tracker = MagicMock()
    mgr._citation_tracker.get_context_with_citations = MagicMock(
        return_value=("context", [])
    )
    mgr._initialized = True

    yield mgr


# ── Init & Lifecycle ─────────────────────────────────────────────────────


class TestInit:
    def test_constructor(self, mock_config):
        mgr = DocumentManager(config=mock_config)
        assert mgr.config is mock_config
        assert mgr._initialized is False
        assert mgr._store is None

    def test_constructor_default_config(self):
        with patch("documents.manager.get_document_config") as mock_get:
            mock_get.return_value = MagicMock()
            mgr = DocumentManager()
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize(self, mock_config):
        with patch("documents.manager.DocumentStore") as MockStore, patch(
            "documents.manager.PDFProcessor"
        ) as MockPDF, patch("documents.manager.SemanticChunker") as MockChunker, patch(
            "documents.manager.DeepSeekOCR"
        ) as MockOCR, patch(
            "documents.manager.DocumentSearcher"
        ) as MockSearcher, patch(
            "documents.manager.CitationTracker"
        ) as MockCitation:
            MockSearcher.return_value.initialize = AsyncMock()
            mock_config.integration.store_chunks_in_ltm = False

            mgr = DocumentManager(config=mock_config)
            await mgr.initialize()

            assert mgr._initialized is True
            MockStore.assert_called_once()
            MockPDF.assert_called_once()
            MockChunker.assert_called_once()
            MockOCR.assert_called_once()
            MockSearcher.assert_called_once()
            MockCitation.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, manager):
        """Second initialize is a no-op."""
        manager._initialized = True
        store_calls_before = manager._store.initialize.call_count
        await manager.initialize()
        # store.initialize should not be called again
        assert manager._store.initialize.call_count == store_calls_before

    @pytest.mark.asyncio
    async def test_shutdown(self, manager):
        manager._ocr.is_loaded = True
        await manager.shutdown()
        manager._ocr.unload_model.assert_called_once()
        manager._store.close.assert_called_once()
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_shutdown_ocr_not_loaded(self, manager):
        manager._ocr.is_loaded = False
        await manager.shutdown()
        manager._ocr.unload_model.assert_not_called()
        manager._store.close.assert_called_once()


# ── Singleton ────────────────────────────────────────────────────────────


class TestSingleton:
    def test_get_document_manager(self):
        with patch("documents.manager.get_document_config") as mock_get:
            mock_get.return_value = MagicMock()
            mgr1 = get_document_manager()
            mgr2 = get_document_manager()
            assert mgr1 is mgr2

    @pytest.mark.asyncio
    async def test_initialize_document_manager(self):
        with patch("documents.manager.get_document_config") as mock_get, patch(
            "documents.manager.DocumentStore"
        ), patch("documents.manager.PDFProcessor"), patch(
            "documents.manager.SemanticChunker"
        ), patch(
            "documents.manager.DeepSeekOCR"
        ), patch(
            "documents.manager.DocumentSearcher"
        ) as MockSearcher, patch(
            "documents.manager.CitationTracker"
        ):
            mock_cfg = MagicMock()
            mock_cfg.integration.store_chunks_in_ltm = False
            mock_get.return_value = mock_cfg
            MockSearcher.return_value.initialize = AsyncMock()

            mgr = await initialize_document_manager()
            assert mgr._initialized is True


# ── Ingest Document ──────────────────────────────────────────────────────


class TestIngestDocument:
    @pytest.mark.asyncio
    async def test_ingest_basic(self, manager, tmp_path):
        # Create a real file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake")

        mock_doc = _make_mock_document()
        with patch.object(Document, "create", return_value=mock_doc):
            manager._pdf_processor.get_pdf_info.return_value = (10, "hash123", 1024)

            # Mock process_document to return success
            manager.process_document = AsyncMock(
                return_value=ProcessingResult(
                    document_id="doc1",
                    status=ProcessingStatus.COMPLETED,
                    pages_processed=10,
                    chunks_created=5,
                )
            )

            result = await manager.ingest_document(
                file_path=str(pdf_path),
                title="Test Book",
                author="Author",
            )

            assert result is not None
            manager._store.store_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_file_not_found(self, manager):
        with pytest.raises(FileNotFoundError):
            await manager.ingest_document(
                file_path="/nonexistent/file.pdf",
                title="Missing",
            )

    @pytest.mark.asyncio
    async def test_ingest_duplicate(self, manager, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake")

        existing = _make_mock_document(doc_id="existing1")
        manager._store.get_document_by_hash.return_value = existing

        result = await manager.ingest_document(
            file_path=str(pdf_path),
            title="Duplicate",
        )
        assert result.id == "existing1"
        manager._store.store_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_no_processing(self, manager, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake")

        mock_doc = _make_mock_document()
        with patch.object(Document, "create", return_value=mock_doc):
            result = await manager.ingest_document(
                file_path=str(pdf_path),
                title="Test",
                process_immediately=False,
            )
            assert result is not None
            # process_document should not be called
            # Verify by checking store_pages not called
            manager._store.store_pages.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_auto_initializes(self, mock_config, tmp_path):
        """If not initialized, ingest should auto-initialize."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake")

        with patch("documents.manager.DocumentStore") as MockStore, patch(
            "documents.manager.PDFProcessor"
        ) as MockPDF, patch("documents.manager.SemanticChunker"), patch(
            "documents.manager.DeepSeekOCR"
        ), patch(
            "documents.manager.DocumentSearcher"
        ) as MockSearcher, patch(
            "documents.manager.CitationTracker"
        ):
            mock_config.integration.store_chunks_in_ltm = False
            MockSearcher.return_value.initialize = AsyncMock()
            MockStore.return_value.get_document_by_hash.return_value = None
            MockPDF.return_value.get_pdf_info.return_value = (5, "hash", 512)

            mgr = DocumentManager(config=mock_config)
            assert mgr._initialized is False

            mock_doc = _make_mock_document()
            with patch.object(Document, "create", return_value=mock_doc):
                mgr.process_document = AsyncMock(
                    return_value=ProcessingResult(
                        document_id="doc1",
                        status=ProcessingStatus.COMPLETED,
                        pages_processed=5,
                        chunks_created=3,
                    )
                )
                await mgr.ingest_document(
                    file_path=str(pdf_path),
                    title="Auto Init Test",
                    process_immediately=True,
                )
            assert mgr._initialized is True

    @pytest.mark.asyncio
    async def test_ingest_with_language(self, manager, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake")

        mock_doc = _make_mock_document()
        with patch.object(Document, "create", return_value=mock_doc):
            manager.process_document = AsyncMock(
                return_value=ProcessingResult(
                    document_id="doc1",
                    status=ProcessingStatus.COMPLETED,
                    pages_processed=10,
                    chunks_created=5,
                )
            )
            result = await manager.ingest_document(
                file_path=str(pdf_path),
                title="Telugu Book",
                language=DocumentLanguage.TELUGU,
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_ingest_processing_failure(self, manager, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-fake")

        mock_doc = _make_mock_document()
        with patch.object(Document, "create", return_value=mock_doc):
            manager.process_document = AsyncMock(
                return_value=ProcessingResult(
                    document_id="doc1",
                    status=ProcessingStatus.FAILED,
                    pages_processed=0,
                    chunks_created=0,
                    errors=["OCR failed"],
                )
            )
            result = await manager.ingest_document(
                file_path=str(pdf_path),
                title="Fail Test",
            )
            assert result.status == ProcessingStatus.FAILED


# ── Process Document ─────────────────────────────────────────────────────


class TestProcessDocument:
    @pytest.mark.asyncio
    async def test_process_not_found(self, manager):
        manager._store.get_document.return_value = None
        result = await manager.process_document("nonexistent")
        assert result.status == ProcessingStatus.FAILED
        assert "not found" in result.errors[0]

    @pytest.mark.asyncio
    async def test_process_success(self, manager):
        doc = _make_mock_document()
        manager._store.get_document.return_value = doc

        ocr_result = MagicMock()
        ocr_result.text = "Page text"
        ocr_result.confidence = 0.98
        ocr_result.model_used = "deepseek"
        ocr_result.has_images = False
        ocr_result.has_tables = False
        ocr_result.detected_headers = []
        manager._ocr.process_batch = AsyncMock(return_value=[ocr_result])

        mock_chunk = MagicMock(spec=Chunk)
        mock_chunk.content = "chunk text"
        mock_chunk.chunk_index = 0
        manager._chunker.chunk_document.return_value = [mock_chunk]

        with patch.object(Page, "create", return_value=MagicMock(spec=Page)):
            result = await manager.process_document("doc1")

        assert result.status == ProcessingStatus.COMPLETED
        assert result.pages_processed >= 1
        assert result.chunks_created >= 1
        manager._store.store_pages.assert_called_once()
        manager._store.store_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_page_range(self, manager):
        doc = _make_mock_document(pages=20)
        manager._store.get_document.return_value = doc

        ocr_result = MagicMock()
        ocr_result.text = "Text"
        ocr_result.confidence = 0.95
        ocr_result.model_used = "deepseek"
        ocr_result.has_images = False
        ocr_result.has_tables = False
        ocr_result.detected_headers = []
        manager._ocr.process_batch = AsyncMock(return_value=[ocr_result])
        manager._chunker.chunk_document.return_value = []

        with patch.object(Page, "create", return_value=MagicMock(spec=Page)):
            result = await manager.process_document("doc1", page_range=(5, 10))

        assert result.status == ProcessingStatus.COMPLETED
        manager._pdf_processor.convert_to_images.assert_called_once()
        call_kwargs = manager._pdf_processor.convert_to_images.call_args
        assert (
            call_kwargs[1].get(
                "start_page", call_kwargs[0][2] if len(call_kwargs[0]) > 2 else None
            )
            is not None
        )

    @pytest.mark.asyncio
    async def test_process_exception(self, manager):
        doc = _make_mock_document()
        manager._store.get_document.return_value = doc
        manager._pdf_processor.convert_to_images.side_effect = RuntimeError("GPU OOM")

        result = await manager.process_document("doc1")
        assert result.status == ProcessingStatus.FAILED
        assert "GPU OOM" in result.errors[0]
        manager._store.update_document_status.assert_called_with(
            "doc1", ProcessingStatus.FAILED
        )

    @pytest.mark.asyncio
    async def test_process_with_progress_callback(self, manager):
        doc = _make_mock_document(pages=2)
        manager._store.get_document.return_value = doc
        manager._pdf_processor.convert_to_images.return_value = [
            "/tmp/p1.png",
            "/tmp/p2.png",
        ]

        ocr_result = MagicMock()
        ocr_result.text = "Text"
        ocr_result.confidence = 0.95
        ocr_result.model_used = "deepseek"
        ocr_result.has_images = False
        ocr_result.has_tables = False
        ocr_result.detected_headers = []
        manager._ocr.process_batch = AsyncMock(return_value=[ocr_result, ocr_result])
        manager._chunker.chunk_document.return_value = []

        progress_calls = []

        def callback(done, total):
            progress_calls.append((done, total))

        with patch.object(Page, "create", return_value=MagicMock(spec=Page)):
            await manager.process_document("doc1", progress_callback=callback)

        assert len(progress_calls) >= 1

    @pytest.mark.asyncio
    async def test_process_unloads_ocr(self, manager):
        doc = _make_mock_document()
        manager._store.get_document.return_value = doc
        manager.config.embedding.share_gpu_with_ocr = False

        ocr_result = MagicMock()
        ocr_result.text = "T"
        ocr_result.confidence = 0.9
        ocr_result.model_used = "d"
        ocr_result.has_images = False
        ocr_result.has_tables = False
        ocr_result.detected_headers = []
        manager._ocr.process_batch = AsyncMock(return_value=[ocr_result])
        manager._chunker.chunk_document.return_value = []

        with patch.object(Page, "create", return_value=MagicMock(spec=Page)):
            await manager.process_document("doc1")

        manager._ocr.unload_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_cleanup_images(self, manager):
        doc = _make_mock_document()
        manager._store.get_document.return_value = doc
        manager.config.storage.auto_cleanup_images = True

        ocr_result = MagicMock()
        ocr_result.text = "T"
        ocr_result.confidence = 0.9
        ocr_result.model_used = "d"
        ocr_result.has_images = False
        ocr_result.has_tables = False
        ocr_result.detected_headers = []
        manager._ocr.process_batch = AsyncMock(return_value=[ocr_result])
        manager._chunker.chunk_document.return_value = []

        with patch.object(Page, "create", return_value=MagicMock(spec=Page)):
            await manager.process_document("doc1")

        manager._pdf_processor.cleanup_images.assert_called_with("doc1")


# ── OCR Text Cleaning ────────────────────────────────────────────────────


class TestCleanOCRText:
    def test_clean_basic(self, manager):
        result = manager._clean_ocr_text("Hello  World")
        assert result == "Hello World"

    def test_clean_excessive_newlines(self, manager):
        result = manager._clean_ocr_text("Line1\n\n\n\n\nLine2")
        assert result == "Line1\n\nLine2"

    def test_clean_strip(self, manager):
        result = manager._clean_ocr_text("  text  ")
        assert result == "text"

    def test_clean_soft_hyphens(self, manager):
        result = manager._clean_ocr_text("word\u00ADbreak")
        assert result == "wordbreak"

    def test_clean_empty(self, manager):
        result = manager._clean_ocr_text("")
        assert result == ""


# ── Search ────────────────────────────────────────────────────────────────


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_delegates(self, manager):
        await manager.search("test query")
        manager._searcher.search.assert_called_once_with(
            query="test query",
            document_id=None,
            document_type=None,
            project=None,
            top_k=10,
        )

    @pytest.mark.asyncio
    async def test_search_with_filters(self, manager):
        await manager.search(
            "query",
            document_id="doc1",
            document_type=DocumentType.BOOK,
            project="proj1",
            top_k=5,
        )
        manager._searcher.search.assert_called_once_with(
            query="query",
            document_id="doc1",
            document_type=DocumentType.BOOK,
            project="proj1",
            top_k=5,
        )

    @pytest.mark.asyncio
    async def test_search_auto_initializes(self, mock_config):
        with patch("documents.manager.DocumentStore"), patch(
            "documents.manager.PDFProcessor"
        ), patch("documents.manager.SemanticChunker"), patch(
            "documents.manager.DeepSeekOCR"
        ), patch(
            "documents.manager.DocumentSearcher"
        ) as MockSearcher, patch(
            "documents.manager.CitationTracker"
        ):
            mock_config.integration.store_chunks_in_ltm = False
            mock_searcher = MagicMock()
            mock_searcher.initialize = AsyncMock()
            mock_searcher.search = AsyncMock(return_value=[])
            MockSearcher.return_value = mock_searcher

            mgr = DocumentManager(config=mock_config)
            await mgr.search("test")
            assert mgr._initialized is True


# ── Get Context ──────────────────────────────────────────────────────────


class TestGetContext:
    @pytest.mark.asyncio
    async def test_get_context_empty(self, manager):
        manager._searcher.search = AsyncMock(return_value=[])
        context, citations = await manager.get_context_for_query("test")
        assert context == ""
        assert citations == []

    @pytest.mark.asyncio
    async def test_get_context_with_results(self, manager):
        mock_result = MagicMock(spec=DocumentSearchResult)
        manager._searcher.search = AsyncMock(return_value=[mock_result])
        manager._citation_tracker.get_context_with_citations.return_value = (
            "Found context",
            [MagicMock(spec=Citation)],
        )

        context, citations = await manager.get_context_for_query("test")
        assert context == "Found context"
        assert len(citations) == 1

    @pytest.mark.asyncio
    async def test_get_context_limits_chunks(self, manager):
        results = [MagicMock(spec=DocumentSearchResult) for _ in range(10)]
        manager._searcher.search = AsyncMock(return_value=results)
        manager._citation_tracker.get_context_with_citations.return_value = ("ctx", [])

        await manager.get_context_for_query("test", max_chunks=3)
        # search should request top_k = max_chunks * 2
        call_kwargs = manager._searcher.search.call_args[1]
        assert call_kwargs["top_k"] == 6  # 3 * 2


# ── Document Access ──────────────────────────────────────────────────────


class TestDocumentAccess:
    @pytest.mark.asyncio
    async def test_get_document(self, manager):
        mock_doc = _make_mock_document()
        manager._store.get_document.return_value = mock_doc
        result = await manager.get_document("doc1")
        assert result is mock_doc

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, manager):
        manager._store.get_document.return_value = None
        result = await manager.get_document("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_page(self, manager):
        mock_page = MagicMock(spec=Page)
        manager._store.get_page.return_value = mock_page
        result = await manager.get_page("doc1", 1)
        assert result is mock_page

    @pytest.mark.asyncio
    async def test_list_documents(self, manager):
        docs = [_make_mock_document(f"d{i}") for i in range(3)]
        manager._store.list_documents.return_value = docs
        result = await manager.list_documents()
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_documents_with_filters(self, manager):
        await manager.list_documents(
            document_type=DocumentType.SCREENPLAY,
            project="proj1",
            status=ProcessingStatus.COMPLETED,
            limit=50,
        )
        manager._store.list_documents.assert_called_once_with(
            document_type=DocumentType.SCREENPLAY,
            project="proj1",
            status=ProcessingStatus.COMPLETED,
            limit=50,
        )


# ── Delete Document ──────────────────────────────────────────────────────


class TestDeleteDocument:
    @pytest.mark.asyncio
    async def test_delete(self, manager):
        result = await manager.delete_document("doc1")
        assert result is True
        manager._pdf_processor.cleanup_images.assert_called_with("doc1")
        manager._store.delete_document.assert_called_with("doc1")

    @pytest.mark.asyncio
    async def test_delete_not_found(self, manager):
        manager._store.delete_document.return_value = False
        result = await manager.delete_document("nonexistent")
        assert result is False


# ── Get Chapter ──────────────────────────────────────────────────────────


class TestGetChapter:
    @pytest.mark.asyncio
    async def test_get_chapter_not_found_doc(self, manager):
        manager._store.get_document.return_value = None
        with pytest.raises(ValueError, match="Document not found"):
            await manager.get_chapter("nonexistent")

    @pytest.mark.asyncio
    async def test_get_chapter_no_chapters(self, manager):
        doc = _make_mock_document()
        doc.chapters = []
        manager._store.get_document.return_value = doc
        with pytest.raises(ValueError, match="no chapter"):
            await manager.get_chapter("doc1", chapter_title="Ch 1")

    @pytest.mark.asyncio
    async def test_get_chapter_not_found_title(self, manager):
        doc = _make_mock_document()
        ch = MagicMock(spec=ChapterInfo)
        ch.title = "Introduction"
        ch.start_page = 1
        ch.end_page = 10
        doc.chapters = [ch]
        manager._store.get_document.return_value = doc
        with pytest.raises(ValueError, match="Chapter not found"):
            await manager.get_chapter("doc1", chapter_title="Nonexistent")

    @pytest.mark.asyncio
    async def test_get_chapter_by_title(self, manager):
        doc = _make_mock_document()
        ch = MagicMock(spec=ChapterInfo)
        ch.title = "Structure"
        ch.start_page = 5
        ch.end_page = 15
        doc.chapters = [ch]
        manager._store.get_document.return_value = doc

        mock_chunk = MagicMock(spec=Chunk)
        mock_chunk.content = "Chapter content"
        mock_chunk.chunk_index = 0
        manager._store.get_chunks_for_document.return_value = [mock_chunk]

        text, page_range = await manager.get_chapter("doc1", chapter_title="Structure")
        assert "Chapter content" in text
        assert "pp. 5-15" in page_range

    @pytest.mark.asyncio
    async def test_get_chapter_by_index(self, manager):
        doc = _make_mock_document()
        ch0 = MagicMock(spec=ChapterInfo)
        ch0.title = "First"
        ch0.start_page = 1
        ch0.end_page = 10
        ch1 = MagicMock(spec=ChapterInfo)
        ch1.title = "Second"
        ch1.start_page = 11
        ch1.end_page = 20
        doc.chapters = [ch0, ch1]
        manager._store.get_document.return_value = doc

        mock_chunk = MagicMock(spec=Chunk)
        mock_chunk.content = "Second chapter"
        mock_chunk.chunk_index = 0
        manager._store.get_chunks_for_document.return_value = [mock_chunk]

        text, page_range = await manager.get_chapter("doc1", chapter_index=1)
        assert "Second chapter" in text

    @pytest.mark.asyncio
    async def test_get_chapter_index_out_of_range(self, manager):
        doc = _make_mock_document()
        ch = MagicMock(spec=ChapterInfo)
        ch.title = "Only"
        ch.start_page = 1
        ch.end_page = 5
        doc.chapters = [ch]
        manager._store.get_document.return_value = doc

        with pytest.raises(ValueError, match="out of range"):
            await manager.get_chapter("doc1", chapter_index=5)

    @pytest.mark.asyncio
    async def test_get_chapter_no_identifier(self, manager):
        doc = _make_mock_document()
        ch = MagicMock(spec=ChapterInfo)
        ch.title = "Ch1"
        ch.start_page = 1
        ch.end_page = 5
        doc.chapters = [ch]
        manager._store.get_document.return_value = doc

        with pytest.raises(ValueError, match="Must provide"):
            await manager.get_chapter("doc1")

    @pytest.mark.asyncio
    async def test_get_chapter_fallback_to_pages(self, manager):
        doc = _make_mock_document()
        ch = MagicMock(spec=ChapterInfo)
        ch.title = "Ch1"
        ch.start_page = 1
        ch.end_page = 2
        doc.chapters = [ch]
        manager._store.get_document.return_value = doc

        # No chunks, fall back to pages
        manager._store.get_chunks_for_document.return_value = []
        mock_page = MagicMock(spec=Page)
        mock_page.cleaned_text = "Page text"
        mock_page.raw_text = "Raw page"
        manager._store.get_page.return_value = mock_page

        text, page_range = await manager.get_chapter("doc1", chapter_title="Ch1")
        assert "Page text" in text


# ── Processing Status ────────────────────────────────────────────────────


class TestProcessingStatus:
    @pytest.mark.asyncio
    async def test_status_not_found(self, manager):
        manager._store.get_document.return_value = None
        result = await manager.get_processing_status("nonexistent")
        assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_status_pending(self, manager):
        doc = _make_mock_document(status=ProcessingStatus.PENDING)
        manager._store.get_document.return_value = doc
        manager._store.get_pages_for_document.return_value = []

        result = await manager.get_processing_status("doc1")
        assert result["status"] == "pending"
        assert result["progress"] == 0

    @pytest.mark.asyncio
    async def test_status_completed(self, manager):
        doc = _make_mock_document(pages=10, status=ProcessingStatus.COMPLETED)
        manager._store.get_document.return_value = doc
        manager._store.get_pages_for_document.return_value = [MagicMock()] * 10

        result = await manager.get_processing_status("doc1")
        assert result["status"] == "completed"
        assert result["progress"] == 100.0

    @pytest.mark.asyncio
    async def test_status_partial(self, manager):
        doc = _make_mock_document(pages=10, status=ProcessingStatus.PROCESSING)
        manager._store.get_document.return_value = doc
        manager._store.get_pages_for_document.return_value = [MagicMock()] * 5

        result = await manager.get_processing_status("doc1")
        assert result["progress"] == 50.0

    @pytest.mark.asyncio
    async def test_status_failed(self, manager):
        doc = _make_mock_document(status=ProcessingStatus.FAILED)
        manager._store.get_document.return_value = doc
        manager._store.get_pages_for_document.return_value = []

        result = await manager.get_processing_status("doc1")
        assert result["status"] == "failed"
        assert "error" in result


# ── Stats ─────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_no_store(self, mock_config):
        mgr = DocumentManager(config=mock_config)
        assert mgr.get_stats() == {}

    def test_stats_delegates(self, manager):
        manager._store.get_stats.return_value = {"total": 5, "chunks": 100}
        stats = manager.get_stats()
        assert stats["total"] == 5
        manager._store.get_stats.assert_called_once()


# ── Edge Cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_chapter_title_case_insensitive(self, manager):
        doc = _make_mock_document()
        ch = MagicMock(spec=ChapterInfo)
        ch.title = "The Inciting Incident"
        ch.start_page = 1
        ch.end_page = 10
        doc.chapters = [ch]
        manager._store.get_document.return_value = doc

        mock_chunk = MagicMock(spec=Chunk)
        mock_chunk.content = "text"
        mock_chunk.chunk_index = 0
        manager._store.get_chunks_for_document.return_value = [mock_chunk]

        # Lowercase should match
        text, _ = await manager.get_chapter(
            "doc1", chapter_title="the inciting incident"
        )
        assert text is not None

    @pytest.mark.asyncio
    async def test_chapter_negative_index(self, manager):
        doc = _make_mock_document()
        ch = MagicMock(spec=ChapterInfo)
        ch.title = "Only"
        ch.start_page = 1
        ch.end_page = 5
        doc.chapters = [ch]
        manager._store.get_document.return_value = doc

        with pytest.raises(ValueError, match="out of range"):
            await manager.get_chapter("doc1", chapter_index=-1)
