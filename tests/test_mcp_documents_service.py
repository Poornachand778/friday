"""
Tests for MCP Document Processing Service
==========================================

Tests the service layer that wraps DocumentManager, BookUnderstandingStore,
BookComprehensionEngine, and MentorEngine for MCP tool access.

All async singletons are mocked to avoid needing real databases, LLMs, or OCR.
Uses asyncio.get_event_loop().run_until_complete() for async test compatibility.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


# ============================================================
# Helpers
# ============================================================


def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


@pytest.fixture(autouse=True)
def reset_service_singletons():
    """Reset all module-level singletons between tests."""
    import mcp.documents.service as svc

    svc._manager = None
    svc._comprehension_engine = None
    svc._mentor_engine = None
    svc._understanding_store = None
    svc._sync_manager = None
    yield
    svc._manager = None
    svc._comprehension_engine = None
    svc._mentor_engine = None
    svc._understanding_store = None
    svc._sync_manager = None


def _mock_document(**overrides):
    """Create a mock Document with all attributes the service accesses."""
    doc = MagicMock()
    doc.id = overrides.get("id", "doc-123")
    doc.file_path = overrides.get("file_path", "/path/to/book.pdf")
    doc.metadata = MagicMock()
    doc.metadata.title = overrides.get("title", "Story")
    doc.metadata.author = overrides.get("author", "Robert McKee")
    doc.metadata.isbn = overrides.get("isbn", "978-0060391683")
    doc.metadata.publication_date = overrides.get("publication_date", None)
    doc.total_pages = overrides.get("total_pages", 500)
    doc.status = MagicMock()
    doc.status.value = overrides.get("status", "completed")
    doc.document_type = MagicMock()
    doc.document_type.value = overrides.get("doc_type", "book")
    doc.language = MagicMock()
    doc.language.value = overrides.get("language", "en")
    doc.project = overrides.get("project", None)
    doc.created_at = overrides.get("created_at", datetime(2025, 1, 1))
    doc.processed_at = overrides.get("processed_at", datetime(2025, 1, 1))
    doc.chapters = overrides.get("chapters", [])
    return doc


def _mock_chapter(number=1, title="Chapter 1", start=1, end=20):
    """Create a mock ChapterInfo."""
    ch = MagicMock()
    ch.number = number
    ch.title = title
    ch.start_page = start
    ch.end_page = end
    return ch


def _mock_understanding(**overrides):
    """Create a mock BookUnderstanding."""
    u = MagicMock()
    u.id = overrides.get("id", "und-123")
    u.document_id = overrides.get("document_id", "doc-123")
    u.title = overrides.get("title", "Story")
    u.author = overrides.get("author", "Robert McKee")
    u.summary = overrides.get("summary", "A guide to screenwriting.")
    u.main_argument = overrides.get("main_argument", "Story is structure")
    u.domains = overrides.get("domains", ["screenwriting"])
    u.concepts = overrides.get("concepts", [MagicMock(), MagicMock()])
    u.principles = overrides.get("principles", [MagicMock()])
    u.techniques = overrides.get("techniques", [MagicMock()])
    u.examples = overrides.get("examples", [MagicMock()])
    u.comprehension_quality = overrides.get("quality", 0.85)
    u.study_completed_at = overrides.get("studied_at", datetime(2025, 1, 15))
    u.to_dict.return_value = {"id": u.id, "title": u.title}
    return u


def _mock_search_result(**overrides):
    """Create a mock DocumentSearchResult."""
    result = MagicMock()
    result.citation = MagicMock()
    result.citation.format_inline.return_value = overrides.get(
        "citation", "[Story, pp. 45-47]"
    )
    result.citation.document_title = overrides.get("doc_title", "Story")
    result.citation.page_range = overrides.get("page_range", "pp. 45-47")
    result.citation.chapter = overrides.get("chapter", "Chapter 3")
    result.highlight = overrides.get("highlight", "The inciting incident...")
    result.relevance = overrides.get("relevance", 0.923)
    return result


def _mock_citation(**overrides):
    """Create a mock Citation."""
    c = MagicMock()
    c.document_title = overrides.get("doc_title", "Story")
    c.page_range = overrides.get("page_range", "pp. 45-47")
    c.chapter = overrides.get("chapter", "Chapter 3")
    c.quote = overrides.get("quote", "Every scene must turn.")
    c.format_inline.return_value = overrides.get("formatted", "[Story, pp. 45-47]")
    return c


# ============================================================
# Document Management Tests
# ============================================================


class TestDocumentIngest:
    """Tests for document_ingest service function."""

    def test_success(self):
        from mcp.documents.service import document_ingest

        mock_mgr = AsyncMock()
        mock_doc = _mock_document()
        mock_mgr.ingest_document.return_value = mock_doc

        with (
            patch("mcp.documents.service.get_document_manager", return_value=mock_mgr),
            patch("mcp.documents.service.Path") as MockPath,
        ):
            MockPath.return_value.exists.return_value = True
            result = _run(
                document_ingest(
                    file_path="/path/to/book.pdf",
                    title="Story",
                    author="Robert McKee",
                )
            )

        assert result["success"] is True
        assert result["document_id"] == "doc-123"
        assert result["title"] == "Story"
        assert result["total_pages"] == 500

    def test_file_not_found(self):
        from mcp.documents.service import document_ingest

        with (
            patch(
                "mcp.documents.service.get_document_manager",
                return_value=AsyncMock(),
            ),
            patch("mcp.documents.service.Path") as MockPath,
        ):
            MockPath.return_value.exists.return_value = False
            result = _run(document_ingest(file_path="/nonexistent.pdf", title="X"))

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_exception_handling(self):
        from mcp.documents.service import document_ingest

        mock_mgr = AsyncMock()
        mock_mgr.ingest_document.side_effect = RuntimeError("OCR failed")

        with (
            patch("mcp.documents.service.get_document_manager", return_value=mock_mgr),
            patch("mcp.documents.service.Path") as MockPath,
        ):
            MockPath.return_value.exists.return_value = True
            result = _run(document_ingest(file_path="/path/to/book.pdf", title="Story"))

        assert result["success"] is False
        assert "OCR failed" in result["error"]


class TestDocumentSearch:
    """Tests for document_search service function."""

    def test_success(self):
        from mcp.documents.service import document_search

        mock_mgr = AsyncMock()
        mock_mgr.search.return_value = [
            _mock_search_result(),
            _mock_search_result(relevance=0.812),
        ]

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_search(query="inciting incident"))

        assert result["success"] is True
        assert result["query"] == "inciting incident"
        assert result["count"] == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["citation"] == "[Story, pp. 45-47]"
        assert result["results"][0]["relevance"] == 0.923

    def test_top_k_capped_at_20(self):
        from mcp.documents.service import document_search

        mock_mgr = AsyncMock()
        mock_mgr.search.return_value = []

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            _run(document_search(query="test", top_k=50))

        call_kwargs = mock_mgr.search.call_args[1]
        assert call_kwargs["top_k"] == 20

    def test_exception_handling(self):
        from mcp.documents.service import document_search

        mock_mgr = AsyncMock()
        mock_mgr.search.side_effect = RuntimeError("DB error")

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_search(query="test"))

        assert result["success"] is False
        assert "DB error" in result["error"]


class TestDocumentGetContext:
    """Tests for document_get_context service function."""

    def test_success(self):
        from mcp.documents.service import document_get_context

        mock_mgr = AsyncMock()
        citations = [_mock_citation(), _mock_citation(chapter="Chapter 5")]
        mock_mgr.get_context_for_query.return_value = (
            "Context about character arcs...",
            citations,
        )

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_get_context(query="character arcs"))

        assert result["success"] is True
        assert "character arcs" in result["context"]
        assert len(result["citations"]) == 2
        assert result["citations"][0]["formatted"] == "[Story, pp. 45-47]"

    def test_long_quote_truncated(self):
        from mcp.documents.service import document_get_context

        mock_mgr = AsyncMock()
        long_quote = "x" * 300
        citation = _mock_citation(quote=long_quote)
        mock_mgr.get_context_for_query.return_value = ("Context...", [citation])

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_get_context(query="test"))

        assert result["citations"][0]["quote"].endswith("...")
        assert len(result["citations"][0]["quote"]) == 203  # 200 + "..."

    def test_exception_handling(self):
        from mcp.documents.service import document_get_context

        mock_mgr = AsyncMock()
        mock_mgr.get_context_for_query.side_effect = RuntimeError("Search failed")

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_get_context(query="test"))

        assert result["success"] is False


class TestDocumentList:
    """Tests for document_list service function."""

    def test_success(self):
        from mcp.documents.service import document_list

        mock_mgr = AsyncMock()
        mock_mgr.list_documents.return_value = [
            _mock_document(),
            _mock_document(id="doc-456", title="Screenplay"),
        ]

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_list())

        assert result["success"] is True
        assert result["count"] == 2
        assert result["documents"][0]["id"] == "doc-123"
        assert result["documents"][0]["title"] == "Story"

    def test_with_filters(self):
        from mcp.documents.service import document_list

        mock_mgr = AsyncMock()
        mock_mgr.list_documents.return_value = []

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(
                document_list(
                    document_type="book", project="my-project", status="completed"
                )
            )

        assert result["success"] is True
        assert result["count"] == 0

    def test_exception_handling(self):
        from mcp.documents.service import document_list

        mock_mgr = AsyncMock()
        mock_mgr.list_documents.side_effect = RuntimeError("DB error")

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_list())

        assert result["success"] is False


class TestDocumentGet:
    """Tests for document_get service function."""

    def test_success_with_chapters(self):
        from mcp.documents.service import document_get

        mock_mgr = AsyncMock()
        chapters = [_mock_chapter(1, "The Inciting Incident", 1, 30)]
        doc = _mock_document(chapters=chapters)
        mock_mgr.get_document.return_value = doc

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_get("doc-123"))

        assert result["success"] is True
        assert result["document"]["id"] == "doc-123"
        assert result["document"]["title"] == "Story"
        assert len(result["document"]["chapters"]) == 1
        assert result["document"]["chapters"][0]["number"] == 1
        assert result["document"]["chapters"][0]["title"] == "The Inciting Incident"
        assert result["document"]["chapters"][0]["start_page"] == 1

    def test_not_found(self):
        from mcp.documents.service import document_get

        mock_mgr = AsyncMock()
        mock_mgr.get_document.return_value = None

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_get("nonexistent-id"))

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_exception_handling(self):
        from mcp.documents.service import document_get

        mock_mgr = AsyncMock()
        mock_mgr.get_document.side_effect = RuntimeError("DB error")

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_get("doc-123"))

        assert result["success"] is False


class TestDocumentGetChapter:
    """Tests for document_get_chapter service function."""

    def test_success_by_title(self):
        from mcp.documents.service import document_get_chapter

        mock_mgr = AsyncMock()
        mock_mgr.get_chapter.return_value = ("Chapter text here...", "pp. 45-67")

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(
                document_get_chapter(
                    document_id="doc-123", chapter_title="The Inciting Incident"
                )
            )

        assert result["success"] is True
        assert result["text"] == "Chapter text here..."
        assert result["page_range"] == "pp. 45-67"

    def test_value_error(self):
        from mcp.documents.service import document_get_chapter

        mock_mgr = AsyncMock()
        mock_mgr.get_chapter.side_effect = ValueError("Chapter not found")

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_get_chapter(document_id="doc-123", chapter_index=99))

        assert result["success"] is False
        assert "Chapter not found" in result["error"]

    def test_exception_handling(self):
        from mcp.documents.service import document_get_chapter

        mock_mgr = AsyncMock()
        mock_mgr.get_chapter.side_effect = RuntimeError("Read error")

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_get_chapter(document_id="doc-123", chapter_index=0))

        assert result["success"] is False


class TestDocumentStatus:
    """Tests for document_status service function."""

    def test_success(self):
        from mcp.documents.service import document_status

        mock_mgr = AsyncMock()
        mock_mgr.get_processing_status.return_value = {
            "status": "processing",
            "progress": 0.45,
            "current_page": 225,
            "total_pages": 500,
        }

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_status("doc-123"))

        assert result["success"] is True
        assert result["status"] == "processing"
        assert result["progress"] == 0.45
        assert result["current_page"] == 225

    def test_exception_handling(self):
        from mcp.documents.service import document_status

        mock_mgr = AsyncMock()
        mock_mgr.get_processing_status.side_effect = RuntimeError("Error")

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_status("doc-123"))

        assert result["success"] is False


class TestDocumentDelete:
    """Tests for document_delete service function."""

    def test_success(self):
        from mcp.documents.service import document_delete

        mock_mgr = AsyncMock()

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_delete("doc-123"))

        assert result["success"] is True
        assert result["deleted"] is True

    def test_value_error(self):
        from mcp.documents.service import document_delete

        mock_mgr = AsyncMock()
        mock_mgr.delete_document.side_effect = ValueError("Document not found")

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_delete("nonexistent"))

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_exception_handling(self):
        from mcp.documents.service import document_delete

        mock_mgr = AsyncMock()
        mock_mgr.delete_document.side_effect = RuntimeError("DB error")

        with patch("mcp.documents.service.get_document_manager", return_value=mock_mgr):
            result = _run(document_delete("doc-123"))

        assert result["success"] is False


# ============================================================
# Book Understanding Tests
# ============================================================


class TestBookStudy:
    """Tests for book_study service function."""

    def test_already_studied(self):
        from mcp.documents.service import book_study

        mock_mgr = AsyncMock()
        mock_store = MagicMock()
        existing = _mock_understanding()
        mock_store.get_understanding_by_document.return_value = existing

        with (
            patch("mcp.documents.service.get_document_manager", return_value=mock_mgr),
            patch(
                "mcp.documents.service._get_understanding_store",
                return_value=mock_store,
            ),
            patch(
                "mcp.documents.service._get_comprehension_engine",
                return_value=MagicMock(),
            ),
        ):
            result = _run(book_study("doc-123"))

        assert result["success"] is True
        assert result["already_studied"] is True
        assert result["understanding_id"] == "und-123"
        assert result["concepts"] == 2

    def test_document_not_found(self):
        from mcp.documents.service import book_study

        mock_mgr = AsyncMock()
        mock_mgr.get_document.return_value = None
        mock_store = MagicMock()
        mock_store.get_understanding_by_document.return_value = None

        with (
            patch("mcp.documents.service.get_document_manager", return_value=mock_mgr),
            patch(
                "mcp.documents.service._get_understanding_store",
                return_value=mock_store,
            ),
            patch(
                "mcp.documents.service._get_comprehension_engine",
                return_value=MagicMock(),
            ),
        ):
            result = _run(book_study("nonexistent"))

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_no_chunks(self):
        from mcp.documents.service import book_study

        mock_mgr = AsyncMock()
        mock_mgr.get_document.return_value = _mock_document()
        mock_mgr.get_chunks.return_value = []
        mock_store = MagicMock()
        mock_store.get_understanding_by_document.return_value = None

        with (
            patch("mcp.documents.service.get_document_manager", return_value=mock_mgr),
            patch(
                "mcp.documents.service._get_understanding_store",
                return_value=mock_store,
            ),
            patch(
                "mcp.documents.service._get_comprehension_engine",
                return_value=MagicMock(),
            ),
        ):
            result = _run(book_study("doc-123"))

        assert result["success"] is False
        assert "no chunks" in result["error"].lower()

    def test_success(self):
        from mcp.documents.service import book_study

        mock_mgr = AsyncMock()
        mock_mgr.get_document.return_value = _mock_document()
        mock_mgr.get_chunks.return_value = [MagicMock()]

        mock_store = MagicMock()
        mock_store.get_understanding_by_document.return_value = None

        mock_engine = MagicMock()
        mock_engine._current_job_id = "job-001"
        mock_engine._config = MagicMock()
        mock_engine._config.thorough_mode = True
        understanding = _mock_understanding()
        mock_engine.comprehend = AsyncMock(return_value=understanding)

        with (
            patch("mcp.documents.service.get_document_manager", return_value=mock_mgr),
            patch(
                "mcp.documents.service._get_understanding_store",
                return_value=mock_store,
            ),
            patch(
                "mcp.documents.service._get_comprehension_engine",
                return_value=mock_engine,
            ),
            # Prevent Knowledge Graph integration from trying to init
            patch("memory.get_memory_manager", return_value=None),
        ):
            result = _run(book_study("doc-123", voice_enabled=False))

        assert result["success"] is True
        assert result["understanding_id"] == "und-123"
        assert result["title"] == "Story"
        assert result["concepts"] == 2
        assert result["principles"] == 1
        mock_store.store_understanding.assert_called_once_with(understanding)

    def test_exception_handling(self):
        from mcp.documents.service import book_study

        mock_mgr = AsyncMock()
        mock_mgr.get_document.side_effect = RuntimeError("DB error")
        mock_store = MagicMock()
        mock_store.get_understanding_by_document.return_value = None

        mock_engine = MagicMock()
        mock_engine._current_job_id = "job-fail"

        with (
            patch("mcp.documents.service.get_document_manager", return_value=mock_mgr),
            patch(
                "mcp.documents.service._get_understanding_store",
                return_value=mock_store,
            ),
            patch(
                "mcp.documents.service._get_comprehension_engine",
                return_value=mock_engine,
            ),
        ):
            result = _run(book_study("doc-123"))

        assert result["success"] is False
        assert "DB error" in result["error"]


class TestBookStudyStatus:
    """Tests for book_study_status service function."""

    def test_by_job_id(self):
        from mcp.documents.service import book_study_status

        mock_tracker = MagicMock()
        mock_tracker.get_status.return_value = {
            "voice_status": "Boss, studying Chapter 3...",
            "status": "studying",
            "progress": 0.3,
        }

        with patch(
            "documents.understanding.job_tracker.get_job_tracker",
            return_value=mock_tracker,
        ):
            result = _run(book_study_status(job_id="job-001"))

        assert result["success"] is True
        assert "Chapter 3" in result["voice_status"]

    def test_job_not_found(self):
        from mcp.documents.service import book_study_status

        mock_tracker = MagicMock()
        mock_tracker.get_status.return_value = None

        with patch(
            "documents.understanding.job_tracker.get_job_tracker",
            return_value=mock_tracker,
        ):
            result = _run(book_study_status(job_id="nonexistent"))

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_by_document_id(self):
        from mcp.documents.service import book_study_status

        mock_job = MagicMock()
        mock_job.get_voice_status.return_value = "Boss, studying Story..."
        mock_job.to_dict.return_value = {"status": "studying"}

        mock_tracker = MagicMock()
        mock_tracker.get_job_by_document.return_value = mock_job

        with patch(
            "documents.understanding.job_tracker.get_job_tracker",
            return_value=mock_tracker,
        ):
            result = _run(book_study_status(document_id="doc-123"))

        assert result["success"] is True
        assert "studying" in result["voice_status"].lower()

    def test_no_active_jobs(self):
        from mcp.documents.service import book_study_status

        mock_tracker = MagicMock()
        mock_tracker.get_active_jobs.return_value = []

        with patch(
            "documents.understanding.job_tracker.get_job_tracker",
            return_value=mock_tracker,
        ):
            result = _run(book_study_status())

        assert result["success"] is True
        assert "no books" in result["voice_status"].lower()
        assert result["active_jobs"] == []

    def test_multiple_active_jobs(self):
        from mcp.documents.service import book_study_status

        mock_job1 = MagicMock()
        mock_job1.to_dict.return_value = {"id": "job-1", "title": "Story"}
        mock_job2 = MagicMock()
        mock_job2.to_dict.return_value = {"id": "job-2", "title": "Screenplay"}

        mock_tracker = MagicMock()
        mock_tracker.get_active_jobs.return_value = [mock_job1, mock_job2]
        mock_tracker.get_active_status_summary.return_value = "Studying 2 books..."

        with patch(
            "documents.understanding.job_tracker.get_job_tracker",
            return_value=mock_tracker,
        ):
            result = _run(book_study_status())

        assert result["success"] is True
        assert result["count"] == 2


class TestBookStudyJobs:
    """Tests for book_study_jobs service function."""

    def test_success(self):
        from mcp.documents.service import book_study_jobs

        mock_tracker = MagicMock()
        mock_tracker.get_all_jobs.return_value = [
            {"id": "job-1", "status": "completed"},
            {"id": "job-2", "status": "studying"},
        ]

        with patch(
            "documents.understanding.job_tracker.get_job_tracker",
            return_value=mock_tracker,
        ):
            result = _run(book_study_jobs())

        assert result["success"] is True
        assert result["count"] == 2

    def test_exception_handling(self):
        from mcp.documents.service import book_study_jobs

        with patch(
            "documents.understanding.job_tracker.get_job_tracker",
            side_effect=RuntimeError("Tracker error"),
        ):
            result = _run(book_study_jobs())

        assert result["success"] is False


class TestBookListStudied:
    """Tests for book_list_studied service function."""

    def test_success(self):
        from mcp.documents.service import book_list_studied

        mock_store = MagicMock()
        mock_store.list_understandings.return_value = [
            _mock_understanding(),
            _mock_understanding(id="und-456", title="Screenplay"),
        ]

        with patch(
            "mcp.documents.service._get_understanding_store",
            return_value=mock_store,
        ):
            result = _run(book_list_studied())

        assert result["success"] is True
        assert result["count"] == 2
        assert result["books"][0]["title"] == "Story"
        assert result["books"][0]["concepts"] == 2

    def test_empty(self):
        from mcp.documents.service import book_list_studied

        mock_store = MagicMock()
        mock_store.list_understandings.return_value = []

        with patch(
            "mcp.documents.service._get_understanding_store",
            return_value=mock_store,
        ):
            result = _run(book_list_studied())

        assert result["success"] is True
        assert result["count"] == 0

    def test_exception_handling(self):
        from mcp.documents.service import book_list_studied

        mock_store = MagicMock()
        mock_store.list_understandings.side_effect = RuntimeError("DB error")

        with patch(
            "mcp.documents.service._get_understanding_store",
            return_value=mock_store,
        ):
            result = _run(book_list_studied())

        assert result["success"] is False


class TestBookGetUnderstanding:
    """Tests for book_get_understanding service function."""

    def test_success(self):
        from mcp.documents.service import book_get_understanding

        mock_store = MagicMock()
        understanding = _mock_understanding()
        mock_store.get_understanding.return_value = understanding

        with patch(
            "mcp.documents.service._get_understanding_store",
            return_value=mock_store,
        ):
            result = _run(book_get_understanding("und-123"))

        assert result["success"] is True
        assert result["understanding"]["id"] == "und-123"

    def test_not_found(self):
        from mcp.documents.service import book_get_understanding

        mock_store = MagicMock()
        mock_store.get_understanding.return_value = None

        with patch(
            "mcp.documents.service._get_understanding_store",
            return_value=mock_store,
        ):
            result = _run(book_get_understanding("nonexistent"))

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_exception_handling(self):
        from mcp.documents.service import book_get_understanding

        mock_store = MagicMock()
        mock_store.get_understanding.side_effect = RuntimeError("DB error")

        with patch(
            "mcp.documents.service._get_understanding_store",
            return_value=mock_store,
        ):
            result = _run(book_get_understanding("und-123"))

        assert result["success"] is False


# ============================================================
# Mentor Tests
# ============================================================


class TestMentorLoadBooks:
    """Tests for mentor_load_books service function."""

    def test_success(self):
        from mcp.documents.service import mentor_load_books

        mock_store = MagicMock()
        book1 = _mock_understanding(id="und-1", title="Story")
        book2 = _mock_understanding(id="und-2", title="Screenplay")
        mock_store.get_understanding.side_effect = [book1, book2]

        mock_mentor = MagicMock()

        with (
            patch(
                "mcp.documents.service._get_understanding_store",
                return_value=mock_store,
            ),
            patch(
                "mcp.documents.service._get_mentor_engine",
                return_value=mock_mentor,
            ),
        ):
            result = _run(mentor_load_books(["und-1", "und-2"]))

        assert result["success"] is True
        assert len(result["loaded_books"]) == 2
        mock_mentor.load_books.assert_called_once()

    def test_some_not_found(self):
        from mcp.documents.service import mentor_load_books

        mock_store = MagicMock()
        book1 = _mock_understanding(id="und-1", title="Story")
        mock_store.get_understanding.side_effect = [book1, None]

        mock_mentor = MagicMock()

        with (
            patch(
                "mcp.documents.service._get_understanding_store",
                return_value=mock_store,
            ),
            patch(
                "mcp.documents.service._get_mentor_engine",
                return_value=mock_mentor,
            ),
        ):
            result = _run(mentor_load_books(["und-1", "und-missing"]))

        assert result["success"] is True
        assert len(result["loaded_books"]) == 1

    def test_exception_handling(self):
        from mcp.documents.service import mentor_load_books

        mock_store = MagicMock()
        mock_store.get_understanding.side_effect = RuntimeError("DB error")

        with (
            patch(
                "mcp.documents.service._get_understanding_store",
                return_value=mock_store,
            ),
            patch(
                "mcp.documents.service._get_mentor_engine",
                return_value=MagicMock(),
            ),
        ):
            result = _run(mentor_load_books(["und-1"]))

        assert result["success"] is False


class TestMentorAnalyze:
    """Tests for mentor_analyze service function."""

    def test_success(self):
        from mcp.documents.service import mentor_analyze

        mock_mentor = MagicMock()
        analysis = MagicMock()
        analysis.to_response.return_value = "Good scene structure..."
        analysis.elements_present = ["conflict", "stakes"]
        analysis.elements_missing = ["subtext"]
        analysis.strengths = ["Strong dialogue"]
        analysis.suggestions = ["Add more subtext"]
        analysis.questions_to_consider = ["What does the character want?"]
        analysis.relevant_principles = [MagicMock(), MagicMock()]
        analysis.applicable_techniques = [MagicMock()]
        analysis.similar_examples = []
        mock_mentor.analyze_scene = AsyncMock(return_value=analysis)

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(mentor_analyze(scene_description="Courtroom confrontation"))

        assert result["success"] is True
        assert "Good scene structure" in result["response"]
        assert result["analysis"]["principles_count"] == 2
        assert result["analysis"]["techniques_count"] == 1
        assert "subtext" in result["analysis"]["elements_missing"]

    def test_exception_handling(self):
        from mcp.documents.service import mentor_analyze

        mock_mentor = MagicMock()
        mock_mentor.analyze_scene = AsyncMock(side_effect=RuntimeError("LLM error"))

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(mentor_analyze(scene_description="test"))

        assert result["success"] is False


class TestMentorBrainstorm:
    """Tests for mentor_brainstorm service function."""

    def test_success(self):
        from mcp.documents.service import mentor_brainstorm

        mock_mentor = MagicMock()
        idea = MagicMock()
        idea.idea = "Use a ticking clock"
        idea.rationale = "Creates urgency"
        idea.based_on = ["Tension principles"]
        idea.source_inspiration = "12 Angry Men"
        brainstorm = MagicMock()
        brainstorm.to_response.return_value = "Here are some ideas..."
        brainstorm.topic = "courtroom climax"
        brainstorm.constraints = ["must resolve in one scene"]
        brainstorm.ideas = [idea]
        brainstorm.suggested_structure = "Build to revelation"
        brainstorm.concepts_applied = ["dramatic irony"]
        brainstorm.techniques_suggested = ["ticking clock"]
        mock_mentor.brainstorm = AsyncMock(return_value=brainstorm)

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(
                mentor_brainstorm(
                    topic="courtroom climax",
                    constraints=["must resolve in one scene"],
                )
            )

        assert result["success"] is True
        assert result["brainstorm"]["topic"] == "courtroom climax"
        assert len(result["brainstorm"]["ideas"]) == 1
        assert result["brainstorm"]["ideas"][0]["idea"] == "Use a ticking clock"

    def test_exception_handling(self):
        from mcp.documents.service import mentor_brainstorm

        mock_mentor = MagicMock()
        mock_mentor.brainstorm = AsyncMock(side_effect=RuntimeError("Error"))

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(mentor_brainstorm(topic="test"))

        assert result["success"] is False


class TestMentorCheckRules:
    """Tests for mentor_check_rules service function."""

    def test_success(self):
        from mcp.documents.service import mentor_check_rules

        mock_mentor = MagicMock()
        followed = MagicMock()
        followed.principle = MagicMock()
        followed.principle.statement = "Every scene needs a turning point"
        followed.evidence = "Scene turns when evidence is revealed"
        violated = MagicMock()
        violated.principle = MagicMock()
        violated.principle.statement = "Show, don't tell"
        violated.evidence = "Character explains feelings directly"
        violated.suggestion = "Use visual metaphor instead"
        unclear = MagicMock()
        unclear.principle = MagicMock()
        unclear.principle.statement = "Subtext in every line"
        unclear.evidence = "Hard to determine from text alone"

        check_result = MagicMock()
        check_result.overall_assessment = "Good but needs work on showing"
        check_result.priority_fixes = ["Replace exposition with action"]
        check_result.rules_followed = [followed]
        check_result.rules_violated = [violated]
        check_result.rules_unclear = [unclear]
        mock_mentor.check_rules = AsyncMock(return_value=check_result)

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(mentor_check_rules(scene_text="INT. COURTROOM - DAY"))

        assert result["success"] is True
        assert len(result["rules_followed"]) == 1
        assert len(result["rules_violated"]) == 1
        assert (
            result["rules_violated"][0]["suggestion"] == "Use visual metaphor instead"
        )
        assert len(result["rules_unclear"]) == 1
        assert "showing" in result["overall_assessment"]

    def test_exception_handling(self):
        from mcp.documents.service import mentor_check_rules

        mock_mentor = MagicMock()
        mock_mentor.check_rules = AsyncMock(side_effect=RuntimeError("Error"))

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(mentor_check_rules(scene_text="test"))

        assert result["success"] is False


class TestMentorFindInspiration:
    """Tests for mentor_find_inspiration service function."""

    def test_success(self):
        from mcp.documents.service import mentor_find_inspiration

        mock_mentor = MagicMock()
        insp = MagicMock()
        insp.example = MagicMock()
        insp.example.work_title = "12 Angry Men"
        insp.example.scene_or_section = "Final vote"
        insp.example.description = "The last holdout changes vote"
        insp.relevance_reason = "Same power dynamics"
        insp.how_to_apply = "Mirror the tension buildup"
        insp.source_book = "Story"
        mock_mentor.find_inspiration = AsyncMock(return_value=[insp])

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(mentor_find_inspiration(situation="courtroom confrontation"))

        assert result["success"] is True
        assert result["count"] == 1
        assert result["inspirations"][0]["film"] == "12 Angry Men"

    def test_exception_handling(self):
        from mcp.documents.service import mentor_find_inspiration

        mock_mentor = MagicMock()
        mock_mentor.find_inspiration = AsyncMock(side_effect=RuntimeError("Error"))

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(mentor_find_inspiration(situation="test"))

        assert result["success"] is False


class TestMentorAsk:
    """Tests for mentor_ask service function."""

    def test_success(self):
        from mcp.documents.service import mentor_ask

        mock_mentor = MagicMock()
        mock_mentor.what_would_books_say = AsyncMock(
            return_value="McKee says that structure is character..."
        )

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(mentor_ask(question="What is character arc?"))

        assert result["success"] is True
        assert result["question"] == "What is character arc?"
        assert "McKee" in result["answer"]

    def test_exception_handling(self):
        from mcp.documents.service import mentor_ask

        mock_mentor = MagicMock()
        mock_mentor.what_would_books_say = AsyncMock(side_effect=RuntimeError("Error"))

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(mentor_ask(question="test"))

        assert result["success"] is False


class TestMentorCompare:
    """Tests for mentor_compare service function."""

    def test_success(self):
        from mcp.documents.service import mentor_compare

        mock_mentor = MagicMock()
        mock_mentor.compare_approaches = AsyncMock(
            return_value="McKee favors classical structure while Field..."
        )

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(mentor_compare(topic="three-act structure"))

        assert result["success"] is True
        assert result["topic"] == "three-act structure"
        assert "McKee" in result["comparison"]

    def test_exception_handling(self):
        from mcp.documents.service import mentor_compare

        mock_mentor = MagicMock()
        mock_mentor.compare_approaches = AsyncMock(side_effect=RuntimeError("Error"))

        with patch(
            "mcp.documents.service._get_mentor_engine", return_value=mock_mentor
        ):
            result = _run(mentor_compare(topic="test"))

        assert result["success"] is False


# ============================================================
# Knowledge Search Tests
# ============================================================


class TestKnowledgeSearch:
    """Tests for knowledge_search service function."""

    def test_success(self):
        from mcp.documents.service import knowledge_search

        mock_store = MagicMock()
        mock_store.search_knowledge.return_value = [
            {"type": "concept", "name": "Inciting Incident", "relevance": 0.95},
            {"type": "principle", "statement": "Every scene turns", "relevance": 0.87},
        ]

        with patch(
            "mcp.documents.service._get_understanding_store",
            return_value=mock_store,
        ):
            result = _run(knowledge_search(query="inciting incident"))

        assert result["success"] is True
        assert result["query"] == "inciting incident"
        assert result["count"] == 2

    def test_with_type_filter(self):
        from mcp.documents.service import knowledge_search

        mock_store = MagicMock()
        mock_store.search_knowledge.return_value = []

        with patch(
            "mcp.documents.service._get_understanding_store",
            return_value=mock_store,
        ):
            result = _run(
                knowledge_search(query="test", knowledge_type="technique", top_k=5)
            )

        mock_store.search_knowledge.assert_called_once_with("test", "technique", 5)
        assert result["success"] is True

    def test_exception_handling(self):
        from mcp.documents.service import knowledge_search

        mock_store = MagicMock()
        mock_store.search_knowledge.side_effect = RuntimeError("Search error")

        with patch(
            "mcp.documents.service._get_understanding_store",
            return_value=mock_store,
        ):
            result = _run(knowledge_search(query="test"))

        assert result["success"] is False
