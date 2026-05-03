"""
MCP Document Processing Service for Friday AI
==============================================

Provides document operations for MCP servers.
Wraps the core DocumentManager for tool-based access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_manager: Optional[Any] = None


async def get_document_manager():
    """Get or create the DocumentManager singleton."""
    global _manager
    if _manager is None:
        from documents import initialize_document_manager

        _manager = await initialize_document_manager()
    return _manager


async def document_ingest(
    file_path: str,
    title: str,
    author: Optional[str] = None,
    document_type: str = "book",
    language: str = "en",
    project: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingest a PDF document for conversational access.

    Args:
        file_path: Path to the PDF file
        title: Document title
        author: Author name (optional)
        document_type: Type of document (book, screenplay, article, manual, reference)
        language: Primary language (en, te, mixed)
        project: Link to Friday project (optional)

    Returns:
        Document info with ID and status
    """
    from documents import DocumentType, DocumentLanguage

    manager = await get_document_manager()

    # Convert string to enum
    doc_type = DocumentType(document_type)
    doc_lang = DocumentLanguage(language)

    # Verify file exists
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}", "success": False}

    try:
        document = await manager.ingest_document(
            file_path=file_path,
            title=title,
            author=author,
            document_type=doc_type,
            language=doc_lang,
            project=project,
        )

        return {
            "success": True,
            "document_id": document.id,
            "title": document.metadata.title,
            "total_pages": document.total_pages,
            "status": document.status.value,
            "chapters": len(document.chapters) if document.chapters else 0,
        }
    except Exception as e:
        LOGGER.exception("Document ingestion failed")
        return {"error": str(e), "success": False}


async def document_search(
    query: str,
    document_id: Optional[str] = None,
    document_type: Optional[str] = None,
    project: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Search across ingested documents with citations.

    Args:
        query: Natural language search query
        document_id: Limit to specific document (optional)
        document_type: Filter by document type (optional)
        project: Filter by project (optional)
        top_k: Number of results (default 5, max 20)

    Returns:
        Search results with citations
    """
    from documents import DocumentType

    manager = await get_document_manager()

    doc_type = DocumentType(document_type) if document_type else None

    try:
        results = await manager.search(
            query=query,
            document_id=document_id,
            document_type=doc_type,
            project=project,
            top_k=min(top_k, 20),
        )

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": [
                {
                    "citation": result.citation.format_inline(),
                    "document_title": result.citation.document_title,
                    "page_range": result.citation.page_range,
                    "chapter": result.citation.chapter,
                    "highlight": result.highlight,
                    "relevance": round(result.relevance, 3),
                }
                for result in results
            ],
        }
    except Exception as e:
        LOGGER.exception("Document search failed")
        return {"error": str(e), "success": False}


async def document_get_context(
    query: str,
    document_id: Optional[str] = None,
    max_chunks: int = 3,
    max_chars: int = 4000,
) -> Dict[str, Any]:
    """
    Get document context for LLM generation with citations.

    Args:
        query: The question or topic to find context for
        document_id: Limit to specific document (optional)
        max_chunks: Maximum number of chunks (default 3)
        max_chars: Maximum total characters (default 4000)

    Returns:
        Context text and citations for LLM use
    """
    manager = await get_document_manager()

    try:
        context, citations = await manager.get_context_for_query(
            query=query,
            document_id=document_id,
            max_chunks=max_chunks,
            max_chars=max_chars,
        )

        return {
            "success": True,
            "context": context,
            "citations": [
                {
                    "document_title": c.document_title,
                    "page_range": c.page_range,
                    "chapter": c.chapter,
                    "quote": c.quote[:200] + "..." if len(c.quote) > 200 else c.quote,
                    "formatted": c.format_inline(),
                }
                for c in citations
            ],
        }
    except Exception as e:
        LOGGER.exception("Get context failed")
        return {"error": str(e), "success": False}


async def document_list(
    document_type: Optional[str] = None,
    project: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List all ingested documents.

    Args:
        document_type: Filter by type (optional)
        project: Filter by project (optional)
        status: Filter by processing status (optional)

    Returns:
        List of documents with metadata
    """
    from documents import DocumentType, ProcessingStatus

    manager = await get_document_manager()

    doc_type = DocumentType(document_type) if document_type else None
    proc_status = ProcessingStatus(status) if status else None

    try:
        documents = await manager.list_documents(
            document_type=doc_type,
            project=project,
            status=proc_status,
        )

        return {
            "success": True,
            "count": len(documents),
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.metadata.title,
                    "author": doc.metadata.author,
                    "type": doc.document_type.value,
                    "language": doc.language.value,
                    "pages": doc.total_pages,
                    "chapters": len(doc.chapters) if doc.chapters else 0,
                    "status": doc.status.value,
                    "project": doc.project,
                    "created_at": doc.created_at.isoformat(),
                }
                for doc in documents
            ],
        }
    except Exception as e:
        LOGGER.exception("List documents failed")
        return {"error": str(e), "success": False}


async def document_get(document_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific document.

    Args:
        document_id: The document UUID

    Returns:
        Full document details including chapters
    """
    manager = await get_document_manager()

    try:
        document = await manager.get_document(document_id)
        if not document:
            return {"error": f"Document not found: {document_id}", "success": False}

        return {
            "success": True,
            "document": {
                "id": document.id,
                "file_path": document.file_path,
                "title": document.metadata.title,
                "author": document.metadata.author,
                "isbn": document.metadata.isbn,
                "publication_date": (
                    document.metadata.publication_date.isoformat()
                    if document.metadata.publication_date
                    else None
                ),
                "type": document.document_type.value,
                "language": document.language.value,
                "total_pages": document.total_pages,
                "status": document.status.value,
                "project": document.project,
                "chapters": [
                    {
                        "number": ch.number,
                        "title": ch.title,
                        "start_page": ch.start_page,
                        "end_page": ch.end_page,
                    }
                    for ch in (document.chapters or [])
                ],
                "created_at": document.created_at.isoformat(),
                "processed_at": (
                    document.processed_at.isoformat() if document.processed_at else None
                ),
            },
        }
    except Exception as e:
        LOGGER.exception("Get document failed")
        return {"error": str(e), "success": False}


async def document_get_chapter(
    document_id: str,
    chapter_title: Optional[str] = None,
    chapter_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Get the full text of a specific chapter.

    Args:
        document_id: The document UUID
        chapter_title: Chapter title to retrieve
        chapter_index: Chapter index (0-based) if title not provided

    Returns:
        Chapter text with page range
    """
    manager = await get_document_manager()

    try:
        chapter_text, page_range = await manager.get_chapter(
            document_id=document_id,
            chapter_title=chapter_title,
            chapter_index=chapter_index,
        )

        return {
            "success": True,
            "document_id": document_id,
            "chapter_title": chapter_title,
            "page_range": page_range,
            "text": chapter_text,
        }
    except ValueError as e:
        return {"error": str(e), "success": False}
    except Exception as e:
        LOGGER.exception("Get chapter failed")
        return {"error": str(e), "success": False}


async def document_status(document_id: str) -> Dict[str, Any]:
    """
    Check the processing status of a document.

    Args:
        document_id: The document UUID

    Returns:
        Processing status and progress info
    """
    manager = await get_document_manager()

    try:
        status_info = await manager.get_processing_status(document_id)

        return {
            "success": True,
            "document_id": document_id,
            "status": status_info.get("status", "unknown"),
            "progress": status_info.get("progress", 0),
            "current_page": status_info.get("current_page"),
            "total_pages": status_info.get("total_pages"),
            "error": status_info.get("error"),
        }
    except Exception as e:
        LOGGER.exception("Get status failed")
        return {"error": str(e), "success": False}


async def document_delete(document_id: str) -> Dict[str, Any]:
    """
    Delete a document and all its chunks.

    Args:
        document_id: The document UUID

    Returns:
        Deletion confirmation
    """
    manager = await get_document_manager()

    try:
        await manager.delete_document(document_id)
        return {"success": True, "document_id": document_id, "deleted": True}
    except ValueError as e:
        return {"error": str(e), "success": False}
    except Exception as e:
        LOGGER.exception("Delete document failed")
        return {"error": str(e), "success": False}


# =============================================================================
# Cloud Sync Functions (for server deployment)
# =============================================================================

_sync_manager: Optional[Any] = None


async def _get_sync_manager():
    """Get or create the CloudSyncManager singleton."""
    global _sync_manager
    if _sync_manager is None:
        import os
        from documents.storage.cloud_sync import create_sync_manager

        # Get config from environment
        backend = os.getenv("FRIDAY_STORAGE_BACKEND", "local")
        inbox_path = os.getenv("FRIDAY_INBOX_PATH", "documents/data/inbox")
        s3_bucket = os.getenv("FRIDAY_S3_BUCKET", "")
        s3_prefix = os.getenv("FRIDAY_S3_PREFIX", "friday-inbox/")

        manager = await get_document_manager()

        _sync_manager = create_sync_manager(
            backend=backend,
            inbox_path=inbox_path,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            document_manager=manager,
        )

    return _sync_manager


async def inbox_scan() -> Dict[str, Any]:
    """
    Scan the document inbox and process new files.

    For server deployment: scans S3 bucket or shared folder for new PDFs.
    For local deployment: scans local inbox folder.

    Returns:
        List of newly processed files
    """
    try:
        sync = await _get_sync_manager()
        new_files = await sync.sync_once()

        return {
            "success": True,
            "files_processed": len(new_files),
            "files": [
                {
                    "filename": f.filename,
                    "size": f.size,
                    "extension": f.extension,
                }
                for f in new_files
            ],
        }
    except Exception as e:
        LOGGER.exception("Inbox scan failed")
        return {"error": str(e), "success": False}


async def inbox_list() -> Dict[str, Any]:
    """
    List files in the inbox waiting to be processed.

    Returns:
        List of pending files
    """
    try:
        sync = await _get_sync_manager()
        pending = await sync.list_pending()

        return {
            "success": True,
            "count": len(pending),
            "files": [
                {
                    "filename": f.filename,
                    "size": f.size,
                    "extension": f.extension,
                    "last_modified": f.last_modified.isoformat(),
                }
                for f in pending
            ],
        }
    except Exception as e:
        LOGGER.exception("List inbox failed")
        return {"error": str(e), "success": False}


async def inbox_status() -> Dict[str, Any]:
    """
    Get the status of the inbox sync system.

    Returns:
        Sync configuration and status
    """
    try:
        sync = await _get_sync_manager()
        status = sync.get_status()

        return {
            "success": True,
            **status,
        }
    except Exception as e:
        LOGGER.exception("Get inbox status failed")
        return {"error": str(e), "success": False}


async def inbox_start_watch() -> Dict[str, Any]:
    """
    Start automatic inbox watching (background polling).

    Returns:
        Status of watch start
    """
    try:
        sync = await _get_sync_manager()
        await sync.start()

        return {
            "success": True,
            "watching": True,
            "poll_interval": sync.config.poll_interval_seconds,
            "backend": sync.config.backend.value,
        }
    except Exception as e:
        LOGGER.exception("Start inbox watch failed")
        return {"error": str(e), "success": False}


async def inbox_stop_watch() -> Dict[str, Any]:
    """
    Stop automatic inbox watching.

    Returns:
        Status of watch stop
    """
    try:
        sync = await _get_sync_manager()
        await sync.stop()

        return {
            "success": True,
            "watching": False,
        }
    except Exception as e:
        LOGGER.exception("Stop inbox watch failed")
        return {"error": str(e), "success": False}


# =============================================================================
# Book Understanding & Mentor Functions
# =============================================================================

_comprehension_engine: Optional[Any] = None
_mentor_engine: Optional[Any] = None
_understanding_store: Optional[Any] = None


async def _get_understanding_store():
    """Get or create the BookUnderstandingStore singleton."""
    global _understanding_store
    if _understanding_store is None:
        from documents.storage.understanding_store import BookUnderstandingStore

        _understanding_store = BookUnderstandingStore()
        _understanding_store.initialize()
    return _understanding_store


async def _get_comprehension_engine():
    """Get or create the BookComprehensionEngine singleton."""
    global _comprehension_engine
    if _comprehension_engine is None:
        from documents.understanding.comprehension import BookComprehensionEngine

        # Create LLM completion function
        # This will use the orchestrator's configured model
        async def llm_complete(prompt: str) -> str:
            # Use a simple completion for now
            # In production, this would call the orchestrator's LLM
            try:
                from orchestrator.core import get_orchestrator

                orch = get_orchestrator()
                if orch and hasattr(orch, "complete"):
                    return await orch.complete(prompt)
            except ImportError:
                pass

            # Fallback: return empty (will be filled in production)
            LOGGER.warning("No LLM available for comprehension - using placeholder")
            return "{}"

        _comprehension_engine = BookComprehensionEngine(llm_complete)
    return _comprehension_engine


async def _get_mentor_engine():
    """Get or create the MentorEngine singleton."""
    global _mentor_engine
    if _mentor_engine is None:
        from documents.understanding.mentor import MentorEngine

        store = await _get_understanding_store()

        # Create LLM completion function
        async def llm_complete(prompt: str) -> str:
            try:
                from orchestrator.core import get_orchestrator

                orch = get_orchestrator()
                if orch and hasattr(orch, "complete"):
                    return await orch.complete(prompt)
            except ImportError:
                pass

            LOGGER.warning("No LLM available for mentor - using placeholder")
            return "{}"

        _mentor_engine = MentorEngine(llm_complete, store)
    return _mentor_engine


async def book_study(
    document_id: str,
    link_to_project: Optional[str] = None,
    thorough_mode: Optional[bool] = None,
    voice_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Study a document and extract structured knowledge.

    This creates a BookUnderstanding with:
    - Summary and thesis
    - Key concepts with definitions
    - Principles (rules/guidelines)
    - Techniques (practical methods)
    - Examples (film references/case studies)

    Args:
        document_id: The document UUID (must be ingested first)
        link_to_project: Optional project to link knowledge to
        thorough_mode: Override config for thorough processing (None = use config default)
        voice_enabled: Whether to enable voice progress announcements

    Returns:
        Book understanding summary with counts
    """
    manager = await get_document_manager()
    engine = await _get_comprehension_engine()
    store = await _get_understanding_store()

    try:
        # Check if already studied
        existing = store.get_understanding_by_document(document_id)
        if existing:
            return {
                "success": True,
                "already_studied": True,
                "understanding_id": existing.id,
                "title": existing.title,
                "concepts": len(existing.concepts),
                "principles": len(existing.principles),
                "techniques": len(existing.techniques),
                "examples": len(existing.examples),
                "quality": existing.comprehension_quality,
            }

        # Get document and chunks
        document = await manager.get_document(document_id)
        if not document:
            return {"error": f"Document not found: {document_id}", "success": False}

        chunks = await manager.get_chunks(document_id)
        if not chunks:
            return {
                "error": "Document has no chunks - ensure it's processed",
                "success": False,
            }

        # Override thorough_mode if specified
        if thorough_mode is not None:
            engine._config.thorough_mode = thorough_mode

        # Progress callback for logging
        def progress_callback(stage: str, progress: float):
            LOGGER.info("Book study: %s (%.0f%%)", stage, progress * 100)

        # Voice callback for announcements
        voice_messages: List[str] = []

        def voice_callback(message: str):
            voice_messages.append(message)
            LOGGER.info("Voice: %s", message)
            # In production, this would call the voice daemon to speak
            # For now, we collect messages to return

        # Comprehend the document
        understanding = await engine.comprehend(
            document,
            chunks,
            progress_callback,
            voice_callback if voice_enabled else None,
        )

        # Store the understanding
        store.store_understanding(understanding)

        # Integrate with Knowledge Graph (if available)
        try:
            from memory import get_memory_manager
            from documents.understanding.graph_integration import BookGraphIntegrator

            memory = get_memory_manager()
            if memory and hasattr(memory, "knowledge_graph"):
                integrator = BookGraphIntegrator(memory.knowledge_graph)
                await integrator.integrate_book(understanding, link_to_project)
        except ImportError:
            LOGGER.info(
                "Knowledge Graph integration skipped - memory module not available"
            )

        return {
            "success": True,
            "job_id": engine._current_job_id,  # For status tracking
            "understanding_id": understanding.id,
            "title": understanding.title,
            "author": understanding.author,
            "summary": (
                understanding.summary[:500] + "..."
                if len(understanding.summary) > 500
                else understanding.summary
            ),
            "main_argument": understanding.main_argument,
            "domains": understanding.domains,
            "concepts": len(understanding.concepts),
            "principles": len(understanding.principles),
            "techniques": len(understanding.techniques),
            "examples": len(understanding.examples),
            "quality": understanding.comprehension_quality,
            "mode": "thorough" if engine._config.thorough_mode else "sampling",
            "voice_messages": voice_messages if voice_enabled else [],
        }
    except Exception as e:
        LOGGER.exception("Book study failed")
        # Get job ID if available for error tracking
        try:
            engine = await _get_comprehension_engine()
            job_id = engine._current_job_id
        except:
            job_id = None
        return {"error": str(e), "success": False, "job_id": job_id}


async def book_study_status(
    job_id: Optional[str] = None,
    document_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get the live status of a book study operation.

    Voice-friendly: Returns messages like "Boss, I'm on Chapter 5 of 15 -
    'The Inciting Incident'. About 8 minutes remaining."

    Can query by job_id, document_id, or get status of all active jobs.

    Args:
        job_id: Specific job ID to check (returned from book_study)
        document_id: Get status for a specific document

    Returns:
        Current study progress with voice-friendly status message
    """
    from documents.understanding.job_tracker import get_job_tracker

    try:
        tracker = get_job_tracker()

        # Query specific job
        if job_id:
            status = tracker.get_status(job_id)
            if status:
                return {
                    "success": True,
                    "voice_status": status["voice_status"],
                    **status,
                }
            return {"error": f"Job not found: {job_id}", "success": False}

        # Query by document
        if document_id:
            job = tracker.get_job_by_document(document_id)
            if job:
                return {
                    "success": True,
                    "voice_status": job.get_voice_status(),
                    **job.to_dict(),
                }
            return {
                "error": f"No active job for document: {document_id}",
                "success": False,
            }

        # Return status of all active jobs
        active_jobs = tracker.get_active_jobs()
        if not active_jobs:
            return {
                "success": True,
                "voice_status": "Boss, no books are currently being studied.",
                "active_jobs": [],
            }

        return {
            "success": True,
            "voice_status": tracker.get_active_status_summary(),
            "active_jobs": [job.to_dict() for job in active_jobs],
            "count": len(active_jobs),
        }
    except Exception as e:
        LOGGER.exception("Book study status check failed")
        return {"error": str(e), "success": False}


async def book_study_jobs() -> Dict[str, Any]:
    """
    List all study jobs (active and recent completed).

    Returns:
        List of all study jobs with their status
    """
    from documents.understanding.job_tracker import get_job_tracker

    try:
        tracker = get_job_tracker()
        jobs = tracker.get_all_jobs()

        return {
            "success": True,
            "count": len(jobs),
            "jobs": jobs,
        }
    except Exception as e:
        LOGGER.exception("List study jobs failed")
        return {"error": str(e), "success": False}


async def book_list_studied() -> Dict[str, Any]:
    """
    List all studied books.

    Returns:
        List of book understandings with knowledge counts
    """
    store = await _get_understanding_store()

    try:
        understandings = store.list_understandings()

        return {
            "success": True,
            "count": len(understandings),
            "books": [
                {
                    "id": u.id,
                    "document_id": u.document_id,
                    "title": u.title,
                    "author": u.author,
                    "domains": u.domains,
                    "concepts": len(u.concepts) if u.concepts else 0,
                    "principles": len(u.principles) if u.principles else 0,
                    "techniques": len(u.techniques) if u.techniques else 0,
                    "examples": len(u.examples) if u.examples else 0,
                    "quality": u.comprehension_quality,
                    "studied_at": (
                        u.study_completed_at.isoformat()
                        if u.study_completed_at
                        else None
                    ),
                }
                for u in understandings
            ],
        }
    except Exception as e:
        LOGGER.exception("List studied books failed")
        return {"error": str(e), "success": False}


async def book_get_understanding(understanding_id: str) -> Dict[str, Any]:
    """
    Get detailed understanding of a specific book.

    Args:
        understanding_id: The understanding UUID

    Returns:
        Full book understanding with all extracted knowledge
    """
    store = await _get_understanding_store()

    try:
        understanding = store.get_understanding(understanding_id)
        if not understanding:
            return {
                "error": f"Understanding not found: {understanding_id}",
                "success": False,
            }

        return {
            "success": True,
            "understanding": understanding.to_dict(),
        }
    except Exception as e:
        LOGGER.exception("Get understanding failed")
        return {"error": str(e), "success": False}


async def mentor_load_books(
    understanding_ids: List[str],
) -> Dict[str, Any]:
    """
    Load books for a mentoring session.

    Args:
        understanding_ids: List of understanding UUIDs to load

    Returns:
        Confirmation of loaded books
    """
    store = await _get_understanding_store()
    mentor = await _get_mentor_engine()

    try:
        books = []
        for uid in understanding_ids:
            understanding = store.get_understanding(uid)
            if understanding:
                books.append(understanding)
            else:
                LOGGER.warning("Understanding not found: %s", uid)

        mentor.load_books(books)

        return {
            "success": True,
            "loaded_books": [
                {"id": b.id, "title": b.title, "author": b.author} for b in books
            ],
        }
    except Exception as e:
        LOGGER.exception("Load books failed")
        return {"error": str(e), "success": False}


async def mentor_analyze(
    scene_description: str,
    project_context: str = "",
) -> Dict[str, Any]:
    """
    Analyze a scene against loaded book knowledge.

    The mentor will:
    - Identify what's working
    - Suggest what might be missing
    - Reference relevant principles and techniques
    - Provide examples from the books

    Args:
        scene_description: Description of the scene to analyze
        project_context: Context about the project (optional)

    Returns:
        Detailed mentor analysis
    """
    mentor = await _get_mentor_engine()

    try:
        analysis = await mentor.analyze_scene(scene_description, project_context)

        return {
            "success": True,
            "response": analysis.to_response(),
            "analysis": {
                "elements_present": analysis.elements_present,
                "elements_missing": analysis.elements_missing,
                "strengths": analysis.strengths,
                "suggestions": analysis.suggestions,
                "questions": analysis.questions_to_consider,
                "principles_count": len(analysis.relevant_principles),
                "techniques_count": len(analysis.applicable_techniques),
                "examples_count": len(analysis.similar_examples),
            },
        }
    except Exception as e:
        LOGGER.exception("Mentor analyze failed")
        return {"error": str(e), "success": False}


async def mentor_brainstorm(
    topic: str,
    constraints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Brainstorm ideas using book knowledge.

    Generates creative ideas grounded in principles and
    techniques from the loaded books.

    Args:
        topic: What to brainstorm about
        constraints: Optional list of requirements/limitations

    Returns:
        Brainstorm results with grounded ideas
    """
    mentor = await _get_mentor_engine()

    try:
        result = await mentor.brainstorm(topic, constraints)

        return {
            "success": True,
            "response": result.to_response(),
            "brainstorm": {
                "topic": result.topic,
                "constraints": result.constraints,
                "ideas": [
                    {
                        "idea": idea.idea,
                        "rationale": idea.rationale,
                        "based_on": idea.based_on,
                        "inspiration": idea.source_inspiration,
                    }
                    for idea in result.ideas
                ],
                "suggested_structure": result.suggested_structure,
                "concepts_applied": result.concepts_applied,
                "techniques_suggested": result.techniques_suggested,
            },
        }
    except Exception as e:
        LOGGER.exception("Mentor brainstorm failed")
        return {"error": str(e), "success": False}


async def mentor_check_rules(
    scene_text: str,
) -> Dict[str, Any]:
    """
    Check a scene against principles from loaded books.

    Identifies which rules are being followed or violated.

    Args:
        scene_text: The scene text to check

    Returns:
        Rule check results showing what's followed vs violated
    """
    mentor = await _get_mentor_engine()

    try:
        result = await mentor.check_rules(scene_text)

        return {
            "success": True,
            "overall_assessment": result.overall_assessment,
            "priority_fixes": result.priority_fixes,
            "rules_followed": [
                {
                    "principle": r.principle.statement,
                    "evidence": r.evidence,
                }
                for r in result.rules_followed
            ],
            "rules_violated": [
                {
                    "principle": r.principle.statement,
                    "evidence": r.evidence,
                    "suggestion": r.suggestion,
                }
                for r in result.rules_violated
            ],
            "rules_unclear": [
                {
                    "principle": r.principle.statement,
                    "question": r.evidence,
                }
                for r in result.rules_unclear
            ],
        }
    except Exception as e:
        LOGGER.exception("Mentor check rules failed")
        return {"error": str(e), "success": False}


async def mentor_find_inspiration(
    situation: str,
) -> Dict[str, Any]:
    """
    Find inspiring examples from books for a situation.

    Args:
        situation: What you're trying to write

    Returns:
        Relevant examples from the books with adaptation suggestions
    """
    mentor = await _get_mentor_engine()

    try:
        inspirations = await mentor.find_inspiration(situation)

        return {
            "success": True,
            "count": len(inspirations),
            "inspirations": [
                {
                    "film": insp.example.work_title,
                    "scene": insp.example.scene_or_section,
                    "description": insp.example.description,
                    "why_relevant": insp.relevance_reason,
                    "how_to_adapt": insp.how_to_apply,
                    "source_book": insp.source_book,
                }
                for insp in inspirations
            ],
        }
    except Exception as e:
        LOGGER.exception("Find inspiration failed")
        return {"error": str(e), "success": False}


async def mentor_ask(
    question: str,
) -> Dict[str, Any]:
    """
    Ask a question and get an answer based on book knowledge.

    Args:
        question: Your question

    Returns:
        Answer grounded in book knowledge
    """
    mentor = await _get_mentor_engine()

    try:
        answer = await mentor.what_would_books_say(question)

        return {
            "success": True,
            "question": question,
            "answer": answer,
        }
    except Exception as e:
        LOGGER.exception("Mentor ask failed")
        return {"error": str(e), "success": False}


async def mentor_compare(
    topic: str,
) -> Dict[str, Any]:
    """
    Compare what different books say about a topic.

    Requires at least 2 books loaded.

    Args:
        topic: Topic to compare views on

    Returns:
        Comparison of different book perspectives
    """
    mentor = await _get_mentor_engine()

    try:
        comparison = await mentor.compare_approaches(topic)

        return {
            "success": True,
            "topic": topic,
            "comparison": comparison,
        }
    except Exception as e:
        LOGGER.exception("Mentor compare failed")
        return {"error": str(e), "success": False}


async def knowledge_search(
    query: str,
    knowledge_type: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Search across all book knowledge.

    Args:
        query: Search query
        knowledge_type: Filter by type (concept, principle, technique, example)
        top_k: Maximum results

    Returns:
        Matching knowledge items
    """
    store = await _get_understanding_store()

    try:
        results = store.search_knowledge(query, knowledge_type, top_k)

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
        }
    except Exception as e:
        LOGGER.exception("Knowledge search failed")
        return {"error": str(e), "success": False}
