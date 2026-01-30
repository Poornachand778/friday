"""
MCP Document Processing Service for Friday AI
==============================================

Provides document operations for MCP servers.
Wraps the core DocumentManager for tool-based access.
"""

from __future__ import annotations

import asyncio
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
                "publisher": document.metadata.publisher,
                "year": document.metadata.year,
                "isbn": document.metadata.isbn,
                "type": document.document_type.value,
                "language": document.language.value,
                "total_pages": document.total_pages,
                "status": document.status.value,
                "project": document.project,
                "chapters": [
                    {
                        "title": ch.title,
                        "start_page": ch.start_page,
                        "end_page": ch.end_page,
                        "level": ch.level,
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
