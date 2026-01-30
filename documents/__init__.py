"""
Friday Document Processing System

DeepSeek-OCR 2 powered document understanding for book ingestion,
search with citations, and knowledge graph integration.

Usage:
    from documents import DocumentManager, initialize_document_manager

    # Initialize
    manager = await initialize_document_manager()

    # Ingest a book
    document = await manager.ingest_document(
        file_path="/path/to/book.pdf",
        title="Story by Robert McKee",
        document_type=DocumentType.BOOK,
    )

    # Search with citations
    results = await manager.search("character arc principles")
    for result in results:
        print(f"{result.citation.format_inline()}: {result.highlight}")

    # Get context for LLM
    context, citations = await manager.get_context_for_query(
        "What does McKee say about three-act structure?"
    )
"""

__version__ = "1.0.0"

# Manager
from documents.manager import (
    DocumentManager,
    get_document_manager,
    initialize_document_manager,
)

# Configuration
from documents.config import (
    DocumentConfig,
    get_document_config,
)

# Models
from documents.models import (
    Document,
    DocumentType,
    DocumentLanguage,
    DocumentMetadata,
    Page,
    Chunk,
    Citation,
    DocumentSearchResult,
    ProcessingStatus,
    ProcessingResult,
    OCRResult,
)

# OCR
from documents.ocr.deepseek_engine import DeepSeekOCR

# Pipeline
from documents.pipeline.pdf_processor import PDFProcessor
from documents.pipeline.chunker import SemanticChunker

# Storage
from documents.storage.document_store import DocumentStore

# Retrieval
from documents.retrieval.searcher import DocumentSearcher
from documents.retrieval.citation import CitationTracker


__all__ = [
    # Version
    "__version__",
    # Manager
    "DocumentManager",
    "get_document_manager",
    "initialize_document_manager",
    # Config
    "DocumentConfig",
    "get_document_config",
    # Models
    "Document",
    "DocumentType",
    "DocumentLanguage",
    "DocumentMetadata",
    "Page",
    "Chunk",
    "Citation",
    "DocumentSearchResult",
    "ProcessingStatus",
    "ProcessingResult",
    "OCRResult",
    # Components
    "DeepSeekOCR",
    "PDFProcessor",
    "SemanticChunker",
    "DocumentStore",
    "DocumentSearcher",
    "CitationTracker",
]
