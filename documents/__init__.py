"""
Friday Document Processing System

DeepSeek-OCR 2 powered document understanding for book ingestion,
search with citations, and knowledge graph integration.

Status:
    DONE: Module structure and interfaces
    DONE: Document/Page/Chunk data models
    DONE: SQLite document store schema
    DONE: Book Understanding Layer (comprehension + mentor)
    DONE: MCP tools for document ingestion, search, and mentor
    TODO: Test DeepSeek-OCR 2 integration with sample PDF
    TODO: Integrate with LTM for chunk embeddings
    TODO: Test Telugu document processing

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

    # Study a book (extract structured knowledge)
    from documents.understanding import BookComprehensionEngine, MentorEngine
    # ... see understanding module for full API
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
from documents.storage.cloud_sync import (
    CloudSyncManager,
    SyncConfig,
    StorageBackend,
    CloudFile,
    create_sync_manager,
)

# Retrieval
from documents.retrieval.searcher import DocumentSearcher
from documents.retrieval.citation import CitationTracker

# Understanding (Book Comprehension & Mentor)
from documents.understanding import (
    BookUnderstanding,
    BookComprehensionEngine,
    MentorEngine,
    BookGraphIntegrator,
    Concept,
    Principle,
    Technique,
    BookExample,
)

# Understanding Storage
from documents.storage.understanding_store import BookUnderstandingStore


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
    # Cloud Sync
    "CloudSyncManager",
    "SyncConfig",
    "StorageBackend",
    "CloudFile",
    "create_sync_manager",
    # Understanding (Book Comprehension & Mentor)
    "BookUnderstanding",
    "BookComprehensionEngine",
    "MentorEngine",
    "BookGraphIntegrator",
    "Concept",
    "Principle",
    "Technique",
    "BookExample",
    "BookUnderstandingStore",
]
