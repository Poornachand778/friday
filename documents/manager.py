"""
Document Manager

Central coordinator for document processing, storage, and retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from documents.config import DocumentConfig, get_document_config
from documents.models import (
    Citation,
    Chunk,
    Document,
    DocumentLanguage,
    DocumentMetadata,
    DocumentSearchResult,
    DocumentType,
    Page,
    ProcessingResult,
    ProcessingStatus,
)
from documents.ocr.deepseek_engine import DeepSeekOCR
from documents.pipeline.chunker import SemanticChunker
from documents.pipeline.pdf_processor import PDFProcessor
from documents.retrieval.citation import CitationTracker
from documents.retrieval.searcher import DocumentSearcher
from documents.storage.document_store import DocumentStore

LOGGER = logging.getLogger(__name__)


class DocumentManager:
    """
    Central coordinator for document processing.

    Handles:
    - Document ingestion (PDF → OCR → Chunks → Storage)
    - Search and retrieval with citations
    - Integration with Memory System
    - Cross-document analysis
    """

    def __init__(self, config: Optional[DocumentConfig] = None):
        self.config = config or get_document_config()

        # Components (lazy initialized)
        self._store: Optional[DocumentStore] = None
        self._ocr: Optional[DeepSeekOCR] = None
        self._pdf_processor: Optional[PDFProcessor] = None
        self._chunker: Optional[SemanticChunker] = None
        self._searcher: Optional[DocumentSearcher] = None
        self._citation_tracker: Optional[CitationTracker] = None
        self._embedding_model = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all components"""
        if self._initialized:
            return

        LOGGER.info("Initializing DocumentManager...")

        # Initialize storage
        self._store = DocumentStore(self.config.storage)
        self._store.initialize()

        # Initialize processors
        self._pdf_processor = PDFProcessor(self.config.storage)
        self._chunker = SemanticChunker(self.config.chunking)

        # Initialize OCR (don't load model yet - lazy load on first use)
        self._ocr = DeepSeekOCR(self.config.ocr)

        # Initialize searcher
        self._searcher = DocumentSearcher(self._store, self.config.retrieval)
        await self._searcher.initialize()

        # Initialize citation tracker
        self._citation_tracker = CitationTracker(self.config.retrieval.citation_style)

        # Load embedding model if needed
        if self.config.integration.store_chunks_in_ltm:
            await self._load_embedding_model()

        self._initialized = True
        LOGGER.info("DocumentManager initialized")

    async def shutdown(self) -> None:
        """Cleanup resources"""
        if self._ocr and self._ocr.is_loaded:
            await self._ocr.unload_model()

        if self._store:
            self._store.close()

        self._initialized = False
        LOGGER.info("DocumentManager shut down")

    async def _load_embedding_model(self) -> None:
        """Load embedding model for chunk embeddings"""
        try:
            from sentence_transformers import SentenceTransformer

            model_name = self.config.embedding.model_name
            self._embedding_model = SentenceTransformer(model_name)
            LOGGER.info("Loaded embedding model: %s", model_name)
        except ImportError:
            LOGGER.warning(
                "sentence-transformers not installed. "
                "Chunks will not have embeddings."
            )

    # ==================== Ingestion ====================

    async def ingest_document(
        self,
        file_path: str,
        title: str,
        author: Optional[str] = None,
        document_type: DocumentType = DocumentType.BOOK,
        language: DocumentLanguage = DocumentLanguage.ENGLISH,
        project: Optional[str] = None,
        process_immediately: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Document:
        """
        Ingest a new document.

        Args:
            file_path: Path to PDF file
            title: Document title
            author: Author name
            document_type: Type of document
            language: Primary language
            project: Associate with Friday project
            process_immediately: Start processing now
            progress_callback: Callback for progress updates (pages_done, total_pages)

        Returns:
            Document object (may be pending processing)
        """
        if not self._initialized:
            await self.initialize()

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get PDF info
        page_count, file_hash, file_size = self._pdf_processor.get_pdf_info(file_path)

        # Check for duplicates
        existing = self._store.get_document_by_hash(file_hash)
        if existing:
            LOGGER.info("Document already exists: %s", existing.id)
            return existing

        # Create document record
        metadata = DocumentMetadata(title=title, author=author)
        document = Document.create(
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            metadata=metadata,
            total_pages=page_count,
            document_type=document_type,
            language=language,
            project=project,
        )

        # Store document
        self._store.store_document(document)
        LOGGER.info(
            "Created document %s: %s (%d pages)",
            document.id,
            title,
            page_count,
        )

        # Process if requested
        if process_immediately:
            result = await self.process_document(
                document.id,
                progress_callback=progress_callback,
            )
            if result.status == ProcessingStatus.COMPLETED:
                document.status = ProcessingStatus.COMPLETED
            else:
                document.status = result.status

        return document

    async def process_document(
        self,
        document_id: str,
        page_range: Optional[Tuple[int, int]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ProcessingResult:
        """
        Process a pending document.

        Runs: PDF → Images → OCR → Chunks → Embeddings → Storage

        Args:
            document_id: Document ID
            page_range: Optional (start, end) pages to process
            progress_callback: Progress callback (pages_done, total_pages)

        Returns:
            ProcessingResult with status and counts
        """
        import time

        start_time = time.time()

        document = self._store.get_document(document_id)
        if not document:
            return ProcessingResult(
                document_id=document_id,
                status=ProcessingStatus.FAILED,
                pages_processed=0,
                chunks_created=0,
                errors=["Document not found"],
            )

        # Update status
        self._store.update_document_status(document_id, ProcessingStatus.PROCESSING)

        errors: List[str] = []
        pages_processed = 0
        chunks_created = 0

        try:
            # Determine page range
            start_page = page_range[0] if page_range else 1
            end_page = page_range[1] if page_range else document.total_pages

            # Convert PDF to images
            LOGGER.info(
                "Converting pages %d-%d of %s to images",
                start_page,
                end_page,
                document_id,
            )
            image_paths = self._pdf_processor.convert_to_images(
                pdf_path=document.file_path,
                document_id=document_id,
                dpi=self.config.ocr.image_dpi,
                start_page=start_page,
                end_page=end_page,
            )

            # Process pages through OCR
            pages: List[Page] = []
            batch_size = self.config.ocr.max_batch_size

            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i : i + batch_size]
                LOGGER.debug(
                    "Processing OCR batch %d-%d",
                    i + 1,
                    min(i + batch_size, len(image_paths)),
                )

                # Language hint for OCR
                lang_hint = None
                if document.language == DocumentLanguage.TELUGU:
                    lang_hint = "te"
                elif document.language == DocumentLanguage.MIXED:
                    lang_hint = "mixed"

                # Run OCR
                results = await self._ocr.process_batch(batch, lang_hint)

                # Create Page objects
                for j, (path, result) in enumerate(zip(batch, results)):
                    page_num = start_page + i + j
                    page = Page.create(
                        document_id=document_id,
                        page_number=page_num,
                        raw_text=result.text,
                        cleaned_text=self._clean_ocr_text(result.text),
                        ocr_confidence=result.confidence,
                        ocr_model=result.model_used,
                    )
                    page.has_images = result.has_images
                    page.has_tables = result.has_tables
                    page.detected_headers = result.detected_headers
                    pages.append(page)
                    pages_processed += 1

                # Progress callback
                if progress_callback:
                    progress_callback(pages_processed, len(image_paths))

            # Store pages
            self._store.store_pages(pages)

            # Unload OCR model to free GPU for embeddings
            if not self.config.embedding.share_gpu_with_ocr:
                await self._ocr.unload_model()

            # Chunk pages
            LOGGER.info("Chunking %d pages", len(pages))
            chunks = self._chunker.chunk_document(
                pages=pages,
                document_id=document_id,
                chapters=document.chapters if document.chapters else None,
            )

            # Generate embeddings
            if self._embedding_model:
                LOGGER.info("Generating embeddings for %d chunks", len(chunks))
                await self._generate_embeddings(chunks)

            # Store chunks
            self._store.store_chunks(chunks)
            chunks_created = len(chunks)

            # Cleanup images if configured
            if self.config.storage.auto_cleanup_images:
                self._pdf_processor.cleanup_images(document_id)

            # Update document status
            self._store.update_document_status(
                document_id,
                ProcessingStatus.COMPLETED,
                pages_processed,
            )

            processing_time = time.time() - start_time
            LOGGER.info(
                "Processed document %s: %d pages, %d chunks in %.1fs",
                document_id,
                pages_processed,
                chunks_created,
                processing_time,
            )

            return ProcessingResult(
                document_id=document_id,
                status=ProcessingStatus.COMPLETED,
                pages_processed=pages_processed,
                chunks_created=chunks_created,
                errors=errors,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            LOGGER.error("Document processing failed: %s", e)
            errors.append(str(e))
            self._store.update_document_status(document_id, ProcessingStatus.FAILED)

            return ProcessingResult(
                document_id=document_id,
                status=ProcessingStatus.FAILED,
                pages_processed=pages_processed,
                chunks_created=chunks_created,
                errors=errors,
            )

    async def _generate_embeddings(self, chunks: List[Chunk]) -> None:
        """Generate embeddings for chunks in batches"""
        if not self._embedding_model:
            return

        batch_size = self.config.embedding.batch_size
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]

            embeddings = self._embedding_model.encode(
                texts,
                normalize_embeddings=self.config.embedding.normalize,
                show_progress_bar=False,
            )

            for chunk, embedding in zip(batch, embeddings):
                chunk.embedding = embedding.tolist()

    def _clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR text"""
        # Remove excessive whitespace
        import re

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Remove common OCR artifacts
        text = text.replace("", "")  # Zero-width characters
        text = text.replace("­", "")  # Soft hyphens

        return text.strip()

    # ==================== Search & Retrieval ====================

    async def search(
        self,
        query: str,
        document_id: Optional[str] = None,
        document_type: Optional[DocumentType] = None,
        project: Optional[str] = None,
        top_k: int = 10,
    ) -> List[DocumentSearchResult]:
        """
        Search across documents.

        Args:
            query: Natural language query
            document_id: Limit to specific document
            document_type: Filter by type
            project: Filter by project
            top_k: Number of results

        Returns:
            List of search results with citations
        """
        if not self._initialized:
            await self.initialize()

        return await self._searcher.search(
            query=query,
            document_id=document_id,
            document_type=document_type,
            project=project,
            top_k=top_k,
        )

    async def get_context_for_query(
        self,
        query: str,
        max_chunks: int = 3,
        max_chars: int = 4000,
        document_id: Optional[str] = None,
    ) -> Tuple[str, List[Citation]]:
        """
        Get relevant document context for LLM generation.

        Used by orchestrator to inject document knowledge.

        Args:
            query: Query to find relevant context for
            max_chunks: Maximum chunks to include
            max_chars: Maximum characters in context
            document_id: Optional document to limit search

        Returns:
            Tuple of (context_string, citations)
        """
        if not self._initialized:
            await self.initialize()

        # Search for relevant chunks
        results = await self.search(
            query=query,
            document_id=document_id,
            top_k=max_chunks * 2,  # Get extra for filtering
        )

        if not results:
            return "", []

        # Build context with citations
        return self._citation_tracker.get_context_with_citations(
            results[:max_chunks],
            max_chars=max_chars,
        )

    # ==================== Document Access ====================

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        if not self._initialized:
            await self.initialize()
        return self._store.get_document(document_id)

    async def get_page(self, document_id: str, page_number: int) -> Optional[Page]:
        """Get specific page"""
        if not self._initialized:
            await self.initialize()
        return self._store.get_page(document_id, page_number)

    async def get_chapter(
        self,
        document_id: str,
        chapter_title: Optional[str] = None,
        chapter_index: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Get full chapter text.

        Args:
            document_id: Document ID
            chapter_title: Title of chapter to retrieve
            chapter_index: Index of chapter (0-based) if title not provided

        Returns:
            Tuple of (chapter_text, page_range)

        Raises:
            ValueError: If chapter not found or no identifier provided
        """
        if not self._initialized:
            await self.initialize()

        document = self._store.get_document(document_id)
        if not document:
            raise ValueError(f"Document not found: {document_id}")

        if not document.chapters:
            raise ValueError("Document has no chapter information")

        # Find the chapter
        chapter_info = None
        if chapter_title:
            for ch in document.chapters:
                if ch.title.lower() == chapter_title.lower():
                    chapter_info = ch
                    break
            if not chapter_info:
                raise ValueError(f"Chapter not found: {chapter_title}")
        elif chapter_index is not None:
            if chapter_index < 0 or chapter_index >= len(document.chapters):
                raise ValueError(
                    f"Chapter index {chapter_index} out of range "
                    f"(0-{len(document.chapters) - 1})"
                )
            chapter_info = document.chapters[chapter_index]
        else:
            raise ValueError("Must provide chapter_title or chapter_index")

        # Get chunks for this chapter
        chunks = self._store.get_chunks_for_document(
            document_id, chapter=chapter_info.title
        )
        if not chunks:
            # Fallback: get pages in chapter range
            pages = []
            for page_num in range(chapter_info.start_page, chapter_info.end_page + 1):
                page = self._store.get_page(document_id, page_num)
                if page:
                    pages.append(page.cleaned_text or page.raw_text)
            return (
                "\n\n".join(pages),
                f"pp. {chapter_info.start_page}-{chapter_info.end_page}",
            )

        # Combine chunks in order
        chunks.sort(key=lambda c: c.chunk_index)
        page_range = f"pp. {chapter_info.start_page}-{chapter_info.end_page}"
        return "\n\n".join(c.content for c in chunks), page_range

    async def list_documents(
        self,
        document_type: Optional[DocumentType] = None,
        project: Optional[str] = None,
        status: Optional[ProcessingStatus] = None,
        limit: int = 100,
    ) -> List[Document]:
        """List documents with optional filters"""
        if not self._initialized:
            await self.initialize()
        return self._store.list_documents(
            document_type=document_type,
            project=project,
            status=status,
            limit=limit,
        )

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all related data"""
        if not self._initialized:
            await self.initialize()

        # Cleanup images
        self._pdf_processor.cleanup_images(document_id)

        # Delete from database
        return self._store.delete_document(document_id)

    async def get_processing_status(self, document_id: str) -> dict:
        """
        Get the processing status of a document.

        Args:
            document_id: Document ID

        Returns:
            Dict with status, progress, and error info
        """
        if not self._initialized:
            await self.initialize()

        document = self._store.get_document(document_id)
        if not document:
            return {"status": "not_found", "error": "Document not found"}

        # Get processed page count
        pages = self._store.get_pages_for_document(document_id)
        pages_processed = len(pages) if pages else 0

        result = {
            "status": document.status.value,
            "total_pages": document.total_pages,
            "current_page": pages_processed,
            "progress": (
                round(pages_processed / document.total_pages * 100, 1)
                if document.total_pages > 0
                else 0
            ),
        }

        if document.status == ProcessingStatus.FAILED:
            result["error"] = "Processing failed. Check logs for details."

        return result

    # ==================== Statistics ====================

    def get_stats(self) -> dict:
        """Get document system statistics"""
        if not self._store:
            return {}
        return self._store.get_stats()


# Singleton pattern
_manager: Optional[DocumentManager] = None


def get_document_manager() -> DocumentManager:
    """Get document manager singleton"""
    global _manager
    if _manager is None:
        _manager = DocumentManager()
    return _manager


async def initialize_document_manager() -> DocumentManager:
    """Initialize and return document manager"""
    manager = get_document_manager()
    await manager.initialize()
    return manager
