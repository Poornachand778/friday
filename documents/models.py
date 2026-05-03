"""
Document Processing Data Models

Defines core data structures for document ingestion, storage, and retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class DocumentType(str, Enum):
    """Types of documents Friday can process"""

    BOOK = "book"
    SCREENPLAY = "screenplay"
    ARTICLE = "article"
    MANUAL = "manual"
    REFERENCE = "reference"
    CUSTOM = "custom"


class DocumentLanguage(str, Enum):
    """Supported document languages"""

    ENGLISH = "en"
    TELUGU = "te"
    MIXED = "mixed"


class ProcessingStatus(str, Enum):
    """Document processing status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class DocumentMetadata:
    """Document metadata (title, author, etc.)"""

    title: str
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    isbn: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "publication_date": (
                self.publication_date.isoformat() if self.publication_date else None
            ),
            "isbn": self.isbn,
            "description": self.description,
            "tags": self.tags,
            "custom_fields": self.custom_fields,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        pub_date = data.get("publication_date")
        if pub_date and isinstance(pub_date, str):
            pub_date = datetime.fromisoformat(pub_date)
        return cls(
            title=data.get("title", "Untitled"),
            author=data.get("author"),
            publication_date=pub_date,
            isbn=data.get("isbn"),
            description=data.get("description"),
            tags=data.get("tags", []),
            custom_fields=data.get("custom_fields", {}),
        )


@dataclass
class ChapterInfo:
    """Chapter/Section metadata detected in document"""

    number: int
    title: str
    start_page: int
    end_page: int
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChapterInfo":
        return cls(
            number=data["number"],
            title=data["title"],
            start_page=data["start_page"],
            end_page=data["end_page"],
            summary=data.get("summary"),
        )


@dataclass
class Document:
    """
    Represents a complete document (book, PDF, screenplay).

    Links to:
        - Pages (for page-level content)
        - Chunks (for semantic retrieval)
        - LTM entries (for memory integration)
    """

    id: str
    file_path: str
    file_hash: str  # SHA256 for deduplication
    file_size: int
    document_type: DocumentType
    metadata: DocumentMetadata
    language: DocumentLanguage
    total_pages: int
    chapters: List[ChapterInfo] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.PENDING
    processed_pages: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    project: Optional[str] = None  # Link to Friday project
    access_count: int = 0

    @classmethod
    def create(
        cls,
        file_path: str,
        file_hash: str,
        file_size: int,
        metadata: DocumentMetadata,
        total_pages: int,
        document_type: DocumentType = DocumentType.BOOK,
        language: DocumentLanguage = DocumentLanguage.ENGLISH,
        project: Optional[str] = None,
    ) -> "Document":
        """Factory method to create a new Document"""
        return cls(
            id=str(uuid.uuid4()),
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            document_type=document_type,
            metadata=metadata,
            language=language,
            total_pages=total_pages,
            project=project,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "file_size": self.file_size,
            "document_type": self.document_type.value,
            "metadata": self.metadata.to_dict(),
            "language": self.language.value,
            "total_pages": self.total_pages,
            "chapters": [c.to_dict() for c in self.chapters],
            "status": self.status.value,
            "processed_pages": self.processed_pages,
            "created_at": self.created_at.isoformat(),
            "processed_at": (
                self.processed_at.isoformat() if self.processed_at else None
            ),
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
            "project": self.project,
            "access_count": self.access_count,
        }


@dataclass
class Page:
    """
    Represents a single page from a document.

    Stores raw OCR output and links to chunks.
    """

    id: str
    document_id: str
    page_number: int
    raw_text: str  # Original OCR output (markdown)
    cleaned_text: str  # Post-processed text
    has_images: bool = False
    has_tables: bool = False
    detected_headers: List[str] = field(default_factory=list)
    ocr_confidence: float = 0.0
    ocr_model: str = ""
    processed_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(
        cls,
        document_id: str,
        page_number: int,
        raw_text: str,
        cleaned_text: Optional[str] = None,
        ocr_confidence: float = 0.0,
        ocr_model: str = "deepseek-ocr-2",
    ) -> "Page":
        """Factory method to create a new Page"""
        return cls(
            id=str(uuid.uuid4()),
            document_id=document_id,
            page_number=page_number,
            raw_text=raw_text,
            cleaned_text=cleaned_text or raw_text,
            ocr_confidence=ocr_confidence,
            ocr_model=ocr_model,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "has_images": self.has_images,
            "has_tables": self.has_tables,
            "detected_headers": self.detected_headers,
            "ocr_confidence": self.ocr_confidence,
            "ocr_model": self.ocr_model,
            "processed_at": self.processed_at.isoformat(),
        }


@dataclass
class Chunk:
    """
    A semantic chunk for retrieval.

    Created from pages using chunking strategy.
    Stored with embedding for vector search.
    Links back to source page and LTM.
    """

    id: str
    document_id: str
    page_ids: List[str]  # Can span multiple pages
    content: str
    page_range: str  # "pp. 45-47" for citation
    chapter: Optional[str] = None
    section: Optional[str] = None
    embedding: Optional[List[float]] = None
    chunk_index: int = 0  # Order in document
    char_count: int = 0
    token_count_approx: int = 0
    ltm_entry_id: Optional[str] = None  # Link to LTM
    entities: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(
        cls,
        document_id: str,
        page_ids: List[str],
        content: str,
        page_range: str,
        chunk_index: int,
        chapter: Optional[str] = None,
        section: Optional[str] = None,
    ) -> "Chunk":
        """Factory method to create a new Chunk"""
        return cls(
            id=str(uuid.uuid4()),
            document_id=document_id,
            page_ids=page_ids,
            content=content,
            page_range=page_range,
            chunk_index=chunk_index,
            chapter=chapter,
            section=section,
            char_count=len(content),
            token_count_approx=len(content) // 4,  # Rough estimate
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "page_ids": self.page_ids,
            "content": self.content,
            "page_range": self.page_range,
            "chapter": self.chapter,
            "section": self.section,
            "embedding": self.embedding,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "token_count_approx": self.token_count_approx,
            "ltm_entry_id": self.ltm_entry_id,
            "entities": self.entities,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Citation:
    """Citation reference for sourcing answers"""

    document_id: str
    document_title: str
    chunk_id: str
    page_range: str
    chapter: Optional[str]
    section: Optional[str]
    quote: str  # Exact quote from source
    relevance: float

    def format_inline(self) -> str:
        """Format as inline citation [Author, p. X]"""
        return f"[{self.document_title}, {self.page_range}]"

    def format_footnote(self) -> str:
        """Format as footnote citation"""
        parts = [self.document_title]
        if self.chapter:
            parts.append(self.chapter)
        parts.append(self.page_range)
        return ". ".join(parts)

    def format(self, style: str = "inline") -> str:
        """Format citation in specified style"""
        if style == "inline":
            return self.format_inline()
        elif style == "footnote":
            return self.format_footnote()
        return f"Source: {self.document_title}, {self.page_range}"


@dataclass
class DocumentSearchResult:
    """Result from document search"""

    chunk: Chunk
    document: Document
    similarity: float
    highlight: str  # Relevant snippet
    citation: Citation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "document_id": self.document.id,
            "document_title": self.document.metadata.title,
            "content": self.chunk.content,
            "page_range": self.chunk.page_range,
            "chapter": self.chunk.chapter,
            "similarity": self.similarity,
            "highlight": self.highlight,
            "citation": self.citation.format_inline(),
        }


@dataclass
class OCRResult:
    """Result from OCR processing"""

    text: str
    confidence: float
    has_images: bool = False
    has_tables: bool = False
    detected_headers: List[str] = field(default_factory=list)
    model_used: str = "deepseek-ocr-2"
    processing_time_ms: int = 0


@dataclass
class ProcessingResult:
    """Result of document processing"""

    document_id: str
    status: ProcessingStatus
    pages_processed: int
    chunks_created: int
    errors: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
