"""
Document Storage Layer

SQLite-based storage for documents, pages, and chunks.
Follows the same patterns as memory/layers/long_term.py.
"""

from __future__ import annotations

import json
import logging
import pickle
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from documents.config import StorageConfig, get_document_config
from documents.models import (
    Chunk,
    ChapterInfo,
    Document,
    DocumentMetadata,
    DocumentLanguage,
    DocumentType,
    Page,
    ProcessingStatus,
)

LOGGER = logging.getLogger(__name__)


class DocumentStore:
    """
    SQLite storage for documents, pages, and chunks.

    Provides:
    - Document CRUD operations
    - Page storage and retrieval
    - Chunk storage with embeddings
    - FTS5 full-text search
    - Vector similarity search
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or get_document_config().storage
        self._db_path = Path(self.config.db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize database and create tables"""
        if self._initialized:
            return

        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        self._create_tables()
        self._initialized = True
        LOGGER.info("DocumentStore initialized at %s", self._db_path)

    def close(self) -> None:
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._initialized = False

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database transactions"""
        if not self._conn:
            self.initialize()
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    def _create_tables(self) -> None:
        """Create database schema"""
        with self._transaction() as cur:
            # Documents table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    file_size INTEGER NOT NULL,
                    document_type TEXT NOT NULL,
                    metadata JSON NOT NULL,
                    language TEXT DEFAULT 'en',
                    total_pages INTEGER DEFAULT 0,
                    chapters JSON,
                    status TEXT DEFAULT 'pending',
                    processed_pages INTEGER DEFAULT 0,
                    project TEXT,
                    access_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    processed_at TEXT,
                    last_accessed TEXT
                )
            """
            )

            # Pages table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pages (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page_number INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    cleaned_text TEXT,
                    has_images BOOLEAN DEFAULT FALSE,
                    has_tables BOOLEAN DEFAULT FALSE,
                    detected_headers JSON,
                    ocr_confidence REAL DEFAULT 0.0,
                    ocr_model TEXT,
                    processed_at TEXT NOT NULL,
                    UNIQUE(document_id, page_number)
                )
            """
            )

            # Chunks table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page_ids JSON NOT NULL,
                    content TEXT NOT NULL,
                    page_range TEXT,
                    chapter TEXT,
                    section TEXT,
                    embedding BLOB,
                    chunk_index INTEGER DEFAULT 0,
                    char_count INTEGER DEFAULT 0,
                    token_count_approx INTEGER DEFAULT 0,
                    ltm_entry_id TEXT,
                    entities JSON,
                    created_at TEXT NOT NULL
                )
            """
            )

            # Indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(document_type)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_project ON documents(project)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_status ON documents(status)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_doc_hash ON documents(file_hash)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_pages_doc ON pages(document_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_ltm ON chunks(ltm_entry_id)"
            )

            # FTS5 for full-text search on chunks
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    id,
                    content,
                    chapter,
                    section,
                    content=chunks,
                    content_rowid=rowid,
                    tokenize='porter unicode61'
                )
            """
            )

            # Triggers to keep FTS in sync
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, id, content, chapter, section)
                    VALUES (NEW.rowid, NEW.id, NEW.content, NEW.chapter, NEW.section);
                END
            """
            )

            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, id, content, chapter, section)
                    VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.chapter, OLD.section);
                END
            """
            )

            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, id, content, chapter, section)
                    VALUES ('delete', OLD.rowid, OLD.id, OLD.content, OLD.chapter, OLD.section);
                    INSERT INTO chunks_fts(rowid, id, content, chapter, section)
                    VALUES (NEW.rowid, NEW.id, NEW.content, NEW.chapter, NEW.section);
                END
            """
            )

    # ========== Document Operations ==========

    def store_document(self, document: Document) -> None:
        """Store a document"""
        with self._transaction() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO documents
                (id, file_path, file_hash, file_size, document_type, metadata,
                 language, total_pages, chapters, status, processed_pages, project,
                 access_count, created_at, processed_at, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.file_path,
                    document.file_hash,
                    document.file_size,
                    document.document_type.value,
                    json.dumps(document.metadata.to_dict()),
                    document.language.value,
                    document.total_pages,
                    json.dumps([c.to_dict() for c in document.chapters]),
                    document.status.value,
                    document.processed_pages,
                    document.project,
                    document.access_count,
                    document.created_at.isoformat(),
                    (
                        document.processed_at.isoformat()
                        if document.processed_at
                        else None
                    ),
                    (
                        document.last_accessed.isoformat()
                        if document.last_accessed
                        else None
                    ),
                ),
            )

    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document by ID"""
        with self._transaction() as cur:
            cur.execute("SELECT * FROM documents WHERE id = ?", (document_id,))
            row = cur.fetchone()
            if row:
                return self._row_to_document(row)
        return None

    def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        """Get a document by file hash (for deduplication)"""
        with self._transaction() as cur:
            cur.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
            row = cur.fetchone()
            if row:
                return self._row_to_document(row)
        return None

    def list_documents(
        self,
        document_type: Optional[DocumentType] = None,
        project: Optional[str] = None,
        status: Optional[ProcessingStatus] = None,
        limit: int = 100,
    ) -> List[Document]:
        """List documents with optional filters"""
        query = "SELECT * FROM documents WHERE 1=1"
        params: List[Any] = []

        if document_type:
            query += " AND document_type = ?"
            params.append(document_type.value)
        if project:
            query += " AND project = ?"
            params.append(project)
        if status:
            query += " AND status = ?"
            params.append(status.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._transaction() as cur:
            cur.execute(query, params)
            return [self._row_to_document(row) for row in cur.fetchall()]

    def update_document_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        processed_pages: Optional[int] = None,
    ) -> None:
        """Update document processing status"""
        with self._transaction() as cur:
            if processed_pages is not None:
                cur.execute(
                    """
                    UPDATE documents
                    SET status = ?, processed_pages = ?,
                        processed_at = CASE WHEN ? = 'completed' THEN ? ELSE processed_at END
                    WHERE id = ?
                    """,
                    (
                        status.value,
                        processed_pages,
                        status.value,
                        datetime.now().isoformat(),
                        document_id,
                    ),
                )
            else:
                cur.execute(
                    "UPDATE documents SET status = ? WHERE id = ?",
                    (status.value, document_id),
                )

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all related data"""
        with self._transaction() as cur:
            cur.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            return cur.rowcount > 0

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        """Convert database row to Document"""
        metadata = DocumentMetadata.from_dict(json.loads(row["metadata"]))
        chapters_data = json.loads(row["chapters"]) if row["chapters"] else []
        chapters = [ChapterInfo.from_dict(c) for c in chapters_data]

        return Document(
            id=row["id"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            file_size=row["file_size"],
            document_type=DocumentType(row["document_type"]),
            metadata=metadata,
            language=DocumentLanguage(row["language"]),
            total_pages=row["total_pages"],
            chapters=chapters,
            status=ProcessingStatus(row["status"]),
            processed_pages=row["processed_pages"],
            project=row["project"],
            access_count=row["access_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            processed_at=(
                datetime.fromisoformat(row["processed_at"])
                if row["processed_at"]
                else None
            ),
            last_accessed=(
                datetime.fromisoformat(row["last_accessed"])
                if row["last_accessed"]
                else None
            ),
        )

    # ========== Page Operations ==========

    def store_page(self, page: Page) -> None:
        """Store a page"""
        with self._transaction() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO pages
                (id, document_id, page_number, raw_text, cleaned_text,
                 has_images, has_tables, detected_headers, ocr_confidence,
                 ocr_model, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    page.id,
                    page.document_id,
                    page.page_number,
                    page.raw_text,
                    page.cleaned_text,
                    page.has_images,
                    page.has_tables,
                    json.dumps(page.detected_headers),
                    page.ocr_confidence,
                    page.ocr_model,
                    page.processed_at.isoformat(),
                ),
            )

    def store_pages(self, pages: List[Page]) -> None:
        """Store multiple pages in a batch"""
        with self._transaction() as cur:
            cur.executemany(
                """
                INSERT OR REPLACE INTO pages
                (id, document_id, page_number, raw_text, cleaned_text,
                 has_images, has_tables, detected_headers, ocr_confidence,
                 ocr_model, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        page.id,
                        page.document_id,
                        page.page_number,
                        page.raw_text,
                        page.cleaned_text,
                        page.has_images,
                        page.has_tables,
                        json.dumps(page.detected_headers),
                        page.ocr_confidence,
                        page.ocr_model,
                        page.processed_at.isoformat(),
                    )
                    for page in pages
                ],
            )

    def get_page(self, document_id: str, page_number: int) -> Optional[Page]:
        """Get a specific page"""
        with self._transaction() as cur:
            cur.execute(
                "SELECT * FROM pages WHERE document_id = ? AND page_number = ?",
                (document_id, page_number),
            )
            row = cur.fetchone()
            if row:
                return self._row_to_page(row)
        return None

    def get_pages(
        self,
        document_id: str,
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> List[Page]:
        """Get pages for a document"""
        query = "SELECT * FROM pages WHERE document_id = ? AND page_number >= ?"
        params: List[Any] = [document_id, start_page]

        if end_page:
            query += " AND page_number <= ?"
            params.append(end_page)

        query += " ORDER BY page_number"

        with self._transaction() as cur:
            cur.execute(query, params)
            return [self._row_to_page(row) for row in cur.fetchall()]

    def get_pages_for_document(self, document_id: str) -> List[Page]:
        """Get all pages for a document"""
        return self.get_pages(document_id)

    def _row_to_page(self, row: sqlite3.Row) -> Page:
        """Convert database row to Page"""
        return Page(
            id=row["id"],
            document_id=row["document_id"],
            page_number=row["page_number"],
            raw_text=row["raw_text"],
            cleaned_text=row["cleaned_text"] or row["raw_text"],
            has_images=bool(row["has_images"]),
            has_tables=bool(row["has_tables"]),
            detected_headers=json.loads(row["detected_headers"] or "[]"),
            ocr_confidence=row["ocr_confidence"],
            ocr_model=row["ocr_model"] or "",
            processed_at=datetime.fromisoformat(row["processed_at"]),
        )

    # ========== Chunk Operations ==========

    def store_chunk(self, chunk: Chunk) -> None:
        """Store a chunk with embedding"""
        embedding_blob = None
        if chunk.embedding:
            embedding_blob = pickle.dumps(np.array(chunk.embedding, dtype=np.float32))

        with self._transaction() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO chunks
                (id, document_id, page_ids, content, page_range, chapter, section,
                 embedding, chunk_index, char_count, token_count_approx,
                 ltm_entry_id, entities, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.document_id,
                    json.dumps(chunk.page_ids),
                    chunk.content,
                    chunk.page_range,
                    chunk.chapter,
                    chunk.section,
                    embedding_blob,
                    chunk.chunk_index,
                    chunk.char_count,
                    chunk.token_count_approx,
                    chunk.ltm_entry_id,
                    json.dumps(chunk.entities),
                    chunk.created_at.isoformat(),
                ),
            )

    def store_chunks(self, chunks: List[Chunk]) -> None:
        """Store multiple chunks in a batch"""
        data = []
        for chunk in chunks:
            embedding_blob = None
            if chunk.embedding:
                embedding_blob = pickle.dumps(
                    np.array(chunk.embedding, dtype=np.float32)
                )
            data.append(
                (
                    chunk.id,
                    chunk.document_id,
                    json.dumps(chunk.page_ids),
                    chunk.content,
                    chunk.page_range,
                    chunk.chapter,
                    chunk.section,
                    embedding_blob,
                    chunk.chunk_index,
                    chunk.char_count,
                    chunk.token_count_approx,
                    chunk.ltm_entry_id,
                    json.dumps(chunk.entities),
                    chunk.created_at.isoformat(),
                )
            )

        with self._transaction() as cur:
            cur.executemany(
                """
                INSERT OR REPLACE INTO chunks
                (id, document_id, page_ids, content, page_range, chapter, section,
                 embedding, chunk_index, char_count, token_count_approx,
                 ltm_entry_id, entities, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                data,
            )

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID"""
        with self._transaction() as cur:
            cur.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
            row = cur.fetchone()
            if row:
                return self._row_to_chunk(row)
        return None

    def get_chunks_for_document(
        self,
        document_id: str,
        chapter: Optional[str] = None,
    ) -> List[Chunk]:
        """Get all chunks for a document"""
        query = "SELECT * FROM chunks WHERE document_id = ?"
        params: List[Any] = [document_id]

        if chapter:
            query += " AND chapter = ?"
            params.append(chapter)

        query += " ORDER BY chunk_index"

        with self._transaction() as cur:
            cur.execute(query, params)
            return [self._row_to_chunk(row) for row in cur.fetchall()]

    def update_chunk_ltm_link(self, chunk_id: str, ltm_entry_id: str) -> None:
        """Update chunk's LTM entry link"""
        with self._transaction() as cur:
            cur.execute(
                "UPDATE chunks SET ltm_entry_id = ? WHERE id = ?",
                (ltm_entry_id, chunk_id),
            )

    def _row_to_chunk(self, row: sqlite3.Row) -> Chunk:
        """Convert database row to Chunk"""
        embedding = None
        if row["embedding"]:
            embedding = pickle.loads(row["embedding"]).tolist()

        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            page_ids=json.loads(row["page_ids"]),
            content=row["content"],
            page_range=row["page_range"],
            chapter=row["chapter"],
            section=row["section"],
            embedding=embedding,
            chunk_index=row["chunk_index"],
            char_count=row["char_count"],
            token_count_approx=row["token_count_approx"],
            ltm_entry_id=row["ltm_entry_id"],
            entities=json.loads(row["entities"] or "[]"),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ========== Search Operations ==========

    def vector_search(
        self,
        query_embedding: List[float],
        document_id: Optional[str] = None,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[Chunk, float]]:
        """Search chunks by vector similarity"""
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        query_vec = query_vec / query_norm

        # Get all chunks with embeddings
        sql = "SELECT * FROM chunks WHERE embedding IS NOT NULL"
        params: List[Any] = []
        if document_id:
            sql += " AND document_id = ?"
            params.append(document_id)

        results: List[Tuple[Chunk, float]] = []

        with self._transaction() as cur:
            cur.execute(sql, params)
            for row in cur.fetchall():
                if not row["embedding"]:
                    continue

                doc_vec = pickle.loads(row["embedding"])
                doc_norm = np.linalg.norm(doc_vec)
                if doc_norm == 0:
                    continue

                doc_vec = doc_vec / doc_norm
                similarity = float(np.dot(query_vec, doc_vec))

                if similarity >= min_similarity:
                    chunk = self._row_to_chunk(row)
                    results.append((chunk, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def keyword_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        """Search chunks using FTS5 full-text search"""
        # Escape special FTS characters
        safe_query = query.replace('"', '""')

        sql = """
            SELECT c.*, bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON chunks_fts.id = c.id
            WHERE chunks_fts MATCH ?
        """
        params: List[Any] = [safe_query]

        if document_id:
            sql += " AND c.document_id = ?"
            params.append(document_id)

        sql += " ORDER BY score LIMIT ?"
        params.append(top_k)

        results: List[Tuple[Chunk, float]] = []

        with self._transaction() as cur:
            try:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    chunk = self._row_to_chunk(row)
                    # Normalize BM25 score to 0-1 range (approximate)
                    score = min(1.0, abs(row["score"]) / 10.0)
                    results.append((chunk, score))
            except sqlite3.OperationalError as e:
                LOGGER.warning("FTS search failed: %s", e)
                # Fall back to LIKE search
                return self._like_search(query, document_id, top_k)

        return results

    def _like_search(
        self,
        query: str,
        document_id: Optional[str],
        top_k: int,
    ) -> List[Tuple[Chunk, float]]:
        """Fallback search using LIKE"""
        sql = "SELECT * FROM chunks WHERE content LIKE ?"
        params: List[Any] = [f"%{query}%"]

        if document_id:
            sql += " AND document_id = ?"
            params.append(document_id)

        sql += " LIMIT ?"
        params.append(top_k)

        results: List[Tuple[Chunk, float]] = []

        with self._transaction() as cur:
            cur.execute(sql, params)
            for row in cur.fetchall():
                chunk = self._row_to_chunk(row)
                results.append((chunk, 0.5))  # Fixed relevance for LIKE

        return results

    # ========== Statistics ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self._transaction() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            doc_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM pages")
            page_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL")
            embedded_count = cur.fetchone()[0]

            cur.execute(
                "SELECT document_type, COUNT(*) FROM documents GROUP BY document_type"
            )
            type_dist = dict(cur.fetchall())

            cur.execute("SELECT status, COUNT(*) FROM documents GROUP BY status")
            status_dist = dict(cur.fetchall())

        return {
            "total_documents": doc_count,
            "total_pages": page_count,
            "total_chunks": chunk_count,
            "chunks_with_embeddings": embedded_count,
            "document_types": type_dist,
            "processing_status": status_dist,
            "db_path": str(self._db_path),
        }
