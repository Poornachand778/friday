"""
Long-Term Memory Layer
======================

Consolidated knowledge with vector embeddings for semantic search.

Features:
    - Vector similarity search using sentence-transformers
    - Dual timestamps for temporal reasoning
    - Memory types (fact, preference, event, pattern, decision)
    - Telugu-aware keyword extraction
    - Decay-resistant important memories

Brain Inspiration:
    Human LTM stores consolidated, strengthened memories permanently.
    Retrieval is cue-based (semantic similarity).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from memory.config import LTMConfig, get_memory_config

LOGGER = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of long-term memories"""

    FACT = "fact"  # "Boss's birthday is October 15"
    PREFERENCE = "preference"  # "Boss prefers direct communication"
    EVENT = "event"  # "Gusagusalu deadline is March"
    PATTERN = "pattern"  # "Boss usually writes in mornings"
    DECISION = "decision"  # "We decided to use flashback structure"
    RELATIONSHIP = "relationship"  # "Ravi is the protagonist's father"


@dataclass
class LTMEntry:
    """A long-term memory entry"""

    id: str

    # Content
    content: str  # High-signal extracted fact
    source_summary: str  # Where this came from

    # Categorization
    memory_type: MemoryType
    domain: str = "general"  # film, personal, technical

    # Dual timestamps
    created_at: datetime = field(default_factory=datetime.now)
    event_date: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # Relationships
    related_memories: List[str] = field(default_factory=list)
    project: Optional[str] = None
    entities: List[str] = field(default_factory=list)

    # Embedding
    embedding: Optional[List[float]] = None

    # Scoring
    confidence: float = 0.8
    trust_level: int = 3  # 1-5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance: float = 0.5

    # Telugu support
    language: str = "en"  # en, te, mixed
    telugu_keywords: List[str] = field(default_factory=list)

    # Source tracking
    source_stm_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "source_summary": self.source_summary,
            "memory_type": self.memory_type.value,
            "domain": self.domain,
            "created_at": self.created_at.isoformat(),
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "related_memories": self.related_memories,
            "project": self.project,
            "entities": self.entities,
            "embedding": self.embedding,
            "confidence": self.confidence,
            "trust_level": self.trust_level,
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
            "importance": self.importance,
            "language": self.language,
            "telugu_keywords": self.telugu_keywords,
            "source_stm_ids": self.source_stm_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LTMEntry":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            content=data["content"],
            source_summary=data.get("source_summary", ""),
            memory_type=MemoryType(data["memory_type"]),
            domain=data.get("domain", "general"),
            created_at=datetime.fromisoformat(data["created_at"]),
            event_date=(
                datetime.fromisoformat(data["event_date"])
                if data.get("event_date")
                else None
            ),
            valid_until=(
                datetime.fromisoformat(data["valid_until"])
                if data.get("valid_until")
                else None
            ),
            related_memories=data.get("related_memories", []),
            project=data.get("project"),
            entities=data.get("entities", []),
            embedding=data.get("embedding"),
            confidence=data.get("confidence", 0.8),
            trust_level=data.get("trust_level", 3),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"])
                if data.get("last_accessed")
                else None
            ),
            importance=data.get("importance", 0.5),
            language=data.get("language", "en"),
            telugu_keywords=data.get("telugu_keywords", []),
            source_stm_ids=data.get("source_stm_ids", []),
        )


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.

    Lazy-loads model on first use to reduce startup time.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                LOGGER.info("Loaded embedding model: %s", self.model_name)
            except ImportError:
                LOGGER.warning(
                    "sentence-transformers not installed, embeddings disabled"
                )
                self._model = False  # Mark as unavailable
            except Exception as e:
                LOGGER.error("Failed to load embedding model: %s", e)
                self._model = False

    def encode(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        self._load_model()

        if self._model is False:
            return None

        try:
            embedding = self._model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            LOGGER.error("Failed to generate embedding: %s", e)
            return None

    def encode_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts"""
        self._load_model()

        if self._model is False:
            return [None] * len(texts)

        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return [e.tolist() for e in embeddings]
        except Exception as e:
            LOGGER.error("Failed to generate batch embeddings: %s", e)
            return [None] * len(texts)

    @property
    def is_available(self) -> bool:
        """Check if embedding model is available"""
        self._load_model()
        return self._model is not False


class LongTermMemory:
    """
    Long-term memory storage with vector search.

    Uses Qdrant HNSW for fast vector search when available,
    falls back to SQLite brute-force cosine similarity.
    Falls back to keyword search if embeddings unavailable.

    Usage:
        ltm = LongTermMemory()
        await ltm.initialize()

        # Store a memory
        entry = await ltm.store(
            content="Boss prefers direct communication without flattery",
            memory_type=MemoryType.PREFERENCE,
            domain="personal",
        )

        # Semantic search
        results = await ltm.search("how should I communicate?", top_k=5)

        # Temporal search
        upcoming = await ltm.search_upcoming_events(days=7)
    """

    def __init__(self, config: Optional[LTMConfig] = None):
        self.config = config or get_memory_config().ltm
        self._db_path = Path(self.config.sqlite_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._embedder: Optional[EmbeddingModel] = None
        self._vector_store = None  # Optional Qdrant backend

    async def initialize(self) -> None:
        """Initialize database, embedding model, and optional Qdrant"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Enable WAL mode
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._create_tables()

        # Initialize embedding model
        self._embedder = EmbeddingModel(self.config.embedding_model)

        # Try to connect to Qdrant
        try:
            from db.vector_store import get_vector_store

            self._vector_store = await get_vector_store()
            if self._vector_store:
                LOGGER.info("LTM using Qdrant for vector search")
            else:
                LOGGER.info("LTM using SQLite vector search fallback")
        except Exception as e:
            LOGGER.debug("Could not initialize Qdrant for LTM: %s", e)

        LOGGER.info("LTM initialized: %s", self._db_path)

    def _create_tables(self) -> None:
        """Create database tables"""
        with self._transaction() as cur:
            # Main table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source_summary TEXT,
                    memory_type TEXT NOT NULL,
                    domain TEXT DEFAULT 'general',
                    created_at TEXT NOT NULL,
                    event_date TEXT,
                    valid_until TEXT,
                    related_memories TEXT,
                    project TEXT,
                    entities TEXT,
                    embedding BLOB,
                    confidence REAL DEFAULT 0.8,
                    trust_level INTEGER DEFAULT 3,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    importance REAL DEFAULT 0.5,
                    language TEXT DEFAULT 'en',
                    telugu_keywords TEXT,
                    source_stm_ids TEXT
                )
            """
            )

            # Indexes
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ltm_type
                ON long_term_memories(memory_type)
            """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ltm_project
                ON long_term_memories(project)
            """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ltm_event_date
                ON long_term_memories(event_date)
            """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ltm_importance
                ON long_term_memories(importance)
            """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ltm_created
                ON long_term_memories(created_at)
            """
            )

            # FTS for keyword search fallback
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS ltm_fts USING fts5(
                    id,
                    content,
                    source_summary,
                    entities,
                    telugu_keywords,
                    content=long_term_memories,
                    content_rowid=rowid,
                    tokenize='porter unicode61'
                )
            """
            )

            # FTS triggers
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS ltm_ai AFTER INSERT ON long_term_memories BEGIN
                    INSERT INTO ltm_fts(rowid, id, content, source_summary, entities, telugu_keywords)
                    VALUES (new.rowid, new.id, new.content, new.source_summary, new.entities, new.telugu_keywords);
                END
            """
            )
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS ltm_ad AFTER DELETE ON long_term_memories BEGIN
                    INSERT INTO ltm_fts(ltm_fts, rowid, id, content, source_summary, entities, telugu_keywords)
                    VALUES ('delete', old.rowid, old.id, old.content, old.source_summary, old.entities, old.telugu_keywords);
                END
            """
            )

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database transactions"""
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    async def close(self) -> None:
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None
        LOGGER.info("LTM closed")

    # =========================================================================
    # Storage Operations
    # =========================================================================

    async def store(
        self,
        content: str,
        memory_type: MemoryType,
        source_summary: str = "",
        domain: str = "general",
        event_date: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
        project: Optional[str] = None,
        entities: Optional[List[str]] = None,
        confidence: float = 0.8,
        trust_level: int = 3,
        importance: float = 0.5,
        language: str = "en",
        telugu_keywords: Optional[List[str]] = None,
        source_stm_ids: Optional[List[str]] = None,
    ) -> LTMEntry:
        """
        Store a new long-term memory.

        Args:
            content: The memory content (high-signal fact)
            memory_type: Type of memory (fact, preference, etc.)
            source_summary: Where this came from
            domain: Domain category (film, personal, technical)
            event_date: When the referenced event occurs
            valid_until: Expiry date for time-bound facts
            project: Associated project
            entities: Named entities mentioned
            confidence: How certain (0-1)
            trust_level: Source reliability (1-5)
            importance: Importance score (0-1)
            language: Language (en, te, mixed)
            telugu_keywords: Telugu keywords for search
            source_stm_ids: STM entries this was consolidated from

        Returns:
            Created LTMEntry
        """
        # Generate embedding
        embedding = None
        if self._embedder and self._embedder.is_available:
            embedding = self._embedder.encode(content)

        entry = LTMEntry(
            id=str(uuid.uuid4()),
            content=content,
            source_summary=source_summary,
            memory_type=memory_type,
            domain=domain,
            created_at=datetime.now(),
            event_date=event_date,
            valid_until=valid_until,
            project=project,
            entities=entities or [],
            embedding=embedding,
            confidence=confidence,
            trust_level=trust_level,
            importance=importance,
            language=language,
            telugu_keywords=telugu_keywords or [],
            source_stm_ids=source_stm_ids or [],
        )

        with self._transaction() as cur:
            cur.execute(
                """
                INSERT INTO long_term_memories (
                    id, content, source_summary, memory_type, domain,
                    created_at, event_date, valid_until, related_memories,
                    project, entities, embedding, confidence, trust_level,
                    access_count, importance, language, telugu_keywords, source_stm_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.id,
                    entry.content,
                    entry.source_summary,
                    entry.memory_type.value,
                    entry.domain,
                    entry.created_at.isoformat(),
                    entry.event_date.isoformat() if entry.event_date else None,
                    entry.valid_until.isoformat() if entry.valid_until else None,
                    json.dumps(entry.related_memories),
                    entry.project,
                    json.dumps(entry.entities),
                    self._serialize_embedding(entry.embedding),
                    entry.confidence,
                    entry.trust_level,
                    0,
                    entry.importance,
                    entry.language,
                    json.dumps(entry.telugu_keywords),
                    json.dumps(entry.source_stm_ids),
                ),
            )

        # Also upsert to Qdrant if available
        if self._vector_store and entry.embedding:
            try:
                from db.vector_store import COLLECTION_MEMORIES

                await self._vector_store.upsert(
                    collection=COLLECTION_MEMORIES,
                    id=entry.id,
                    vector=entry.embedding,
                    payload={
                        "memory_type": entry.memory_type.value,
                        "domain": entry.domain,
                        "project": entry.project or "",
                        "importance": entry.importance,
                        "language": entry.language,
                    },
                )
            except Exception as e:
                LOGGER.warning("Failed to upsert LTM to Qdrant: %s", e)

        LOGGER.debug("Stored LTM: %s (%s)", entry.id[:8], memory_type.value)
        return entry

    async def get(self, memory_id: str) -> Optional[LTMEntry]:
        """Get a memory by ID"""
        with self._transaction() as cur:
            cur.execute("SELECT * FROM long_term_memories WHERE id = ?", (memory_id,))
            row = cur.fetchone()

        if not row:
            return None

        # Update access tracking
        await self._record_access(memory_id)

        return self._row_to_entry(row)

    async def update(self, memory_id: str, **kwargs) -> Optional[LTMEntry]:
        """Update a memory's fields"""
        allowed = {
            "content",
            "importance",
            "confidence",
            "trust_level",
            "event_date",
            "valid_until",
            "project",
            "entities",
            "related_memories",
            "telugu_keywords",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}

        if not updates:
            return await self.get(memory_id)

        # Handle special serializations
        if "entities" in updates:
            updates["entities"] = json.dumps(updates["entities"])
        if "related_memories" in updates:
            updates["related_memories"] = json.dumps(updates["related_memories"])
        if "telugu_keywords" in updates:
            updates["telugu_keywords"] = json.dumps(updates["telugu_keywords"])
        if "event_date" in updates and updates["event_date"]:
            updates["event_date"] = updates["event_date"].isoformat()
        if "valid_until" in updates and updates["valid_until"]:
            updates["valid_until"] = updates["valid_until"].isoformat()

        # Re-generate embedding if content changed
        new_embedding = None
        if "content" in updates and self._embedder and self._embedder.is_available:
            new_embedding = self._embedder.encode(updates["content"])
            updates["embedding"] = self._serialize_embedding(new_embedding)

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [memory_id]

        with self._transaction() as cur:
            cur.execute(
                f"UPDATE long_term_memories SET {set_clause} WHERE id = ?", values
            )

        # Update Qdrant if embedding changed
        if new_embedding and self._vector_store:
            try:
                from db.vector_store import COLLECTION_MEMORIES

                entry = await self.get(memory_id)
                if entry:
                    await self._vector_store.upsert(
                        collection=COLLECTION_MEMORIES,
                        id=memory_id,
                        vector=new_embedding,
                        payload={
                            "memory_type": entry.memory_type.value,
                            "domain": entry.domain,
                            "project": entry.project or "",
                            "importance": entry.importance,
                            "language": entry.language,
                        },
                    )
            except Exception as e:
                LOGGER.warning("Failed to update Qdrant embedding: %s", e)

        LOGGER.debug("Updated LTM: %s", memory_id[:8])
        return await self.get(memory_id)

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory from SQLite and Qdrant"""
        with self._transaction() as cur:
            cur.execute("DELETE FROM long_term_memories WHERE id = ?", (memory_id,))
            deleted = cur.rowcount > 0

        if deleted:
            # Also remove from Qdrant
            if self._vector_store:
                try:
                    from db.vector_store import COLLECTION_MEMORIES

                    await self._vector_store.delete(COLLECTION_MEMORIES, memory_id)
                except Exception as e:
                    LOGGER.warning("Failed to delete from Qdrant: %s", e)

            LOGGER.debug("Deleted LTM: %s", memory_id[:8])
        return deleted

    async def boost_importance(
        self, memory_id: str, boost: float = 0.1
    ) -> Optional[LTMEntry]:
        """Boost a memory's importance (when user reinforces)"""
        entry = await self.get(memory_id)
        if entry:
            new_importance = min(entry.importance + boost, 1.0)
            return await self.update(memory_id, importance=new_importance)
        return None

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        top_k: int = 10,
        memory_type: Optional[MemoryType] = None,
        project: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[Tuple[LTMEntry, float]]:
        """
        Search memories using vector similarity.

        Falls back to FTS if embeddings unavailable.

        Args:
            query: Search query
            top_k: Maximum results
            memory_type: Filter by type
            project: Filter by project
            min_importance: Minimum importance threshold

        Returns:
            List of (LTMEntry, similarity_score) tuples
        """
        # Try vector search first
        if self._embedder and self._embedder.is_available:
            query_embedding = self._embedder.encode(query)
            if query_embedding:
                return await self._vector_search(
                    query_embedding, top_k, memory_type, project, min_importance
                )

        # Fall back to FTS
        return await self._keyword_search(
            query, top_k, memory_type, project, min_importance
        )

    async def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        memory_type: Optional[MemoryType],
        project: Optional[str],
        min_importance: float,
    ) -> List[Tuple[LTMEntry, float]]:
        """Vector similarity search (Qdrant HNSW first, SQLite fallback)"""

        # Try Qdrant first
        if self._vector_store:
            try:
                results = await self._qdrant_vector_search(
                    query_embedding, top_k, memory_type, project, min_importance
                )
                if results:
                    return results
            except Exception as e:
                LOGGER.warning("Qdrant LTM search failed, using SQLite: %s", e)

        # SQLite brute-force fallback
        return await self._sqlite_vector_search(
            query_embedding, top_k, memory_type, project, min_importance
        )

    async def _qdrant_vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        memory_type: Optional[MemoryType],
        project: Optional[str],
        min_importance: float,
    ) -> List[Tuple[LTMEntry, float]]:
        """Vector search using Qdrant HNSW index"""
        from db.vector_store import COLLECTION_MEMORIES

        filters = {}
        if memory_type:
            filters["memory_type"] = memory_type.value
        if project:
            filters["project"] = project

        qdrant_results = await self._vector_store.search(
            collection=COLLECTION_MEMORIES,
            query_vector=query_embedding,
            top_k=top_k,
            min_score=0.0,
            filters=filters if filters else None,
        )

        # Convert Qdrant results to LTMEntry tuples
        results: List[Tuple[LTMEntry, float]] = []
        for r in qdrant_results:
            entry = await self.get(r.id)
            if entry:
                # Apply importance filter (Qdrant doesn't do range filters easily)
                if entry.importance >= min_importance:
                    results.append((entry, r.score))

        # Record access for retrieved memories
        for entry, _ in results:
            await self._record_access(entry.id)

        return results

    async def _sqlite_vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        memory_type: Optional[MemoryType],
        project: Optional[str],
        min_importance: float,
    ) -> List[Tuple[LTMEntry, float]]:
        """SQLite brute-force cosine similarity fallback"""
        query = "SELECT * FROM long_term_memories WHERE embedding IS NOT NULL"
        params: List[Any] = []

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)

        if project:
            query += " AND project = ?"
            params.append(project)

        if min_importance > 0:
            query += " AND importance >= ?"
            params.append(min_importance)

        with self._transaction() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        if not rows:
            return []

        # Calculate similarities
        query_vec = np.array(query_embedding)
        results = []

        for row in rows:
            embedding = self._deserialize_embedding(row["embedding"])
            if embedding is None:
                continue

            # Cosine similarity
            doc_vec = np.array(embedding)
            similarity = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            )

            entry = self._row_to_entry(row)
            results.append((entry, float(similarity)))

        # Sort by similarity and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]

        # Record access for retrieved memories
        for entry, _ in top_results:
            await self._record_access(entry.id)

        return top_results

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        memory_type: Optional[MemoryType],
        project: Optional[str],
        min_importance: float,
    ) -> List[Tuple[LTMEntry, float]]:
        """FTS-based keyword search fallback"""
        fts_query = """
            SELECT ltm.* FROM long_term_memories ltm
            JOIN ltm_fts ON ltm.id = ltm_fts.id
            WHERE ltm_fts MATCH ?
        """
        params: List[Any] = [query]

        if memory_type:
            fts_query += " AND ltm.memory_type = ?"
            params.append(memory_type.value)

        if project:
            fts_query += " AND ltm.project = ?"
            params.append(project)

        if min_importance > 0:
            fts_query += " AND ltm.importance >= ?"
            params.append(min_importance)

        fts_query += " ORDER BY rank LIMIT ?"
        params.append(top_k)

        try:
            with self._transaction() as cur:
                cur.execute(fts_query, params)
                rows = cur.fetchall()

            results = []
            for i, row in enumerate(rows):
                entry = self._row_to_entry(row)
                # Use rank position as pseudo-similarity
                similarity = 1.0 - (i / max(len(rows), 1))
                results.append((entry, similarity))
                await self._record_access(entry.id)

            return results

        except sqlite3.OperationalError as e:
            LOGGER.warning("FTS search failed: %s", e)
            return []

    async def search_by_entity(
        self,
        entity: str,
        top_k: int = 10,
    ) -> List[LTMEntry]:
        """Search memories mentioning a specific entity"""
        with self._transaction() as cur:
            cur.execute(
                """
                SELECT * FROM long_term_memories
                WHERE entities LIKE ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """,
                (f'%"{entity}"%', top_k),
            )
            rows = cur.fetchall()

        results = [self._row_to_entry(row) for row in rows]

        for entry in results:
            await self._record_access(entry.id)

        return results

    async def search_upcoming_events(
        self,
        days: int = 7,
        project: Optional[str] = None,
    ) -> List[LTMEntry]:
        """Search for memories with upcoming event dates"""
        now = datetime.now()
        cutoff = now + timedelta(days=days)

        query = """
            SELECT * FROM long_term_memories
            WHERE event_date IS NOT NULL
            AND event_date >= ? AND event_date <= ?
        """
        params: List[Any] = [now.isoformat(), cutoff.isoformat()]

        if project:
            query += " AND project = ?"
            params.append(project)

        query += " ORDER BY event_date ASC"

        with self._transaction() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def search_by_type(
        self,
        memory_type: MemoryType,
        top_k: int = 20,
        project: Optional[str] = None,
    ) -> List[LTMEntry]:
        """Get memories of a specific type"""
        query = """
            SELECT * FROM long_term_memories
            WHERE memory_type = ?
        """
        params: List[Any] = [memory_type.value]

        if project:
            query += " AND project = ?"
            params.append(project)

        query += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(top_k)

        with self._transaction() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    # =========================================================================
    # Maintenance Operations
    # =========================================================================

    async def get_decay_candidates(
        self,
        threshold: float = 0.2,
        exclude_types: Optional[List[MemoryType]] = None,
    ) -> List[LTMEntry]:
        """Get memories that should be considered for decay"""
        query = """
            SELECT * FROM long_term_memories
            WHERE importance < ?
        """
        params: List[Any] = [threshold]

        # Exclude certain types (preferences never fully decay)
        if exclude_types:
            placeholders = ", ".join("?" * len(exclude_types))
            query += f" AND memory_type NOT IN ({placeholders})"
            params.extend(t.value for t in exclude_types)

        query += " ORDER BY importance ASC"

        with self._transaction() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def get_expired_memories(self) -> List[LTMEntry]:
        """Get memories past their valid_until date"""
        now = datetime.now().isoformat()

        with self._transaction() as cur:
            cur.execute(
                """
                SELECT * FROM long_term_memories
                WHERE valid_until IS NOT NULL AND valid_until < ?
            """,
                (now,),
            )
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def get_stats(self) -> Dict[str, Any]:
        """Get LTM statistics"""
        with self._transaction() as cur:
            cur.execute("SELECT COUNT(*) FROM long_term_memories")
            total = cur.fetchone()[0]

            cur.execute(
                "SELECT memory_type, COUNT(*) FROM long_term_memories GROUP BY memory_type"
            )
            type_counts = dict(cur.fetchall())

            cur.execute(
                "SELECT COUNT(*) FROM long_term_memories WHERE embedding IS NOT NULL"
            )
            with_embeddings = cur.fetchone()[0]

            cur.execute("SELECT AVG(importance) FROM long_term_memories")
            avg_importance = cur.fetchone()[0] or 0

            cur.execute(
                "SELECT COUNT(DISTINCT project) FROM long_term_memories WHERE project IS NOT NULL"
            )
            projects = cur.fetchone()[0]

        # Check Qdrant status
        qdrant_count = None
        if self._vector_store:
            try:
                from db.vector_store import COLLECTION_MEMORIES

                qdrant_count = await self._vector_store.count(COLLECTION_MEMORIES)
            except Exception:
                pass

        stats = {
            "total": total,
            "by_type": type_counts,
            "with_embeddings": with_embeddings,
            "embedding_coverage": with_embeddings / total if total > 0 else 0,
            "avg_importance": round(avg_importance, 3),
            "projects": projects,
            "embedding_model": self.config.embedding_model,
            "embedding_available": (
                self._embedder.is_available if self._embedder else False
            ),
            "vector_backend": "qdrant" if self._vector_store else "sqlite",
        }
        if qdrant_count is not None:
            stats["qdrant_vectors"] = qdrant_count
        return stats

    # =========================================================================
    # Vector Store Sync
    # =========================================================================

    async def sync_to_vector_store(self, batch_size: int = 100) -> int:
        """
        Sync all LTM embeddings from SQLite to Qdrant.

        Used for initial migration or re-sync after Qdrant restart.

        Returns:
            Number of embeddings synced
        """
        if not self._vector_store:
            LOGGER.warning("No vector store available for sync")
            return 0

        from db.vector_store import COLLECTION_MEMORIES

        await self._vector_store.ensure_collection(COLLECTION_MEMORIES)

        with self._transaction() as cur:
            cur.execute("SELECT * FROM long_term_memories WHERE embedding IS NOT NULL")
            rows = cur.fetchall()

        if not rows:
            return 0

        total = 0
        ids_batch: list[str] = []
        vectors_batch: list[list[float]] = []
        payloads_batch: list[dict] = []

        for row in rows:
            embedding = self._deserialize_embedding(row["embedding"])
            if embedding is None:
                continue

            ids_batch.append(row["id"])
            vectors_batch.append(embedding)
            payloads_batch.append(
                {
                    "memory_type": row["memory_type"],
                    "domain": row["domain"],
                    "project": row["project"] or "",
                    "importance": row["importance"],
                    "language": row["language"],
                }
            )

            if len(ids_batch) >= batch_size:
                count = await self._vector_store.upsert_batch(
                    collection=COLLECTION_MEMORIES,
                    ids=ids_batch,
                    vectors=vectors_batch,
                    payloads=payloads_batch,
                )
                total += count
                ids_batch, vectors_batch, payloads_batch = [], [], []

        # Final batch
        if ids_batch:
            count = await self._vector_store.upsert_batch(
                collection=COLLECTION_MEMORIES,
                ids=ids_batch,
                vectors=vectors_batch,
                payloads=payloads_batch,
            )
            total += count

        LOGGER.info("Synced %d LTM embeddings to Qdrant", total)
        return total

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _record_access(self, memory_id: str) -> None:
        """Record that a memory was accessed"""
        with self._transaction() as cur:
            cur.execute(
                """
                UPDATE long_term_memories
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """,
                (datetime.now().isoformat(), memory_id),
            )

    def _serialize_embedding(self, embedding: Optional[List[float]]) -> Optional[bytes]:
        """Serialize embedding to bytes"""
        if embedding is None:
            return None
        return np.array(embedding, dtype=np.float32).tobytes()

    def _deserialize_embedding(self, data: Optional[bytes]) -> Optional[List[float]]:
        """Deserialize embedding from bytes"""
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.float32).tolist()

    def _row_to_entry(self, row: sqlite3.Row) -> LTMEntry:
        """Convert database row to LTMEntry"""
        return LTMEntry(
            id=row["id"],
            content=row["content"],
            source_summary=row["source_summary"] or "",
            memory_type=MemoryType(row["memory_type"]),
            domain=row["domain"],
            created_at=datetime.fromisoformat(row["created_at"]),
            event_date=(
                datetime.fromisoformat(row["event_date"]) if row["event_date"] else None
            ),
            valid_until=(
                datetime.fromisoformat(row["valid_until"])
                if row["valid_until"]
                else None
            ),
            related_memories=(
                json.loads(row["related_memories"]) if row["related_memories"] else []
            ),
            project=row["project"],
            entities=json.loads(row["entities"]) if row["entities"] else [],
            embedding=self._deserialize_embedding(row["embedding"]),
            confidence=row["confidence"],
            trust_level=row["trust_level"],
            access_count=row["access_count"],
            last_accessed=(
                datetime.fromisoformat(row["last_accessed"])
                if row["last_accessed"]
                else None
            ),
            importance=row["importance"],
            language=row["language"],
            telugu_keywords=(
                json.loads(row["telugu_keywords"]) if row["telugu_keywords"] else []
            ),
            source_stm_ids=(
                json.loads(row["source_stm_ids"]) if row["source_stm_ids"] else []
            ),
        )

    def __repr__(self) -> str:
        return f"LongTermMemory(db={self._db_path})"
