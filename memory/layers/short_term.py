"""
Short-Term Memory Layer
=======================

Recent conversations stored for 7 days before consolidation.

Features:
    - SQLite storage with FTS5 for text search
    - Dual timestamps (document_time + event_time)
    - Fact extraction from conversations
    - Automatic consolidation to LTM

Brain Inspiration:
    Human STM holds information for seconds to days before
    either forgetting or consolidating to long-term memory.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from memory.config import STMConfig, get_memory_config

LOGGER = logging.getLogger(__name__)


@dataclass
class STMEntry:
    """A short-term memory entry"""

    id: str
    session_id: str

    # Content
    summary: str  # Compressed representation
    key_facts: List[str]  # Extracted facts
    raw_turns: List[Dict[str, Any]]  # Original conversation

    # Dual timestamps (Supermemory-inspired)
    created_at: datetime  # When conversation happened
    event_dates: List[datetime]  # Referenced future/past events

    # Metadata
    room: str = "general"
    project: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    language: str = "mixed"  # en, te, mixed

    # Scoring for decay
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance: float = 0.5

    # Status
    status: str = "active"  # active, archived, consolidated

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "summary": self.summary,
            "key_facts": self.key_facts,
            "raw_turns": self.raw_turns,
            "created_at": self.created_at.isoformat(),
            "event_dates": [d.isoformat() for d in self.event_dates],
            "room": self.room,
            "project": self.project,
            "topics": self.topics,
            "language": self.language,
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
            "importance": self.importance,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "STMEntry":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            summary=data["summary"],
            key_facts=data.get("key_facts", []),
            raw_turns=data.get("raw_turns", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            event_dates=[
                datetime.fromisoformat(d) for d in data.get("event_dates", [])
            ],
            room=data.get("room", "general"),
            project=data.get("project"),
            topics=data.get("topics", []),
            language=data.get("language", "mixed"),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"])
                if data.get("last_accessed")
                else None
            ),
            importance=data.get("importance", 0.5),
            status=data.get("status", "active"),
        )


class ShortTermMemory:
    """
    Short-term memory storage using SQLite.

    Stores recent conversations (7 days by default) with:
        - FTS5 full-text search
        - Dual timestamps for temporal reasoning
        - Extracted key facts
        - Scoring for decay decisions

    Usage:
        stm = ShortTermMemory()
        await stm.initialize()

        # Store a conversation
        entry = await stm.store(
            session_id="abc123",
            summary="Discussed climax scene structure",
            key_facts=["Boss wants more emotional punch"],
            raw_turns=[{"user": "...", "assistant": "..."}],
        )

        # Search
        results = await stm.search("climax scene")

        # Get by session
        entries = await stm.get_by_session("abc123")
    """

    def __init__(self, config: Optional[STMConfig] = None):
        self.config = config or get_memory_config().stm
        self._db_path = Path(self.config.db_path)
        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self) -> None:
        """Initialize database and create tables"""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Enable WAL mode for better concurrency
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._create_tables()
        LOGGER.info("STM initialized: %s", self._db_path)

    def _create_tables(self) -> None:
        """Create database tables"""
        with self._transaction() as cur:
            # Main table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS short_term_memories (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    key_facts TEXT,
                    raw_turns TEXT,
                    created_at TEXT NOT NULL,
                    event_dates TEXT,
                    room TEXT DEFAULT 'general',
                    project TEXT,
                    topics TEXT,
                    language TEXT DEFAULT 'mixed',
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    importance REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'active'
                )
            """
            )

            # Indexes
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stm_session
                ON short_term_memories(session_id)
            """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stm_created
                ON short_term_memories(created_at)
            """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stm_project
                ON short_term_memories(project)
            """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stm_status
                ON short_term_memories(status)
            """
            )

            # FTS5 virtual table for full-text search
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS stm_fts USING fts5(
                    id,
                    summary,
                    key_facts,
                    topics,
                    content=short_term_memories,
                    content_rowid=rowid,
                    tokenize='porter unicode61'
                )
            """
            )

            # Triggers to keep FTS in sync
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS stm_ai AFTER INSERT ON short_term_memories BEGIN
                    INSERT INTO stm_fts(rowid, id, summary, key_facts, topics)
                    VALUES (new.rowid, new.id, new.summary, new.key_facts, new.topics);
                END
            """
            )
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS stm_ad AFTER DELETE ON short_term_memories BEGIN
                    INSERT INTO stm_fts(stm_fts, rowid, id, summary, key_facts, topics)
                    VALUES ('delete', old.rowid, old.id, old.summary, old.key_facts, old.topics);
                END
            """
            )
            cur.execute(
                """
                CREATE TRIGGER IF NOT EXISTS stm_au AFTER UPDATE ON short_term_memories BEGIN
                    INSERT INTO stm_fts(stm_fts, rowid, id, summary, key_facts, topics)
                    VALUES ('delete', old.rowid, old.id, old.summary, old.key_facts, old.topics);
                    INSERT INTO stm_fts(rowid, id, summary, key_facts, topics)
                    VALUES (new.rowid, new.id, new.summary, new.key_facts, new.topics);
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
        LOGGER.info("STM closed")

    # =========================================================================
    # Storage Operations
    # =========================================================================

    async def store(
        self,
        session_id: str,
        summary: str,
        key_facts: Optional[List[str]] = None,
        raw_turns: Optional[List[Dict]] = None,
        event_dates: Optional[List[datetime]] = None,
        room: str = "general",
        project: Optional[str] = None,
        topics: Optional[List[str]] = None,
        language: str = "mixed",
        importance: float = 0.5,
    ) -> STMEntry:
        """
        Store a new short-term memory.

        Args:
            session_id: Conversation session identifier
            summary: Compressed summary of the conversation
            key_facts: Extracted atomic facts
            raw_turns: Original conversation turns
            event_dates: Referenced future/past event dates
            room: Context room (writers_room, kitchen, etc.)
            project: Associated project
            topics: Topic tags
            language: Dominant language (en, te, mixed)
            importance: Initial importance score (0-1)

        Returns:
            Created STMEntry
        """
        entry = STMEntry(
            id=str(uuid.uuid4()),
            session_id=session_id,
            summary=summary,
            key_facts=key_facts or [],
            raw_turns=raw_turns or [],
            created_at=datetime.now(),
            event_dates=event_dates or [],
            room=room,
            project=project,
            topics=topics or [],
            language=language,
            importance=importance,
        )

        with self._transaction() as cur:
            cur.execute(
                """
                INSERT INTO short_term_memories (
                    id, session_id, summary, key_facts, raw_turns,
                    created_at, event_dates, room, project, topics,
                    language, importance, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.id,
                    entry.session_id,
                    entry.summary,
                    json.dumps(entry.key_facts),
                    json.dumps(entry.raw_turns),
                    entry.created_at.isoformat(),
                    json.dumps([d.isoformat() for d in entry.event_dates]),
                    entry.room,
                    entry.project,
                    json.dumps(entry.topics),
                    entry.language,
                    entry.importance,
                    entry.status,
                ),
            )

        LOGGER.debug("Stored STM: %s (session: %s)", entry.id[:8], session_id[:8])
        return entry

    async def get(self, memory_id: str) -> Optional[STMEntry]:
        """Get a memory by ID"""
        with self._transaction() as cur:
            cur.execute("SELECT * FROM short_term_memories WHERE id = ?", (memory_id,))
            row = cur.fetchone()

        if not row:
            return None

        # Update access tracking
        await self._record_access(memory_id)

        return self._row_to_entry(row)

    async def get_by_session(self, session_id: str) -> List[STMEntry]:
        """Get all memories for a session"""
        with self._transaction() as cur:
            cur.execute(
                "SELECT * FROM short_term_memories WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            )
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def get_recent(
        self,
        days: int = 7,
        room: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 50,
    ) -> List[STMEntry]:
        """Get recent memories with optional filters"""
        cutoff = datetime.now() - timedelta(days=days)

        query = """
            SELECT * FROM short_term_memories
            WHERE created_at >= ? AND status = 'active'
        """
        params: List[Any] = [cutoff.isoformat()]

        if room:
            query += " AND room = ?"
            params.append(room)

        if project:
            query += " AND project = ?"
            params.append(project)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._transaction() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def update(self, memory_id: str, **kwargs) -> Optional[STMEntry]:
        """Update a memory's fields"""
        allowed = {"summary", "key_facts", "topics", "importance", "status", "project"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}

        if not updates:
            return await self.get(memory_id)

        # Serialize lists
        if "key_facts" in updates:
            updates["key_facts"] = json.dumps(updates["key_facts"])
        if "topics" in updates:
            updates["topics"] = json.dumps(updates["topics"])

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [memory_id]

        with self._transaction() as cur:
            cur.execute(
                f"UPDATE short_term_memories SET {set_clause} WHERE id = ?", values
            )

        LOGGER.debug("Updated STM: %s", memory_id[:8])
        return await self.get(memory_id)

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory"""
        with self._transaction() as cur:
            cur.execute("DELETE FROM short_term_memories WHERE id = ?", (memory_id,))
            deleted = cur.rowcount > 0

        if deleted:
            LOGGER.debug("Deleted STM: %s", memory_id[:8])
        return deleted

    async def archive(self, memory_id: str) -> Optional[STMEntry]:
        """Archive a memory (soft delete)"""
        return await self.update(memory_id, status="archived")

    async def mark_consolidated(self, memory_id: str) -> Optional[STMEntry]:
        """Mark memory as consolidated to LTM"""
        return await self.update(memory_id, status="consolidated")

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        top_k: int = 10,
        room: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[STMEntry]:
        """
        Search memories using FTS5 full-text search.

        Args:
            query: Search query (supports FTS5 syntax)
            top_k: Maximum results
            room: Filter by room
            project: Filter by project

        Returns:
            Matching STMEntry objects, ranked by relevance
        """
        # Build FTS query
        fts_query = f"""
            SELECT stm.* FROM short_term_memories stm
            JOIN stm_fts ON stm.id = stm_fts.id
            WHERE stm_fts MATCH ? AND stm.status = 'active'
        """
        params: List[Any] = [query]

        if room:
            fts_query += " AND stm.room = ?"
            params.append(room)

        if project:
            fts_query += " AND stm.project = ?"
            params.append(project)

        fts_query += " ORDER BY rank LIMIT ?"
        params.append(top_k)

        try:
            with self._transaction() as cur:
                cur.execute(fts_query, params)
                rows = cur.fetchall()

            results = [self._row_to_entry(row) for row in rows]

            # Record access for retrieved memories
            for entry in results:
                await self._record_access(entry.id)

            return results

        except sqlite3.OperationalError as e:
            LOGGER.warning("FTS search failed: %s, falling back to LIKE", e)
            return await self._fallback_search(query, top_k, room, project)

    async def _fallback_search(
        self,
        query: str,
        top_k: int,
        room: Optional[str],
        project: Optional[str],
    ) -> List[STMEntry]:
        """Fallback to LIKE-based search if FTS fails"""
        like_query = """
            SELECT * FROM short_term_memories
            WHERE status = 'active' AND (
                summary LIKE ? OR key_facts LIKE ? OR topics LIKE ?
            )
        """
        pattern = f"%{query}%"
        params: List[Any] = [pattern, pattern, pattern]

        if room:
            like_query += " AND room = ?"
            params.append(room)

        if project:
            like_query += " AND project = ?"
            params.append(project)

        like_query += " ORDER BY created_at DESC LIMIT ?"
        params.append(top_k)

        with self._transaction() as cur:
            cur.execute(like_query, params)
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def search_temporal(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        has_event_date: bool = False,
    ) -> List[STMEntry]:
        """Search by time range"""
        query = "SELECT * FROM short_term_memories WHERE status = 'active'"
        params: List[Any] = []

        if start:
            query += " AND created_at >= ?"
            params.append(start.isoformat())

        if end:
            query += " AND created_at <= ?"
            params.append(end.isoformat())

        if has_event_date:
            query += " AND event_dates != '[]'"

        query += " ORDER BY created_at DESC"

        with self._transaction() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    # =========================================================================
    # Maintenance Operations
    # =========================================================================

    async def get_consolidation_candidates(
        self,
        threshold: Optional[float] = None,
        max_age_days: Optional[int] = None,
    ) -> List[STMEntry]:
        """
        Get memories ready for consolidation to LTM.

        Candidates are memories that:
        - Are older than threshold (or importance below threshold)
        - Have been accessed multiple times
        - Are still active (not already consolidated)
        """
        threshold = threshold or self.config.consolidation_threshold
        max_age = max_age_days or self.config.retention_days

        cutoff = datetime.now() - timedelta(days=max_age)

        with self._transaction() as cur:
            cur.execute(
                """
                SELECT * FROM short_term_memories
                WHERE status = 'active'
                AND (created_at < ? OR importance < ?)
                ORDER BY created_at ASC
            """,
                (cutoff.isoformat(), threshold),
            )
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def get_decay_candidates(self, threshold: float = 0.2) -> List[STMEntry]:
        """Get memories that should be decayed (archived/deleted)"""
        with self._transaction() as cur:
            cur.execute(
                """
                SELECT * FROM short_term_memories
                WHERE status = 'active' AND importance < ?
                ORDER BY importance ASC
            """,
                (threshold,),
            )
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    async def cleanup_old(self, days: Optional[int] = None) -> int:
        """Delete memories older than retention period"""
        days = days or self.config.retention_days
        cutoff = datetime.now() - timedelta(days=days)

        with self._transaction() as cur:
            cur.execute(
                """
                DELETE FROM short_term_memories
                WHERE status IN ('archived', 'consolidated')
                AND created_at < ?
            """,
                (cutoff.isoformat(),),
            )
            deleted = cur.rowcount

        if deleted:
            LOGGER.info("Cleaned up %d old STM entries", deleted)
        return deleted

    async def get_stats(self) -> Dict[str, Any]:
        """Get STM statistics"""
        with self._transaction() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM short_term_memories WHERE status = 'active'"
            )
            active = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM short_term_memories WHERE status = 'archived'"
            )
            archived = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM short_term_memories WHERE status = 'consolidated'"
            )
            consolidated = cur.fetchone()[0]

            cur.execute(
                "SELECT AVG(importance) FROM short_term_memories WHERE status = 'active'"
            )
            avg_importance = cur.fetchone()[0] or 0

            cur.execute("SELECT COUNT(DISTINCT session_id) FROM short_term_memories")
            sessions = cur.fetchone()[0]

        return {
            "active": active,
            "archived": archived,
            "consolidated": consolidated,
            "total": active + archived + consolidated,
            "avg_importance": round(avg_importance, 3),
            "sessions": sessions,
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _record_access(self, memory_id: str) -> None:
        """Record that a memory was accessed"""
        with self._transaction() as cur:
            cur.execute(
                """
                UPDATE short_term_memories
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """,
                (datetime.now().isoformat(), memory_id),
            )

    async def mark_accessed(self, memory_id: str) -> None:
        """Public method to record access (for decay daemon)"""
        await self._record_access(memory_id)

    async def update_importance(self, memory_id: str, new_importance: float) -> bool:
        """Update a memory's importance score"""
        with self._transaction() as cur:
            cur.execute(
                """
                UPDATE short_term_memories
                SET importance = ?
                WHERE id = ?
            """,
                (new_importance, memory_id),
            )
            return cur.rowcount > 0

    async def get_entries_before(
        self,
        cutoff: datetime,
        status: str = "active",
    ) -> List[STMEntry]:
        """
        Get all entries created before a cutoff time.

        Used by decay daemon to find entries for processing.

        Args:
            cutoff: Only return entries created before this time
            status: Filter by status (default: active)

        Returns:
            List of STMEntry objects
        """
        with self._transaction() as cur:
            cur.execute(
                """
                SELECT * FROM short_term_memories
                WHERE created_at < ? AND status = ?
                ORDER BY created_at ASC
            """,
                (cutoff.isoformat(), status),
            )
            rows = cur.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def _row_to_entry(self, row: sqlite3.Row) -> STMEntry:
        """Convert database row to STMEntry"""
        return STMEntry(
            id=row["id"],
            session_id=row["session_id"],
            summary=row["summary"],
            key_facts=json.loads(row["key_facts"]) if row["key_facts"] else [],
            raw_turns=json.loads(row["raw_turns"]) if row["raw_turns"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
            event_dates=[
                datetime.fromisoformat(d)
                for d in json.loads(row["event_dates"] or "[]")
            ],
            room=row["room"],
            project=row["project"],
            topics=json.loads(row["topics"]) if row["topics"] else [],
            language=row["language"],
            access_count=row["access_count"],
            last_accessed=(
                datetime.fromisoformat(row["last_accessed"])
                if row["last_accessed"]
                else None
            ),
            importance=row["importance"],
            status=row["status"],
        )

    def __repr__(self) -> str:
        return f"ShortTermMemory(db={self._db_path})"
