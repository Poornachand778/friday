"""
Book Understanding Storage Layer
================================

SQLite storage for BookUnderstanding and extracted knowledge.

This stores the comprehension results - concepts, principles, techniques,
and examples - separately from the raw document data.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from documents.config import StorageConfig, get_document_config
from documents.understanding.models import (
    BookUnderstanding,
    ChapterSummary,
    Concept,
    Principle,
    Technique,
    BookExample,
    ConfidenceLevel,
)

LOGGER = logging.getLogger(__name__)


class BookUnderstandingStore:
    """
    SQLite storage for book understanding and extracted knowledge.

    Stores:
    - BookUnderstanding (summary, thesis, chapters)
    - Concepts
    - Principles
    - Techniques
    - Examples

    All linked back to the source document.
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or get_document_config().storage
        # Use same database as document store for consistency
        self._db_path = Path(self.config.db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize database and create understanding tables"""
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
        LOGGER.info("BookUnderstandingStore initialized at %s", self._db_path)

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
        """Create understanding tables"""
        with self._transaction() as cur:
            # Book Understanding (main comprehension record)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS book_understandings (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    author TEXT,
                    summary TEXT,
                    main_argument TEXT,
                    target_audience TEXT,
                    chapters JSON,
                    domains JSON,
                    agrees_with JSON,
                    disagrees_with JSON,
                    extends JSON,
                    comprehension_quality REAL DEFAULT 0.0,
                    study_completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
            """
            )

            # Concepts table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS book_concepts (
                    id TEXT PRIMARY KEY,
                    understanding_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    definition TEXT,
                    importance TEXT,
                    source_pages TEXT,
                    related_concepts JSON,
                    parent_concept TEXT,
                    sub_concepts JSON,
                    synonyms JSON,
                    domain TEXT DEFAULT 'screenwriting',
                    confidence REAL DEFAULT 0.9,
                    extracted_at TEXT NOT NULL,
                    FOREIGN KEY (understanding_id) REFERENCES book_understandings(id) ON DELETE CASCADE
                )
            """
            )

            # Principles table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS book_principles (
                    id TEXT PRIMARY KEY,
                    understanding_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    rationale TEXT,
                    source_page TEXT,
                    confidence_level TEXT DEFAULT 'strong',
                    applies_to JSON,
                    exceptions JSON,
                    prerequisites JSON,
                    related_concepts JSON,
                    related_techniques JSON,
                    checkable BOOLEAN DEFAULT TRUE,
                    check_question TEXT,
                    extracted_at TEXT NOT NULL,
                    FOREIGN KEY (understanding_id) REFERENCES book_understandings(id) ON DELETE CASCADE
                )
            """
            )

            # Techniques table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS book_techniques (
                    id TEXT PRIMARY KEY,
                    understanding_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    steps JSON,
                    source_page TEXT,
                    use_cases JSON,
                    when_to_use TEXT,
                    when_not_to_use TEXT,
                    example_films JSON,
                    example_description TEXT,
                    related_concepts JSON,
                    related_principles JSON,
                    alternative_techniques JSON,
                    difficulty TEXT DEFAULT 'intermediate',
                    extracted_at TEXT NOT NULL,
                    FOREIGN KEY (understanding_id) REFERENCES book_understandings(id) ON DELETE CASCADE
                )
            """
            )

            # Examples table
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS book_examples (
                    id TEXT PRIMARY KEY,
                    understanding_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    work_title TEXT NOT NULL,
                    work_type TEXT DEFAULT 'film',
                    scene_or_section TEXT,
                    source_page TEXT,
                    description TEXT,
                    lesson TEXT,
                    what_works TEXT,
                    demonstrates_concept JSON,
                    demonstrates_technique JSON,
                    demonstrates_principle JSON,
                    situation_type JSON,
                    emotional_beat TEXT,
                    extracted_at TEXT NOT NULL,
                    FOREIGN KEY (understanding_id) REFERENCES book_understandings(id) ON DELETE CASCADE
                )
            """
            )

            # Indexes
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_understanding_doc ON book_understandings(document_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_concepts_understanding ON book_concepts(understanding_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_concepts_name ON book_concepts(name)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_principles_understanding ON book_principles(understanding_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_techniques_understanding ON book_techniques(understanding_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_techniques_name ON book_techniques(name)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_examples_understanding ON book_examples(understanding_id)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_examples_work ON book_examples(work_title)"
            )

            # FTS for searching across all knowledge
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    id,
                    type,
                    content,
                    tokenize='porter unicode61'
                )
            """
            )

    # ========== BookUnderstanding Operations ==========

    def store_understanding(self, understanding: BookUnderstanding) -> None:
        """Store a complete book understanding with all extracted knowledge"""
        now = datetime.now().isoformat()

        with self._transaction() as cur:
            # Store main understanding record
            cur.execute(
                """
                INSERT OR REPLACE INTO book_understandings
                (id, document_id, title, author, summary, main_argument,
                 target_audience, chapters, domains, agrees_with, disagrees_with,
                 extends, comprehension_quality, study_completed_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    understanding.id,
                    understanding.document_id,
                    understanding.title,
                    understanding.author,
                    understanding.summary,
                    understanding.main_argument,
                    understanding.target_audience,
                    json.dumps([c.to_dict() for c in understanding.chapters]),
                    json.dumps(understanding.domains),
                    json.dumps(understanding.agrees_with),
                    json.dumps(understanding.disagrees_with),
                    json.dumps(understanding.extends),
                    understanding.comprehension_quality,
                    (
                        understanding.study_completed_at.isoformat()
                        if understanding.study_completed_at
                        else None
                    ),
                    now,
                    now,
                ),
            )

            # Store concepts
            for concept in understanding.concepts:
                self._store_concept(cur, understanding.id, concept)

            # Store principles
            for principle in understanding.principles:
                self._store_principle(cur, understanding.id, principle)

            # Store techniques
            for technique in understanding.techniques:
                self._store_technique(cur, understanding.id, technique)

            # Store examples
            for example in understanding.examples:
                self._store_example(cur, understanding.id, example)

        # Update FTS index
        self._update_fts_index(understanding)

        LOGGER.info(
            "Stored understanding for '%s': %d concepts, %d principles, %d techniques, %d examples",
            understanding.title,
            len(understanding.concepts),
            len(understanding.principles),
            len(understanding.techniques),
            len(understanding.examples),
        )

    def _store_concept(
        self, cur: sqlite3.Cursor, understanding_id: str, concept: Concept
    ) -> None:
        """Store a concept"""
        cur.execute(
            """
            INSERT OR REPLACE INTO book_concepts
            (id, understanding_id, document_id, name, definition, importance,
             source_pages, related_concepts, parent_concept, sub_concepts,
             synonyms, domain, confidence, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                concept.id,
                understanding_id,
                concept.source_document_id,
                concept.name,
                concept.definition,
                concept.importance,
                concept.source_pages,
                json.dumps(concept.related_concepts),
                concept.parent_concept,
                json.dumps(concept.sub_concepts),
                json.dumps(concept.synonyms),
                concept.domain,
                concept.confidence,
                concept.extracted_at.isoformat(),
            ),
        )

    def _store_principle(
        self, cur: sqlite3.Cursor, understanding_id: str, principle: Principle
    ) -> None:
        """Store a principle"""
        cur.execute(
            """
            INSERT OR REPLACE INTO book_principles
            (id, understanding_id, document_id, statement, rationale, source_page,
             confidence_level, applies_to, exceptions, prerequisites,
             related_concepts, related_techniques, checkable, check_question, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                principle.id,
                understanding_id,
                principle.source_document_id,
                principle.statement,
                principle.rationale,
                principle.source_page,
                principle.confidence_level.value,
                json.dumps(principle.applies_to),
                json.dumps(principle.exceptions),
                json.dumps(principle.prerequisites),
                json.dumps(principle.related_concepts),
                json.dumps(principle.related_techniques),
                principle.checkable,
                principle.check_question,
                principle.extracted_at.isoformat(),
            ),
        )

    def _store_technique(
        self, cur: sqlite3.Cursor, understanding_id: str, technique: Technique
    ) -> None:
        """Store a technique"""
        cur.execute(
            """
            INSERT OR REPLACE INTO book_techniques
            (id, understanding_id, document_id, name, description, steps,
             source_page, use_cases, when_to_use, when_not_to_use,
             example_films, example_description, related_concepts,
             related_principles, alternative_techniques, difficulty, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                technique.id,
                understanding_id,
                technique.source_document_id,
                technique.name,
                technique.description,
                json.dumps(technique.steps),
                technique.source_page,
                json.dumps(technique.use_cases),
                technique.when_to_use,
                technique.when_not_to_use,
                json.dumps(technique.example_films),
                technique.example_description,
                json.dumps(technique.related_concepts),
                json.dumps(technique.related_principles),
                json.dumps(technique.alternative_techniques),
                technique.difficulty,
                technique.extracted_at.isoformat(),
            ),
        )

    def _store_example(
        self, cur: sqlite3.Cursor, understanding_id: str, example: BookExample
    ) -> None:
        """Store an example"""
        cur.execute(
            """
            INSERT OR REPLACE INTO book_examples
            (id, understanding_id, document_id, work_title, work_type,
             scene_or_section, source_page, description, lesson, what_works,
             demonstrates_concept, demonstrates_technique, demonstrates_principle,
             situation_type, emotional_beat, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                example.id,
                understanding_id,
                example.source_document_id,
                example.work_title,
                example.work_type,
                example.scene_or_section,
                example.source_page,
                example.description,
                example.lesson,
                example.what_works,
                json.dumps(example.demonstrates_concept),
                json.dumps(example.demonstrates_technique),
                json.dumps(example.demonstrates_principle),
                json.dumps(example.situation_type),
                example.emotional_beat,
                example.extracted_at.isoformat(),
            ),
        )

    def _update_fts_index(self, understanding: BookUnderstanding) -> None:
        """Update FTS index with all knowledge from understanding"""
        with self._transaction() as cur:
            # Delete existing entries for this understanding
            cur.execute(
                "DELETE FROM knowledge_fts WHERE id LIKE ?",
                (f"{understanding.id}%",),
            )

            # Add concepts
            for concept in understanding.concepts:
                cur.execute(
                    "INSERT INTO knowledge_fts (id, type, content) VALUES (?, ?, ?)",
                    (
                        concept.id,
                        "concept",
                        f"{concept.name} {concept.definition} {concept.importance}",
                    ),
                )

            # Add principles
            for principle in understanding.principles:
                cur.execute(
                    "INSERT INTO knowledge_fts (id, type, content) VALUES (?, ?, ?)",
                    (
                        principle.id,
                        "principle",
                        f"{principle.statement} {principle.rationale}",
                    ),
                )

            # Add techniques
            for technique in understanding.techniques:
                cur.execute(
                    "INSERT INTO knowledge_fts (id, type, content) VALUES (?, ?, ?)",
                    (
                        technique.id,
                        "technique",
                        f"{technique.name} {technique.description} {technique.when_to_use}",
                    ),
                )

            # Add examples
            for example in understanding.examples:
                cur.execute(
                    "INSERT INTO knowledge_fts (id, type, content) VALUES (?, ?, ?)",
                    (
                        example.id,
                        "example",
                        f"{example.work_title} {example.description} {example.lesson}",
                    ),
                )

    def get_understanding(self, understanding_id: str) -> Optional[BookUnderstanding]:
        """Get a book understanding by ID"""
        with self._transaction() as cur:
            cur.execute(
                "SELECT * FROM book_understandings WHERE id = ?", (understanding_id,)
            )
            row = cur.fetchone()
            if not row:
                return None

            understanding = self._row_to_understanding(row)

            # Load related knowledge
            understanding.concepts = self._get_concepts(cur, understanding_id)
            understanding.principles = self._get_principles(cur, understanding_id)
            understanding.techniques = self._get_techniques(cur, understanding_id)
            understanding.examples = self._get_examples(cur, understanding_id)

            return understanding

    def get_understanding_by_document(
        self, document_id: str
    ) -> Optional[BookUnderstanding]:
        """Get understanding for a document"""
        with self._transaction() as cur:
            cur.execute(
                "SELECT * FROM book_understandings WHERE document_id = ?",
                (document_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            understanding = self._row_to_understanding(row)

            # Load related knowledge
            understanding.concepts = self._get_concepts(cur, understanding.id)
            understanding.principles = self._get_principles(cur, understanding.id)
            understanding.techniques = self._get_techniques(cur, understanding.id)
            understanding.examples = self._get_examples(cur, understanding.id)

            return understanding

    def list_understandings(
        self,
        domain: Optional[str] = None,
        limit: int = 100,
    ) -> List[BookUnderstanding]:
        """List all book understandings"""
        query = "SELECT * FROM book_understandings WHERE 1=1"
        params: List[Any] = []

        if domain:
            query += " AND domains LIKE ?"
            params.append(f'%"{domain}"%')

        query += " ORDER BY study_completed_at DESC LIMIT ?"
        params.append(limit)

        understandings = []
        with self._transaction() as cur:
            cur.execute(query, params)
            for row in cur.fetchall():
                understanding = self._row_to_understanding(row)
                # Load counts only for list view
                cur.execute(
                    "SELECT COUNT(*) FROM book_concepts WHERE understanding_id = ?",
                    (understanding.id,),
                )
                concept_count = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(*) FROM book_principles WHERE understanding_id = ?",
                    (understanding.id,),
                )
                principle_count = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(*) FROM book_techniques WHERE understanding_id = ?",
                    (understanding.id,),
                )
                technique_count = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(*) FROM book_examples WHERE understanding_id = ?",
                    (understanding.id,),
                )
                example_count = cur.fetchone()[0]

                # Set counts without loading full objects
                understanding.concepts = [None] * concept_count  # type: ignore
                understanding.principles = [None] * principle_count  # type: ignore
                understanding.techniques = [None] * technique_count  # type: ignore
                understanding.examples = [None] * example_count  # type: ignore

                understandings.append(understanding)

        return understandings

    def delete_understanding(self, understanding_id: str) -> bool:
        """Delete a book understanding and all related knowledge"""
        with self._transaction() as cur:
            # FTS cleanup
            cur.execute(
                "DELETE FROM knowledge_fts WHERE id IN (SELECT id FROM book_concepts WHERE understanding_id = ?)",
                (understanding_id,),
            )
            cur.execute(
                "DELETE FROM knowledge_fts WHERE id IN (SELECT id FROM book_principles WHERE understanding_id = ?)",
                (understanding_id,),
            )
            cur.execute(
                "DELETE FROM knowledge_fts WHERE id IN (SELECT id FROM book_techniques WHERE understanding_id = ?)",
                (understanding_id,),
            )
            cur.execute(
                "DELETE FROM knowledge_fts WHERE id IN (SELECT id FROM book_examples WHERE understanding_id = ?)",
                (understanding_id,),
            )

            # Main delete (cascades to knowledge tables)
            cur.execute(
                "DELETE FROM book_understandings WHERE id = ?", (understanding_id,)
            )
            return cur.rowcount > 0

    def _row_to_understanding(self, row: sqlite3.Row) -> BookUnderstanding:
        """Convert row to BookUnderstanding"""
        chapters_data = json.loads(row["chapters"]) if row["chapters"] else []
        chapters = []
        for c in chapters_data:
            chapters.append(
                ChapterSummary(
                    number=c.get("number", 0),
                    title=c.get("title", ""),
                    summary=c.get("summary", ""),
                    key_points=c.get("key_points", []),
                    concepts_introduced=c.get("concepts_introduced", []),
                    principles_taught=c.get("principles_taught", []),
                    page_range=c.get("page_range", ""),
                )
            )

        return BookUnderstanding(
            id=row["id"],
            document_id=row["document_id"],
            title=row["title"],
            author=row["author"] or "",
            summary=row["summary"] or "",
            main_argument=row["main_argument"] or "",
            target_audience=row["target_audience"] or "",
            chapters=chapters,
            domains=json.loads(row["domains"]) if row["domains"] else [],
            agrees_with=json.loads(row["agrees_with"]) if row["agrees_with"] else {},
            disagrees_with=(
                json.loads(row["disagrees_with"]) if row["disagrees_with"] else {}
            ),
            extends=json.loads(row["extends"]) if row["extends"] else {},
            comprehension_quality=row["comprehension_quality"],
            study_completed_at=(
                datetime.fromisoformat(row["study_completed_at"])
                if row["study_completed_at"]
                else None
            ),
        )

    def _get_concepts(
        self, cur: sqlite3.Cursor, understanding_id: str
    ) -> List[Concept]:
        """Get concepts for an understanding"""
        cur.execute(
            "SELECT * FROM book_concepts WHERE understanding_id = ?",
            (understanding_id,),
        )
        concepts = []
        for row in cur.fetchall():
            concepts.append(
                Concept(
                    id=row["id"],
                    name=row["name"],
                    definition=row["definition"] or "",
                    importance=row["importance"] or "",
                    source_document_id=row["document_id"],
                    source_pages=row["source_pages"] or "",
                    related_concepts=(
                        json.loads(row["related_concepts"])
                        if row["related_concepts"]
                        else []
                    ),
                    parent_concept=row["parent_concept"],
                    sub_concepts=(
                        json.loads(row["sub_concepts"]) if row["sub_concepts"] else []
                    ),
                    synonyms=json.loads(row["synonyms"]) if row["synonyms"] else [],
                    domain=row["domain"],
                    confidence=row["confidence"],
                    extracted_at=datetime.fromisoformat(row["extracted_at"]),
                )
            )
        return concepts

    def _get_principles(
        self, cur: sqlite3.Cursor, understanding_id: str
    ) -> List[Principle]:
        """Get principles for an understanding"""
        cur.execute(
            "SELECT * FROM book_principles WHERE understanding_id = ?",
            (understanding_id,),
        )
        principles = []
        for row in cur.fetchall():
            confidence_level = ConfidenceLevel.STRONG
            if row["confidence_level"]:
                try:
                    confidence_level = ConfidenceLevel(row["confidence_level"])
                except ValueError:
                    pass

            principles.append(
                Principle(
                    id=row["id"],
                    statement=row["statement"],
                    rationale=row["rationale"] or "",
                    source_document_id=row["document_id"],
                    source_page=row["source_page"] or "",
                    confidence_level=confidence_level,
                    applies_to=(
                        json.loads(row["applies_to"]) if row["applies_to"] else []
                    ),
                    exceptions=(
                        json.loads(row["exceptions"]) if row["exceptions"] else []
                    ),
                    prerequisites=(
                        json.loads(row["prerequisites"]) if row["prerequisites"] else []
                    ),
                    related_concepts=(
                        json.loads(row["related_concepts"])
                        if row["related_concepts"]
                        else []
                    ),
                    related_techniques=(
                        json.loads(row["related_techniques"])
                        if row["related_techniques"]
                        else []
                    ),
                    checkable=bool(row["checkable"]),
                    check_question=row["check_question"] or "",
                    extracted_at=datetime.fromisoformat(row["extracted_at"]),
                )
            )
        return principles

    def _get_techniques(
        self, cur: sqlite3.Cursor, understanding_id: str
    ) -> List[Technique]:
        """Get techniques for an understanding"""
        cur.execute(
            "SELECT * FROM book_techniques WHERE understanding_id = ?",
            (understanding_id,),
        )
        techniques = []
        for row in cur.fetchall():
            techniques.append(
                Technique(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"] or "",
                    steps=json.loads(row["steps"]) if row["steps"] else [],
                    source_document_id=row["document_id"],
                    source_page=row["source_page"] or "",
                    use_cases=json.loads(row["use_cases"]) if row["use_cases"] else [],
                    when_to_use=row["when_to_use"] or "",
                    when_not_to_use=row["when_not_to_use"] or "",
                    example_films=(
                        json.loads(row["example_films"]) if row["example_films"] else []
                    ),
                    example_description=row["example_description"] or "",
                    related_concepts=(
                        json.loads(row["related_concepts"])
                        if row["related_concepts"]
                        else []
                    ),
                    related_principles=(
                        json.loads(row["related_principles"])
                        if row["related_principles"]
                        else []
                    ),
                    alternative_techniques=(
                        json.loads(row["alternative_techniques"])
                        if row["alternative_techniques"]
                        else []
                    ),
                    difficulty=row["difficulty"],
                    extracted_at=datetime.fromisoformat(row["extracted_at"]),
                )
            )
        return techniques

    def _get_examples(
        self, cur: sqlite3.Cursor, understanding_id: str
    ) -> List[BookExample]:
        """Get examples for an understanding"""
        cur.execute(
            "SELECT * FROM book_examples WHERE understanding_id = ?",
            (understanding_id,),
        )
        examples = []
        for row in cur.fetchall():
            examples.append(
                BookExample(
                    id=row["id"],
                    work_title=row["work_title"],
                    work_type=row["work_type"],
                    scene_or_section=row["scene_or_section"] or "",
                    source_document_id=row["document_id"],
                    source_page=row["source_page"] or "",
                    description=row["description"] or "",
                    lesson=row["lesson"] or "",
                    what_works=row["what_works"] or "",
                    demonstrates_concept=(
                        json.loads(row["demonstrates_concept"])
                        if row["demonstrates_concept"]
                        else []
                    ),
                    demonstrates_technique=(
                        json.loads(row["demonstrates_technique"])
                        if row["demonstrates_technique"]
                        else []
                    ),
                    demonstrates_principle=(
                        json.loads(row["demonstrates_principle"])
                        if row["demonstrates_principle"]
                        else []
                    ),
                    situation_type=(
                        json.loads(row["situation_type"])
                        if row["situation_type"]
                        else []
                    ),
                    emotional_beat=row["emotional_beat"] or "",
                    extracted_at=datetime.fromisoformat(row["extracted_at"]),
                )
            )
        return examples

    # ========== Search Operations ==========

    def search_knowledge(
        self,
        query: str,
        knowledge_type: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Search across all knowledge using FTS.

        Args:
            query: Search query
            knowledge_type: Filter by type (concept, principle, technique, example)
            top_k: Max results

        Returns:
            List of dicts with {id, type, content, score}
        """
        safe_query = query.replace('"', '""')

        sql = """
            SELECT id, type, content, bm25(knowledge_fts) as score
            FROM knowledge_fts
            WHERE knowledge_fts MATCH ?
        """
        params: List[Any] = [safe_query]

        if knowledge_type:
            sql += " AND type = ?"
            params.append(knowledge_type)

        sql += " ORDER BY score LIMIT ?"
        params.append(top_k)

        results = []
        with self._transaction() as cur:
            try:
                cur.execute(sql, params)
                for row in cur.fetchall():
                    results.append(
                        {
                            "id": row["id"],
                            "type": row["type"],
                            "content": row["content"],
                            "score": abs(row["score"]),
                        }
                    )
            except sqlite3.OperationalError as e:
                LOGGER.warning("FTS search failed: %s", e)

        return results

    def find_concepts_by_name(self, name: str) -> List[Concept]:
        """Find concepts by name (partial match)"""
        with self._transaction() as cur:
            cur.execute(
                "SELECT * FROM book_concepts WHERE name LIKE ?",
                (f"%{name}%",),
            )
            return [self._row_to_concept(row) for row in cur.fetchall()]

    def find_techniques_by_name(self, name: str) -> List[Technique]:
        """Find techniques by name (partial match)"""
        with self._transaction() as cur:
            cur.execute(
                "SELECT * FROM book_techniques WHERE name LIKE ?",
                (f"%{name}%",),
            )
            return [self._row_to_technique(row) for row in cur.fetchall()]

    def find_examples_by_work(self, work_title: str) -> List[BookExample]:
        """Find examples by film/work title"""
        with self._transaction() as cur:
            cur.execute(
                "SELECT * FROM book_examples WHERE work_title LIKE ?",
                (f"%{work_title}%",),
            )
            return [self._row_to_example(row) for row in cur.fetchall()]

    def _row_to_concept(self, row: sqlite3.Row) -> Concept:
        """Convert single row to Concept"""
        return Concept(
            id=row["id"],
            name=row["name"],
            definition=row["definition"] or "",
            importance=row["importance"] or "",
            source_document_id=row["document_id"],
            source_pages=row["source_pages"] or "",
            related_concepts=(
                json.loads(row["related_concepts"]) if row["related_concepts"] else []
            ),
            parent_concept=row["parent_concept"],
            sub_concepts=json.loads(row["sub_concepts"]) if row["sub_concepts"] else [],
            synonyms=json.loads(row["synonyms"]) if row["synonyms"] else [],
            domain=row["domain"],
            confidence=row["confidence"],
            extracted_at=datetime.fromisoformat(row["extracted_at"]),
        )

    def _row_to_technique(self, row: sqlite3.Row) -> Technique:
        """Convert single row to Technique"""
        return Technique(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            steps=json.loads(row["steps"]) if row["steps"] else [],
            source_document_id=row["document_id"],
            source_page=row["source_page"] or "",
            use_cases=json.loads(row["use_cases"]) if row["use_cases"] else [],
            when_to_use=row["when_to_use"] or "",
            when_not_to_use=row["when_not_to_use"] or "",
            example_films=(
                json.loads(row["example_films"]) if row["example_films"] else []
            ),
            example_description=row["example_description"] or "",
            related_concepts=(
                json.loads(row["related_concepts"]) if row["related_concepts"] else []
            ),
            related_principles=(
                json.loads(row["related_principles"])
                if row["related_principles"]
                else []
            ),
            alternative_techniques=(
                json.loads(row["alternative_techniques"])
                if row["alternative_techniques"]
                else []
            ),
            difficulty=row["difficulty"],
            extracted_at=datetime.fromisoformat(row["extracted_at"]),
        )

    def _row_to_example(self, row: sqlite3.Row) -> BookExample:
        """Convert single row to BookExample"""
        return BookExample(
            id=row["id"],
            work_title=row["work_title"],
            work_type=row["work_type"],
            scene_or_section=row["scene_or_section"] or "",
            source_document_id=row["document_id"],
            source_page=row["source_page"] or "",
            description=row["description"] or "",
            lesson=row["lesson"] or "",
            what_works=row["what_works"] or "",
            demonstrates_concept=(
                json.loads(row["demonstrates_concept"])
                if row["demonstrates_concept"]
                else []
            ),
            demonstrates_technique=(
                json.loads(row["demonstrates_technique"])
                if row["demonstrates_technique"]
                else []
            ),
            demonstrates_principle=(
                json.loads(row["demonstrates_principle"])
                if row["demonstrates_principle"]
                else []
            ),
            situation_type=(
                json.loads(row["situation_type"]) if row["situation_type"] else []
            ),
            emotional_beat=row["emotional_beat"] or "",
            extracted_at=datetime.fromisoformat(row["extracted_at"]),
        )

    # ========== Statistics ==========

    def get_stats(self) -> Dict[str, Any]:
        """Get understanding storage statistics"""
        with self._transaction() as cur:
            cur.execute("SELECT COUNT(*) FROM book_understandings")
            understanding_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM book_concepts")
            concept_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM book_principles")
            principle_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM book_techniques")
            technique_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM book_examples")
            example_count = cur.fetchone()[0]

            # Get domain distribution
            cur.execute("SELECT domains FROM book_understandings")
            domain_counts: Dict[str, int] = {}
            for row in cur.fetchall():
                domains = json.loads(row["domains"]) if row["domains"] else []
                for domain in domains:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1

        return {
            "total_understandings": understanding_count,
            "total_concepts": concept_count,
            "total_principles": principle_count,
            "total_techniques": technique_count,
            "total_examples": example_count,
            "total_knowledge_items": concept_count
            + principle_count
            + technique_count
            + example_count,
            "domain_distribution": domain_counts,
            "db_path": str(self._db_path),
        }
