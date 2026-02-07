"""
Tests for VectorStore abstraction
=================================

Tests the VectorStore interface, config, fallback behavior,
and integration with DocumentSearcher and LongTermMemory.

Run with: pytest tests/test_vector_store.py -v
"""

import asyncio
import json
import os
import pickle
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from db.vector_store import (
    VectorStore,
    VectorStoreConfig,
    VectorSearchResult,
    QdrantVectorStore,
    get_vector_store,
    reset_vector_store,
    VECTOR_DIM,
    COLLECTION_DOCUMENTS,
    COLLECTION_MEMORIES,
    COLLECTION_SCENES,
)


class TestVectorStoreConfig:
    """Test VectorStoreConfig creation"""

    def test_default_config(self):
        config = VectorStoreConfig()
        assert config.host == "localhost"
        assert config.port == 6333
        assert config.grpc_port == 6334
        assert config.vector_dim == 768
        assert config.distance_metric == "cosine"
        assert config.api_key is None

    def test_from_env_defaults(self):
        """Test from_env with no env vars set"""
        with patch.dict(os.environ, {}, clear=True):
            config = VectorStoreConfig.from_env()
            assert config.host == "localhost"
            assert config.port == 6333

    def test_from_env_with_host_port(self):
        """Test from_env with QDRANT_HOST and QDRANT_PORT"""
        with patch.dict(
            os.environ,
            {"QDRANT_HOST": "qdrant-server", "QDRANT_PORT": "7333"},
            clear=True,
        ):
            config = VectorStoreConfig.from_env()
            assert config.host == "qdrant-server"
            assert config.port == 7333

    def test_from_env_with_url(self):
        """Test from_env with QDRANT_URL (Docker compose style)"""
        with patch.dict(
            os.environ,
            {"QDRANT_URL": "http://qdrant:6333"},
            clear=True,
        ):
            config = VectorStoreConfig.from_env()
            assert config.host == "qdrant"
            assert config.port == 6333

    def test_from_env_url_with_custom_port(self):
        """Test URL parsing with custom port"""
        with patch.dict(
            os.environ,
            {"QDRANT_URL": "http://my-qdrant:9333"},
            clear=True,
        ):
            config = VectorStoreConfig.from_env()
            assert config.host == "my-qdrant"
            assert config.port == 9333

    def test_from_env_api_key(self):
        """Test API key from env"""
        with patch.dict(
            os.environ,
            {"QDRANT_API_KEY": "my-secret-key"},
            clear=True,
        ):
            config = VectorStoreConfig.from_env()
            assert config.api_key == "my-secret-key"

    def test_hnsw_defaults(self):
        """Test HNSW index parameters"""
        config = VectorStoreConfig()
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construct == 100
        assert config.hnsw_ef_search == 128


class TestVectorSearchResult:
    """Test VectorSearchResult dataclass"""

    def test_basic_result(self):
        result = VectorSearchResult(id="test-id", score=0.95)
        assert result.id == "test-id"
        assert result.score == 0.95
        assert result.payload == {}

    def test_result_with_payload(self):
        result = VectorSearchResult(
            id="test-id",
            score=0.85,
            payload={"document_id": "doc-1", "chapter": "Introduction"},
        )
        assert result.payload["document_id"] == "doc-1"
        assert result.payload["chapter"] == "Introduction"


class TestCollectionNames:
    """Test collection name constants"""

    def test_collection_names(self):
        assert COLLECTION_DOCUMENTS == "friday_documents"
        assert COLLECTION_MEMORIES == "friday_memories"
        assert COLLECTION_SCENES == "friday_scenes"

    def test_vector_dim(self):
        assert VECTOR_DIM == 768


class TestQdrantVectorStore:
    """Test QdrantVectorStore initialization"""

    def test_init_default_config(self):
        store = QdrantVectorStore()
        assert store.config.host == "localhost"
        assert store._client is None
        assert store._async_client is None

    def test_init_custom_config(self):
        config = VectorStoreConfig(host="custom-host", port=9333)
        store = QdrantVectorStore(config)
        assert store.config.host == "custom-host"
        assert store.config.port == 9333

    def test_lazy_client_init(self):
        """Clients should not be created until first use"""
        store = QdrantVectorStore()
        assert store._client is None
        assert store._async_client is None


class TestGetVectorStore:
    """Test the singleton factory"""

    def setup_method(self):
        reset_vector_store()

    def test_returns_none_when_qdrant_unavailable(self):
        """Should return None gracefully when Qdrant is not running"""
        result = asyncio.get_event_loop().run_until_complete(get_vector_store())
        # Will be None since Qdrant isn't running in test environment
        assert result is None

    def test_singleton_caching(self):
        """Should cache the result after first call"""
        r1 = asyncio.get_event_loop().run_until_complete(get_vector_store())
        r2 = asyncio.get_event_loop().run_until_complete(get_vector_store())
        assert r1 is r2

    def test_reset_clears_cache(self):
        """Reset should allow re-initialization"""
        asyncio.get_event_loop().run_until_complete(get_vector_store())
        reset_vector_store()
        # After reset, it should try to connect again
        result = asyncio.get_event_loop().run_until_complete(get_vector_store())
        assert result is None  # Still None without Qdrant


class TestVectorStoreInterface:
    """Verify the abstract interface is correct"""

    def test_is_abstract(self):
        """VectorStore should not be instantiable directly"""
        with pytest.raises(TypeError):
            VectorStore()

    def test_required_methods(self):
        """VectorStore should define all required abstract methods"""
        required = {
            "ensure_collection",
            "upsert",
            "upsert_batch",
            "search",
            "delete",
            "delete_by_filter",
            "get",
            "count",
            "health_check",
        }
        actual = {
            name
            for name in dir(VectorStore)
            if not name.startswith("_") and callable(getattr(VectorStore, name, None))
        }
        assert required.issubset(actual), f"Missing methods: {required - actual}"


# =========================================================================
# Integration Tests: LongTermMemory + VectorStore
# =========================================================================


def _create_mock_vector_store():
    """Create a mock VectorStore with all async methods"""
    mock = AsyncMock()
    mock.ensure_collection = AsyncMock()
    mock.upsert = AsyncMock()
    # Return the actual batch length so counts are accurate
    mock.upsert_batch = AsyncMock(
        side_effect=lambda collection, ids, vectors, payloads=None: len(ids)
    )
    mock.search = AsyncMock(return_value=[])
    mock.delete = AsyncMock(return_value=True)
    mock.count = AsyncMock(return_value=42)
    mock.health_check = AsyncMock(return_value={"status": "healthy"})
    return mock


class TestLTMQdrantIntegration:
    """Test LongTermMemory with mocked Qdrant"""

    def _run(self, coro):
        """Helper to run async code"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

    def setup_method(self):
        """Set up LTM with in-memory SQLite and mocked Qdrant"""
        reset_vector_store()

        from memory.layers.long_term import LongTermMemory, MemoryType
        from memory.config import LTMConfig

        self.tmpdir = tempfile.mkdtemp()
        config = LTMConfig(sqlite_path=os.path.join(self.tmpdir, "test_ltm.db"))
        self.ltm = LongTermMemory(config)

        # Initialize without Qdrant (patch at the source module)
        with patch(
            "db.vector_store.get_vector_store", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = None
            self._run(self.ltm.initialize())

        # Now inject a mock vector store
        self.mock_vs = _create_mock_vector_store()
        self.ltm._vector_store = self.mock_vs

    def test_store_upserts_to_qdrant(self):
        """Storing a memory should also upsert to Qdrant"""
        from memory.layers.long_term import MemoryType

        # Mock embedder to return a real embedding
        self.ltm._embedder = MagicMock()
        self.ltm._embedder.is_available = True
        self.ltm._embedder.encode.return_value = [0.1] * 768

        entry = self._run(
            self.ltm.store(
                content="Boss prefers direct communication",
                memory_type=MemoryType.PREFERENCE,
                domain="personal",
            )
        )

        assert entry is not None
        assert entry.embedding is not None

        # Verify Qdrant upsert was called
        self.mock_vs.upsert.assert_called_once()
        call_args = self.mock_vs.upsert.call_args
        assert call_args.kwargs["collection"] == COLLECTION_MEMORIES
        assert call_args.kwargs["id"] == entry.id
        assert len(call_args.kwargs["vector"]) == 768
        assert call_args.kwargs["payload"]["memory_type"] == "preference"

    def test_store_without_embedding_skips_qdrant(self):
        """If no embedding generated, should skip Qdrant upsert"""
        from memory.layers.long_term import MemoryType

        self.ltm._embedder = MagicMock()
        self.ltm._embedder.is_available = False

        entry = self._run(
            self.ltm.store(
                content="Some fact",
                memory_type=MemoryType.FACT,
            )
        )

        assert entry.embedding is None
        self.mock_vs.upsert.assert_not_called()

    def test_delete_removes_from_qdrant(self):
        """Deleting a memory should also remove from Qdrant"""
        from memory.layers.long_term import MemoryType

        self.ltm._embedder = MagicMock()
        self.ltm._embedder.is_available = True
        self.ltm._embedder.encode.return_value = [0.1] * 768

        entry = self._run(self.ltm.store("test memory", MemoryType.FACT))

        # Reset mock to track delete calls
        self.mock_vs.reset_mock()

        deleted = self._run(self.ltm.delete(entry.id))
        assert deleted is True
        self.mock_vs.delete.assert_called_once_with(COLLECTION_MEMORIES, entry.id)

    def test_search_uses_qdrant_first(self):
        """Search should try Qdrant first"""
        from memory.layers.long_term import MemoryType

        self.ltm._embedder = MagicMock()
        self.ltm._embedder.is_available = True
        self.ltm._embedder.encode.return_value = [0.1] * 768

        # Store a memory first
        entry = self._run(self.ltm.store("Boss likes coffee", MemoryType.PREFERENCE))

        # Set up Qdrant to return this memory
        self.mock_vs.search.return_value = [
            VectorSearchResult(id=entry.id, score=0.92, payload={})
        ]

        results = self._run(self.ltm.search("what does Boss like?", top_k=5))

        # Verify Qdrant search was called
        self.mock_vs.search.assert_called_once()
        assert len(results) >= 1
        assert results[0][0].id == entry.id
        assert results[0][1] == 0.92

    def test_search_falls_back_to_sqlite(self):
        """If Qdrant search fails, should fall back to SQLite"""
        from memory.layers.long_term import MemoryType

        self.ltm._embedder = MagicMock()
        self.ltm._embedder.is_available = True
        self.ltm._embedder.encode.return_value = [0.1] * 768

        # Store a memory
        self._run(self.ltm.store("Test fallback memory", MemoryType.FACT))

        # Make Qdrant fail
        self.mock_vs.search.side_effect = Exception("Qdrant connection refused")

        # Should still work via SQLite fallback
        results = self._run(self.ltm.search("fallback", top_k=5))
        # May or may not find results depending on cosine similarity,
        # but should NOT raise
        assert isinstance(results, list)

    def test_search_without_vector_store_uses_sqlite(self):
        """With no vector store at all, should use SQLite directly"""
        from memory.layers.long_term import MemoryType

        self.ltm._vector_store = None  # No Qdrant
        self.ltm._embedder = MagicMock()
        self.ltm._embedder.is_available = True
        self.ltm._embedder.encode.return_value = [0.1] * 768

        self._run(self.ltm.store("SQLite only", MemoryType.FACT))

        results = self._run(self.ltm.search("SQLite", top_k=5))
        assert isinstance(results, list)

    def test_sync_to_vector_store(self):
        """sync_to_vector_store should batch upsert all embeddings"""
        from memory.layers.long_term import MemoryType

        self.ltm._embedder = MagicMock()
        self.ltm._embedder.is_available = True
        self.ltm._embedder.encode.return_value = [0.1] * 768

        # Temporarily disable Qdrant during storage to only store in SQLite
        self.ltm._vector_store = None
        for i in range(5):
            self._run(self.ltm.store(f"Memory {i}", MemoryType.FACT))

        # Re-attach mock and sync
        self.ltm._vector_store = self.mock_vs
        count = self._run(self.ltm.sync_to_vector_store(batch_size=3))

        assert count == 5
        # Should have been called twice: batch of 3, then batch of 2
        assert self.mock_vs.upsert_batch.call_count == 2

    def test_stats_includes_vector_backend(self):
        """Stats should report which vector backend is in use"""
        stats = self._run(self.ltm.get_stats())
        assert stats["vector_backend"] == "qdrant"

    def test_stats_without_qdrant(self):
        """Stats should report sqlite when no Qdrant"""
        self.ltm._vector_store = None
        stats = self._run(self.ltm.get_stats())
        assert stats["vector_backend"] == "sqlite"

    def test_update_content_updates_qdrant(self):
        """Updating memory content should re-embed and update Qdrant"""
        from memory.layers.long_term import MemoryType

        self.ltm._embedder = MagicMock()
        self.ltm._embedder.is_available = True
        self.ltm._embedder.encode.return_value = [0.2] * 768

        entry = self._run(self.ltm.store("original", MemoryType.FACT))
        self.mock_vs.reset_mock()

        # Update content (triggers re-embedding)
        self._run(self.ltm.update(entry.id, content="updated content"))

        # Should have upserted new embedding to Qdrant
        self.mock_vs.upsert.assert_called_once()
        call_args = self.mock_vs.upsert.call_args
        assert call_args.kwargs["id"] == entry.id


class TestDocumentSearcherQdrantIntegration:
    """Test DocumentSearcher with mocked Qdrant"""

    def _run(self, coro):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

    def setup_method(self):
        """Set up DocumentStore and DocumentSearcher with mocked Qdrant"""
        reset_vector_store()

        from documents.storage.document_store import DocumentStore
        from documents.retrieval.searcher import DocumentSearcher
        from documents.config import RetrievalConfig, StorageConfig

        self.tmpdir = tempfile.mkdtemp()
        storage_config = StorageConfig(
            db_path=os.path.join(self.tmpdir, "test_docs.db")
        )
        self.store = DocumentStore(storage_config)
        self.store.initialize()

        retrieval_config = RetrievalConfig(
            use_hybrid_search=False,
            min_similarity=0.0,
        )

        self.mock_vs = _create_mock_vector_store()
        self.searcher = DocumentSearcher(
            store=self.store,
            config=retrieval_config,
            vector_store=self.mock_vs,
        )

        # Mock embedding model
        self.searcher._embedding_model = MagicMock()
        self.searcher._embedding_model.encode.return_value = MagicMock(
            tolist=MagicMock(return_value=[0.1] * 768)
        )

    def test_search_uses_qdrant_when_available(self):
        """Search should use Qdrant when vector store is present"""
        # Set up empty results (no chunks in DB)
        self.mock_vs.search.return_value = []

        results = self._run(self.searcher.search("test query"))

        # Qdrant should have been called
        self.mock_vs.search.assert_called_once()
        call_args = self.mock_vs.search.call_args
        assert call_args.kwargs["collection"] == COLLECTION_DOCUMENTS

    def test_search_falls_back_on_qdrant_error(self):
        """If Qdrant fails, should fall back to SQLite"""
        self.mock_vs.search.side_effect = Exception("Qdrant down")

        # Should not raise, falls back to SQLite
        results = self._run(self.searcher.search("test query"))
        assert isinstance(results, list)

    def test_search_without_vector_store(self):
        """Without vector store, should use SQLite directly"""
        self.searcher._vector_store = None

        results = self._run(self.searcher.search("test"))
        assert isinstance(results, list)

    def test_search_without_embedding_model(self):
        """Without embedding model, should fall back to keyword search"""
        self.searcher._embedding_model = None

        results = self._run(self.searcher.search("test"))
        assert isinstance(results, list)


class TestDocumentStoreSyncIntegration:
    """Test DocumentStore vector sync methods"""

    def _run(self, coro):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)

    def setup_method(self):
        from documents.storage.document_store import DocumentStore
        from documents.config import StorageConfig

        self.tmpdir = tempfile.mkdtemp()
        config = StorageConfig(db_path=os.path.join(self.tmpdir, "test_sync.db"))
        self.store = DocumentStore(config)
        self.store.initialize()
        self.mock_vs = _create_mock_vector_store()

    def _ensure_test_document(self):
        """Insert a parent document for foreign key compliance"""
        conn = self.store._conn
        conn.execute(
            """INSERT OR IGNORE INTO documents
               (id, file_path, file_hash, file_size, document_type,
                metadata, language, total_pages, chapters, status,
                processed_pages, project, access_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "test-doc-1",
                "/test/doc.pdf",
                "abc123hash",
                1024,
                "book",
                '{"title": "Test Book"}',
                "en",
                10,
                "[]",
                "completed",
                10,
                None,
                0,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

    def _insert_test_chunks(self, count: int = 5):
        """Insert test chunks with embeddings directly into SQLite"""
        self._ensure_test_document()
        conn = self.store._conn
        for i in range(count):
            chunk_id = str(uuid.uuid4())
            doc_id = "test-doc-1"
            embedding = pickle.dumps(np.random.rand(768).astype(np.float32))
            conn.execute(
                """INSERT INTO chunks
                   (id, document_id, page_ids, content, page_range,
                    chapter, section, embedding, chunk_index, char_count,
                    token_count_approx, entities, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk_id,
                    doc_id,
                    "[]",
                    f"Chunk content {i}",
                    f"p. {i}",
                    f"Chapter {i}",
                    None,
                    embedding,
                    i,
                    100,
                    25,
                    "[]",
                    datetime.now().isoformat(),
                ),
            )
        conn.commit()

    def test_sync_embeddings(self):
        """sync_embeddings_to_vector_store should batch upsert all chunks"""
        self._insert_test_chunks(7)

        count = self._run(
            self.store.sync_embeddings_to_vector_store(self.mock_vs, batch_size=3)
        )

        assert count == 7
        # 3 batches: 3 + 3 + 1
        assert self.mock_vs.upsert_batch.call_count == 3

    def test_sync_single_document(self):
        """Should filter by document_id"""
        self._insert_test_chunks(3)

        count = self._run(
            self.store.sync_embeddings_to_vector_store(
                self.mock_vs, document_id="test-doc-1"
            )
        )

        assert count == 3

    def test_sync_no_embeddings(self):
        """Empty store should return 0"""
        count = self._run(self.store.sync_embeddings_to_vector_store(self.mock_vs))
        assert count == 0
        self.mock_vs.upsert_batch.assert_not_called()

    def test_upsert_chunk_to_vector_store(self):
        """upsert_chunk_to_vector_store should call upsert"""
        from documents.models import Chunk

        chunk = Chunk(
            id="test-chunk-1",
            document_id="test-doc-1",
            page_ids=["p1"],
            content="Test chunk content",
            page_range="p. 1",
            chapter="Chapter 1",
            section=None,
            embedding=[0.1] * 768,
            chunk_index=0,
            char_count=100,
            token_count_approx=25,
            entities=[],
            created_at=datetime.now(),
        )

        self._run(self.store.upsert_chunk_to_vector_store(self.mock_vs, chunk))

        self.mock_vs.upsert.assert_called_once()
        call_args = self.mock_vs.upsert.call_args
        assert call_args.kwargs["id"] == "test-chunk-1"
        assert call_args.kwargs["payload"]["document_id"] == "test-doc-1"

    def test_upsert_chunk_without_embedding_is_noop(self):
        """Chunk without embedding should not call Qdrant"""
        from documents.models import Chunk

        chunk = Chunk(
            id="test-chunk-2",
            document_id="test-doc-1",
            page_ids=["p1"],
            content="No embedding",
            page_range="p. 1",
            embedding=None,
            chunk_index=0,
            char_count=50,
            token_count_approx=10,
            entities=[],
            created_at=datetime.now(),
        )

        self._run(self.store.upsert_chunk_to_vector_store(self.mock_vs, chunk))
        self.mock_vs.upsert.assert_not_called()
