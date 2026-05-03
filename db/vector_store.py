"""
Vector Store Abstraction for Friday AI
=======================================

Provides a clean interface for vector similarity search with:
- Qdrant as primary backend (HNSW indexing, production-grade)
- In-memory fallback for local development without Qdrant

Collections:
- friday_documents: Document chunk embeddings
- friday_memories: LTM memory embeddings
- friday_scenes: Screenplay scene embeddings

All collections use 768-dimensional vectors from
paraphrase-multilingual-mpnet-base-v2.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# Vector dimension for paraphrase-multilingual-mpnet-base-v2
VECTOR_DIM = 768

# Collection names
COLLECTION_DOCUMENTS = "friday_documents"
COLLECTION_MEMORIES = "friday_memories"
COLLECTION_SCENES = "friday_scenes"


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search"""

    id: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """Configuration for the vector store"""

    # Qdrant connection
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    use_grpc: bool = False
    api_key: Optional[str] = None

    # Collection settings
    vector_dim: int = VECTOR_DIM
    distance_metric: str = "cosine"  # cosine, euclid, dot

    # HNSW index parameters
    hnsw_m: int = 16  # Number of edges per node (higher = better recall, more RAM)
    hnsw_ef_construct: int = (
        100  # Construction accuracy (higher = better index, slower build)
    )
    hnsw_ef_search: int = 128  # Search accuracy (higher = better recall, slower search)

    # Optimizers
    indexing_threshold: int = 20000  # Start HNSW after this many vectors
    memmap_threshold: int = 50000  # Switch to mmap storage after this many

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        """Create config from environment variables.

        Supports both QDRANT_HOST/QDRANT_PORT (explicit) and
        QDRANT_URL (e.g. http://qdrant:6333 from Docker compose).
        """
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", "6333"))

        # Parse QDRANT_URL if set (docker-compose style)
        qdrant_url = os.environ.get("QDRANT_URL")
        if qdrant_url:
            from urllib.parse import urlparse

            parsed = urlparse(qdrant_url)
            host = parsed.hostname or host
            port = parsed.port or port

        return cls(
            host=host,
            port=port,
            grpc_port=int(os.environ.get("QDRANT_GRPC_PORT", "6334")),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )


class VectorStore(ABC):
    """Abstract vector store interface.

    All vector operations go through this interface, making it easy
    to swap backends (Qdrant, pgvector, in-memory) without changing
    consuming code.
    """

    @abstractmethod
    async def ensure_collection(self, name: str, vector_dim: int = VECTOR_DIM) -> None:
        """Create collection if it doesn't exist"""

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a vector with metadata"""

    @abstractmethod
    async def upsert_batch(
        self,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Batch insert/update vectors. Returns count of upserted vectors."""

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors with optional metadata filtering"""

    @abstractmethod
    async def delete(self, collection: str, id: str) -> bool:
        """Delete a vector by ID"""

    @abstractmethod
    async def delete_by_filter(self, collection: str, filters: Dict[str, Any]) -> int:
        """Delete vectors matching filter. Returns count deleted."""

    @abstractmethod
    async def get(self, collection: str, id: str) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID"""

    @abstractmethod
    async def count(self, collection: str) -> int:
        """Count vectors in a collection"""

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check if the vector store is healthy"""


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation.

    Uses qdrant-client for high-performance vector similarity search
    with HNSW indexing. Supports filtering, batch operations, and
    automatic collection management.
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig.from_env()
        self._client = None
        self._async_client = None

    def _get_client(self):
        """Lazy-load the synchronous Qdrant client"""
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.grpc_port if self.config.use_grpc else None,
                prefer_grpc=self.config.use_grpc,
                api_key=self.config.api_key,
                timeout=10,
            )
        return self._client

    async def _get_async_client(self):
        """Lazy-load the async Qdrant client"""
        if self._async_client is None:
            from qdrant_client import AsyncQdrantClient

            self._async_client = AsyncQdrantClient(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.grpc_port if self.config.use_grpc else None,
                prefer_grpc=self.config.use_grpc,
                api_key=self.config.api_key,
                timeout=10,
            )
        return self._async_client

    async def ensure_collection(self, name: str, vector_dim: int = VECTOR_DIM) -> None:
        """Create a Qdrant collection with HNSW indexing if it doesn't exist"""
        from qdrant_client.models import (
            Distance,
            VectorParams,
            HnswConfigDiff,
            OptimizersConfigDiff,
        )

        client = await self._get_async_client()

        # Check if collection exists
        collections = await client.get_collections()
        existing = {c.name for c in collections.collections}

        if name in existing:
            LOGGER.debug("Collection '%s' already exists", name)
            return

        # Map distance metric
        distance_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        distance = distance_map.get(self.config.distance_metric, Distance.COSINE)

        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=distance,
            ),
            hnsw_config=HnswConfigDiff(
                m=self.config.hnsw_m,
                ef_construct=self.config.hnsw_ef_construct,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=self.config.indexing_threshold,
                memmap_threshold=self.config.memmap_threshold,
            ),
        )
        LOGGER.info(
            "Created collection '%s' (dim=%d, distance=%s, HNSW m=%d)",
            name,
            vector_dim,
            self.config.distance_metric,
            self.config.hnsw_m,
        )

    async def upsert(
        self,
        collection: str,
        id: str,
        vector: List[float],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert a single vector into Qdrant"""
        from qdrant_client.models import PointStruct

        client = await self._get_async_client()
        await client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=id,
                    vector=vector,
                    payload=payload or {},
                )
            ],
        )

    async def upsert_batch(
        self,
        collection: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Batch upsert vectors into Qdrant"""
        from qdrant_client.models import PointStruct

        if not ids:
            return 0

        client = await self._get_async_client()
        plds = payloads or [{} for _ in ids]

        points = [
            PointStruct(id=id_, vector=vec, payload=pld)
            for id_, vec, pld in zip(ids, vectors, plds)
        ]

        # Batch in chunks of 100
        batch_size = 100
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await client.upsert(
                collection_name=collection,
                points=batch,
            )
            total += len(batch)

        return total

    async def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors in Qdrant"""
        from qdrant_client.models import (
            Filter,
            FieldCondition,
            MatchValue,
            SearchParams,
        )

        client = await self._get_async_client()

        # Build filter from dict
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if value is not None:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        results = await client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=min_score if min_score > 0 else None,
            query_filter=qdrant_filter,
            search_params=SearchParams(
                hnsw_ef=self.config.hnsw_ef_search,
                exact=False,  # Use HNSW approximation
            ),
        )

        return [
            VectorSearchResult(
                id=str(point.id),
                score=point.score,
                payload=point.payload or {},
            )
            for point in results
        ]

    async def delete(self, collection: str, id: str) -> bool:
        """Delete a vector from Qdrant"""
        from qdrant_client.models import PointIdsList

        client = await self._get_async_client()
        await client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=[id]),
        )
        return True

    async def delete_by_filter(self, collection: str, filters: Dict[str, Any]) -> int:
        """Delete vectors matching filter conditions"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = await self._get_async_client()

        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
            if v is not None
        ]

        if not conditions:
            return 0

        await client.delete(
            collection_name=collection,
            points_selector=Filter(must=conditions),
        )
        # Qdrant delete doesn't return count, return -1 to indicate success
        return -1

    async def get(self, collection: str, id: str) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID"""
        client = await self._get_async_client()
        results = await client.retrieve(
            collection_name=collection,
            ids=[id],
            with_vectors=False,
        )
        if results:
            point = results[0]
            return VectorSearchResult(
                id=str(point.id),
                score=1.0,
                payload=point.payload or {},
            )
        return None

    async def count(self, collection: str) -> int:
        """Count vectors in a collection"""
        client = await self._get_async_client()
        info = await client.get_collection(collection_name=collection)
        return info.points_count

    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant health"""
        try:
            client = await self._get_async_client()
            collections = await client.get_collections()
            collection_info = {}
            for c in collections.collections:
                info = await client.get_collection(c.name)
                collection_info[c.name] = {
                    "vectors": info.points_count,
                    "status": (
                        info.status.value
                        if hasattr(info.status, "value")
                        else str(info.status)
                    ),
                }

            return {
                "status": "healthy",
                "backend": "qdrant",
                "host": f"{self.config.host}:{self.config.port}",
                "collections": collection_info,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "qdrant",
                "error": str(e),
            }


# =========================================================================
# Singleton + Fallback Management
# =========================================================================

_vector_store: Optional[VectorStore] = None
_initialized = False


async def get_vector_store() -> Optional[VectorStore]:
    """Get the vector store singleton.

    Returns None if Qdrant is not available and no fallback is configured.
    Callers should fall back to their existing SQLite-based cosine
    similarity when this returns None.
    """
    global _vector_store, _initialized

    if _initialized:
        return _vector_store

    _initialized = True

    # Try Qdrant
    config = VectorStoreConfig.from_env()
    store = QdrantVectorStore(config)

    try:
        health = await store.health_check()
        if health.get("status") == "healthy":
            _vector_store = store
            LOGGER.info(
                "Qdrant vector store connected at %s:%d",
                config.host,
                config.port,
            )

            # Ensure collections exist
            await store.ensure_collection(COLLECTION_DOCUMENTS)
            await store.ensure_collection(COLLECTION_MEMORIES)
            await store.ensure_collection(COLLECTION_SCENES)

            return _vector_store
    except Exception as e:
        LOGGER.warning("Qdrant not available (%s), using SQLite fallback", e)

    # No Qdrant available — callers use their existing SQLite vector search
    _vector_store = None
    return None


def reset_vector_store() -> None:
    """Reset the singleton (for testing)"""
    global _vector_store, _initialized
    _vector_store = None
    _initialized = False
