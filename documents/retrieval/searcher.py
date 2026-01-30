"""
Document Searcher

Handles vector and hybrid search across document chunks.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from documents.config import RetrievalConfig, get_document_config
from documents.models import (
    Chunk,
    Citation,
    Document,
    DocumentSearchResult,
    DocumentType,
)
from documents.storage.document_store import DocumentStore

LOGGER = logging.getLogger(__name__)


class DocumentSearcher:
    """
    Document search with vector similarity and hybrid search.

    Provides:
    - Vector similarity search on chunk embeddings
    - FTS5 keyword search
    - Hybrid search combining both
    - Result ranking and formatting
    """

    def __init__(
        self,
        store: DocumentStore,
        config: Optional[RetrievalConfig] = None,
    ):
        self.store = store
        self.config = config or get_document_config().retrieval
        self._embedding_model = None

    async def initialize(self) -> None:
        """Initialize embedding model for query encoding"""
        try:
            from sentence_transformers import SentenceTransformer

            model_name = get_document_config().embedding.model_name
            self._embedding_model = SentenceTransformer(model_name)
            LOGGER.info("Loaded embedding model: %s", model_name)
        except ImportError:
            LOGGER.warning(
                "sentence-transformers not installed. "
                "Vector search will fall back to keyword search."
            )

    async def search(
        self,
        query: str,
        document_id: Optional[str] = None,
        document_type: Optional[DocumentType] = None,
        project: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[DocumentSearchResult]:
        """
        Search documents using hybrid search.

        Args:
            query: Search query
            document_id: Limit to specific document
            document_type: Filter by document type
            project: Filter by project
            top_k: Number of results (default from config)

        Returns:
            List of DocumentSearchResult with citations
        """
        top_k = top_k or self.config.vector_search_top_k

        if self.config.use_hybrid_search:
            return await self._hybrid_search(
                query, document_id, document_type, project, top_k
            )
        else:
            return await self._vector_search(query, document_id, top_k)

    async def _vector_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[DocumentSearchResult]:
        """Pure vector similarity search"""
        if not self._embedding_model:
            LOGGER.warning("No embedding model, falling back to keyword search")
            return await self._keyword_search(query, document_id, top_k)

        # Encode query
        try:
            query_embedding = self._embedding_model.encode(
                query, normalize_embeddings=True
            ).tolist()
        except Exception as e:
            LOGGER.error("Failed to encode query: %s", e)
            return await self._keyword_search(query, document_id, top_k)

        # Search
        results = self.store.vector_search(
            query_embedding=query_embedding,
            document_id=document_id,
            top_k=top_k,
            min_similarity=self.config.min_similarity,
        )

        return self._format_results(results)

    async def _keyword_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[DocumentSearchResult]:
        """FTS5 keyword search"""
        results = self.store.keyword_search(
            query=query,
            document_id=document_id,
            top_k=top_k,
        )

        return self._format_results(results)

    async def _hybrid_search(
        self,
        query: str,
        document_id: Optional[str] = None,
        document_type: Optional[DocumentType] = None,
        project: Optional[str] = None,
        top_k: int = 10,
    ) -> List[DocumentSearchResult]:
        """
        Hybrid search combining vector and keyword.

        Uses reciprocal rank fusion (RRF) to combine results.
        """
        # Get more results for merging
        fetch_k = top_k * 2

        # Vector search
        vector_results: List[Tuple[Chunk, float]] = []
        if self._embedding_model:
            try:
                query_embedding = self._embedding_model.encode(
                    query, normalize_embeddings=True
                ).tolist()
                vector_results = self.store.vector_search(
                    query_embedding=query_embedding,
                    document_id=document_id,
                    top_k=fetch_k,
                    min_similarity=0.0,  # Get more for fusion
                )
            except Exception as e:
                LOGGER.warning("Vector search failed: %s", e)

        # Keyword search
        keyword_results = self.store.keyword_search(
            query=query,
            document_id=document_id,
            top_k=fetch_k,
        )

        # Combine using RRF
        combined = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            k=60,  # RRF constant
            vector_weight=1 - self.config.keyword_weight,
            keyword_weight=self.config.keyword_weight,
        )

        # Filter by document type and project if specified
        if document_type or project:
            filtered = []
            for chunk, score in combined:
                doc = self.store.get_document(chunk.document_id)
                if doc:
                    if document_type and doc.document_type != document_type:
                        continue
                    if project and doc.project != project:
                        continue
                    filtered.append((chunk, score))
            combined = filtered

        # Take top_k and filter by min_similarity
        combined = [
            (chunk, score)
            for chunk, score in combined[:top_k]
            if score >= self.config.min_similarity
        ]

        return self._format_results(combined)

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[Chunk, float]],
        keyword_results: List[Tuple[Chunk, float]],
        k: int = 60,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> List[Tuple[Chunk, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank))
        """
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}

        # Process vector results
        for rank, (chunk, _) in enumerate(vector_results):
            rrf_score = vector_weight * (1 / (k + rank + 1))
            scores[chunk.id] = scores.get(chunk.id, 0) + rrf_score
            chunks[chunk.id] = chunk

        # Process keyword results
        for rank, (chunk, _) in enumerate(keyword_results):
            rrf_score = keyword_weight * (1 / (k + rank + 1))
            scores[chunk.id] = scores.get(chunk.id, 0) + rrf_score
            chunks[chunk.id] = chunk

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Normalize scores to 0-1 range
        max_score = max(scores.values()) if scores else 1.0
        return [
            (chunks[chunk_id], scores[chunk_id] / max_score) for chunk_id in sorted_ids
        ]

    def _format_results(
        self,
        results: List[Tuple[Chunk, float]],
    ) -> List[DocumentSearchResult]:
        """Format chunk results as DocumentSearchResult"""
        formatted = []

        for chunk, similarity in results:
            # Get document
            document = self.store.get_document(chunk.document_id)
            if not document:
                continue

            # Extract highlight
            highlight = self._extract_highlight(chunk.content)

            # Create citation
            citation = Citation(
                document_id=document.id,
                document_title=document.metadata.title,
                chunk_id=chunk.id,
                page_range=chunk.page_range,
                chapter=chunk.chapter,
                section=chunk.section,
                quote=highlight,
                relevance=similarity,
            )

            formatted.append(
                DocumentSearchResult(
                    chunk=chunk,
                    document=document,
                    similarity=similarity,
                    highlight=highlight,
                    citation=citation,
                )
            )

        return formatted

    def _extract_highlight(self, content: str, max_length: int = 200) -> str:
        """Extract the most relevant snippet from content"""
        # Clean content
        text = content.strip()

        # Get first substantial paragraph
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) > 30:
                text = para
                break

        # Truncate if needed
        if len(text) > max_length:
            # Try to end at sentence
            for end in [". ", "? ", "! "]:
                idx = text.rfind(end, 0, max_length)
                if idx > max_length // 2:
                    return text[: idx + 1].strip()

            return text[:max_length].strip() + "..."

        return text

    async def search_in_chapter(
        self,
        query: str,
        document_id: str,
        chapter: str,
        top_k: int = 5,
    ) -> List[DocumentSearchResult]:
        """Search within a specific chapter"""
        # Get chapter chunks
        all_chunks = self.store.get_chunks_for_document(document_id, chapter=chapter)

        if not all_chunks or not self._embedding_model:
            return []

        # Encode query
        try:
            query_embedding = self._embedding_model.encode(
                query, normalize_embeddings=True
            )
        except Exception:
            return []

        # Score chunks
        import numpy as np

        results: List[Tuple[Chunk, float]] = []
        for chunk in all_chunks:
            if chunk.embedding:
                chunk_vec = np.array(chunk.embedding)
                similarity = float(np.dot(query_embedding, chunk_vec))
                if similarity >= self.config.min_similarity:
                    results.append((chunk, similarity))

        # Sort and limit
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        return self._format_results(results)
