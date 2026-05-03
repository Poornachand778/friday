"""
Tests for DocumentSearcher
============================

Comprehensive tests for the document search engine covering:
- Initialization with default/custom config
- Highlight extraction from content
- Reciprocal Rank Fusion (RRF) score merging
- Result formatting with citations
- Vector search (Qdrant + SQLite fallback)
- Keyword search (FTS5)
- Hybrid search combining vector + keyword
- Top-level search dispatcher
- Chapter-scoped search

Run with: pytest tests/test_document_searcher.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from documents.retrieval.searcher import DocumentSearcher
from documents.config import RetrievalConfig
from documents.models import (
    Chunk,
    Citation,
    Document,
    DocumentMetadata,
    DocumentLanguage,
    DocumentSearchResult,
    DocumentType,
    ProcessingStatus,
)


# =========================================================================
# Helpers
# =========================================================================


def _make_chunk(
    chunk_id="chunk-1",
    document_id="doc-1",
    content="This is chunk content for testing purposes.",
    page_range="pp. 1-3",
    chapter="Chapter 1",
    section="Introduction",
    embedding=None,
    page_ids=None,
):
    """Create a Chunk with sensible defaults for testing."""
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        page_ids=page_ids or ["page-1"],
        content=content,
        page_range=page_range,
        chapter=chapter,
        section=section,
        embedding=embedding,
        chunk_index=0,
        char_count=len(content),
        token_count_approx=len(content) // 4,
    )


def _make_document(
    doc_id="doc-1",
    title="Test Book",
    document_type=DocumentType.BOOK,
    project=None,
):
    """Create a Document with sensible defaults for testing."""
    return Document(
        id=doc_id,
        file_path="/fake/path.pdf",
        file_hash="abc123",
        file_size=1024,
        document_type=document_type,
        metadata=DocumentMetadata(title=title),
        language=DocumentLanguage.ENGLISH,
        total_pages=100,
        project=project,
    )


def _make_store():
    """Create a mock DocumentStore."""
    store = MagicMock()
    store.vector_search = MagicMock(return_value=[])
    store.keyword_search = MagicMock(return_value=[])
    store.get_chunk = MagicMock(return_value=None)
    store.get_document = MagicMock(return_value=None)
    store.get_chunks_for_document = MagicMock(return_value=[])
    return store


def _make_vector_store():
    """Create a mock VectorStore with async search."""
    vs = MagicMock()
    vs.search = AsyncMock(return_value=[])
    return vs


def _make_embedding_model(embedding_dim=768):
    """Create a mock SentenceTransformer model."""
    model = MagicMock()
    embedding = np.random.randn(embedding_dim).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    model.encode = MagicMock(return_value=embedding)
    return model


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def store():
    return _make_store()


@pytest.fixture
def config():
    return RetrievalConfig()


@pytest.fixture
def custom_config():
    return RetrievalConfig(
        vector_search_top_k=5,
        min_similarity=0.3,
        use_hybrid_search=False,
        keyword_weight=0.5,
    )


@pytest.fixture
def searcher(store, config):
    return DocumentSearcher(store=store, config=config)


@pytest.fixture
def searcher_with_model(store, config):
    s = DocumentSearcher(store=store, config=config)
    s._embedding_model = _make_embedding_model()
    return s


@pytest.fixture
def searcher_with_qdrant(store, config):
    vs = _make_vector_store()
    s = DocumentSearcher(store=store, config=config, vector_store=vs)
    s._embedding_model = _make_embedding_model()
    return s


@pytest.fixture
def sample_chunk():
    return _make_chunk()


@pytest.fixture
def sample_document():
    return _make_document()


# =========================================================================
# TestSearcherInit
# =========================================================================


class TestSearcherInit:
    """Tests for DocumentSearcher constructor."""

    def test_default_config(self, store):
        """When no config is provided, loads from get_document_config."""
        with patch("documents.retrieval.searcher.get_document_config") as mock_cfg:
            mock_cfg.return_value.retrieval = RetrievalConfig(vector_search_top_k=20)
            searcher = DocumentSearcher(store=store)
            assert searcher.config.vector_search_top_k == 20
            assert searcher.store is store

    def test_custom_config(self, store, custom_config):
        """When config is provided, uses it directly."""
        searcher = DocumentSearcher(store=store, config=custom_config)
        assert searcher.config.vector_search_top_k == 5
        assert searcher.config.min_similarity == 0.3
        assert searcher.config.use_hybrid_search is False
        assert searcher.config.keyword_weight == 0.5

    def test_vector_store_param(self, store, config):
        """Vector store is stored when provided."""
        vs = _make_vector_store()
        searcher = DocumentSearcher(store=store, config=config, vector_store=vs)
        assert searcher._vector_store is vs

    def test_initial_state_no_model(self, store, config):
        """Embedding model is None until initialize() is called."""
        searcher = DocumentSearcher(store=store, config=config)
        assert searcher._embedding_model is None


# =========================================================================
# TestInitialize
# =========================================================================


class TestInitialize:
    """Tests for the async initialize() method."""

    @pytest.mark.asyncio
    async def test_initialize_loads_model(self, store, config):
        """initialize() loads the SentenceTransformer model."""
        searcher = DocumentSearcher(store=store, config=config)
        mock_model = MagicMock()
        with patch(
            "documents.retrieval.searcher.get_document_config"
        ) as mock_cfg, patch.dict(
            "sys.modules",
            {
                "sentence_transformers": MagicMock(
                    SentenceTransformer=MagicMock(return_value=mock_model)
                )
            },
        ):
            mock_cfg.return_value.embedding.model_name = "test-model"
            await searcher.initialize()
            assert searcher._embedding_model is not None

    @pytest.mark.asyncio
    async def test_initialize_without_sentence_transformers(self, store, config):
        """initialize() gracefully handles missing sentence-transformers."""
        searcher = DocumentSearcher(store=store, config=config)

        # Simulate ImportError by patching the import inside initialize
        original_import = (
            __builtins__.__import__
            if hasattr(__builtins__, "__import__")
            else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            await searcher.initialize()
            assert searcher._embedding_model is None

    @pytest.mark.asyncio
    async def test_initialize_connects_qdrant(self, store, config):
        """initialize() attempts to connect to Qdrant when no vector_store provided."""
        searcher = DocumentSearcher(store=store, config=config)
        mock_vs = MagicMock()

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        mock_get_vs = AsyncMock(return_value=mock_vs)

        with patch.dict(
            "sys.modules",
            {
                "sentence_transformers": mock_st_module,
                "db.vector_store": MagicMock(get_vector_store=mock_get_vs),
            },
        ), patch("documents.retrieval.searcher.get_document_config") as mock_cfg:
            mock_cfg.return_value.embedding.model_name = "test-model"
            await searcher.initialize()
            assert searcher._vector_store is mock_vs


# =========================================================================
# TestExtractHighlight
# =========================================================================


class TestExtractHighlight:
    """Tests for _extract_highlight()."""

    def test_short_text_returned_as_is(self, searcher):
        """Text shorter than max_length is returned unchanged."""
        text = "Short text for testing."
        assert searcher._extract_highlight(text) == "Short text for testing."

    def test_long_text_truncated_at_sentence_boundary(self, searcher):
        """Long text is truncated at the last sentence boundary within max_length."""
        sentence1 = "A" * 120 + ". "
        sentence2 = "B" * 120 + ". "
        text = sentence1 + sentence2
        result = searcher._extract_highlight(text, max_length=200)
        # Should end at the first sentence boundary
        assert result.endswith(".")
        assert len(result) <= 200

    def test_long_text_truncated_with_ellipsis(self, searcher):
        """Long text with no sentence boundary gets hard-truncated with ellipsis."""
        text = "A" * 300
        result = searcher._extract_highlight(text, max_length=200)
        assert result.endswith("...")
        assert len(result) == 203  # 200 + "..."

    def test_multiple_paragraphs_uses_first_substantial(self, searcher):
        """With multiple paragraphs, the first substantial one (>30 chars) is used."""
        text = "Short.\n\nThis is a substantially longer paragraph with enough characters to be considered meaningful."
        result = searcher._extract_highlight(text)
        assert "substantially longer" in result

    def test_short_paragraphs_skipped(self, searcher):
        """Paragraphs shorter than 30 chars are skipped."""
        text = "Hi.\n\nBye.\n\nThis paragraph is long enough to be a real paragraph for the test extraction purposes absolutely."
        result = searcher._extract_highlight(text)
        assert "long enough" in result

    def test_empty_content(self, searcher):
        """Empty content returns empty string."""
        result = searcher._extract_highlight("")
        assert result == ""

    def test_whitespace_only_content(self, searcher):
        """Whitespace-only content returns empty string."""
        result = searcher._extract_highlight("   \n\n   ")
        assert result == ""

    def test_custom_max_length(self, searcher):
        """Custom max_length is respected."""
        text = "Word " * 50  # 250 chars
        result = searcher._extract_highlight(text, max_length=50)
        assert len(result) <= 53  # 50 + "..."

    def test_question_mark_sentence_boundary(self, searcher):
        """Truncation respects question-mark sentence boundaries."""
        # The sentence ending must be past max_length//2 for rfind to find it
        text = "A" * 110 + "? " + "B" * 200
        result = searcher._extract_highlight(text, max_length=200)
        assert result.endswith("?")
        assert len(result) <= 200

    def test_exclamation_mark_sentence_boundary(self, searcher):
        """Truncation respects exclamation-mark sentence boundaries."""
        text = "A" * 110 + "! " + "B" * 200
        result = searcher._extract_highlight(text, max_length=200)
        assert result.endswith("!")


# =========================================================================
# TestReciprocalRankFusion
# =========================================================================


class TestReciprocalRankFusion:
    """Tests for _reciprocal_rank_fusion()."""

    def test_empty_inputs(self, searcher):
        """No results in either list yields empty output."""
        result = searcher._reciprocal_rank_fusion([], [])
        assert result == []

    def test_only_vector_results(self, searcher):
        """Only vector results: all should appear with normalized scores."""
        c1 = _make_chunk(chunk_id="c1")
        c2 = _make_chunk(chunk_id="c2")
        vector_results = [(c1, 0.9), (c2, 0.7)]
        result = searcher._reciprocal_rank_fusion(vector_results, [])
        assert len(result) == 2
        # First result should have score 1.0 (normalized max)
        assert result[0][0].id == "c1"
        assert result[0][1] == pytest.approx(1.0)

    def test_only_keyword_results(self, searcher):
        """Only keyword results: all should appear with normalized scores."""
        c1 = _make_chunk(chunk_id="c1")
        c2 = _make_chunk(chunk_id="c2")
        keyword_results = [(c1, 5.0), (c2, 3.0)]
        result = searcher._reciprocal_rank_fusion([], keyword_results)
        assert len(result) == 2
        assert result[0][0].id == "c1"
        assert result[0][1] == pytest.approx(1.0)

    def test_both_vector_and_keyword(self, searcher):
        """Both lists combined: should contain union of chunk IDs."""
        c1 = _make_chunk(chunk_id="c1")
        c2 = _make_chunk(chunk_id="c2")
        c3 = _make_chunk(chunk_id="c3")
        vector_results = [(c1, 0.9), (c2, 0.7)]
        keyword_results = [(c3, 5.0), (c1, 3.0)]
        result = searcher._reciprocal_rank_fusion(vector_results, keyword_results)
        result_ids = [r[0].id for r in result]
        assert set(result_ids) == {"c1", "c2", "c3"}

    def test_overlapping_chunks_get_combined_scores(self, searcher):
        """A chunk appearing in both lists gets boosted via combined RRF score."""
        c1 = _make_chunk(chunk_id="c1")
        c2 = _make_chunk(chunk_id="c2")
        # c1 appears in both lists so should get a higher combined score
        vector_results = [(c1, 0.9), (c2, 0.7)]
        keyword_results = [(c1, 5.0)]
        result = searcher._reciprocal_rank_fusion(vector_results, keyword_results)
        # c1 should be ranked first because it appears in both
        assert result[0][0].id == "c1"
        assert result[0][1] == pytest.approx(1.0)
        # c2 should have a lower score
        assert result[1][0].id == "c2"
        assert result[1][1] < 1.0

    def test_score_normalization_max_becomes_one(self, searcher):
        """The highest-scoring result is normalized to 1.0."""
        c1 = _make_chunk(chunk_id="c1")
        c2 = _make_chunk(chunk_id="c2")
        result = searcher._reciprocal_rank_fusion([(c1, 0.9)], [(c2, 5.0)])
        scores = [r[1] for r in result]
        assert max(scores) == pytest.approx(1.0)

    def test_custom_weights(self, searcher):
        """Custom vector_weight and keyword_weight affect scores."""
        c1 = _make_chunk(chunk_id="c1")
        c2 = _make_chunk(chunk_id="c2")
        # With high keyword_weight, keyword-only result should rank higher
        result = searcher._reciprocal_rank_fusion(
            [(c1, 0.9)],
            [(c2, 5.0)],
            vector_weight=0.1,
            keyword_weight=0.9,
        )
        # c2 (keyword only) should be ranked first
        assert result[0][0].id == "c2"

    def test_custom_k_parameter(self, searcher):
        """Custom k constant changes the RRF scoring curve."""
        c1 = _make_chunk(chunk_id="c1")
        c2 = _make_chunk(chunk_id="c2")
        # Small k = higher differentiation between ranks
        result_small_k = searcher._reciprocal_rank_fusion(
            [(c1, 0.9), (c2, 0.7)], [], k=1
        )
        # Large k = less differentiation
        result_large_k = searcher._reciprocal_rank_fusion(
            [(c1, 0.9), (c2, 0.7)], [], k=1000
        )
        # With small k, score ratio between rank 1 and rank 2 should be more extreme
        ratio_small = result_small_k[1][1] / result_small_k[0][1]
        ratio_large = result_large_k[1][1] / result_large_k[0][1]
        assert ratio_small < ratio_large

    def test_ranking_order_preserved_descending(self, searcher):
        """Results are sorted in descending score order."""
        chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(5)]
        vector_results = [(chunks[i], 1.0 - i * 0.1) for i in range(5)]
        result = searcher._reciprocal_rank_fusion(vector_results, [])
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_single_chunk_both_lists_score_is_one(self, searcher):
        """A single chunk in both lists normalizes to 1.0."""
        c1 = _make_chunk(chunk_id="c1")
        result = searcher._reciprocal_rank_fusion([(c1, 0.8)], [(c1, 3.0)])
        assert len(result) == 1
        assert result[0][1] == pytest.approx(1.0)


# =========================================================================
# TestFormatResults
# =========================================================================


class TestFormatResults:
    """Tests for _format_results()."""

    def test_empty_results(self, searcher):
        """Empty input yields empty output."""
        assert searcher._format_results([]) == []

    def test_single_result_with_citation(self, store):
        """Single result is formatted with chunk, document, similarity, highlight, citation."""
        chunk = _make_chunk()
        doc = _make_document()
        store.get_document = MagicMock(return_value=doc)
        searcher = DocumentSearcher(store=store, config=RetrievalConfig())

        results = searcher._format_results([(chunk, 0.85)])
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, DocumentSearchResult)
        assert r.chunk is chunk
        assert r.document is doc
        assert r.similarity == 0.85
        assert isinstance(r.highlight, str)
        assert isinstance(r.citation, Citation)
        assert r.citation.document_id == doc.id
        assert r.citation.document_title == "Test Book"
        assert r.citation.chunk_id == chunk.id
        assert r.citation.page_range == "pp. 1-3"
        assert r.citation.chapter == "Chapter 1"
        assert r.citation.section == "Introduction"
        assert r.citation.relevance == 0.85

    def test_missing_document_skipped(self, store):
        """If get_document returns None, that result is skipped."""
        chunk = _make_chunk()
        store.get_document = MagicMock(return_value=None)
        searcher = DocumentSearcher(store=store, config=RetrievalConfig())

        results = searcher._format_results([(chunk, 0.9)])
        assert results == []

    def test_multiple_results(self, store):
        """Multiple results are all formatted correctly."""
        c1 = _make_chunk(chunk_id="c1", document_id="doc-1")
        c2 = _make_chunk(chunk_id="c2", document_id="doc-2")
        doc1 = _make_document(doc_id="doc-1", title="Book One")
        doc2 = _make_document(doc_id="doc-2", title="Book Two")

        def get_doc(doc_id):
            return {"doc-1": doc1, "doc-2": doc2}.get(doc_id)

        store.get_document = MagicMock(side_effect=get_doc)
        searcher = DocumentSearcher(store=store, config=RetrievalConfig())

        results = searcher._format_results([(c1, 0.9), (c2, 0.7)])
        assert len(results) == 2
        assert results[0].citation.document_title == "Book One"
        assert results[1].citation.document_title == "Book Two"

    def test_partial_missing_documents(self, store):
        """Only results with valid documents are returned; others are skipped."""
        c1 = _make_chunk(chunk_id="c1", document_id="doc-1")
        c2 = _make_chunk(chunk_id="c2", document_id="doc-missing")
        c3 = _make_chunk(chunk_id="c3", document_id="doc-1")
        doc1 = _make_document(doc_id="doc-1")

        def get_doc(doc_id):
            if doc_id == "doc-1":
                return doc1
            return None

        store.get_document = MagicMock(side_effect=get_doc)
        searcher = DocumentSearcher(store=store, config=RetrievalConfig())

        results = searcher._format_results([(c1, 0.9), (c2, 0.8), (c3, 0.7)])
        assert len(results) == 2

    def test_citation_quote_matches_highlight(self, store):
        """Citation quote should match the extracted highlight."""
        chunk = _make_chunk(content="This is a test paragraph with real content.")
        doc = _make_document()
        store.get_document = MagicMock(return_value=doc)
        searcher = DocumentSearcher(store=store, config=RetrievalConfig())

        results = searcher._format_results([(chunk, 0.9)])
        assert results[0].citation.quote == results[0].highlight


# =========================================================================
# TestVectorSearch
# =========================================================================


class TestVectorSearch:
    """Tests for _vector_search()."""

    @pytest.mark.asyncio
    async def test_no_embedding_model_falls_back_to_keyword(self, store, config):
        """Without an embedding model, falls back to keyword search."""
        searcher = DocumentSearcher(store=store, config=config)
        c = _make_chunk()
        doc = _make_document()
        store.keyword_search = MagicMock(return_value=[(c, 3.0)])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher._vector_search("test query")
        store.keyword_search.assert_called_once_with(
            query="test query", document_id=None, top_k=10
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_encoding_failure_falls_back_to_keyword(self, store, config):
        """If encoding fails, falls back to keyword search."""
        searcher = DocumentSearcher(store=store, config=config)
        model = MagicMock()
        model.encode = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        searcher._embedding_model = model

        c = _make_chunk()
        doc = _make_document()
        store.keyword_search = MagicMock(return_value=[(c, 3.0)])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher._vector_search("test query")
        store.keyword_search.assert_called_once()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_qdrant_available_uses_qdrant(self, store, config):
        """When Qdrant is available, uses _qdrant_vector_search."""
        vs = _make_vector_store()
        searcher = DocumentSearcher(store=store, config=config, vector_store=vs)
        searcher._embedding_model = _make_embedding_model()

        # Create mock Qdrant results
        mock_result = MagicMock()
        mock_result.id = "chunk-1"
        mock_result.score = 0.92
        vs.search = AsyncMock(return_value=[mock_result])

        chunk = _make_chunk(chunk_id="chunk-1")
        doc = _make_document()
        store.get_chunk = MagicMock(return_value=chunk)
        store.get_document = MagicMock(return_value=doc)

        with patch(
            "documents.retrieval.searcher.COLLECTION_DOCUMENTS", "docs", create=True
        ), patch.dict(
            "sys.modules", {"db.vector_store": MagicMock(COLLECTION_DOCUMENTS="docs")}
        ):
            results = await searcher._vector_search("test query")

        vs.search.assert_called_once()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_qdrant_fails_falls_back_to_sqlite(self, store, config):
        """When Qdrant fails, falls back to SQLite vector search."""
        vs = _make_vector_store()
        vs.search = AsyncMock(side_effect=Exception("Connection refused"))
        searcher = DocumentSearcher(store=store, config=config, vector_store=vs)
        searcher._embedding_model = _make_embedding_model()

        chunk = _make_chunk()
        doc = _make_document()
        store.vector_search = MagicMock(return_value=[(chunk, 0.8)])
        store.get_document = MagicMock(return_value=doc)

        with patch.dict(
            "sys.modules", {"db.vector_store": MagicMock(COLLECTION_DOCUMENTS="docs")}
        ):
            results = await searcher._vector_search("test query")

        store.vector_search.assert_called_once()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_sqlite_vector_search(self, store, config):
        """Without Qdrant, uses SQLite brute-force vector search."""
        searcher = DocumentSearcher(store=store, config=config)
        searcher._embedding_model = _make_embedding_model()

        chunk = _make_chunk()
        doc = _make_document()
        store.vector_search = MagicMock(return_value=[(chunk, 0.75)])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher._vector_search("test query")

        store.vector_search.assert_called_once()
        call_kwargs = store.vector_search.call_args
        assert call_kwargs[1]["min_similarity"] == config.min_similarity
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_document_id_filter(self, store, config):
        """document_id is passed through to the store search methods."""
        searcher = DocumentSearcher(store=store, config=config)
        searcher._embedding_model = _make_embedding_model()

        store.vector_search = MagicMock(return_value=[])

        await searcher._vector_search("test", document_id="doc-42")

        call_kwargs = store.vector_search.call_args
        assert call_kwargs[1]["document_id"] == "doc-42"

    @pytest.mark.asyncio
    async def test_top_k_passed_through(self, store, config):
        """top_k parameter is forwarded to the store."""
        searcher = DocumentSearcher(store=store, config=config)
        searcher._embedding_model = _make_embedding_model()
        store.vector_search = MagicMock(return_value=[])

        await searcher._vector_search("test", top_k=3)

        call_kwargs = store.vector_search.call_args
        assert call_kwargs[1]["top_k"] == 3


# =========================================================================
# TestQdrantVectorSearch
# =========================================================================


class TestQdrantVectorSearch:
    """Tests for _qdrant_vector_search()."""

    @pytest.mark.asyncio
    async def test_basic_qdrant_search(self, store, config):
        """Qdrant search returns formatted results from matching chunks."""
        vs = _make_vector_store()
        searcher = DocumentSearcher(store=store, config=config, vector_store=vs)

        mock_r1 = MagicMock()
        mock_r1.id = "chunk-1"
        mock_r1.score = 0.95
        mock_r2 = MagicMock()
        mock_r2.id = "chunk-2"
        mock_r2.score = 0.80
        vs.search = AsyncMock(return_value=[mock_r1, mock_r2])

        c1 = _make_chunk(chunk_id="chunk-1")
        c2 = _make_chunk(chunk_id="chunk-2")
        doc = _make_document()

        def get_chunk(cid):
            return {"chunk-1": c1, "chunk-2": c2}.get(cid)

        store.get_chunk = MagicMock(side_effect=get_chunk)
        store.get_document = MagicMock(return_value=doc)

        with patch.dict(
            "sys.modules", {"db.vector_store": MagicMock(COLLECTION_DOCUMENTS="docs")}
        ):
            results = await searcher._qdrant_vector_search([0.1] * 768)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_qdrant_skips_missing_chunks(self, store, config):
        """Qdrant results with missing chunks in store are skipped."""
        vs = _make_vector_store()
        searcher = DocumentSearcher(store=store, config=config, vector_store=vs)

        mock_r1 = MagicMock()
        mock_r1.id = "chunk-exists"
        mock_r1.score = 0.9
        mock_r2 = MagicMock()
        mock_r2.id = "chunk-missing"
        mock_r2.score = 0.8
        vs.search = AsyncMock(return_value=[mock_r1, mock_r2])

        chunk = _make_chunk(chunk_id="chunk-exists")
        doc = _make_document()

        def get_chunk(cid):
            if cid == "chunk-exists":
                return chunk
            return None

        store.get_chunk = MagicMock(side_effect=get_chunk)
        store.get_document = MagicMock(return_value=doc)

        with patch.dict(
            "sys.modules", {"db.vector_store": MagicMock(COLLECTION_DOCUMENTS="docs")}
        ):
            results = await searcher._qdrant_vector_search([0.1] * 768)

        # Only 1 result because chunk-missing doesn't exist in the store
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_qdrant_document_id_filter(self, store, config):
        """document_id filter is passed to Qdrant search."""
        vs = _make_vector_store()
        vs.search = AsyncMock(return_value=[])
        searcher = DocumentSearcher(store=store, config=config, vector_store=vs)

        with patch.dict(
            "sys.modules", {"db.vector_store": MagicMock(COLLECTION_DOCUMENTS="docs")}
        ):
            await searcher._qdrant_vector_search([0.1] * 768, document_id="doc-42")

        call_kwargs = vs.search.call_args[1]
        assert call_kwargs["filters"] == {"document_id": "doc-42"}

    @pytest.mark.asyncio
    async def test_qdrant_no_document_id_no_filter(self, store, config):
        """Without document_id, filters is None in Qdrant call."""
        vs = _make_vector_store()
        vs.search = AsyncMock(return_value=[])
        searcher = DocumentSearcher(store=store, config=config, vector_store=vs)

        with patch.dict(
            "sys.modules", {"db.vector_store": MagicMock(COLLECTION_DOCUMENTS="docs")}
        ):
            await searcher._qdrant_vector_search([0.1] * 768)

        call_kwargs = vs.search.call_args[1]
        assert call_kwargs["filters"] is None


# =========================================================================
# TestKeywordSearch
# =========================================================================


class TestKeywordSearch:
    """Tests for _keyword_search()."""

    @pytest.mark.asyncio
    async def test_basic_keyword_search(self, store, config):
        """Keyword search delegates to store.keyword_search and formats results."""
        chunk = _make_chunk()
        doc = _make_document()
        store.keyword_search = MagicMock(return_value=[(chunk, 3.5)])
        store.get_document = MagicMock(return_value=doc)
        searcher = DocumentSearcher(store=store, config=config)

        results = await searcher._keyword_search("screenplay structure")
        store.keyword_search.assert_called_once_with(
            query="screenplay structure", document_id=None, top_k=10
        )
        assert len(results) == 1
        assert results[0].similarity == 3.5

    @pytest.mark.asyncio
    async def test_keyword_search_with_document_id(self, store, config):
        """document_id is passed through to store.keyword_search."""
        store.keyword_search = MagicMock(return_value=[])
        searcher = DocumentSearcher(store=store, config=config)

        await searcher._keyword_search("test", document_id="doc-7")
        call_kwargs = store.keyword_search.call_args
        assert call_kwargs[1]["document_id"] == "doc-7"

    @pytest.mark.asyncio
    async def test_keyword_search_with_top_k(self, store, config):
        """top_k is passed through to store.keyword_search."""
        store.keyword_search = MagicMock(return_value=[])
        searcher = DocumentSearcher(store=store, config=config)

        await searcher._keyword_search("test", top_k=3)
        call_kwargs = store.keyword_search.call_args
        assert call_kwargs[1]["top_k"] == 3

    @pytest.mark.asyncio
    async def test_keyword_search_empty_results(self, store, config):
        """No matches returns empty list."""
        store.keyword_search = MagicMock(return_value=[])
        searcher = DocumentSearcher(store=store, config=config)

        results = await searcher._keyword_search("nonexistent_term_xyz")
        assert results == []


# =========================================================================
# TestHybridSearch
# =========================================================================


class TestHybridSearch:
    """Tests for _hybrid_search()."""

    @pytest.mark.asyncio
    async def test_both_vector_and_keyword_combined(self, store):
        """Hybrid search runs both vector and keyword, combining with RRF."""
        config = RetrievalConfig(min_similarity=0.0)  # No filtering so both appear
        searcher = DocumentSearcher(store=store, config=config)
        searcher._embedding_model = _make_embedding_model()

        cv = _make_chunk(chunk_id="c-vec", document_id="doc-1")
        ck = _make_chunk(chunk_id="c-key", document_id="doc-1")
        doc = _make_document()

        store.vector_search = MagicMock(return_value=[(cv, 0.9)])
        store.keyword_search = MagicMock(return_value=[(ck, 5.0)])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher._hybrid_search("test query")

        store.vector_search.assert_called_once()
        store.keyword_search.assert_called_once()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_no_embedding_model_keyword_only(self, store, config):
        """Without embedding model, hybrid uses keyword search only."""
        searcher = DocumentSearcher(store=store, config=config)
        # No embedding model set

        chunk = _make_chunk()
        doc = _make_document()
        store.keyword_search = MagicMock(return_value=[(chunk, 3.0)])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher._hybrid_search("test query")

        store.vector_search.assert_not_called()
        store.keyword_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_document_type_filter(self, store, config):
        """document_type filter removes non-matching results."""
        searcher = DocumentSearcher(store=store, config=config)
        # No embedding model - uses keyword only

        c1 = _make_chunk(chunk_id="c1", document_id="doc-book")
        c2 = _make_chunk(chunk_id="c2", document_id="doc-article")
        doc_book = _make_document(doc_id="doc-book", document_type=DocumentType.BOOK)
        doc_article = _make_document(
            doc_id="doc-article", document_type=DocumentType.ARTICLE
        )

        store.keyword_search = MagicMock(return_value=[(c1, 5.0), (c2, 4.0)])

        def get_doc(doc_id):
            return {"doc-book": doc_book, "doc-article": doc_article}.get(doc_id)

        store.get_document = MagicMock(side_effect=get_doc)

        results = await searcher._hybrid_search("test", document_type=DocumentType.BOOK)
        assert len(results) == 1
        assert results[0].document.document_type == DocumentType.BOOK

    @pytest.mark.asyncio
    async def test_project_filter(self, store, config):
        """project filter removes non-matching results."""
        searcher = DocumentSearcher(store=store, config=config)

        c1 = _make_chunk(chunk_id="c1", document_id="doc-p1")
        c2 = _make_chunk(chunk_id="c2", document_id="doc-p2")
        doc_p1 = _make_document(doc_id="doc-p1", project="project-alpha")
        doc_p2 = _make_document(doc_id="doc-p2", project="project-beta")

        store.keyword_search = MagicMock(return_value=[(c1, 5.0), (c2, 4.0)])

        def get_doc(doc_id):
            return {"doc-p1": doc_p1, "doc-p2": doc_p2}.get(doc_id)

        store.get_document = MagicMock(side_effect=get_doc)

        results = await searcher._hybrid_search("test", project="project-alpha")
        assert len(results) == 1
        assert results[0].document.project == "project-alpha"

    @pytest.mark.asyncio
    async def test_min_similarity_filter(self, store):
        """Results below min_similarity are filtered out."""
        config = RetrievalConfig(min_similarity=0.8)
        searcher = DocumentSearcher(store=store, config=config)
        searcher._embedding_model = _make_embedding_model()

        # Use both vector and keyword with non-overlapping results to create
        # a wide score gap. The top chunk (in both lists) normalizes to 1.0.
        # A keyword-only chunk at rank 2 with k=60 gets a much lower score.
        c_top = _make_chunk(chunk_id="c-top")
        c_low = _make_chunk(chunk_id="c-low")
        doc = _make_document()

        # c_top in vector rank-1, c_low not in vector at all
        store.vector_search = MagicMock(return_value=[(c_top, 0.95)])
        # c_top in keyword rank-1, c_low in keyword rank-2
        store.keyword_search = MagicMock(return_value=[(c_top, 5.0), (c_low, 4.0)])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher._hybrid_search("test")

        # c_top: vector_rrf=0.7/61 + keyword_rrf=0.3/61 = 1.0/61 (normalized to 1.0)
        # c_low: keyword_rrf=0.3/62 only, normalized ~ 0.3/62 / (1.0/61) ~ 0.295
        # So c_low < 0.8 => filtered out
        assert len(results) == 1
        assert results[0].similarity >= 0.8
        assert results[0].chunk.id == "c-top"

    @pytest.mark.asyncio
    async def test_top_k_limit(self, store, config):
        """Only top_k results are returned even when more are available."""
        searcher = DocumentSearcher(store=store, config=config)

        chunks = [_make_chunk(chunk_id=f"c{i}", document_id="doc-1") for i in range(20)]
        doc = _make_document()
        store.keyword_search = MagicMock(
            return_value=[(c, 5.0 - i * 0.1) for i, c in enumerate(chunks)]
        )
        store.get_document = MagicMock(return_value=doc)

        results = await searcher._hybrid_search("test", top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_fetch_k_is_doubled(self, store, config):
        """hybrid_search fetches top_k*2 candidates for merging."""
        searcher = DocumentSearcher(store=store, config=config)
        store.keyword_search = MagicMock(return_value=[])

        await searcher._hybrid_search("test", top_k=5)
        call_kwargs = store.keyword_search.call_args
        assert call_kwargs[1]["top_k"] == 10  # 5 * 2

    @pytest.mark.asyncio
    async def test_document_type_and_project_combined(self, store, config):
        """Both document_type and project filters work together."""
        searcher = DocumentSearcher(store=store, config=config)

        c1 = _make_chunk(chunk_id="c1", document_id="doc-match")
        c2 = _make_chunk(chunk_id="c2", document_id="doc-wrong-type")
        c3 = _make_chunk(chunk_id="c3", document_id="doc-wrong-project")

        doc_match = _make_document(
            doc_id="doc-match",
            document_type=DocumentType.BOOK,
            project="alpha",
        )
        doc_wrong_type = _make_document(
            doc_id="doc-wrong-type",
            document_type=DocumentType.ARTICLE,
            project="alpha",
        )
        doc_wrong_project = _make_document(
            doc_id="doc-wrong-project",
            document_type=DocumentType.BOOK,
            project="beta",
        )

        store.keyword_search = MagicMock(return_value=[(c1, 5.0), (c2, 4.0), (c3, 3.0)])

        def get_doc(doc_id):
            return {
                "doc-match": doc_match,
                "doc-wrong-type": doc_wrong_type,
                "doc-wrong-project": doc_wrong_project,
            }.get(doc_id)

        store.get_document = MagicMock(side_effect=get_doc)

        results = await searcher._hybrid_search(
            "test",
            document_type=DocumentType.BOOK,
            project="alpha",
        )
        assert len(results) == 1
        assert results[0].document.id == "doc-match"


# =========================================================================
# TestSearch
# =========================================================================


class TestSearch:
    """Tests for the top-level search() method."""

    @pytest.mark.asyncio
    async def test_hybrid_search_when_enabled(self, store):
        """When use_hybrid_search=True, delegates to _hybrid_search."""
        config = RetrievalConfig(use_hybrid_search=True)
        searcher = DocumentSearcher(store=store, config=config)

        with patch.object(
            searcher, "_hybrid_search", new_callable=AsyncMock
        ) as mock_hybrid:
            mock_hybrid.return_value = []
            await searcher.search("test query")

            mock_hybrid.assert_called_once_with("test query", None, None, None, 10)

    @pytest.mark.asyncio
    async def test_vector_search_when_hybrid_disabled(self, store):
        """When use_hybrid_search=False, delegates to _vector_search."""
        config = RetrievalConfig(use_hybrid_search=False)
        searcher = DocumentSearcher(store=store, config=config)

        with patch.object(
            searcher, "_vector_search", new_callable=AsyncMock
        ) as mock_vec:
            mock_vec.return_value = []
            await searcher.search("test query")

            mock_vec.assert_called_once_with("test query", None, 10)

    @pytest.mark.asyncio
    async def test_custom_top_k(self, store):
        """Custom top_k overrides config default."""
        config = RetrievalConfig(use_hybrid_search=True, vector_search_top_k=10)
        searcher = DocumentSearcher(store=store, config=config)

        with patch.object(
            searcher, "_hybrid_search", new_callable=AsyncMock
        ) as mock_hybrid:
            mock_hybrid.return_value = []
            await searcher.search("test", top_k=3)

            mock_hybrid.assert_called_once_with("test", None, None, None, 3)

    @pytest.mark.asyncio
    async def test_default_top_k_from_config(self, store):
        """When top_k is None, uses config.vector_search_top_k."""
        config = RetrievalConfig(use_hybrid_search=True, vector_search_top_k=25)
        searcher = DocumentSearcher(store=store, config=config)

        with patch.object(
            searcher, "_hybrid_search", new_callable=AsyncMock
        ) as mock_hybrid:
            mock_hybrid.return_value = []
            await searcher.search("test")

            mock_hybrid.assert_called_once_with("test", None, None, None, 25)

    @pytest.mark.asyncio
    async def test_search_passes_all_filters(self, store):
        """document_id, document_type, project are all forwarded."""
        config = RetrievalConfig(use_hybrid_search=True)
        searcher = DocumentSearcher(store=store, config=config)

        with patch.object(
            searcher, "_hybrid_search", new_callable=AsyncMock
        ) as mock_hybrid:
            mock_hybrid.return_value = []
            await searcher.search(
                "test",
                document_id="doc-1",
                document_type=DocumentType.SCREENPLAY,
                project="my-project",
                top_k=7,
            )

            mock_hybrid.assert_called_once_with(
                "test", "doc-1", DocumentType.SCREENPLAY, "my-project", 7
            )


# =========================================================================
# TestSearchInChapter
# =========================================================================


class TestSearchInChapter:
    """Tests for search_in_chapter()."""

    @pytest.mark.asyncio
    async def test_basic_chapter_search(self, store, config):
        """search_in_chapter returns results within the specified chapter."""
        searcher = DocumentSearcher(store=store, config=config)
        searcher._embedding_model = _make_embedding_model()

        # Make the encode return a known normalized vector
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        searcher._embedding_model.encode = MagicMock(return_value=query_vec)

        # Create chunks with embeddings that have known similarity
        c1 = _make_chunk(
            chunk_id="c1",
            embedding=[1.0, 0.0, 0.0],  # similarity=1.0
            chapter="Chapter 1",
        )
        c2 = _make_chunk(
            chunk_id="c2",
            embedding=[0.0, 1.0, 0.0],  # similarity=0.0
            chapter="Chapter 1",
        )
        doc = _make_document()

        store.get_chunks_for_document = MagicMock(return_value=[c1, c2])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher.search_in_chapter(
            "test", document_id="doc-1", chapter="Chapter 1"
        )

        store.get_chunks_for_document.assert_called_once_with(
            "doc-1", chapter="Chapter 1"
        )
        # Only c1 should pass min_similarity=0.5 default
        assert len(results) == 1
        assert results[0].chunk.id == "c1"

    @pytest.mark.asyncio
    async def test_no_chunks_returns_empty(self, store, config):
        """No chunks for the chapter returns empty list."""
        searcher = DocumentSearcher(store=store, config=config)
        searcher._embedding_model = _make_embedding_model()
        store.get_chunks_for_document = MagicMock(return_value=[])

        results = await searcher.search_in_chapter(
            "test", document_id="doc-1", chapter="Nonexistent"
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_no_embedding_model_returns_empty(self, store, config):
        """Without embedding model, returns empty list."""
        searcher = DocumentSearcher(store=store, config=config)
        # No embedding model
        store.get_chunks_for_document = MagicMock(return_value=[_make_chunk()])

        results = await searcher.search_in_chapter(
            "test", document_id="doc-1", chapter="Chapter 1"
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_similarity_filtering(self, store):
        """Chunks below min_similarity are excluded."""
        config = RetrievalConfig(min_similarity=0.7)
        searcher = DocumentSearcher(store=store, config=config)

        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        model = MagicMock()
        model.encode = MagicMock(return_value=query_vec)
        searcher._embedding_model = model

        c_high = _make_chunk(chunk_id="c-high", embedding=[0.9, 0.4, 0.1])
        c_low = _make_chunk(chunk_id="c-low", embedding=[0.3, 0.9, 0.1])
        doc = _make_document()

        store.get_chunks_for_document = MagicMock(return_value=[c_high, c_low])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher.search_in_chapter(
            "test", document_id="doc-1", chapter="Chapter 1"
        )

        # c_high: dot([1,0,0], [0.9,0.4,0.1]) = 0.9 >= 0.7 => included
        # c_low:  dot([1,0,0], [0.3,0.9,0.1]) = 0.3 < 0.7 => excluded
        assert len(results) == 1
        assert results[0].chunk.id == "c-high"

    @pytest.mark.asyncio
    async def test_top_k_limit_respected(self, store, config):
        """Only top_k results are returned, sorted by similarity."""
        searcher = DocumentSearcher(store=store, config=config)

        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        model = MagicMock()
        model.encode = MagicMock(return_value=query_vec)
        searcher._embedding_model = model

        # Create 10 chunks with descending similarity
        chunks = []
        for i in range(10):
            sim_val = 1.0 - i * 0.05
            embedding = [sim_val, 0.0, 0.0]
            chunks.append(_make_chunk(chunk_id=f"c{i}", embedding=embedding))
        doc = _make_document()

        store.get_chunks_for_document = MagicMock(return_value=chunks)
        store.get_document = MagicMock(return_value=doc)

        results = await searcher.search_in_chapter(
            "test", document_id="doc-1", chapter="Chapter 1", top_k=3
        )
        assert len(results) == 3
        # Verify descending order
        assert results[0].similarity >= results[1].similarity
        assert results[1].similarity >= results[2].similarity

    @pytest.mark.asyncio
    async def test_chunks_without_embedding_skipped(self, store, config):
        """Chunks with no embedding are skipped in similarity calculation."""
        searcher = DocumentSearcher(store=store, config=config)

        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        model = MagicMock()
        model.encode = MagicMock(return_value=query_vec)
        searcher._embedding_model = model

        c_with = _make_chunk(chunk_id="c-with", embedding=[1.0, 0.0, 0.0])
        c_without = _make_chunk(chunk_id="c-without", embedding=None)
        doc = _make_document()

        store.get_chunks_for_document = MagicMock(return_value=[c_with, c_without])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher.search_in_chapter(
            "test", document_id="doc-1", chapter="Chapter 1"
        )
        assert len(results) == 1
        assert results[0].chunk.id == "c-with"

    @pytest.mark.asyncio
    async def test_encoding_failure_returns_empty(self, store, config):
        """If encoding the query fails, returns empty list."""
        searcher = DocumentSearcher(store=store, config=config)
        model = MagicMock()
        model.encode = MagicMock(side_effect=RuntimeError("encode failed"))
        searcher._embedding_model = model

        store.get_chunks_for_document = MagicMock(
            return_value=[_make_chunk(embedding=[1.0, 0.0])]
        )

        results = await searcher.search_in_chapter(
            "test", document_id="doc-1", chapter="Chapter 1"
        )
        assert results == []


# =========================================================================
# Integration-style tests
# =========================================================================


class TestSearchIntegration:
    """Higher-level integration-style tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_hybrid_pipeline(self, store):
        """End-to-end: search() -> _hybrid_search -> RRF -> format."""
        config = RetrievalConfig(
            use_hybrid_search=True,
            keyword_weight=0.3,
            min_similarity=0.0,
            vector_search_top_k=5,
        )
        searcher = DocumentSearcher(store=store, config=config)
        searcher._embedding_model = _make_embedding_model()

        # Vector results
        cv1 = _make_chunk(
            chunk_id="cv1", document_id="doc-1", content="Vector match alpha."
        )
        cv2 = _make_chunk(
            chunk_id="cv2", document_id="doc-1", content="Vector match beta."
        )

        # Keyword results (cv1 also appears here for boost)
        ck1 = cv1  # Same chunk in both
        ck2 = _make_chunk(
            chunk_id="ck2", document_id="doc-1", content="Keyword only result."
        )

        doc = _make_document()

        store.vector_search = MagicMock(return_value=[(cv1, 0.9), (cv2, 0.7)])
        store.keyword_search = MagicMock(return_value=[(ck1, 5.0), (ck2, 3.0)])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher.search("test query")

        assert len(results) > 0
        # cv1 appears in both, should be ranked highest
        assert results[0].chunk.id == "cv1"
        # All results should have proper citations
        for r in results:
            assert r.citation is not None
            assert r.citation.document_title == "Test Book"

    @pytest.mark.asyncio
    async def test_full_vector_only_pipeline(self, store):
        """End-to-end with hybrid disabled: search() -> _vector_search -> format."""
        config = RetrievalConfig(use_hybrid_search=False, min_similarity=0.0)
        searcher = DocumentSearcher(store=store, config=config)
        searcher._embedding_model = _make_embedding_model()

        chunk = _make_chunk(content="A relevant passage about storytelling.")
        doc = _make_document()

        store.vector_search = MagicMock(return_value=[(chunk, 0.85)])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher.search("storytelling")
        assert len(results) == 1
        assert results[0].similarity == 0.85
        assert "storytelling" in results[0].highlight

    @pytest.mark.asyncio
    async def test_search_with_no_model_end_to_end(self, store):
        """End-to-end with no embedding model: falls back to keyword entirely."""
        config = RetrievalConfig(use_hybrid_search=True, min_similarity=0.0)
        searcher = DocumentSearcher(store=store, config=config)
        # No embedding model

        chunk = _make_chunk(content="A keyword match on structure.")
        doc = _make_document()

        store.keyword_search = MagicMock(return_value=[(chunk, 4.0)])
        store.get_document = MagicMock(return_value=doc)

        results = await searcher.search("structure")
        assert len(results) == 1
        # Should only have used keyword, not vector
        store.vector_search.assert_not_called()
