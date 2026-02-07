"""
Tests for Document Processing Configuration
=============================================

Comprehensive tests for documents/config.py covering all dataclasses,
YAML loading, environment variable overrides, singleton behavior, and
serialization.

Run with: pytest tests/test_document_config.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import documents.config as cfg_mod
from documents.config import (
    DATA_DIR,
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    ChunkingConfig,
    ComprehensionConfig,
    DocumentConfig,
    EmbeddingConfig,
    IntegrationConfig,
    OCRConfig,
    RetrievalConfig,
    StorageConfig,
    get_document_config,
    reset_document_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton before and after every test."""
    from documents.config import reset_document_config

    reset_document_config()
    yield
    reset_document_config()


@pytest.fixture()
def clean_env():
    """Remove DOCUMENT_OCR_FALLBACK_API_KEY env var that may leak."""
    keys = ["DOCUMENT_OCR_FALLBACK_API_KEY"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ===========================================================================
# 1. Module-level constants
# ===========================================================================


class TestModuleLevelConstants:
    """Tests for REPO_ROOT, DEFAULT_CONFIG_PATH, DATA_DIR."""

    def test_repo_root_is_parent_of_documents(self):
        documents_dir = REPO_ROOT / "documents"
        assert documents_dir.is_dir()

    def test_repo_root_is_absolute(self):
        assert REPO_ROOT.is_absolute()

    def test_default_config_path_under_repo_root(self):
        assert DEFAULT_CONFIG_PATH == REPO_ROOT / "config" / "document_config.yaml"

    def test_data_dir_is_under_documents(self):
        assert DATA_DIR.name == "data"
        assert DATA_DIR.parent.name == "documents"


# ===========================================================================
# 2. OCRConfig defaults
# ===========================================================================


class TestOCRConfigDefaults:
    """Every field of OCRConfig must have the documented default."""

    def test_model_path_default(self):
        assert OCRConfig().model_path == "deepseek-ai/DeepSeek-OCR-2"

    def test_use_flash_attention_default(self):
        assert OCRConfig().use_flash_attention is True

    def test_quantization_default(self):
        assert OCRConfig().quantization == "4bit"

    def test_max_batch_size_default(self):
        assert OCRConfig().max_batch_size == 4

    def test_device_default(self):
        assert OCRConfig().device == "cuda"

    def test_timeout_seconds_default(self):
        assert OCRConfig().timeout_seconds == 60.0

    def test_image_dpi_default(self):
        assert OCRConfig().image_dpi == 300

    def test_max_image_dimension_default(self):
        assert OCRConfig().max_image_dimension == 2048

    def test_fallback_enabled_default(self):
        assert OCRConfig().fallback_enabled is True

    def test_fallback_provider_default(self):
        assert OCRConfig().fallback_provider == "google"

    def test_fallback_api_key_default(self):
        assert OCRConfig().fallback_api_key == ""

    def test_retry_on_low_confidence_default(self):
        assert OCRConfig().retry_on_low_confidence is True

    def test_low_confidence_threshold_default(self):
        assert OCRConfig().low_confidence_threshold == 0.5


# ===========================================================================
# 3. OCRConfig.from_dict
# ===========================================================================


class TestOCRConfigFromDict:
    """Tests for OCRConfig.from_dict including alternate keys and env vars."""

    def test_empty_dict_returns_defaults(self, clean_env):
        cfg = OCRConfig.from_dict({})
        assert cfg.model_path == "deepseek-ai/DeepSeek-OCR-2"
        assert cfg.device == "cuda"
        assert cfg.fallback_api_key == ""

    def test_custom_values(self, clean_env):
        cfg = OCRConfig.from_dict(
            {
                "model_path": "custom/model",
                "use_flash_attention": False,
                "quantization": "8bit",
                "max_batch_size": 8,
                "device": "cpu",
                "timeout_seconds": 120.0,
                "image_dpi": 600,
                "max_image_dimension": 4096,
                "fallback_enabled": False,
                "fallback_provider": "azure",
                "fallback_api_key": "my-key",
                "retry_on_low_confidence": False,
                "low_confidence_threshold": 0.8,
            }
        )
        assert cfg.model_path == "custom/model"
        assert cfg.use_flash_attention is False
        assert cfg.quantization == "8bit"
        assert cfg.max_batch_size == 8
        assert cfg.device == "cpu"
        assert cfg.timeout_seconds == 120.0
        assert cfg.image_dpi == 600
        assert cfg.max_image_dimension == 4096
        assert cfg.fallback_enabled is False
        assert cfg.fallback_provider == "azure"
        assert cfg.fallback_api_key == "my-key"
        assert cfg.retry_on_low_confidence is False
        assert cfg.low_confidence_threshold == 0.8

    def test_alternate_key_model(self, clean_env):
        """'model' key should map to model_path."""
        cfg = OCRConfig.from_dict({"model": "alt/model"})
        assert cfg.model_path == "alt/model"

    def test_model_path_takes_precedence_over_model_key(self, clean_env):
        """When both 'model' and 'model_path' are present, 'model' wins (first in get chain)."""
        cfg = OCRConfig.from_dict({"model": "primary", "model_path": "secondary"})
        assert cfg.model_path == "primary"

    def test_env_var_fallback_api_key(self, clean_env):
        """DOCUMENT_OCR_FALLBACK_API_KEY env var should be used when key not in dict."""
        with patch.dict(os.environ, {"DOCUMENT_OCR_FALLBACK_API_KEY": "env-key"}):
            cfg = OCRConfig.from_dict({})
            assert cfg.fallback_api_key == "env-key"

    def test_dict_api_key_overrides_env_var(self):
        """When fallback_api_key is provided in dict, it takes precedence over env."""
        with patch.dict(os.environ, {"DOCUMENT_OCR_FALLBACK_API_KEY": "env-key"}):
            cfg = OCRConfig.from_dict({"fallback_api_key": "dict-key"})
            assert cfg.fallback_api_key == "dict-key"

    def test_unknown_keys_are_ignored(self, clean_env):
        cfg = OCRConfig.from_dict({"unknown_field": "value"})
        assert cfg.model_path == "deepseek-ai/DeepSeek-OCR-2"


# ===========================================================================
# 4. ChunkingConfig defaults
# ===========================================================================


class TestChunkingConfigDefaults:
    """Every field of ChunkingConfig must have the documented default."""

    def test_strategy_default(self):
        assert ChunkingConfig().strategy == "semantic"

    def test_min_chunk_chars_default(self):
        assert ChunkingConfig().min_chunk_chars == 500

    def test_max_chunk_chars_default(self):
        assert ChunkingConfig().max_chunk_chars == 2000

    def test_overlap_chars_default(self):
        assert ChunkingConfig().overlap_chars == 100

    def test_respect_chapters_default(self):
        assert ChunkingConfig().respect_chapters is True

    def test_respect_sections_default(self):
        assert ChunkingConfig().respect_sections is True

    def test_detect_headers_default(self):
        assert ChunkingConfig().detect_headers is True

    def test_screenplay_mode_default(self):
        assert ChunkingConfig().screenplay_mode is False

    def test_dialogue_grouping_default(self):
        assert ChunkingConfig().dialogue_grouping is True

    def test_chapter_patterns_default(self):
        patterns = ChunkingConfig().chapter_patterns
        assert isinstance(patterns, list)
        assert len(patterns) == 4
        assert r"^Chapter\s+\d+" in patterns
        assert r"^CHAPTER\s+\d+" in patterns
        assert r"^Part\s+\d+" in patterns
        assert r"^\d+\.\s+[A-Z]" in patterns

    def test_chapter_patterns_default_factory_creates_separate_lists(self):
        """Each instance gets its own list (not shared)."""
        a = ChunkingConfig()
        b = ChunkingConfig()
        a.chapter_patterns.append("new-pattern")
        assert "new-pattern" not in b.chapter_patterns


# ===========================================================================
# 5. ChunkingConfig.from_dict
# ===========================================================================


class TestChunkingConfigFromDict:

    def test_empty_dict_returns_defaults(self):
        cfg = ChunkingConfig.from_dict({})
        assert cfg.strategy == "semantic"
        assert cfg.min_chunk_chars == 500
        assert cfg.max_chunk_chars == 2000

    def test_custom_values(self):
        cfg = ChunkingConfig.from_dict(
            {
                "strategy": "fixed",
                "min_chunk_chars": 200,
                "max_chunk_chars": 5000,
                "overlap_chars": 50,
                "respect_chapters": False,
                "respect_sections": False,
                "detect_headers": False,
                "screenplay_mode": True,
                "dialogue_grouping": False,
                "chapter_patterns": [r"^ACT\s+\d+"],
            }
        )
        assert cfg.strategy == "fixed"
        assert cfg.min_chunk_chars == 200
        assert cfg.max_chunk_chars == 5000
        assert cfg.overlap_chars == 50
        assert cfg.respect_chapters is False
        assert cfg.respect_sections is False
        assert cfg.detect_headers is False
        assert cfg.screenplay_mode is True
        assert cfg.dialogue_grouping is False
        assert cfg.chapter_patterns == [r"^ACT\s+\d+"]

    def test_alternate_key_min_chars(self):
        """'min_chars' key should map to min_chunk_chars."""
        cfg = ChunkingConfig.from_dict({"min_chars": 300})
        assert cfg.min_chunk_chars == 300

    def test_alternate_key_max_chars(self):
        """'max_chars' key should map to max_chunk_chars."""
        cfg = ChunkingConfig.from_dict({"max_chars": 3000})
        assert cfg.max_chunk_chars == 3000

    def test_min_chunk_chars_key_used_when_min_chars_absent(self):
        cfg = ChunkingConfig.from_dict({"min_chunk_chars": 400})
        assert cfg.min_chunk_chars == 400

    def test_min_chars_takes_precedence_over_min_chunk_chars(self):
        """When both are present, 'min_chars' wins (first in get chain)."""
        cfg = ChunkingConfig.from_dict({"min_chars": 100, "min_chunk_chars": 900})
        assert cfg.min_chunk_chars == 100

    def test_max_chars_takes_precedence_over_max_chunk_chars(self):
        """When both are present, 'max_chars' wins (first in get chain)."""
        cfg = ChunkingConfig.from_dict({"max_chars": 1000, "max_chunk_chars": 9000})
        assert cfg.max_chunk_chars == 1000


# ===========================================================================
# 6. EmbeddingConfig defaults
# ===========================================================================


class TestEmbeddingConfigDefaults:

    def test_model_name_default(self):
        assert EmbeddingConfig().model_name == "paraphrase-multilingual-mpnet-base-v2"

    def test_embedding_dim_default(self):
        assert EmbeddingConfig().embedding_dim == 768

    def test_batch_size_default(self):
        assert EmbeddingConfig().batch_size == 32

    def test_normalize_default(self):
        assert EmbeddingConfig().normalize is True

    def test_share_gpu_with_ocr_default(self):
        assert EmbeddingConfig().share_gpu_with_ocr is False

    def test_gpu_memory_fraction_default(self):
        assert EmbeddingConfig().gpu_memory_fraction == 0.5


# ===========================================================================
# 7. EmbeddingConfig.from_dict
# ===========================================================================


class TestEmbeddingConfigFromDict:

    def test_empty_dict_returns_defaults(self):
        cfg = EmbeddingConfig.from_dict({})
        assert cfg.model_name == "paraphrase-multilingual-mpnet-base-v2"

    def test_custom_values(self):
        cfg = EmbeddingConfig.from_dict(
            {
                "model_name": "custom-embed-model",
                "embedding_dim": 1024,
                "batch_size": 64,
                "normalize": False,
                "share_gpu_with_ocr": True,
                "gpu_memory_fraction": 0.8,
            }
        )
        assert cfg.model_name == "custom-embed-model"
        assert cfg.embedding_dim == 1024
        assert cfg.batch_size == 64
        assert cfg.normalize is False
        assert cfg.share_gpu_with_ocr is True
        assert cfg.gpu_memory_fraction == 0.8

    def test_alternate_key_model(self):
        """'model' key should map to model_name."""
        cfg = EmbeddingConfig.from_dict({"model": "alt-embed-model"})
        assert cfg.model_name == "alt-embed-model"

    def test_model_key_takes_precedence_over_model_name(self):
        """When both 'model' and 'model_name' are present, 'model' wins."""
        cfg = EmbeddingConfig.from_dict({"model": "primary", "model_name": "secondary"})
        assert cfg.model_name == "primary"


# ===========================================================================
# 8. StorageConfig defaults
# ===========================================================================


class TestStorageConfigDefaults:

    def test_db_path_default(self):
        assert StorageConfig().db_path == str(DATA_DIR / "documents.db")

    def test_documents_dir_default(self):
        assert StorageConfig().documents_dir == str(DATA_DIR / "raw")

    def test_images_dir_default(self):
        assert StorageConfig().images_dir == str(DATA_DIR / "images")

    def test_cache_enabled_default(self):
        assert StorageConfig().cache_enabled is True

    def test_cache_ttl_hours_default(self):
        assert StorageConfig().cache_ttl_hours == 24

    def test_auto_cleanup_images_default(self):
        assert StorageConfig().auto_cleanup_images is True

    def test_max_storage_gb_default(self):
        assert StorageConfig().max_storage_gb == 10.0


# ===========================================================================
# 9. StorageConfig.from_dict
# ===========================================================================


class TestStorageConfigFromDict:

    def test_empty_dict_returns_defaults(self):
        cfg = StorageConfig.from_dict({})
        assert cfg.db_path == str(DATA_DIR / "documents.db")
        assert cfg.documents_dir == str(DATA_DIR / "raw")
        assert cfg.images_dir == str(DATA_DIR / "images")

    def test_custom_values(self):
        cfg = StorageConfig.from_dict(
            {
                "db_path": "/custom/path.db",
                "documents_dir": "/custom/docs",
                "images_dir": "/custom/imgs",
                "cache_enabled": False,
                "cache_ttl_hours": 48,
                "auto_cleanup_images": False,
                "max_storage_gb": 50.0,
            }
        )
        assert cfg.db_path == "/custom/path.db"
        assert cfg.documents_dir == "/custom/docs"
        assert cfg.images_dir == "/custom/imgs"
        assert cfg.cache_enabled is False
        assert cfg.cache_ttl_hours == 48
        assert cfg.auto_cleanup_images is False
        assert cfg.max_storage_gb == 50.0

    def test_data_dir_key_sets_base_paths(self, tmp_path):
        """'data_dir' key sets the base directory for db_path, documents_dir, images_dir."""
        custom_dir = str(tmp_path / "mydata")
        cfg = StorageConfig.from_dict({"data_dir": custom_dir})
        expected_data_dir = Path(custom_dir)
        assert cfg.db_path == str(expected_data_dir / "documents.db")
        assert cfg.documents_dir == str(expected_data_dir / "raw")
        assert cfg.images_dir == str(expected_data_dir / "images")

    def test_explicit_paths_override_data_dir(self, tmp_path):
        """Explicit db_path/documents_dir/images_dir override the data_dir-based defaults."""
        custom_dir = str(tmp_path / "mydata")
        cfg = StorageConfig.from_dict(
            {
                "data_dir": custom_dir,
                "db_path": "/override/db.sqlite",
                "documents_dir": "/override/docs",
                "images_dir": "/override/imgs",
            }
        )
        assert cfg.db_path == "/override/db.sqlite"
        assert cfg.documents_dir == "/override/docs"
        assert cfg.images_dir == "/override/imgs"


# ===========================================================================
# 10. RetrievalConfig defaults
# ===========================================================================


class TestRetrievalConfigDefaults:

    def test_vector_search_top_k_default(self):
        assert RetrievalConfig().vector_search_top_k == 10

    def test_min_similarity_default(self):
        assert RetrievalConfig().min_similarity == 0.5

    def test_use_hybrid_search_default(self):
        assert RetrievalConfig().use_hybrid_search is True

    def test_keyword_weight_default(self):
        assert RetrievalConfig().keyword_weight == 0.3

    def test_rerank_enabled_default(self):
        assert RetrievalConfig().rerank_enabled is False

    def test_rerank_model_default(self):
        assert RetrievalConfig().rerank_model == ""

    def test_include_citations_default(self):
        assert RetrievalConfig().include_citations is True

    def test_citation_style_default(self):
        assert RetrievalConfig().citation_style == "inline"

    def test_max_quote_length_default(self):
        assert RetrievalConfig().max_quote_length == 200


# ===========================================================================
# 11. RetrievalConfig.from_dict
# ===========================================================================


class TestRetrievalConfigFromDict:

    def test_empty_dict_returns_defaults(self):
        cfg = RetrievalConfig.from_dict({})
        assert cfg.vector_search_top_k == 10
        assert cfg.min_similarity == 0.5

    def test_custom_values(self):
        cfg = RetrievalConfig.from_dict(
            {
                "vector_search_top_k": 20,
                "min_similarity": 0.7,
                "use_hybrid_search": False,
                "keyword_weight": 0.5,
                "rerank_enabled": True,
                "rerank_model": "cross-encoder/ms-marco",
                "include_citations": False,
                "citation_style": "footnote",
                "max_quote_length": 500,
            }
        )
        assert cfg.vector_search_top_k == 20
        assert cfg.min_similarity == 0.7
        assert cfg.use_hybrid_search is False
        assert cfg.keyword_weight == 0.5
        assert cfg.rerank_enabled is True
        assert cfg.rerank_model == "cross-encoder/ms-marco"
        assert cfg.include_citations is False
        assert cfg.citation_style == "footnote"
        assert cfg.max_quote_length == 500

    def test_alternate_key_top_k(self):
        """'top_k' key should map to vector_search_top_k."""
        cfg = RetrievalConfig.from_dict({"top_k": 25})
        assert cfg.vector_search_top_k == 25

    def test_top_k_takes_precedence_over_vector_search_top_k(self):
        """When both are present, 'top_k' wins (first in get chain)."""
        cfg = RetrievalConfig.from_dict({"top_k": 5, "vector_search_top_k": 50})
        assert cfg.vector_search_top_k == 5


# ===========================================================================
# 12. ComprehensionConfig defaults
# ===========================================================================


class TestComprehensionConfigDefaults:

    def test_thorough_mode_default(self):
        assert ComprehensionConfig().thorough_mode is True

    def test_max_chunks_per_extraction_default(self):
        assert ComprehensionConfig().max_chunks_per_extraction == 20

    def test_process_by_chapter_default(self):
        assert ComprehensionConfig().process_by_chapter is True

    def test_max_chunks_per_chapter_default(self):
        assert ComprehensionConfig().max_chunks_per_chapter == 30

    def test_parallel_chapters_default(self):
        assert ComprehensionConfig().parallel_chapters == 1

    def test_min_concepts_per_chapter_default(self):
        assert ComprehensionConfig().min_concepts_per_chapter == 3

    def test_min_principles_per_book_default(self):
        assert ComprehensionConfig().min_principles_per_book == 10

    def test_min_techniques_per_book_default(self):
        assert ComprehensionConfig().min_techniques_per_book == 5

    def test_min_examples_per_book_default(self):
        assert ComprehensionConfig().min_examples_per_book == 5

    def test_voice_progress_enabled_default(self):
        assert ComprehensionConfig().voice_progress_enabled is True

    def test_voice_progress_interval_default(self):
        assert ComprehensionConfig().voice_progress_interval == "chapter"

    def test_voice_announce_start_default(self):
        assert ComprehensionConfig().voice_announce_start is True

    def test_voice_announce_complete_default(self):
        assert ComprehensionConfig().voice_announce_complete is True

    def test_max_llm_calls_per_book_default(self):
        assert ComprehensionConfig().max_llm_calls_per_book == 50

    def test_estimated_cost_warning_threshold_default(self):
        assert ComprehensionConfig().estimated_cost_warning_threshold == 5.0

    def test_deduplication_enabled_default(self):
        assert ComprehensionConfig().deduplication_enabled is True

    def test_concept_similarity_threshold_default(self):
        assert ComprehensionConfig().concept_similarity_threshold == 0.85


# ===========================================================================
# 13. ComprehensionConfig.from_dict
# ===========================================================================


class TestComprehensionConfigFromDict:

    def test_empty_dict_returns_defaults(self):
        cfg = ComprehensionConfig.from_dict({})
        assert cfg.thorough_mode is True
        assert cfg.max_chunks_per_extraction == 20

    def test_custom_values(self):
        cfg = ComprehensionConfig.from_dict(
            {
                "thorough_mode": False,
                "max_chunks_per_extraction": 10,
                "process_by_chapter": False,
                "max_chunks_per_chapter": 15,
                "parallel_chapters": 4,
                "min_concepts_per_chapter": 5,
                "min_principles_per_book": 20,
                "min_techniques_per_book": 10,
                "min_examples_per_book": 8,
                "voice_progress_enabled": False,
                "voice_progress_interval": "percentage",
                "voice_announce_start": False,
                "voice_announce_complete": False,
                "max_llm_calls_per_book": 100,
                "estimated_cost_warning_threshold": 10.0,
                "deduplication_enabled": False,
                "concept_similarity_threshold": 0.9,
            }
        )
        assert cfg.thorough_mode is False
        assert cfg.max_chunks_per_extraction == 10
        assert cfg.process_by_chapter is False
        assert cfg.max_chunks_per_chapter == 15
        assert cfg.parallel_chapters == 4
        assert cfg.min_concepts_per_chapter == 5
        assert cfg.min_principles_per_book == 20
        assert cfg.min_techniques_per_book == 10
        assert cfg.min_examples_per_book == 8
        assert cfg.voice_progress_enabled is False
        assert cfg.voice_progress_interval == "percentage"
        assert cfg.voice_announce_start is False
        assert cfg.voice_announce_complete is False
        assert cfg.max_llm_calls_per_book == 100
        assert cfg.estimated_cost_warning_threshold == 10.0
        assert cfg.deduplication_enabled is False
        assert cfg.concept_similarity_threshold == 0.9

    def test_partial_dict_preserves_other_defaults(self):
        cfg = ComprehensionConfig.from_dict({"thorough_mode": False})
        assert cfg.thorough_mode is False
        assert cfg.max_chunks_per_extraction == 20
        assert cfg.voice_progress_enabled is True


# ===========================================================================
# 14. IntegrationConfig defaults
# ===========================================================================


class TestIntegrationConfigDefaults:

    def test_store_chunks_in_ltm_default(self):
        assert IntegrationConfig().store_chunks_in_ltm is True

    def test_ltm_memory_type_default(self):
        assert IntegrationConfig().ltm_memory_type == "fact"

    def test_ltm_importance_default(self):
        assert IntegrationConfig().ltm_importance == 0.7

    def test_extract_triplets_default(self):
        assert IntegrationConfig().extract_triplets is True

    def test_triplet_confidence_threshold_default(self):
        assert IntegrationConfig().triplet_confidence_threshold == 0.6

    def test_add_document_node_default(self):
        assert IntegrationConfig().add_document_node is True

    def test_auto_associate_project_default(self):
        assert IntegrationConfig().auto_associate_project is True


# ===========================================================================
# 15. IntegrationConfig.from_dict
# ===========================================================================


class TestIntegrationConfigFromDict:

    def test_empty_dict_returns_defaults(self):
        cfg = IntegrationConfig.from_dict({})
        assert cfg.store_chunks_in_ltm is True
        assert cfg.triplet_confidence_threshold == 0.6

    def test_custom_values(self):
        cfg = IntegrationConfig.from_dict(
            {
                "store_chunks_in_ltm": False,
                "ltm_memory_type": "concept",
                "ltm_importance": 0.9,
                "extract_triplets": False,
                "triplet_confidence_threshold": 0.8,
                "add_document_node": False,
                "auto_associate_project": False,
            }
        )
        assert cfg.store_chunks_in_ltm is False
        assert cfg.ltm_memory_type == "concept"
        assert cfg.ltm_importance == 0.9
        assert cfg.extract_triplets is False
        assert cfg.triplet_confidence_threshold == 0.8
        assert cfg.add_document_node is False
        assert cfg.auto_associate_project is False

    def test_alternate_key_store_in_ltm(self):
        """'store_in_ltm' key should map to store_chunks_in_ltm."""
        cfg = IntegrationConfig.from_dict({"store_in_ltm": False})
        assert cfg.store_chunks_in_ltm is False

    def test_store_in_ltm_takes_precedence_over_store_chunks_in_ltm(self):
        """When both are present, 'store_in_ltm' wins (first in get chain)."""
        cfg = IntegrationConfig.from_dict(
            {
                "store_in_ltm": False,
                "store_chunks_in_ltm": True,
            }
        )
        assert cfg.store_chunks_in_ltm is False

    def test_alternate_key_triplet_confidence(self):
        """'triplet_confidence' key should map to triplet_confidence_threshold."""
        cfg = IntegrationConfig.from_dict({"triplet_confidence": 0.9})
        assert cfg.triplet_confidence_threshold == 0.9

    def test_triplet_confidence_takes_precedence_over_triplet_confidence_threshold(
        self,
    ):
        """When both are present, 'triplet_confidence' wins (first in get chain)."""
        cfg = IntegrationConfig.from_dict(
            {
                "triplet_confidence": 0.3,
                "triplet_confidence_threshold": 0.7,
            }
        )
        assert cfg.triplet_confidence_threshold == 0.3


# ===========================================================================
# 16. DocumentConfig defaults
# ===========================================================================


class TestDocumentConfigDefaults:

    def test_ocr_sub_config_default(self):
        cfg = DocumentConfig()
        assert isinstance(cfg.ocr, OCRConfig)
        assert cfg.ocr.model_path == "deepseek-ai/DeepSeek-OCR-2"

    def test_chunking_sub_config_default(self):
        cfg = DocumentConfig()
        assert isinstance(cfg.chunking, ChunkingConfig)
        assert cfg.chunking.strategy == "semantic"

    def test_embedding_sub_config_default(self):
        cfg = DocumentConfig()
        assert isinstance(cfg.embedding, EmbeddingConfig)
        assert cfg.embedding.model_name == "paraphrase-multilingual-mpnet-base-v2"

    def test_storage_sub_config_default(self):
        cfg = DocumentConfig()
        assert isinstance(cfg.storage, StorageConfig)
        assert cfg.storage.cache_enabled is True

    def test_retrieval_sub_config_default(self):
        cfg = DocumentConfig()
        assert isinstance(cfg.retrieval, RetrievalConfig)
        assert cfg.retrieval.vector_search_top_k == 10

    def test_integration_sub_config_default(self):
        cfg = DocumentConfig()
        assert isinstance(cfg.integration, IntegrationConfig)
        assert cfg.integration.store_chunks_in_ltm is True

    def test_comprehension_sub_config_default(self):
        cfg = DocumentConfig()
        assert isinstance(cfg.comprehension, ComprehensionConfig)
        assert cfg.comprehension.thorough_mode is True

    def test_max_concurrent_pages_default(self):
        assert DocumentConfig().max_concurrent_pages == 4

    def test_batch_processing_enabled_default(self):
        assert DocumentConfig().batch_processing_enabled is True

    def test_progress_callback_interval_default(self):
        assert DocumentConfig().progress_callback_interval == 10

    def test_log_level_default(self):
        assert DocumentConfig().log_level == "INFO"

    def test_log_processing_details_default(self):
        assert DocumentConfig().log_processing_details is True


# ===========================================================================
# 17. DocumentConfig.from_yaml
# ===========================================================================


class TestDocumentConfigFromYaml:

    def test_nonexistent_file_returns_defaults(self, tmp_path):
        cfg = DocumentConfig.from_yaml(tmp_path / "does_not_exist.yaml")
        assert cfg.max_concurrent_pages == 4
        assert cfg.ocr.model_path == "deepseek-ai/DeepSeek-OCR-2"

    def test_empty_yaml_returns_defaults(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.max_concurrent_pages == 4
        assert cfg.ocr.device == "cuda"

    def test_yaml_with_only_comments_returns_defaults(self, tmp_path):
        yaml_file = tmp_path / "comments.yaml"
        yaml_file.write_text("# just a comment\n# nothing else\n")
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.log_level == "INFO"

    def test_yaml_null_document_returns_defaults(self, tmp_path):
        yaml_file = tmp_path / "null.yaml"
        yaml_file.write_text("---\n~\n")
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.max_concurrent_pages == 4

    def test_valid_yaml_ocr_section(self, tmp_path, clean_env):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "ocr": {
                        "model": "custom/ocr-model",
                        "device": "mps",
                        "quantization": "none",
                    }
                }
            )
        )
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.ocr.model_path == "custom/ocr-model"
        assert cfg.ocr.device == "mps"
        assert cfg.ocr.quantization == "none"
        # Untouched defaults
        assert cfg.ocr.use_flash_attention is True

    def test_valid_yaml_chunking_section(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "chunking": {
                        "strategy": "fixed",
                        "min_chars": 100,
                        "max_chars": 1000,
                    }
                }
            )
        )
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.chunking.strategy == "fixed"
        assert cfg.chunking.min_chunk_chars == 100
        assert cfg.chunking.max_chunk_chars == 1000

    def test_valid_yaml_embedding_section(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "embedding": {
                        "model": "custom-embed",
                        "embedding_dim": 512,
                    }
                }
            )
        )
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.embedding.model_name == "custom-embed"
        assert cfg.embedding.embedding_dim == 512

    def test_valid_yaml_storage_section(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "storage": {
                        "data_dir": str(tmp_path / "storage"),
                        "cache_ttl_hours": 72,
                    }
                }
            )
        )
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.storage.db_path == str(tmp_path / "storage" / "documents.db")
        assert cfg.storage.cache_ttl_hours == 72

    def test_valid_yaml_retrieval_section(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "retrieval": {
                        "top_k": 30,
                        "citation_style": "endnote",
                    }
                }
            )
        )
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.retrieval.vector_search_top_k == 30
        assert cfg.retrieval.citation_style == "endnote"

    def test_valid_yaml_integration_section(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "integration": {
                        "store_in_ltm": False,
                        "triplet_confidence": 0.95,
                    }
                }
            )
        )
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.integration.store_chunks_in_ltm is False
        assert cfg.integration.triplet_confidence_threshold == 0.95

    def test_valid_yaml_comprehension_section(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "comprehension": {
                        "thorough_mode": False,
                        "parallel_chapters": 8,
                    }
                }
            )
        )
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.comprehension.thorough_mode is False
        assert cfg.comprehension.parallel_chapters == 8

    def test_valid_yaml_top_level_settings(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "max_concurrent_pages": 8,
                    "batch_processing_enabled": False,
                    "progress_callback_interval": 25,
                    "log_level": "DEBUG",
                    "log_processing_details": False,
                }
            )
        )
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.max_concurrent_pages == 8
        assert cfg.batch_processing_enabled is False
        assert cfg.progress_callback_interval == 25
        assert cfg.log_level == "DEBUG"
        assert cfg.log_processing_details is False

    def test_valid_yaml_full(self, tmp_path, clean_env):
        data = {
            "ocr": {"model": "custom-ocr", "device": "cpu"},
            "chunking": {"strategy": "hybrid", "min_chars": 250},
            "embedding": {"model": "custom-embed", "batch_size": 16},
            "storage": {"cache_enabled": False, "max_storage_gb": 25.0},
            "retrieval": {"top_k": 15, "use_hybrid_search": False},
            "integration": {"store_in_ltm": False, "extract_triplets": False},
            "comprehension": {"thorough_mode": False, "max_llm_calls_per_book": 25},
            "max_concurrent_pages": 16,
            "batch_processing_enabled": False,
            "progress_callback_interval": 5,
            "log_level": "WARNING",
            "log_processing_details": False,
        }
        yaml_file = tmp_path / "full.yaml"
        yaml_file.write_text(yaml.dump(data))
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.ocr.model_path == "custom-ocr"
        assert cfg.ocr.device == "cpu"
        assert cfg.chunking.strategy == "hybrid"
        assert cfg.chunking.min_chunk_chars == 250
        assert cfg.embedding.model_name == "custom-embed"
        assert cfg.embedding.batch_size == 16
        assert cfg.storage.cache_enabled is False
        assert cfg.storage.max_storage_gb == 25.0
        assert cfg.retrieval.vector_search_top_k == 15
        assert cfg.retrieval.use_hybrid_search is False
        assert cfg.integration.store_chunks_in_ltm is False
        assert cfg.integration.extract_triplets is False
        assert cfg.comprehension.thorough_mode is False
        assert cfg.comprehension.max_llm_calls_per_book == 25
        assert cfg.max_concurrent_pages == 16
        assert cfg.batch_processing_enabled is False
        assert cfg.progress_callback_interval == 5
        assert cfg.log_level == "WARNING"
        assert cfg.log_processing_details is False

    def test_default_path_used_when_none(self):
        """When path is None, DEFAULT_CONFIG_PATH is used."""
        cfg = DocumentConfig.from_yaml(None)
        assert isinstance(cfg, DocumentConfig)


# ===========================================================================
# 18. DocumentConfig._from_dict
# ===========================================================================


class TestDocumentConfigFromDict:

    def test_empty_dict_returns_defaults(self, clean_env):
        cfg = DocumentConfig._from_dict({})
        assert cfg.ocr.model_path == "deepseek-ai/DeepSeek-OCR-2"
        assert cfg.max_concurrent_pages == 4

    def test_partial_ocr_section(self, clean_env):
        cfg = DocumentConfig._from_dict({"ocr": {"device": "mps"}})
        assert cfg.ocr.device == "mps"
        assert cfg.ocr.model_path == "deepseek-ai/DeepSeek-OCR-2"

    def test_partial_chunking_section(self):
        cfg = DocumentConfig._from_dict({"chunking": {"strategy": "fixed"}})
        assert cfg.chunking.strategy == "fixed"
        assert cfg.chunking.min_chunk_chars == 500

    def test_partial_embedding_section(self):
        cfg = DocumentConfig._from_dict({"embedding": {"batch_size": 64}})
        assert cfg.embedding.batch_size == 64
        assert cfg.embedding.model_name == "paraphrase-multilingual-mpnet-base-v2"

    def test_partial_storage_section(self):
        cfg = DocumentConfig._from_dict({"storage": {"cache_enabled": False}})
        assert cfg.storage.cache_enabled is False
        assert cfg.storage.auto_cleanup_images is True

    def test_partial_retrieval_section(self):
        cfg = DocumentConfig._from_dict({"retrieval": {"top_k": 50}})
        assert cfg.retrieval.vector_search_top_k == 50
        assert cfg.retrieval.min_similarity == 0.5

    def test_partial_integration_section(self):
        cfg = DocumentConfig._from_dict({"integration": {"extract_triplets": False}})
        assert cfg.integration.extract_triplets is False
        assert cfg.integration.store_chunks_in_ltm is True

    def test_partial_comprehension_section(self):
        cfg = DocumentConfig._from_dict({"comprehension": {"parallel_chapters": 4}})
        assert cfg.comprehension.parallel_chapters == 4
        assert cfg.comprehension.thorough_mode is True

    def test_all_sections_populated(self, clean_env):
        data = {
            "ocr": {"device": "cpu"},
            "chunking": {"strategy": "fixed"},
            "embedding": {"batch_size": 16},
            "storage": {"max_storage_gb": 20.0},
            "retrieval": {"top_k": 5},
            "integration": {"store_in_ltm": False},
            "comprehension": {"thorough_mode": False},
            "max_concurrent_pages": 2,
            "batch_processing_enabled": False,
            "progress_callback_interval": 20,
            "log_level": "ERROR",
            "log_processing_details": False,
        }
        cfg = DocumentConfig._from_dict(data)
        assert cfg.ocr.device == "cpu"
        assert cfg.chunking.strategy == "fixed"
        assert cfg.embedding.batch_size == 16
        assert cfg.storage.max_storage_gb == 20.0
        assert cfg.retrieval.vector_search_top_k == 5
        assert cfg.integration.store_chunks_in_ltm is False
        assert cfg.comprehension.thorough_mode is False
        assert cfg.max_concurrent_pages == 2
        assert cfg.batch_processing_enabled is False
        assert cfg.progress_callback_interval == 20
        assert cfg.log_level == "ERROR"
        assert cfg.log_processing_details is False

    def test_unknown_keys_are_ignored(self, clean_env):
        cfg = DocumentConfig._from_dict({"unknown_key": "value", "another": 123})
        assert cfg.max_concurrent_pages == 4


# ===========================================================================
# 19. DocumentConfig.to_dict
# ===========================================================================


class TestDocumentConfigToDict:

    def test_to_dict_contains_all_top_level_keys(self):
        d = DocumentConfig().to_dict()
        expected_keys = {
            "ocr",
            "chunking",
            "embedding",
            "storage",
            "retrieval",
            "integration",
            "comprehension",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_ocr_section_keys(self):
        d = DocumentConfig().to_dict()
        expected = {"model", "quantization", "device", "image_dpi", "fallback_enabled"}
        assert set(d["ocr"].keys()) == expected

    def test_to_dict_ocr_model_uses_model_key(self):
        """to_dict exports model_path as 'model' key."""
        d = DocumentConfig().to_dict()
        assert d["ocr"]["model"] == "deepseek-ai/DeepSeek-OCR-2"

    def test_to_dict_chunking_section_keys(self):
        d = DocumentConfig().to_dict()
        expected = {
            "strategy",
            "min_chars",
            "max_chars",
            "overlap_chars",
            "respect_chapters",
        }
        assert set(d["chunking"].keys()) == expected

    def test_to_dict_chunking_uses_min_chars_key(self):
        """to_dict exports min_chunk_chars as 'min_chars' key."""
        d = DocumentConfig().to_dict()
        assert d["chunking"]["min_chars"] == 500

    def test_to_dict_chunking_uses_max_chars_key(self):
        """to_dict exports max_chunk_chars as 'max_chars' key."""
        d = DocumentConfig().to_dict()
        assert d["chunking"]["max_chars"] == 2000

    def test_to_dict_embedding_section_keys(self):
        d = DocumentConfig().to_dict()
        expected = {"model", "batch_size"}
        assert set(d["embedding"].keys()) == expected

    def test_to_dict_embedding_model_uses_model_key(self):
        """to_dict exports model_name as 'model' key."""
        d = DocumentConfig().to_dict()
        assert d["embedding"]["model"] == "paraphrase-multilingual-mpnet-base-v2"

    def test_to_dict_storage_section_keys(self):
        d = DocumentConfig().to_dict()
        expected = {"db_path", "auto_cleanup_images"}
        assert set(d["storage"].keys()) == expected

    def test_to_dict_retrieval_section_keys(self):
        d = DocumentConfig().to_dict()
        expected = {"top_k", "min_similarity", "citation_style"}
        assert set(d["retrieval"].keys()) == expected

    def test_to_dict_retrieval_uses_top_k_key(self):
        """to_dict exports vector_search_top_k as 'top_k' key."""
        d = DocumentConfig().to_dict()
        assert d["retrieval"]["top_k"] == 10

    def test_to_dict_integration_section_keys(self):
        d = DocumentConfig().to_dict()
        expected = {"store_in_ltm", "extract_triplets"}
        assert set(d["integration"].keys()) == expected

    def test_to_dict_integration_uses_store_in_ltm_key(self):
        """to_dict exports store_chunks_in_ltm as 'store_in_ltm' key."""
        d = DocumentConfig().to_dict()
        assert d["integration"]["store_in_ltm"] is True

    def test_to_dict_comprehension_section_keys(self):
        d = DocumentConfig().to_dict()
        expected = {
            "thorough_mode",
            "process_by_chapter",
            "voice_progress_enabled",
            "voice_progress_interval",
        }
        assert set(d["comprehension"].keys()) == expected

    def test_to_dict_roundtrip_values(self, clean_env):
        """Values placed into config should appear in the dict output."""
        cfg = DocumentConfig._from_dict(
            {
                "ocr": {"model": "test-ocr"},
                "chunking": {"strategy": "hybrid", "min_chars": 250, "max_chars": 3000},
                "embedding": {"model": "test-embed", "batch_size": 16},
                "retrieval": {"top_k": 20, "citation_style": "footnote"},
                "integration": {"store_in_ltm": False},
                "comprehension": {
                    "thorough_mode": False,
                    "voice_progress_interval": "step",
                },
            }
        )
        d = cfg.to_dict()
        assert d["ocr"]["model"] == "test-ocr"
        assert d["chunking"]["strategy"] == "hybrid"
        assert d["chunking"]["min_chars"] == 250
        assert d["chunking"]["max_chars"] == 3000
        assert d["embedding"]["model"] == "test-embed"
        assert d["embedding"]["batch_size"] == 16
        assert d["retrieval"]["top_k"] == 20
        assert d["retrieval"]["citation_style"] == "footnote"
        assert d["integration"]["store_in_ltm"] is False
        assert d["comprehension"]["thorough_mode"] is False
        assert d["comprehension"]["voice_progress_interval"] == "step"

    def test_to_dict_then_from_dict_preserves_key_values(self, clean_env):
        """Exporting to dict and re-importing should preserve key values."""
        original = DocumentConfig._from_dict(
            {
                "ocr": {"model": "roundtrip-ocr", "device": "mps"},
                "chunking": {"min_chars": 300},
                "retrieval": {"top_k": 7},
            }
        )
        exported = original.to_dict()
        reimported = DocumentConfig._from_dict(exported)
        assert reimported.ocr.model_path == "roundtrip-ocr"
        assert reimported.chunking.min_chunk_chars == 300
        assert reimported.retrieval.vector_search_top_k == 7


# ===========================================================================
# 20. get_document_config singleton
# ===========================================================================


class TestGetDocumentConfig:

    def test_returns_document_config_instance(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        cfg = get_document_config(yaml_file)
        assert isinstance(cfg, DocumentConfig)

    def test_singleton_returns_same_object(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("")
        cfg1 = get_document_config(yaml_file)
        cfg2 = get_document_config(yaml_file)
        assert cfg1 is cfg2

    def test_singleton_ignores_second_path(self, tmp_path):
        """Once created, the singleton ignores subsequent path arguments."""
        f1 = tmp_path / "a.yaml"
        f1.write_text(yaml.dump({"log_level": "DEBUG"}))
        f2 = tmp_path / "b.yaml"
        f2.write_text(yaml.dump({"log_level": "ERROR"}))
        cfg1 = get_document_config(f1)
        cfg2 = get_document_config(f2)
        assert cfg1 is cfg2
        assert cfg1.log_level == "DEBUG"

    def test_get_config_with_nonexistent_path(self, tmp_path):
        cfg = get_document_config(tmp_path / "missing.yaml")
        assert cfg.max_concurrent_pages == 4

    def test_get_config_default_path_used_when_none(self):
        """When path is None, DEFAULT_CONFIG_PATH is used."""
        cfg = get_document_config(None)
        assert isinstance(cfg, DocumentConfig)

    def test_get_config_no_argument(self):
        """Calling without arguments also works."""
        cfg = get_document_config()
        assert isinstance(cfg, DocumentConfig)


# ===========================================================================
# 21. reset_document_config
# ===========================================================================


class TestResetDocumentConfig:

    def test_reset_clears_singleton(self, tmp_path):
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(yaml.dump({"log_level": "DEBUG"}))
        cfg1 = get_document_config(yaml_file)
        assert cfg1.log_level == "DEBUG"

        reset_document_config()

        yaml_file.write_text(yaml.dump({"log_level": "ERROR"}))
        cfg2 = get_document_config(yaml_file)
        assert cfg2.log_level == "ERROR"
        assert cfg1 is not cfg2

    def test_reset_allows_new_config_to_be_created(self, tmp_path):
        f1 = tmp_path / "a.yaml"
        f1.write_text(yaml.dump({"max_concurrent_pages": 2}))
        cfg1 = get_document_config(f1)
        assert cfg1.max_concurrent_pages == 2

        reset_document_config()

        f2 = tmp_path / "b.yaml"
        f2.write_text(yaml.dump({"max_concurrent_pages": 16}))
        cfg2 = get_document_config(f2)
        assert cfg2.max_concurrent_pages == 16

    def test_reset_sets_internal_config_to_none(self):
        get_document_config()
        assert cfg_mod._config is not None
        reset_document_config()
        assert cfg_mod._config is None

    def test_double_reset_is_safe(self):
        """Calling reset twice should not raise an error."""
        reset_document_config()
        reset_document_config()
        assert cfg_mod._config is None


# ===========================================================================
# 22. Edge cases
# ===========================================================================


class TestEdgeCases:

    def test_from_dict_with_empty_ocr_section(self, clean_env):
        cfg = DocumentConfig._from_dict({"ocr": {}})
        assert cfg.ocr.model_path == "deepseek-ai/DeepSeek-OCR-2"
        assert cfg.ocr.fallback_api_key == ""

    def test_from_dict_with_empty_chunking_section(self):
        cfg = DocumentConfig._from_dict({"chunking": {}})
        assert cfg.chunking.strategy == "semantic"

    def test_from_dict_with_empty_embedding_section(self):
        cfg = DocumentConfig._from_dict({"embedding": {}})
        assert cfg.embedding.model_name == "paraphrase-multilingual-mpnet-base-v2"

    def test_from_dict_with_empty_storage_section(self):
        cfg = DocumentConfig._from_dict({"storage": {}})
        assert cfg.storage.cache_enabled is True

    def test_from_dict_with_empty_retrieval_section(self):
        cfg = DocumentConfig._from_dict({"retrieval": {}})
        assert cfg.retrieval.vector_search_top_k == 10

    def test_from_dict_with_empty_integration_section(self):
        cfg = DocumentConfig._from_dict({"integration": {}})
        assert cfg.integration.store_chunks_in_ltm is True

    def test_from_dict_with_empty_comprehension_section(self):
        cfg = DocumentConfig._from_dict({"comprehension": {}})
        assert cfg.comprehension.thorough_mode is True

    def test_yaml_with_extra_sections(self, tmp_path, clean_env):
        yaml_file = tmp_path / "extra.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "ocr": {"device": "cpu"},
                    "some_unknown_section": {"a": 1},
                }
            )
        )
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert cfg.ocr.device == "cpu"

    def test_multiple_configs_are_independent(self):
        """Two DocumentConfig instances should not share mutable state."""
        a = DocumentConfig()
        b = DocumentConfig()
        a.chunking.chapter_patterns.append("custom-pattern")
        assert "custom-pattern" not in b.chunking.chapter_patterns

    def test_dataclass_equality_ocr(self):
        assert OCRConfig() == OCRConfig()

    def test_dataclass_equality_chunking(self):
        assert ChunkingConfig() == ChunkingConfig()

    def test_dataclass_equality_embedding(self):
        assert EmbeddingConfig() == EmbeddingConfig()

    def test_dataclass_equality_storage(self):
        assert StorageConfig() == StorageConfig()

    def test_dataclass_equality_retrieval(self):
        assert RetrievalConfig() == RetrievalConfig()

    def test_dataclass_equality_integration(self):
        assert IntegrationConfig() == IntegrationConfig()

    def test_dataclass_equality_comprehension(self):
        assert ComprehensionConfig() == ComprehensionConfig()

    def test_dataclass_inequality_ocr(self):
        assert OCRConfig(device="cpu") != OCRConfig(device="cuda")

    def test_dataclass_inequality_chunking(self):
        assert ChunkingConfig(strategy="fixed") != ChunkingConfig(strategy="semantic")

    def test_from_yaml_reads_utf8(self, tmp_path, clean_env):
        yaml_file = tmp_path / "utf8.yaml"
        yaml_file.write_text(
            yaml.dump({"ocr": {"model": "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41-model"}}),
            encoding="utf-8",
        )
        cfg = DocumentConfig.from_yaml(yaml_file)
        assert "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41" in cfg.ocr.model_path

    def test_ocr_from_dict_unknown_keys_are_ignored(self, clean_env):
        cfg = OCRConfig.from_dict({"unknown_sub_key": True})
        assert cfg.model_path == "deepseek-ai/DeepSeek-OCR-2"

    def test_env_var_override_through_yaml_load(self, tmp_path):
        """DOCUMENT_OCR_FALLBACK_API_KEY env var should work through from_yaml."""
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(yaml.dump({"ocr": {"device": "cpu"}}))
        with patch.dict(os.environ, {"DOCUMENT_OCR_FALLBACK_API_KEY": "yaml-env-key"}):
            cfg = DocumentConfig.from_yaml(yaml_file)
            assert cfg.ocr.fallback_api_key == "yaml-env-key"

    def test_storage_from_dict_data_dir_as_path_object(self, tmp_path):
        """data_dir can be a string; Path conversion happens internally."""
        cfg = StorageConfig.from_dict({"data_dir": str(tmp_path)})
        assert str(tmp_path) in cfg.db_path

    def test_comprehension_voice_progress_interval_step(self):
        cfg = ComprehensionConfig.from_dict({"voice_progress_interval": "step"})
        assert cfg.voice_progress_interval == "step"

    def test_comprehension_voice_progress_interval_percentage(self):
        cfg = ComprehensionConfig.from_dict({"voice_progress_interval": "percentage"})
        assert cfg.voice_progress_interval == "percentage"

    def test_retrieval_citation_style_endnote(self):
        cfg = RetrievalConfig.from_dict({"citation_style": "endnote"})
        assert cfg.citation_style == "endnote"

    def test_retrieval_citation_style_footnote(self):
        cfg = RetrievalConfig.from_dict({"citation_style": "footnote"})
        assert cfg.citation_style == "footnote"

    def test_ocr_device_mps(self, clean_env):
        cfg = OCRConfig.from_dict({"device": "mps"})
        assert cfg.device == "mps"

    def test_ocr_quantization_none(self, clean_env):
        cfg = OCRConfig.from_dict({"quantization": "none"})
        assert cfg.quantization == "none"

    def test_ocr_quantization_8bit(self, clean_env):
        cfg = OCRConfig.from_dict({"quantization": "8bit"})
        assert cfg.quantization == "8bit"
