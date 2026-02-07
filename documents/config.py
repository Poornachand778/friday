"""
Document Processing Configuration

Configuration dataclass with YAML loading for document processing settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "document_config.yaml"
DATA_DIR = Path(__file__).parent / "data"


@dataclass
class OCRConfig:
    """OCR engine configuration"""

    # DeepSeek-OCR 2 settings
    model_path: str = "deepseek-ai/DeepSeek-OCR-2"
    use_flash_attention: bool = True
    quantization: str = "4bit"  # "4bit", "8bit", "none"
    max_batch_size: int = 4
    device: str = "cuda"  # "cuda", "cpu", "mps"
    timeout_seconds: float = 60.0

    # Image settings
    image_dpi: int = 300
    max_image_dimension: int = 2048

    # Fallback API (when local GPU unavailable)
    fallback_enabled: bool = True
    fallback_provider: str = "google"  # "google", "azure", "aws"
    fallback_api_key: str = ""

    # Processing
    retry_on_low_confidence: bool = True
    low_confidence_threshold: float = 0.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRConfig":
        return cls(
            model_path=data.get("model", data.get("model_path", cls.model_path)),
            use_flash_attention=data.get(
                "use_flash_attention", cls.use_flash_attention
            ),
            quantization=data.get("quantization", cls.quantization),
            max_batch_size=data.get("max_batch_size", cls.max_batch_size),
            device=data.get("device", cls.device),
            timeout_seconds=data.get("timeout_seconds", cls.timeout_seconds),
            image_dpi=data.get("image_dpi", cls.image_dpi),
            max_image_dimension=data.get(
                "max_image_dimension", cls.max_image_dimension
            ),
            fallback_enabled=data.get("fallback_enabled", cls.fallback_enabled),
            fallback_provider=data.get("fallback_provider", cls.fallback_provider),
            fallback_api_key=data.get(
                "fallback_api_key",
                os.getenv("DOCUMENT_OCR_FALLBACK_API_KEY", ""),
            ),
            retry_on_low_confidence=data.get(
                "retry_on_low_confidence", cls.retry_on_low_confidence
            ),
            low_confidence_threshold=data.get(
                "low_confidence_threshold", cls.low_confidence_threshold
            ),
        )


@dataclass
class ChunkingConfig:
    """Chunking strategy configuration"""

    strategy: str = "semantic"  # "semantic", "fixed", "hybrid"

    # Size constraints
    min_chunk_chars: int = 500
    max_chunk_chars: int = 2000
    overlap_chars: int = 100

    # Section detection
    respect_chapters: bool = True
    respect_sections: bool = True
    detect_headers: bool = True

    # Special handling
    screenplay_mode: bool = False  # Scene-based chunking for screenplays
    dialogue_grouping: bool = True  # Keep dialogues together in screenplays

    # Chapter detection patterns
    chapter_patterns: List[str] = field(
        default_factory=lambda: [
            r"^Chapter\s+\d+",
            r"^CHAPTER\s+\d+",
            r"^Part\s+\d+",
            r"^\d+\.\s+[A-Z]",
        ]
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkingConfig":
        return cls(
            strategy=data.get("strategy", cls.strategy),
            min_chunk_chars=data.get(
                "min_chars", data.get("min_chunk_chars", cls.min_chunk_chars)
            ),
            max_chunk_chars=data.get(
                "max_chars", data.get("max_chunk_chars", cls.max_chunk_chars)
            ),
            overlap_chars=data.get("overlap_chars", cls.overlap_chars),
            respect_chapters=data.get("respect_chapters", cls.respect_chapters),
            respect_sections=data.get("respect_sections", cls.respect_sections),
            detect_headers=data.get("detect_headers", cls.detect_headers),
            screenplay_mode=data.get("screenplay_mode", cls.screenplay_mode),
            dialogue_grouping=data.get("dialogue_grouping", cls.dialogue_grouping),
            chapter_patterns=data.get("chapter_patterns", cls().chapter_patterns),
        )


@dataclass
class EmbeddingConfig:
    """Embedding configuration (shares model with LTM)"""

    model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    embedding_dim: int = 768
    batch_size: int = 32
    normalize: bool = True

    # GPU management
    share_gpu_with_ocr: bool = False  # If false, unload OCR before embedding
    gpu_memory_fraction: float = 0.5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingConfig":
        return cls(
            model_name=data.get("model", data.get("model_name", cls.model_name)),
            embedding_dim=data.get("embedding_dim", cls.embedding_dim),
            batch_size=data.get("batch_size", cls.batch_size),
            normalize=data.get("normalize", cls.normalize),
            share_gpu_with_ocr=data.get("share_gpu_with_ocr", cls.share_gpu_with_ocr),
            gpu_memory_fraction=data.get(
                "gpu_memory_fraction", cls.gpu_memory_fraction
            ),
        )


@dataclass
class StorageConfig:
    """Storage configuration"""

    db_path: str = str(DATA_DIR / "documents.db")
    documents_dir: str = str(DATA_DIR / "raw")  # Original PDFs
    images_dir: str = str(DATA_DIR / "images")  # Converted page images

    # Cache
    cache_enabled: bool = True
    cache_ttl_hours: int = 24

    # Cleanup
    auto_cleanup_images: bool = True  # Delete images after processing
    max_storage_gb: float = 10.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageConfig":
        data_dir = Path(data.get("data_dir", DATA_DIR))
        return cls(
            db_path=data.get("db_path", str(data_dir / "documents.db")),
            documents_dir=data.get("documents_dir", str(data_dir / "raw")),
            images_dir=data.get("images_dir", str(data_dir / "images")),
            cache_enabled=data.get("cache_enabled", cls.cache_enabled),
            cache_ttl_hours=data.get("cache_ttl_hours", cls.cache_ttl_hours),
            auto_cleanup_images=data.get(
                "auto_cleanup_images", cls.auto_cleanup_images
            ),
            max_storage_gb=data.get("max_storage_gb", cls.max_storage_gb),
        )


@dataclass
class RetrievalConfig:
    """Retrieval configuration"""

    vector_search_top_k: int = 10
    min_similarity: float = 0.5
    use_hybrid_search: bool = True  # Combine vector + keyword
    keyword_weight: float = 0.3  # Weight for keyword results in hybrid

    # Reranking
    rerank_enabled: bool = False
    rerank_model: str = ""

    # Citation
    include_citations: bool = True
    citation_style: str = "inline"  # "inline", "footnote", "endnote"
    max_quote_length: int = 200

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalConfig":
        return cls(
            vector_search_top_k=data.get(
                "top_k", data.get("vector_search_top_k", cls.vector_search_top_k)
            ),
            min_similarity=data.get("min_similarity", cls.min_similarity),
            use_hybrid_search=data.get("use_hybrid_search", cls.use_hybrid_search),
            keyword_weight=data.get("keyword_weight", cls.keyword_weight),
            rerank_enabled=data.get("rerank_enabled", cls.rerank_enabled),
            rerank_model=data.get("rerank_model", cls.rerank_model),
            include_citations=data.get("include_citations", cls.include_citations),
            citation_style=data.get("citation_style", cls.citation_style),
            max_quote_length=data.get("max_quote_length", cls.max_quote_length),
        )


@dataclass
class ComprehensionConfig:
    """Book comprehension configuration for the understanding layer"""

    # Processing mode
    thorough_mode: bool = True  # True = process all chapters, False = sample chunks
    max_chunks_per_extraction: int = 20  # For sampling mode only

    # Chapter-by-chapter settings (thorough mode)
    process_by_chapter: bool = True
    max_chunks_per_chapter: int = 30  # Chunks to include per chapter LLM call
    parallel_chapters: int = 1  # Process N chapters in parallel (1 = sequential)

    # Knowledge extraction targets
    min_concepts_per_chapter: int = 3
    min_principles_per_book: int = 10
    min_techniques_per_book: int = 5
    min_examples_per_book: int = 5

    # Voice progress settings
    voice_progress_enabled: bool = True
    voice_progress_interval: str = "chapter"  # "chapter", "percentage", "step"
    voice_announce_start: bool = True
    voice_announce_complete: bool = True

    # Cost management
    max_llm_calls_per_book: int = 50  # Safety limit
    estimated_cost_warning_threshold: float = 5.0  # Warn if > $5

    # Quality
    deduplication_enabled: bool = True
    concept_similarity_threshold: float = 0.85  # For deduplication

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComprehensionConfig":
        return cls(
            thorough_mode=data.get("thorough_mode", cls.thorough_mode),
            max_chunks_per_extraction=data.get(
                "max_chunks_per_extraction", cls.max_chunks_per_extraction
            ),
            process_by_chapter=data.get("process_by_chapter", cls.process_by_chapter),
            max_chunks_per_chapter=data.get(
                "max_chunks_per_chapter", cls.max_chunks_per_chapter
            ),
            parallel_chapters=data.get("parallel_chapters", cls.parallel_chapters),
            min_concepts_per_chapter=data.get(
                "min_concepts_per_chapter", cls.min_concepts_per_chapter
            ),
            min_principles_per_book=data.get(
                "min_principles_per_book", cls.min_principles_per_book
            ),
            min_techniques_per_book=data.get(
                "min_techniques_per_book", cls.min_techniques_per_book
            ),
            min_examples_per_book=data.get(
                "min_examples_per_book", cls.min_examples_per_book
            ),
            voice_progress_enabled=data.get(
                "voice_progress_enabled", cls.voice_progress_enabled
            ),
            voice_progress_interval=data.get(
                "voice_progress_interval", cls.voice_progress_interval
            ),
            voice_announce_start=data.get(
                "voice_announce_start", cls.voice_announce_start
            ),
            voice_announce_complete=data.get(
                "voice_announce_complete", cls.voice_announce_complete
            ),
            max_llm_calls_per_book=data.get(
                "max_llm_calls_per_book", cls.max_llm_calls_per_book
            ),
            estimated_cost_warning_threshold=data.get(
                "estimated_cost_warning_threshold", cls.estimated_cost_warning_threshold
            ),
            deduplication_enabled=data.get(
                "deduplication_enabled", cls.deduplication_enabled
            ),
            concept_similarity_threshold=data.get(
                "concept_similarity_threshold", cls.concept_similarity_threshold
            ),
        )


@dataclass
class IntegrationConfig:
    """Memory/Knowledge Graph integration configuration"""

    # LTM integration
    store_chunks_in_ltm: bool = True
    ltm_memory_type: str = "fact"  # MemoryType for document facts
    ltm_importance: float = 0.7  # Default importance for document chunks

    # Knowledge graph
    extract_triplets: bool = True
    triplet_confidence_threshold: float = 0.6
    add_document_node: bool = True  # Add document as entity

    # Project association
    auto_associate_project: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationConfig":
        return cls(
            store_chunks_in_ltm=data.get(
                "store_in_ltm", data.get("store_chunks_in_ltm", cls.store_chunks_in_ltm)
            ),
            ltm_memory_type=data.get("ltm_memory_type", cls.ltm_memory_type),
            ltm_importance=data.get("ltm_importance", cls.ltm_importance),
            extract_triplets=data.get("extract_triplets", cls.extract_triplets),
            triplet_confidence_threshold=data.get(
                "triplet_confidence",
                data.get(
                    "triplet_confidence_threshold", cls.triplet_confidence_threshold
                ),
            ),
            add_document_node=data.get("add_document_node", cls.add_document_node),
            auto_associate_project=data.get(
                "auto_associate_project", cls.auto_associate_project
            ),
        )


@dataclass
class DocumentConfig:
    """Complete document processing configuration"""

    ocr: OCRConfig = field(default_factory=OCRConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    comprehension: ComprehensionConfig = field(default_factory=ComprehensionConfig)

    # Processing
    max_concurrent_pages: int = 4
    batch_processing_enabled: bool = True
    progress_callback_interval: int = 10  # Report every N pages

    # Logging
    log_level: str = "INFO"
    log_processing_details: bool = True

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "DocumentConfig":
        """Load configuration from YAML file"""
        config_path = path or DEFAULT_CONFIG_PATH

        if not config_path.exists():
            return cls()

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "DocumentConfig":
        """Create config from dictionary"""
        return cls(
            ocr=OCRConfig.from_dict(data.get("ocr", {})),
            chunking=ChunkingConfig.from_dict(data.get("chunking", {})),
            embedding=EmbeddingConfig.from_dict(data.get("embedding", {})),
            storage=StorageConfig.from_dict(data.get("storage", {})),
            retrieval=RetrievalConfig.from_dict(data.get("retrieval", {})),
            integration=IntegrationConfig.from_dict(data.get("integration", {})),
            comprehension=ComprehensionConfig.from_dict(data.get("comprehension", {})),
            max_concurrent_pages=data.get("max_concurrent_pages", 4),
            batch_processing_enabled=data.get("batch_processing_enabled", True),
            progress_callback_interval=data.get("progress_callback_interval", 10),
            log_level=data.get("log_level", "INFO"),
            log_processing_details=data.get("log_processing_details", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Export config to dictionary"""
        return {
            "ocr": {
                "model": self.ocr.model_path,
                "quantization": self.ocr.quantization,
                "device": self.ocr.device,
                "image_dpi": self.ocr.image_dpi,
                "fallback_enabled": self.ocr.fallback_enabled,
            },
            "chunking": {
                "strategy": self.chunking.strategy,
                "min_chars": self.chunking.min_chunk_chars,
                "max_chars": self.chunking.max_chunk_chars,
                "overlap_chars": self.chunking.overlap_chars,
                "respect_chapters": self.chunking.respect_chapters,
            },
            "embedding": {
                "model": self.embedding.model_name,
                "batch_size": self.embedding.batch_size,
            },
            "storage": {
                "db_path": self.storage.db_path,
                "auto_cleanup_images": self.storage.auto_cleanup_images,
            },
            "retrieval": {
                "top_k": self.retrieval.vector_search_top_k,
                "min_similarity": self.retrieval.min_similarity,
                "citation_style": self.retrieval.citation_style,
            },
            "integration": {
                "store_in_ltm": self.integration.store_chunks_in_ltm,
                "extract_triplets": self.integration.extract_triplets,
            },
            "comprehension": {
                "thorough_mode": self.comprehension.thorough_mode,
                "process_by_chapter": self.comprehension.process_by_chapter,
                "voice_progress_enabled": self.comprehension.voice_progress_enabled,
                "voice_progress_interval": self.comprehension.voice_progress_interval,
            },
        }


# Singleton pattern
_config: Optional[DocumentConfig] = None


def get_document_config(path: Optional[Path] = None) -> DocumentConfig:
    """Get document configuration singleton"""
    global _config
    if _config is None:
        _config = DocumentConfig.from_yaml(path)
    return _config


def reset_document_config() -> None:
    """Reset configuration singleton (for testing)"""
    global _config
    _config = None
