"""Database package exposing configuration and schema utilities."""

from .config import DatabaseSettings, get_engine
from .schema import Base
from .screenplay_schema import (
    ScreenplayProject,
    ScreenplayScene,
    ScreenplayCharacter,
    SceneElement,
    DialogueLine,
    SceneEmbedding,
    SceneRelation,
    SceneRevision,
    ExportConfig,
)
from .vector_store import (
    VectorStore,
    VectorStoreConfig,
    VectorSearchResult,
    QdrantVectorStore,
    get_vector_store,
    reset_vector_store,
    COLLECTION_DOCUMENTS,
    COLLECTION_MEMORIES,
    COLLECTION_SCENES,
)

__all__ = [
    "DatabaseSettings",
    "get_engine",
    "Base",
    "ScreenplayProject",
    "ScreenplayScene",
    "ScreenplayCharacter",
    "SceneElement",
    "DialogueLine",
    "SceneEmbedding",
    "SceneRelation",
    "SceneRevision",
    "ExportConfig",
    # Vector store
    "VectorStore",
    "VectorStoreConfig",
    "VectorSearchResult",
    "QdrantVectorStore",
    "get_vector_store",
    "reset_vector_store",
    "COLLECTION_DOCUMENTS",
    "COLLECTION_MEMORIES",
    "COLLECTION_SCENES",
]
