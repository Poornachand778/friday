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
]
