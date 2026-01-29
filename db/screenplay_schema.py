"""
Screenplay Database Schema for Friday AI
=========================================

Proper screenplay format based on Celtx/Industry Standard:
- Scene headings with INT/EXT, location, time
- Action blocks (description paragraphs)
- Dialogue blocks (character, parenthetical, lines)
- Transitions (CUT TO, FADE IN, etc.)

Designed for:
1. Semantic search on scene content
2. Export to proper screenplay format (PDF/Fountain)
3. MCP tool integration
4. Email delivery with proper formatting
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional, List
from enum import Enum

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    Enum as SQLEnum,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ============================================================================
# ENUMS
# ============================================================================


class IntExtType(str, Enum):
    INT = "INT"
    EXT = "EXT"
    INT_EXT = "INT/EXT"


class ElementType(str, Enum):
    ACTION = "action"
    DIALOGUE = "dialogue"
    TRANSITION = "transition"
    SHOT = "shot"


class ScriptStatus(str, Enum):
    DRAFT = "draft"
    REVISION = "revision"
    LOCKED = "locked"
    PRODUCTION = "production"


class SceneStatus(str, Enum):
    ACTIVE = "active"
    BACKLOG = "backlog"
    CUT = "cut"
    ARCHIVED = "archived"


# ============================================================================
# SCRIPT PROJECT
# ============================================================================


class ScreenplayProject(Base):
    """A screenplay/script project"""

    __tablename__ = "screenplay_projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Basic Info
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    slug: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    logline: Mapped[Optional[str]] = mapped_column(Text)

    # Metadata
    author: Mapped[Optional[str]] = mapped_column(String(256))
    contact: Mapped[Optional[str]] = mapped_column(String(512))  # Email/phone
    draft_date: Mapped[Optional[str]] = mapped_column(String(64))  # "January 2026"
    copyright_notice: Mapped[Optional[str]] = mapped_column(String(512))

    # Status
    status: Mapped[str] = mapped_column(String(32), default=ScriptStatus.DRAFT.value)
    version: Mapped[int] = mapped_column(Integer, default=1)

    # Language settings
    primary_language: Mapped[str] = mapped_column(String(8), default="te")  # Telugu
    secondary_language: Mapped[Optional[str]] = mapped_column(String(8), default="en")

    # Notes
    notes: Mapped[Optional[str]] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    scenes: Mapped[List["ScreenplayScene"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="ScreenplayScene.scene_number",
    )
    characters: Mapped[List["ScreenplayCharacter"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


# ============================================================================
# CHARACTERS
# ============================================================================


class ScreenplayCharacter(Base):
    """Character registry for a screenplay"""

    __tablename__ = "screenplay_characters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("screenplay_projects.id"), nullable=False
    )

    # Character Info
    name: Mapped[str] = mapped_column(
        String(128), nullable=False
    )  # Display name (NEELIMA)
    full_name: Mapped[Optional[str]] = mapped_column(String(256))  # Neelima Sharma
    description: Mapped[Optional[str]] = mapped_column(Text)  # "Mid 20s, curly hair..."

    # Character traits for AI understanding
    age_range: Mapped[Optional[str]] = mapped_column(String(32))  # "mid 20s"
    role_type: Mapped[Optional[str]] = mapped_column(
        String(32)
    )  # protagonist, antagonist, supporting

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    project: Mapped[ScreenplayProject] = relationship(back_populates="characters")

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_character_project_name"),
    )


# ============================================================================
# SCENES
# ============================================================================


class ScreenplayScene(Base):
    """A single scene in a screenplay"""

    __tablename__ = "screenplay_scenes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("screenplay_projects.id"), nullable=False
    )

    # Scene Heading Components (for the gray box)
    scene_number: Mapped[int] = mapped_column(Integer, nullable=False)  # 1, 2, 3...
    int_ext: Mapped[str] = mapped_column(
        String(16), default=IntExtType.INT.value
    )  # INT, EXT, INT/EXT
    location: Mapped[str] = mapped_column(
        String(256), nullable=False
    )  # "HOUSE- TWO STORIED BUILDING"
    sub_location: Mapped[Optional[str]] = mapped_column(
        String(256)
    )  # "THIRD FLOOR - FRONT BALCONY"
    time_of_day: Mapped[Optional[str]] = mapped_column(
        String(64)
    )  # "MORNING - 8 A.M", "NIGHT - 7:00 P.M."

    # Computed full heading (for display)
    # Format: "INT. HOUSE - THIRD FLOOR - MORNING 8 A.M"

    # Scene metadata
    title: Mapped[Optional[str]] = mapped_column(
        String(256)
    )  # Internal title for reference
    summary: Mapped[Optional[str]] = mapped_column(
        Text
    )  # AI-generated or manual summary

    # For ordering (supports fractional for insertions)
    narrative_order: Mapped[float] = mapped_column(Float, default=0.0)

    # Status
    status: Mapped[str] = mapped_column(String(32), default=SceneStatus.ACTIVE.value)

    # Tags for search
    tags: Mapped[list] = mapped_column(
        JSON, default=list
    )  # ["emotional", "conflict", "romance"]

    # Page estimate (for production)
    estimated_pages: Mapped[Optional[float]] = mapped_column(Float)  # 1.5 pages

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    project: Mapped[ScreenplayProject] = relationship(back_populates="scenes")
    elements: Mapped[List["SceneElement"]] = relationship(
        back_populates="scene",
        cascade="all, delete-orphan",
        order_by="SceneElement.order_index",
    )
    embeddings: Mapped[List["SceneEmbedding"]] = relationship(
        back_populates="scene", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_scene_project_number", "project_id", "scene_number"),
        UniqueConstraint("project_id", "scene_number", name="uq_scene_project_number"),
    )

    @property
    def scene_heading(self) -> str:
        """Generate the full scene heading string"""
        parts = [f"{self.int_ext}."]

        if self.location:
            parts.append(self.location.upper())
        if self.sub_location:
            parts.append(f"- {self.sub_location.upper()}")
        if self.time_of_day:
            parts.append(f"- {self.time_of_day.upper()}")

        return " ".join(parts)


# ============================================================================
# SCENE ELEMENTS (Action, Dialogue, Transitions)
# ============================================================================


class SceneElement(Base):
    """
    Individual elements within a scene.
    Each element is either: action, dialogue, transition, or shot.
    """

    __tablename__ = "scene_elements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scene_id: Mapped[int] = mapped_column(
        ForeignKey("screenplay_scenes.id"), nullable=False
    )

    # Element type
    element_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # action, dialogue, transition

    # Order within scene
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    scene: Mapped[ScreenplayScene] = relationship(back_populates="elements")

    # Polymorphic content - use JSON for flexibility
    # For ACTION: {"text": "A quiet morning. Two pairs of sparrows..."}
    # For DIALOGUE: {"character": "NEELIMA", "parenthetical": "PHONE CALL V.O.",
    #                "lines": [{"text": "nanna tho godava pettukunna", "translation": "i fought with daddy."}]}
    # For TRANSITION: {"text": "CUT TO"}
    content: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    __table_args__ = (Index("ix_element_scene_order", "scene_id", "order_index"),)


# ============================================================================
# DIALOGUE DETAILS (Separate table for better querying)
# ============================================================================


class DialogueLine(Base):
    """
    Individual dialogue lines within a dialogue element.
    Allows for bilingual dialogue with translations.
    """

    __tablename__ = "dialogue_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    element_id: Mapped[int] = mapped_column(
        ForeignKey("scene_elements.id"), nullable=False
    )

    # Character speaking
    character_name: Mapped[str] = mapped_column(String(128), nullable=False)

    # Parenthetical direction (V.O., O.S., CONT'D, etc.)
    parenthetical: Mapped[Optional[str]] = mapped_column(String(128))

    # The dialogue text
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Translation (if dialogue is in Telugu, this is English translation)
    translation: Mapped[Optional[str]] = mapped_column(Text)

    # Language of the dialogue text
    language: Mapped[str] = mapped_column(String(8), default="te")  # te, en, mixed

    # Order within the element (for back-and-forth dialogue)
    line_order: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ============================================================================
# SCENE EMBEDDINGS (For Semantic Search)
# ============================================================================


class SceneEmbedding(Base):
    """Vector embeddings for semantic search on scenes"""

    __tablename__ = "scene_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scene_id: Mapped[int] = mapped_column(
        ForeignKey("screenplay_scenes.id"), nullable=False
    )

    # What was embedded
    content_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # "full", "summary", "dialogue"
    content_hash: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # SHA256 to detect changes

    # Embedding model info
    model_name: Mapped[str] = mapped_column(String(160), nullable=False)

    # The embedding vector (stored as JSON array of floats)
    # For production, consider using pgvector extension
    vector: Mapped[list] = mapped_column(JSON, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    scene: Mapped[ScreenplayScene] = relationship(back_populates="embeddings")


# ============================================================================
# SCENE LINKS (For story structure)
# ============================================================================


class SceneRelation(Base):
    """Links between scenes for story structure"""

    __tablename__ = "scene_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(
        ForeignKey("screenplay_projects.id"), nullable=False
    )

    from_scene_id: Mapped[int] = mapped_column(
        ForeignKey("screenplay_scenes.id"), nullable=False
    )
    to_scene_id: Mapped[int] = mapped_column(
        ForeignKey("screenplay_scenes.id"), nullable=False
    )

    # Relation type
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # Types: "sequence", "flashback", "parallel", "callback", "setup_payoff"

    notes: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ============================================================================
# REVISION HISTORY
# ============================================================================


class SceneRevision(Base):
    """Revision history for scenes"""

    __tablename__ = "scene_revisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scene_id: Mapped[int] = mapped_column(
        ForeignKey("screenplay_scenes.id"), nullable=False
    )

    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # What changed
    change_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # "created", "edited", "restructured"
    change_summary: Mapped[Optional[str]] = mapped_column(Text)

    # Snapshot of the scene at this revision (JSON blob)
    snapshot: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Who made the change
    author: Mapped[Optional[str]] = mapped_column(String(128))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================


class ExportConfig(Base):
    """Configuration for screenplay export formatting"""

    __tablename__ = "export_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # Font settings
    font_family: Mapped[str] = mapped_column(String(64), default="Courier Prime")
    font_size: Mapped[int] = mapped_column(Integer, default=12)

    # Page settings
    page_width: Mapped[float] = mapped_column(Float, default=8.5)  # inches
    page_height: Mapped[float] = mapped_column(Float, default=11.0)
    margin_top: Mapped[float] = mapped_column(Float, default=1.0)
    margin_bottom: Mapped[float] = mapped_column(Float, default=1.0)
    margin_left: Mapped[float] = mapped_column(Float, default=1.5)
    margin_right: Mapped[float] = mapped_column(Float, default=1.0)

    # Scene heading style
    scene_heading_bg_color: Mapped[str] = mapped_column(
        String(16), default="#CCCCCC"
    )  # Gray box
    scene_heading_bold: Mapped[bool] = mapped_column(Boolean, default=True)

    # Dialogue formatting
    character_name_caps: Mapped[bool] = mapped_column(Boolean, default=True)
    parenthetical_italics: Mapped[bool] = mapped_column(Boolean, default=False)

    # Translation display
    show_translations: Mapped[bool] = mapped_column(Boolean, default=True)
    translation_in_parentheses: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
