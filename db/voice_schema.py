"""
Voice Database Schema for Friday AI
====================================

Tables for voice conversation pipeline:
- VoiceSession: A conversation session triggered by wake word
- VoiceTurn: Individual turns (user speech + system response)
- VoiceProfile: TTS voice cloning profiles
- VoiceTrainingExample: Approved examples for training export
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
    Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.screenplay_schema import Base


# ============================================================================
# ENUMS
# ============================================================================


class SessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class TurnRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TrainingStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPORTED = "exported"


# ============================================================================
# VOICE SESSIONS
# ============================================================================


class VoiceSession(Base):
    """A voice conversation session triggered by wake word"""

    __tablename__ = "voice_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Wake word that triggered the session
    wake_word_detected: Mapped[str] = mapped_column(String(64), nullable=False)
    wake_word_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Session info
    session_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False
    )  # UUID
    status: Mapped[str] = mapped_column(String(32), default=SessionStatus.ACTIVE.value)

    # Device/location info
    device_id: Mapped[Optional[str]] = mapped_column(String(128))
    location: Mapped[Optional[str]] = mapped_column(
        String(128)
    )  # "writers_room", "kitchen"

    # Duration tracking
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)

    # Aggregate stats
    total_turns: Mapped[int] = mapped_column(Integer, default=0)
    total_audio_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    turns: Mapped[List["VoiceTurn"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="VoiceTurn.turn_number",
    )

    __table_args__ = (
        Index("ix_voice_session_started", "started_at"),
        Index("ix_voice_session_status", "status"),
    )


# ============================================================================
# VOICE TURNS
# ============================================================================


class VoiceTurn(Base):
    """Individual turn in a voice conversation"""

    __tablename__ = "voice_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("voice_sessions.id"), nullable=False
    )

    # Turn ordering
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user, assistant

    # User audio (STT input)
    user_audio_path: Mapped[Optional[str]] = mapped_column(
        String(512)
    )  # Path to WAV file
    user_audio_duration: Mapped[Optional[float]] = mapped_column(Float)  # seconds
    user_audio_sample_rate: Mapped[int] = mapped_column(Integer, default=16000)

    # STT result
    transcript: Mapped[Optional[str]] = mapped_column(Text)
    transcript_confidence: Mapped[Optional[float]] = mapped_column(Float)
    detected_language: Mapped[Optional[str]] = mapped_column(String(8))  # te, en, mixed

    # STT timing
    stt_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    stt_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    stt_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # LLM response
    response_text: Mapped[Optional[str]] = mapped_column(Text)
    llm_model: Mapped[Optional[str]] = mapped_column(String(128))
    llm_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # Tool calls made during this turn
    tool_calls: Mapped[list] = mapped_column(JSON, default=list)
    # Format: [{"tool": "scene_search", "args": {...}, "result": {...}}]

    # TTS output
    response_audio_path: Mapped[Optional[str]] = mapped_column(String(512))
    response_audio_duration: Mapped[Optional[float]] = mapped_column(Float)
    tts_voice_profile: Mapped[Optional[str]] = mapped_column(String(64))
    tts_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # Total turn timing
    turn_started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    turn_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    total_latency_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # Training eligibility
    training_status: Mapped[str] = mapped_column(
        String(32), default=TrainingStatus.PENDING.value
    )

    # Relationships
    session: Mapped[VoiceSession] = relationship(back_populates="turns")

    __table_args__ = (
        Index("ix_voice_turn_session_number", "session_id", "turn_number"),
        Index("ix_voice_turn_training", "training_status"),
    )


# ============================================================================
# VOICE PROFILES (for TTS cloning)
# ============================================================================


class VoiceProfile(Base):
    """TTS voice cloning profiles"""

    __tablename__ = "voice_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Profile info
    name: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False
    )  # "friday_telugu"
    description: Mapped[Optional[str]] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(8), default="te")  # Primary language

    # Reference audio for cloning
    reference_audio_paths: Mapped[list] = mapped_column(JSON, default=list)
    # Format: ["voice/data/voice_samples/friday_te_01.wav", ...]

    # TTS settings
    tts_engine: Mapped[str] = mapped_column(String(32), default="xtts_v2")
    speaker_embedding: Mapped[Optional[list]] = mapped_column(JSON)  # Cached embedding
    gpt_cond_latent: Mapped[Optional[list]] = mapped_column(JSON)  # XTTS conditioning

    # Voice characteristics
    speaking_rate: Mapped[float] = mapped_column(Float, default=1.0)
    pitch_adjustment: Mapped[float] = mapped_column(Float, default=0.0)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


# ============================================================================
# VOICE TRAINING EXAMPLES
# ============================================================================


class VoiceTrainingExample(Base):
    """Approved examples for training data generation"""

    __tablename__ = "voice_training_examples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    turn_id: Mapped[int] = mapped_column(ForeignKey("voice_turns.id"), nullable=False)

    # Export info
    export_batch: Mapped[Optional[str]] = mapped_column(
        String(64)
    )  # Batch ID when exported
    exported_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Quality metadata
    quality_score: Mapped[Optional[float]] = mapped_column(Float)  # 0.0-1.0
    review_notes: Mapped[Optional[str]] = mapped_column(Text)
    reviewed_by: Mapped[Optional[str]] = mapped_column(String(64))

    # Training format
    # Stored in ChatML format for SFT
    formatted_example: Mapped[Optional[dict]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_voice_training_batch", "export_batch"),)
