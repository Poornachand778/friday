"""
Audio Storage for Friday Voice Pipeline
========================================

Persistent storage for voice conversations.
Organizes WAV files and transcripts by date.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError as e:
    raise ImportError("soundfile not installed: pip install soundfile") from e

from voice.config import StorageConfig, get_voice_config


LOGGER = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class StoredTurn:
    """A stored voice turn"""

    turn_id: str
    session_id: str
    turn_number: int
    timestamp: str  # ISO format

    # User side
    user_audio_path: Optional[str]
    user_audio_duration: Optional[float]
    transcript: Optional[str]
    detected_language: Optional[str]

    # Assistant side
    response_text: Optional[str]
    response_audio_path: Optional[str]
    response_audio_duration: Optional[float]

    # Tool calls
    tool_calls: List[dict]

    @classmethod
    def from_dict(cls, data: dict) -> "StoredTurn":
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


class AudioStorage:
    """
    Persistent storage for voice conversations.

    Features:
    - Organizes files by date (YYYY/MM/DD)
    - Stores both user and response audio
    - Maintains transcript JSONL files
    - Supports retention policies

    Usage:
        storage = AudioStorage()

        # Store a user turn
        turn = storage.store_user_audio(
            session_id="abc123",
            turn_number=1,
            audio=audio_data,
            transcript="Hello Friday",
        )

        # Store assistant response
        storage.store_response_audio(
            turn_id=turn.turn_id,
            audio=response_audio,
            text="Hello Boss, how can I help?",
        )

        # Retrieve turns
        turns = storage.get_session_turns("abc123")
    """

    def __init__(
        self,
        config: Optional[StorageConfig] = None,
        base_path: Optional[Path] = None,
    ):
        self.config = config or get_voice_config().storage
        self.base_path = base_path or REPO_ROOT / self.config.base_path

        # Ensure directories exist
        self.base_path.mkdir(parents=True, exist_ok=True)

        self._current_date: Optional[str] = None
        self._transcript_file: Optional[Path] = None

    def _get_date_dir(self, dt: Optional[datetime] = None) -> Path:
        """Get storage directory for a date"""
        dt = dt or datetime.now()

        if self.config.organize_by_date:
            date_path = self.base_path / dt.strftime("%Y/%m/%d")
        else:
            date_path = self.base_path

        date_path.mkdir(parents=True, exist_ok=True)
        return date_path

    def _get_transcript_path(self, dt: Optional[datetime] = None) -> Path:
        """Get transcript file path"""
        date_dir = self._get_date_dir(dt)
        return date_dir / "transcripts.jsonl"

    def store_user_audio(
        self,
        session_id: str,
        turn_number: int,
        audio: np.ndarray,
        sample_rate: int = 16000,
        transcript: Optional[str] = None,
        language: Optional[str] = None,
    ) -> StoredTurn:
        """
        Store user audio for a turn.

        Args:
            session_id: Session identifier
            turn_number: Turn number within session
            audio: Audio data
            sample_rate: Audio sample rate
            transcript: Transcribed text
            language: Detected language

        Returns:
            StoredTurn with file paths
        """
        now = datetime.now()
        turn_id = str(uuid.uuid4())[:8]
        date_dir = self._get_date_dir(now)

        # Save audio file
        audio_filename = f"{session_id}_{turn_number:03d}_user_{turn_id}.wav"
        audio_path = date_dir / audio_filename

        if self.config.save_user_audio:
            self._save_audio(audio_path, audio, sample_rate)
            duration = len(audio) / sample_rate
        else:
            audio_path = None
            duration = len(audio) / sample_rate

        # Create turn record
        turn = StoredTurn(
            turn_id=turn_id,
            session_id=session_id,
            turn_number=turn_number,
            timestamp=now.isoformat(),
            user_audio_path=str(audio_path) if audio_path else None,
            user_audio_duration=duration,
            transcript=transcript,
            detected_language=language,
            response_text=None,
            response_audio_path=None,
            response_audio_duration=None,
            tool_calls=[],
        )

        # Append to transcript file
        self._append_transcript(turn, now)

        LOGGER.info(
            "Stored user audio: session=%s turn=%d duration=%.2fs",
            session_id,
            turn_number,
            duration,
        )

        return turn

    def store_response_audio(
        self,
        turn_id: str,
        session_id: str,
        turn_number: int,
        audio: np.ndarray,
        sample_rate: int = 24000,
        text: Optional[str] = None,
        tool_calls: Optional[List[dict]] = None,
    ) -> Path:
        """
        Store assistant response audio.

        Args:
            turn_id: Turn identifier
            session_id: Session identifier
            turn_number: Turn number
            audio: Audio data
            sample_rate: Audio sample rate
            text: Response text
            tool_calls: Tool calls made

        Returns:
            Path to saved audio file
        """
        now = datetime.now()
        date_dir = self._get_date_dir(now)

        # Save audio file
        audio_filename = f"{session_id}_{turn_number:03d}_response_{turn_id}.wav"
        audio_path = date_dir / audio_filename

        if self.config.save_response_audio:
            self._save_audio(audio_path, audio, sample_rate)
            duration = len(audio) / sample_rate
        else:
            duration = len(audio) / sample_rate

        # Update transcript with response info
        self._update_transcript_response(
            turn_id=turn_id,
            response_text=text,
            response_audio_path=(
                str(audio_path) if self.config.save_response_audio else None
            ),
            response_audio_duration=duration,
            tool_calls=tool_calls or [],
            dt=now,
        )

        LOGGER.info(
            "Stored response audio: turn=%s duration=%.2fs",
            turn_id,
            duration,
        )

        return audio_path

    def _save_audio(
        self,
        path: Path,
        audio: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Save audio to file"""
        # Convert to appropriate format
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            # Keep as float for quality
            pass
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        sf.write(str(path), audio, sample_rate)

    def _append_transcript(self, turn: StoredTurn, dt: datetime) -> None:
        """Append turn to transcript file"""
        transcript_path = self._get_transcript_path(dt)

        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn.to_dict(), ensure_ascii=False) + "\n")

    def _update_transcript_response(
        self,
        turn_id: str,
        response_text: Optional[str],
        response_audio_path: Optional[str],
        response_audio_duration: Optional[float],
        tool_calls: List[dict],
        dt: datetime,
    ) -> None:
        """Update transcript with response info"""
        transcript_path = self._get_transcript_path(dt)

        if not transcript_path.exists():
            return

        # Read all turns
        turns = []
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    turns.append(json.loads(line))

        # Update matching turn
        for turn in turns:
            if turn.get("turn_id") == turn_id:
                turn["response_text"] = response_text
                turn["response_audio_path"] = response_audio_path
                turn["response_audio_duration"] = response_audio_duration
                turn["tool_calls"] = tool_calls
                break

        # Rewrite file
        with open(transcript_path, "w", encoding="utf-8") as f:
            for turn in turns:
                f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    def get_session_turns(self, session_id: str) -> List[StoredTurn]:
        """Get all turns for a session"""
        turns = []

        # Search through all transcript files
        for transcript_path in self.base_path.rglob("transcripts.jsonl"):
            with open(transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get("session_id") == session_id:
                            turns.append(StoredTurn.from_dict(data))

        return sorted(turns, key=lambda t: t.turn_number)

    def get_recent_turns(self, limit: int = 100) -> List[StoredTurn]:
        """Get most recent turns across all sessions"""
        all_turns = []

        # Collect from recent transcript files
        transcript_files = sorted(
            self.base_path.rglob("transcripts.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for transcript_path in transcript_files[:10]:  # Last 10 days
            with open(transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        all_turns.append(StoredTurn.from_dict(data))

                        if len(all_turns) >= limit:
                            return all_turns

        return all_turns

    def cleanup_old_files(self, days: Optional[int] = None) -> int:
        """
        Remove files older than retention period.

        Args:
            days: Override retention days (default: config.retention_days)

        Returns:
            Number of files removed
        """
        retention_days = days or self.config.retention_days
        cutoff = datetime.now().timestamp() - (retention_days * 86400)

        removed = 0

        for audio_file in self.base_path.rglob("*.wav"):
            if audio_file.stat().st_mtime < cutoff:
                audio_file.unlink()
                removed += 1
                LOGGER.debug("Removed old audio file: %s", audio_file)

        LOGGER.info("Cleanup complete: removed %d files", removed)
        return removed

    def get_storage_stats(self) -> dict:
        """Get storage statistics"""
        total_size = 0
        audio_count = 0
        transcript_count = 0

        for audio_file in self.base_path.rglob("*.wav"):
            total_size += audio_file.stat().st_size
            audio_count += 1

        for transcript_file in self.base_path.rglob("*.jsonl"):
            total_size += transcript_file.stat().st_size
            transcript_count += sum(1 for _ in open(transcript_file))

        return {
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "audio_files": audio_count,
            "transcript_entries": transcript_count,
            "base_path": str(self.base_path),
        }
