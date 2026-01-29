"""
Training Data Generator for Friday AI
=====================================

Exports voice conversation data in SFT training format (ChatML).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

from voice.config import get_voice_config
from voice.storage.audio_storage import AudioStorage, StoredTurn


LOGGER = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "sft" / "voice"


@dataclass
class TrainingExample:
    """A training example in ChatML format"""

    messages: List[
        dict
    ]  # [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]
    metadata: dict = field(default_factory=dict)

    def to_jsonl_line(self) -> str:
        """Convert to JSONL format"""
        return json.dumps(
            {
                "messages": self.messages,
                **self.metadata,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_turn(
        cls,
        turn: StoredTurn,
        system_prompt: str,
        include_tool_calls: bool = True,
    ) -> Optional["TrainingExample"]:
        """Create training example from a stored turn"""
        if not turn.transcript or not turn.response_text:
            return None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": turn.transcript},
        ]

        # Add tool calls if present
        if include_tool_calls and turn.tool_calls:
            # Format tool calls for training
            assistant_content = turn.response_text
            tool_call_text = (
                "\n\n[Tool calls: "
                + ", ".join(tc.get("tool", "unknown") for tc in turn.tool_calls)
                + "]"
            )
            # Don't append tool call text to response for cleaner training
        else:
            assistant_content = turn.response_text

        messages.append({"role": "assistant", "content": assistant_content})

        metadata = {
            "source": "voice",
            "turn_id": turn.turn_id,
            "session_id": turn.session_id,
            "language": turn.detected_language or "unknown",
            "timestamp": turn.timestamp,
        }

        if turn.tool_calls:
            metadata["tool_calls"] = turn.tool_calls

        return cls(messages=messages, metadata=metadata)


class TrainingDataGenerator:
    """
    Generates training data from voice conversations.

    Features:
    - Exports to ChatML JSONL format
    - Quality filtering (length, completeness)
    - Language filtering
    - Batch export with metadata

    Usage:
        generator = TrainingDataGenerator()

        # Export approved turns
        generator.export_approved_turns(
            output_path="voice_training.jsonl",
            min_quality_score=0.8,
        )

        # Or iterate through examples
        for example in generator.generate_examples():
            print(example.to_jsonl_line())
    """

    # Default Friday system prompt
    DEFAULT_SYSTEM_PROMPT = (
        "You are Friday, Poorna's AI assistant. "
        "You blend Telugu and English naturally, addressing him as 'Boss'. "
        "Be concise, helpful, and direct. No flattery or excessive formality."
    )

    def __init__(
        self,
        storage: Optional[AudioStorage] = None,
        system_prompt: Optional[str] = None,
    ):
        self.storage = storage or AudioStorage()
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def generate_examples(
        self,
        min_transcript_length: int = 5,
        min_response_length: int = 10,
        languages: Optional[List[str]] = None,
        include_tool_calls: bool = True,
    ) -> Iterator[TrainingExample]:
        """
        Generate training examples from stored turns.

        Args:
            min_transcript_length: Minimum user transcript length
            min_response_length: Minimum response length
            languages: Filter by languages (None = all)
            include_tool_calls: Include tool call metadata

        Yields:
            TrainingExample objects
        """
        turns = self.storage.get_recent_turns(limit=10000)

        for turn in turns:
            # Skip incomplete turns
            if not turn.transcript or not turn.response_text:
                continue

            # Length filters
            if len(turn.transcript) < min_transcript_length:
                continue
            if len(turn.response_text) < min_response_length:
                continue

            # Language filter
            if languages and turn.detected_language not in languages:
                continue

            example = TrainingExample.from_turn(
                turn=turn,
                system_prompt=self.system_prompt,
                include_tool_calls=include_tool_calls,
            )

            if example:
                yield example

    def export_to_jsonl(
        self,
        output_path: Optional[Path] = None,
        min_transcript_length: int = 5,
        min_response_length: int = 10,
        languages: Optional[List[str]] = None,
        include_tool_calls: bool = True,
    ) -> Tuple[Path, int]:
        """
        Export training data to JSONL file.

        Args:
            output_path: Output file path
            min_transcript_length: Minimum user transcript length
            min_response_length: Minimum response length
            languages: Filter by languages
            include_tool_calls: Include tool call metadata

        Returns:
            Tuple of (output_path, example_count)
        """
        from typing import Tuple

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"voice_train_{timestamp}.jsonl"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for example in self.generate_examples(
                min_transcript_length=min_transcript_length,
                min_response_length=min_response_length,
                languages=languages,
                include_tool_calls=include_tool_calls,
            ):
                f.write(example.to_jsonl_line() + "\n")
                count += 1

        LOGGER.info("Exported %d training examples to %s", count, output_path)
        return output_path, count

    def export_approved_turns(
        self,
        output_path: Optional[Path] = None,
    ) -> Tuple[Path, int]:
        """
        Export only approved training examples from database.

        Returns:
            Tuple of (output_path, example_count)
        """
        from typing import Tuple

        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import Session
            from db.voice_schema import VoiceTurn, VoiceTrainingExample, TrainingStatus

            db_url = os.environ.get(
                "DATABASE_URL", "postgresql://friday:friday@localhost:5432/friday"
            )
            engine = create_engine(db_url)

            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = OUTPUT_DIR / f"voice_approved_{timestamp}.jsonl"
            else:
                output_path = Path(output_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            count = 0
            with Session(engine) as session:
                # Get approved examples
                approved = (
                    session.query(VoiceTrainingExample)
                    .filter_by(training_status=TrainingStatus.APPROVED.value)
                    .all()
                )

                with open(output_path, "w", encoding="utf-8") as f:
                    for example in approved:
                        if example.formatted_example:
                            f.write(
                                json.dumps(
                                    example.formatted_example, ensure_ascii=False
                                )
                                + "\n"
                            )
                            count += 1

            LOGGER.info("Exported %d approved examples to %s", count, output_path)
            return output_path, count

        except Exception as e:
            LOGGER.error("Failed to export approved turns: %s", e)
            # Fall back to file-based export
            return self.export_to_jsonl(output_path)

    def get_statistics(self) -> dict:
        """Get training data statistics"""
        turns = self.storage.get_recent_turns(limit=10000)

        total = len(turns)
        complete = sum(1 for t in turns if t.transcript and t.response_text)
        with_tools = sum(1 for t in turns if t.tool_calls)

        languages = {}
        for turn in turns:
            lang = turn.detected_language or "unknown"
            languages[lang] = languages.get(lang, 0) + 1

        return {
            "total_turns": total,
            "complete_turns": complete,
            "with_tool_calls": with_tools,
            "languages": languages,
            "completion_rate": round(complete / total * 100, 1) if total > 0 else 0,
        }

    def approve_turn(self, turn_id: str, quality_score: float = 1.0) -> bool:
        """
        Mark a turn as approved for training.

        Args:
            turn_id: Turn identifier
            quality_score: Quality score (0-1)

        Returns:
            True if successful
        """
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import Session
            from db.voice_schema import VoiceTurn, VoiceTrainingExample, TrainingStatus

            db_url = os.environ.get(
                "DATABASE_URL", "postgresql://friday:friday@localhost:5432/friday"
            )
            engine = create_engine(db_url)

            with Session(engine) as session:
                # Find the turn
                turn = session.query(VoiceTurn).filter_by(turn_id=turn_id).first()
                if not turn:
                    LOGGER.error("Turn not found: %s", turn_id)
                    return False

                # Update training status
                turn.training_status = TrainingStatus.APPROVED.value

                # Create or update training example
                example = (
                    session.query(VoiceTrainingExample)
                    .filter_by(turn_id=turn.id)
                    .first()
                )
                if not example:
                    # Create formatted example
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": turn.transcript},
                        {"role": "assistant", "content": turn.response_text},
                    ]

                    example = VoiceTrainingExample(
                        turn_id=turn.id,
                        quality_score=quality_score,
                        formatted_example={"messages": messages},
                    )
                    session.add(example)
                else:
                    example.quality_score = quality_score

                session.commit()

            LOGGER.info("Approved turn %s with quality %.2f", turn_id, quality_score)
            return True

        except Exception as e:
            LOGGER.error("Failed to approve turn: %s", e)
            return False

    def reject_turn(self, turn_id: str, reason: str = "") -> bool:
        """Mark a turn as rejected for training"""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import Session
            from db.voice_schema import VoiceTurn, TrainingStatus

            db_url = os.environ.get(
                "DATABASE_URL", "postgresql://friday:friday@localhost:5432/friday"
            )
            engine = create_engine(db_url)

            with Session(engine) as session:
                turn = session.query(VoiceTurn).filter_by(turn_id=turn_id).first()
                if turn:
                    turn.training_status = TrainingStatus.REJECTED.value
                    session.commit()

            LOGGER.info("Rejected turn %s: %s", turn_id, reason)
            return True

        except Exception as e:
            LOGGER.error("Failed to reject turn: %s", e)
            return False


# Add missing import
from typing import Tuple
