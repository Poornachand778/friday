"""
Custom Wake Word Training for Friday AI
========================================

Tools for training custom wake words like "Hey Friday" or "Wake up Daddy's home".
Uses OpenWakeWord's training utilities.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

from voice.config import get_voice_config


LOGGER = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_DIR = REPO_ROOT / "voice" / "data" / "wakeword_training"
MODELS_DIR = REPO_ROOT / "voice" / "models"


@dataclass
class TrainingConfig:
    """Configuration for wake word training"""

    wake_word: str  # The wake phrase to train
    output_name: str  # Output model filename
    positive_samples_dir: Path
    negative_samples_dir: Optional[Path] = None

    # Training parameters
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.001

    # Audio preprocessing
    sample_rate: int = 16000
    target_length_ms: int = 1500  # Typical wake word duration

    # Augmentation
    augment_noise: bool = True
    augment_speed: bool = True
    augment_pitch: bool = False

    # Output
    output_dir: Path = field(default_factory=lambda: MODELS_DIR)


class WakeWordTrainer:
    """
    Custom wake word model trainer.

    Uses OpenWakeWord's training pipeline to create custom wake word models.

    Workflow:
    1. Collect positive samples (recordings of the wake phrase)
    2. Collect negative samples (other speech/noise)
    3. Run training
    4. Export ONNX model

    Usage:
        trainer = WakeWordTrainer()

        # Prepare data
        trainer.record_positive_samples("hey_friday", num_samples=50)
        trainer.add_negative_samples(["general_speech.wav", "noise.wav"])

        # Train
        config = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=trainer.positive_dir,
        )
        trainer.train(config)

        # Use the model
        model_path = trainer.output_dir / "hey_friday.onnx"
    """

    def __init__(self):
        self.training_data_dir = TRAINING_DATA_DIR
        self.models_dir = MODELS_DIR

        # Create directories
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def prepare_training_dir(self, wake_word_name: str) -> Path:
        """
        Prepare a training directory for a wake word.

        Args:
            wake_word_name: Identifier for the wake word (e.g., "hey_friday")

        Returns:
            Path to training directory
        """
        training_dir = self.training_data_dir / wake_word_name
        positive_dir = training_dir / "positive"
        negative_dir = training_dir / "negative"

        positive_dir.mkdir(parents=True, exist_ok=True)
        negative_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file
        metadata = {
            "wake_word_name": wake_word_name,
            "positive_samples": 0,
            "negative_samples": 0,
            "status": "collecting",
        }
        metadata_path = training_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        LOGGER.info("Prepared training directory: %s", training_dir)
        return training_dir

    def add_positive_sample(
        self,
        wake_word_name: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
        sample_id: Optional[str] = None,
    ) -> Path:
        """
        Add a positive sample (recording of the wake phrase).

        Args:
            wake_word_name: Wake word identifier
            audio: Audio data
            sample_rate: Sample rate
            sample_id: Optional sample identifier

        Returns:
            Path to saved sample
        """
        if sf is None:
            raise ImportError("soundfile required: pip install soundfile")

        training_dir = self.training_data_dir / wake_word_name
        positive_dir = training_dir / "positive"
        positive_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        existing = list(positive_dir.glob("*.wav"))
        if sample_id:
            filename = f"{sample_id}.wav"
        else:
            filename = f"sample_{len(existing):04d}.wav"

        output_path = positive_dir / filename

        # Resample if needed (should be 16kHz)
        if sample_rate != 16000:
            LOGGER.warning("Resampling from %d to 16000 Hz", sample_rate)
            # Simple resampling (for production, use librosa or scipy)
            ratio = 16000 / sample_rate
            audio = np.interp(
                np.linspace(0, len(audio), int(len(audio) * ratio)),
                np.arange(len(audio)),
                audio,
            )

        # Convert to int16 if needed
        if audio.dtype != np.int16:
            if audio.max() <= 1.0:
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)

        sf.write(str(output_path), audio, 16000)

        # Update metadata
        self._update_metadata(wake_word_name)

        LOGGER.info("Added positive sample: %s", output_path)
        return output_path

    def add_negative_sample(
        self,
        wake_word_name: str,
        audio_path: str,
    ) -> bool:
        """
        Add a negative sample (non-wake-word audio).

        Args:
            wake_word_name: Wake word identifier
            audio_path: Path to negative sample audio

        Returns:
            True if successful
        """
        training_dir = self.training_data_dir / wake_word_name
        negative_dir = training_dir / "negative"
        negative_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(audio_path)
        if not source_path.exists():
            LOGGER.error("Negative sample not found: %s", audio_path)
            return False

        # Copy to negative samples directory
        dest_path = negative_dir / source_path.name
        shutil.copy2(source_path, dest_path)

        self._update_metadata(wake_word_name)

        LOGGER.info("Added negative sample: %s", dest_path)
        return True

    def _update_metadata(self, wake_word_name: str) -> None:
        """Update training metadata"""
        training_dir = self.training_data_dir / wake_word_name
        positive_dir = training_dir / "positive"
        negative_dir = training_dir / "negative"
        metadata_path = training_dir / "metadata.json"

        positive_count = (
            len(list(positive_dir.glob("*.wav"))) if positive_dir.exists() else 0
        )
        negative_count = (
            len(list(negative_dir.glob("*.wav"))) if negative_dir.exists() else 0
        )

        metadata = {
            "wake_word_name": wake_word_name,
            "positive_samples": positive_count,
            "negative_samples": negative_count,
            "status": "collecting",
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def get_training_status(self, wake_word_name: str) -> dict:
        """Get training status for a wake word"""
        training_dir = self.training_data_dir / wake_word_name
        metadata_path = training_dir / "metadata.json"

        if not metadata_path.exists():
            return {"status": "not_found"}

        with open(metadata_path) as f:
            return json.load(f)

    def train(self, config: TrainingConfig) -> Optional[Path]:
        """
        Train a custom wake word model.

        Note: Full training requires OpenWakeWord's training dependencies
        (tensorflow, etc.) which are not included in the base installation.

        Args:
            config: Training configuration

        Returns:
            Path to trained model if successful, None otherwise
        """
        # Check for training dependencies
        try:
            from openwakeword.train import train_model
        except ImportError:
            LOGGER.error(
                "OpenWakeWord training dependencies not installed. "
                "See: https://github.com/dscripka/openWakeWord/tree/main/openwakeword/train"
            )
            return None

        # Validate samples
        positive_dir = config.positive_samples_dir
        positive_samples = list(positive_dir.glob("*.wav"))

        if len(positive_samples) < 10:
            LOGGER.error(
                "Insufficient positive samples (%d). Need at least 10.",
                len(positive_samples),
            )
            return None

        LOGGER.info(
            "Starting training: %s with %d positive samples",
            config.wake_word,
            len(positive_samples),
        )

        # Run training
        try:
            output_path = config.output_dir / f"{config.output_name}.onnx"

            # OpenWakeWord training (simplified - actual implementation varies)
            train_model(
                positive_dir=str(positive_dir),
                negative_dir=(
                    str(config.negative_samples_dir)
                    if config.negative_samples_dir
                    else None
                ),
                output_path=str(output_path),
                epochs=config.epochs,
                batch_size=config.batch_size,
            )

            if output_path.exists():
                LOGGER.info("Training complete: %s", output_path)

                # Update metadata
                training_dir = self.training_data_dir / config.output_name
                metadata_path = training_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    metadata["status"] = "trained"
                    metadata["model_path"] = str(output_path)
                    with open(metadata_path, "w") as f:
                        json.dump(metadata, f, indent=2)

                return output_path
            else:
                LOGGER.error("Training failed - no output model")
                return None

        except Exception as e:
            LOGGER.error("Training failed: %s", e)
            return None

    def list_available_models(self) -> List[dict]:
        """List available wake word models"""
        models = []

        # Check for trained models
        for model_path in self.models_dir.glob("*.onnx"):
            models.append(
                {
                    "name": model_path.stem,
                    "path": str(model_path),
                    "type": "custom",
                }
            )

        # Add built-in models
        builtin = ["alexa", "hey_jarvis", "hey_mycroft", "timer", "weather"]
        for name in builtin:
            models.append(
                {
                    "name": name,
                    "path": None,
                    "type": "builtin",
                }
            )

        return models
