"""
OpenWakeWord Service for Friday AI
===================================

Wake word detection using OpenWakeWord.
Supports multiple wake phrases with adjustable thresholds.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import openwakeword
    from openwakeword.model import Model as OWWModel
except ImportError as e:
    raise ImportError(
        "OpenWakeWord not installed. Run: pip install openwakeword"
    ) from e

from voice.config import WakeWordConfig, get_voice_config
from voice.audio.capture import AudioChunk


LOGGER = logging.getLogger(__name__)


# OpenWakeWord expects 16kHz, 16-bit audio
SAMPLE_RATE = 16000


@dataclass
class WakeWordDetection:
    """Wake word detection result"""

    wake_word: str
    confidence: float
    timestamp: float
    audio_offset_samples: int

    @property
    def is_confident(self) -> bool:
        """Check if detection meets threshold"""
        return self.confidence >= 0.5


class OpenWakeWordService:
    """
    Wake word detection using OpenWakeWord.

    Features:
    - Multiple wake words with individual thresholds
    - Continuous monitoring mode
    - Custom model support
    - Noise suppression

    Built-in models:
    - alexa, hey_jarvis, hey_mycroft, timer, weather

    Custom models:
    - Train "hey_friday" or "wake_up_daddys_home" using trainer

    Usage:
        wakeword = OpenWakeWordService()
        wakeword.add_model("alexa", threshold=0.5)

        for chunk in audio_stream:
            detections = wakeword.process(chunk)
            for detection in detections:
                if detection.is_confident:
                    print(f"Detected: {detection.wake_word}")
    """

    # Built-in OpenWakeWord models
    BUILTIN_MODELS = [
        "alexa",
        "hey_jarvis",
        "hey_mycroft",
        "timer",
        "weather",
        "hey_siri",  # May not be available
    ]

    def __init__(
        self,
        config: Optional[WakeWordConfig] = None,
    ):
        self.config = config or get_voice_config().wakeword
        self._model: Optional[OWWModel] = None
        self._thresholds: Dict[str, float] = {}
        self._detection_history: List[WakeWordDetection] = []

    def _ensure_model_loaded(self) -> OWWModel:
        """Lazy load the wake word model"""
        if self._model is None:
            LOGGER.info("Loading OpenWakeWord models...")
            start_time = time.time()

            # Collect model names and paths
            model_paths = []
            for model_config in self.config.models:
                name = model_config.get("name", "")
                path = model_config.get("path")
                threshold = model_config.get("threshold", 0.5)

                if path:
                    # Custom model
                    model_paths.append(path)
                    self._thresholds[Path(path).stem] = threshold
                elif name in self.BUILTIN_MODELS:
                    # Built-in model
                    self._thresholds[name] = threshold
                else:
                    LOGGER.warning("Unknown wake word model: %s", name)

            # Initialize model
            self._model = OWWModel(
                wakeword_models=model_paths if model_paths else None,
                inference_framework=self.config.inference_framework,
            )

            load_time = time.time() - start_time
            LOGGER.info(
                "OpenWakeWord loaded in %.2fs with models: %s",
                load_time,
                list(self._thresholds.keys()),
            )

        return self._model

    def add_model(
        self,
        name: str,
        path: Optional[str] = None,
        threshold: float = 0.5,
    ) -> None:
        """
        Add a wake word model.

        Args:
            name: Model name (builtin) or identifier (custom)
            path: Path to custom model file (.onnx)
            threshold: Detection threshold (0-1)
        """
        self._thresholds[name] = threshold

        # If model already loaded, need to reload with new model
        if self._model is not None and path:
            LOGGER.warning("Model already loaded. Restart to add custom models.")

    def process(self, chunk: AudioChunk) -> List[WakeWordDetection]:
        """
        Process audio chunk for wake word detection.

        Args:
            chunk: Audio chunk to process

        Returns:
            List of wake word detections (may be empty)
        """
        model = self._ensure_model_loaded()

        # Prepare audio (OpenWakeWord expects int16)
        audio = chunk.data
        if audio.dtype != np.int16:
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)

        # Run detection
        predictions = model.predict(audio)

        detections = []
        for model_name, score in predictions.items():
            threshold = self._thresholds.get(model_name, 0.5)

            if score >= threshold:
                detection = WakeWordDetection(
                    wake_word=model_name,
                    confidence=float(score),
                    timestamp=chunk.timestamp,
                    audio_offset_samples=chunk.chunk_index * len(chunk.data),
                )
                detections.append(detection)
                self._detection_history.append(detection)

                LOGGER.info(
                    "Wake word detected: %s (confidence=%.3f)",
                    model_name,
                    score,
                )

        return detections

    def process_batch(
        self,
        audio: np.ndarray,
        chunk_size: int = 1280,  # 80ms at 16kHz
    ) -> List[WakeWordDetection]:
        """
        Process a batch of audio for wake word detection.

        Args:
            audio: Audio data (int16, 16kHz)
            chunk_size: Samples per processing chunk

        Returns:
            List of all detections
        """
        model = self._ensure_model_loaded()

        all_detections = []
        timestamp = time.time()

        # Process in chunks
        for i in range(0, len(audio), chunk_size):
            chunk_data = audio[i : i + chunk_size]
            if len(chunk_data) < chunk_size:
                # Pad last chunk
                chunk_data = np.pad(chunk_data, (0, chunk_size - len(chunk_data)))

            chunk = AudioChunk(
                data=chunk_data,
                timestamp=timestamp + (i / SAMPLE_RATE),
                sample_rate=SAMPLE_RATE,
                chunk_index=i // chunk_size,
            )

            detections = self.process(chunk)
            all_detections.extend(detections)

        return all_detections

    def monitor_stream(
        self,
        audio_chunks: Iterator[AudioChunk],
        timeout_seconds: float = 60.0,
    ) -> Optional[WakeWordDetection]:
        """
        Monitor audio stream for wake word.

        Blocks until wake word detected or timeout.

        Args:
            audio_chunks: Iterator of audio chunks
            timeout_seconds: Maximum time to wait

        Returns:
            WakeWordDetection if detected, None if timeout
        """
        start_time = time.time()

        for chunk in audio_chunks:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                LOGGER.debug("Wake word monitoring timed out")
                return None

            detections = self.process(chunk)
            if detections:
                return detections[0]  # Return first detection

        return None

    def get_detection_history(
        self,
        limit: int = 100,
    ) -> List[WakeWordDetection]:
        """Get recent detection history"""
        return self._detection_history[-limit:]

    def clear_history(self) -> None:
        """Clear detection history"""
        self._detection_history = []

    def reset(self) -> None:
        """Reset model state"""
        if self._model is not None:
            self._model.reset()
        self.clear_history()
        LOGGER.debug("Wake word detector reset")

    @property
    def active_models(self) -> List[str]:
        """List of active wake word models"""
        return list(self._thresholds.keys())

    def get_threshold(self, model_name: str) -> float:
        """Get threshold for a model"""
        return self._thresholds.get(model_name, 0.5)

    def set_threshold(self, model_name: str, threshold: float) -> None:
        """Set threshold for a model"""
        if 0.0 <= threshold <= 1.0:
            self._thresholds[model_name] = threshold
            LOGGER.debug("Set threshold for %s: %.2f", model_name, threshold)
        else:
            raise ValueError("Threshold must be between 0 and 1")


def detect_wake_word(
    audio: np.ndarray,
    models: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Optional[WakeWordDetection]:
    """
    Quick wake word detection function.

    Args:
        audio: Audio data (int16, 16kHz)
        models: List of model names to use
        threshold: Detection threshold

    Returns:
        First detection if any, None otherwise
    """
    config = WakeWordConfig(
        models=[{"name": m, "threshold": threshold} for m in (models or ["alexa"])]
    )
    service = OpenWakeWordService(config)
    detections = service.process_batch(audio)
    return detections[0] if detections else None
