"""
Faster-Whisper STT Service for Friday AI
=========================================

Real-time speech-to-text using Faster-Whisper (CTranslate2).
Optimized for Telugu-English bilingual transcription.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError as e:
    raise ImportError(
        "Faster-Whisper not installed. Run: " "pip install faster-whisper ctranslate2"
    ) from e

from voice.config import STTConfig, get_voice_config


LOGGER = logging.getLogger(__name__)


@dataclass
class WordTiming:
    """Word-level timing information"""

    word: str
    start: float  # seconds
    end: float
    probability: float


@dataclass
class TranscriptionResult:
    """Complete transcription result"""

    text: str
    language: str
    language_probability: float
    duration: float  # audio duration in seconds
    processing_time: float  # STT processing time in seconds

    # Word-level timings (if requested)
    words: List[WordTiming] = field(default_factory=list)

    # Segment-level info
    segments: List[dict] = field(default_factory=list)

    @property
    def is_telugu(self) -> bool:
        return self.language == "te"

    @property
    def is_english(self) -> bool:
        return self.language == "en"

    @property
    def rtf(self) -> float:
        """Real-time factor (processing_time / duration)"""
        if self.duration > 0:
            return self.processing_time / self.duration
        return 0.0


class FasterWhisperSTT:
    """
    Speech-to-Text service using Faster-Whisper.

    Features:
    - Automatic language detection (Telugu/English)
    - Word-level timestamps
    - VAD filtering for better accuracy
    - GPU acceleration (CUDA)

    Usage:
        stt = FasterWhisperSTT()
        result = stt.transcribe(audio_data)
        print(result.text, result.language)
    """

    # Supported models
    MODELS = [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large-v2",
        "large-v3",
        "distil-large-v2",
        "distil-large-v3",
    ]

    def __init__(
        self,
        config: Optional[STTConfig] = None,
        model_path: Optional[str] = None,
    ):
        self.config = config or get_voice_config().stt
        self._model: Optional[WhisperModel] = None
        self._model_path = model_path

    def _ensure_model_loaded(self) -> WhisperModel:
        """Lazy load the model"""
        if self._model is None:
            model_name = self._model_path or self.config.model
            LOGGER.info(
                "Loading Faster-Whisper model: %s (device=%s, compute=%s)",
                model_name,
                self.config.device,
                self.config.compute_type,
            )

            start_time = time.time()
            self._model = WhisperModel(
                model_name,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            load_time = time.time() - start_time
            LOGGER.info("Model loaded in %.2f seconds", load_time)

        return self._model

    def transcribe(
        self,
        audio: Union[np.ndarray, str, Path],
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data (np.ndarray), file path, or Path object
            language: Force specific language (None = auto-detect)
            **kwargs: Additional whisper options

        Returns:
            TranscriptionResult with text, language, and timing info
        """
        model = self._ensure_model_loaded()

        # Prepare audio
        if isinstance(audio, (str, Path)):
            audio_input = str(audio)
            audio_duration = self._get_audio_duration_from_file(audio_input)
        else:
            audio_input = self._prepare_audio(audio)
            audio_duration = len(audio) / 16000  # Assume 16kHz

        # Transcription options
        options = {
            "beam_size": self.config.beam_size,
            "word_timestamps": self.config.word_timestamps,
            "vad_filter": self.config.vad_filter,
        }

        # Language setting
        if language:
            options["language"] = language
        elif self.config.language:
            options["language"] = self.config.language

        options.update(kwargs)

        # Run transcription
        LOGGER.debug("Starting transcription (duration=%.2fs)", audio_duration)
        start_time = time.time()

        segments, info = model.transcribe(audio_input, **options)

        # Collect results
        text_parts = []
        words = []
        segment_list = []

        for segment in segments:
            text_parts.append(segment.text)
            segment_list.append(
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }
            )

            # Collect word timings if available
            if (
                self.config.word_timestamps
                and hasattr(segment, "words")
                and segment.words
            ):
                for word in segment.words:
                    words.append(
                        WordTiming(
                            word=word.word,
                            start=word.start,
                            end=word.end,
                            probability=word.probability,
                        )
                    )

        processing_time = time.time() - start_time
        full_text = "".join(text_parts).strip()

        result = TranscriptionResult(
            text=full_text,
            language=info.language,
            language_probability=info.language_probability,
            duration=audio_duration,
            processing_time=processing_time,
            words=words,
            segments=segment_list,
        )

        LOGGER.info(
            "Transcription complete: '%s' (lang=%s@%.2f, RTF=%.2f)",
            full_text[:50] + "..." if len(full_text) > 50 else full_text,
            result.language,
            result.language_probability,
            result.rtf,
        )

        return result

    def transcribe_stream(
        self,
        audio_chunks: Iterator[np.ndarray],
        chunk_duration: float = 0.5,
        language: Optional[str] = None,
    ) -> Iterator[TranscriptionResult]:
        """
        Stream transcription for real-time processing.

        Buffers audio chunks and transcribes when sufficient audio is collected.

        Args:
            audio_chunks: Iterator of audio chunks
            chunk_duration: Minimum duration before transcribing
            language: Force specific language

        Yields:
            TranscriptionResult for each transcribed segment
        """
        buffer = []
        buffer_duration = 0.0
        sample_rate = 16000

        for chunk in audio_chunks:
            buffer.append(chunk)
            buffer_duration += len(chunk) / sample_rate

            if buffer_duration >= chunk_duration:
                # Transcribe buffered audio
                audio = np.concatenate(buffer)
                result = self.transcribe(audio, language=language)

                if result.text.strip():
                    yield result

                # Reset buffer
                buffer = []
                buffer_duration = 0.0

        # Transcribe remaining audio
        if buffer:
            audio = np.concatenate(buffer)
            result = self.transcribe(audio, language=language)
            if result.text.strip():
                yield result

    def detect_language(
        self,
        audio: Union[np.ndarray, str, Path],
    ) -> Tuple[str, float]:
        """
        Detect language of audio.

        Args:
            audio: Audio data or file path

        Returns:
            Tuple of (language_code, probability)
        """
        model = self._ensure_model_loaded()

        if isinstance(audio, (str, Path)):
            audio_input = str(audio)
        else:
            audio_input = self._prepare_audio(audio)

        # Get first 30 seconds for language detection
        _, info = model.transcribe(
            audio_input,
            beam_size=1,
            word_timestamps=False,
            vad_filter=False,
        )

        return info.language, info.language_probability

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """Prepare audio for Whisper (float32, mono, 16kHz)"""
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        return audio

    def _get_audio_duration_from_file(self, path: str) -> float:
        """Get audio duration from file"""
        try:
            import soundfile as sf

            info = sf.info(path)
            return info.duration
        except Exception:
            return 0.0

    def unload_model(self) -> None:
        """Unload model to free GPU memory"""
        if self._model is not None:
            del self._model
            self._model = None
            LOGGER.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None


# Convenience function
def transcribe_audio(
    audio: Union[np.ndarray, str, Path],
    language: Optional[str] = None,
    model: str = "large-v3",
    device: str = "cuda",
) -> TranscriptionResult:
    """
    Quick transcription function.

    Args:
        audio: Audio data or file path
        language: Force language (None = auto-detect)
        model: Whisper model name
        device: Device to use (cuda/cpu)

    Returns:
        TranscriptionResult
    """
    config = STTConfig(model=model, device=device)
    stt = FasterWhisperSTT(config)
    return stt.transcribe(audio, language=language)
