"""
XTTS v2 TTS Service for Friday AI
==================================

Text-to-Speech using Coqui XTTS v2 with voice cloning.
Optimized for Telugu and English synthesis.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
except ImportError as e:
    raise ImportError("Coqui TTS not installed. Run: pip install TTS") from e

from voice.config import TTSConfig, get_voice_config


LOGGER = logging.getLogger(__name__)


# Language code mapping for XTTS
LANGUAGE_MAP = {
    "te": "te",  # Telugu - Not natively supported, use multilingual mode
    "en": "en",  # English
    "hi": "hi",  # Hindi
    "ta": "ta",  # Tamil
    # XTTS v2 supported: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko
}

# For unsupported languages, use closest alternative
LANGUAGE_FALLBACK = {
    "te": "hi",  # Telugu -> Hindi (closest Indic language supported)
}


@dataclass
class TTSResult:
    """TTS synthesis result"""

    audio: np.ndarray
    sample_rate: int
    duration: float  # seconds
    processing_time: float  # seconds
    language: str
    voice_profile: str

    @property
    def rtf(self) -> float:
        """Real-time factor (processing_time / duration)"""
        if self.duration > 0:
            return self.processing_time / self.duration
        return 0.0


class XTTSService:
    """
    Text-to-Speech service using XTTS v2.

    Features:
    - Voice cloning from reference audio
    - Telugu language support (via multilingual model)
    - Streaming synthesis for low-latency
    - Multiple voice profiles

    Usage:
        tts = XTTSService()

        # Basic synthesis
        result = tts.synthesize("Hello world", language="en")
        play_audio(result.audio, result.sample_rate)

        # With voice cloning
        tts.load_voice_profile("friday_telugu", ["voice_sample.wav"])
        result = tts.synthesize("నమస్కారం", language="te", profile="friday_telugu")
    """

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        model_path: Optional[str] = None,
    ):
        self.config = config or get_voice_config().tts
        self._model: Optional[TTS] = None
        self._xtts_model: Optional[Xtts] = None
        self._model_path = model_path

        # Voice profile cache
        self._voice_profiles: dict[str, dict] = {}

        # Sample rate for XTTS v2
        self.sample_rate = 24000

    def _ensure_model_loaded(self) -> TTS:
        """Lazy load the XTTS model"""
        if self._model is None:
            LOGGER.info("Loading XTTS v2 model (device=%s)...", self.config.device)
            start_time = time.time()

            # Use XTTS v2 model
            model_name = (
                self._model_path or "tts_models/multilingual/multi-dataset/xtts_v2"
            )

            self._model = TTS(model_name=model_name).to(self.config.device)

            load_time = time.time() - start_time
            LOGGER.info("XTTS model loaded in %.2f seconds", load_time)

        return self._model

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        profile: Optional[str] = None,
        speaker_wav: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> TTSResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            language: Language code (te, en, hi, etc.)
            profile: Voice profile name (if loaded)
            speaker_wav: Reference audio for cloning (alternative to profile)
            **kwargs: Additional XTTS options

        Returns:
            TTSResult with audio data
        """
        model = self._ensure_model_loaded()

        # Determine language
        lang = language or self.config.default_language
        xtts_lang = self._get_xtts_language(lang)

        # Get speaker reference
        speaker_audio = None
        profile_name = profile or self.config.default_profile

        if speaker_wav:
            speaker_audio = (
                speaker_wav if isinstance(speaker_wav, list) else [speaker_wav]
            )
        elif profile_name and profile_name in self._voice_profiles:
            speaker_audio = self._voice_profiles[profile_name].get("reference_audio")

        LOGGER.debug(
            "Synthesizing: text='%s' lang=%s profile=%s",
            text[:50] + "..." if len(text) > 50 else text,
            xtts_lang,
            profile_name,
        )

        start_time = time.time()

        # Synthesize
        if speaker_audio:
            # Voice cloning mode
            audio = model.tts(
                text=text,
                language=xtts_lang,
                speaker_wav=(
                    speaker_audio[0] if len(speaker_audio) == 1 else speaker_audio
                ),
                split_sentences=True,
            )
        else:
            # Use default speaker (first speaker in model)
            audio = model.tts(
                text=text,
                language=xtts_lang,
                split_sentences=True,
            )

        processing_time = time.time() - start_time

        # Convert to numpy
        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = np.array(audio, dtype=np.float32)

        duration = len(audio) / self.sample_rate

        result = TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration=duration,
            processing_time=processing_time,
            language=lang,
            voice_profile=profile_name,
        )

        LOGGER.info(
            "Synthesis complete: %.2fs audio in %.2fs (RTF=%.2f)",
            result.duration,
            result.processing_time,
            result.rtf,
        )

        return result

    def synthesize_stream(
        self,
        text: str,
        language: Optional[str] = None,
        profile: Optional[str] = None,
        chunk_size: int = 8192,
    ) -> Iterator[np.ndarray]:
        """
        Stream synthesis for low-latency playback.

        Yields audio chunks as they're generated.

        Args:
            text: Text to synthesize
            language: Language code
            profile: Voice profile name
            chunk_size: Samples per chunk

        Yields:
            Audio chunks (numpy arrays)
        """
        # For now, synthesize full audio then chunk it
        # TODO: Implement true streaming when XTTS supports it
        result = self.synthesize(text, language=language, profile=profile)

        # Yield in chunks
        audio = result.audio
        offset = 0

        while offset < len(audio):
            end = min(offset + chunk_size, len(audio))
            yield audio[offset:end]
            offset = end

    def load_voice_profile(
        self,
        name: str,
        reference_audio: List[Union[str, Path]],
        language: str = "te",
    ) -> None:
        """
        Load a voice profile for cloning.

        Args:
            name: Profile name
            reference_audio: List of reference audio file paths
            language: Primary language for this voice
        """
        # Validate audio files exist
        audio_paths = []
        for path in reference_audio:
            path = Path(path)
            if not path.exists():
                LOGGER.warning("Reference audio not found: %s", path)
                continue
            audio_paths.append(str(path))

        if not audio_paths:
            raise ValueError(f"No valid reference audio files for profile '{name}'")

        self._voice_profiles[name] = {
            "reference_audio": audio_paths,
            "language": language,
        }

        LOGGER.info(
            "Loaded voice profile '%s' with %d reference audio files",
            name,
            len(audio_paths),
        )

    def list_profiles(self) -> List[str]:
        """List loaded voice profiles"""
        return list(self._voice_profiles.keys())

    def _get_xtts_language(self, lang: str) -> str:
        """Map language code to XTTS-supported language"""
        if lang in LANGUAGE_MAP:
            xtts_lang = LANGUAGE_MAP[lang]
            # Check if XTTS actually supports this
            # XTTS v2 supported: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi
            xtts_supported = {
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cs",
                "ar",
                "zh-cn",
                "ja",
                "hu",
                "ko",
                "hi",
            }
            if xtts_lang not in xtts_supported:
                # Use fallback
                xtts_lang = LANGUAGE_FALLBACK.get(lang, "en")
                LOGGER.debug(
                    "Language %s not supported, falling back to %s", lang, xtts_lang
                )
            return xtts_lang

        LOGGER.warning("Unknown language '%s', defaulting to 'en'", lang)
        return "en"

    def unload_model(self) -> None:
        """Unload model to free GPU memory"""
        if self._model is not None:
            del self._model
            self._model = None
            LOGGER.info("TTS model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None


def synthesize_speech(
    text: str,
    language: str = "te",
    speaker_wav: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[np.ndarray, int]:
    """
    Quick synthesis function.

    Args:
        text: Text to synthesize
        language: Language code
        speaker_wav: Reference audio for voice cloning
        device: Device (cuda/cpu)

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    config = TTSConfig(device=device, default_language=language)
    tts = XTTSService(config)
    result = tts.synthesize(text, language=language, speaker_wav=speaker_wav)
    return result.audio, result.sample_rate


def save_audio(
    audio: np.ndarray,
    path: Union[str, Path],
    sample_rate: int = 24000,
) -> Path:
    """Save audio to file"""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(path), audio, sample_rate)
    LOGGER.info("Saved audio to %s", path)
    return path
