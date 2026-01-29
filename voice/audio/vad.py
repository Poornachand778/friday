"""
Voice Activity Detection for Friday Voice Pipeline
==================================================

WebRTC VAD integration for detecting speech in audio streams.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterator, Optional

import numpy as np

try:
    import webrtcvad
except ImportError as e:
    raise ImportError(
        "VAD dependencies not installed. Run: pip install webrtcvad"
    ) from e

from voice.config import AudioConfig, get_voice_config
from voice.audio.capture import AudioChunk


LOGGER = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Voice Activity Detection using WebRTC VAD.

    Detects speech segments in audio stream for:
    - Filtering silence before STT
    - Detecting end of utterance
    - Improving wake word detection

    Usage:
        vad = VoiceActivityDetector()

        for chunk in audio_stream:
            is_speech = vad.is_speech(chunk.data)
            chunk.is_speech = is_speech

            if vad.utterance_ended():
                process_utterance(vad.get_utterance())
    """

    # WebRTC VAD only supports specific frame durations (10, 20, 30 ms)
    VALID_FRAME_DURATIONS_MS = (10, 20, 30)

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        aggressiveness: Optional[int] = None,
    ):
        self.config = config or get_voice_config().audio
        self.aggressiveness = aggressiveness or self.config.vad_aggressiveness

        # Validate aggressiveness (0-3)
        if not 0 <= self.aggressiveness <= 3:
            raise ValueError("VAD aggressiveness must be 0-3")

        self._vad = webrtcvad.Vad(self.aggressiveness)

        # Frame size for WebRTC VAD (must be 10, 20, or 30ms)
        self._frame_duration_ms = 30  # 30ms frames work well
        self._frame_size = int(self.config.sample_rate * self._frame_duration_ms / 1000)

        # State tracking
        self._ring_buffer: deque = deque(maxlen=30)  # ~900ms of history
        self._triggered = False  # Currently in speech segment
        self._utterance_buffer: list[np.ndarray] = []

        # Silence detection
        self._silence_frames = 0
        self._silence_threshold_frames = int(
            self.config.silence_threshold_ms / self._frame_duration_ms
        )

        LOGGER.debug(
            "VAD initialized: aggressiveness=%d, frame_size=%d, silence_threshold=%d frames",
            self.aggressiveness,
            self._frame_size,
            self._silence_threshold_frames,
        )

    def is_speech(self, audio_data: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech.

        Args:
            audio_data: Audio samples (int16)

        Returns:
            True if speech detected
        """
        # Ensure int16 format
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        # Process in valid frame sizes
        frames = self._split_into_frames(audio_data)
        if not frames:
            return False

        # Check each frame
        speech_frames = 0
        for frame in frames:
            try:
                is_speech = self._vad.is_speech(
                    frame.tobytes(), self.config.sample_rate
                )
                if is_speech:
                    speech_frames += 1
            except Exception as e:
                LOGGER.debug("VAD error: %s", e)
                continue

        # Consider speech if majority of frames are speech
        return speech_frames > len(frames) / 2

    def process_chunk(self, chunk: AudioChunk) -> AudioChunk:
        """
        Process audio chunk and update speech state.

        Args:
            chunk: Audio chunk to process

        Returns:
            Chunk with is_speech flag set
        """
        is_speech = self.is_speech(chunk.data)
        chunk.is_speech = is_speech

        # Update state machine
        self._update_state(chunk.data, is_speech)

        return chunk

    def _update_state(self, audio_data: np.ndarray, is_speech: bool) -> None:
        """Update VAD state machine"""
        self._ring_buffer.append((audio_data, is_speech))

        if not self._triggered:
            # Not in speech - look for start
            num_voiced = sum(1 for _, speech in self._ring_buffer if speech)
            if num_voiced > 0.9 * len(self._ring_buffer):
                # Enough speech frames - trigger
                self._triggered = True
                self._silence_frames = 0
                LOGGER.debug("Speech started")

                # Add buffered audio to utterance
                for data, _ in self._ring_buffer:
                    self._utterance_buffer.append(data)
                self._ring_buffer.clear()

        else:
            # In speech - accumulate and look for end
            self._utterance_buffer.append(audio_data)

            if not is_speech:
                self._silence_frames += 1
            else:
                self._silence_frames = 0

    def utterance_ended(self) -> bool:
        """Check if current utterance has ended (silence detected)"""
        if not self._triggered:
            return False

        if self._silence_frames >= self._silence_threshold_frames:
            LOGGER.debug("Speech ended (silence detected)")
            return True

        return False

    def get_utterance(self) -> np.ndarray:
        """
        Get accumulated utterance audio and reset state.

        Returns:
            Concatenated audio from speech segment
        """
        if not self._utterance_buffer:
            return np.array([], dtype=np.int16)

        # Concatenate all buffers
        utterance = np.concatenate(self._utterance_buffer)

        # Reset state
        self._triggered = False
        self._utterance_buffer = []
        self._silence_frames = 0
        self._ring_buffer.clear()

        return utterance

    def reset(self) -> None:
        """Reset VAD state"""
        self._triggered = False
        self._utterance_buffer = []
        self._silence_frames = 0
        self._ring_buffer.clear()
        LOGGER.debug("VAD state reset")

    def _split_into_frames(self, audio_data: np.ndarray) -> list[np.ndarray]:
        """Split audio into VAD-compatible frames"""
        frames = []
        offset = 0

        while offset + self._frame_size <= len(audio_data):
            frame = audio_data[offset : offset + self._frame_size]
            frames.append(frame)
            offset += self._frame_size

        return frames

    @property
    def is_triggered(self) -> bool:
        """Check if currently in speech segment"""
        return self._triggered

    @property
    def utterance_duration_ms(self) -> float:
        """Get current utterance duration in milliseconds"""
        if not self._utterance_buffer:
            return 0.0

        total_samples = sum(len(buf) for buf in self._utterance_buffer)
        return (total_samples / self.config.sample_rate) * 1000


def vad_filter_stream(
    audio_stream: Iterator[AudioChunk],
    config: Optional[AudioConfig] = None,
) -> Iterator[AudioChunk]:
    """
    Filter audio stream to only yield speech chunks.

    Args:
        audio_stream: Iterator of audio chunks
        config: Audio configuration

    Yields:
        Audio chunks that contain speech
    """
    vad = VoiceActivityDetector(config)

    for chunk in audio_stream:
        processed = vad.process_chunk(chunk)
        if processed.is_speech:
            yield processed


def collect_utterance(
    audio_stream: Iterator[AudioChunk],
    config: Optional[AudioConfig] = None,
    max_duration_seconds: float = 30.0,
) -> tuple[np.ndarray, float]:
    """
    Collect a complete utterance from audio stream.

    Waits for speech to start, then collects until silence.

    Args:
        audio_stream: Iterator of audio chunks
        config: Audio configuration
        max_duration_seconds: Maximum utterance duration

    Returns:
        Tuple of (audio_data, duration_seconds)
    """
    cfg = config or get_voice_config().audio
    vad = VoiceActivityDetector(cfg)

    max_samples = int(max_duration_seconds * cfg.sample_rate)
    collected_samples = 0

    for chunk in audio_stream:
        vad.process_chunk(chunk)
        collected_samples += len(chunk.data)

        if vad.utterance_ended() or collected_samples >= max_samples:
            utterance = vad.get_utterance()
            duration = len(utterance) / cfg.sample_rate
            return utterance, duration

    # Stream ended without complete utterance
    utterance = vad.get_utterance()
    duration = len(utterance) / cfg.sample_rate if len(utterance) > 0 else 0.0
    return utterance, duration
