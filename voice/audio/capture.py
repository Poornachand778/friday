"""
Audio Capture for Friday Voice Pipeline
=======================================

Microphone input handling using sounddevice.
Streams audio chunks for real-time processing.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError as e:
    raise ImportError(
        "Audio capture dependencies not installed. Run: "
        "pip install sounddevice soundfile"
    ) from e

from voice.config import AudioConfig, get_voice_config


LOGGER = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """A chunk of audio data with metadata"""

    data: np.ndarray
    timestamp: float  # Unix timestamp
    sample_rate: int
    is_speech: bool = False  # Set by VAD
    chunk_index: int = 0


class AudioCapture:
    """
    Microphone audio capture with real-time streaming.

    Usage:
        capture = AudioCapture()
        capture.start()
        for chunk in capture.stream():
            process(chunk)
        capture.stop()
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        device: Optional[int] = None,
    ):
        self.config = config or get_voice_config().audio
        self.device = device  # None = default device

        self._queue: queue.Queue[Optional[AudioChunk]] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._chunk_index = 0
        self._lock = threading.Lock()

        # Recording buffer for saving to file
        self._recording_buffer: list[np.ndarray] = []
        self._is_recording = False
        self._recording_start: float = 0.0

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for audio stream - runs in separate thread"""
        if status:
            LOGGER.warning("Audio callback status: %s", status)

        timestamp = time.time()
        chunk = AudioChunk(
            data=indata.copy().flatten(),
            timestamp=timestamp,
            sample_rate=self.config.sample_rate,
            chunk_index=self._chunk_index,
        )
        self._chunk_index += 1

        # Add to recording buffer if recording
        if self._is_recording:
            self._recording_buffer.append(indata.copy())

        # Put in queue for processing
        self._queue.put(chunk)

    def start(self) -> None:
        """Start audio capture"""
        with self._lock:
            if self._running:
                LOGGER.warning("AudioCapture already running")
                return

            self._chunk_index = 0
            self._running = True

            # Create and start the stream
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.chunk_size,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
            LOGGER.info(
                "AudioCapture started: %d Hz, %d channels, device=%s",
                self.config.sample_rate,
                self.config.channels,
                self.device or "default",
            )

    def stop(self) -> None:
        """Stop audio capture"""
        with self._lock:
            if not self._running:
                return

            self._running = False

            # Stop and close stream
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            # Signal end of stream
            self._queue.put(None)

            LOGGER.info("AudioCapture stopped")

    def stream(self, timeout: float = 1.0) -> Iterator[AudioChunk]:
        """
        Iterate over audio chunks as they arrive.

        Yields:
            AudioChunk objects until stop() is called
        """
        while self._running or not self._queue.empty():
            try:
                chunk = self._queue.get(timeout=timeout)
                if chunk is None:
                    break
                yield chunk
            except queue.Empty:
                continue

    def start_recording(self) -> None:
        """Start recording audio to buffer"""
        self._recording_buffer = []
        self._recording_start = time.time()
        self._is_recording = True
        LOGGER.debug("Recording started")

    def stop_recording(self) -> tuple[np.ndarray, float]:
        """
        Stop recording and return captured audio.

        Returns:
            Tuple of (audio_data, duration_seconds)
        """
        self._is_recording = False
        duration = time.time() - self._recording_start

        if not self._recording_buffer:
            return np.array([], dtype=self.config.dtype), 0.0

        audio_data = np.concatenate(self._recording_buffer)
        self._recording_buffer = []

        LOGGER.debug("Recording stopped: %.2f seconds", duration)
        return audio_data, duration

    def save_recording(
        self,
        audio_data: np.ndarray,
        path: Path,
        sample_rate: Optional[int] = None,
    ) -> Path:
        """
        Save audio data to file.

        Args:
            audio_data: Audio samples
            path: Output file path
            sample_rate: Sample rate (default: config sample rate)

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sr = sample_rate or self.config.sample_rate
        sf.write(str(path), audio_data, sr)

        LOGGER.info("Saved audio to %s", path)
        return path

    def get_input_devices(self) -> list[dict]:
        """List available input devices"""
        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append(
                    {
                        "index": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "sample_rate": device["default_samplerate"],
                        "is_default": i == sd.default.device[0],
                    }
                )

        return input_devices

    @property
    def is_running(self) -> bool:
        """Check if capture is running"""
        return self._running

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()


def record_audio(
    duration_seconds: float,
    sample_rate: int = 16000,
    device: Optional[int] = None,
) -> np.ndarray:
    """
    Simple blocking audio recording.

    Args:
        duration_seconds: How long to record
        sample_rate: Sample rate in Hz
        device: Audio device index (None = default)

    Returns:
        Audio data as numpy array
    """
    LOGGER.info("Recording for %.1f seconds...", duration_seconds)
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        device=device,
    )
    sd.wait()
    return audio.flatten()


def get_default_input_device() -> dict:
    """Get information about the default input device"""
    device_id = sd.default.device[0]
    device = sd.query_devices(device_id)
    return {
        "index": device_id,
        "name": device["name"],
        "channels": device["max_input_channels"],
        "sample_rate": device["default_samplerate"],
    }
