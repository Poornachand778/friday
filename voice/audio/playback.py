"""
Audio Playback for Friday Voice Pipeline
=========================================

Speaker output handling using sounddevice.
Supports streaming playback for TTS output.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import sounddevice as sd
    import soundfile as sf
except ImportError as e:
    raise ImportError(
        "Audio playback dependencies not installed. Run: "
        "pip install sounddevice soundfile"
    ) from e

from voice.config import AudioConfig, get_voice_config


LOGGER = logging.getLogger(__name__)


class AudioPlayback:
    """
    Audio playback for TTS output.

    Supports both blocking and streaming playback modes.

    Usage:
        playback = AudioPlayback()

        # Blocking playback
        playback.play(audio_data)

        # Or play from file
        playback.play_file("response.wav")

        # Streaming playback
        playback.start_stream()
        playback.stream_chunk(chunk1)
        playback.stream_chunk(chunk2)
        playback.stop_stream()
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        device: Optional[int] = None,
    ):
        self.config = config or get_voice_config().audio
        self.device = device  # None = default output device

        self._stream: Optional[sd.OutputStream] = None
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._is_playing = False
        self._lock = threading.Lock()

    def play(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None,
        blocking: bool = True,
    ) -> None:
        """
        Play audio data.

        Args:
            audio_data: Audio samples to play
            sample_rate: Sample rate (default: 22050 for TTS, config for STT)
            blocking: Wait for playback to complete
        """
        sr = sample_rate or 22050  # XTTS default
        audio = audio_data.astype(np.float32)

        # Normalize if needed
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))

        LOGGER.debug("Playing audio: %.2f seconds at %d Hz", len(audio) / sr, sr)

        if blocking:
            sd.play(audio, sr, device=self.device)
            sd.wait()
        else:
            sd.play(audio, sr, device=self.device)

    def play_file(
        self,
        path: Union[str, Path],
        blocking: bool = True,
    ) -> float:
        """
        Play audio from file.

        Args:
            path: Path to audio file
            blocking: Wait for playback to complete

        Returns:
            Duration in seconds
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio, sr = sf.read(str(path))
        duration = len(audio) / sr

        LOGGER.info("Playing %s (%.2f seconds)", path.name, duration)
        self.play(audio, sample_rate=sr, blocking=blocking)

        return duration

    def stop(self) -> None:
        """Stop any currently playing audio"""
        sd.stop()
        LOGGER.debug("Playback stopped")

    # -----------------------------------------------------------------
    # Streaming playback (for real-time TTS)
    # -----------------------------------------------------------------

    def _stream_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for streaming playback"""
        if status:
            LOGGER.warning("Stream callback status: %s", status)

        try:
            data = self._queue.get_nowait()
            if data is None:
                # End of stream signal
                raise sd.CallbackStop()

            # Pad or truncate to match expected frames
            if len(data) < frames:
                outdata[: len(data), 0] = data
                outdata[len(data) :, 0] = 0
            else:
                outdata[:, 0] = data[:frames]

        except queue.Empty:
            # No data available, output silence
            outdata.fill(0)

    def start_stream(self, sample_rate: int = 22050) -> None:
        """Start streaming playback mode"""
        with self._lock:
            if self._is_playing:
                LOGGER.warning("Stream already active")
                return

            self._is_playing = True
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=1024,
                device=self.device,
                callback=self._stream_callback,
            )
            self._stream.start()
            LOGGER.debug("Streaming playback started")

    def stream_chunk(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to streaming queue"""
        if not self._is_playing:
            LOGGER.warning("Stream not active, starting...")
            self.start_stream()

        # Ensure float32 and normalized
        chunk = audio_chunk.astype(np.float32)
        if chunk.max() > 1.0 or chunk.min() < -1.0:
            chunk = chunk / max(abs(chunk.max()), abs(chunk.min()))

        self._queue.put(chunk)

    def stop_stream(self) -> None:
        """Stop streaming playback"""
        with self._lock:
            if not self._is_playing:
                return

            # Signal end of stream
            self._queue.put(None)

            # Wait for queue to drain
            time.sleep(0.1)

            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            self._is_playing = False
            LOGGER.debug("Streaming playback stopped")

    def get_output_devices(self) -> list[dict]:
        """List available output devices"""
        devices = sd.query_devices()
        output_devices = []

        for i, device in enumerate(devices):
            if device["max_output_channels"] > 0:
                output_devices.append(
                    {
                        "index": i,
                        "name": device["name"],
                        "channels": device["max_output_channels"],
                        "sample_rate": device["default_samplerate"],
                        "is_default": i == sd.default.device[1],
                    }
                )

        return output_devices

    @property
    def is_playing(self) -> bool:
        """Check if playback is active"""
        return self._is_playing

    def __enter__(self) -> "AudioPlayback":
        return self

    def __exit__(self, *args) -> None:
        self.stop()
        self.stop_stream()


def get_default_output_device() -> dict:
    """Get information about the default output device"""
    device_id = sd.default.device[1]
    device = sd.query_devices(device_id)
    return {
        "index": device_id,
        "name": device["name"],
        "channels": device["max_output_channels"],
        "sample_rate": device["default_samplerate"],
    }
