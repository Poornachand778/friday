#!/usr/bin/env python3
"""
Friday AI Voice Daemon
======================

Main voice daemon process for JARVIS-style interaction.
Handles wake word detection, speech recognition, and voice synthesis.

Usage:
    python -m voice.daemon [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voice.config import VoiceConfig, get_voice_config, reload_config
from voice.audio.capture import AudioCapture
from voice.audio.playback import AudioPlayback
from voice.audio.vad import VoiceActivityDetector
from voice.orchestrator_client import OrchestratorClient, LocalFallbackClient


LOGGER = logging.getLogger(__name__)


class DaemonState(str, Enum):
    """Voice daemon state machine"""

    IDLE = "idle"  # Not listening
    LISTENING = "listening"  # Waiting for wake word
    WAKE_DETECTED = "wake_detected"  # Wake word heard, ready for command
    CAPTURING = "capturing"  # Recording user speech
    PROCESSING = "processing"  # Running STT + LLM
    SPEAKING = "speaking"  # TTS playback
    ERROR = "error"


@dataclass
class DaemonSession:
    """Active voice session"""

    session_id: str
    started_at: float
    wake_word: str
    wake_confidence: float
    turn_count: int = 0
    last_activity: float = 0.0


class VoiceDaemon:
    """
    Main voice daemon for Friday AI.

    State Machine:
        IDLE -> LISTENING (start())
        LISTENING -> WAKE_DETECTED (wake word)
        WAKE_DETECTED -> CAPTURING (voice detected)
        CAPTURING -> PROCESSING (silence detected)
        PROCESSING -> SPEAKING (response ready)
        SPEAKING -> LISTENING (playback complete)
        Any -> ERROR (exception)
        Any -> IDLE (stop())

    Usage:
        daemon = VoiceDaemon()
        daemon.start()  # Blocking
        # Or:
        await daemon.run_async()
    """

    def __init__(
        self,
        config: Optional[VoiceConfig] = None,
        config_path: Optional[Path] = None,
    ):
        self.config = config or get_voice_config(config_path)
        self._state = DaemonState.IDLE
        self._running = False

        # Components (lazy loaded)
        self._audio_capture: Optional[AudioCapture] = None
        self._audio_playback: Optional[AudioPlayback] = None
        self._vad: Optional[VoiceActivityDetector] = None
        self._wakeword_service = None
        self._stt_service = None
        self._tts_service = None
        self._orchestrator_client = None

        # Session tracking
        self._current_session: Optional[DaemonSession] = None
        self._session_timeout = self.config.daemon.idle_timeout_seconds

        # Signal handling
        self._shutdown_event = asyncio.Event()

    @property
    def state(self) -> DaemonState:
        return self._state

    def _set_state(self, state: DaemonState) -> None:
        """Update daemon state"""
        old_state = self._state
        self._state = state
        LOGGER.info("State: %s -> %s", old_state.value, state.value)

    def _load_components(self) -> None:
        """Lazy load all components"""
        LOGGER.info("Loading voice components...")

        # Audio I/O
        self._audio_capture = AudioCapture(self.config.audio)
        self._audio_playback = AudioPlayback(self.config.audio)
        self._vad = VoiceActivityDetector(self.config.audio)

        # Wake word detection
        try:
            from voice.wakeword import OpenWakeWordService

            self._wakeword_service = OpenWakeWordService(self.config.wakeword)
            LOGGER.info("Wake word service loaded")
        except ImportError as e:
            LOGGER.warning("Wake word not available: %s", e)

        # STT
        try:
            from voice.stt import FasterWhisperSTT

            self._stt_service = FasterWhisperSTT(self.config.stt)
            LOGGER.info("STT service loaded")
        except ImportError as e:
            LOGGER.warning("STT not available: %s", e)

        # TTS
        try:
            from voice.tts import XTTSService

            self._tts_service = XTTSService(self.config.tts)
            LOGGER.info("TTS service loaded")
        except ImportError as e:
            LOGGER.warning("TTS not available: %s", e)

        # Orchestrator client
        orchestrator_url = os.environ.get(
            "FRIDAY_ORCHESTRATOR_URL", "http://localhost:8000"
        )
        self._orchestrator_client = OrchestratorClient(base_url=orchestrator_url)
        LOGGER.info("Orchestrator client configured: %s", orchestrator_url)

        LOGGER.info("Voice components loaded")

    def start(self) -> None:
        """Start the daemon (blocking)"""
        LOGGER.info("Starting Friday Voice Daemon...")

        # Load components
        self._load_components()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self._running = True
        self._set_state(DaemonState.LISTENING)

        try:
            # Run the main loop
            asyncio.run(self._main_loop())
        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the daemon"""
        LOGGER.info("Stopping Voice Daemon...")
        self._running = False
        self._set_state(DaemonState.IDLE)

        # Cleanup
        if self._audio_capture:
            self._audio_capture.stop()
        if self._audio_playback:
            self._audio_playback.stop()
        if self._orchestrator_client:
            asyncio.get_event_loop().run_until_complete(
                self._orchestrator_client.close()
            )

        self._current_session = None

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals"""
        LOGGER.info("Received signal %d, shutting down...", signum)
        self._running = False
        self._shutdown_event.set()

    async def _main_loop(self) -> None:
        """Main daemon loop"""
        LOGGER.info("Daemon ready. Listening for wake word...")

        self._audio_capture.start()

        try:
            for chunk in self._audio_capture.stream():
                if not self._running:
                    break

                await self._process_chunk(chunk)

                # Check session timeout
                if self._current_session:
                    if (
                        time.time() - self._current_session.last_activity
                        > self._session_timeout
                    ):
                        LOGGER.info("Session timed out")
                        self._end_session()

        except Exception as e:
            LOGGER.error("Daemon error: %s", e)
            self._set_state(DaemonState.ERROR)
            raise

    async def _process_chunk(self, chunk) -> None:
        """Process an audio chunk based on current state"""
        if self._state == DaemonState.LISTENING:
            await self._handle_listening(chunk)
        elif self._state == DaemonState.WAKE_DETECTED:
            await self._handle_wake_detected(chunk)
        elif self._state == DaemonState.CAPTURING:
            await self._handle_capturing(chunk)

    async def _handle_listening(self, chunk) -> None:
        """Handle LISTENING state - wait for wake word"""
        if not self._wakeword_service:
            # No wake word detection - use VAD trigger
            self._vad.process_chunk(chunk)
            if self._vad.is_triggered:
                self._start_session("voice_trigger", 1.0)
                self._set_state(DaemonState.CAPTURING)
            return

        # Check for wake word
        detections = self._wakeword_service.process(chunk)
        if detections:
            detection = detections[0]
            LOGGER.info(
                "Wake word detected: %s (confidence=%.2f)",
                detection.wake_word,
                detection.confidence,
            )

            self._start_session(detection.wake_word, detection.confidence)
            self._set_state(DaemonState.WAKE_DETECTED)

            # Play acknowledgment sound or speak
            await self._play_acknowledgment()

    async def _handle_wake_detected(self, chunk) -> None:
        """Handle WAKE_DETECTED state - ready for speech"""
        # Use VAD to detect speech start
        self._vad.process_chunk(chunk)

        if self._vad.is_triggered:
            self._set_state(DaemonState.CAPTURING)
            self._audio_capture.start_recording()

    async def _handle_capturing(self, chunk) -> None:
        """Handle CAPTURING state - record user speech"""
        self._vad.process_chunk(chunk)

        if self._vad.utterance_ended():
            # Get recorded audio
            audio, duration = self._audio_capture.stop_recording()

            if duration > 0.5:  # Minimum utterance duration
                self._set_state(DaemonState.PROCESSING)
                await self._process_utterance(audio)
            else:
                LOGGER.debug("Utterance too short, ignoring")
                self._set_state(DaemonState.LISTENING)

    async def _process_utterance(self, audio: np.ndarray) -> None:
        """Process recorded utterance through STT -> Orchestrator -> TTS"""
        try:
            # STT
            if self._stt_service:
                LOGGER.info("Running STT...")
                result = self._stt_service.transcribe(audio)
                transcript = result.text
                language = result.language
                LOGGER.info("Transcript: '%s' (lang=%s)", transcript, language)
            else:
                transcript = "[STT not available]"
                language = "en"

            # Store user turn
            await self._store_turn(audio, transcript, language)

            # Get response from Friday Orchestrator
            session_id = (
                self._current_session.session_id if self._current_session else None
            )
            location = self.config.daemon.location  # e.g., "writers_room"

            LOGGER.info("Calling orchestrator...")
            orch_response = await self._orchestrator_client.chat(
                transcript=transcript,
                location=location,
                session_id=session_id,
            )
            response_text = orch_response.response
            LOGGER.info(
                "Response: '%s' (context=%s)", response_text[:50], orch_response.context
            )

            # TTS
            if self._tts_service and response_text:
                self._set_state(DaemonState.SPEAKING)
                LOGGER.info("Running TTS...")

                # Detect language for TTS (use transcript language or fallback to en)
                tts_language = language if language in ["en", "te"] else "en"

                tts_result = self._tts_service.synthesize(
                    response_text,
                    language=tts_language,
                )

                # Play response
                self._audio_playback.play(tts_result.audio, tts_result.sample_rate)

                # Store response audio
                await self._store_response(tts_result.audio, response_text)

            # Update session
            if self._current_session:
                self._current_session.turn_count += 1
                self._current_session.last_activity = time.time()

        except Exception as e:
            LOGGER.error("Processing failed: %s", e)
            self._set_state(DaemonState.ERROR)

        finally:
            self._set_state(DaemonState.LISTENING)
            self._vad.reset()

    async def _play_acknowledgment(self) -> None:
        """Play acknowledgment sound after wake word"""
        # Simple beep or short phrase
        if self._tts_service:
            try:
                result = self._tts_service.synthesize("Yes?", language="en")
                self._audio_playback.play(result.audio, result.sample_rate)
            except Exception as e:
                LOGGER.debug("Acknowledgment failed: %s", e)

    async def _store_turn(
        self,
        audio: np.ndarray,
        transcript: str,
        language: str,
    ) -> None:
        """Store user turn to storage"""
        try:
            from voice.storage import AudioStorage

            storage = AudioStorage()
            storage.store_user_audio(
                session_id=(
                    self._current_session.session_id
                    if self._current_session
                    else "unknown"
                ),
                turn_number=(
                    self._current_session.turn_count if self._current_session else 0
                ),
                audio=audio,
                transcript=transcript,
                language=language,
            )
        except Exception as e:
            LOGGER.debug("Storage failed: %s", e)

    async def _store_response(
        self,
        audio: np.ndarray,
        text: str,
    ) -> None:
        """Store response audio to storage"""
        try:
            from voice.storage import AudioStorage

            storage = AudioStorage()
            storage.store_response_audio(
                turn_id="",  # TODO: get from store_user_audio
                session_id=(
                    self._current_session.session_id
                    if self._current_session
                    else "unknown"
                ),
                turn_number=(
                    self._current_session.turn_count if self._current_session else 0
                ),
                audio=audio,
                text=text,
            )
        except Exception as e:
            LOGGER.debug("Response storage failed: %s", e)

    def _start_session(self, wake_word: str, confidence: float) -> None:
        """Start a new voice session"""
        self._current_session = DaemonSession(
            session_id=str(uuid.uuid4())[:8],
            started_at=time.time(),
            wake_word=wake_word,
            wake_confidence=confidence,
            last_activity=time.time(),
        )
        LOGGER.info("Started session: %s", self._current_session.session_id)

    def _end_session(self) -> None:
        """End current voice session"""
        if self._current_session:
            duration = time.time() - self._current_session.started_at
            LOGGER.info(
                "Ended session %s: %d turns, %.1fs duration",
                self._current_session.session_id,
                self._current_session.turn_count,
                duration,
            )
            self._current_session = None

        self._set_state(DaemonState.LISTENING)


def main(argv: list[str] | None = None) -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Friday AI Voice Daemon")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to voice config YAML",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        daemon = VoiceDaemon(config_path=args.config)
        daemon.start()
        return 0
    except Exception as e:
        LOGGER.error("Daemon failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
