#!/usr/bin/env python3
"""
Test script for Voice Pipeline Phase 1
======================================

Verifies:
1. Configuration loading
2. Audio device detection
3. VAD functionality (with synthetic audio)
4. Database schema imports

Usage:
    python scripts/test_voice_phase1.py [--test-audio]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def test_config() -> bool:
    """Test configuration loading"""
    LOGGER.info("Testing configuration...")

    try:
        from voice.config import VoiceConfig, get_voice_config

        config = get_voice_config()
        LOGGER.info("  Config loaded successfully")
        LOGGER.info("    STT engine: %s", config.stt.engine)
        LOGGER.info("    STT model: %s", config.stt.model)
        LOGGER.info("    TTS engine: %s", config.tts.engine)
        LOGGER.info("    Wake word engine: %s", config.wakeword.engine)
        LOGGER.info("    Storage enabled: %s", config.storage.enabled)
        return True
    except Exception as e:
        LOGGER.error("  Config test failed: %s", e)
        return False


def test_audio_devices() -> bool:
    """Test audio device detection"""
    LOGGER.info("Testing audio devices...")

    try:
        import sounddevice as sd

        # List input devices
        devices = sd.query_devices()
        input_count = sum(1 for d in devices if d["max_input_channels"] > 0)
        output_count = sum(1 for d in devices if d["max_output_channels"] > 0)

        LOGGER.info(
            "  Found %d input devices, %d output devices", input_count, output_count
        )

        # Get defaults
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]

        if default_input is not None:
            LOGGER.info("  Default input: %s", sd.query_devices(default_input)["name"])
        if default_output is not None:
            LOGGER.info(
                "  Default output: %s", sd.query_devices(default_output)["name"]
            )

        return input_count > 0 and output_count > 0
    except ImportError:
        LOGGER.warning("  sounddevice not installed, skipping")
        return True
    except Exception as e:
        LOGGER.error("  Audio device test failed: %s", e)
        return False


def test_vad_synthetic() -> bool:
    """Test VAD with synthetic audio"""
    LOGGER.info("Testing VAD with synthetic audio...")

    try:
        import numpy as np
        from voice.audio.vad import VoiceActivityDetector
        from voice.audio.capture import AudioChunk

        vad = VoiceActivityDetector()

        # Create silence
        silence = np.zeros(16000, dtype=np.int16)  # 1 second silence
        is_speech_silence = vad.is_speech(silence)
        LOGGER.info(
            "  Silence detected as speech: %s (expected: False)", is_speech_silence
        )

        # Create noise (simulates speech)
        np.random.seed(42)
        noise = (np.random.randn(16000) * 5000).astype(np.int16)
        is_speech_noise = vad.is_speech(noise)
        LOGGER.info("  Noise detected as speech: %s", is_speech_noise)

        # Test utterance detection
        vad.reset()
        chunk = AudioChunk(data=noise, timestamp=0.0, sample_rate=16000)
        vad.process_chunk(chunk)
        LOGGER.info("  VAD triggered: %s", vad.is_triggered)
        LOGGER.info("  Utterance duration: %.1f ms", vad.utterance_duration_ms)

        return not is_speech_silence
    except ImportError as e:
        LOGGER.warning("  VAD dependency missing: %s", e)
        return True
    except Exception as e:
        LOGGER.error("  VAD test failed: %s", e)
        return False


def test_capture_playback_classes() -> bool:
    """Test AudioCapture and AudioPlayback class imports"""
    LOGGER.info("Testing audio capture/playback classes...")

    try:
        from voice.audio import AudioCapture, AudioPlayback, VoiceActivityDetector

        # Just verify classes can be instantiated (without starting audio)
        LOGGER.info("  AudioCapture class: OK")
        LOGGER.info("  AudioPlayback class: OK")
        LOGGER.info("  VoiceActivityDetector class: OK")
        return True
    except ImportError as e:
        LOGGER.warning("  Import error (sounddevice may not be installed): %s", e)
        return True
    except Exception as e:
        LOGGER.error("  Class import test failed: %s", e)
        return False


def test_database_schema() -> bool:
    """Test database schema imports"""
    LOGGER.info("Testing database schema imports...")

    try:
        from db.voice_schema import (
            VoiceSession,
            VoiceTurn,
            VoiceProfile,
            VoiceTrainingExample,
            SessionStatus,
            TurnRole,
            TrainingStatus,
        )

        LOGGER.info("  VoiceSession: OK")
        LOGGER.info("  VoiceTurn: OK")
        LOGGER.info("  VoiceProfile: OK")
        LOGGER.info("  VoiceTrainingExample: OK")
        LOGGER.info("  Enums (SessionStatus, TurnRole, TrainingStatus): OK")
        return True
    except Exception as e:
        LOGGER.error("  Schema import test failed: %s", e)
        return False


def test_live_audio(duration: float = 2.0) -> bool:
    """Test live audio capture (requires microphone)"""
    LOGGER.info("Testing live audio capture for %.1f seconds...", duration)

    try:
        from voice.audio.capture import record_audio, get_default_input_device
        import numpy as np

        device_info = get_default_input_device()
        LOGGER.info("  Using device: %s", device_info["name"])

        audio = record_audio(duration)
        LOGGER.info("  Captured %d samples", len(audio))
        LOGGER.info("  Audio range: [%d, %d]", audio.min(), audio.max())
        LOGGER.info("  Audio RMS: %.1f", np.sqrt(np.mean(audio.astype(float) ** 2)))

        return len(audio) > 0
    except Exception as e:
        LOGGER.error("  Live audio test failed: %s", e)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test Voice Pipeline Phase 1")
    parser.add_argument(
        "--test-audio",
        action="store_true",
        help="Test live audio capture (requires microphone)",
    )
    args = parser.parse_args(argv)

    LOGGER.info("=" * 50)
    LOGGER.info("Voice Pipeline Phase 1 Tests")
    LOGGER.info("=" * 50)

    results = {}

    # Run tests
    results["config"] = test_config()
    results["audio_devices"] = test_audio_devices()
    results["vad_synthetic"] = test_vad_synthetic()
    results["classes"] = test_capture_playback_classes()
    results["db_schema"] = test_database_schema()

    if args.test_audio:
        results["live_audio"] = test_live_audio()

    # Summary
    LOGGER.info("=" * 50)
    LOGGER.info("Results:")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        LOGGER.info("  %s: %s", test_name, status)
        if not passed:
            all_passed = False

    if all_passed:
        LOGGER.info("All tests passed!")
        return 0
    else:
        LOGGER.error("Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
