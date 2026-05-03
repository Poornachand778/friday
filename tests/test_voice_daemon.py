"""
Tests for voice/daemon.py
=========================

Comprehensive tests for DaemonState, DaemonSession, VoiceDaemon,
and the main() entry point.  Covers state-machine transitions,
lazy component loading, wake-word / VAD handling, STT->orchestrator->TTS
pipeline, session lifecycle, cleanup, and argparse CLI.

Tests: 90+

IMPORTANT: sounddevice, soundfile, webrtcvad, openwakeword, faster_whisper,
and TTS (Coqui) are NOT installed in the test environment.
We patch sys.modules *before* importing the module under test so that
none of these native-extension imports blow up.
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import signal
import sys
import time
import uuid
from dataclasses import fields
from pathlib import Path
from unittest.mock import (
    AsyncMock,
    MagicMock,
    PropertyMock,
    call,
    patch,
)

import numpy as np
import pytest

# ── Pre-import module mocking ────────────────────────────────────────────
# All modules that would fail at import time must be stubbed out before
# we import *anything* from the voice package.

_MOCK_MODULES = {
    "sounddevice": MagicMock(),
    "soundfile": MagicMock(),
    "webrtcvad": MagicMock(),
    "openwakeword": MagicMock(),
    "openwakeword.model": MagicMock(),
    "faster_whisper": MagicMock(),
    "TTS": MagicMock(),
    "TTS.api": MagicMock(),
    "TTS.tts": MagicMock(),
    "TTS.tts.configs": MagicMock(),
    "TTS.tts.configs.xtts_config": MagicMock(),
    "TTS.tts.models": MagicMock(),
    "TTS.tts.models.xtts": MagicMock(),
}

# Patch them in once; they stay for the entire test-module lifetime.
for _mod_name, _mock in _MOCK_MODULES.items():
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _mock

# Now it is safe to import from voice.*
from voice.config import (
    AudioConfig,
    DaemonConfig,
    STTConfig,
    TTSConfig,
    VoiceConfig,
    WakeWordConfig,
)
from voice.daemon import (
    DaemonSession,
    DaemonState,
    VoiceDaemon,
    main,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_config(**daemon_overrides) -> VoiceConfig:
    """Return a default VoiceConfig with optional daemon overrides."""
    daemon_kw = {"idle_timeout_seconds": 300, "location": "writers_room"}
    daemon_kw.update(daemon_overrides)
    return VoiceConfig(daemon=DaemonConfig(**daemon_kw))


def _make_daemon(config: VoiceConfig | None = None) -> VoiceDaemon:
    """Create a VoiceDaemon with a mocked config (no lazy loading)."""
    cfg = config or _make_config()
    with patch("voice.daemon.get_voice_config", return_value=cfg):
        return VoiceDaemon(config=cfg)


def _make_chunk(data: np.ndarray | None = None) -> MagicMock:
    """Create a mock AudioChunk."""
    chunk = MagicMock()
    chunk.data = data if data is not None else np.zeros(512, dtype=np.int16)
    chunk.timestamp = time.time()
    chunk.sample_rate = 16000
    chunk.is_speech = False
    chunk.chunk_index = 0
    return chunk


def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── 1. DaemonState enum ─────────────────────────────────────────────────


class TestDaemonState:
    """Tests for the DaemonState enum."""

    def test_idle_value(self):
        assert DaemonState.IDLE == "idle"

    def test_listening_value(self):
        assert DaemonState.LISTENING == "listening"

    def test_wake_detected_value(self):
        assert DaemonState.WAKE_DETECTED == "wake_detected"

    def test_capturing_value(self):
        assert DaemonState.CAPTURING == "capturing"

    def test_processing_value(self):
        assert DaemonState.PROCESSING == "processing"

    def test_speaking_value(self):
        assert DaemonState.SPEAKING == "speaking"

    def test_error_value(self):
        assert DaemonState.ERROR == "error"

    def test_member_count(self):
        assert len(DaemonState) == 7

    def test_is_str_subclass(self):
        assert isinstance(DaemonState.IDLE, str)

    def test_all_values_unique(self):
        values = [s.value for s in DaemonState]
        assert len(values) == len(set(values))

    def test_lookup_by_value(self):
        assert DaemonState("idle") is DaemonState.IDLE
        assert DaemonState("error") is DaemonState.ERROR

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            DaemonState("nonexistent")


# ── 2. DaemonSession dataclass ──────────────────────────────────────────


class TestDaemonSession:
    """Tests for the DaemonSession dataclass."""

    def test_creation_with_required_fields(self):
        s = DaemonSession(
            session_id="abc",
            started_at=100.0,
            wake_word="friday",
            wake_confidence=0.95,
        )
        assert s.session_id == "abc"
        assert s.started_at == 100.0
        assert s.wake_word == "friday"
        assert s.wake_confidence == 0.95

    def test_default_turn_count(self):
        s = DaemonSession("id", 0.0, "w", 0.5)
        assert s.turn_count == 0

    def test_default_last_activity(self):
        s = DaemonSession("id", 0.0, "w", 0.5)
        assert s.last_activity == 0.0

    def test_custom_turn_count(self):
        s = DaemonSession("id", 0.0, "w", 0.5, turn_count=5)
        assert s.turn_count == 5

    def test_custom_last_activity(self):
        s = DaemonSession("id", 0.0, "w", 0.5, last_activity=42.0)
        assert s.last_activity == 42.0

    def test_mutation(self):
        s = DaemonSession("id", 0.0, "w", 0.5)
        s.turn_count = 3
        s.last_activity = 99.9
        assert s.turn_count == 3
        assert s.last_activity == 99.9

    def test_field_count(self):
        assert len(fields(DaemonSession)) == 6

    def test_field_names(self):
        names = {f.name for f in fields(DaemonSession)}
        expected = {
            "session_id",
            "started_at",
            "wake_word",
            "wake_confidence",
            "turn_count",
            "last_activity",
        }
        assert names == expected


# ── 3. VoiceDaemon.__init__ ─────────────────────────────────────────────


class TestVoiceDaemonInit:
    """Tests for VoiceDaemon constructor."""

    def test_default_config(self):
        d = _make_daemon()
        assert isinstance(d.config, VoiceConfig)

    def test_initial_state_idle(self):
        d = _make_daemon()
        assert d.state is DaemonState.IDLE

    def test_not_running(self):
        d = _make_daemon()
        assert d._running is False

    def test_lazy_components_none(self):
        d = _make_daemon()
        assert d._audio_capture is None
        assert d._audio_playback is None
        assert d._vad is None
        assert d._wakeword_service is None
        assert d._stt_service is None
        assert d._tts_service is None
        assert d._orchestrator_client is None

    def test_no_current_session(self):
        d = _make_daemon()
        assert d._current_session is None

    def test_session_timeout_from_config(self):
        cfg = _make_config(idle_timeout_seconds=120)
        d = _make_daemon(config=cfg)
        assert d._session_timeout == 120

    def test_custom_config_used(self):
        cfg = _make_config(location="kitchen")
        d = _make_daemon(config=cfg)
        assert d.config.daemon.location == "kitchen"

    def test_config_path_fallback(self):
        """If config=None, get_voice_config(config_path) is called."""
        mock_cfg = _make_config()
        with patch("voice.daemon.get_voice_config", return_value=mock_cfg) as gvc:
            d = VoiceDaemon(config=None, config_path=Path("/tmp/test.yaml"))
            gvc.assert_called_once_with(Path("/tmp/test.yaml"))
            assert d.config is mock_cfg

    def test_shutdown_event_created(self):
        d = _make_daemon()
        assert isinstance(d._shutdown_event, asyncio.Event)


# ── 4. State property / _set_state ──────────────────────────────────────


class TestStateTransitions:
    """Tests for the state property and _set_state method."""

    def test_state_property_returns_current(self):
        d = _make_daemon()
        assert d.state is DaemonState.IDLE

    def test_set_state_updates(self):
        d = _make_daemon()
        d._set_state(DaemonState.LISTENING)
        assert d.state is DaemonState.LISTENING

    def test_set_state_logs(self):
        d = _make_daemon()
        with patch("voice.daemon.LOGGER") as mock_logger:
            d._set_state(DaemonState.LISTENING)
            mock_logger.info.assert_called_once()
            args = mock_logger.info.call_args
            assert "idle" in str(args)
            assert "listening" in str(args)

    def test_idle_to_listening(self):
        d = _make_daemon()
        d._set_state(DaemonState.LISTENING)
        assert d.state is DaemonState.LISTENING

    def test_listening_to_wake_detected(self):
        d = _make_daemon()
        d._set_state(DaemonState.LISTENING)
        d._set_state(DaemonState.WAKE_DETECTED)
        assert d.state is DaemonState.WAKE_DETECTED

    def test_wake_detected_to_capturing(self):
        d = _make_daemon()
        d._set_state(DaemonState.WAKE_DETECTED)
        d._set_state(DaemonState.CAPTURING)
        assert d.state is DaemonState.CAPTURING

    def test_capturing_to_processing(self):
        d = _make_daemon()
        d._set_state(DaemonState.CAPTURING)
        d._set_state(DaemonState.PROCESSING)
        assert d.state is DaemonState.PROCESSING

    def test_processing_to_speaking(self):
        d = _make_daemon()
        d._set_state(DaemonState.PROCESSING)
        d._set_state(DaemonState.SPEAKING)
        assert d.state is DaemonState.SPEAKING

    def test_speaking_to_listening(self):
        d = _make_daemon()
        d._set_state(DaemonState.SPEAKING)
        d._set_state(DaemonState.LISTENING)
        assert d.state is DaemonState.LISTENING

    def test_any_to_error(self):
        d = _make_daemon()
        for st in DaemonState:
            d._set_state(st)
            d._set_state(DaemonState.ERROR)
            assert d.state is DaemonState.ERROR

    def test_any_to_idle(self):
        d = _make_daemon()
        for st in DaemonState:
            d._set_state(st)
            d._set_state(DaemonState.IDLE)
            assert d.state is DaemonState.IDLE


# ── 5. _load_components ─────────────────────────────────────────────────


class TestLoadComponents:
    """Tests for lazy component loading."""

    def test_audio_capture_loaded(self):
        d = _make_daemon()
        with (
            patch("voice.daemon.AudioCapture") as mc,
            patch("voice.daemon.AudioPlayback"),
            patch("voice.daemon.VoiceActivityDetector"),
            patch("voice.daemon.OrchestratorClient"),
        ):
            d._load_components()
            mc.assert_called_once_with(d.config.audio)
            assert d._audio_capture is not None

    def test_audio_playback_loaded(self):
        d = _make_daemon()
        with (
            patch("voice.daemon.AudioCapture"),
            patch("voice.daemon.AudioPlayback") as mp,
            patch("voice.daemon.VoiceActivityDetector"),
            patch("voice.daemon.OrchestratorClient"),
        ):
            d._load_components()
            mp.assert_called_once_with(d.config.audio)
            assert d._audio_playback is not None

    def test_vad_loaded(self):
        d = _make_daemon()
        with (
            patch("voice.daemon.AudioCapture"),
            patch("voice.daemon.AudioPlayback"),
            patch("voice.daemon.VoiceActivityDetector") as mv,
            patch("voice.daemon.OrchestratorClient"),
        ):
            d._load_components()
            mv.assert_called_once_with(d.config.audio)
            assert d._vad is not None

    def test_orchestrator_client_loaded(self):
        d = _make_daemon()
        with (
            patch("voice.daemon.AudioCapture"),
            patch("voice.daemon.AudioPlayback"),
            patch("voice.daemon.VoiceActivityDetector"),
            patch("voice.daemon.OrchestratorClient") as mo,
            patch.dict("os.environ", {}, clear=False),
        ):
            d._load_components()
            mo.assert_called_once()
            assert d._orchestrator_client is not None

    def test_orchestrator_url_from_env(self):
        d = _make_daemon()
        with (
            patch("voice.daemon.AudioCapture"),
            patch("voice.daemon.AudioPlayback"),
            patch("voice.daemon.VoiceActivityDetector"),
            patch("voice.daemon.OrchestratorClient") as mo,
            patch.dict(
                "os.environ",
                {"FRIDAY_ORCHESTRATOR_URL": "http://myhost:9000"},
            ),
        ):
            d._load_components()
            mo.assert_called_once_with(base_url="http://myhost:9000")

    def test_wakeword_import_success(self):
        d = _make_daemon()
        mock_ww = MagicMock()
        with (
            patch("voice.daemon.AudioCapture"),
            patch("voice.daemon.AudioPlayback"),
            patch("voice.daemon.VoiceActivityDetector"),
            patch("voice.daemon.OrchestratorClient"),
            patch.dict("sys.modules", {"voice.wakeword": mock_ww}),
        ):
            mock_ww.OpenWakeWordService = MagicMock()
            d._load_components()
            assert d._wakeword_service is not None

    def test_wakeword_import_failure_graceful(self):
        d = _make_daemon()
        with (
            patch("voice.daemon.AudioCapture"),
            patch("voice.daemon.AudioPlayback"),
            patch("voice.daemon.VoiceActivityDetector"),
            patch("voice.daemon.OrchestratorClient"),
            patch(
                "builtins.__import__",
                side_effect=_selective_import_error("voice.wakeword"),
            ),
        ):
            d._load_components()
            assert d._wakeword_service is None

    def test_stt_import_failure_graceful(self):
        d = _make_daemon()
        with (
            patch("voice.daemon.AudioCapture"),
            patch("voice.daemon.AudioPlayback"),
            patch("voice.daemon.VoiceActivityDetector"),
            patch("voice.daemon.OrchestratorClient"),
            patch(
                "builtins.__import__", side_effect=_selective_import_error("voice.stt")
            ),
        ):
            d._load_components()
            assert d._stt_service is None

    def test_tts_import_failure_graceful(self):
        d = _make_daemon()
        with (
            patch("voice.daemon.AudioCapture"),
            patch("voice.daemon.AudioPlayback"),
            patch("voice.daemon.VoiceActivityDetector"),
            patch("voice.daemon.OrchestratorClient"),
            patch(
                "builtins.__import__", side_effect=_selective_import_error("voice.tts")
            ),
        ):
            d._load_components()
            assert d._tts_service is None


def _selective_import_error(module_name):
    """Return a side_effect function that raises ImportError only for *module_name*."""
    real_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    def _importer(name, *args, **kwargs):
        if name == module_name:
            raise ImportError(f"Mocked missing: {name}")
        return real_import(name, *args, **kwargs)

    return _importer


# ── 6. _process_chunk dispatch ───────────────────────────────────────────


class TestProcessChunk:
    """Tests for _process_chunk state-based dispatch."""

    def test_dispatches_to_handle_listening(self):
        d = _make_daemon()
        d._state = DaemonState.LISTENING
        chunk = _make_chunk()
        with patch.object(d, "_handle_listening", new_callable=AsyncMock) as m:
            _run(d._process_chunk(chunk))
            m.assert_awaited_once_with(chunk)

    def test_dispatches_to_handle_wake_detected(self):
        d = _make_daemon()
        d._state = DaemonState.WAKE_DETECTED
        chunk = _make_chunk()
        with patch.object(d, "_handle_wake_detected", new_callable=AsyncMock) as m:
            _run(d._process_chunk(chunk))
            m.assert_awaited_once_with(chunk)

    def test_dispatches_to_handle_capturing(self):
        d = _make_daemon()
        d._state = DaemonState.CAPTURING
        chunk = _make_chunk()
        with patch.object(d, "_handle_capturing", new_callable=AsyncMock) as m:
            _run(d._process_chunk(chunk))
            m.assert_awaited_once_with(chunk)

    def test_no_dispatch_for_processing(self):
        d = _make_daemon()
        d._state = DaemonState.PROCESSING
        chunk = _make_chunk()
        # Should not raise, just silently skip
        _run(d._process_chunk(chunk))

    def test_no_dispatch_for_speaking(self):
        d = _make_daemon()
        d._state = DaemonState.SPEAKING
        chunk = _make_chunk()
        _run(d._process_chunk(chunk))

    def test_no_dispatch_for_idle(self):
        d = _make_daemon()
        d._state = DaemonState.IDLE
        chunk = _make_chunk()
        _run(d._process_chunk(chunk))

    def test_no_dispatch_for_error(self):
        d = _make_daemon()
        d._state = DaemonState.ERROR
        chunk = _make_chunk()
        _run(d._process_chunk(chunk))


# ── 7. _handle_listening ────────────────────────────────────────────────


class TestHandleListening:
    """Tests for wake-word and VAD-trigger logic in LISTENING state."""

    def test_no_wakeword_vad_not_triggered(self):
        d = _make_daemon()
        d._wakeword_service = None
        d._vad = MagicMock()
        d._vad.is_triggered = False
        chunk = _make_chunk()

        _run(d._handle_listening(chunk))

        d._vad.process_chunk.assert_called_once_with(chunk)
        assert d.state is DaemonState.IDLE  # unchanged

    def test_no_wakeword_vad_triggered_starts_session(self):
        d = _make_daemon()
        d._wakeword_service = None
        d._vad = MagicMock()
        d._vad.is_triggered = True
        chunk = _make_chunk()

        with patch.object(d, "_start_session") as ss:
            _run(d._handle_listening(chunk))
            ss.assert_called_once_with("voice_trigger", 1.0)
        assert d.state is DaemonState.CAPTURING

    def test_wakeword_no_detections(self):
        d = _make_daemon()
        d._wakeword_service = MagicMock()
        d._wakeword_service.process.return_value = []
        chunk = _make_chunk()

        _run(d._handle_listening(chunk))
        d._wakeword_service.process.assert_called_once_with(chunk)
        assert d.state is DaemonState.IDLE  # unchanged

    def test_wakeword_detection(self):
        d = _make_daemon()
        detection = MagicMock()
        detection.wake_word = "hey_friday"
        detection.confidence = 0.92
        d._wakeword_service = MagicMock()
        d._wakeword_service.process.return_value = [detection]
        d._audio_playback = MagicMock()
        d._tts_service = None  # no TTS for acknowledgment
        chunk = _make_chunk()

        with patch.object(d, "_start_session") as ss:
            _run(d._handle_listening(chunk))
            ss.assert_called_once_with("hey_friday", 0.92)
        assert d.state is DaemonState.WAKE_DETECTED

    def test_wakeword_detection_plays_acknowledgment(self):
        d = _make_daemon()
        detection = MagicMock()
        detection.wake_word = "hey_friday"
        detection.confidence = 0.9
        d._wakeword_service = MagicMock()
        d._wakeword_service.process.return_value = [detection]
        chunk = _make_chunk()

        with (
            patch.object(d, "_start_session"),
            patch.object(d, "_play_acknowledgment", new_callable=AsyncMock) as pa,
        ):
            _run(d._handle_listening(chunk))
            pa.assert_awaited_once()

    def test_wakeword_uses_first_detection_only(self):
        d = _make_daemon()
        det1 = MagicMock(wake_word="friday", confidence=0.9)
        det2 = MagicMock(wake_word="jarvis", confidence=0.8)
        d._wakeword_service = MagicMock()
        d._wakeword_service.process.return_value = [det1, det2]
        chunk = _make_chunk()

        with (
            patch.object(d, "_start_session") as ss,
            patch.object(d, "_play_acknowledgment", new_callable=AsyncMock),
        ):
            _run(d._handle_listening(chunk))
            ss.assert_called_once_with("friday", 0.9)


# ── 8. _handle_wake_detected ────────────────────────────────────────────


class TestHandleWakeDetected:
    """Tests for WAKE_DETECTED state - speech detection."""

    def test_vad_not_triggered(self):
        d = _make_daemon()
        d._state = DaemonState.WAKE_DETECTED
        d._vad = MagicMock()
        d._vad.is_triggered = False
        chunk = _make_chunk()

        _run(d._handle_wake_detected(chunk))

        d._vad.process_chunk.assert_called_once_with(chunk)
        assert d.state is DaemonState.WAKE_DETECTED  # unchanged

    def test_vad_triggered_starts_capturing(self):
        d = _make_daemon()
        d._state = DaemonState.WAKE_DETECTED
        d._vad = MagicMock()
        d._vad.is_triggered = True
        d._audio_capture = MagicMock()
        chunk = _make_chunk()

        _run(d._handle_wake_detected(chunk))

        assert d.state is DaemonState.CAPTURING
        d._audio_capture.start_recording.assert_called_once()

    def test_vad_triggered_processes_chunk_first(self):
        d = _make_daemon()
        d._state = DaemonState.WAKE_DETECTED
        d._vad = MagicMock()
        d._vad.is_triggered = True
        d._audio_capture = MagicMock()
        chunk = _make_chunk()

        _run(d._handle_wake_detected(chunk))
        # process_chunk must be called before checking is_triggered
        d._vad.process_chunk.assert_called_once_with(chunk)


# ── 9. _handle_capturing ────────────────────────────────────────────────


class TestHandleCapturing:
    """Tests for CAPTURING state - recording until silence."""

    def test_utterance_not_ended(self):
        d = _make_daemon()
        d._state = DaemonState.CAPTURING
        d._vad = MagicMock()
        d._vad.utterance_ended.return_value = False
        chunk = _make_chunk()

        _run(d._handle_capturing(chunk))

        d._vad.process_chunk.assert_called_once_with(chunk)
        assert d.state is DaemonState.CAPTURING  # unchanged

    def test_utterance_too_short_returns_to_listening(self):
        d = _make_daemon()
        d._state = DaemonState.CAPTURING
        d._vad = MagicMock()
        d._vad.utterance_ended.return_value = True
        d._audio_capture = MagicMock()
        d._audio_capture.stop_recording.return_value = (np.zeros(100), 0.3)
        chunk = _make_chunk()

        _run(d._handle_capturing(chunk))

        assert d.state is DaemonState.LISTENING

    def test_valid_utterance_triggers_processing(self):
        d = _make_daemon()
        d._state = DaemonState.CAPTURING
        d._vad = MagicMock()
        d._vad.utterance_ended.return_value = True
        audio = np.zeros(16000, dtype=np.int16)
        d._audio_capture = MagicMock()
        d._audio_capture.stop_recording.return_value = (audio, 1.0)
        chunk = _make_chunk()

        with patch.object(d, "_process_utterance", new_callable=AsyncMock) as pu:
            _run(d._handle_capturing(chunk))
            pu.assert_awaited_once()
            np.testing.assert_array_equal(pu.call_args[0][0], audio)

    def test_boundary_duration_exactly_half_second(self):
        """0.5 seconds is the threshold - exactly 0.5 should be ignored."""
        d = _make_daemon()
        d._state = DaemonState.CAPTURING
        d._vad = MagicMock()
        d._vad.utterance_ended.return_value = True
        d._audio_capture = MagicMock()
        d._audio_capture.stop_recording.return_value = (np.zeros(8000), 0.5)
        chunk = _make_chunk()

        _run(d._handle_capturing(chunk))
        # 0.5 is NOT > 0.5, so goes back to LISTENING
        assert d.state is DaemonState.LISTENING

    def test_boundary_duration_just_over_half(self):
        """0.51 seconds is above the threshold."""
        d = _make_daemon()
        d._state = DaemonState.CAPTURING
        d._vad = MagicMock()
        d._vad.utterance_ended.return_value = True
        audio = np.zeros(8160, dtype=np.int16)
        d._audio_capture = MagicMock()
        d._audio_capture.stop_recording.return_value = (audio, 0.51)
        chunk = _make_chunk()

        with patch.object(d, "_process_utterance", new_callable=AsyncMock) as pu:
            _run(d._handle_capturing(chunk))
            pu.assert_awaited_once()

    def test_set_state_to_processing_before_process_utterance(self):
        d = _make_daemon()
        d._state = DaemonState.CAPTURING
        d._vad = MagicMock()
        d._vad.utterance_ended.return_value = True
        audio = np.zeros(16000, dtype=np.int16)
        d._audio_capture = MagicMock()
        d._audio_capture.stop_recording.return_value = (audio, 1.0)
        chunk = _make_chunk()

        states_at_call = []

        async def capture_state(a):
            states_at_call.append(d.state)

        with patch.object(d, "_process_utterance", side_effect=capture_state):
            _run(d._handle_capturing(chunk))
        assert states_at_call[0] is DaemonState.PROCESSING


# ── 10. _process_utterance ──────────────────────────────────────────────


class TestProcessUtterance:
    """Tests for the full STT -> orchestrator -> TTS pipeline."""

    def _setup_daemon(self):
        d = _make_daemon()
        d._stt_service = MagicMock()
        stt_result = MagicMock()
        stt_result.text = "Hello Boss"
        stt_result.language = "en"
        d._stt_service.transcribe.return_value = stt_result

        d._orchestrator_client = AsyncMock()
        orch_response = MagicMock()
        orch_response.response = "Yes Boss, right away"
        orch_response.context = "general"
        d._orchestrator_client.chat.return_value = orch_response

        d._tts_service = MagicMock()
        tts_result = MagicMock()
        tts_result.audio = np.zeros(22050, dtype=np.float32)
        tts_result.sample_rate = 22050
        d._tts_service.synthesize.return_value = tts_result

        d._audio_playback = MagicMock()
        d._vad = MagicMock()

        d._start_session("friday", 0.9)
        return d

    def test_stt_called_with_audio(self):
        d = self._setup_daemon()
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        d._stt_service.transcribe.assert_called_once_with(audio)

    def test_orchestrator_called_with_transcript(self):
        d = self._setup_daemon()
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        d._orchestrator_client.chat.assert_awaited_once()
        kwargs = d._orchestrator_client.chat.call_args
        assert kwargs.kwargs["transcript"] == "Hello Boss"
        assert kwargs.kwargs["location"] == "writers_room"

    def test_tts_called_with_response(self):
        d = self._setup_daemon()
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        d._tts_service.synthesize.assert_called_once()
        call_kwargs = d._tts_service.synthesize.call_args
        assert call_kwargs.args[0] == "Yes Boss, right away"

    def test_playback_called(self):
        d = self._setup_daemon()
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        d._audio_playback.play.assert_called_once()

    def test_state_set_to_speaking_during_tts(self):
        d = self._setup_daemon()
        states_during = []

        original_synthesize = d._tts_service.synthesize

        def capture_state(*args, **kwargs):
            states_during.append(d.state)
            return original_synthesize(*args, **kwargs)

        d._tts_service.synthesize.side_effect = capture_state
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        assert DaemonState.SPEAKING in states_during

    def test_state_returns_to_listening_after(self):
        d = self._setup_daemon()
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        assert d.state is DaemonState.LISTENING

    def test_vad_reset_after(self):
        d = self._setup_daemon()
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        d._vad.reset.assert_called_once()

    def test_session_turn_count_incremented(self):
        d = self._setup_daemon()
        initial_count = d._current_session.turn_count
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        assert d._current_session.turn_count == initial_count + 1

    def test_session_last_activity_updated(self):
        d = self._setup_daemon()
        old_activity = d._current_session.last_activity
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        assert d._current_session.last_activity > old_activity

    def test_no_stt_service_fallback(self):
        d = self._setup_daemon()
        d._stt_service = None
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        d._orchestrator_client.chat.assert_awaited_once()
        kwargs = d._orchestrator_client.chat.call_args
        assert kwargs.kwargs["transcript"] == "[STT not available]"

    def test_no_tts_service_skips_speak(self):
        d = self._setup_daemon()
        d._tts_service = None
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        d._audio_playback.play.assert_not_called()

    def test_no_session_uses_none_session_id(self):
        d = self._setup_daemon()
        d._current_session = None
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        kwargs = d._orchestrator_client.chat.call_args
        assert kwargs.kwargs["session_id"] is None

    def test_error_in_pipeline_sets_error_state_then_listening(self):
        d = self._setup_daemon()
        d._stt_service.transcribe.side_effect = RuntimeError("STT crash")
        audio = np.zeros(16000, dtype=np.int16)

        states_seen = []
        original_set = d._set_state

        def track_set(state):
            states_seen.append(state)
            original_set(state)

        d._set_state = track_set
        _run(d._process_utterance(audio))
        assert DaemonState.ERROR in states_seen
        # Finally clause runs, so LISTENING is set after ERROR
        assert d.state is DaemonState.LISTENING

    def test_tts_language_mapping_en(self):
        d = self._setup_daemon()
        d._stt_service.transcribe.return_value.language = "en"
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        call_kwargs = d._tts_service.synthesize.call_args
        assert call_kwargs.kwargs["language"] == "en"

    def test_tts_language_mapping_te(self):
        d = self._setup_daemon()
        d._stt_service.transcribe.return_value.language = "te"
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        call_kwargs = d._tts_service.synthesize.call_args
        assert call_kwargs.kwargs["language"] == "te"

    def test_tts_language_mapping_unknown_falls_back_to_en(self):
        d = self._setup_daemon()
        d._stt_service.transcribe.return_value.language = "fr"
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        call_kwargs = d._tts_service.synthesize.call_args
        assert call_kwargs.kwargs["language"] == "en"

    def test_empty_response_skips_tts(self):
        d = self._setup_daemon()
        orch_response = MagicMock()
        orch_response.response = ""
        orch_response.context = "general"
        d._orchestrator_client.chat.return_value = orch_response
        audio = np.zeros(16000, dtype=np.int16)
        _run(d._process_utterance(audio))
        d._tts_service.synthesize.assert_not_called()


# ── 11. _play_acknowledgment ────────────────────────────────────────────


class TestPlayAcknowledgment:
    """Tests for acknowledgment sound after wake word."""

    def test_with_tts_service(self):
        d = _make_daemon()
        result = MagicMock()
        result.audio = np.zeros(1000)
        result.sample_rate = 22050
        d._tts_service = MagicMock()
        d._tts_service.synthesize.return_value = result
        d._audio_playback = MagicMock()

        _run(d._play_acknowledgment())

        d._tts_service.synthesize.assert_called_once_with("Yes?", language="en")
        d._audio_playback.play.assert_called_once_with(result.audio, result.sample_rate)

    def test_without_tts_service(self):
        d = _make_daemon()
        d._tts_service = None
        d._audio_playback = MagicMock()

        _run(d._play_acknowledgment())
        d._audio_playback.play.assert_not_called()

    def test_tts_exception_caught(self):
        d = _make_daemon()
        d._tts_service = MagicMock()
        d._tts_service.synthesize.side_effect = RuntimeError("TTS error")
        d._audio_playback = MagicMock()

        # Should not raise
        _run(d._play_acknowledgment())


# ── 12. _store_turn / _store_response ────────────────────────────────────


class TestStorage:
    """Tests for _store_turn and _store_response lazy storage."""

    def test_store_turn_with_session(self):
        d = _make_daemon()
        d._start_session("friday", 0.9)
        audio = np.zeros(16000, dtype=np.int16)

        mock_storage_cls = MagicMock()
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage

        with patch.dict(
            "sys.modules", {"voice.storage": MagicMock(AudioStorage=mock_storage_cls)}
        ):
            _run(d._store_turn(audio, "hello", "en"))
            mock_storage.store_user_audio.assert_called_once()

    def test_store_turn_without_session(self):
        d = _make_daemon()
        d._current_session = None
        audio = np.zeros(16000, dtype=np.int16)

        mock_storage_cls = MagicMock()
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage

        with patch.dict(
            "sys.modules", {"voice.storage": MagicMock(AudioStorage=mock_storage_cls)}
        ):
            _run(d._store_turn(audio, "hello", "en"))
            call_kwargs = mock_storage.store_user_audio.call_args
            assert call_kwargs.kwargs["session_id"] == "unknown"
            assert call_kwargs.kwargs["turn_number"] == 0

    def test_store_turn_import_error_caught(self):
        d = _make_daemon()
        d._current_session = None
        audio = np.zeros(16000, dtype=np.int16)

        # Should not raise even if voice.storage import fails
        with patch(
            "builtins.__import__", side_effect=_selective_import_error("voice.storage")
        ):
            _run(d._store_turn(audio, "hello", "en"))

    def test_store_response_with_session(self):
        d = _make_daemon()
        d._start_session("friday", 0.9)
        audio = np.zeros(22050, dtype=np.float32)

        mock_storage_cls = MagicMock()
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage

        with patch.dict(
            "sys.modules", {"voice.storage": MagicMock(AudioStorage=mock_storage_cls)}
        ):
            _run(d._store_response(audio, "Yes Boss"))
            mock_storage.store_response_audio.assert_called_once()

    def test_store_response_without_session(self):
        d = _make_daemon()
        d._current_session = None
        audio = np.zeros(22050, dtype=np.float32)

        mock_storage_cls = MagicMock()
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage

        with patch.dict(
            "sys.modules", {"voice.storage": MagicMock(AudioStorage=mock_storage_cls)}
        ):
            _run(d._store_response(audio, "Yes Boss"))
            call_kwargs = mock_storage.store_response_audio.call_args
            assert call_kwargs.kwargs["session_id"] == "unknown"
            assert call_kwargs.kwargs["turn_number"] == 0

    def test_store_response_exception_caught(self):
        d = _make_daemon()
        audio = np.zeros(22050, dtype=np.float32)

        with patch(
            "builtins.__import__", side_effect=_selective_import_error("voice.storage")
        ):
            _run(d._store_response(audio, "test"))


# ── 13. _start_session / _end_session ────────────────────────────────────


class TestSessionLifecycle:
    """Tests for session start and end."""

    def test_start_session_creates_session(self):
        d = _make_daemon()
        d._start_session("hey_friday", 0.88)
        assert d._current_session is not None

    def test_start_session_fields(self):
        d = _make_daemon()
        before = time.time()
        d._start_session("hey_friday", 0.88)
        after = time.time()

        s = d._current_session
        assert s.wake_word == "hey_friday"
        assert s.wake_confidence == 0.88
        assert s.turn_count == 0
        assert before <= s.started_at <= after
        assert before <= s.last_activity <= after

    def test_start_session_id_format(self):
        d = _make_daemon()
        d._start_session("friday", 0.9)
        # UUID[:8]
        assert len(d._current_session.session_id) == 8

    def test_end_session_clears(self):
        d = _make_daemon()
        d._start_session("friday", 0.9)
        d._end_session()
        assert d._current_session is None

    def test_end_session_sets_listening(self):
        d = _make_daemon()
        d._state = DaemonState.PROCESSING
        d._start_session("friday", 0.9)
        d._end_session()
        assert d.state is DaemonState.LISTENING

    def test_end_session_no_session_is_safe(self):
        d = _make_daemon()
        d._current_session = None
        d._end_session()  # Should not raise
        assert d.state is DaemonState.LISTENING

    def test_start_session_replaces_previous(self):
        d = _make_daemon()
        d._start_session("friday", 0.8)
        old_id = d._current_session.session_id
        d._start_session("jarvis", 0.95)
        assert d._current_session.session_id != old_id
        assert d._current_session.wake_word == "jarvis"

    def test_end_session_logs_duration(self):
        d = _make_daemon()
        d._start_session("friday", 0.9)
        d._current_session.turn_count = 5
        with patch("voice.daemon.LOGGER") as mock_logger:
            d._end_session()
            mock_logger.info.assert_called()


# ── 14. stop() ──────────────────────────────────────────────────────────


class TestStop:
    """Tests for daemon stop / cleanup."""

    def test_stop_sets_not_running(self):
        d = _make_daemon()
        d._running = True
        d._audio_capture = None
        d._audio_playback = None
        d._orchestrator_client = None
        d.stop()
        assert d._running is False

    def test_stop_sets_idle(self):
        d = _make_daemon()
        d._state = DaemonState.LISTENING
        d._audio_capture = None
        d._audio_playback = None
        d._orchestrator_client = None
        d.stop()
        assert d.state is DaemonState.IDLE

    def test_stop_calls_audio_capture_stop(self):
        d = _make_daemon()
        d._audio_capture = MagicMock()
        d._audio_playback = None
        d._orchestrator_client = None
        d.stop()
        d._audio_capture.stop.assert_called_once()

    def test_stop_calls_audio_playback_stop(self):
        d = _make_daemon()
        d._audio_capture = None
        d._audio_playback = MagicMock()
        d._orchestrator_client = None
        d.stop()
        d._audio_playback.stop.assert_called_once()

    def test_stop_closes_orchestrator_client(self):
        d = _make_daemon()
        d._audio_capture = None
        d._audio_playback = None
        mock_client = AsyncMock()

        # We need to mock get_event_loop to avoid issues
        mock_loop = MagicMock()
        mock_loop.run_until_complete = MagicMock()
        d._orchestrator_client = mock_client

        with patch("asyncio.get_event_loop", return_value=mock_loop):
            d.stop()
            mock_loop.run_until_complete.assert_called_once()

    def test_stop_clears_session(self):
        d = _make_daemon()
        d._audio_capture = None
        d._audio_playback = None
        d._orchestrator_client = None
        d._start_session("friday", 0.9)
        d.stop()
        assert d._current_session is None

    def test_stop_with_no_components_is_safe(self):
        d = _make_daemon()
        d.stop()  # Should not raise


# ── 15. _signal_handler ─────────────────────────────────────────────────


class TestSignalHandler:
    """Tests for shutdown signal handling."""

    def test_sets_running_false(self):
        d = _make_daemon()
        d._running = True
        d._signal_handler(signal.SIGINT, None)
        assert d._running is False

    def test_sets_shutdown_event(self):
        d = _make_daemon()
        d._signal_handler(signal.SIGTERM, None)
        assert d._shutdown_event.is_set()

    def test_sigint(self):
        d = _make_daemon()
        d._running = True
        d._signal_handler(signal.SIGINT, None)
        assert d._running is False
        assert d._shutdown_event.is_set()

    def test_sigterm(self):
        d = _make_daemon()
        d._running = True
        d._signal_handler(signal.SIGTERM, None)
        assert d._running is False
        assert d._shutdown_event.is_set()


# ── 16. _main_loop ──────────────────────────────────────────────────────


class TestMainLoop:
    """Tests for the async main loop."""

    def test_starts_audio_capture(self):
        d = _make_daemon()
        d._audio_capture = MagicMock()
        d._audio_capture.stream.return_value = iter([])
        d._running = True

        _run(d._main_loop())
        d._audio_capture.start.assert_called_once()

    def test_stops_on_running_false(self):
        d = _make_daemon()
        d._audio_capture = MagicMock()
        chunk = _make_chunk()
        d._audio_capture.stream.return_value = iter([chunk, chunk, chunk])
        d._running = False  # immediately stop

        _run(d._main_loop())

    def test_processes_chunks(self):
        d = _make_daemon()
        d._audio_capture = MagicMock()
        chunk = _make_chunk()

        call_count = 0

        async def process_and_stop(c):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                d._running = False

        d._audio_capture.stream.return_value = iter([chunk, chunk, chunk])
        d._running = True
        d._current_session = None

        with patch.object(d, "_process_chunk", side_effect=process_and_stop):
            _run(d._main_loop())
        assert call_count == 2

    def test_session_timeout_check(self):
        d = _make_daemon()
        d._session_timeout = 0  # immediate timeout
        d._audio_capture = MagicMock()
        chunk = _make_chunk()

        d._start_session("friday", 0.9)
        d._current_session.last_activity = time.time() - 1000  # long ago

        call_count = 0

        async def stop_after_one(c):
            nonlocal call_count
            call_count += 1
            d._running = False

        d._audio_capture.stream.return_value = iter([chunk])
        d._running = True

        with patch.object(d, "_process_chunk", side_effect=stop_after_one):
            with patch.object(d, "_end_session") as es:
                _run(d._main_loop())
                es.assert_called_once()

    def test_exception_sets_error_state(self):
        d = _make_daemon()
        d._audio_capture = MagicMock()
        # Error inside the try block (stream iterator raises)
        d._audio_capture.stream.return_value = iter([])
        d._audio_capture.stream.side_effect = RuntimeError("stream error")
        d._running = True

        with pytest.raises(RuntimeError):
            _run(d._main_loop())
        assert d.state is DaemonState.ERROR

    def test_exception_before_try_propagates(self):
        """Error in start() (before try) propagates without setting ERROR."""
        d = _make_daemon()
        d._audio_capture = MagicMock()
        d._audio_capture.start.side_effect = RuntimeError("mic error")
        d._running = True

        with pytest.raises(RuntimeError):
            _run(d._main_loop())
        # State remains IDLE since error was before the try/except
        assert d.state is DaemonState.IDLE


# ── 17. start() ─────────────────────────────────────────────────────────


class TestStart:
    """Tests for the blocking start() method."""

    def test_start_loads_components(self):
        d = _make_daemon()

        with (
            patch.object(d, "_load_components") as lc,
            patch("asyncio.run"),
            patch("signal.signal"),
            patch.object(d, "stop"),
        ):
            d.start()
            lc.assert_called_once()

    def test_start_sets_signal_handlers(self):
        d = _make_daemon()

        with (
            patch.object(d, "_load_components"),
            patch("asyncio.run"),
            patch("signal.signal") as sig,
            patch.object(d, "stop"),
        ):
            d.start()
            sig_calls = [c[0][0] for c in sig.call_args_list]
            assert signal.SIGINT in sig_calls
            assert signal.SIGTERM in sig_calls

    def test_start_sets_running(self):
        d = _make_daemon()
        running_during_loop = []

        def capture_running(*a, **kw):
            running_during_loop.append(d._running)

        with (
            patch.object(d, "_load_components"),
            patch("asyncio.run", side_effect=capture_running),
            patch("signal.signal"),
            patch.object(d, "stop"),
        ):
            d.start()
        assert running_during_loop[0] is True

    def test_start_sets_listening_state(self):
        d = _make_daemon()
        state_during_loop = []

        def capture_state(*a, **kw):
            state_during_loop.append(d.state)

        with (
            patch.object(d, "_load_components"),
            patch("asyncio.run", side_effect=capture_state),
            patch("signal.signal"),
            patch.object(d, "stop"),
        ):
            d.start()
        assert state_during_loop[0] is DaemonState.LISTENING

    def test_start_calls_stop_in_finally(self):
        d = _make_daemon()

        with (
            patch.object(d, "_load_components"),
            patch("asyncio.run"),
            patch("signal.signal"),
            patch.object(d, "stop") as mock_stop,
        ):
            d.start()
            mock_stop.assert_called_once()

    def test_start_handles_keyboard_interrupt(self):
        d = _make_daemon()

        with (
            patch.object(d, "_load_components"),
            patch("asyncio.run", side_effect=KeyboardInterrupt),
            patch("signal.signal"),
            patch.object(d, "stop") as mock_stop,
        ):
            d.start()  # Should not raise
            mock_stop.assert_called_once()


# ── 18. main() argparse ─────────────────────────────────────────────────


class TestMain:
    """Tests for the main() entry point with argparse."""

    def test_main_default_args(self):
        with (
            patch("voice.daemon.VoiceDaemon") as mock_cls,
            patch("logging.basicConfig"),
        ):
            mock_daemon = MagicMock()
            mock_cls.return_value = mock_daemon
            result = main([])
            mock_cls.assert_called_once_with(config_path=None)
            mock_daemon.start.assert_called_once()
            assert result == 0

    def test_main_with_config(self):
        with (
            patch("voice.daemon.VoiceDaemon") as mock_cls,
            patch("logging.basicConfig"),
        ):
            mock_daemon = MagicMock()
            mock_cls.return_value = mock_daemon
            result = main(["--config", "/tmp/test.yaml"])
            mock_cls.assert_called_once_with(config_path=Path("/tmp/test.yaml"))
            assert result == 0

    def test_main_with_log_level_debug(self):
        with (
            patch("voice.daemon.VoiceDaemon") as mock_cls,
            patch("logging.basicConfig") as mock_basic,
        ):
            mock_cls.return_value = MagicMock()
            main(["--log-level", "DEBUG"])
            mock_basic.assert_called_once()
            assert mock_basic.call_args.kwargs["level"] == logging.DEBUG

    def test_main_with_log_level_warning(self):
        with (
            patch("voice.daemon.VoiceDaemon") as mock_cls,
            patch("logging.basicConfig") as mock_basic,
        ):
            mock_cls.return_value = MagicMock()
            main(["--log-level", "WARNING"])
            assert mock_basic.call_args.kwargs["level"] == logging.WARNING

    def test_main_with_log_level_error(self):
        with (
            patch("voice.daemon.VoiceDaemon") as mock_cls,
            patch("logging.basicConfig") as mock_basic,
        ):
            mock_cls.return_value = MagicMock()
            main(["--log-level", "ERROR"])
            assert mock_basic.call_args.kwargs["level"] == logging.ERROR

    def test_main_with_log_level_info_default(self):
        with (
            patch("voice.daemon.VoiceDaemon") as mock_cls,
            patch("logging.basicConfig") as mock_basic,
        ):
            mock_cls.return_value = MagicMock()
            main([])
            assert mock_basic.call_args.kwargs["level"] == logging.INFO

    def test_main_returns_1_on_exception(self):
        with (
            patch("voice.daemon.VoiceDaemon") as mock_cls,
            patch("logging.basicConfig"),
        ):
            mock_cls.side_effect = RuntimeError("boom")
            result = main([])
            assert result == 1

    def test_main_invalid_log_level_exits(self):
        with pytest.raises(SystemExit):
            main(["--log-level", "INVALID"])

    def test_main_config_and_log_combined(self):
        with (
            patch("voice.daemon.VoiceDaemon") as mock_cls,
            patch("logging.basicConfig") as mock_basic,
        ):
            mock_cls.return_value = MagicMock()
            result = main(["--config", "/etc/friday.yaml", "--log-level", "DEBUG"])
            mock_cls.assert_called_once_with(config_path=Path("/etc/friday.yaml"))
            assert mock_basic.call_args.kwargs["level"] == logging.DEBUG
            assert result == 0
