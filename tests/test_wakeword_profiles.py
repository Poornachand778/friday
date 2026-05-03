"""
Comprehensive tests for:
  1. voice/wakeword/openwakeword_service.py
  2. voice/tts/voice_profiles.py
"""

from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Mock heavy / unavailable C-extension dependencies BEFORE any project import
# ---------------------------------------------------------------------------
sys.modules.setdefault("openwakeword", MagicMock())
sys.modules.setdefault("openwakeword.model", MagicMock())
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("webrtcvad", MagicMock())

# TTS / Coqui dependencies (imported by voice.tts.__init__ -> xtts_service)
sys.modules.setdefault("TTS", MagicMock())
sys.modules.setdefault("TTS.api", MagicMock())
sys.modules.setdefault("TTS.tts", MagicMock())
sys.modules.setdefault("TTS.tts.configs", MagicMock())
sys.modules.setdefault("TTS.tts.configs.xtts_config", MagicMock())
sys.modules.setdefault("TTS.tts.models", MagicMock())
sys.modules.setdefault("TTS.tts.models.xtts", MagicMock())

import numpy as np
import pytest
from pathlib import Path
from dataclasses import FrozenInstanceError

# Now safe to import project modules
from voice.config import WakeWordConfig
from voice.audio.capture import AudioChunk
from voice.wakeword.openwakeword_service import (
    SAMPLE_RATE,
    WakeWordDetection,
    OpenWakeWordService,
    detect_wake_word,
)
from voice.tts.voice_profiles import (
    VOICE_SAMPLES_DIR,
    VoiceProfileInfo,
    VoiceProfileManager,
)


# ============================================================================
#  Helpers / Fixtures
# ============================================================================


def _make_audio_chunk(
    data=None,
    timestamp=None,
    sample_rate=16000,
    chunk_index=0,
    is_speech=False,
):
    """Build a minimal AudioChunk for testing."""
    if data is None:
        data = np.zeros(1280, dtype=np.int16)
    return AudioChunk(
        data=data,
        timestamp=timestamp or time.time(),
        sample_rate=sample_rate,
        chunk_index=chunk_index,
        is_speech=is_speech,
    )


def _make_wakeword_config(models=None, framework="onnx"):
    """Build a WakeWordConfig for testing."""
    if models is None:
        models = [{"name": "alexa", "threshold": 0.5}]
    return WakeWordConfig(
        models=models,
        inference_framework=framework,
    )


@pytest.fixture
def oww_service():
    """Return an OpenWakeWordService with a mocked underlying model."""
    config = _make_wakeword_config()
    svc = OpenWakeWordService(config)
    # Inject a mock model so we avoid lazy-loading
    mock_model = MagicMock()
    mock_model.predict.return_value = {}
    svc._model = mock_model
    svc._thresholds = {"alexa": 0.5}
    return svc


@pytest.fixture
def profile_manager(tmp_path):
    """Return a VoiceProfileManager with VOICE_SAMPLES_DIR patched to tmp_path."""
    with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
        mgr = VoiceProfileManager(use_database=False)
    return mgr


@pytest.fixture
def profile_manager_with_samples(tmp_path):
    """Manager with fake Telugu + English sample files present."""
    (tmp_path / "friday_te_01.wav").write_bytes(b"\x00" * 100)
    (tmp_path / "friday_te_02.wav").write_bytes(b"\x00" * 100)
    (tmp_path / "friday_en_01.wav").write_bytes(b"\x00" * 100)
    with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
        mgr = VoiceProfileManager(use_database=False)
    return mgr


# ============================================================================
#  SECTION A -- WakeWordDetection dataclass
# ============================================================================


class TestWakeWordDetection:
    """Tests for the WakeWordDetection dataclass."""

    def test_creation_basic(self):
        d = WakeWordDetection(
            wake_word="alexa",
            confidence=0.85,
            timestamp=1000.0,
            audio_offset_samples=0,
        )
        assert d.wake_word == "alexa"
        assert d.confidence == 0.85
        assert d.timestamp == 1000.0
        assert d.audio_offset_samples == 0

    def test_is_confident_above_threshold(self):
        d = WakeWordDetection("w", 0.75, 0.0, 0)
        assert d.is_confident is True

    def test_is_confident_at_exact_threshold(self):
        d = WakeWordDetection("w", 0.5, 0.0, 0)
        assert d.is_confident is True

    def test_is_confident_below_threshold(self):
        d = WakeWordDetection("w", 0.49, 0.0, 0)
        assert d.is_confident is False

    def test_is_confident_zero(self):
        d = WakeWordDetection("w", 0.0, 0.0, 0)
        assert d.is_confident is False

    def test_is_confident_one(self):
        d = WakeWordDetection("w", 1.0, 0.0, 0)
        assert d.is_confident is True

    def test_is_confident_just_below_half(self):
        d = WakeWordDetection("w", 0.4999, 0.0, 0)
        assert d.is_confident is False

    def test_is_confident_just_above_half(self):
        d = WakeWordDetection("w", 0.5001, 0.0, 0)
        assert d.is_confident is True

    def test_different_wake_words(self):
        for name in ("alexa", "hey_jarvis", "hey_friday", "custom_model"):
            d = WakeWordDetection(name, 0.9, 0.0, 0)
            assert d.wake_word == name

    def test_audio_offset_samples_positive(self):
        d = WakeWordDetection("w", 0.9, 0.0, 12345)
        assert d.audio_offset_samples == 12345

    def test_timestamp_precision(self):
        ts = 1700000000.123456
        d = WakeWordDetection("w", 0.5, ts, 0)
        assert d.timestamp == ts


# ============================================================================
#  SECTION B -- OpenWakeWordService
# ============================================================================


class TestOpenWakeWordServiceInit:
    """Tests for service initialization."""

    def test_init_with_default_config(self):
        config = _make_wakeword_config()
        svc = OpenWakeWordService(config)
        assert svc.config is config
        assert svc._model is None
        assert svc._thresholds == {}
        assert svc._detection_history == []

    def test_init_uses_provided_config(self):
        config = _make_wakeword_config([{"name": "timer", "threshold": 0.7}])
        svc = OpenWakeWordService(config)
        assert svc.config.models[0]["name"] == "timer"

    def test_init_model_not_loaded_yet(self):
        svc = OpenWakeWordService(_make_wakeword_config())
        assert svc._model is None

    def test_builtin_models_list(self):
        expected = {
            "alexa",
            "hey_jarvis",
            "hey_mycroft",
            "timer",
            "weather",
            "hey_siri",
        }
        assert set(OpenWakeWordService.BUILTIN_MODELS) == expected


class TestEnsureModelLoaded:
    """Tests for lazy model loading."""

    def test_ensure_model_loaded_creates_model(self):
        config = _make_wakeword_config([{"name": "alexa", "threshold": 0.6}])
        svc = OpenWakeWordService(config)
        with patch("voice.wakeword.openwakeword_service.OWWModel") as MockModel:
            mock_inst = MagicMock()
            MockModel.return_value = mock_inst
            model = svc._ensure_model_loaded()
            MockModel.assert_called_once_with(
                wakeword_models=None,
                inference_framework="onnx",
            )
            assert model is mock_inst
            assert svc._thresholds["alexa"] == 0.6

    def test_ensure_model_loaded_with_custom_path(self, tmp_path):
        custom = str(tmp_path / "hey_friday.onnx")
        config = _make_wakeword_config(
            [{"name": "hey_friday", "path": custom, "threshold": 0.4}]
        )
        svc = OpenWakeWordService(config)
        with patch("voice.wakeword.openwakeword_service.OWWModel") as MockModel:
            MockModel.return_value = MagicMock()
            svc._ensure_model_loaded()
            MockModel.assert_called_once()
            call_args = MockModel.call_args
            assert call_args.kwargs["wakeword_models"] == [custom]
            assert svc._thresholds["hey_friday"] == 0.4

    def test_ensure_model_loaded_idempotent(self):
        config = _make_wakeword_config()
        svc = OpenWakeWordService(config)
        fake_model = MagicMock()
        svc._model = fake_model
        assert svc._ensure_model_loaded() is fake_model

    def test_ensure_model_loaded_unknown_model_skipped(self):
        config = _make_wakeword_config([{"name": "totally_unknown", "threshold": 0.3}])
        svc = OpenWakeWordService(config)
        with patch("voice.wakeword.openwakeword_service.OWWModel") as MockModel:
            MockModel.return_value = MagicMock()
            svc._ensure_model_loaded()
            # Unknown model should NOT appear in thresholds
            assert "totally_unknown" not in svc._thresholds

    def test_ensure_model_loaded_multiple_models(self):
        config = _make_wakeword_config(
            [
                {"name": "alexa", "threshold": 0.5},
                {"name": "timer", "threshold": 0.8},
            ]
        )
        svc = OpenWakeWordService(config)
        with patch("voice.wakeword.openwakeword_service.OWWModel") as MockModel:
            MockModel.return_value = MagicMock()
            svc._ensure_model_loaded()
            assert svc._thresholds == {"alexa": 0.5, "timer": 0.8}

    def test_ensure_model_loaded_default_threshold(self):
        config = _make_wakeword_config([{"name": "alexa"}])
        svc = OpenWakeWordService(config)
        with patch("voice.wakeword.openwakeword_service.OWWModel") as MockModel:
            MockModel.return_value = MagicMock()
            svc._ensure_model_loaded()
            assert svc._thresholds["alexa"] == 0.5


class TestAddModel:
    """Tests for add_model method."""

    def test_add_model_stores_threshold(self, oww_service):
        oww_service.add_model("timer", threshold=0.8)
        assert oww_service._thresholds["timer"] == 0.8

    def test_add_model_default_threshold(self, oww_service):
        oww_service.add_model("weather")
        assert oww_service._thresholds["weather"] == 0.5

    def test_add_model_overrides_existing(self, oww_service):
        oww_service.add_model("alexa", threshold=0.9)
        assert oww_service._thresholds["alexa"] == 0.9

    def test_add_model_with_path_logs_warning_when_loaded(self, oww_service):
        with patch("voice.wakeword.openwakeword_service.LOGGER") as mock_log:
            oww_service.add_model("custom", path="/some/model.onnx", threshold=0.3)
            mock_log.warning.assert_called()
        assert oww_service._thresholds["custom"] == 0.3

    def test_add_model_with_path_no_warning_when_not_loaded(self):
        svc = OpenWakeWordService(_make_wakeword_config())
        with patch("voice.wakeword.openwakeword_service.LOGGER") as mock_log:
            svc.add_model("custom", path="/model.onnx", threshold=0.6)
            mock_log.warning.assert_not_called()
        assert svc._thresholds["custom"] == 0.6


class TestProcess:
    """Tests for the process method."""

    def test_process_with_detection_above_threshold(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.8}
        chunk = _make_audio_chunk()
        detections = oww_service.process(chunk)
        assert len(detections) == 1
        assert detections[0].wake_word == "alexa"
        assert detections[0].confidence == 0.8

    def test_process_with_detection_below_threshold(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.3}
        chunk = _make_audio_chunk()
        detections = oww_service.process(chunk)
        assert len(detections) == 0

    def test_process_at_exact_threshold(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.5}
        detections = oww_service.process(_make_audio_chunk())
        assert len(detections) == 1

    def test_process_just_below_threshold(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.4999}
        detections = oww_service.process(_make_audio_chunk())
        assert len(detections) == 0

    def test_process_empty_predictions(self, oww_service):
        oww_service._model.predict.return_value = {}
        assert oww_service.process(_make_audio_chunk()) == []

    def test_process_multiple_models(self, oww_service):
        oww_service._thresholds = {"alexa": 0.5, "timer": 0.5}
        oww_service._model.predict.return_value = {"alexa": 0.9, "timer": 0.6}
        detections = oww_service.process(_make_audio_chunk())
        assert len(detections) == 2
        names = {d.wake_word for d in detections}
        assert names == {"alexa", "timer"}

    def test_process_some_above_some_below(self, oww_service):
        oww_service._thresholds = {"alexa": 0.5, "timer": 0.8}
        oww_service._model.predict.return_value = {"alexa": 0.9, "timer": 0.6}
        detections = oww_service.process(_make_audio_chunk())
        assert len(detections) == 1
        assert detections[0].wake_word == "alexa"

    def test_process_adds_to_history(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.8}
        oww_service.process(_make_audio_chunk())
        assert len(oww_service._detection_history) == 1

    def test_process_history_grows(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.8}
        for _ in range(5):
            oww_service.process(_make_audio_chunk())
        assert len(oww_service._detection_history) == 5

    def test_process_no_history_below_threshold(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.1}
        oww_service.process(_make_audio_chunk())
        assert len(oww_service._detection_history) == 0

    def test_process_float32_audio_conversion(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.7}
        chunk = _make_audio_chunk(data=np.ones(1280, dtype=np.float32) * 0.5)
        detections = oww_service.process(chunk)
        assert len(detections) == 1
        # Verify predict was called with int16 data
        call_args = oww_service._model.predict.call_args[0][0]
        assert call_args.dtype == np.int16

    def test_process_float64_audio_conversion(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.7}
        chunk = _make_audio_chunk(data=np.ones(1280, dtype=np.float64) * 0.5)
        detections = oww_service.process(chunk)
        assert len(detections) == 1
        call_args = oww_service._model.predict.call_args[0][0]
        assert call_args.dtype == np.int16

    def test_process_int16_no_conversion(self, oww_service):
        oww_service._model.predict.return_value = {}
        data = np.array([100, 200, 300], dtype=np.int16)
        chunk = _make_audio_chunk(data=data)
        oww_service.process(chunk)
        call_args = oww_service._model.predict.call_args[0][0]
        np.testing.assert_array_equal(call_args, data)

    def test_process_uint8_conversion(self, oww_service):
        oww_service._model.predict.return_value = {}
        data = np.array([10, 20, 30], dtype=np.uint8)
        chunk = _make_audio_chunk(data=data)
        oww_service.process(chunk)
        call_args = oww_service._model.predict.call_args[0][0]
        assert call_args.dtype == np.int16

    def test_process_timestamp_propagated(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.9}
        ts = 1700000000.0
        chunk = _make_audio_chunk(timestamp=ts)
        detections = oww_service.process(chunk)
        assert detections[0].timestamp == ts

    def test_process_audio_offset_calculated(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.9}
        data = np.zeros(1280, dtype=np.int16)
        chunk = _make_audio_chunk(data=data, chunk_index=3)
        detections = oww_service.process(chunk)
        expected_offset = 3 * 1280
        assert detections[0].audio_offset_samples == expected_offset

    def test_process_default_threshold_for_unknown_model(self, oww_service):
        # Model not in _thresholds should default to 0.5
        oww_service._model.predict.return_value = {"mystery_model": 0.6}
        detections = oww_service.process(_make_audio_chunk())
        assert len(detections) == 1
        assert detections[0].wake_word == "mystery_model"


class TestProcessBatch:
    """Tests for process_batch method."""

    def test_process_batch_single_chunk(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.8}
        audio = np.zeros(1280, dtype=np.int16)
        detections = oww_service.process_batch(audio, chunk_size=1280)
        assert len(detections) == 1

    def test_process_batch_multiple_chunks(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.8}
        audio = np.zeros(3840, dtype=np.int16)  # 3 chunks of 1280
        detections = oww_service.process_batch(audio, chunk_size=1280)
        assert len(detections) == 3

    def test_process_batch_pads_last_chunk(self, oww_service):
        call_count = 0

        def track_predict(data):
            nonlocal call_count
            call_count += 1
            assert len(data) == 1280, f"Chunk {call_count} not padded: {len(data)}"
            return {}

        oww_service._model.predict.side_effect = track_predict
        # 1280 + 500 = 1780 samples -> 2 chunks, last padded
        audio = np.zeros(1780, dtype=np.int16)
        oww_service.process_batch(audio, chunk_size=1280)
        assert call_count == 2

    def test_process_batch_empty_audio(self, oww_service):
        audio = np.array([], dtype=np.int16)
        detections = oww_service.process_batch(audio)
        assert detections == []

    def test_process_batch_exact_multiple(self, oww_service):
        oww_service._model.predict.return_value = {}
        audio = np.zeros(2560, dtype=np.int16)  # exactly 2 chunks
        oww_service.process_batch(audio, chunk_size=1280)
        assert oww_service._model.predict.call_count == 2

    def test_process_batch_collects_all_detections(self, oww_service):
        # Return detection only on second chunk
        call_num = [0]

        def pred(data):
            call_num[0] += 1
            if call_num[0] == 2:
                return {"alexa": 0.9}
            return {"alexa": 0.1}

        oww_service._model.predict.side_effect = pred
        audio = np.zeros(2560, dtype=np.int16)
        detections = oww_service.process_batch(audio, chunk_size=1280)
        assert len(detections) == 1
        assert detections[0].wake_word == "alexa"

    def test_process_batch_default_chunk_size(self, oww_service):
        oww_service._model.predict.return_value = {}
        audio = np.zeros(1280, dtype=np.int16)
        oww_service.process_batch(audio)
        assert oww_service._model.predict.call_count == 1

    def test_process_batch_custom_chunk_size(self, oww_service):
        oww_service._model.predict.return_value = {}
        audio = np.zeros(640, dtype=np.int16)
        oww_service.process_batch(audio, chunk_size=320)
        assert oww_service._model.predict.call_count == 2


class TestMonitorStream:
    """Tests for monitor_stream method."""

    def test_monitor_stream_returns_first_detection(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.8}
        chunks = iter([_make_audio_chunk(), _make_audio_chunk()])
        result = oww_service.monitor_stream(chunks)
        assert result is not None
        assert result.wake_word == "alexa"

    def test_monitor_stream_returns_none_on_empty_iterator(self, oww_service):
        result = oww_service.monitor_stream(iter([]))
        assert result is None

    def test_monitor_stream_returns_none_on_timeout(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.1}

        def slow_chunks():
            while True:
                time.sleep(0.01)
                yield _make_audio_chunk()

        result = oww_service.monitor_stream(slow_chunks(), timeout_seconds=0.05)
        assert result is None

    def test_monitor_stream_stops_after_first_detection(self, oww_service):
        call_count = [0]

        def pred(data):
            call_count[0] += 1
            return {"alexa": 0.9}

        oww_service._model.predict.side_effect = pred
        chunks = [_make_audio_chunk() for _ in range(10)]
        oww_service.monitor_stream(iter(chunks))
        assert call_count[0] == 1

    def test_monitor_stream_skips_until_detection(self, oww_service):
        call_count = [0]

        def pred(data):
            call_count[0] += 1
            if call_count[0] == 3:
                return {"alexa": 0.9}
            return {"alexa": 0.1}

        oww_service._model.predict.side_effect = pred
        chunks = [_make_audio_chunk() for _ in range(5)]
        result = oww_service.monitor_stream(iter(chunks))
        assert result is not None
        assert call_count[0] == 3


class TestDetectionHistory:
    """Tests for detection history management."""

    def test_get_detection_history_empty(self, oww_service):
        assert oww_service.get_detection_history() == []

    def test_get_detection_history_with_entries(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.9}
        for _ in range(3):
            oww_service.process(_make_audio_chunk())
        history = oww_service.get_detection_history()
        assert len(history) == 3

    def test_get_detection_history_with_limit(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.9}
        for _ in range(10):
            oww_service.process(_make_audio_chunk())
        assert len(oww_service.get_detection_history(limit=5)) == 5

    def test_get_detection_history_limit_larger_than_list(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.9}
        oww_service.process(_make_audio_chunk())
        assert len(oww_service.get_detection_history(limit=100)) == 1

    def test_get_detection_history_default_limit(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.9}
        for _ in range(150):
            oww_service.process(_make_audio_chunk())
        assert len(oww_service.get_detection_history()) == 100

    def test_clear_history(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.9}
        oww_service.process(_make_audio_chunk())
        oww_service.clear_history()
        assert oww_service._detection_history == []

    def test_clear_history_when_already_empty(self, oww_service):
        oww_service.clear_history()
        assert oww_service._detection_history == []


class TestReset:
    """Tests for reset method."""

    def test_reset_calls_model_reset(self, oww_service):
        oww_service.reset()
        oww_service._model.reset.assert_called_once()

    def test_reset_clears_history(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.9}
        oww_service.process(_make_audio_chunk())
        oww_service.reset()
        assert oww_service._detection_history == []

    def test_reset_when_model_not_loaded(self):
        svc = OpenWakeWordService(_make_wakeword_config())
        svc.reset()  # should not raise
        assert svc._detection_history == []


class TestActiveModels:
    """Tests for active_models property."""

    def test_active_models_empty(self):
        svc = OpenWakeWordService(_make_wakeword_config())
        assert svc.active_models == []

    def test_active_models_after_add(self, oww_service):
        oww_service.add_model("timer", threshold=0.5)
        assert "timer" in oww_service.active_models

    def test_active_models_returns_list(self, oww_service):
        result = oww_service.active_models
        assert isinstance(result, list)

    def test_active_models_reflects_thresholds(self, oww_service):
        oww_service._thresholds = {"a": 0.5, "b": 0.6, "c": 0.7}
        assert set(oww_service.active_models) == {"a", "b", "c"}


class TestThresholds:
    """Tests for get_threshold / set_threshold."""

    def test_get_threshold_existing(self, oww_service):
        assert oww_service.get_threshold("alexa") == 0.5

    def test_get_threshold_default(self, oww_service):
        assert oww_service.get_threshold("nonexistent") == 0.5

    def test_set_threshold_valid(self, oww_service):
        oww_service.set_threshold("alexa", 0.8)
        assert oww_service._thresholds["alexa"] == 0.8

    def test_set_threshold_zero(self, oww_service):
        oww_service.set_threshold("alexa", 0.0)
        assert oww_service._thresholds["alexa"] == 0.0

    def test_set_threshold_one(self, oww_service):
        oww_service.set_threshold("alexa", 1.0)
        assert oww_service._thresholds["alexa"] == 1.0

    def test_set_threshold_negative_raises(self, oww_service):
        with pytest.raises(ValueError, match="between 0 and 1"):
            oww_service.set_threshold("alexa", -0.1)

    def test_set_threshold_above_one_raises(self, oww_service):
        with pytest.raises(ValueError, match="between 0 and 1"):
            oww_service.set_threshold("alexa", 1.1)

    def test_set_threshold_large_value_raises(self, oww_service):
        with pytest.raises(ValueError):
            oww_service.set_threshold("alexa", 100.0)

    def test_set_threshold_creates_new_entry(self, oww_service):
        oww_service.set_threshold("new_model", 0.7)
        assert oww_service._thresholds["new_model"] == 0.7

    def test_set_threshold_preserves_others(self, oww_service):
        oww_service._thresholds = {"a": 0.3, "b": 0.4}
        oww_service.set_threshold("a", 0.9)
        assert oww_service._thresholds["b"] == 0.4


class TestDetectWakeWordConvenience:
    """Tests for the module-level detect_wake_word function."""

    @patch("voice.wakeword.openwakeword_service.OWWModel")
    def test_detect_wake_word_returns_detection(self, MockModel):
        mock_inst = MagicMock()
        mock_inst.predict.return_value = {"alexa": 0.9}
        MockModel.return_value = mock_inst
        audio = np.zeros(1280, dtype=np.int16)
        result = detect_wake_word(audio, models=["alexa"], threshold=0.5)
        assert result is not None
        assert result.wake_word == "alexa"

    @patch("voice.wakeword.openwakeword_service.OWWModel")
    def test_detect_wake_word_returns_none(self, MockModel):
        mock_inst = MagicMock()
        mock_inst.predict.return_value = {"alexa": 0.1}
        MockModel.return_value = mock_inst
        audio = np.zeros(1280, dtype=np.int16)
        result = detect_wake_word(audio, models=["alexa"], threshold=0.5)
        assert result is None

    @patch("voice.wakeword.openwakeword_service.OWWModel")
    def test_detect_wake_word_default_models(self, MockModel):
        mock_inst = MagicMock()
        mock_inst.predict.return_value = {"alexa": 0.9}
        MockModel.return_value = mock_inst
        audio = np.zeros(1280, dtype=np.int16)
        result = detect_wake_word(audio)
        assert result is not None

    @patch("voice.wakeword.openwakeword_service.OWWModel")
    def test_detect_wake_word_custom_threshold(self, MockModel):
        mock_inst = MagicMock()
        mock_inst.predict.return_value = {"alexa": 0.7}
        MockModel.return_value = mock_inst
        audio = np.zeros(1280, dtype=np.int16)
        # High threshold should still match 0.7 < 0.8
        result = detect_wake_word(audio, threshold=0.8)
        assert result is None


class TestSampleRate:
    """Tests for module-level constants."""

    def test_sample_rate_value(self):
        assert SAMPLE_RATE == 16000

    def test_sample_rate_is_int(self):
        assert isinstance(SAMPLE_RATE, int)


# ============================================================================
#  SECTION C -- VoiceProfileInfo dataclass
# ============================================================================


class TestVoiceProfileInfo:
    """Tests for the VoiceProfileInfo dataclass."""

    def test_creation_basic(self):
        p = VoiceProfileInfo(
            name="test",
            description="Test profile",
            language="en",
            reference_audio_paths=["/tmp/audio.wav"],
            tts_engine="xtts_v2",
            is_active=True,
            is_default=False,
        )
        assert p.name == "test"
        assert p.language == "en"
        assert p.tts_engine == "xtts_v2"
        assert p.is_active is True
        assert p.is_default is False

    def test_has_audio_with_existing_files(self, tmp_path):
        f = tmp_path / "audio.wav"
        f.write_bytes(b"\x00")
        p = VoiceProfileInfo("test", "", "en", [str(f)], "xtts_v2", True, False)
        assert p.has_audio is True

    def test_has_audio_with_nonexistent_file(self):
        p = VoiceProfileInfo(
            "test", "", "en", ["/no/such/file.wav"], "xtts_v2", True, False
        )
        assert p.has_audio is False

    def test_has_audio_empty_list(self):
        p = VoiceProfileInfo("test", "", "en", [], "xtts_v2", True, False)
        assert p.has_audio is False

    def test_has_audio_mixed_existing_and_not(self, tmp_path):
        f = tmp_path / "exists.wav"
        f.write_bytes(b"\x00")
        p = VoiceProfileInfo(
            "test",
            "",
            "en",
            [str(f), "/does/not/exist.wav"],
            "xtts_v2",
            True,
            False,
        )
        # all() requires all to exist
        assert p.has_audio is False

    def test_has_audio_multiple_existing(self, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"audio_{i}.wav"
            f.write_bytes(b"\x00")
            files.append(str(f))
        p = VoiceProfileInfo("test", "", "en", files, "xtts_v2", True, False)
        assert p.has_audio is True

    def test_description_field(self):
        p = VoiceProfileInfo(
            "test", "My custom voice", "te", [], "xtts_v2", True, False
        )
        assert p.description == "My custom voice"

    def test_language_codes(self):
        for lang in ("en", "te", "hi", "fr", "de"):
            p = VoiceProfileInfo("t", "", lang, [], "xtts_v2", True, False)
            assert p.language == lang


# ============================================================================
#  SECTION D -- VoiceProfileManager
# ============================================================================


class TestVoiceProfileManagerInit:
    """Tests for manager initialization."""

    def test_init_without_database(self, profile_manager):
        assert profile_manager._use_database is False
        assert isinstance(profile_manager._profiles, dict)

    def test_init_no_samples_dir_gives_empty_profiles(self, profile_manager):
        # tmp_path is empty -> no glob matches -> no default profiles
        assert profile_manager._profiles == {}

    def test_init_with_samples_creates_defaults(self, profile_manager_with_samples):
        mgr = profile_manager_with_samples
        assert "friday_telugu" in mgr._profiles
        assert "friday_english" in mgr._profiles

    def test_init_telugu_profile_is_default(self, profile_manager_with_samples):
        mgr = profile_manager_with_samples
        assert mgr._profiles["friday_telugu"].is_default is True
        assert mgr._profiles["friday_english"].is_default is False

    def test_init_telugu_profile_language(self, profile_manager_with_samples):
        assert profile_manager_with_samples._profiles["friday_telugu"].language == "te"

    def test_init_english_profile_language(self, profile_manager_with_samples):
        assert profile_manager_with_samples._profiles["friday_english"].language == "en"

    def test_init_profiles_have_sample_paths(self, profile_manager_with_samples):
        te = profile_manager_with_samples._profiles["friday_telugu"]
        assert len(te.reference_audio_paths) == 2  # friday_te_01 + friday_te_02


class TestCreateProfile:
    """Tests for create_profile."""

    def test_create_profile_valid(self, profile_manager, tmp_path):
        f = tmp_path / "voice.wav"
        f.write_bytes(b"\x00")
        p = profile_manager.create_profile("custom", [str(f)], "en", "My voice")
        assert p.name == "custom"
        assert p.language == "en"
        assert p.description == "My voice"
        assert p.is_active is True

    def test_create_profile_no_valid_paths_raises(self, profile_manager):
        with pytest.raises(ValueError, match="No valid reference audio"):
            profile_manager.create_profile("bad", ["/no/file.wav"])

    def test_create_profile_mixed_paths(self, profile_manager, tmp_path):
        good = tmp_path / "good.wav"
        good.write_bytes(b"\x00")
        p = profile_manager.create_profile("mix", [str(good), "/nonexistent.wav"], "en")
        assert len(p.reference_audio_paths) == 1
        assert str(good) in p.reference_audio_paths[0]

    def test_create_profile_default_description(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        p = profile_manager.create_profile("myvoice", [str(f)])
        assert p.description == "Voice profile: myvoice"

    def test_create_profile_make_default(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        p = profile_manager.create_profile("first", [str(f)], make_default=True)
        assert p.is_default is True

    def test_create_profile_make_default_unsets_previous(
        self, profile_manager, tmp_path
    ):
        f1 = tmp_path / "a.wav"
        f1.write_bytes(b"\x00")
        f2 = tmp_path / "b.wav"
        f2.write_bytes(b"\x00")
        p1 = profile_manager.create_profile("first", [str(f1)], make_default=True)
        p2 = profile_manager.create_profile("second", [str(f2)], make_default=True)
        # p1's is_default should be unset
        assert profile_manager._profiles["first"].is_default is False
        assert p2.is_default is True

    def test_create_profile_stored_in_manager(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        profile_manager.create_profile("stored", [str(f)])
        assert "stored" in profile_manager._profiles

    def test_create_profile_default_language(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        p = profile_manager.create_profile("test", [str(f)])
        assert p.language == "te"

    def test_create_profile_tts_engine(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        p = profile_manager.create_profile("test", [str(f)])
        assert p.tts_engine == "xtts_v2"

    def test_create_profile_multiple_valid_paths(self, profile_manager, tmp_path):
        files = []
        for i in range(4):
            f = tmp_path / f"s{i}.wav"
            f.write_bytes(b"\x00")
            files.append(str(f))
        p = profile_manager.create_profile("multi", files)
        assert len(p.reference_audio_paths) == 4

    def test_create_profile_overwrites_existing(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        profile_manager.create_profile("same", [str(f)], description="first")
        profile_manager.create_profile("same", [str(f)], description="second")
        assert profile_manager._profiles["same"].description == "second"


class TestGetProfile:
    """Tests for get_profile."""

    def test_get_profile_existing(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        profile_manager.create_profile("x", [str(f)])
        assert profile_manager.get_profile("x") is not None

    def test_get_profile_nonexistent(self, profile_manager):
        assert profile_manager.get_profile("nope") is None

    def test_get_profile_returns_correct_instance(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        created = profile_manager.create_profile("exact", [str(f)])
        fetched = profile_manager.get_profile("exact")
        assert fetched is created


class TestGetDefaultProfile:
    """Tests for get_default_profile."""

    def test_get_default_profile_none_when_empty(self, profile_manager):
        assert profile_manager.get_default_profile() is None

    def test_get_default_profile_returns_default(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        profile_manager.create_profile("d", [str(f)], make_default=True)
        d = profile_manager.get_default_profile()
        assert d is not None
        assert d.name == "d"

    def test_get_default_profile_falls_back_to_first(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        profile_manager.create_profile("nond", [str(f)], make_default=False)
        d = profile_manager.get_default_profile()
        assert d is not None
        assert d.name == "nond"

    def test_get_default_profile_with_samples(self, profile_manager_with_samples):
        d = profile_manager_with_samples.get_default_profile()
        assert d is not None
        assert d.name == "friday_telugu"

    def test_get_default_profile_after_change(self, profile_manager, tmp_path):
        f1 = tmp_path / "a.wav"
        f1.write_bytes(b"\x00")
        f2 = tmp_path / "b.wav"
        f2.write_bytes(b"\x00")
        profile_manager.create_profile("first", [str(f1)], make_default=True)
        profile_manager.create_profile("second", [str(f2)], make_default=True)
        d = profile_manager.get_default_profile()
        assert d.name == "second"


class TestListProfiles:
    """Tests for list_profiles."""

    def test_list_profiles_empty(self, profile_manager):
        assert profile_manager.list_profiles() == []

    def test_list_profiles_returns_all(self, profile_manager, tmp_path):
        for i in range(3):
            f = tmp_path / f"f{i}.wav"
            f.write_bytes(b"\x00")
            profile_manager.create_profile(f"p{i}", [str(f)])
        assert len(profile_manager.list_profiles()) == 3

    def test_list_profiles_returns_list(self, profile_manager):
        assert isinstance(profile_manager.list_profiles(), list)

    def test_list_profiles_with_defaults(self, profile_manager_with_samples):
        profiles = profile_manager_with_samples.list_profiles()
        assert len(profiles) == 2
        names = {p.name for p in profiles}
        assert names == {"friday_telugu", "friday_english"}


class TestDeleteProfile:
    """Tests for delete_profile."""

    def test_delete_profile_existing(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        profile_manager.create_profile("del_me", [str(f)])
        assert profile_manager.delete_profile("del_me") is True
        assert "del_me" not in profile_manager._profiles

    def test_delete_profile_nonexistent(self, profile_manager):
        assert profile_manager.delete_profile("nope") is False

    def test_delete_profile_reduces_count(self, profile_manager, tmp_path):
        for i in range(3):
            f = tmp_path / f"f{i}.wav"
            f.write_bytes(b"\x00")
            profile_manager.create_profile(f"p{i}", [str(f)])
        profile_manager.delete_profile("p1")
        assert len(profile_manager.list_profiles()) == 2

    def test_delete_profile_idempotent(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        profile_manager.create_profile("once", [str(f)])
        assert profile_manager.delete_profile("once") is True
        assert profile_manager.delete_profile("once") is False


class TestAddReferenceAudio:
    """Tests for add_reference_audio."""

    def test_add_reference_audio_to_existing(self, profile_manager, tmp_path):
        f1 = tmp_path / "a.wav"
        f1.write_bytes(b"\x00")
        f2 = tmp_path / "b.wav"
        f2.write_bytes(b"\x00")
        profile_manager.create_profile("test", [str(f1)])
        result = profile_manager.add_reference_audio("test", str(f2))
        assert result is True
        p = profile_manager.get_profile("test")
        assert len(p.reference_audio_paths) == 2

    def test_add_reference_audio_nonexistent_profile(self, profile_manager, tmp_path):
        f = tmp_path / "a.wav"
        f.write_bytes(b"\x00")
        result = profile_manager.add_reference_audio("no_such", str(f))
        assert result is False

    def test_add_reference_audio_nonexistent_file(self, profile_manager, tmp_path):
        f = tmp_path / "a.wav"
        f.write_bytes(b"\x00")
        profile_manager.create_profile("test", [str(f)])
        result = profile_manager.add_reference_audio("test", "/no/file.wav")
        assert result is False

    def test_add_reference_audio_preserves_existing(self, profile_manager, tmp_path):
        f1 = tmp_path / "a.wav"
        f1.write_bytes(b"\x00")
        f2 = tmp_path / "b.wav"
        f2.write_bytes(b"\x00")
        profile_manager.create_profile("test", [str(f1)])
        profile_manager.add_reference_audio("test", str(f2))
        p = profile_manager.get_profile("test")
        assert str(f1) in p.reference_audio_paths[0]
        assert str(f2) in p.reference_audio_paths[1]


class TestDatabaseOperations:
    """Tests for database-related methods (mocked)."""

    def test_create_profile_calls_save_when_db_enabled(self, tmp_path):
        with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
            mgr = VoiceProfileManager(use_database=True)
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        with patch.object(mgr, "_save_to_database") as mock_save:
            mgr.create_profile("db_test", [str(f)])
            mock_save.assert_called_once()

    def test_create_profile_no_save_when_db_disabled(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        with patch.object(profile_manager, "_save_to_database") as mock_save:
            profile_manager.create_profile("nodb", [str(f)])
            mock_save.assert_not_called()

    def test_delete_profile_calls_delete_from_db(self, tmp_path):
        with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
            mgr = VoiceProfileManager(use_database=True)
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        with patch.object(mgr, "_save_to_database"):
            mgr.create_profile("del_db", [str(f)])
        with patch.object(mgr, "_delete_from_database") as mock_del:
            mgr.delete_profile("del_db")
            mock_del.assert_called_once_with("del_db")

    def test_delete_profile_no_db_call_when_disabled(self, profile_manager, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"\x00")
        profile_manager.create_profile("nodb", [str(f)])
        with patch.object(profile_manager, "_delete_from_database") as mock_del:
            profile_manager.delete_profile("nodb")
            mock_del.assert_not_called()

    def test_add_reference_audio_saves_when_db_enabled(self, tmp_path):
        with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
            mgr = VoiceProfileManager(use_database=True)
        f1 = tmp_path / "a.wav"
        f1.write_bytes(b"\x00")
        f2 = tmp_path / "b.wav"
        f2.write_bytes(b"\x00")
        with patch.object(mgr, "_save_to_database") as mock_save:
            mgr.create_profile("ref", [str(f1)])
            mock_save.reset_mock()
            mgr.add_reference_audio("ref", str(f2))
            mock_save.assert_called_once()

    def test_save_to_database_handles_exception(self, tmp_path):
        with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
            mgr = VoiceProfileManager(use_database=True)
        profile = VoiceProfileInfo("t", "d", "en", [], "xtts_v2", True, False)
        # _save_to_database will try to import sqlalchemy, which may not exist,
        # but the method catches all exceptions and logs them.
        mgr._save_to_database(profile)  # should not raise

    def test_delete_from_database_handles_exception(self, tmp_path):
        with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
            mgr = VoiceProfileManager(use_database=True)
        mgr._delete_from_database("nonexistent")  # should not raise

    def test_load_from_database_returns_zero_on_failure(self, profile_manager):
        result = profile_manager.load_from_database()
        assert result == 0

    @patch("voice.tts.voice_profiles.os.environ", {"DATABASE_URL": "sqlite:///test.db"})
    def test_load_from_database_uses_env_url(self, tmp_path):
        with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
            mgr = VoiceProfileManager(use_database=False)
        # Will fail at import but return 0 gracefully
        result = mgr.load_from_database()
        assert result == 0

    def test_save_to_database_with_mocked_sqlalchemy(self, tmp_path):
        """Full mock of the database save path."""
        with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
            mgr = VoiceProfileManager(use_database=True)

        mock_session_cls = MagicMock()
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        mock_engine = MagicMock()
        mock_voice_profile_cls = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "sqlalchemy": MagicMock(),
                "sqlalchemy.orm": MagicMock(Session=mock_session_cls),
                "sqlalchemy.orm.Session": mock_session_cls,
                "db": MagicMock(),
                "db.voice_schema": MagicMock(VoiceProfile=mock_voice_profile_cls),
            },
        ):
            with patch(
                "voice.tts.voice_profiles.os.environ", {"DATABASE_URL": "sqlite:///"}
            ):
                profile = VoiceProfileInfo(
                    "dbtest", "desc", "en", ["/a.wav"], "xtts_v2", True, False
                )
                mgr._save_to_database(profile)

    def test_load_from_database_with_mocked_sqlalchemy(self, tmp_path):
        """Full mock of the database load path."""
        with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
            mgr = VoiceProfileManager(use_database=True)

        mock_db_profile = MagicMock()
        mock_db_profile.name = "loaded"
        mock_db_profile.description = "Loaded from DB"
        mock_db_profile.language = "en"
        mock_db_profile.reference_audio_paths = ["/some/audio.wav"]
        mock_db_profile.tts_engine = "xtts_v2"
        mock_db_profile.is_active = True
        mock_db_profile.is_default = False

        mock_session_cls = MagicMock()
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter_by.return_value.all.return_value = [
            mock_db_profile
        ]

        mock_engine_factory = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "sqlalchemy": MagicMock(create_engine=mock_engine_factory),
                "sqlalchemy.orm": MagicMock(Session=mock_session_cls),
                "db": MagicMock(),
                "db.voice_schema": MagicMock(VoiceProfile=MagicMock()),
            },
        ):
            with patch(
                "voice.tts.voice_profiles.os.environ", {"DATABASE_URL": "sqlite:///"}
            ):
                count = mgr.load_from_database()
                assert count == 1
                assert "loaded" in mgr._profiles
                assert mgr._profiles["loaded"].language == "en"


class TestVoiceSamplesDir:
    """Tests for the VOICE_SAMPLES_DIR constant."""

    def test_voice_samples_dir_is_path(self):
        assert isinstance(VOICE_SAMPLES_DIR, Path)

    def test_voice_samples_dir_ends_correctly(self):
        assert VOICE_SAMPLES_DIR.name == "voice_samples"
        assert VOICE_SAMPLES_DIR.parent.name == "data"


# ============================================================================
#  SECTION E -- Edge cases and integration-style tests
# ============================================================================


class TestEdgeCases:
    """Additional edge-case tests for both modules."""

    def test_oww_service_process_batch_small_audio(self, oww_service):
        oww_service._model.predict.return_value = {}
        audio = np.zeros(10, dtype=np.int16)
        detections = oww_service.process_batch(audio, chunk_size=1280)
        # Should pad and process 1 chunk
        assert oww_service._model.predict.call_count == 1

    def test_oww_service_multiple_process_calls_accumulate_history(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.9}
        for i in range(7):
            oww_service.process(_make_audio_chunk(chunk_index=i))
        assert len(oww_service._detection_history) == 7

    def test_voice_profile_all_files_must_exist(self, tmp_path):
        f = tmp_path / "exists.wav"
        f.write_bytes(b"\x00")
        p = VoiceProfileInfo(
            "test",
            "",
            "en",
            [str(f), str(tmp_path / "ghost.wav")],
            "xtts_v2",
            True,
            False,
        )
        assert p.has_audio is False

    def test_create_profile_empty_audio_list_raises(self, profile_manager):
        with pytest.raises(ValueError, match="No valid reference audio"):
            profile_manager.create_profile("empty", [])

    def test_process_batch_with_float32_audio(self, oww_service):
        oww_service._model.predict.return_value = {}
        audio = np.zeros(1280, dtype=np.float32)
        oww_service.process_batch(audio, chunk_size=1280)
        # Should still call predict (conversion happens in process)
        assert oww_service._model.predict.call_count == 1

    def test_process_detection_confidence_is_float(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.85}
        detections = oww_service.process(_make_audio_chunk())
        assert isinstance(detections[0].confidence, float)

    def test_get_detection_history_returns_most_recent(self, oww_service):
        oww_service._model.predict.return_value = {"alexa": 0.9}
        for i in range(20):
            oww_service.process(_make_audio_chunk(chunk_index=i))
        history = oww_service.get_detection_history(limit=3)
        # Last 3 should have chunk_index 17, 18, 19
        offsets = [d.audio_offset_samples for d in history]
        expected_base = [17, 18, 19]
        for h, e in zip(history, expected_base):
            assert h.audio_offset_samples == e * 1280

    def test_oww_service_init_with_no_config_uses_default(self):
        with patch("voice.wakeword.openwakeword_service.get_voice_config") as mock_cfg:
            mock_cfg.return_value.wakeword = WakeWordConfig()
            svc = OpenWakeWordService()
            assert svc.config is not None

    def test_profile_manager_load_only_telugu(self, tmp_path):
        (tmp_path / "friday_te_01.wav").write_bytes(b"\x00")
        with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
            mgr = VoiceProfileManager(use_database=False)
        assert "friday_telugu" in mgr._profiles
        assert "friday_english" not in mgr._profiles

    def test_profile_manager_load_only_english(self, tmp_path):
        (tmp_path / "friday_en_01.wav").write_bytes(b"\x00")
        with patch("voice.tts.voice_profiles.VOICE_SAMPLES_DIR", tmp_path):
            mgr = VoiceProfileManager(use_database=False)
        assert "friday_english" in mgr._profiles
        assert "friday_telugu" not in mgr._profiles
