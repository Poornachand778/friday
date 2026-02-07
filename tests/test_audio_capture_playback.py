"""
Comprehensive tests for voice/audio/capture.py and voice/audio/playback.py
==========================================================================

Tests AudioChunk, AudioCapture, AudioPlayback, and standalone functions.
Mocks sounddevice/soundfile at sys.modules level since they may not be installed.
"""

import queue
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock sounddevice, soundfile, AND webrtcvad BEFORE any voice.audio imports.
# voice/audio/__init__.py imports vad.py which requires webrtcvad, so we
# must inject all three into sys.modules early.
# ---------------------------------------------------------------------------
mock_sd = MagicMock()
mock_sf = MagicMock()
mock_webrtcvad = MagicMock()
sys.modules.setdefault("sounddevice", mock_sd)
sys.modules.setdefault("soundfile", mock_sf)
sys.modules.setdefault("webrtcvad", mock_webrtcvad)

# Create a real AudioConfig / VoiceConfig for tests
from voice.config import AudioConfig, VoiceConfig

# Now import the modules under test (webrtcvad is already in sys.modules
# so voice.audio.__init__ -> vad.py will not blow up).
from voice.audio.capture import (
    AudioCapture,
    AudioChunk,
    get_default_input_device,
    record_audio,
)
from voice.audio.playback import (
    AudioPlayback,
    get_default_output_device,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_config(**overrides) -> AudioConfig:
    defaults = dict(
        sample_rate=16000,
        channels=1,
        dtype="int16",
        chunk_size=512,
        vad_aggressiveness=2,
        silence_threshold_ms=500,
        max_recording_seconds=30.0,
    )
    defaults.update(overrides)
    return AudioConfig(**defaults)


@pytest.fixture(autouse=True)
def _reset_sd_mocks():
    """Reset all sounddevice / soundfile mocks between tests."""
    mock_sd.reset_mock()
    mock_sf.reset_mock()
    mock_sd.CallbackFlags = MagicMock
    mock_sd.CallbackStop = type("CallbackStop", (Exception,), {})
    yield


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def capture(config):
    with patch("voice.audio.capture.get_voice_config", return_value=VoiceConfig()):
        c = AudioCapture(config=config)
    return c


@pytest.fixture
def playback(config):
    with patch("voice.audio.playback.get_voice_config", return_value=VoiceConfig()):
        p = AudioPlayback(config=config)
    return p


# ===================================================================
# SECTION 1 - AudioChunk dataclass
# ===================================================================


class TestAudioChunk:

    def test_creation_basic(self):
        data = np.zeros(512, dtype="int16")
        chunk = AudioChunk(data=data, timestamp=1.0, sample_rate=16000)
        assert chunk.sample_rate == 16000
        assert chunk.timestamp == 1.0
        np.testing.assert_array_equal(chunk.data, data)

    def test_defaults_is_speech_false(self):
        chunk = AudioChunk(data=np.array([1, 2, 3]), timestamp=0.0, sample_rate=16000)
        assert chunk.is_speech is False

    def test_defaults_chunk_index_zero(self):
        chunk = AudioChunk(data=np.array([1]), timestamp=0.0, sample_rate=16000)
        assert chunk.chunk_index == 0

    def test_custom_is_speech(self):
        chunk = AudioChunk(
            data=np.array([1]), timestamp=0.0, sample_rate=16000, is_speech=True
        )
        assert chunk.is_speech is True

    def test_custom_chunk_index(self):
        chunk = AudioChunk(
            data=np.array([1]), timestamp=0.0, sample_rate=16000, chunk_index=42
        )
        assert chunk.chunk_index == 42

    def test_data_is_ndarray(self):
        data = np.random.randn(1024).astype(np.float32)
        chunk = AudioChunk(data=data, timestamp=0.5, sample_rate=22050)
        assert isinstance(chunk.data, np.ndarray)

    def test_various_sample_rates(self):
        for sr in [8000, 16000, 22050, 44100, 48000]:
            chunk = AudioChunk(data=np.array([0]), timestamp=0.0, sample_rate=sr)
            assert chunk.sample_rate == sr

    def test_data_preserved_exactly(self):
        data = np.array([1, -1, 32767, -32768], dtype="int16")
        chunk = AudioChunk(data=data, timestamp=0.0, sample_rate=16000)
        np.testing.assert_array_equal(chunk.data, data)

    def test_float_timestamp(self):
        ts = time.time()
        chunk = AudioChunk(data=np.array([0]), timestamp=ts, sample_rate=16000)
        assert chunk.timestamp == ts

    def test_large_chunk_index(self):
        chunk = AudioChunk(
            data=np.array([0]), timestamp=0.0, sample_rate=16000, chunk_index=999999
        )
        assert chunk.chunk_index == 999999


# ===================================================================
# SECTION 2 - AudioCapture.__init__
# ===================================================================


class TestAudioCaptureInit:

    def test_init_with_provided_config(self, config):
        with patch("voice.audio.capture.get_voice_config"):
            c = AudioCapture(config=config)
        assert c.config is config

    def test_init_default_config(self):
        mock_vc = MagicMock()
        mock_vc.return_value.audio = _make_config()
        with patch("voice.audio.capture.get_voice_config", mock_vc):
            c = AudioCapture()
        assert c.config is mock_vc.return_value.audio

    def test_init_device_none_by_default(self, config):
        with patch("voice.audio.capture.get_voice_config"):
            c = AudioCapture(config=config)
        assert c.device is None

    def test_init_custom_device(self, config):
        with patch("voice.audio.capture.get_voice_config"):
            c = AudioCapture(config=config, device=3)
        assert c.device == 3

    def test_init_running_false(self, capture):
        assert capture._running is False

    def test_init_chunk_index_zero(self, capture):
        assert capture._chunk_index == 0

    def test_init_queue_empty(self, capture):
        assert capture._queue.empty()

    def test_init_stream_none(self, capture):
        assert capture._stream is None

    def test_init_recording_buffer_empty(self, capture):
        assert capture._recording_buffer == []

    def test_init_is_recording_false(self, capture):
        assert capture._is_recording is False

    def test_init_recording_start_zero(self, capture):
        assert capture._recording_start == 0.0


# ===================================================================
# SECTION 3 - AudioCapture._audio_callback
# ===================================================================


class TestAudioCaptureCallback:

    def test_callback_creates_chunk(self, capture):
        indata = np.ones((512, 1), dtype="int16")
        capture._audio_callback(indata, 512, {}, None)
        chunk = capture._queue.get_nowait()
        assert isinstance(chunk, AudioChunk)

    def test_callback_increments_chunk_index(self, capture):
        indata = np.ones((512, 1), dtype="int16")
        capture._audio_callback(indata, 512, {}, None)
        capture._audio_callback(indata, 512, {}, None)
        c1 = capture._queue.get_nowait()
        c2 = capture._queue.get_nowait()
        assert c1.chunk_index == 0
        assert c2.chunk_index == 1

    def test_callback_copies_data(self, capture):
        indata = np.ones((512, 1), dtype="int16")
        capture._audio_callback(indata, 512, {}, None)
        chunk = capture._queue.get_nowait()
        # Modify original - chunk should be unaffected
        indata[:] = 99
        assert not np.all(chunk.data == 99)

    def test_callback_flattens_data(self, capture):
        indata = np.ones((512, 1), dtype="int16")
        capture._audio_callback(indata, 512, {}, None)
        chunk = capture._queue.get_nowait()
        assert chunk.data.ndim == 1

    def test_callback_sets_sample_rate(self, capture):
        indata = np.ones((512, 1), dtype="int16")
        capture._audio_callback(indata, 512, {}, None)
        chunk = capture._queue.get_nowait()
        assert chunk.sample_rate == capture.config.sample_rate

    def test_callback_sets_timestamp(self, capture):
        before = time.time()
        indata = np.ones((512, 1), dtype="int16")
        capture._audio_callback(indata, 512, {}, None)
        after = time.time()
        chunk = capture._queue.get_nowait()
        assert before <= chunk.timestamp <= after

    def test_callback_with_status_logs_warning(self, capture):
        indata = np.ones((512, 1), dtype="int16")
        with patch("voice.audio.capture.LOGGER") as mock_logger:
            capture._audio_callback(indata, 512, {}, "overflow")
            mock_logger.warning.assert_called_once()

    def test_callback_no_status_no_warning(self, capture):
        indata = np.ones((512, 1), dtype="int16")
        with patch("voice.audio.capture.LOGGER") as mock_logger:
            capture._audio_callback(indata, 512, {}, None)
            mock_logger.warning.assert_not_called()

    def test_callback_recording_appends_buffer(self, capture):
        capture._is_recording = True
        indata = np.ones((512, 1), dtype="int16")
        capture._audio_callback(indata, 512, {}, None)
        assert len(capture._recording_buffer) == 1

    def test_callback_not_recording_no_buffer(self, capture):
        capture._is_recording = False
        indata = np.ones((512, 1), dtype="int16")
        capture._audio_callback(indata, 512, {}, None)
        assert len(capture._recording_buffer) == 0

    def test_callback_recording_buffer_copies_data(self, capture):
        capture._is_recording = True
        indata = np.ones((512, 1), dtype="int16")
        capture._audio_callback(indata, 512, {}, None)
        # Modify original
        indata[:] = 0
        assert np.all(capture._recording_buffer[0] == 1)


# ===================================================================
# SECTION 4 - AudioCapture.start / stop
# ===================================================================


class TestAudioCaptureStartStop:

    def test_start_creates_input_stream(self, capture):
        capture.start()
        mock_sd.InputStream.assert_called_once()

    def test_start_passes_config_params(self, capture):
        capture.start()
        mock_sd.InputStream.assert_called_once_with(
            samplerate=capture.config.sample_rate,
            channels=capture.config.channels,
            dtype=capture.config.dtype,
            blocksize=capture.config.chunk_size,
            device=capture.device,
            callback=capture._audio_callback,
        )

    def test_start_calls_stream_start(self, capture):
        capture.start()
        mock_sd.InputStream.return_value.start.assert_called_once()

    def test_start_sets_running_true(self, capture):
        capture.start()
        assert capture._running is True

    def test_start_resets_chunk_index(self, capture):
        capture._chunk_index = 42
        capture.start()
        assert capture._chunk_index == 0

    def test_start_when_already_running_no_op(self, capture):
        capture._running = True
        capture.start()
        mock_sd.InputStream.assert_not_called()

    def test_stop_sets_running_false(self, capture):
        capture._running = True
        capture._stream = MagicMock()
        capture.stop()
        assert capture._running is False

    def test_stop_stops_stream(self, capture):
        stream_mock = MagicMock()
        capture._running = True
        capture._stream = stream_mock
        capture.stop()
        stream_mock.stop.assert_called_once()

    def test_stop_closes_stream(self, capture):
        stream_mock = MagicMock()
        capture._running = True
        capture._stream = stream_mock
        capture.stop()
        stream_mock.close.assert_called_once()

    def test_stop_sets_stream_none(self, capture):
        capture._running = True
        capture._stream = MagicMock()
        capture.stop()
        assert capture._stream is None

    def test_stop_puts_none_sentinel(self, capture):
        capture._running = True
        capture._stream = MagicMock()
        capture.stop()
        sentinel = capture._queue.get_nowait()
        assert sentinel is None

    def test_stop_when_not_running_no_op(self, capture):
        capture._running = False
        capture.stop()  # Should not raise
        assert capture._queue.empty()

    def test_stop_without_stream(self, capture):
        capture._running = True
        capture._stream = None
        capture.stop()
        assert capture._running is False


# ===================================================================
# SECTION 5 - AudioCapture.stream()
# ===================================================================


class TestAudioCaptureStream:

    def test_stream_yields_chunks(self, capture):
        chunk = AudioChunk(data=np.array([1, 2, 3]), timestamp=1.0, sample_rate=16000)
        capture._running = True
        capture._queue.put(chunk)
        capture._queue.put(None)  # sentinel

        results = list(capture.stream(timeout=0.1))
        assert len(results) == 1
        assert results[0] is chunk

    def test_stream_stops_on_none(self, capture):
        capture._running = True
        capture._queue.put(None)
        results = list(capture.stream(timeout=0.1))
        assert results == []

    def test_stream_yields_multiple_chunks(self, capture):
        capture._running = True
        for i in range(5):
            capture._queue.put(
                AudioChunk(
                    data=np.array([i]),
                    timestamp=float(i),
                    sample_rate=16000,
                    chunk_index=i,
                )
            )
        capture._queue.put(None)

        results = list(capture.stream(timeout=0.1))
        assert len(results) == 5
        for i, c in enumerate(results):
            assert c.chunk_index == i

    def test_stream_handles_empty_queue_timeout(self, capture):
        """When queue is empty and not running, stream should exit."""
        capture._running = False
        results = list(capture.stream(timeout=0.05))
        assert results == []

    def test_stream_continues_on_queue_empty(self, capture):
        """When running but queue empty, stream continues until data arrives."""
        capture._running = True

        collected = []

        def feeder():
            time.sleep(0.05)
            capture._queue.put(
                AudioChunk(data=np.array([1]), timestamp=0.0, sample_rate=16000)
            )
            time.sleep(0.05)
            capture._queue.put(None)

        t = threading.Thread(target=feeder)
        t.start()

        for chunk in capture.stream(timeout=0.2):
            collected.append(chunk)
            # After getting one chunk, set running to False so stream checks exit
            capture._running = False

        t.join(timeout=2)
        assert len(collected) >= 1


# ===================================================================
# SECTION 6 - AudioCapture.start_recording / stop_recording
# ===================================================================


class TestAudioCaptureRecording:

    def test_start_recording_clears_buffer(self, capture):
        capture._recording_buffer = [np.array([1])]
        capture.start_recording()
        assert capture._recording_buffer == []

    def test_start_recording_sets_flag(self, capture):
        capture.start_recording()
        assert capture._is_recording is True

    def test_start_recording_sets_start_time(self, capture):
        before = time.time()
        capture.start_recording()
        after = time.time()
        assert before <= capture._recording_start <= after

    def test_stop_recording_clears_flag(self, capture):
        capture._is_recording = True
        capture._recording_start = time.time()
        capture.stop_recording()
        assert capture._is_recording is False

    def test_stop_recording_empty_buffer_returns_empty(self, capture):
        capture._is_recording = True
        capture._recording_start = time.time()
        audio, duration = capture.stop_recording()
        assert len(audio) == 0
        assert duration == 0.0

    def test_stop_recording_returns_concatenated_audio(self, capture):
        capture._is_recording = True
        capture._recording_start = time.time() - 1.0
        capture._recording_buffer = [
            np.array([1, 2], dtype="int16"),
            np.array([3, 4], dtype="int16"),
        ]
        audio, duration = capture.stop_recording()
        np.testing.assert_array_equal(audio, np.array([1, 2, 3, 4], dtype="int16"))

    def test_stop_recording_returns_duration(self, capture):
        capture._is_recording = True
        capture._recording_start = time.time() - 2.5
        capture._recording_buffer = [np.array([0], dtype="int16")]
        _, duration = capture.stop_recording()
        assert 2.0 <= duration <= 3.0

    def test_stop_recording_clears_buffer(self, capture):
        capture._is_recording = True
        capture._recording_start = time.time()
        capture._recording_buffer = [np.array([1], dtype="int16")]
        capture.stop_recording()
        assert capture._recording_buffer == []

    def test_start_stop_recording_cycle(self, capture):
        capture.start_recording()
        assert capture._is_recording is True
        capture._recording_buffer.append(np.array([10, 20], dtype="int16"))
        audio, dur = capture.stop_recording()
        assert capture._is_recording is False
        np.testing.assert_array_equal(audio, np.array([10, 20], dtype="int16"))


# ===================================================================
# SECTION 7 - AudioCapture.save_recording
# ===================================================================


class TestAudioCaptureSaveRecording:

    def test_save_recording_calls_sf_write(self, capture, tmp_path):
        audio = np.array([1, 2, 3], dtype="int16")
        out_path = tmp_path / "test.wav"
        capture.save_recording(audio, out_path)
        mock_sf.write.assert_called_once_with(
            str(out_path), audio, capture.config.sample_rate
        )

    def test_save_recording_custom_sample_rate(self, capture, tmp_path):
        audio = np.array([1, 2, 3], dtype="int16")
        out_path = tmp_path / "test.wav"
        capture.save_recording(audio, out_path, sample_rate=44100)
        mock_sf.write.assert_called_once_with(str(out_path), audio, 44100)

    def test_save_recording_creates_parent_dirs(self, capture, tmp_path):
        audio = np.array([1], dtype="int16")
        out_path = tmp_path / "sub" / "dir" / "test.wav"
        capture.save_recording(audio, out_path)
        assert out_path.parent.exists()

    def test_save_recording_returns_path(self, capture, tmp_path):
        audio = np.array([1], dtype="int16")
        out_path = tmp_path / "test.wav"
        result = capture.save_recording(audio, out_path)
        assert result == out_path

    def test_save_recording_accepts_string_path(self, capture, tmp_path):
        audio = np.array([1], dtype="int16")
        out_path = str(tmp_path / "test.wav")
        result = capture.save_recording(audio, out_path)
        assert isinstance(result, Path)


# ===================================================================
# SECTION 8 - AudioCapture.get_input_devices
# ===================================================================


class TestAudioCaptureGetInputDevices:

    def test_get_input_devices_filters_input(self, capture):
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 44100},
        ]
        mock_sd.default.device = (0, 1)
        devices = capture.get_input_devices()
        assert len(devices) == 1
        assert devices[0]["name"] == "Mic"

    def test_get_input_devices_returns_correct_fields(self, capture):
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_input_channels": 2, "default_samplerate": 48000},
        ]
        mock_sd.default.device = (0, 1)
        devices = capture.get_input_devices()
        d = devices[0]
        assert d["index"] == 0
        assert d["name"] == "Mic"
        assert d["channels"] == 2
        assert d["sample_rate"] == 48000
        assert d["is_default"] is True

    def test_get_input_devices_empty_list(self, capture):
        mock_sd.query_devices.return_value = []
        devices = capture.get_input_devices()
        assert devices == []

    def test_get_input_devices_multiple(self, capture):
        mock_sd.query_devices.return_value = [
            {"name": "Mic1", "max_input_channels": 1, "default_samplerate": 16000},
            {"name": "Mic2", "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 44100},
        ]
        mock_sd.default.device = (0, 2)
        devices = capture.get_input_devices()
        assert len(devices) == 2

    def test_get_input_devices_default_marker(self, capture):
        mock_sd.query_devices.return_value = [
            {"name": "Mic1", "max_input_channels": 1, "default_samplerate": 16000},
            {"name": "Mic2", "max_input_channels": 2, "default_samplerate": 44100},
        ]
        mock_sd.default.device = (1, 0)
        devices = capture.get_input_devices()
        assert devices[0]["is_default"] is False
        assert devices[1]["is_default"] is True


# ===================================================================
# SECTION 9 - AudioCapture.is_running property
# ===================================================================


class TestAudioCaptureIsRunning:

    def test_is_running_initially_false(self, capture):
        assert capture.is_running is False

    def test_is_running_after_start(self, capture):
        capture.start()
        assert capture.is_running is True

    def test_is_running_after_stop(self, capture):
        capture.start()
        capture.stop()
        assert capture.is_running is False


# ===================================================================
# SECTION 10 - AudioCapture context manager
# ===================================================================


class TestAudioCaptureContextManager:

    def test_enter_calls_start(self, capture):
        with patch.object(capture, "start") as mock_start:
            with patch.object(capture, "stop"):
                capture.__enter__()
                mock_start.assert_called_once()

    def test_enter_returns_self(self, capture):
        with patch.object(capture, "start"):
            result = capture.__enter__()
        assert result is capture

    def test_exit_calls_stop(self, capture):
        with patch.object(capture, "start"):
            with patch.object(capture, "stop") as mock_stop:
                capture.__enter__()
                capture.__exit__(None, None, None)
                mock_stop.assert_called_once()

    def test_context_manager_full(self, capture):
        with patch.object(capture, "start") as s, patch.object(capture, "stop") as st:
            with capture:
                s.assert_called_once()
            st.assert_called_once()


# ===================================================================
# SECTION 11 - record_audio standalone function
# ===================================================================


class TestRecordAudio:

    def test_record_audio_calls_sd_rec(self):
        mock_sd.rec.return_value = np.zeros((16000, 1), dtype="int16")
        result = record_audio(1.0)
        mock_sd.rec.assert_called_once_with(
            16000,
            samplerate=16000,
            channels=1,
            dtype="int16",
            device=None,
        )

    def test_record_audio_calls_sd_wait(self):
        mock_sd.rec.return_value = np.zeros((16000, 1), dtype="int16")
        record_audio(1.0)
        mock_sd.wait.assert_called_once()

    def test_record_audio_custom_sample_rate(self):
        mock_sd.rec.return_value = np.zeros((44100, 1), dtype="int16")
        record_audio(1.0, sample_rate=44100)
        mock_sd.rec.assert_called_once_with(
            44100,
            samplerate=44100,
            channels=1,
            dtype="int16",
            device=None,
        )

    def test_record_audio_custom_device(self):
        mock_sd.rec.return_value = np.zeros((16000, 1), dtype="int16")
        record_audio(1.0, device=2)
        mock_sd.rec.assert_called_once_with(
            16000,
            samplerate=16000,
            channels=1,
            dtype="int16",
            device=2,
        )

    def test_record_audio_returns_flattened(self):
        mock_sd.rec.return_value = np.ones((16000, 1), dtype="int16")
        result = record_audio(1.0)
        assert result.ndim == 1
        assert len(result) == 16000

    def test_record_audio_duration_multiplied(self):
        mock_sd.rec.return_value = np.zeros((48000, 1), dtype="int16")
        record_audio(3.0, sample_rate=16000)
        mock_sd.rec.assert_called_once_with(
            48000,
            samplerate=16000,
            channels=1,
            dtype="int16",
            device=None,
        )


# ===================================================================
# SECTION 12 - get_default_input_device standalone function
# ===================================================================


class TestGetDefaultInputDevice:

    def test_returns_device_dict(self):
        mock_sd.default.device = (0, 1)
        mock_sd.query_devices.return_value = {
            "name": "Default Mic",
            "max_input_channels": 2,
            "default_samplerate": 44100,
        }
        result = get_default_input_device()
        assert result["index"] == 0
        assert result["name"] == "Default Mic"
        assert result["channels"] == 2
        assert result["sample_rate"] == 44100

    def test_calls_query_devices_with_id(self):
        mock_sd.default.device = (5, 1)
        mock_sd.query_devices.return_value = {
            "name": "Mic",
            "max_input_channels": 1,
            "default_samplerate": 16000,
        }
        get_default_input_device()
        mock_sd.query_devices.assert_called_with(5)


# ===================================================================
# SECTION 13 - AudioPlayback.__init__
# ===================================================================


class TestAudioPlaybackInit:

    def test_init_with_config(self, config):
        with patch("voice.audio.playback.get_voice_config"):
            p = AudioPlayback(config=config)
        assert p.config is config

    def test_init_default_config(self):
        mock_vc = MagicMock()
        mock_vc.return_value.audio = _make_config()
        with patch("voice.audio.playback.get_voice_config", mock_vc):
            p = AudioPlayback()
        assert p.config is mock_vc.return_value.audio

    def test_init_device_none(self, config):
        with patch("voice.audio.playback.get_voice_config"):
            p = AudioPlayback(config=config)
        assert p.device is None

    def test_init_custom_device(self, config):
        with patch("voice.audio.playback.get_voice_config"):
            p = AudioPlayback(config=config, device=7)
        assert p.device == 7

    def test_init_is_playing_false(self, playback):
        assert playback._is_playing is False

    def test_init_stream_none(self, playback):
        assert playback._stream is None

    def test_init_queue_empty(self, playback):
        assert playback._queue.empty()


# ===================================================================
# SECTION 14 - AudioPlayback.play
# ===================================================================


class TestAudioPlaybackPlay:

    def test_play_blocking_calls_sd_play_and_wait(self, playback):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        playback.play(audio, sample_rate=22050, blocking=True)
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()

    def test_play_non_blocking_no_wait(self, playback):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        playback.play(audio, sample_rate=22050, blocking=False)
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_not_called()

    def test_play_default_sample_rate_22050(self, playback):
        audio = np.array([0.5], dtype=np.float32)
        playback.play(audio)
        args, kwargs = mock_sd.play.call_args
        assert args[1] == 22050  # sr

    def test_play_custom_sample_rate(self, playback):
        audio = np.array([0.5], dtype=np.float32)
        playback.play(audio, sample_rate=44100)
        args, kwargs = mock_sd.play.call_args
        assert args[1] == 44100

    def test_play_passes_device(self, playback):
        playback.device = 3
        audio = np.array([0.5], dtype=np.float32)
        playback.play(audio)
        _, kwargs = mock_sd.play.call_args
        assert kwargs["device"] == 3

    def test_play_normalizes_audio_above_one(self, playback):
        audio = np.array([2.0, -2.0], dtype=np.float32)
        playback.play(audio, blocking=True)
        played = mock_sd.play.call_args[0][0]
        assert played.max() <= 1.0
        assert played.min() >= -1.0

    def test_play_does_not_normalize_within_range(self, playback):
        audio = np.array([0.5, -0.3], dtype=np.float32)
        playback.play(audio, blocking=True)
        played = mock_sd.play.call_args[0][0]
        np.testing.assert_allclose(played, np.array([0.5, -0.3], dtype=np.float32))

    def test_play_converts_to_float32(self, playback):
        audio = np.array([100, -100], dtype=np.int16)
        playback.play(audio, blocking=True)
        played = mock_sd.play.call_args[0][0]
        assert played.dtype == np.float32

    def test_play_normalizes_large_int_values(self, playback):
        audio = np.array([32767, -32768], dtype=np.int16)
        playback.play(audio, blocking=True)
        played = mock_sd.play.call_args[0][0]
        assert played.max() <= 1.0
        assert played.min() >= -1.0

    def test_play_default_blocking_true(self, playback):
        audio = np.array([0.1], dtype=np.float32)
        playback.play(audio)
        mock_sd.wait.assert_called_once()


# ===================================================================
# SECTION 15 - AudioPlayback.play_file
# ===================================================================


class TestAudioPlaybackPlayFile:

    def test_play_file_raises_file_not_found(self, playback):
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            playback.play_file("/nonexistent/audio.wav")

    def test_play_file_calls_sf_read(self, playback, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.touch()
        mock_sf.read.return_value = (np.array([0.1, 0.2], dtype=np.float32), 22050)
        playback.play_file(wav_file)
        mock_sf.read.assert_called_once_with(str(wav_file))

    def test_play_file_returns_duration(self, playback, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.touch()
        # 22050 samples at 22050 Hz = 1.0 second
        mock_sf.read.return_value = (np.zeros(22050, dtype=np.float32), 22050)
        duration = playback.play_file(wav_file)
        assert duration == pytest.approx(1.0)

    def test_play_file_calls_play(self, playback, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.touch()
        audio = np.zeros(1000, dtype=np.float32)
        mock_sf.read.return_value = (audio, 22050)
        with patch.object(playback, "play") as mock_play:
            playback.play_file(wav_file, blocking=False)
            mock_play.assert_called_once()
            _, kwargs = mock_play.call_args
            assert kwargs["sample_rate"] == 22050
            assert kwargs["blocking"] is False

    def test_play_file_accepts_string_path(self, playback, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.touch()
        mock_sf.read.return_value = (np.zeros(100, dtype=np.float32), 16000)
        playback.play_file(str(wav_file))
        mock_sf.read.assert_called_once()

    def test_play_file_blocking_default_true(self, playback, tmp_path):
        wav_file = tmp_path / "test.wav"
        wav_file.touch()
        mock_sf.read.return_value = (np.zeros(100, dtype=np.float32), 16000)
        with patch.object(playback, "play") as mock_play:
            playback.play_file(wav_file)
            _, kwargs = mock_play.call_args
            assert kwargs["blocking"] is True


# ===================================================================
# SECTION 16 - AudioPlayback.stop
# ===================================================================


class TestAudioPlaybackStop:

    def test_stop_calls_sd_stop(self, playback):
        playback.stop()
        mock_sd.stop.assert_called_once()

    def test_stop_can_be_called_multiple_times(self, playback):
        playback.stop()
        playback.stop()
        assert mock_sd.stop.call_count == 2


# ===================================================================
# SECTION 17 - AudioPlayback streaming
# ===================================================================


class TestAudioPlaybackStreaming:

    def test_start_stream_creates_output_stream(self, playback):
        playback.start_stream()
        mock_sd.OutputStream.assert_called_once()

    def test_start_stream_passes_params(self, playback):
        playback.start_stream(sample_rate=44100)
        mock_sd.OutputStream.assert_called_once_with(
            samplerate=44100,
            channels=1,
            dtype="float32",
            blocksize=1024,
            device=playback.device,
            callback=playback._stream_callback,
        )

    def test_start_stream_calls_stream_start(self, playback):
        playback.start_stream()
        mock_sd.OutputStream.return_value.start.assert_called_once()

    def test_start_stream_sets_is_playing_true(self, playback):
        playback.start_stream()
        assert playback._is_playing is True

    def test_start_stream_default_sample_rate(self, playback):
        playback.start_stream()
        args, kwargs = mock_sd.OutputStream.call_args
        assert kwargs["samplerate"] == 22050

    def test_start_stream_when_already_playing_no_op(self, playback):
        playback._is_playing = True
        playback.start_stream()
        mock_sd.OutputStream.assert_not_called()

    def test_stream_chunk_puts_data_in_queue(self, playback):
        playback._is_playing = True
        chunk = np.array([0.1, 0.2], dtype=np.float32)
        playback.stream_chunk(chunk)
        queued = playback._queue.get_nowait()
        assert isinstance(queued, np.ndarray)

    def test_stream_chunk_normalizes(self, playback):
        playback._is_playing = True
        chunk = np.array([5.0, -5.0], dtype=np.float32)
        playback.stream_chunk(chunk)
        queued = playback._queue.get_nowait()
        assert queued.max() <= 1.0
        assert queued.min() >= -1.0

    def test_stream_chunk_converts_to_float32(self, playback):
        playback._is_playing = True
        chunk = np.array([100, -100], dtype=np.int16)
        playback.stream_chunk(chunk)
        queued = playback._queue.get_nowait()
        assert queued.dtype == np.float32

    def test_stream_chunk_auto_starts_stream(self, playback):
        playback._is_playing = False
        with patch.object(playback, "start_stream") as mock_start:
            # After calling start_stream, set _is_playing to True to avoid recursion
            def side_effect(*a, **k):
                playback._is_playing = True

            mock_start.side_effect = side_effect
            chunk = np.array([0.5], dtype=np.float32)
            playback.stream_chunk(chunk)
            mock_start.assert_called_once()

    def test_stream_chunk_does_not_normalize_in_range(self, playback):
        playback._is_playing = True
        chunk = np.array([0.5, -0.3], dtype=np.float32)
        playback.stream_chunk(chunk)
        queued = playback._queue.get_nowait()
        np.testing.assert_allclose(queued, np.array([0.5, -0.3], dtype=np.float32))

    def test_stop_stream_sets_is_playing_false(self, playback):
        playback._is_playing = True
        playback._stream = MagicMock()
        playback.stop_stream()
        assert playback._is_playing is False

    def test_stop_stream_stops_and_closes_stream(self, playback):
        stream_mock = MagicMock()
        playback._is_playing = True
        playback._stream = stream_mock
        playback.stop_stream()
        stream_mock.stop.assert_called_once()
        stream_mock.close.assert_called_once()

    def test_stop_stream_sets_stream_none(self, playback):
        playback._is_playing = True
        playback._stream = MagicMock()
        playback.stop_stream()
        assert playback._stream is None

    def test_stop_stream_puts_none_sentinel(self, playback):
        playback._is_playing = True
        playback._stream = MagicMock()
        playback.stop_stream()
        # The None sentinel should be in the queue
        found_none = False
        while not playback._queue.empty():
            item = playback._queue.get_nowait()
            if item is None:
                found_none = True
        assert found_none

    def test_stop_stream_when_not_playing_no_op(self, playback):
        playback._is_playing = False
        playback.stop_stream()  # Should not raise

    def test_stop_stream_without_stream_object(self, playback):
        playback._is_playing = True
        playback._stream = None
        playback.stop_stream()
        assert playback._is_playing is False


# ===================================================================
# SECTION 18 - AudioPlayback._stream_callback
# ===================================================================


class TestAudioPlaybackStreamCallback:

    def test_callback_fills_outdata_from_queue(self, playback):
        data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        playback._queue.put(data)
        outdata = np.zeros((4, 1), dtype=np.float32)
        playback._stream_callback(outdata, 4, {}, None)
        np.testing.assert_array_equal(outdata[:, 0], data)

    def test_callback_pads_short_data(self, playback):
        data = np.array([0.5, 0.6], dtype=np.float32)
        playback._queue.put(data)
        outdata = np.zeros((4, 1), dtype=np.float32)
        playback._stream_callback(outdata, 4, {}, None)
        np.testing.assert_array_equal(outdata[:2, 0], data)
        np.testing.assert_array_equal(outdata[2:, 0], [0.0, 0.0])

    def test_callback_truncates_long_data(self, playback):
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        playback._queue.put(data)
        outdata = np.zeros((4, 1), dtype=np.float32)
        playback._stream_callback(outdata, 4, {}, None)
        np.testing.assert_array_equal(outdata[:, 0], data[:4])

    def test_callback_silence_on_empty_queue(self, playback):
        outdata = np.ones((4, 1), dtype=np.float32)
        playback._stream_callback(outdata, 4, {}, None)
        np.testing.assert_array_equal(outdata, np.zeros((4, 1)))

    def test_callback_raises_stop_on_none(self, playback):
        playback._queue.put(None)
        outdata = np.zeros((4, 1), dtype=np.float32)
        with pytest.raises(mock_sd.CallbackStop):
            playback._stream_callback(outdata, 4, {}, None)

    def test_callback_logs_status_warning(self, playback):
        outdata = np.zeros((4, 1), dtype=np.float32)
        with patch("voice.audio.playback.LOGGER") as mock_logger:
            playback._stream_callback(outdata, 4, {}, "underflow")
            mock_logger.warning.assert_called_once()

    def test_callback_no_status_no_warning(self, playback):
        outdata = np.zeros((4, 1), dtype=np.float32)
        with patch("voice.audio.playback.LOGGER") as mock_logger:
            playback._stream_callback(outdata, 4, {}, None)
            mock_logger.warning.assert_not_called()


# ===================================================================
# SECTION 19 - AudioPlayback.get_output_devices
# ===================================================================


class TestAudioPlaybackGetOutputDevices:

    def test_get_output_devices_filters_output(self, playback):
        mock_sd.query_devices.return_value = [
            {"name": "Mic", "max_output_channels": 0, "default_samplerate": 44100},
            {"name": "Speaker", "max_output_channels": 2, "default_samplerate": 44100},
        ]
        mock_sd.default.device = (0, 1)
        devices = playback.get_output_devices()
        assert len(devices) == 1
        assert devices[0]["name"] == "Speaker"

    def test_get_output_devices_returns_correct_fields(self, playback):
        mock_sd.query_devices.return_value = [
            {"name": "Speakers", "max_output_channels": 6, "default_samplerate": 48000},
        ]
        mock_sd.default.device = (1, 0)
        devices = playback.get_output_devices()
        d = devices[0]
        assert d["index"] == 0
        assert d["name"] == "Speakers"
        assert d["channels"] == 6
        assert d["sample_rate"] == 48000
        assert d["is_default"] is True

    def test_get_output_devices_empty(self, playback):
        mock_sd.query_devices.return_value = []
        devices = playback.get_output_devices()
        assert devices == []

    def test_get_output_devices_multiple(self, playback):
        mock_sd.query_devices.return_value = [
            {"name": "Speaker1", "max_output_channels": 2, "default_samplerate": 44100},
            {"name": "Speaker2", "max_output_channels": 2, "default_samplerate": 48000},
            {"name": "Mic", "max_output_channels": 0, "default_samplerate": 16000},
        ]
        mock_sd.default.device = (2, 0)
        devices = playback.get_output_devices()
        assert len(devices) == 2

    def test_get_output_devices_default_marker(self, playback):
        mock_sd.query_devices.return_value = [
            {"name": "Speaker1", "max_output_channels": 2, "default_samplerate": 44100},
            {"name": "Speaker2", "max_output_channels": 2, "default_samplerate": 48000},
        ]
        mock_sd.default.device = (0, 1)
        devices = playback.get_output_devices()
        assert devices[0]["is_default"] is False
        assert devices[1]["is_default"] is True


# ===================================================================
# SECTION 20 - AudioPlayback.is_playing property
# ===================================================================


class TestAudioPlaybackIsPlaying:

    def test_is_playing_initially_false(self, playback):
        assert playback.is_playing is False

    def test_is_playing_after_start_stream(self, playback):
        playback.start_stream()
        assert playback.is_playing is True

    def test_is_playing_after_stop_stream(self, playback):
        playback._is_playing = True
        playback._stream = MagicMock()
        playback.stop_stream()
        assert playback.is_playing is False


# ===================================================================
# SECTION 21 - AudioPlayback context manager
# ===================================================================


class TestAudioPlaybackContextManager:

    def test_enter_returns_self(self, playback):
        result = playback.__enter__()
        assert result is playback

    def test_exit_calls_stop(self, playback):
        with patch.object(playback, "stop") as mock_stop:
            with patch.object(playback, "stop_stream"):
                playback.__exit__(None, None, None)
                mock_stop.assert_called_once()

    def test_exit_calls_stop_stream(self, playback):
        with patch.object(playback, "stop"):
            with patch.object(playback, "stop_stream") as mock_stop_stream:
                playback.__exit__(None, None, None)
                mock_stop_stream.assert_called_once()

    def test_context_manager_full(self, playback):
        with patch.object(playback, "stop") as s, patch.object(
            playback, "stop_stream"
        ) as ss:
            with playback:
                pass
            s.assert_called_once()
            ss.assert_called_once()

    def test_context_manager_on_exception(self, playback):
        with patch.object(playback, "stop") as s, patch.object(
            playback, "stop_stream"
        ) as ss:
            try:
                with playback:
                    raise ValueError("test error")
            except ValueError:
                pass
            s.assert_called_once()
            ss.assert_called_once()


# ===================================================================
# SECTION 22 - get_default_output_device standalone function
# ===================================================================


class TestGetDefaultOutputDevice:

    def test_returns_device_dict(self):
        mock_sd.default.device = (0, 2)
        mock_sd.query_devices.return_value = {
            "name": "Default Speaker",
            "max_output_channels": 2,
            "default_samplerate": 48000,
        }
        result = get_default_output_device()
        assert result["index"] == 2
        assert result["name"] == "Default Speaker"
        assert result["channels"] == 2
        assert result["sample_rate"] == 48000

    def test_calls_query_devices_with_output_id(self):
        mock_sd.default.device = (0, 7)
        mock_sd.query_devices.return_value = {
            "name": "Sp",
            "max_output_channels": 2,
            "default_samplerate": 44100,
        }
        get_default_output_device()
        mock_sd.query_devices.assert_called_with(7)


# ===================================================================
# SECTION 23 - Integration / edge case tests
# ===================================================================


class TestIntegrationEdgeCases:

    def test_capture_start_stop_start(self, capture):
        """Start, stop, then start again should work."""
        capture.start()
        capture.stop()
        mock_sd.reset_mock()
        capture.start()
        mock_sd.InputStream.assert_called_once()
        assert capture.is_running is True

    def test_capture_double_stop_safe(self, capture):
        capture.start()
        capture.stop()
        capture.stop()  # Should not raise

    def test_playback_play_zero_length_audio(self, playback):
        audio = np.array([], dtype=np.float32)
        # Should not raise (max/min of empty array may cause issues, but
        # we test the call does not crash)
        try:
            playback.play(audio)
        except ValueError:
            pass  # Acceptable for empty audio

    def test_capture_callback_multiple_rapid_calls(self, capture):
        indata = np.ones((512, 1), dtype="int16")
        for _ in range(100):
            capture._audio_callback(indata, 512, {}, None)
        assert capture._queue.qsize() == 100
        assert capture._chunk_index == 100

    def test_playback_stream_callback_exact_frames(self, playback):
        """When data length exactly matches frames, no padding needed."""
        data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        playback._queue.put(data)
        outdata = np.zeros((4, 1), dtype=np.float32)
        playback._stream_callback(outdata, 4, {}, None)
        np.testing.assert_array_equal(outdata[:, 0], data)

    def test_capture_recording_across_multiple_callbacks(self, capture):
        capture.start_recording()
        for i in range(10):
            indata = np.full((512, 1), i, dtype="int16")
            capture._audio_callback(indata, 512, {}, None)
        audio, duration = capture.stop_recording()
        assert len(audio) == 5120  # 10 * 512

    def test_playback_normalizes_negative_heavy_audio(self, playback):
        audio = np.array([-10.0, -5.0, -1.0], dtype=np.float32)
        playback.play(audio, blocking=True)
        played = mock_sd.play.call_args[0][0]
        assert played.min() >= -1.0

    def test_playback_normalizes_positive_heavy_audio(self, playback):
        audio = np.array([10.0, 5.0, 1.0], dtype=np.float32)
        playback.play(audio, blocking=True)
        played = mock_sd.play.call_args[0][0]
        assert played.max() <= 1.0

    def test_capture_save_recording_with_nested_path(self, capture, tmp_path):
        audio = np.array([1, 2, 3], dtype="int16")
        out_path = tmp_path / "a" / "b" / "c" / "test.wav"
        result = capture.save_recording(audio, out_path)
        assert result == out_path
        assert out_path.parent.exists()

    def test_playback_play_file_with_path_object(self, playback, tmp_path):
        wav = tmp_path / "audio.wav"
        wav.touch()
        mock_sf.read.return_value = (np.zeros(100, dtype=np.float32), 16000)
        playback.play_file(Path(wav))
        mock_sf.read.assert_called_once()

    def test_capture_is_running_thread_safe(self, capture):
        """is_running property should reflect _running flag."""
        assert capture.is_running is False
        capture._running = True
        assert capture.is_running is True
        capture._running = False
        assert capture.is_running is False

    def test_playback_stream_chunk_multiple(self, playback):
        playback._is_playing = True
        for i in range(5):
            playback.stream_chunk(np.array([float(i) * 0.1], dtype=np.float32))
        assert playback._queue.qsize() == 5
