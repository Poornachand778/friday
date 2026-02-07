"""
Comprehensive tests for:
  1. voice/audio/vad.py - VoiceActivityDetector, vad_filter_stream, collect_utterance
  2. voice/stt/language_detector.py - detect_language, split_by_language, is_code_switched

Target: 120+ tests covering all public methods, edge cases, state transitions,
and property behaviors.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock external C-extension / hardware modules BEFORE importing vad.py
# and language_detector.py (which goes through voice.stt.__init__ which
# imports faster_whisper_service).
# ---------------------------------------------------------------------------
mock_webrtcvad = sys.modules.setdefault("webrtcvad", MagicMock())
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("faster_whisper", MagicMock())
sys.modules.setdefault("ctranslate2", MagicMock())

# Now safe to import everything from the voice package
from voice.config import AudioConfig
from voice.audio.capture import AudioChunk
from voice.audio.vad import (
    VoiceActivityDetector,
    vad_filter_stream,
    collect_utterance,
)
from voice.stt.language_detector import (
    TELUGU_RANGE,
    TELUGU_ROMANIZED_PATTERNS,
    ENGLISH_STOP_WORDS,
    LanguageInfo,
    detect_language,
    split_by_language,
    is_code_switched,
)


# ===================================================================
#  Helpers / fixtures
# ===================================================================


def _make_config(**overrides) -> AudioConfig:
    """Create an AudioConfig with sensible test defaults."""
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


def _make_chunk(
    data: np.ndarray, sample_rate: int = 16000, is_speech: bool = False
) -> AudioChunk:
    """Create an AudioChunk for testing."""
    return AudioChunk(
        data=data,
        timestamp=time.time(),
        sample_rate=sample_rate,
        is_speech=is_speech,
        chunk_index=0,
    )


def _silence(n_samples: int = 480) -> np.ndarray:
    """Return n_samples of silence (zeros) as int16."""
    return np.zeros(n_samples, dtype=np.int16)


def _tone(n_samples: int = 480, freq: float = 440.0, sr: int = 16000) -> np.ndarray:
    """Return a short sine-tone as int16."""
    t = np.arange(n_samples) / sr
    return (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)


# Telugu string helpers (Unicode range 0x0C00 - 0x0C7F)
TELUGU_HELLO = "\u0C28\u0C2E\u0C38\u0C4D\u0C15\u0C3E\u0C30\u0C02"  # నమస్కారం
TELUGU_WORD = "\u0C24\u0C46\u0C32\u0C41\u0C17\u0C41"  # తెలుగు
TELUGU_SENTENCE = "\u0C28\u0C47\u0C28\u0C41 \u0C24\u0C46\u0C32\u0C41\u0C17\u0C41 \u0C2E\u0C3E\u0C1F\u0C4D\u0C32\u0C3E\u0C21\u0C24\u0C3E\u0C28\u0C41"


@pytest.fixture
def config():
    return _make_config()


@pytest.fixture
def vad(config):
    """Return a VoiceActivityDetector with mocked webrtcvad backend."""
    mock_webrtcvad.Vad.reset_mock()
    detector = VoiceActivityDetector(config=config, aggressiveness=2)
    # The underlying Vad mock is a singleton (Vad.return_value).
    # Reset its is_speech so side_effect/return_value from a prior test
    # does not leak into the current test.
    detector._vad.is_speech.reset_mock()
    detector._vad.is_speech.side_effect = None
    detector._vad.is_speech.return_value = False  # safe default
    return detector


# ===================================================================
#  PART 1 – VoiceActivityDetector Tests
# ===================================================================


class TestVadInit:
    """Tests for VoiceActivityDetector.__init__"""

    def test_init_with_default_config(self):
        """VAD initializes with default AudioConfig."""
        vad = VoiceActivityDetector(config=_make_config())
        assert vad.config.sample_rate == 16000
        assert vad.aggressiveness == 2

    def test_init_aggressiveness_0_falls_back_to_config(self):
        """aggressiveness=0 is falsy, so `or` falls through to config default."""
        cfg = _make_config(vad_aggressiveness=2)
        vad = VoiceActivityDetector(config=cfg, aggressiveness=0)
        # 0 or 2 => 2 (Python falsy-or behavior)
        assert vad.aggressiveness == 2

    def test_init_aggressiveness_1(self):
        vad = VoiceActivityDetector(config=_make_config(), aggressiveness=1)
        assert vad.aggressiveness == 1

    def test_init_aggressiveness_2(self):
        vad = VoiceActivityDetector(config=_make_config(), aggressiveness=2)
        assert vad.aggressiveness == 2

    def test_init_aggressiveness_3(self):
        vad = VoiceActivityDetector(config=_make_config(), aggressiveness=3)
        assert vad.aggressiveness == 3

    def test_init_aggressiveness_negative_raises(self):
        with pytest.raises(ValueError, match="aggressiveness must be 0-3"):
            VoiceActivityDetector(config=_make_config(), aggressiveness=-1)

    def test_init_aggressiveness_4_raises(self):
        with pytest.raises(ValueError, match="aggressiveness must be 0-3"):
            VoiceActivityDetector(config=_make_config(), aggressiveness=4)

    def test_init_aggressiveness_100_raises(self):
        with pytest.raises(ValueError, match="aggressiveness must be 0-3"):
            VoiceActivityDetector(config=_make_config(), aggressiveness=100)

    def test_init_creates_webrtcvad_instance(self):
        mock_webrtcvad.Vad.reset_mock()
        VoiceActivityDetector(config=_make_config(), aggressiveness=2)
        mock_webrtcvad.Vad.assert_called_with(2)

    def test_init_frame_size_calculation(self):
        """frame_size = sample_rate * frame_duration_ms / 1000"""
        vad = VoiceActivityDetector(config=_make_config(sample_rate=16000))
        # 16000 * 30 / 1000 = 480
        assert vad._frame_size == 480

    def test_init_frame_size_8khz(self):
        vad = VoiceActivityDetector(config=_make_config(sample_rate=8000))
        # 8000 * 30 / 1000 = 240
        assert vad._frame_size == 240

    def test_init_silence_threshold_frames(self):
        """silence_threshold_frames = silence_threshold_ms / frame_duration_ms"""
        vad = VoiceActivityDetector(config=_make_config(silence_threshold_ms=600))
        # 600 / 30 = 20 frames
        assert vad._silence_threshold_frames == 20

    def test_init_not_triggered(self, vad):
        assert not vad._triggered

    def test_init_empty_utterance_buffer(self, vad):
        assert vad._utterance_buffer == []

    def test_init_silence_frames_zero(self, vad):
        assert vad._silence_frames == 0

    def test_init_ring_buffer_empty(self, vad):
        assert len(vad._ring_buffer) == 0

    def test_valid_frame_durations_constant(self):
        assert VoiceActivityDetector.VALID_FRAME_DURATIONS_MS == (10, 20, 30)


class TestSplitIntoFrames:
    """Tests for VoiceActivityDetector._split_into_frames"""

    def test_exact_single_frame(self, vad):
        data = _silence(480)  # exactly 1 frame at 16kHz/30ms
        frames = vad._split_into_frames(data)
        assert len(frames) == 1
        assert len(frames[0]) == 480

    def test_exact_multiple_frames(self, vad):
        data = _silence(960)  # exactly 2 frames
        frames = vad._split_into_frames(data)
        assert len(frames) == 2

    def test_trailing_samples_ignored(self, vad):
        data = _silence(500)  # 480 + 20 leftover
        frames = vad._split_into_frames(data)
        assert len(frames) == 1

    def test_too_short_returns_empty(self, vad):
        data = _silence(100)  # less than one frame
        frames = vad._split_into_frames(data)
        assert frames == []

    def test_empty_array(self, vad):
        data = np.array([], dtype=np.int16)
        frames = vad._split_into_frames(data)
        assert frames == []

    def test_three_frames(self, vad):
        data = _silence(1440)  # 480 * 3
        frames = vad._split_into_frames(data)
        assert len(frames) == 3

    def test_frame_data_preserved(self, vad):
        """Content of frames matches the original slices."""
        data = np.arange(960, dtype=np.int16)
        frames = vad._split_into_frames(data)
        np.testing.assert_array_equal(frames[0], data[:480])
        np.testing.assert_array_equal(frames[1], data[480:960])


class TestIsSpeech:
    """Tests for VoiceActivityDetector.is_speech"""

    def test_all_frames_speech(self, vad):
        """Majority rule: all speech -> True."""
        vad._vad.is_speech.return_value = True
        data = _silence(960)  # 2 frames
        assert vad.is_speech(data) is True

    def test_no_frames_speech(self, vad):
        """All frames silence -> False."""
        vad._vad.is_speech.return_value = False
        data = _silence(960)
        assert vad.is_speech(data) is False

    def test_majority_speech(self, vad):
        """More than half speech -> True."""
        # 3 frames: True, True, False -> 2/3 > 0.5 -> True
        vad._vad.is_speech.side_effect = [True, True, False]
        data = _silence(1440)
        assert vad.is_speech(data) is True

    def test_minority_speech(self, vad):
        """Less than half speech -> False."""
        # 3 frames: True, False, False -> 1/3 < 0.5 -> False
        vad._vad.is_speech.side_effect = [True, False, False]
        data = _silence(1440)
        assert vad.is_speech(data) is False

    def test_exactly_half_speech(self, vad):
        """Exactly half is NOT > half, so False."""
        # 2 frames: True, False -> 1/2 == 0.5, not > 0.5 -> False
        vad._vad.is_speech.side_effect = [True, False]
        data = _silence(960)
        assert vad.is_speech(data) is False

    def test_data_too_short_returns_false(self, vad):
        """If no frames can be split, return False."""
        data = _silence(100)
        assert vad.is_speech(data) is False

    def test_float_data_converted_to_int16(self, vad):
        """Float data gets converted via * 32767."""
        vad._vad.is_speech.return_value = True
        data = np.zeros(480, dtype=np.float64)
        result = vad.is_speech(data)
        assert result is True

    def test_vad_exception_skips_frame(self, vad):
        """If underlying VAD raises, that frame is skipped."""
        vad._vad.is_speech.side_effect = [Exception("bad frame"), True]
        data = _silence(960)
        # 1 valid frame out of 2, that 1 is speech -> 1/1 > 0.5?
        # No: 1 speech out of 2 total frames (exception in loop still increments)
        # Actually: speech_frames=1, len(frames)=2 -> 1 > 1 => False
        assert vad.is_speech(data) is False

    def test_vad_exception_all_frames_still_returns_false(self, vad):
        """All frames throw -> 0 speech frames -> False."""
        vad._vad.is_speech.side_effect = Exception("broken")
        data = _silence(960)
        assert vad.is_speech(data) is False

    def test_single_frame_speech(self, vad):
        """1 frame that is speech -> 1 > 0.5 -> True."""
        vad._vad.is_speech.return_value = True
        data = _silence(480)
        assert vad.is_speech(data) is True

    def test_single_frame_silence(self, vad):
        vad._vad.is_speech.return_value = False
        data = _silence(480)
        assert vad.is_speech(data) is False


class TestProcessChunk:
    """Tests for VoiceActivityDetector.process_chunk"""

    def test_sets_is_speech_true(self, vad):
        vad._vad.is_speech.return_value = True
        chunk = _make_chunk(_silence(480))
        result = vad.process_chunk(chunk)
        assert result.is_speech is True

    def test_sets_is_speech_false(self, vad):
        vad._vad.is_speech.return_value = False
        chunk = _make_chunk(_silence(480))
        result = vad.process_chunk(chunk)
        assert result.is_speech is False

    def test_returns_same_chunk(self, vad):
        vad._vad.is_speech.return_value = False
        chunk = _make_chunk(_silence(480))
        result = vad.process_chunk(chunk)
        assert result is chunk

    def test_updates_ring_buffer(self, vad):
        vad._vad.is_speech.return_value = False
        chunk = _make_chunk(_silence(480))
        vad.process_chunk(chunk)
        assert len(vad._ring_buffer) == 1


class TestUpdateState:
    """Tests for VoiceActivityDetector._update_state (state machine)."""

    def test_not_triggered_adds_to_ring_buffer(self, vad):
        vad._update_state(_silence(480), False)
        assert len(vad._ring_buffer) == 1

    def test_triggers_on_first_voiced_frame_with_empty_buffer(self, vad):
        """With empty ring buffer, a single voiced frame triggers (1 > 0.9*1)."""
        vad._update_state(_silence(480), True)
        assert vad._triggered is True

    def test_triggers_after_enough_voiced_in_full_buffer(self, vad):
        """When buffer is mostly silence, need >90% voiced to trigger."""
        # Fill with 29 silence frames (not enough to be >90% of 30)
        for _ in range(29):
            vad._update_state(_silence(480), False)
        assert vad._triggered is False
        # Now add voiced frames until >90% of buffer is voiced
        # Buffer is maxlen=30. We need >27 voiced out of 30.
        # Currently 29 silence + the next voiced = 30 items, 1/30 < 90%
        # Keep adding voiced frames to push out silence
        for _ in range(28):
            vad._update_state(_silence(480), True)
        # Now buffer should have 2 silence + 28 voiced = 30, 28/30=0.933 > 0.9
        assert vad._triggered is True

    def test_does_not_trigger_with_insufficient_voiced(self, vad):
        """If the ring buffer is pre-filled with silence, 50% voiced won't trigger.

        Note: the trigger condition is `num_voiced > 0.9 * len(ring_buffer)`.
        On an empty buffer, even a single voiced frame would trigger (1 > 0.9*1).
        So we first fill the buffer completely with silence, then feed alternating.
        """
        # Fill ring buffer completely with silence first
        for _ in range(30):
            vad._update_state(_silence(480), False)
        assert vad._triggered is False

        # Now feed alternating speech/silence -- oldest silence frames get pushed out
        # but the buffer stays at ~50% voiced max, never reaching 90%.
        for i in range(30):
            vad._update_state(_silence(480), i % 2 == 0)
        assert vad._triggered is False

    def test_trigger_clears_ring_buffer(self, vad):
        """When trigger fires, ring buffer is cleared. But subsequent voiced
        frames go through the 'else' (triggered) branch which still appends
        to ring_buffer at the top of _update_state."""
        # Single voiced frame on empty buffer triggers immediately
        vad._update_state(_silence(480), True)
        # Ring buffer was cleared during trigger
        # But note: the append happens BEFORE the if/else, so:
        # Step: append (ring=[1 item]) -> trigger fires -> ring cleared -> utterance gets 1 item
        # After clear, ring is empty. No further appends in this call.
        # Wait - let's verify: the ring_buffer.clear() is in the trigger block.
        # The append already happened before the trigger. But clear() empties it after.
        assert len(vad._ring_buffer) == 0

    def test_trigger_copies_ring_to_utterance_buffer(self, vad):
        """On trigger, all ring buffer contents are moved to utterance buffer."""
        vad._update_state(_silence(480), True)
        # Ring buffer had 1 item when trigger fired -> copied to utterance
        assert len(vad._utterance_buffer) == 1

    def test_triggered_accumulates_audio(self, vad):
        # Force trigger
        vad._triggered = True
        vad._update_state(_silence(480), True)
        assert len(vad._utterance_buffer) == 1

    def test_triggered_silence_increments_counter(self, vad):
        vad._triggered = True
        vad._update_state(_silence(480), False)
        assert vad._silence_frames == 1

    def test_triggered_speech_resets_silence_counter(self, vad):
        vad._triggered = True
        vad._silence_frames = 5
        vad._update_state(_silence(480), True)
        assert vad._silence_frames == 0

    def test_triggered_multiple_silences(self, vad):
        vad._triggered = True
        for _ in range(10):
            vad._update_state(_silence(480), False)
        assert vad._silence_frames == 10

    def test_ring_buffer_respects_maxlen(self, vad):
        """Ring buffer maxlen is 30; adding 35 items should keep only 30."""
        for _ in range(35):
            vad._update_state(_silence(480), False)
        assert len(vad._ring_buffer) == 30


class TestUtteranceEnded:
    """Tests for VoiceActivityDetector.utterance_ended"""

    def test_not_triggered_returns_false(self, vad):
        assert vad.utterance_ended() is False

    def test_triggered_no_silence_returns_false(self, vad):
        vad._triggered = True
        vad._silence_frames = 0
        assert vad.utterance_ended() is False

    def test_triggered_below_threshold_returns_false(self, vad):
        vad._triggered = True
        vad._silence_frames = vad._silence_threshold_frames - 1
        assert vad.utterance_ended() is False

    def test_triggered_at_threshold_returns_true(self, vad):
        vad._triggered = True
        vad._silence_frames = vad._silence_threshold_frames
        assert vad.utterance_ended() is True

    def test_triggered_above_threshold_returns_true(self, vad):
        vad._triggered = True
        vad._silence_frames = vad._silence_threshold_frames + 10
        assert vad.utterance_ended() is True


class TestGetUtterance:
    """Tests for VoiceActivityDetector.get_utterance"""

    def test_empty_buffer_returns_empty(self, vad):
        result = vad.get_utterance()
        assert len(result) == 0
        assert result.dtype == np.int16

    def test_single_buffer(self, vad):
        vad._utterance_buffer = [np.ones(480, dtype=np.int16)]
        result = vad.get_utterance()
        assert len(result) == 480

    def test_multiple_buffers_concatenated(self, vad):
        vad._utterance_buffer = [
            np.ones(480, dtype=np.int16),
            np.ones(480, dtype=np.int16) * 2,
        ]
        result = vad.get_utterance()
        assert len(result) == 960
        assert result[0] == 1
        assert result[480] == 2

    def test_resets_triggered(self, vad):
        vad._triggered = True
        vad._utterance_buffer = [_silence(480)]
        vad.get_utterance()
        assert vad._triggered is False

    def test_resets_utterance_buffer(self, vad):
        vad._utterance_buffer = [_silence(480)]
        vad.get_utterance()
        assert vad._utterance_buffer == []

    def test_resets_silence_frames(self, vad):
        vad._silence_frames = 10
        vad._utterance_buffer = [_silence(480)]
        vad.get_utterance()
        assert vad._silence_frames == 0

    def test_resets_ring_buffer(self, vad):
        vad._ring_buffer.append((_silence(480), True))
        vad._utterance_buffer = [_silence(480)]
        vad.get_utterance()
        assert len(vad._ring_buffer) == 0


class TestReset:
    """Tests for VoiceActivityDetector.reset"""

    def test_clears_triggered(self, vad):
        vad._triggered = True
        vad.reset()
        assert vad._triggered is False

    def test_clears_utterance_buffer(self, vad):
        vad._utterance_buffer = [_silence(480)]
        vad.reset()
        assert vad._utterance_buffer == []

    def test_clears_silence_frames(self, vad):
        vad._silence_frames = 15
        vad.reset()
        assert vad._silence_frames == 0

    def test_clears_ring_buffer(self, vad):
        vad._ring_buffer.append((_silence(480), True))
        vad.reset()
        assert len(vad._ring_buffer) == 0


class TestVadProperties:
    """Tests for VAD properties."""

    def test_is_triggered_false_initially(self, vad):
        assert vad.is_triggered is False

    def test_is_triggered_true_after_set(self, vad):
        vad._triggered = True
        assert vad.is_triggered is True

    def test_utterance_duration_ms_empty(self, vad):
        assert vad.utterance_duration_ms == 0.0

    def test_utterance_duration_ms_single_buffer(self, vad):
        # 480 samples at 16000 Hz = 30ms
        vad._utterance_buffer = [np.zeros(480, dtype=np.int16)]
        assert vad.utterance_duration_ms == pytest.approx(30.0)

    def test_utterance_duration_ms_multiple_buffers(self, vad):
        # 2 * 480 = 960 samples at 16000 Hz = 60ms
        vad._utterance_buffer = [
            np.zeros(480, dtype=np.int16),
            np.zeros(480, dtype=np.int16),
        ]
        assert vad.utterance_duration_ms == pytest.approx(60.0)

    def test_utterance_duration_ms_large(self, vad):
        # 16000 samples = 1 second = 1000ms
        vad._utterance_buffer = [np.zeros(16000, dtype=np.int16)]
        assert vad.utterance_duration_ms == pytest.approx(1000.0)


class TestVadFilterStream:
    """Tests for vad_filter_stream function."""

    def test_yields_speech_chunks(self):
        config = _make_config()
        # Create chunks with enough samples for at least one frame
        speech_chunk = _make_chunk(_silence(480))
        silence_chunk = _make_chunk(_silence(480))

        with patch("voice.audio.vad.VoiceActivityDetector") as MockVAD:
            instance = MockVAD.return_value

            # process_chunk: first is speech, second is not
            def process_side_effect(chunk):
                return chunk

            instance.process_chunk.side_effect = [speech_chunk, silence_chunk]

            # Set up the is_speech attrs
            speech_chunk.is_speech = True
            silence_chunk.is_speech = False

            stream = iter([speech_chunk, silence_chunk])
            result = list(vad_filter_stream(stream, config))
            assert len(result) == 1
            assert result[0] is speech_chunk

    def test_empty_stream(self):
        config = _make_config()
        with patch("voice.audio.vad.VoiceActivityDetector"):
            result = list(vad_filter_stream(iter([]), config))
            assert result == []

    def test_all_speech(self):
        config = _make_config()
        chunks = [_make_chunk(_silence(480)) for _ in range(3)]

        with patch("voice.audio.vad.VoiceActivityDetector") as MockVAD:
            instance = MockVAD.return_value
            for c in chunks:
                c.is_speech = True
            instance.process_chunk.side_effect = chunks

            result = list(vad_filter_stream(iter(chunks), config))
            assert len(result) == 3

    def test_all_silence(self):
        config = _make_config()
        chunks = [_make_chunk(_silence(480)) for _ in range(3)]

        with patch("voice.audio.vad.VoiceActivityDetector") as MockVAD:
            instance = MockVAD.return_value
            for c in chunks:
                c.is_speech = False
            instance.process_chunk.side_effect = chunks

            result = list(vad_filter_stream(iter(chunks), config))
            assert len(result) == 0


class TestCollectUtterance:
    """Tests for collect_utterance function."""

    def test_collects_until_utterance_ended(self):
        config = _make_config()
        chunks = [_make_chunk(_silence(480)) for _ in range(5)]

        with patch("voice.audio.vad.VoiceActivityDetector") as MockVAD:
            instance = MockVAD.return_value
            instance.process_chunk.side_effect = lambda c: c
            # utterance_ended returns False 4 times, then True
            instance.utterance_ended.side_effect = [False, False, False, False, True]
            instance.get_utterance.return_value = np.zeros(2400, dtype=np.int16)

            audio, duration = collect_utterance(iter(chunks), config)
            assert len(audio) == 2400
            assert duration == pytest.approx(2400 / 16000)

    def test_collects_until_max_duration(self):
        config = _make_config()
        # Each chunk is 480 samples. max_duration=0.03s -> max_samples = 480
        chunks = [_make_chunk(_silence(480)) for _ in range(5)]

        with patch("voice.audio.vad.VoiceActivityDetector") as MockVAD:
            instance = MockVAD.return_value
            instance.process_chunk.side_effect = lambda c: c
            instance.utterance_ended.return_value = False
            instance.get_utterance.return_value = np.zeros(480, dtype=np.int16)

            audio, duration = collect_utterance(
                iter(chunks), config, max_duration_seconds=0.03
            )
            assert len(audio) == 480

    def test_stream_ends_without_utterance(self):
        config = _make_config()
        chunks = [_make_chunk(_silence(480)) for _ in range(2)]

        with patch("voice.audio.vad.VoiceActivityDetector") as MockVAD:
            instance = MockVAD.return_value
            instance.process_chunk.side_effect = lambda c: c
            instance.utterance_ended.return_value = False
            instance.get_utterance.return_value = np.array([], dtype=np.int16)

            audio, duration = collect_utterance(iter(chunks), config)
            assert len(audio) == 0
            assert duration == 0.0

    def test_empty_stream(self):
        config = _make_config()
        with patch("voice.audio.vad.VoiceActivityDetector") as MockVAD:
            instance = MockVAD.return_value
            instance.get_utterance.return_value = np.array([], dtype=np.int16)

            audio, duration = collect_utterance(iter([]), config)
            assert len(audio) == 0
            assert duration == 0.0


class TestVadIntegrationScenario:
    """End-to-end state machine scenarios."""

    def test_full_speech_lifecycle(self, vad):
        """Simulate: silence -> speech triggers -> silence ends utterance."""
        vad._vad.is_speech.return_value = True

        # Feed 30 voiced frames to trigger
        for _ in range(30):
            chunk = _make_chunk(_silence(480))
            vad.process_chunk(chunk)
        assert vad.is_triggered is True

        # Feed some more speech frames
        for _ in range(5):
            chunk = _make_chunk(_silence(480))
            vad.process_chunk(chunk)

        # Now feed silence until utterance ends
        vad._vad.is_speech.return_value = False
        threshold = vad._silence_threshold_frames
        for _ in range(threshold):
            chunk = _make_chunk(_silence(480))
            vad.process_chunk(chunk)

        assert vad.utterance_ended() is True

        utterance = vad.get_utterance()
        assert len(utterance) > 0
        assert vad.is_triggered is False

    def test_reset_during_speech(self, vad):
        """Reset mid-speech clears everything."""
        vad._triggered = True
        vad._utterance_buffer = [_silence(480)] * 10
        vad._silence_frames = 5
        vad._ring_buffer.append((_silence(480), True))

        vad.reset()

        assert vad.is_triggered is False
        assert vad._utterance_buffer == []
        assert vad._silence_frames == 0
        assert len(vad._ring_buffer) == 0


# ===================================================================
#  PART 2 – Language Detector Tests
# ===================================================================


class TestLanguageInfoDataclass:
    """Tests for LanguageInfo dataclass and its properties."""

    def test_is_telugu(self):
        info = LanguageInfo("te", 0.9, 0.1, 0.0, True, False, 0.9)
        assert info.is_telugu is True
        assert info.is_english is False
        assert info.is_mixed is False

    def test_is_english(self):
        info = LanguageInfo("en", 0.0, 1.0, 0.0, False, False, 1.0)
        assert info.is_english is True
        assert info.is_telugu is False
        assert info.is_mixed is False

    def test_is_mixed(self):
        info = LanguageInfo("mixed", 0.4, 0.4, 0.8, True, False, 0.8)
        assert info.is_mixed is True
        assert info.is_telugu is False
        assert info.is_english is False

    def test_has_telugu_script(self):
        info = LanguageInfo("te", 1.0, 0.0, 0.0, True, False, 1.0)
        assert info.has_telugu_script is True

    def test_has_romanized_telugu(self):
        info = LanguageInfo("te", 0.5, 0.5, 0.0, False, True, 0.5)
        assert info.has_romanized_telugu is True

    def test_confidence_stored(self):
        info = LanguageInfo("en", 0.0, 1.0, 0.0, False, False, 0.85)
        assert info.confidence == 0.85

    def test_ratios_stored(self):
        info = LanguageInfo("mixed", 0.3, 0.5, 0.7, True, True, 0.7)
        assert info.telugu_ratio == 0.3
        assert info.english_ratio == 0.5
        assert info.mixed_ratio == 0.7


class TestDetectLanguageEmpty:
    """Tests for detect_language with empty/whitespace input."""

    def test_empty_string(self):
        result = detect_language("")
        assert result.primary_language == "en"
        assert result.confidence == 0.0
        assert result.telugu_ratio == 0.0
        assert result.english_ratio == 0.0

    def test_none_input(self):
        result = detect_language(None)
        assert result.primary_language == "en"
        assert result.confidence == 0.0

    def test_whitespace_only(self):
        result = detect_language("   \t\n  ")
        assert result.primary_language == "en"
        assert result.confidence == 0.0

    def test_single_space(self):
        result = detect_language(" ")
        assert result.primary_language == "en"
        assert result.confidence == 0.0


class TestDetectLanguagePureEnglish:
    """Tests for detect_language with pure English text."""

    def test_simple_english(self):
        result = detect_language("Hello world this is a test")
        assert result.primary_language == "en"
        assert result.has_telugu_script is False

    def test_english_sentence(self):
        result = detect_language("The quick brown fox jumps over the lazy dog")
        assert result.is_english is True
        assert result.english_ratio > 0.0

    def test_english_no_telugu_script(self):
        result = detect_language("Python programming language")
        assert result.has_telugu_script is False
        assert result.has_romanized_telugu is False

    def test_english_confidence_positive(self):
        result = detect_language("I have a very important meeting today")
        assert result.confidence > 0.0

    def test_english_mixed_ratio_zero(self):
        result = detect_language("Good morning everyone")
        assert result.mixed_ratio == 0.0

    def test_english_with_numbers(self):
        result = detect_language("I have 3 cats and 2 dogs")
        assert result.primary_language == "en"

    def test_english_with_punctuation(self):
        result = detect_language("Hello! How are you? I'm fine, thanks.")
        assert result.primary_language == "en"


class TestDetectLanguagePureTelugu:
    """Tests for detect_language with pure Telugu script text."""

    def test_telugu_script(self):
        result = detect_language(TELUGU_HELLO)
        assert result.has_telugu_script is True
        assert result.telugu_ratio > 0.0

    def test_telugu_primary_language(self):
        result = detect_language(TELUGU_SENTENCE)
        assert result.primary_language == "te"

    def test_telugu_high_ratio(self):
        result = detect_language(TELUGU_WORD)
        assert result.telugu_ratio > 0.7

    def test_telugu_confidence(self):
        result = detect_language(TELUGU_SENTENCE)
        assert result.confidence > 0.0

    def test_telugu_single_char(self):
        result = detect_language("\u0C05")  # Telugu 'a'
        assert result.has_telugu_script is True

    def test_telugu_range_boundary_low_non_alpha(self):
        """U+0C00 is a combining mark, not alpha -> not counted as Telugu script."""
        result = detect_language(chr(0x0C00))
        assert result.has_telugu_script is False

    def test_telugu_range_first_alpha(self):
        """U+0C05 (TELUGU LETTER A) is the first alphabetic Telugu char."""
        result = detect_language(chr(0x0C05))
        assert result.has_telugu_script is True

    def test_telugu_range_boundary_high_non_alpha(self):
        """U+0C7F is TELUGU SIGN TUUMU, not alpha -> not counted."""
        result = detect_language(chr(0x0C7F))
        assert result.has_telugu_script is False

    def test_telugu_range_vowels(self):
        """Telugu vowels in the range should be detected."""
        # U+0C05 to U+0C14 are Telugu vowels
        for cp in range(0x0C05, 0x0C15):
            c = chr(cp)
            if c.isalpha():
                result = detect_language(c)
                assert result.has_telugu_script is True, f"Failed for U+{cp:04X}"


class TestDetectLanguageRomanizedTelugu:
    """Tests for detect_language with Romanized Telugu (Latin script)."""

    def test_romanized_telugu_detected(self):
        result = detect_language("nenu vellu chesthanu")
        assert result.has_romanized_telugu is True

    def test_romanized_telugu_primary_te(self):
        # Many romanized words, no English stop words
        result = detect_language("nenu chesthanu amma nanna cheppandi idi")
        assert result.primary_language == "te"

    def test_romanized_needs_at_least_two_matches(self):
        """has_romanized_telugu requires >= 2 pattern matches."""
        # "nenu" alone = 1 match -> not enough
        result = detect_language("nenu is here")
        assert result.has_romanized_telugu is False

    def test_romanized_two_matches(self):
        result = detect_language("nenu meeru")
        assert result.has_romanized_telugu is True

    def test_romanized_family_words(self):
        result = detect_language("anna akka amma nanna thammudu chelli")
        assert result.has_romanized_telugu is True

    def test_romanized_time_words(self):
        result = detect_language("ippudu appudu mundu taruvatha")
        assert result.has_romanized_telugu is True

    def test_romanized_mixed_with_english_stops(self):
        """Romanized Telugu + many English stop words -> 'mixed'."""
        result = detect_language(
            "nenu meeru the is a very good undi ippudu it was being have"
        )
        # Many English stop words + romanized Telugu -> mixed or depends on ratio
        assert result.has_romanized_telugu is True


class TestDetectLanguageMixed:
    """Tests for detect_language with mixed Telugu-English text."""

    def test_mixed_script_and_english(self):
        """Telugu script + English text."""
        text = TELUGU_WORD + " is a language"
        result = detect_language(text)
        assert result.has_telugu_script is True
        assert result.english_ratio > 0.0

    def test_mixed_produces_mixed_or_telugu(self):
        """When both ratios are significant, result is mixed."""
        # Build text with ~50-50 Telugu/English characters
        text = TELUGU_WORD + " programming is fun " + TELUGU_HELLO
        result = detect_language(text)
        assert result.primary_language in ("mixed", "te")

    def test_mixed_telugu_dominant(self):
        """Mostly Telugu script with a bit of English."""
        text = TELUGU_SENTENCE + " ok"
        result = detect_language(text)
        # Telugu chars >> English chars
        assert result.primary_language == "te"


class TestDetectLanguageConfidence:
    """Tests for confidence value correctness."""

    def test_confidence_between_0_and_1(self):
        for text in [
            "hello world",
            TELUGU_SENTENCE,
            "nenu meeru chesthanu",
            "",
            "   ",
        ]:
            result = detect_language(text)
            assert 0.0 <= result.confidence <= 1.0, f"Bad confidence for: {text!r}"

    def test_empty_confidence_zero(self):
        assert detect_language("").confidence == 0.0

    def test_pure_english_confidence_positive(self):
        result = detect_language("The cat sat on the mat")
        assert result.confidence > 0.0

    def test_pure_telugu_confidence_positive(self):
        result = detect_language(TELUGU_SENTENCE)
        assert result.confidence > 0.0

    def test_ratios_are_rounded(self):
        """Ratios should have at most 3 decimal places."""
        result = detect_language("hello nenu meeru chesthanu world test abc")
        for val in [
            result.telugu_ratio,
            result.english_ratio,
            result.mixed_ratio,
            result.confidence,
        ]:
            assert val == round(val, 3)


class TestTeluguRangeConstant:
    """Tests for TELUGU_RANGE constant."""

    def test_tuple(self):
        assert isinstance(TELUGU_RANGE, tuple)

    def test_length(self):
        assert len(TELUGU_RANGE) == 2

    def test_values(self):
        assert TELUGU_RANGE == (0x0C00, 0x0C7F)

    def test_range_contains_telugu_chars(self):
        for code in [0x0C05, 0x0C15, 0x0C28, 0x0C35, 0x0C7F]:
            assert TELUGU_RANGE[0] <= code <= TELUGU_RANGE[1]


class TestRomanizedPatterns:
    """Tests for TELUGU_ROMANIZED_PATTERNS."""

    def test_is_list(self):
        assert isinstance(TELUGU_ROMANIZED_PATTERNS, list)

    def test_eight_patterns(self):
        assert len(TELUGU_ROMANIZED_PATTERNS) == 8

    def test_patterns_are_strings(self):
        for p in TELUGU_ROMANIZED_PATTERNS:
            assert isinstance(p, str)

    def test_patterns_compile(self):
        import re

        for p in TELUGU_ROMANIZED_PATTERNS:
            re.compile(p)  # Should not raise


class TestEnglishStopWords:
    """Tests for ENGLISH_STOP_WORDS."""

    def test_is_set(self):
        assert isinstance(ENGLISH_STOP_WORDS, set)

    def test_contains_common_words(self):
        for word in ["the", "is", "a", "and", "or", "not", "i", "you"]:
            assert word in ENGLISH_STOP_WORDS

    def test_lowercase(self):
        for word in ENGLISH_STOP_WORDS:
            assert word == word.lower()


class TestSplitByLanguage:
    """Tests for split_by_language function."""

    def test_empty_string(self):
        assert split_by_language("") == []

    def test_pure_english(self):
        segments = split_by_language("hello world")
        assert len(segments) >= 1
        assert all(lang == "en" for _, lang in segments)

    def test_pure_telugu(self):
        segments = split_by_language(TELUGU_SENTENCE)
        assert len(segments) >= 1
        assert all(lang == "te" for _, lang in segments)

    def test_mixed_produces_multiple_segments(self):
        text = "hello " + TELUGU_WORD + " world"
        segments = split_by_language(text)
        languages = [lang for _, lang in segments]
        assert "en" in languages
        assert "te" in languages

    def test_segment_text_not_empty(self):
        text = "hello " + TELUGU_WORD
        segments = split_by_language(text)
        for seg_text, _ in segments:
            assert len(seg_text.strip()) > 0

    def test_single_telugu_char(self):
        segments = split_by_language("\u0C05")
        assert len(segments) == 1
        assert segments[0][1] == "te"

    def test_single_english_char(self):
        segments = split_by_language("a")
        assert len(segments) == 1
        assert segments[0][1] == "en"

    def test_numbers_only(self):
        # Numbers are not alpha, so no language segments produced
        segments = split_by_language("12345")
        assert segments == []

    def test_preserves_content(self):
        text = "hello"
        segments = split_by_language(text)
        combined = "".join(seg for seg, _ in segments)
        assert combined.strip() == text


class TestIsCodeSwitched:
    """Tests for is_code_switched function."""

    def test_pure_english_not_switched(self):
        assert is_code_switched("Hello, how are you today?") is False

    def test_pure_telugu_not_switched(self):
        assert is_code_switched(TELUGU_SENTENCE) is False

    def test_mixed_script_is_switched(self):
        text = TELUGU_WORD + " is a beautiful language and " + TELUGU_HELLO
        result = is_code_switched(text)
        # Should detect mixed or Telugu+English > 0.1
        assert result is True

    def test_empty_string_not_switched(self):
        assert is_code_switched("") is False

    def test_telugu_script_with_english_words(self):
        """Telugu script + enough English chars -> code-switched."""
        text = TELUGU_SENTENCE + " programming and coding together"
        result = is_code_switched(text)
        assert result is True

    def test_romanized_telugu_with_english(self):
        """Romanized Telugu + English stop words might detect as mixed."""
        text = "nenu meeru the very good chesthanu ippudu appudu it is a was being have"
        result = is_code_switched(text)
        # With many romanized + English words, should be mixed
        assert result is True


class TestDetectLanguageEdgeCases:
    """Edge cases and boundary conditions."""

    def test_only_digits(self):
        result = detect_language("123456")
        assert result.primary_language == "en"
        assert result.telugu_ratio == 0.0

    def test_only_punctuation(self):
        result = detect_language("!@#$%^&*()")
        assert result.primary_language == "en"

    def test_single_english_word(self):
        result = detect_language("hello")
        assert result.primary_language == "en"

    def test_unicode_non_telugu_non_english(self):
        """Non-Telugu, non-ASCII chars (e.g., Chinese) treated as 'other'."""
        result = detect_language("\u4e16\u754c")  # 世界 (Chinese)
        # These are alpha but neither Telugu nor ASCII
        assert result.primary_language == "en"  # default fallback

    def test_very_long_english_text(self):
        text = " ".join(["the quick brown fox"] * 100)
        result = detect_language(text)
        assert result.primary_language == "en"
        assert result.confidence > 0.0

    def test_very_long_telugu_text(self):
        text = " ".join([TELUGU_SENTENCE] * 50)
        result = detect_language(text)
        assert result.primary_language == "te"
        assert result.confidence > 0.0

    def test_romanized_case_insensitive(self):
        """Pattern matching should be case insensitive."""
        result1 = detect_language("Nenu Meeru chesthanu amma")
        result2 = detect_language("nenu meeru CHESTHANU AMMA")
        assert result1.has_romanized_telugu == result2.has_romanized_telugu

    def test_tabs_and_newlines(self):
        result = detect_language("hello\tworld\nnenu\tmeeru")
        assert result.primary_language in ("en", "te", "mixed")

    def test_mixed_script_ratios_sum_reasonable(self):
        """Telugu + English ratios should be <= 1.0 before romanized adjustment."""
        result = detect_language(TELUGU_WORD + " hello world test")
        assert (
            result.telugu_ratio + result.english_ratio <= 1.01
        )  # small float tolerance


class TestVadStateMachineEdgeCases:
    """Additional VAD edge case tests."""

    def test_double_get_utterance_second_empty(self, vad):
        """Calling get_utterance twice - second should be empty."""
        vad._utterance_buffer = [_silence(480)]
        first = vad.get_utterance()
        second = vad.get_utterance()
        assert len(first) == 480
        assert len(second) == 0

    def test_utterance_ended_after_reset(self, vad):
        """After reset, utterance_ended should be False."""
        vad._triggered = True
        vad._silence_frames = 100
        vad.reset()
        assert vad.utterance_ended() is False

    def test_process_chunk_with_float_data(self, vad):
        """Float audio data in chunk should be handled."""
        vad._vad.is_speech.return_value = True
        data = np.zeros(480, dtype=np.float32)
        chunk = _make_chunk(data)
        result = vad.process_chunk(chunk)
        assert result.is_speech is True

    def test_ring_buffer_maxlen(self, vad):
        assert vad._ring_buffer.maxlen == 30

    def test_silence_threshold_with_custom_config(self):
        """Custom silence_threshold_ms changes _silence_threshold_frames."""
        config = _make_config(silence_threshold_ms=900)
        vad = VoiceActivityDetector(config=config)
        # 900 / 30 = 30
        assert vad._silence_threshold_frames == 30

    def test_utterance_duration_increases(self, vad):
        """Duration grows as more audio is buffered."""
        vad._triggered = True
        vad._vad.is_speech.return_value = True

        durations = []
        for _ in range(5):
            vad._utterance_buffer.append(np.zeros(480, dtype=np.int16))
            durations.append(vad.utterance_duration_ms)

        # Each appends 480 samples = 30ms
        for i in range(1, len(durations)):
            assert durations[i] > durations[i - 1]
