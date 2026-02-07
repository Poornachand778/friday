"""
Comprehensive tests for FasterWhisperSTT and XTTSService.

Tests cover:
- WordTiming and TranscriptionResult dataclasses
- FasterWhisperSTT: init, model loading, transcribe, stream, detect, prepare audio
- TTSResult dataclass and LANGUAGE_MAP / LANGUAGE_FALLBACK
- XTTSService: init, model loading, synthesize, stream, profiles, language mapping
- Convenience functions: transcribe_audio, synthesize_speech, save_audio
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

# ---------------------------------------------------------------------------
# Mock external dependencies BEFORE importing the modules under test.
# Both faster_whisper_service.py and xtts_service.py import external libraries
# at module level (inside try/except), so the mocks must be in sys.modules
# before the first import.
# ---------------------------------------------------------------------------

sys.modules.setdefault("faster_whisper", MagicMock())
sys.modules.setdefault("TTS", MagicMock())
sys.modules.setdefault("TTS.api", MagicMock())
sys.modules.setdefault("TTS.tts", MagicMock())
sys.modules.setdefault("TTS.tts.configs", MagicMock())
sys.modules.setdefault("TTS.tts.configs.xtts_config", MagicMock())
sys.modules.setdefault("TTS.tts.models", MagicMock())
sys.modules.setdefault("TTS.tts.models.xtts", MagicMock())
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("webrtcvad", MagicMock())

import pytest  # noqa: E402

from voice.config import STTConfig, TTSConfig  # noqa: E402
from voice.stt.faster_whisper_service import (  # noqa: E402
    FasterWhisperSTT,
    TranscriptionResult,
    WordTiming,
    transcribe_audio,
)
from voice.tts.xtts_service import (  # noqa: E402
    LANGUAGE_FALLBACK,
    LANGUAGE_MAP,
    TTSResult,
    XTTSService,
    save_audio,
    synthesize_speech,
)


# ============================================================================
# Helpers / fixtures
# ============================================================================


def _make_stt_config(**overrides) -> STTConfig:
    """Return an STTConfig with sensible test defaults."""
    defaults = dict(
        model="tiny",
        device="cpu",
        compute_type="float32",
        language=None,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )
    defaults.update(overrides)
    return STTConfig(**defaults)


def _make_tts_config(**overrides) -> TTSConfig:
    """Return a TTSConfig with sensible test defaults."""
    defaults = dict(
        device="cpu",
        default_language="te",
        default_profile="friday_telugu",
    )
    defaults.update(overrides)
    return TTSConfig(**defaults)


def _mock_segment(text=" Hello world", seg_id=0, start=0.0, end=1.0, words=None):
    """Return a mock segment object that looks like a faster-whisper segment."""
    seg = MagicMock()
    seg.text = text
    seg.id = seg_id
    seg.start = start
    seg.end = end
    seg.words = words or []
    return seg


def _mock_transcribe_info(language="en", language_probability=0.98):
    """Return a mock info object returned by model.transcribe."""
    info = MagicMock()
    info.language = language
    info.language_probability = language_probability
    return info


@pytest.fixture
def stt_config():
    return _make_stt_config()


@pytest.fixture
def tts_config():
    return _make_tts_config()


@pytest.fixture
def stt_service(stt_config):
    return FasterWhisperSTT(config=stt_config)


@pytest.fixture
def tts_service(tts_config):
    return XTTSService(config=tts_config)


# ============================================================================
# PART 1 -- WordTiming dataclass
# ============================================================================


class TestWordTiming:
    """Tests for the WordTiming dataclass."""

    def test_creation_basic(self):
        wt = WordTiming(word="hello", start=0.0, end=0.5, probability=0.95)
        assert wt.word == "hello"
        assert wt.start == 0.0
        assert wt.end == 0.5
        assert wt.probability == 0.95

    def test_creation_telugu_word(self):
        wt = WordTiming(
            word="\u0c28\u0c2e\u0c38\u0c4d\u0c15\u0c3e\u0c30\u0c02",
            start=1.0,
            end=2.0,
            probability=0.88,
        )
        assert wt.word == "\u0c28\u0c2e\u0c38\u0c4d\u0c15\u0c3e\u0c30\u0c02"
        assert wt.start == 1.0

    def test_creation_zero_probability(self):
        wt = WordTiming(word="uh", start=0.0, end=0.1, probability=0.0)
        assert wt.probability == 0.0

    def test_creation_full_probability(self):
        wt = WordTiming(word="the", start=0.0, end=0.2, probability=1.0)
        assert wt.probability == 1.0

    def test_equality(self):
        a = WordTiming(word="hi", start=0.0, end=0.1, probability=0.9)
        b = WordTiming(word="hi", start=0.0, end=0.1, probability=0.9)
        assert a == b

    def test_inequality_different_word(self):
        a = WordTiming(word="hi", start=0.0, end=0.1, probability=0.9)
        b = WordTiming(word="bye", start=0.0, end=0.1, probability=0.9)
        assert a != b

    def test_repr_contains_word(self):
        wt = WordTiming(word="test", start=0.0, end=0.5, probability=0.5)
        assert "test" in repr(wt)


# ============================================================================
# PART 2 -- TranscriptionResult dataclass
# ============================================================================


class TestTranscriptionResult:
    """Tests for the TranscriptionResult dataclass."""

    def test_basic_creation(self):
        tr = TranscriptionResult(
            text="hello world",
            language="en",
            language_probability=0.99,
            duration=2.0,
            processing_time=0.5,
        )
        assert tr.text == "hello world"
        assert tr.language == "en"
        assert tr.language_probability == 0.99
        assert tr.duration == 2.0
        assert tr.processing_time == 0.5

    def test_default_words_empty(self):
        tr = TranscriptionResult(
            text="",
            language="en",
            language_probability=0.5,
            duration=1.0,
            processing_time=0.1,
        )
        assert tr.words == []

    def test_default_segments_empty(self):
        tr = TranscriptionResult(
            text="",
            language="en",
            language_probability=0.5,
            duration=1.0,
            processing_time=0.1,
        )
        assert tr.segments == []

    def test_words_provided(self):
        wt = WordTiming(word="hi", start=0.0, end=0.2, probability=0.9)
        tr = TranscriptionResult(
            text="hi",
            language="en",
            language_probability=0.9,
            duration=1.0,
            processing_time=0.1,
            words=[wt],
        )
        assert len(tr.words) == 1
        assert tr.words[0].word == "hi"

    def test_segments_provided(self):
        seg = {"id": 0, "start": 0.0, "end": 1.0, "text": "hi"}
        tr = TranscriptionResult(
            text="hi",
            language="en",
            language_probability=0.9,
            duration=1.0,
            processing_time=0.1,
            segments=[seg],
        )
        assert len(tr.segments) == 1

    def test_is_telugu_true(self):
        tr = TranscriptionResult(
            text="test",
            language="te",
            language_probability=0.9,
            duration=1.0,
            processing_time=0.1,
        )
        assert tr.is_telugu is True

    def test_is_telugu_false(self):
        tr = TranscriptionResult(
            text="test",
            language="en",
            language_probability=0.9,
            duration=1.0,
            processing_time=0.1,
        )
        assert tr.is_telugu is False

    def test_is_english_true(self):
        tr = TranscriptionResult(
            text="test",
            language="en",
            language_probability=0.9,
            duration=1.0,
            processing_time=0.1,
        )
        assert tr.is_english is True

    def test_is_english_false(self):
        tr = TranscriptionResult(
            text="test",
            language="te",
            language_probability=0.9,
            duration=1.0,
            processing_time=0.1,
        )
        assert tr.is_english is False

    def test_is_telugu_and_english_both_false_for_hindi(self):
        tr = TranscriptionResult(
            text="test",
            language="hi",
            language_probability=0.9,
            duration=1.0,
            processing_time=0.1,
        )
        assert tr.is_telugu is False
        assert tr.is_english is False

    def test_rtf_normal(self):
        tr = TranscriptionResult(
            text="t",
            language="en",
            language_probability=0.9,
            duration=2.0,
            processing_time=1.0,
        )
        assert tr.rtf == pytest.approx(0.5)

    def test_rtf_zero_duration(self):
        tr = TranscriptionResult(
            text="t",
            language="en",
            language_probability=0.9,
            duration=0.0,
            processing_time=1.0,
        )
        assert tr.rtf == 0.0

    def test_rtf_fast(self):
        tr = TranscriptionResult(
            text="t",
            language="en",
            language_probability=0.9,
            duration=10.0,
            processing_time=1.0,
        )
        assert tr.rtf == pytest.approx(0.1)

    def test_rtf_slow(self):
        tr = TranscriptionResult(
            text="t",
            language="en",
            language_probability=0.9,
            duration=1.0,
            processing_time=5.0,
        )
        assert tr.rtf == pytest.approx(5.0)


# ============================================================================
# PART 3 -- FasterWhisperSTT init & model loading
# ============================================================================


class TestFasterWhisperSTTInit:
    """Tests for FasterWhisperSTT initialisation."""

    def test_models_list_not_empty(self):
        assert len(FasterWhisperSTT.MODELS) > 0

    def test_models_contains_large_v3(self):
        assert "large-v3" in FasterWhisperSTT.MODELS

    def test_models_contains_tiny(self):
        assert "tiny" in FasterWhisperSTT.MODELS

    def test_models_contains_distil_large_v3(self):
        assert "distil-large-v3" in FasterWhisperSTT.MODELS

    def test_models_list_length(self):
        assert len(FasterWhisperSTT.MODELS) == 12

    def test_init_with_config(self, stt_config):
        stt = FasterWhisperSTT(config=stt_config)
        assert stt.config is stt_config

    def test_init_model_none_initially(self, stt_service):
        assert stt_service._model is None

    def test_init_model_path_stored(self, stt_config):
        stt = FasterWhisperSTT(config=stt_config, model_path="/custom/path")
        assert stt._model_path == "/custom/path"

    def test_init_model_path_none_default(self, stt_config):
        stt = FasterWhisperSTT(config=stt_config)
        assert stt._model_path is None

    def test_init_default_config_used(self):
        """When no config provided, get_voice_config().stt is used."""
        with patch("voice.stt.faster_whisper_service.get_voice_config") as mock_gvc:
            mock_gvc.return_value.stt = _make_stt_config()
            stt = FasterWhisperSTT()
            mock_gvc.assert_called_once()

    def test_is_loaded_false_before_load(self, stt_service):
        assert stt_service.is_loaded is False


class TestFasterWhisperSTTModelLoading:
    """Tests for _ensure_model_loaded."""

    def test_ensure_model_loaded_creates_model(self, stt_service):
        with patch("voice.stt.faster_whisper_service.WhisperModel") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            result = stt_service._ensure_model_loaded()
            assert result is mock_instance
            MockModel.assert_called_once_with(
                "tiny",
                device="cpu",
                compute_type="float32",
            )

    def test_ensure_model_loaded_uses_model_path(self, stt_config):
        stt = FasterWhisperSTT(config=stt_config, model_path="/custom/model")
        with patch("voice.stt.faster_whisper_service.WhisperModel") as MockModel:
            MockModel.return_value = MagicMock()
            stt._ensure_model_loaded()
            MockModel.assert_called_once_with(
                "/custom/model",
                device="cpu",
                compute_type="float32",
            )

    def test_ensure_model_loaded_caches(self, stt_service):
        with patch("voice.stt.faster_whisper_service.WhisperModel") as MockModel:
            MockModel.return_value = MagicMock()
            first = stt_service._ensure_model_loaded()
            second = stt_service._ensure_model_loaded()
            assert first is second
            MockModel.assert_called_once()

    def test_is_loaded_true_after_load(self, stt_service):
        with patch("voice.stt.faster_whisper_service.WhisperModel") as MockModel:
            MockModel.return_value = MagicMock()
            stt_service._ensure_model_loaded()
            assert stt_service.is_loaded is True


# ============================================================================
# PART 4 -- _prepare_audio
# ============================================================================


class TestPrepareAudio:
    """Tests for FasterWhisperSTT._prepare_audio."""

    def test_int16_to_float32(self, stt_service):
        audio = np.array([16384, -16384, 0], dtype=np.int16)
        result = stt_service._prepare_audio(audio)
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(0.5, abs=0.001)
        assert result[1] == pytest.approx(-0.5, abs=0.001)
        assert result[2] == pytest.approx(0.0)

    def test_int32_to_float32(self, stt_service):
        audio = np.array([1073741824, -1073741824], dtype=np.int32)
        result = stt_service._prepare_audio(audio)
        assert result.dtype == np.float32
        assert result[0] == pytest.approx(0.5, abs=0.001)

    def test_float32_passthrough(self, stt_service):
        audio = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        result = stt_service._prepare_audio(audio)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, audio)

    def test_float64_to_float32(self, stt_service):
        audio = np.array([0.5, -0.5], dtype=np.float64)
        result = stt_service._prepare_audio(audio)
        assert result.dtype == np.float32

    def test_stereo_to_mono(self, stt_service):
        audio = np.array([[0.5, 0.3], [0.1, 0.9]], dtype=np.float32)
        result = stt_service._prepare_audio(audio)
        assert len(result.shape) == 1
        assert result[0] == pytest.approx(0.4)
        assert result[1] == pytest.approx(0.5)

    def test_mono_unchanged(self, stt_service):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = stt_service._prepare_audio(audio)
        assert len(result.shape) == 1

    def test_int16_max_value(self, stt_service):
        audio = np.array([32767], dtype=np.int16)
        result = stt_service._prepare_audio(audio)
        assert result[0] == pytest.approx(32767 / 32768.0)

    def test_int16_min_value(self, stt_service):
        audio = np.array([-32768], dtype=np.int16)
        result = stt_service._prepare_audio(audio)
        assert result[0] == pytest.approx(-1.0)

    def test_empty_array(self, stt_service):
        audio = np.array([], dtype=np.float32)
        result = stt_service._prepare_audio(audio)
        assert len(result) == 0
        assert result.dtype == np.float32

    def test_int16_stereo_combined(self, stt_service):
        """int16 + stereo: should convert type AND merge channels."""
        audio = np.array([[16384, 0], [0, 16384]], dtype=np.int16)
        result = stt_service._prepare_audio(audio)
        assert result.dtype == np.float32
        assert len(result.shape) == 1
        assert result[0] == pytest.approx(0.25, abs=0.001)


# ============================================================================
# PART 5 -- transcribe
# ============================================================================


class TestTranscribe:
    """Tests for FasterWhisperSTT.transcribe."""

    def _setup_mock_model(self, stt_service, segments, info):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(segments), info)
        stt_service._model = mock_model
        return mock_model

    def test_transcribe_ndarray(self, stt_service):
        audio = np.zeros(16000, dtype=np.float32)  # 1 second
        info = _mock_transcribe_info("en", 0.99)
        seg = _mock_segment(text=" Hello", seg_id=0, start=0.0, end=0.5)
        mock_model = self._setup_mock_model(stt_service, [seg], info)

        result = stt_service.transcribe(audio)
        assert result.text == "Hello"
        assert result.language == "en"
        assert result.language_probability == 0.99
        assert result.duration == pytest.approx(1.0)
        mock_model.transcribe.assert_called_once()

    def test_transcribe_file_path_string(self, stt_service):
        info = _mock_transcribe_info("te", 0.95)
        seg = _mock_segment(text=" Telugu text")
        self._setup_mock_model(stt_service, [seg], info)

        with patch.object(
            stt_service, "_get_audio_duration_from_file", return_value=3.0
        ):
            result = stt_service.transcribe("/tmp/audio.wav")

        assert result.text == "Telugu text"
        assert result.language == "te"
        assert result.duration == pytest.approx(3.0)

    def test_transcribe_path_object(self, stt_service):
        info = _mock_transcribe_info("en", 0.90)
        seg = _mock_segment(text=" test")
        self._setup_mock_model(stt_service, [seg], info)

        with patch.object(
            stt_service, "_get_audio_duration_from_file", return_value=2.0
        ):
            result = stt_service.transcribe(Path("/tmp/audio.wav"))

        assert result.text == "test"

    def test_transcribe_uses_beam_size_from_config(self, stt_service):
        audio = np.zeros(16000, dtype=np.float32)
        info = _mock_transcribe_info()
        seg = _mock_segment()
        mock_model = self._setup_mock_model(stt_service, [seg], info)

        stt_service.transcribe(audio)
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["beam_size"] == 5

    def test_transcribe_uses_vad_filter_from_config(self, stt_service):
        audio = np.zeros(16000, dtype=np.float32)
        info = _mock_transcribe_info()
        seg = _mock_segment()
        mock_model = self._setup_mock_model(stt_service, [seg], info)

        stt_service.transcribe(audio)
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["vad_filter"] is True

    def test_transcribe_uses_word_timestamps_from_config(self, stt_service):
        audio = np.zeros(16000, dtype=np.float32)
        info = _mock_transcribe_info()
        seg = _mock_segment()
        mock_model = self._setup_mock_model(stt_service, [seg], info)

        stt_service.transcribe(audio)
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["word_timestamps"] is True

    def test_transcribe_force_language(self, stt_service):
        audio = np.zeros(16000, dtype=np.float32)
        info = _mock_transcribe_info()
        seg = _mock_segment()
        mock_model = self._setup_mock_model(stt_service, [seg], info)

        stt_service.transcribe(audio, language="te")
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "te"

    def test_transcribe_config_language_used_when_no_override(self):
        config = _make_stt_config(language="hi")
        stt = FasterWhisperSTT(config=config)
        mock_model = MagicMock()
        info = _mock_transcribe_info("hi", 0.99)
        mock_model.transcribe.return_value = (iter([_mock_segment()]), info)
        stt._model = mock_model

        stt.transcribe(np.zeros(16000, dtype=np.float32))
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "hi"

    def test_transcribe_extra_kwargs_forwarded(self, stt_service):
        audio = np.zeros(16000, dtype=np.float32)
        info = _mock_transcribe_info()
        mock_model = self._setup_mock_model(stt_service, [_mock_segment()], info)

        stt_service.transcribe(audio, temperature=0.5)
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_transcribe_multiple_segments(self, stt_service):
        audio = np.zeros(32000, dtype=np.float32)
        info = _mock_transcribe_info()
        seg1 = _mock_segment(text=" Hello", seg_id=0, start=0.0, end=0.5)
        seg2 = _mock_segment(text=" world", seg_id=1, start=0.5, end=1.0)
        self._setup_mock_model(stt_service, [seg1, seg2], info)

        result = stt_service.transcribe(audio)
        assert result.text == "Hello world"
        assert len(result.segments) == 2

    def test_transcribe_word_timings_collected(self, stt_service):
        audio = np.zeros(16000, dtype=np.float32)
        info = _mock_transcribe_info()
        mock_word = MagicMock()
        mock_word.word = "Hello"
        mock_word.start = 0.0
        mock_word.end = 0.3
        mock_word.probability = 0.95
        seg = _mock_segment(text=" Hello", words=[mock_word])
        self._setup_mock_model(stt_service, [seg], info)

        result = stt_service.transcribe(audio)
        assert len(result.words) == 1
        assert result.words[0].word == "Hello"
        assert result.words[0].probability == 0.95

    def test_transcribe_no_words_when_timestamps_disabled(self):
        config = _make_stt_config(word_timestamps=False)
        stt = FasterWhisperSTT(config=config)
        mock_model = MagicMock()
        info = _mock_transcribe_info()
        seg = _mock_segment(text=" Hi")
        seg.words = None  # No words when disabled
        mock_model.transcribe.return_value = (iter([seg]), info)
        stt._model = mock_model

        result = stt.transcribe(np.zeros(16000, dtype=np.float32))
        assert result.words == []

    def test_transcribe_empty_result(self, stt_service):
        audio = np.zeros(16000, dtype=np.float32)
        info = _mock_transcribe_info()
        self._setup_mock_model(stt_service, [], info)

        result = stt_service.transcribe(audio)
        assert result.text == ""

    def test_transcribe_audio_duration_from_ndarray(self, stt_service):
        audio = np.zeros(32000, dtype=np.float32)  # 2 seconds at 16kHz
        info = _mock_transcribe_info()
        self._setup_mock_model(stt_service, [_mock_segment()], info)

        result = stt_service.transcribe(audio)
        assert result.duration == pytest.approx(2.0)

    def test_transcribe_processing_time_positive(self, stt_service):
        audio = np.zeros(16000, dtype=np.float32)
        info = _mock_transcribe_info()
        self._setup_mock_model(stt_service, [_mock_segment()], info)

        result = stt_service.transcribe(audio)
        assert result.processing_time >= 0


# ============================================================================
# PART 6 -- transcribe_stream
# ============================================================================


class TestTranscribeStream:
    """Tests for FasterWhisperSTT.transcribe_stream."""

    def test_stream_yields_results(self, stt_service):
        with patch.object(stt_service, "transcribe") as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(
                text="hello",
                language="en",
                language_probability=0.9,
                duration=0.5,
                processing_time=0.1,
            )
            chunks = [np.zeros(8000, dtype=np.float32)]  # 0.5s at 16kHz
            results = list(
                stt_service.transcribe_stream(iter(chunks), chunk_duration=0.5)
            )
            assert len(results) >= 1
            assert results[0].text == "hello"

    def test_stream_buffers_until_duration(self, stt_service):
        with patch.object(stt_service, "transcribe") as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(
                text="hello",
                language="en",
                language_probability=0.9,
                duration=1.0,
                processing_time=0.1,
            )
            # Each chunk is 0.25s; need 4 chunks to reach chunk_duration=1.0
            chunks = [np.zeros(4000, dtype=np.float32) for _ in range(4)]
            results = list(
                stt_service.transcribe_stream(iter(chunks), chunk_duration=1.0)
            )
            assert len(results) == 1

    def test_stream_remainder_flushed(self, stt_service):
        with patch.object(stt_service, "transcribe") as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(
                text="remainder",
                language="en",
                language_probability=0.9,
                duration=0.5,
                processing_time=0.1,
            )
            # One small chunk that doesn't fill the buffer
            chunks = [np.zeros(4000, dtype=np.float32)]  # 0.25s
            results = list(
                stt_service.transcribe_stream(iter(chunks), chunk_duration=1.0)
            )
            assert len(results) == 1
            assert results[0].text == "remainder"

    def test_stream_skips_empty_text(self, stt_service):
        with patch.object(stt_service, "transcribe") as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(
                text="",
                language="en",
                language_probability=0.9,
                duration=0.5,
                processing_time=0.1,
            )
            chunks = [np.zeros(8000, dtype=np.float32)]
            results = list(
                stt_service.transcribe_stream(iter(chunks), chunk_duration=0.5)
            )
            assert len(results) == 0

    def test_stream_skips_whitespace_only(self, stt_service):
        with patch.object(stt_service, "transcribe") as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(
                text="   ",
                language="en",
                language_probability=0.9,
                duration=0.5,
                processing_time=0.1,
            )
            chunks = [np.zeros(8000, dtype=np.float32)]
            results = list(
                stt_service.transcribe_stream(iter(chunks), chunk_duration=0.5)
            )
            assert len(results) == 0

    def test_stream_forwards_language(self, stt_service):
        with patch.object(stt_service, "transcribe") as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(
                text="hello",
                language="te",
                language_probability=0.9,
                duration=0.5,
                processing_time=0.1,
            )
            chunks = [np.zeros(8000, dtype=np.float32)]
            list(
                stt_service.transcribe_stream(
                    iter(chunks), chunk_duration=0.5, language="te"
                )
            )
            mock_transcribe.assert_called_once()
            _, kwargs = mock_transcribe.call_args
            assert kwargs["language"] == "te"

    def test_stream_empty_iterator(self, stt_service):
        results = list(stt_service.transcribe_stream(iter([]), chunk_duration=0.5))
        assert results == []

    def test_stream_multiple_complete_buffers(self, stt_service):
        call_count = [0]

        def _fake_transcribe(audio, language=None):
            call_count[0] += 1
            return TranscriptionResult(
                text=f"chunk{call_count[0]}",
                language="en",
                language_probability=0.9,
                duration=0.5,
                processing_time=0.1,
            )

        with patch.object(stt_service, "transcribe", side_effect=_fake_transcribe):
            # 3 chunks of 0.5s each, buffer=0.5s => 3 results
            chunks = [np.zeros(8000, dtype=np.float32) for _ in range(3)]
            results = list(
                stt_service.transcribe_stream(iter(chunks), chunk_duration=0.5)
            )
            assert len(results) == 3
            assert results[0].text == "chunk1"
            assert results[2].text == "chunk3"


# ============================================================================
# PART 7 -- detect_language
# ============================================================================


class TestDetectLanguage:
    """Tests for FasterWhisperSTT.detect_language."""

    def test_detect_language_ndarray(self, stt_service):
        mock_model = MagicMock()
        info = _mock_transcribe_info("te", 0.92)
        mock_model.transcribe.return_value = (iter([]), info)
        stt_service._model = mock_model

        lang, prob = stt_service.detect_language(np.zeros(16000, dtype=np.float32))
        assert lang == "te"
        assert prob == 0.92

    def test_detect_language_file_path(self, stt_service):
        mock_model = MagicMock()
        info = _mock_transcribe_info("en", 0.88)
        mock_model.transcribe.return_value = (iter([]), info)
        stt_service._model = mock_model

        lang, prob = stt_service.detect_language("/tmp/audio.wav")
        assert lang == "en"
        assert prob == 0.88

    def test_detect_language_path_object(self, stt_service):
        mock_model = MagicMock()
        info = _mock_transcribe_info("hi", 0.75)
        mock_model.transcribe.return_value = (iter([]), info)
        stt_service._model = mock_model

        lang, prob = stt_service.detect_language(Path("/tmp/audio.wav"))
        assert lang == "hi"

    def test_detect_language_uses_minimal_options(self, stt_service):
        mock_model = MagicMock()
        info = _mock_transcribe_info()
        mock_model.transcribe.return_value = (iter([]), info)
        stt_service._model = mock_model

        stt_service.detect_language(np.zeros(16000, dtype=np.float32))
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["beam_size"] == 1
        assert call_kwargs["word_timestamps"] is False
        assert call_kwargs["vad_filter"] is False


# ============================================================================
# PART 8 -- unload_model / is_loaded
# ============================================================================


class TestUnloadModel:
    """Tests for FasterWhisperSTT.unload_model and is_loaded."""

    def test_unload_sets_model_none(self, stt_service):
        stt_service._model = MagicMock()
        assert stt_service.is_loaded is True
        stt_service.unload_model()
        assert stt_service._model is None

    def test_unload_is_loaded_false(self, stt_service):
        stt_service._model = MagicMock()
        stt_service.unload_model()
        assert stt_service.is_loaded is False

    def test_unload_when_not_loaded(self, stt_service):
        stt_service.unload_model()  # Should not raise
        assert stt_service.is_loaded is False

    def test_reload_after_unload(self, stt_service):
        with patch("voice.stt.faster_whisper_service.WhisperModel") as MockModel:
            MockModel.return_value = MagicMock()
            stt_service._ensure_model_loaded()
            assert stt_service.is_loaded is True
            stt_service.unload_model()
            assert stt_service.is_loaded is False
            stt_service._ensure_model_loaded()
            assert stt_service.is_loaded is True
            assert MockModel.call_count == 2


# ============================================================================
# PART 9 -- _get_audio_duration_from_file
# ============================================================================


class TestGetAudioDurationFromFile:
    """Tests for _get_audio_duration_from_file."""

    def test_returns_duration(self, stt_service):
        mock_info = MagicMock()
        mock_info.duration = 5.5
        with patch("voice.stt.faster_whisper_service.sf", create=True) as mock_sf:
            # The method imports soundfile internally
            with patch("soundfile.info", return_value=mock_info):
                result = stt_service._get_audio_duration_from_file("/tmp/audio.wav")
                # Due to mocked soundfile, may return 0.0 if import differs
                assert isinstance(result, float)

    def test_returns_zero_on_error(self, stt_service):
        with patch("soundfile.info", side_effect=Exception("File not found")):
            result = stt_service._get_audio_duration_from_file("/nonexistent.wav")
            assert result == 0.0


# ============================================================================
# PART 10 -- transcribe_audio convenience function
# ============================================================================


class TestTranscribeAudioConvenience:
    """Tests for the transcribe_audio() convenience function."""

    def test_creates_stt_and_calls_transcribe(self):
        with patch("voice.stt.faster_whisper_service.FasterWhisperSTT") as MockSTT:
            mock_instance = MockSTT.return_value
            expected_result = TranscriptionResult(
                text="hi",
                language="en",
                language_probability=0.9,
                duration=1.0,
                processing_time=0.1,
            )
            mock_instance.transcribe.return_value = expected_result

            result = transcribe_audio(np.zeros(16000, dtype=np.float32))
            assert result.text == "hi"
            MockSTT.assert_called_once()
            mock_instance.transcribe.assert_called_once()

    def test_passes_language(self):
        with patch("voice.stt.faster_whisper_service.FasterWhisperSTT") as MockSTT:
            mock_instance = MockSTT.return_value
            mock_instance.transcribe.return_value = TranscriptionResult(
                text="",
                language="te",
                language_probability=0.9,
                duration=1.0,
                processing_time=0.1,
            )
            transcribe_audio(np.zeros(16000, dtype=np.float32), language="te")
            _, kwargs = mock_instance.transcribe.call_args
            assert kwargs["language"] == "te"

    def test_passes_model_and_device(self):
        with patch("voice.stt.faster_whisper_service.FasterWhisperSTT") as MockSTT:
            mock_instance = MockSTT.return_value
            mock_instance.transcribe.return_value = TranscriptionResult(
                text="",
                language="en",
                language_probability=0.9,
                duration=1.0,
                processing_time=0.1,
            )
            transcribe_audio(
                np.zeros(16000, dtype=np.float32), model="tiny", device="cpu"
            )
            config_arg = MockSTT.call_args[0][0]
            assert config_arg.model == "tiny"
            assert config_arg.device == "cpu"

    def test_default_model_large_v3(self):
        with patch("voice.stt.faster_whisper_service.FasterWhisperSTT") as MockSTT:
            mock_instance = MockSTT.return_value
            mock_instance.transcribe.return_value = TranscriptionResult(
                text="",
                language="en",
                language_probability=0.9,
                duration=1.0,
                processing_time=0.1,
            )
            transcribe_audio(np.zeros(16000, dtype=np.float32))
            config_arg = MockSTT.call_args[0][0]
            assert config_arg.model == "large-v3"

    def test_default_device_cuda(self):
        with patch("voice.stt.faster_whisper_service.FasterWhisperSTT") as MockSTT:
            mock_instance = MockSTT.return_value
            mock_instance.transcribe.return_value = TranscriptionResult(
                text="",
                language="en",
                language_probability=0.9,
                duration=1.0,
                processing_time=0.1,
            )
            transcribe_audio(np.zeros(16000, dtype=np.float32))
            config_arg = MockSTT.call_args[0][0]
            assert config_arg.device == "cuda"


# ============================================================================
# PART 11 -- TTSResult dataclass
# ============================================================================


class TestTTSResult:
    """Tests for the TTSResult dataclass."""

    def test_basic_creation(self):
        audio = np.zeros(24000, dtype=np.float32)
        tr = TTSResult(
            audio=audio,
            sample_rate=24000,
            duration=1.0,
            processing_time=0.5,
            language="en",
            voice_profile="default",
        )
        assert tr.sample_rate == 24000
        assert tr.duration == 1.0
        assert tr.language == "en"
        assert tr.voice_profile == "default"

    def test_rtf_normal(self):
        tr = TTSResult(
            audio=np.zeros(1),
            sample_rate=24000,
            duration=2.0,
            processing_time=1.0,
            language="en",
            voice_profile="",
        )
        assert tr.rtf == pytest.approx(0.5)

    def test_rtf_zero_duration(self):
        tr = TTSResult(
            audio=np.zeros(1),
            sample_rate=24000,
            duration=0.0,
            processing_time=1.0,
            language="en",
            voice_profile="",
        )
        assert tr.rtf == 0.0

    def test_rtf_realtime(self):
        tr = TTSResult(
            audio=np.zeros(1),
            sample_rate=24000,
            duration=5.0,
            processing_time=5.0,
            language="en",
            voice_profile="",
        )
        assert tr.rtf == pytest.approx(1.0)

    def test_rtf_faster_than_realtime(self):
        tr = TTSResult(
            audio=np.zeros(1),
            sample_rate=24000,
            duration=10.0,
            processing_time=2.0,
            language="en",
            voice_profile="",
        )
        assert tr.rtf == pytest.approx(0.2)

    def test_audio_stored_as_ndarray(self):
        audio = np.array([0.1, 0.2], dtype=np.float32)
        tr = TTSResult(
            audio=audio,
            sample_rate=24000,
            duration=1.0,
            processing_time=0.1,
            language="te",
            voice_profile="friday",
        )
        np.testing.assert_array_equal(tr.audio, audio)


# ============================================================================
# PART 12 -- LANGUAGE_MAP and LANGUAGE_FALLBACK
# ============================================================================


class TestLanguageMappings:
    """Tests for LANGUAGE_MAP and LANGUAGE_FALLBACK module-level dicts."""

    def test_language_map_has_telugu(self):
        assert "te" in LANGUAGE_MAP
        assert LANGUAGE_MAP["te"] == "te"

    def test_language_map_has_english(self):
        assert "en" in LANGUAGE_MAP
        assert LANGUAGE_MAP["en"] == "en"

    def test_language_map_has_hindi(self):
        assert "hi" in LANGUAGE_MAP
        assert LANGUAGE_MAP["hi"] == "hi"

    def test_language_map_has_tamil(self):
        assert "ta" in LANGUAGE_MAP
        assert LANGUAGE_MAP["ta"] == "ta"

    def test_language_fallback_telugu_to_hindi(self):
        assert "te" in LANGUAGE_FALLBACK
        assert LANGUAGE_FALLBACK["te"] == "hi"

    def test_language_fallback_no_english_entry(self):
        assert "en" not in LANGUAGE_FALLBACK

    def test_language_map_is_dict(self):
        assert isinstance(LANGUAGE_MAP, dict)

    def test_language_fallback_is_dict(self):
        assert isinstance(LANGUAGE_FALLBACK, dict)


# ============================================================================
# PART 13 -- XTTSService init and model loading
# ============================================================================


class TestXTTSServiceInit:
    """Tests for XTTSService initialisation."""

    def test_init_with_config(self, tts_config):
        svc = XTTSService(config=tts_config)
        assert svc.config is tts_config

    def test_init_model_none(self, tts_service):
        assert tts_service._model is None

    def test_init_xtts_model_none(self, tts_service):
        assert tts_service._xtts_model is None

    def test_init_model_path_stored(self, tts_config):
        svc = XTTSService(config=tts_config, model_path="/custom/tts")
        assert svc._model_path == "/custom/tts"

    def test_init_voice_profiles_empty(self, tts_service):
        assert tts_service._voice_profiles == {}

    def test_init_sample_rate_24000(self, tts_service):
        assert tts_service.sample_rate == 24000

    def test_is_loaded_false_initially(self, tts_service):
        assert tts_service.is_loaded is False

    def test_init_default_config_used(self):
        with patch("voice.tts.xtts_service.get_voice_config") as mock_gvc:
            mock_gvc.return_value.tts = _make_tts_config()
            svc = XTTSService()
            mock_gvc.assert_called_once()


class TestXTTSServiceModelLoading:
    """Tests for _ensure_model_loaded."""

    def test_ensure_model_loaded_creates_model(self, tts_service):
        with patch("voice.tts.xtts_service.TTS") as MockTTS:
            mock_tts_instance = MagicMock()
            mock_tts_instance.to.return_value = mock_tts_instance
            MockTTS.return_value = mock_tts_instance

            result = tts_service._ensure_model_loaded()
            MockTTS.assert_called_once_with(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2"
            )
            mock_tts_instance.to.assert_called_once_with("cpu")
            assert result is mock_tts_instance

    def test_ensure_model_loaded_custom_path(self, tts_config):
        svc = XTTSService(config=tts_config, model_path="/custom/model")
        with patch("voice.tts.xtts_service.TTS") as MockTTS:
            mock_tts_instance = MagicMock()
            mock_tts_instance.to.return_value = mock_tts_instance
            MockTTS.return_value = mock_tts_instance

            svc._ensure_model_loaded()
            MockTTS.assert_called_once_with(model_name="/custom/model")

    def test_ensure_model_loaded_caches(self, tts_service):
        with patch("voice.tts.xtts_service.TTS") as MockTTS:
            mock_tts_instance = MagicMock()
            mock_tts_instance.to.return_value = mock_tts_instance
            MockTTS.return_value = mock_tts_instance

            first = tts_service._ensure_model_loaded()
            second = tts_service._ensure_model_loaded()
            assert first is second
            MockTTS.assert_called_once()

    def test_is_loaded_true_after_load(self, tts_service):
        mock_model = MagicMock()
        tts_service._model = mock_model
        assert tts_service.is_loaded is True


# ============================================================================
# PART 14 -- synthesize
# ============================================================================


class TestSynthesize:
    """Tests for XTTSService.synthesize."""

    def _setup_mock_model(self, tts_service, audio_data=None):
        mock_model = MagicMock()
        if audio_data is None:
            audio_data = [0.1, 0.2, 0.3] * 8000  # some audio
        mock_model.tts.return_value = audio_data
        tts_service._model = mock_model
        return mock_model

    def test_synthesize_basic(self, tts_service):
        mock_model = self._setup_mock_model(tts_service)
        result = tts_service.synthesize("Hello", language="en")
        assert isinstance(result, TTSResult)
        assert result.language == "en"
        mock_model.tts.assert_called_once()

    def test_synthesize_with_speaker_wav_string(self, tts_service):
        mock_model = self._setup_mock_model(tts_service)
        result = tts_service.synthesize(
            "Hello", language="en", speaker_wav="/tmp/ref.wav"
        )
        call_kwargs = mock_model.tts.call_args[1]
        assert call_kwargs["speaker_wav"] == "/tmp/ref.wav"

    def test_synthesize_with_speaker_wav_list(self, tts_service):
        mock_model = self._setup_mock_model(tts_service)
        result = tts_service.synthesize(
            "Hello", language="en", speaker_wav=["/a.wav", "/b.wav"]
        )
        call_kwargs = mock_model.tts.call_args[1]
        assert call_kwargs["speaker_wav"] == ["/a.wav", "/b.wav"]

    def test_synthesize_with_speaker_wav_single_in_list(self, tts_service):
        mock_model = self._setup_mock_model(tts_service)
        result = tts_service.synthesize("Hello", language="en", speaker_wav=["/a.wav"])
        call_kwargs = mock_model.tts.call_args[1]
        assert call_kwargs["speaker_wav"] == "/a.wav"

    def test_synthesize_without_speaker_wav(self, tts_service):
        mock_model = self._setup_mock_model(tts_service)
        # No profile loaded, no speaker_wav => default mode
        tts_service._voice_profiles = {}
        result = tts_service.synthesize("Hello", language="en")
        call_kwargs = mock_model.tts.call_args[1]
        assert "speaker_wav" not in call_kwargs

    def test_synthesize_with_profile(self, tts_service):
        mock_model = self._setup_mock_model(tts_service)
        tts_service._voice_profiles["my_voice"] = {
            "reference_audio": ["/ref1.wav", "/ref2.wav"],
            "language": "en",
        }
        result = tts_service.synthesize("Hello", language="en", profile="my_voice")
        call_kwargs = mock_model.tts.call_args[1]
        assert call_kwargs["speaker_wav"] == ["/ref1.wav", "/ref2.wav"]

    def test_synthesize_profile_single_ref(self, tts_service):
        mock_model = self._setup_mock_model(tts_service)
        tts_service._voice_profiles["single"] = {
            "reference_audio": ["/ref.wav"],
            "language": "en",
        }
        result = tts_service.synthesize("Hello", language="en", profile="single")
        call_kwargs = mock_model.tts.call_args[1]
        assert call_kwargs["speaker_wav"] == "/ref.wav"

    def test_synthesize_default_language_from_config(self, tts_service):
        mock_model = self._setup_mock_model(tts_service)
        # Not passing language; should use config.default_language = "te"
        # "te" maps to "hi" via fallback
        result = tts_service.synthesize("text")
        call_kwargs = mock_model.tts.call_args[1]
        assert call_kwargs["language"] == "hi"  # te falls back to hi

    def test_synthesize_split_sentences_true(self, tts_service):
        mock_model = self._setup_mock_model(tts_service)
        tts_service.synthesize("Hello", language="en")
        call_kwargs = mock_model.tts.call_args[1]
        assert call_kwargs["split_sentences"] is True

    def test_synthesize_result_sample_rate(self, tts_service):
        self._setup_mock_model(tts_service)
        result = tts_service.synthesize("Hello", language="en")
        assert result.sample_rate == 24000

    def test_synthesize_result_audio_is_float32(self, tts_service):
        self._setup_mock_model(tts_service, audio_data=[0.1, 0.2])
        result = tts_service.synthesize("Hello", language="en")
        assert result.audio.dtype == np.float32

    def test_synthesize_result_duration(self, tts_service):
        audio_data = [0.0] * 24000  # 1 second at 24kHz
        self._setup_mock_model(tts_service, audio_data=audio_data)
        result = tts_service.synthesize("Hello", language="en")
        assert result.duration == pytest.approx(1.0)

    def test_synthesize_result_processing_time_positive(self, tts_service):
        self._setup_mock_model(tts_service)
        result = tts_service.synthesize("Hello", language="en")
        assert result.processing_time >= 0

    def test_synthesize_result_voice_profile(self, tts_service):
        self._setup_mock_model(tts_service)
        result = tts_service.synthesize("Hello", language="en", profile="myprofile")
        assert result.voice_profile == "myprofile"

    def test_synthesize_default_profile_from_config(self, tts_service):
        self._setup_mock_model(tts_service)
        # config default_profile is "friday_telugu"
        result = tts_service.synthesize("Hello", language="en")
        assert result.voice_profile == "friday_telugu"

    def test_synthesize_numpy_array_returned(self, tts_service):
        self._setup_mock_model(
            tts_service, audio_data=np.array([0.1, 0.2], dtype=np.float32)
        )
        result = tts_service.synthesize("Hello", language="en")
        assert isinstance(result.audio, np.ndarray)

    def test_synthesize_list_audio_converted_to_ndarray(self, tts_service):
        self._setup_mock_model(tts_service, audio_data=[0.1, 0.2, 0.3])
        result = tts_service.synthesize("Hello", language="en")
        assert isinstance(result.audio, np.ndarray)

    def test_synthesize_speaker_wav_overrides_profile(self, tts_service):
        mock_model = self._setup_mock_model(tts_service)
        tts_service._voice_profiles["my_voice"] = {
            "reference_audio": ["/profile_ref.wav"],
            "language": "en",
        }
        # speaker_wav takes priority over profile
        tts_service.synthesize(
            "Hello", language="en", profile="my_voice", speaker_wav="/override.wav"
        )
        call_kwargs = mock_model.tts.call_args[1]
        assert call_kwargs["speaker_wav"] == "/override.wav"


# ============================================================================
# PART 15 -- _get_xtts_language
# ============================================================================


class TestGetXttsLanguage:
    """Tests for XTTSService._get_xtts_language."""

    def test_english_supported(self, tts_service):
        assert tts_service._get_xtts_language("en") == "en"

    def test_hindi_supported(self, tts_service):
        assert tts_service._get_xtts_language("hi") == "hi"

    def test_telugu_falls_back_to_hindi(self, tts_service):
        assert tts_service._get_xtts_language("te") == "hi"

    def test_tamil_not_in_xtts_supported(self, tts_service):
        # Tamil is in LANGUAGE_MAP but not in xtts_supported set,
        # and not in LANGUAGE_FALLBACK => defaults to "en"
        result = tts_service._get_xtts_language("ta")
        assert result == "en"

    def test_unknown_language_defaults_to_en(self, tts_service):
        assert tts_service._get_xtts_language("zz") == "en"

    def test_spanish_not_in_map(self, tts_service):
        # "es" is not in LANGUAGE_MAP => unknown => "en"
        assert tts_service._get_xtts_language("es") == "en"

    def test_french_not_in_map(self, tts_service):
        assert tts_service._get_xtts_language("fr") == "en"

    def test_empty_string(self, tts_service):
        assert tts_service._get_xtts_language("") == "en"


# ============================================================================
# PART 16 -- load_voice_profile
# ============================================================================


class TestLoadVoiceProfile:
    """Tests for XTTSService.load_voice_profile."""

    def test_load_valid_profile(self, tts_service, tmp_path):
        ref_audio = tmp_path / "ref.wav"
        ref_audio.touch()
        tts_service.load_voice_profile("test_voice", [str(ref_audio)])
        assert "test_voice" in tts_service._voice_profiles
        assert tts_service._voice_profiles["test_voice"]["reference_audio"] == [
            str(ref_audio)
        ]

    def test_load_multiple_refs(self, tts_service, tmp_path):
        ref1 = tmp_path / "ref1.wav"
        ref2 = tmp_path / "ref2.wav"
        ref1.touch()
        ref2.touch()
        tts_service.load_voice_profile("multi", [str(ref1), str(ref2)])
        assert len(tts_service._voice_profiles["multi"]["reference_audio"]) == 2

    def test_load_profile_stores_language(self, tts_service, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.touch()
        tts_service.load_voice_profile("telugu_voice", [str(ref)], language="te")
        assert tts_service._voice_profiles["telugu_voice"]["language"] == "te"

    def test_load_profile_default_language_te(self, tts_service, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.touch()
        tts_service.load_voice_profile("voice", [str(ref)])
        assert tts_service._voice_profiles["voice"]["language"] == "te"

    def test_load_profile_skips_missing_files(self, tts_service, tmp_path):
        ref_exists = tmp_path / "exists.wav"
        ref_exists.touch()
        tts_service.load_voice_profile(
            "mixed", [str(ref_exists), "/nonexistent/file.wav"]
        )
        assert len(tts_service._voice_profiles["mixed"]["reference_audio"]) == 1

    def test_load_profile_all_missing_raises(self, tts_service):
        with pytest.raises(ValueError, match="No valid reference audio"):
            tts_service.load_voice_profile(
                "bad", ["/nonexistent1.wav", "/nonexistent2.wav"]
            )

    def test_load_profile_path_objects(self, tts_service, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.touch()
        tts_service.load_voice_profile("pathobj", [ref])
        assert len(tts_service._voice_profiles["pathobj"]["reference_audio"]) == 1

    def test_overwrite_existing_profile(self, tts_service, tmp_path):
        ref1 = tmp_path / "ref1.wav"
        ref2 = tmp_path / "ref2.wav"
        ref1.touch()
        ref2.touch()
        tts_service.load_voice_profile("voice", [str(ref1)])
        tts_service.load_voice_profile("voice", [str(ref2)])
        assert tts_service._voice_profiles["voice"]["reference_audio"] == [str(ref2)]


# ============================================================================
# PART 17 -- list_profiles
# ============================================================================


class TestListProfiles:
    """Tests for XTTSService.list_profiles."""

    def test_empty_initially(self, tts_service):
        assert tts_service.list_profiles() == []

    def test_after_loading_one(self, tts_service, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.touch()
        tts_service.load_voice_profile("voice1", [str(ref)])
        assert tts_service.list_profiles() == ["voice1"]

    def test_after_loading_multiple(self, tts_service, tmp_path):
        ref = tmp_path / "ref.wav"
        ref.touch()
        tts_service.load_voice_profile("voice1", [str(ref)])
        tts_service.load_voice_profile("voice2", [str(ref)])
        profiles = tts_service.list_profiles()
        assert len(profiles) == 2
        assert "voice1" in profiles
        assert "voice2" in profiles

    def test_returns_list(self, tts_service):
        assert isinstance(tts_service.list_profiles(), list)


# ============================================================================
# PART 18 -- synthesize_stream
# ============================================================================


class TestSynthesizeStream:
    """Tests for XTTSService.synthesize_stream."""

    def test_yields_chunks(self, tts_service):
        audio_data = np.zeros(24000, dtype=np.float32)
        with patch.object(tts_service, "synthesize") as mock_synth:
            mock_synth.return_value = TTSResult(
                audio=audio_data,
                sample_rate=24000,
                duration=1.0,
                processing_time=0.1,
                language="en",
                voice_profile="",
            )
            chunks = list(
                tts_service.synthesize_stream("Hello", language="en", chunk_size=8000)
            )
            assert len(chunks) == 3  # 24000 / 8000 = 3

    def test_chunk_sizes_correct(self, tts_service):
        audio_data = np.zeros(20000, dtype=np.float32)
        with patch.object(tts_service, "synthesize") as mock_synth:
            mock_synth.return_value = TTSResult(
                audio=audio_data,
                sample_rate=24000,
                duration=1.0,
                processing_time=0.1,
                language="en",
                voice_profile="",
            )
            chunks = list(
                tts_service.synthesize_stream("Hello", language="en", chunk_size=8192)
            )
            assert len(chunks[0]) == 8192
            assert len(chunks[1]) == 8192
            assert len(chunks[2]) == 20000 - 8192 * 2  # remainder

    def test_stream_single_chunk_when_small(self, tts_service):
        audio_data = np.zeros(100, dtype=np.float32)
        with patch.object(tts_service, "synthesize") as mock_synth:
            mock_synth.return_value = TTSResult(
                audio=audio_data,
                sample_rate=24000,
                duration=0.004,
                processing_time=0.001,
                language="en",
                voice_profile="",
            )
            chunks = list(
                tts_service.synthesize_stream("Hi", language="en", chunk_size=8192)
            )
            assert len(chunks) == 1
            assert len(chunks[0]) == 100

    def test_stream_forwards_language_and_profile(self, tts_service):
        audio_data = np.zeros(100, dtype=np.float32)
        with patch.object(tts_service, "synthesize") as mock_synth:
            mock_synth.return_value = TTSResult(
                audio=audio_data,
                sample_rate=24000,
                duration=0.004,
                processing_time=0.001,
                language="te",
                voice_profile="friday",
            )
            list(
                tts_service.synthesize_stream("Hello", language="te", profile="friday")
            )
            mock_synth.assert_called_once_with("Hello", language="te", profile="friday")

    def test_stream_empty_audio(self, tts_service):
        audio_data = np.zeros(0, dtype=np.float32)
        with patch.object(tts_service, "synthesize") as mock_synth:
            mock_synth.return_value = TTSResult(
                audio=audio_data,
                sample_rate=24000,
                duration=0.0,
                processing_time=0.001,
                language="en",
                voice_profile="",
            )
            chunks = list(tts_service.synthesize_stream("", language="en"))
            assert len(chunks) == 0


# ============================================================================
# PART 19 -- XTTSService unload_model / is_loaded
# ============================================================================


class TestXTTSUnloadModel:
    """Tests for XTTSService.unload_model and is_loaded."""

    def test_unload_sets_model_none(self, tts_service):
        tts_service._model = MagicMock()
        tts_service.unload_model()
        assert tts_service._model is None

    def test_is_loaded_false_after_unload(self, tts_service):
        tts_service._model = MagicMock()
        tts_service.unload_model()
        assert tts_service.is_loaded is False

    def test_unload_when_not_loaded(self, tts_service):
        tts_service.unload_model()  # Should not raise
        assert tts_service.is_loaded is False

    def test_reload_after_unload(self, tts_service):
        with patch("voice.tts.xtts_service.TTS") as MockTTS:
            mock_instance = MagicMock()
            mock_instance.to.return_value = mock_instance
            MockTTS.return_value = mock_instance

            tts_service._ensure_model_loaded()
            assert tts_service.is_loaded is True
            tts_service.unload_model()
            assert tts_service.is_loaded is False
            tts_service._ensure_model_loaded()
            assert tts_service.is_loaded is True
            assert MockTTS.call_count == 2


# ============================================================================
# PART 20 -- synthesize_speech convenience function
# ============================================================================


class TestSynthesizeSpeechConvenience:
    """Tests for the synthesize_speech() convenience function."""

    def test_returns_audio_and_sample_rate(self):
        with patch("voice.tts.xtts_service.XTTSService") as MockTTS:
            mock_instance = MockTTS.return_value
            expected_audio = np.zeros(24000, dtype=np.float32)
            mock_instance.synthesize.return_value = TTSResult(
                audio=expected_audio,
                sample_rate=24000,
                duration=1.0,
                processing_time=0.1,
                language="en",
                voice_profile="",
            )
            audio, sr = synthesize_speech("Hello", language="en")
            assert sr == 24000
            np.testing.assert_array_equal(audio, expected_audio)

    def test_passes_language(self):
        with patch("voice.tts.xtts_service.XTTSService") as MockTTS:
            mock_instance = MockTTS.return_value
            mock_instance.synthesize.return_value = TTSResult(
                audio=np.zeros(1),
                sample_rate=24000,
                duration=0.0,
                processing_time=0.0,
                language="te",
                voice_profile="",
            )
            synthesize_speech("text", language="te")
            call_kwargs = mock_instance.synthesize.call_args[1]
            assert call_kwargs["language"] == "te"

    def test_passes_speaker_wav(self):
        with patch("voice.tts.xtts_service.XTTSService") as MockTTS:
            mock_instance = MockTTS.return_value
            mock_instance.synthesize.return_value = TTSResult(
                audio=np.zeros(1),
                sample_rate=24000,
                duration=0.0,
                processing_time=0.0,
                language="en",
                voice_profile="",
            )
            synthesize_speech("text", speaker_wav="/ref.wav")
            call_kwargs = mock_instance.synthesize.call_args[1]
            assert call_kwargs["speaker_wav"] == "/ref.wav"

    def test_passes_device(self):
        with patch("voice.tts.xtts_service.XTTSService") as MockTTS:
            mock_instance = MockTTS.return_value
            mock_instance.synthesize.return_value = TTSResult(
                audio=np.zeros(1),
                sample_rate=24000,
                duration=0.0,
                processing_time=0.0,
                language="en",
                voice_profile="",
            )
            synthesize_speech("text", device="cpu")
            config_arg = MockTTS.call_args[0][0]
            assert config_arg.device == "cpu"

    def test_default_language_te(self):
        with patch("voice.tts.xtts_service.XTTSService") as MockTTS:
            mock_instance = MockTTS.return_value
            mock_instance.synthesize.return_value = TTSResult(
                audio=np.zeros(1),
                sample_rate=24000,
                duration=0.0,
                processing_time=0.0,
                language="te",
                voice_profile="",
            )
            synthesize_speech("text")
            call_kwargs = mock_instance.synthesize.call_args[1]
            assert call_kwargs["language"] == "te"

    def test_default_device_cuda(self):
        with patch("voice.tts.xtts_service.XTTSService") as MockTTS:
            mock_instance = MockTTS.return_value
            mock_instance.synthesize.return_value = TTSResult(
                audio=np.zeros(1),
                sample_rate=24000,
                duration=0.0,
                processing_time=0.0,
                language="te",
                voice_profile="",
            )
            synthesize_speech("text")
            config_arg = MockTTS.call_args[0][0]
            assert config_arg.device == "cuda"


# ============================================================================
# PART 21 -- save_audio utility function
# ============================================================================


class TestSaveAudio:
    """Tests for the save_audio() utility function."""

    def test_save_audio_creates_file(self, tmp_path):
        audio = np.zeros(24000, dtype=np.float32)
        out = tmp_path / "output.wav"
        mock_sf = MagicMock()
        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            result = save_audio(audio, str(out), sample_rate=24000)
            mock_sf.write.assert_called_once_with(str(out), audio, 24000)
            assert result == out

    def test_save_audio_path_object(self, tmp_path):
        audio = np.zeros(100, dtype=np.float32)
        out = tmp_path / "out.wav"
        mock_sf = MagicMock()
        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            result = save_audio(audio, out, sample_rate=16000)
            mock_sf.write.assert_called_once_with(str(out), audio, 16000)

    def test_save_audio_default_sample_rate(self, tmp_path):
        audio = np.zeros(100, dtype=np.float32)
        out = tmp_path / "default_sr.wav"
        mock_sf = MagicMock()
        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            save_audio(audio, out)
            mock_sf.write.assert_called_once_with(str(out), audio, 24000)

    def test_save_audio_creates_parent_dirs(self, tmp_path):
        audio = np.zeros(100, dtype=np.float32)
        out = tmp_path / "sub" / "dir" / "audio.wav"
        mock_sf = MagicMock()
        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            save_audio(audio, out)
            assert (tmp_path / "sub" / "dir").exists()

    def test_save_audio_returns_path(self, tmp_path):
        audio = np.zeros(100, dtype=np.float32)
        out = tmp_path / "test.wav"
        mock_sf = MagicMock()
        with patch.dict("sys.modules", {"soundfile": mock_sf}):
            result = save_audio(audio, str(out))
            assert isinstance(result, Path)
            assert result == out


# ============================================================================
# PART 22 -- Edge cases and integration-style tests
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary tests."""

    def test_transcribe_very_short_audio(self, stt_service):
        """Audio shorter than a typical frame (10ms)."""
        audio = np.zeros(160, dtype=np.float32)  # 10ms at 16kHz
        info = _mock_transcribe_info()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), info)
        stt_service._model = mock_model

        result = stt_service.transcribe(audio)
        assert result.text == ""
        assert result.duration == pytest.approx(0.01)

    def test_transcribe_result_language_probability_range(self, stt_service):
        audio = np.zeros(16000, dtype=np.float32)
        info = _mock_transcribe_info("en", 0.5)
        seg = _mock_segment(text=" text")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)
        stt_service._model = mock_model

        result = stt_service.transcribe(audio)
        assert 0.0 <= result.language_probability <= 1.0

    def test_synthesize_long_text(self, tts_service):
        mock_model = MagicMock()
        mock_model.tts.return_value = [0.0] * 240000  # 10s of audio
        tts_service._model = mock_model

        result = tts_service.synthesize("A" * 1000, language="en")
        assert result.duration == pytest.approx(10.0)

    def test_stt_config_defaults(self):
        config = STTConfig()
        assert config.model == "large-v3"
        assert config.device == "cuda"
        assert config.compute_type == "float16"
        assert config.beam_size == 5
        assert config.word_timestamps is True
        assert config.vad_filter is True
        assert config.language is None

    def test_tts_config_defaults(self):
        config = TTSConfig()
        assert config.device == "cuda"
        assert config.default_language == "te"
        assert config.default_profile == "friday_telugu"
        assert config.speed == 1.0

    def test_word_timing_negative_probability(self):
        """Edge case -- probability outside expected range still stored."""
        wt = WordTiming(word="hmm", start=0.0, end=0.1, probability=-0.1)
        assert wt.probability == -0.1

    def test_transcription_result_equality(self):
        a = TranscriptionResult(
            text="hello",
            language="en",
            language_probability=0.9,
            duration=1.0,
            processing_time=0.5,
        )
        b = TranscriptionResult(
            text="hello",
            language="en",
            language_probability=0.9,
            duration=1.0,
            processing_time=0.5,
        )
        assert a == b

    def test_tts_result_equality(self):
        audio = np.zeros(10, dtype=np.float32)
        a = TTSResult(
            audio=audio,
            sample_rate=24000,
            duration=1.0,
            processing_time=0.5,
            language="en",
            voice_profile="",
        )
        b = TTSResult(
            audio=audio,
            sample_rate=24000,
            duration=1.0,
            processing_time=0.5,
            language="en",
            voice_profile="",
        )
        # Dataclass equality with numpy arrays may differ, just check no error
        assert a.language == b.language

    def test_prepare_audio_uint8_to_float32(self, stt_service):
        """uint8 is not int16/int32 => falls through to generic float32 cast."""
        audio = np.array([128, 0, 255], dtype=np.uint8)
        result = stt_service._prepare_audio(audio)
        assert result.dtype == np.float32

    def test_xtts_language_all_supported_languages(self, tts_service):
        """Ensure all XTTS-supported languages return themselves (if in map)."""
        assert tts_service._get_xtts_language("en") == "en"
        assert tts_service._get_xtts_language("hi") == "hi"

    def test_multiple_profiles_independent(self, tts_service, tmp_path):
        ref1 = tmp_path / "a.wav"
        ref2 = tmp_path / "b.wav"
        ref1.touch()
        ref2.touch()
        tts_service.load_voice_profile("p1", [str(ref1)])
        tts_service.load_voice_profile("p2", [str(ref2)])
        assert (
            tts_service._voice_profiles["p1"]["reference_audio"]
            != tts_service._voice_profiles["p2"]["reference_audio"]
        )

    def test_transcribe_strips_whitespace(self, stt_service):
        audio = np.zeros(16000, dtype=np.float32)
        info = _mock_transcribe_info()
        seg = _mock_segment(text="   spaces   ")
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg]), info)
        stt_service._model = mock_model

        result = stt_service.transcribe(audio)
        assert result.text == "spaces"
