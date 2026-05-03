"""
Comprehensive tests for voice/wakeword/trainer.py
===================================================

Tests for TrainingConfig dataclass and WakeWordTrainer class:
- TrainingConfig defaults and custom values
- WakeWordTrainer.__init__ directory creation
- prepare_training_dir creates subdirs and metadata
- add_positive_sample (success, resampling, float-to-int16, sample_id, ImportError)
- add_negative_sample (success, missing file)
- _update_metadata counts files correctly
- get_training_status (found and not_found)
- train (success, insufficient samples, ImportError, failure, model not created)
- list_available_models (custom + builtin)

Run with: pytest tests/test_wakeword_trainer.py -x -q --tb=short
"""

from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock heavy / unavailable C-extension dependencies BEFORE any project import
# ---------------------------------------------------------------------------
sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("webrtcvad", MagicMock())
sys.modules.setdefault("openwakeword", MagicMock())
sys.modules.setdefault("openwakeword.model", MagicMock())
sys.modules.setdefault("openwakeword.train", MagicMock())

# soundfile: we need a controllable mock since trainer.py does `import soundfile as sf`
# and checks `sf is None`. We set up a proper mock so it's "available" by default.
_sf_mock = MagicMock()
sys.modules.setdefault("soundfile", _sf_mock)

import numpy as np
import pytest

# Now safe to import project modules
import voice.wakeword.trainer as trainer_mod
from voice.wakeword.trainer import TrainingConfig, WakeWordTrainer


# ============================================================================
#  Fixtures
# ============================================================================


@pytest.fixture()
def tmp_dirs(tmp_path):
    """
    Patch TRAINING_DATA_DIR and MODELS_DIR to use tmp_path subdirectories.
    Returns (training_data_dir, models_dir).
    """
    td = tmp_path / "wakeword_training"
    md = tmp_path / "models"
    td.mkdir(parents=True, exist_ok=True)
    md.mkdir(parents=True, exist_ok=True)
    with patch.object(trainer_mod, "TRAINING_DATA_DIR", td), patch.object(
        trainer_mod, "MODELS_DIR", md
    ):
        yield td, md


@pytest.fixture()
def trainer(tmp_dirs):
    """Return a WakeWordTrainer with patched directories."""
    td, md = tmp_dirs
    t = WakeWordTrainer()
    return t


@pytest.fixture()
def prepared_trainer(trainer, tmp_dirs):
    """Return a trainer with a prepared 'hey_friday' directory."""
    trainer.prepare_training_dir("hey_friday")
    return trainer


@pytest.fixture()
def sf_available():
    """Ensure soundfile mock is available (not None) in trainer module."""
    mock_sf = MagicMock()
    with patch.object(trainer_mod, "sf", mock_sf):
        yield mock_sf


@pytest.fixture()
def sf_unavailable():
    """Simulate soundfile not being installed (sf = None)."""
    with patch.object(trainer_mod, "sf", None):
        yield


# ============================================================================
#  SECTION 1 -- TrainingConfig dataclass
# ============================================================================


class TestTrainingConfigDefaults:
    """Tests for TrainingConfig default values."""

    def test_epochs_default(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.epochs == 50

    def test_batch_size_default(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.batch_size == 64

    def test_learning_rate_default(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.learning_rate == 0.001

    def test_sample_rate_default(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.sample_rate == 16000

    def test_target_length_ms_default(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.target_length_ms == 1500

    def test_augment_noise_default(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.augment_noise is True

    def test_augment_speed_default(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.augment_speed is True

    def test_augment_pitch_default(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.augment_pitch is False

    def test_negative_samples_dir_default(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.negative_samples_dir is None

    def test_output_dir_default_is_models_dir(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.output_dir == trainer_mod.MODELS_DIR

    def test_wake_word_stored(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.wake_word == "hey friday"

    def test_output_name_stored(self):
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=Path("/tmp/pos"),
        )
        assert cfg.output_name == "hey_friday"

    def test_positive_samples_dir_stored(self):
        p = Path("/tmp/pos")
        cfg = TrainingConfig(
            wake_word="hey friday",
            output_name="hey_friday",
            positive_samples_dir=p,
        )
        assert cfg.positive_samples_dir == p


class TestTrainingConfigCustomValues:
    """Tests for TrainingConfig with custom values."""

    def test_custom_epochs(self):
        cfg = TrainingConfig(
            wake_word="w",
            output_name="w",
            positive_samples_dir=Path("/tmp"),
            epochs=100,
        )
        assert cfg.epochs == 100

    def test_custom_batch_size(self):
        cfg = TrainingConfig(
            wake_word="w",
            output_name="w",
            positive_samples_dir=Path("/tmp"),
            batch_size=32,
        )
        assert cfg.batch_size == 32

    def test_custom_learning_rate(self):
        cfg = TrainingConfig(
            wake_word="w",
            output_name="w",
            positive_samples_dir=Path("/tmp"),
            learning_rate=0.0005,
        )
        assert cfg.learning_rate == 0.0005

    def test_custom_sample_rate(self):
        cfg = TrainingConfig(
            wake_word="w",
            output_name="w",
            positive_samples_dir=Path("/tmp"),
            sample_rate=48000,
        )
        assert cfg.sample_rate == 48000

    def test_custom_target_length_ms(self):
        cfg = TrainingConfig(
            wake_word="w",
            output_name="w",
            positive_samples_dir=Path("/tmp"),
            target_length_ms=2000,
        )
        assert cfg.target_length_ms == 2000

    def test_custom_augment_flags(self):
        cfg = TrainingConfig(
            wake_word="w",
            output_name="w",
            positive_samples_dir=Path("/tmp"),
            augment_noise=False,
            augment_speed=False,
            augment_pitch=True,
        )
        assert cfg.augment_noise is False
        assert cfg.augment_speed is False
        assert cfg.augment_pitch is True

    def test_custom_negative_samples_dir(self):
        neg = Path("/tmp/neg")
        cfg = TrainingConfig(
            wake_word="w",
            output_name="w",
            positive_samples_dir=Path("/tmp"),
            negative_samples_dir=neg,
        )
        assert cfg.negative_samples_dir == neg

    def test_custom_output_dir(self, tmp_path):
        cfg = TrainingConfig(
            wake_word="w",
            output_name="w",
            positive_samples_dir=Path("/tmp"),
            output_dir=tmp_path,
        )
        assert cfg.output_dir == tmp_path

    def test_is_dataclass(self):
        f = fields(TrainingConfig)
        names = {fld.name for fld in f}
        expected = {
            "wake_word",
            "output_name",
            "positive_samples_dir",
            "negative_samples_dir",
            "epochs",
            "batch_size",
            "learning_rate",
            "sample_rate",
            "target_length_ms",
            "augment_noise",
            "augment_speed",
            "augment_pitch",
            "output_dir",
        }
        assert names == expected

    def test_all_custom_values(self, tmp_path):
        cfg = TrainingConfig(
            wake_word="wake up daddy",
            output_name="wake_daddy",
            positive_samples_dir=tmp_path / "pos",
            negative_samples_dir=tmp_path / "neg",
            epochs=200,
            batch_size=128,
            learning_rate=0.01,
            sample_rate=8000,
            target_length_ms=3000,
            augment_noise=False,
            augment_speed=False,
            augment_pitch=True,
            output_dir=tmp_path / "out",
        )
        assert cfg.wake_word == "wake up daddy"
        assert cfg.output_name == "wake_daddy"
        assert cfg.epochs == 200
        assert cfg.batch_size == 128
        assert cfg.learning_rate == 0.01
        assert cfg.sample_rate == 8000
        assert cfg.target_length_ms == 3000

    def test_output_dir_default_factory_creates_new_each_time(self):
        """Each TrainingConfig instance should get its own output_dir default."""
        cfg1 = TrainingConfig(
            wake_word="a",
            output_name="a",
            positive_samples_dir=Path("/tmp"),
        )
        cfg2 = TrainingConfig(
            wake_word="b",
            output_name="b",
            positive_samples_dir=Path("/tmp"),
        )
        # They should both equal MODELS_DIR
        assert cfg1.output_dir == cfg2.output_dir


# ============================================================================
#  SECTION 2 -- WakeWordTrainer.__init__
# ============================================================================


class TestWakeWordTrainerInit:
    """Tests for WakeWordTrainer initialization."""

    def test_init_creates_training_data_dir(self, tmp_dirs):
        td, md = tmp_dirs
        # Remove first to verify it gets recreated
        import shutil

        shutil.rmtree(td)
        assert not td.exists()
        t = WakeWordTrainer()
        assert t.training_data_dir.exists()

    def test_init_creates_models_dir(self, tmp_dirs):
        td, md = tmp_dirs
        import shutil

        shutil.rmtree(md)
        assert not md.exists()
        t = WakeWordTrainer()
        assert t.models_dir.exists()

    def test_init_sets_training_data_dir(self, tmp_dirs):
        td, md = tmp_dirs
        t = WakeWordTrainer()
        assert t.training_data_dir == td

    def test_init_sets_models_dir(self, tmp_dirs):
        td, md = tmp_dirs
        t = WakeWordTrainer()
        assert t.models_dir == md

    def test_init_idempotent(self, tmp_dirs):
        """Creating multiple trainers should not fail."""
        WakeWordTrainer()
        WakeWordTrainer()

    def test_init_existing_dirs_ok(self, tmp_dirs):
        """Directories already exist before init -- should not raise."""
        td, md = tmp_dirs
        assert td.exists()
        assert md.exists()
        t = WakeWordTrainer()
        assert t.training_data_dir.exists()
        assert t.models_dir.exists()


# ============================================================================
#  SECTION 3 -- prepare_training_dir
# ============================================================================


class TestPrepareTrainingDir:
    """Tests for prepare_training_dir method."""

    def test_returns_path(self, trainer, tmp_dirs):
        td, _ = tmp_dirs
        result = trainer.prepare_training_dir("hey_friday")
        assert isinstance(result, Path)
        assert result == td / "hey_friday"

    def test_creates_positive_subdir(self, trainer, tmp_dirs):
        td, _ = tmp_dirs
        trainer.prepare_training_dir("hey_friday")
        assert (td / "hey_friday" / "positive").is_dir()

    def test_creates_negative_subdir(self, trainer, tmp_dirs):
        td, _ = tmp_dirs
        trainer.prepare_training_dir("hey_friday")
        assert (td / "hey_friday" / "negative").is_dir()

    def test_creates_metadata_json(self, trainer, tmp_dirs):
        td, _ = tmp_dirs
        trainer.prepare_training_dir("hey_friday")
        meta_path = td / "hey_friday" / "metadata.json"
        assert meta_path.exists()

    def test_metadata_initial_content(self, trainer, tmp_dirs):
        td, _ = tmp_dirs
        trainer.prepare_training_dir("hey_friday")
        meta_path = td / "hey_friday" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["wake_word_name"] == "hey_friday"
        assert meta["positive_samples"] == 0
        assert meta["negative_samples"] == 0
        assert meta["status"] == "collecting"

    def test_metadata_wake_word_name_matches(self, trainer, tmp_dirs):
        td, _ = tmp_dirs
        trainer.prepare_training_dir("wake_daddy")
        meta_path = td / "wake_daddy" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["wake_word_name"] == "wake_daddy"

    def test_idempotent_call(self, trainer, tmp_dirs):
        """Calling prepare_training_dir twice should not fail."""
        trainer.prepare_training_dir("hey_friday")
        trainer.prepare_training_dir("hey_friday")
        td, _ = tmp_dirs
        assert (td / "hey_friday" / "positive").is_dir()

    def test_different_wake_words(self, trainer, tmp_dirs):
        td, _ = tmp_dirs
        trainer.prepare_training_dir("hey_friday")
        trainer.prepare_training_dir("ok_friday")
        assert (td / "hey_friday").is_dir()
        assert (td / "ok_friday").is_dir()

    def test_metadata_is_valid_json(self, trainer, tmp_dirs):
        td, _ = tmp_dirs
        trainer.prepare_training_dir("test_word")
        meta_path = td / "test_word" / "metadata.json"
        with open(meta_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)


# ============================================================================
#  SECTION 4 -- add_positive_sample
# ============================================================================


class TestAddPositiveSample:
    """Tests for add_positive_sample method."""

    def test_success_returns_path(self, prepared_trainer, tmp_dirs, sf_available):
        td, _ = tmp_dirs
        audio = np.zeros(16000, dtype=np.int16)
        result = prepared_trainer.add_positive_sample("hey_friday", audio)
        assert isinstance(result, Path)
        assert "hey_friday" in str(result)
        assert result.name.endswith(".wav")

    def test_default_filename_sequential(
        self, prepared_trainer, tmp_dirs, sf_available
    ):
        td, _ = tmp_dirs
        audio = np.zeros(16000, dtype=np.int16)
        p = prepared_trainer.add_positive_sample("hey_friday", audio)
        assert p.name == "sample_0000.wav"

    def test_sequential_numbering(self, prepared_trainer, tmp_dirs, sf_available):
        td, _ = tmp_dirs
        audio = np.zeros(16000, dtype=np.int16)
        # Create a fake existing sample to bump the counter
        pos_dir = td / "hey_friday" / "positive"
        (pos_dir / "sample_0000.wav").write_bytes(b"\x00")
        p = prepared_trainer.add_positive_sample("hey_friday", audio)
        assert p.name == "sample_0001.wav"

    def test_custom_sample_id(self, prepared_trainer, tmp_dirs, sf_available):
        td, _ = tmp_dirs
        audio = np.zeros(16000, dtype=np.int16)
        p = prepared_trainer.add_positive_sample(
            "hey_friday", audio, sample_id="custom_001"
        )
        assert p.name == "custom_001.wav"

    def test_calls_sf_write(self, prepared_trainer, tmp_dirs, sf_available):
        audio = np.zeros(16000, dtype=np.int16)
        prepared_trainer.add_positive_sample("hey_friday", audio)
        sf_available.write.assert_called_once()

    def test_sf_write_called_with_16000(self, prepared_trainer, tmp_dirs, sf_available):
        audio = np.zeros(16000, dtype=np.int16)
        prepared_trainer.add_positive_sample("hey_friday", audio)
        call_args = sf_available.write.call_args
        assert call_args[0][2] == 16000  # sample_rate argument

    def test_resampling_from_8000(self, prepared_trainer, tmp_dirs, sf_available):
        """Audio at 8kHz should be resampled to 16kHz."""
        audio = np.zeros(8000, dtype=np.int16)  # 1 second at 8kHz
        prepared_trainer.add_positive_sample("hey_friday", audio, sample_rate=8000)
        sf_available.write.assert_called_once()
        written_audio = sf_available.write.call_args[0][1]
        # Resampled from 8000 to 16000 => length should be ~16000
        assert len(written_audio) == 16000

    def test_resampling_from_44100(self, prepared_trainer, tmp_dirs, sf_available):
        """Audio at 44.1kHz should be resampled to 16kHz."""
        audio = np.zeros(44100, dtype=np.int16)  # 1 second at 44.1kHz
        prepared_trainer.add_positive_sample("hey_friday", audio, sample_rate=44100)
        sf_available.write.assert_called_once()
        written_audio = sf_available.write.call_args[0][1]
        expected_length = int(44100 * (16000 / 44100))
        assert len(written_audio) == expected_length

    def test_no_resampling_at_16000(self, prepared_trainer, tmp_dirs, sf_available):
        """Audio already at 16kHz should not be resampled."""
        audio = np.arange(16000, dtype=np.int16)
        prepared_trainer.add_positive_sample("hey_friday", audio, sample_rate=16000)
        written_audio = sf_available.write.call_args[0][1]
        assert len(written_audio) == 16000

    def test_float_to_int16_conversion(self, prepared_trainer, tmp_dirs, sf_available):
        """Float audio with max <= 1.0 should be scaled to int16."""
        audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        prepared_trainer.add_positive_sample("hey_friday", audio)
        written_audio = sf_available.write.call_args[0][1]
        assert written_audio.dtype == np.int16
        # 0.5 * 32767 = 16383
        assert written_audio[1] == 16383

    def test_float_large_values_to_int16(
        self, prepared_trainer, tmp_dirs, sf_available
    ):
        """Float audio with max > 1.0 should be cast directly to int16."""
        audio = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        prepared_trainer.add_positive_sample("hey_friday", audio)
        written_audio = sf_available.write.call_args[0][1]
        assert written_audio.dtype == np.int16

    def test_int16_audio_not_converted(self, prepared_trainer, tmp_dirs, sf_available):
        """int16 audio should pass through without conversion."""
        audio = np.array([100, 200, 300], dtype=np.int16)
        prepared_trainer.add_positive_sample("hey_friday", audio)
        written_audio = sf_available.write.call_args[0][1]
        assert written_audio.dtype == np.int16
        np.testing.assert_array_equal(written_audio, audio)

    def test_soundfile_import_error(self, prepared_trainer, tmp_dirs, sf_unavailable):
        """When soundfile is not installed, should raise ImportError."""
        audio = np.zeros(16000, dtype=np.int16)
        with pytest.raises(ImportError, match="soundfile required"):
            prepared_trainer.add_positive_sample("hey_friday", audio)

    def test_updates_metadata(self, prepared_trainer, tmp_dirs, sf_available):
        td, _ = tmp_dirs
        audio = np.zeros(16000, dtype=np.int16)
        prepared_trainer.add_positive_sample("hey_friday", audio)
        meta_path = td / "hey_friday" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        # sf.write is mocked so no actual file is created,
        # _update_metadata counts *.wav files on disk
        assert "positive_samples" in meta

    def test_creates_positive_dir_if_not_exists(self, trainer, tmp_dirs, sf_available):
        """add_positive_sample should create the positive dir even without prepare."""
        td, _ = tmp_dirs
        audio = np.zeros(16000, dtype=np.int16)
        # Don't call prepare_training_dir first
        (td / "hey_friday").mkdir(parents=True, exist_ok=True)
        trainer.add_positive_sample("hey_friday", audio)
        assert (td / "hey_friday" / "positive").is_dir()

    def test_output_path_under_positive_dir(
        self, prepared_trainer, tmp_dirs, sf_available
    ):
        td, _ = tmp_dirs
        audio = np.zeros(16000, dtype=np.int16)
        result = prepared_trainer.add_positive_sample("hey_friday", audio)
        assert result.parent == td / "hey_friday" / "positive"

    def test_sf_write_path_is_string(self, prepared_trainer, tmp_dirs, sf_available):
        """sf.write should receive a string path, not a Path object."""
        audio = np.zeros(16000, dtype=np.int16)
        prepared_trainer.add_positive_sample("hey_friday", audio)
        path_arg = sf_available.write.call_args[0][0]
        assert isinstance(path_arg, str)

    def test_resampling_preserves_content(
        self, prepared_trainer, tmp_dirs, sf_available
    ):
        """Resampled audio should still contain interpolated values."""
        audio = np.ones(8000, dtype=np.float32) * 0.5
        prepared_trainer.add_positive_sample("hey_friday", audio, sample_rate=8000)
        written_audio = sf_available.write.call_args[0][1]
        # After resampling, values should still be around 0.5 * 32767
        assert written_audio.dtype == np.int16
        assert np.all(written_audio > 0)


# ============================================================================
#  SECTION 5 -- add_negative_sample
# ============================================================================


class TestAddNegativeSample:
    """Tests for add_negative_sample method."""

    def test_success_returns_true(self, prepared_trainer, tmp_dirs, tmp_path):
        # Create a fake source audio file
        source = tmp_path / "noise.wav"
        source.write_bytes(b"\x00" * 100)
        result = prepared_trainer.add_negative_sample("hey_friday", str(source))
        assert result is True

    def test_copies_file_to_negative_dir(self, prepared_trainer, tmp_dirs, tmp_path):
        td, _ = tmp_dirs
        source = tmp_path / "noise.wav"
        source.write_bytes(b"\x00" * 100)
        prepared_trainer.add_negative_sample("hey_friday", str(source))
        dest = td / "hey_friday" / "negative" / "noise.wav"
        assert dest.exists()

    def test_missing_file_returns_false(self, prepared_trainer, tmp_dirs):
        result = prepared_trainer.add_negative_sample(
            "hey_friday", "/nonexistent/audio.wav"
        )
        assert result is False

    def test_updates_metadata(self, prepared_trainer, tmp_dirs, tmp_path):
        td, _ = tmp_dirs
        source = tmp_path / "noise.wav"
        source.write_bytes(b"\x00" * 100)
        prepared_trainer.add_negative_sample("hey_friday", str(source))
        meta_path = td / "hey_friday" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert "negative_samples" in meta

    def test_creates_negative_dir_if_not_exists(self, trainer, tmp_dirs, tmp_path):
        td, _ = tmp_dirs
        source = tmp_path / "noise.wav"
        source.write_bytes(b"\x00" * 100)
        # Create the base dir but not negative subdir
        (td / "test_ww").mkdir(parents=True, exist_ok=True)
        trainer.add_negative_sample("test_ww", str(source))
        assert (td / "test_ww" / "negative").is_dir()

    def test_preserves_filename(self, prepared_trainer, tmp_dirs, tmp_path):
        td, _ = tmp_dirs
        source = tmp_path / "my_special_noise.wav"
        source.write_bytes(b"\x00" * 100)
        prepared_trainer.add_negative_sample("hey_friday", str(source))
        dest = td / "hey_friday" / "negative" / "my_special_noise.wav"
        assert dest.exists()

    def test_multiple_negative_samples(self, prepared_trainer, tmp_dirs, tmp_path):
        td, _ = tmp_dirs
        for i in range(5):
            source = tmp_path / f"noise_{i}.wav"
            source.write_bytes(b"\x00" * 100)
            prepared_trainer.add_negative_sample("hey_friday", str(source))
        neg_dir = td / "hey_friday" / "negative"
        assert len(list(neg_dir.glob("*.wav"))) == 5

    def test_copies_file_content(self, prepared_trainer, tmp_dirs, tmp_path):
        td, _ = tmp_dirs
        content = b"\x01\x02\x03\x04\x05"
        source = tmp_path / "data.wav"
        source.write_bytes(content)
        prepared_trainer.add_negative_sample("hey_friday", str(source))
        dest = td / "hey_friday" / "negative" / "data.wav"
        assert dest.read_bytes() == content


# ============================================================================
#  SECTION 6 -- _update_metadata
# ============================================================================


class TestUpdateMetadata:
    """Tests for _update_metadata method."""

    def test_counts_positive_wav_files(self, prepared_trainer, tmp_dirs):
        td, _ = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(7):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")
        prepared_trainer._update_metadata("hey_friday")
        meta_path = td / "hey_friday" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["positive_samples"] == 7

    def test_counts_negative_wav_files(self, prepared_trainer, tmp_dirs):
        td, _ = tmp_dirs
        neg_dir = td / "hey_friday" / "negative"
        for i in range(3):
            (neg_dir / f"noise_{i}.wav").write_bytes(b"\x00")
        prepared_trainer._update_metadata("hey_friday")
        meta_path = td / "hey_friday" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["negative_samples"] == 3

    def test_ignores_non_wav_files(self, prepared_trainer, tmp_dirs):
        td, _ = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        (pos_dir / "sample.wav").write_bytes(b"\x00")
        (pos_dir / "notes.txt").write_text("notes")
        (pos_dir / "data.mp3").write_bytes(b"\x00")
        prepared_trainer._update_metadata("hey_friday")
        meta_path = td / "hey_friday" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["positive_samples"] == 1

    def test_zero_when_dirs_empty(self, prepared_trainer, tmp_dirs):
        td, _ = tmp_dirs
        prepared_trainer._update_metadata("hey_friday")
        meta_path = td / "hey_friday" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["positive_samples"] == 0
        assert meta["negative_samples"] == 0

    def test_handles_missing_positive_dir(self, trainer, tmp_dirs):
        td, _ = tmp_dirs
        # Create only the training dir with a metadata file, no subdirs
        ww_dir = td / "incomplete"
        ww_dir.mkdir(parents=True, exist_ok=True)
        trainer._update_metadata("incomplete")
        meta_path = ww_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["positive_samples"] == 0

    def test_handles_missing_negative_dir(self, trainer, tmp_dirs):
        td, _ = tmp_dirs
        ww_dir = td / "incomplete"
        (ww_dir / "positive").mkdir(parents=True, exist_ok=True)
        trainer._update_metadata("incomplete")
        meta_path = ww_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["negative_samples"] == 0

    def test_metadata_status_is_collecting(self, prepared_trainer, tmp_dirs):
        td, _ = tmp_dirs
        prepared_trainer._update_metadata("hey_friday")
        meta_path = td / "hey_friday" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["status"] == "collecting"

    def test_metadata_wake_word_name(self, prepared_trainer, tmp_dirs):
        td, _ = tmp_dirs
        prepared_trainer._update_metadata("hey_friday")
        meta_path = td / "hey_friday" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["wake_word_name"] == "hey_friday"

    def test_counts_both_positive_and_negative(self, prepared_trainer, tmp_dirs):
        td, _ = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        neg_dir = td / "hey_friday" / "negative"
        for i in range(5):
            (pos_dir / f"pos_{i}.wav").write_bytes(b"\x00")
        for i in range(3):
            (neg_dir / f"neg_{i}.wav").write_bytes(b"\x00")
        prepared_trainer._update_metadata("hey_friday")
        meta_path = td / "hey_friday" / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["positive_samples"] == 5
        assert meta["negative_samples"] == 3


# ============================================================================
#  SECTION 7 -- get_training_status
# ============================================================================


class TestGetTrainingStatus:
    """Tests for get_training_status method."""

    def test_found_returns_metadata(self, prepared_trainer, tmp_dirs):
        status = prepared_trainer.get_training_status("hey_friday")
        assert status["status"] == "collecting"
        assert status["wake_word_name"] == "hey_friday"

    def test_not_found_returns_not_found_status(self, trainer, tmp_dirs):
        status = trainer.get_training_status("nonexistent")
        assert status == {"status": "not_found"}

    def test_positive_samples_count(self, prepared_trainer, tmp_dirs):
        status = prepared_trainer.get_training_status("hey_friday")
        assert "positive_samples" in status
        assert status["positive_samples"] == 0

    def test_negative_samples_count(self, prepared_trainer, tmp_dirs):
        status = prepared_trainer.get_training_status("hey_friday")
        assert "negative_samples" in status
        assert status["negative_samples"] == 0

    def test_after_adding_samples(self, prepared_trainer, tmp_dirs):
        td, _ = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(3):
            (pos_dir / f"sample_{i}.wav").write_bytes(b"\x00")
        prepared_trainer._update_metadata("hey_friday")
        status = prepared_trainer.get_training_status("hey_friday")
        assert status["positive_samples"] == 3

    def test_returns_dict(self, prepared_trainer, tmp_dirs):
        status = prepared_trainer.get_training_status("hey_friday")
        assert isinstance(status, dict)

    def test_not_found_is_dict(self, trainer, tmp_dirs):
        status = trainer.get_training_status("missing")
        assert isinstance(status, dict)

    def test_multiple_wake_words(self, trainer, tmp_dirs):
        trainer.prepare_training_dir("word_a")
        trainer.prepare_training_dir("word_b")
        status_a = trainer.get_training_status("word_a")
        status_b = trainer.get_training_status("word_b")
        assert status_a["wake_word_name"] == "word_a"
        assert status_b["wake_word_name"] == "word_b"


# ============================================================================
#  SECTION 8 -- train
# ============================================================================


class TestTrain:
    """Tests for train method."""

    def test_success_returns_path(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(15):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        mock_train = MagicMock()

        def create_model_file(**kwargs):
            # Simulate train_model creating the output file
            Path(kwargs["output_path"]).write_bytes(b"\x00" * 100)

        mock_train.side_effect = create_model_file

        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            result = prepared_trainer.train(config)
            assert result is not None
            assert isinstance(result, Path)
            assert result.name == "hey_friday.onnx"

    def test_success_updates_metadata(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(15):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        mock_train = MagicMock()

        def create_model_file(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"\x00" * 100)

        mock_train.side_effect = create_model_file

        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            prepared_trainer.train(config)
            meta_path = td / "hey_friday" / "metadata.json"
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["status"] == "trained"
            assert "model_path" in meta

    def test_insufficient_samples_returns_none(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(5):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        mock_train = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            result = prepared_trainer.train(config)
            assert result is None

    def test_exactly_9_samples_returns_none(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(9):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        mock_train = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            result = prepared_trainer.train(config)
            assert result is None

    def test_exactly_10_samples_proceeds(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(10):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        mock_train = MagicMock()

        def create_model_file(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"\x00" * 100)

        mock_train.side_effect = create_model_file

        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            result = prepared_trainer.train(config)
            assert result is not None

    def test_import_error_returns_none(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(15):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        # Simulate ImportError on from openwakeword.train import train_model
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openwakeword.train":
                raise ImportError("No module named 'openwakeword.train'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            result = prepared_trainer.train(config)
            assert result is None

    def test_training_failure_exception_returns_none(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(15):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        mock_train = MagicMock(side_effect=RuntimeError("Training crashed"))

        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            result = prepared_trainer.train(config)
            assert result is None

    def test_model_not_created_returns_none(self, prepared_trainer, tmp_dirs):
        """train_model runs but doesn't produce an output file."""
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(15):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        mock_train = MagicMock()  # Does nothing, no file created

        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            result = prepared_trainer.train(config)
            assert result is None

    def test_train_with_negative_samples_dir(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        neg_dir = td / "hey_friday" / "negative"
        for i in range(15):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")
        for i in range(5):
            (neg_dir / f"noise_{i}.wav").write_bytes(b"\x00")

        mock_train = MagicMock()

        def create_model_file(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"\x00" * 100)
            assert kwargs["negative_dir"] == str(neg_dir)

        mock_train.side_effect = create_model_file

        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                negative_samples_dir=neg_dir,
                output_dir=md,
            )
            result = prepared_trainer.train(config)
            assert result is not None

    def test_train_without_negative_dir_passes_none(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(15):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        mock_train = MagicMock()

        def create_model_file(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"\x00" * 100)
            assert kwargs["negative_dir"] is None

        mock_train.side_effect = create_model_file

        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            prepared_trainer.train(config)

    def test_train_passes_epochs_and_batch_size(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(15):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        mock_train = MagicMock()

        def create_model_file(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"\x00" * 100)
            assert kwargs["epochs"] == 100
            assert kwargs["batch_size"] == 32

        mock_train.side_effect = create_model_file

        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
                epochs=100,
                batch_size=32,
            )
            prepared_trainer.train(config)

    def test_train_output_path_correct(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(15):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        mock_train = MagicMock()

        def create_model_file(**kwargs):
            Path(kwargs["output_path"]).write_bytes(b"\x00" * 100)

        mock_train.side_effect = create_model_file

        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            result = prepared_trainer.train(config)
            assert result == md / "hey_friday.onnx"

    def test_zero_samples_returns_none(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"

        mock_train = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            result = prepared_trainer.train(config)
            assert result is None

    def test_metadata_not_updated_on_failure(self, prepared_trainer, tmp_dirs):
        td, md = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(15):
            (pos_dir / f"sample_{i:04d}.wav").write_bytes(b"\x00")

        mock_train = MagicMock()  # No file created

        with patch.dict(
            "sys.modules",
            {
                "openwakeword.train": MagicMock(train_model=mock_train),
            },
        ):
            config = TrainingConfig(
                wake_word="hey friday",
                output_name="hey_friday",
                positive_samples_dir=pos_dir,
                output_dir=md,
            )
            prepared_trainer.train(config)
            meta_path = td / "hey_friday" / "metadata.json"
            with open(meta_path) as f:
                meta = json.load(f)
            # Status should still be collecting, not trained
            assert meta["status"] == "collecting"


# ============================================================================
#  SECTION 9 -- list_available_models
# ============================================================================


class TestListAvailableModels:
    """Tests for list_available_models method."""

    def test_builtin_models_present(self, trainer, tmp_dirs):
        models = trainer.list_available_models()
        builtin_names = {m["name"] for m in models if m["type"] == "builtin"}
        expected = {"alexa", "hey_jarvis", "hey_mycroft", "timer", "weather"}
        assert builtin_names == expected

    def test_builtin_models_have_none_path(self, trainer, tmp_dirs):
        models = trainer.list_available_models()
        for m in models:
            if m["type"] == "builtin":
                assert m["path"] is None

    def test_no_custom_models_initially(self, trainer, tmp_dirs):
        models = trainer.list_available_models()
        custom = [m for m in models if m["type"] == "custom"]
        assert len(custom) == 0

    def test_custom_onnx_model_detected(self, trainer, tmp_dirs):
        _, md = tmp_dirs
        (md / "hey_friday.onnx").write_bytes(b"\x00" * 100)
        models = trainer.list_available_models()
        custom = [m for m in models if m["type"] == "custom"]
        assert len(custom) == 1
        assert custom[0]["name"] == "hey_friday"

    def test_custom_model_has_path(self, trainer, tmp_dirs):
        _, md = tmp_dirs
        model_file = md / "hey_friday.onnx"
        model_file.write_bytes(b"\x00" * 100)
        models = trainer.list_available_models()
        custom = [m for m in models if m["type"] == "custom"]
        assert custom[0]["path"] == str(model_file)

    def test_multiple_custom_models(self, trainer, tmp_dirs):
        _, md = tmp_dirs
        for name in ["hey_friday", "wake_daddy", "ok_assistant"]:
            (md / f"{name}.onnx").write_bytes(b"\x00")
        models = trainer.list_available_models()
        custom = [m for m in models if m["type"] == "custom"]
        assert len(custom) == 3
        custom_names = {m["name"] for m in custom}
        assert custom_names == {"hey_friday", "wake_daddy", "ok_assistant"}

    def test_non_onnx_files_ignored(self, trainer, tmp_dirs):
        _, md = tmp_dirs
        (md / "model.pt").write_bytes(b"\x00")
        (md / "model.tflite").write_bytes(b"\x00")
        (md / "notes.txt").write_text("notes")
        models = trainer.list_available_models()
        custom = [m for m in models if m["type"] == "custom"]
        assert len(custom) == 0

    def test_returns_list(self, trainer, tmp_dirs):
        result = trainer.list_available_models()
        assert isinstance(result, list)

    def test_total_count_builtin_only(self, trainer, tmp_dirs):
        models = trainer.list_available_models()
        assert len(models) == 5  # 5 builtins

    def test_total_count_with_custom(self, trainer, tmp_dirs):
        _, md = tmp_dirs
        (md / "custom.onnx").write_bytes(b"\x00")
        models = trainer.list_available_models()
        assert len(models) == 6  # 5 builtins + 1 custom

    def test_model_dict_keys(self, trainer, tmp_dirs):
        models = trainer.list_available_models()
        for m in models:
            assert "name" in m
            assert "path" in m
            assert "type" in m

    def test_custom_model_type_value(self, trainer, tmp_dirs):
        _, md = tmp_dirs
        (md / "test.onnx").write_bytes(b"\x00")
        models = trainer.list_available_models()
        custom = [m for m in models if m["name"] == "test"]
        assert custom[0]["type"] == "custom"

    def test_builtin_model_type_value(self, trainer, tmp_dirs):
        models = trainer.list_available_models()
        builtin = [m for m in models if m["name"] == "alexa"]
        assert builtin[0]["type"] == "builtin"


# ============================================================================
#  SECTION 10 -- Edge cases / integration
# ============================================================================


class TestEdgeCases:
    """Additional edge-case and integration tests."""

    def test_prepare_then_get_status(self, trainer, tmp_dirs):
        trainer.prepare_training_dir("test")
        status = trainer.get_training_status("test")
        assert status["status"] == "collecting"
        assert status["positive_samples"] == 0

    def test_full_workflow_prepare_add_get(
        self, trainer, tmp_dirs, sf_available, tmp_path
    ):
        td, _ = tmp_dirs
        trainer.prepare_training_dir("workflow")
        # Add positive sample
        audio = np.zeros(16000, dtype=np.int16)
        trainer.add_positive_sample("workflow", audio)
        # Add negative sample
        src = tmp_path / "noise.wav"
        src.write_bytes(b"\x00" * 100)
        trainer.add_negative_sample("workflow", str(src))
        # Check status
        status = trainer.get_training_status("workflow")
        assert status["status"] == "collecting"
        assert status["negative_samples"] == 1

    def test_add_positive_sample_without_prepare(self, trainer, tmp_dirs, sf_available):
        """add_positive_sample should work even without calling prepare first."""
        td, _ = tmp_dirs
        audio = np.zeros(16000, dtype=np.int16)
        result = trainer.add_positive_sample("no_prepare", audio)
        assert isinstance(result, Path)

    def test_add_negative_sample_without_prepare(self, trainer, tmp_dirs, tmp_path):
        td, _ = tmp_dirs
        src = tmp_path / "test.wav"
        src.write_bytes(b"\x00")
        result = trainer.add_negative_sample("no_prepare", str(src))
        assert result is True

    def test_get_status_after_metadata_update(self, prepared_trainer, tmp_dirs):
        td, _ = tmp_dirs
        pos_dir = td / "hey_friday" / "positive"
        for i in range(12):
            (pos_dir / f"s_{i}.wav").write_bytes(b"\x00")
        prepared_trainer._update_metadata("hey_friday")
        status = prepared_trainer.get_training_status("hey_friday")
        assert status["positive_samples"] == 12

    def test_training_config_equality(self):
        cfg1 = TrainingConfig(
            wake_word="w",
            output_name="w",
            positive_samples_dir=Path("/tmp"),
        )
        cfg2 = TrainingConfig(
            wake_word="w",
            output_name="w",
            positive_samples_dir=Path("/tmp"),
        )
        assert cfg1 == cfg2

    def test_training_config_inequality(self):
        cfg1 = TrainingConfig(
            wake_word="a",
            output_name="a",
            positive_samples_dir=Path("/tmp"),
        )
        cfg2 = TrainingConfig(
            wake_word="b",
            output_name="b",
            positive_samples_dir=Path("/tmp"),
        )
        assert cfg1 != cfg2

    def test_module_level_dirs_are_path_objects(self):
        assert isinstance(trainer_mod.TRAINING_DATA_DIR, Path)
        assert isinstance(trainer_mod.MODELS_DIR, Path)

    def test_module_level_dirs_derived_from_repo_root(self):
        assert "voice" in str(trainer_mod.TRAINING_DATA_DIR)
        assert "voice" in str(trainer_mod.MODELS_DIR) or "models" in str(
            trainer_mod.MODELS_DIR
        )

    def test_prepare_multiple_then_list(self, trainer, tmp_dirs):
        trainer.prepare_training_dir("word1")
        trainer.prepare_training_dir("word2")
        # Both should have metadata
        s1 = trainer.get_training_status("word1")
        s2 = trainer.get_training_status("word2")
        assert s1["status"] == "collecting"
        assert s2["status"] == "collecting"

    def test_add_positive_sample_float64_to_int16(
        self, prepared_trainer, tmp_dirs, sf_available
    ):
        """float64 audio with max <= 1.0 should be scaled to int16."""
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float64)
        prepared_trainer.add_positive_sample("hey_friday", audio)
        written = sf_available.write.call_args[0][1]
        assert written.dtype == np.int16

    def test_add_positive_sample_int32_cast(
        self, prepared_trainer, tmp_dirs, sf_available
    ):
        """int32 audio with max > 1.0 should be cast directly to int16."""
        audio = np.array([100, 200, 300], dtype=np.int32)
        prepared_trainer.add_positive_sample("hey_friday", audio)
        written = sf_available.write.call_args[0][1]
        assert written.dtype == np.int16

    def test_custom_model_stem_name(self, trainer, tmp_dirs):
        """Model name should be stem (without extension)."""
        _, md = tmp_dirs
        (md / "my_wake_word.onnx").write_bytes(b"\x00")
        models = trainer.list_available_models()
        custom = [m for m in models if m["type"] == "custom"]
        assert custom[0]["name"] == "my_wake_word"

    def test_empty_models_dir(self, trainer, tmp_dirs):
        models = trainer.list_available_models()
        custom = [m for m in models if m["type"] == "custom"]
        assert len(custom) == 0
