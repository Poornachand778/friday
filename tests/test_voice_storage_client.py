"""
Comprehensive tests for voice storage and orchestrator client modules.

Tests cover:
- voice/storage/audio_storage.py (StoredTurn, AudioStorage)
- voice/storage/training_generator.py (TrainingExample, TrainingDataGenerator)
- voice/orchestrator_client.py (VoiceChatResponse, OrchestratorClient, LocalFallbackClient)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ── Mock external dependencies before any voice imports ──────────────────────
# httpx is real (installed) and must stay real for exception handling in
# OrchestratorClient, so we import it before mocking the others.
import httpx  # noqa: E402 — must import before sys.modules manipulation

sys.modules.setdefault("sounddevice", MagicMock())
sys.modules.setdefault("soundfile", MagicMock())
sys.modules.setdefault("webrtcvad", MagicMock())

from voice.config import StorageConfig
from voice.storage.audio_storage import AudioStorage, StoredTurn
from voice.storage.training_generator import TrainingDataGenerator, TrainingExample
from voice.orchestrator_client import (
    LocalFallbackClient,
    OrchestratorClient,
    VoiceChatResponse,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ═════════════════════════════════════════════════════════════════════════════


def _make_storage_config(**overrides) -> StorageConfig:
    """Create a StorageConfig with sensible test defaults."""
    defaults = dict(
        enabled=True,
        base_path="voice/data/recordings",
        organize_by_date=True,
        retention_days=90,
        save_user_audio=True,
        save_response_audio=True,
        transcript_format="jsonl",
    )
    defaults.update(overrides)
    return StorageConfig(**defaults)


@pytest.fixture
def storage_config():
    return _make_storage_config()


@pytest.fixture
def storage(tmp_path, storage_config):
    """AudioStorage backed by a temp directory."""
    return AudioStorage(config=storage_config, base_path=tmp_path)


@pytest.fixture
def sample_audio():
    """1 second of silence at 16 kHz (float32)."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def sample_response_audio():
    """1 second of silence at 24 kHz (float32)."""
    return np.zeros(24000, dtype=np.float32)


@pytest.fixture
def sample_turn_dict():
    """A fully-populated turn dictionary."""
    return {
        "turn_id": "abc12345",
        "session_id": "sess-001",
        "turn_number": 1,
        "timestamp": "2025-06-01T12:00:00",
        "user_audio_path": "/tmp/user.wav",
        "user_audio_duration": 2.5,
        "transcript": "Hello Friday",
        "detected_language": "en",
        "response_text": "Hello Boss, how can I help?",
        "response_audio_path": "/tmp/response.wav",
        "response_audio_duration": 1.8,
        "tool_calls": [{"tool": "search", "args": {"q": "test"}}],
    }


@pytest.fixture
def sample_turn(sample_turn_dict) -> StoredTurn:
    return StoredTurn.from_dict(sample_turn_dict)


@pytest.fixture
def incomplete_turn_dict():
    """A turn missing transcript / response_text."""
    return {
        "turn_id": "inc00001",
        "session_id": "sess-002",
        "turn_number": 1,
        "timestamp": "2025-06-01T12:00:00",
        "user_audio_path": None,
        "user_audio_duration": 1.0,
        "transcript": None,
        "detected_language": None,
        "response_text": None,
        "response_audio_path": None,
        "response_audio_duration": None,
        "tool_calls": [],
    }


def _write_transcript(base: Path, turns: List[dict], subdir: str = "") -> Path:
    """Helper: write a transcripts.jsonl in base/subdir."""
    d = base / subdir if subdir else base
    d.mkdir(parents=True, exist_ok=True)
    p = d / "transcripts.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return p


# ═════════════════════════════════════════════════════════════════════════════
#  StoredTurn tests
# ═════════════════════════════════════════════════════════════════════════════


class TestStoredTurn:
    """Tests for the StoredTurn dataclass."""

    def test_from_dict_creates_instance(self, sample_turn_dict):
        turn = StoredTurn.from_dict(sample_turn_dict)
        assert isinstance(turn, StoredTurn)

    def test_from_dict_all_fields(self, sample_turn_dict):
        turn = StoredTurn.from_dict(sample_turn_dict)
        assert turn.turn_id == "abc12345"
        assert turn.session_id == "sess-001"
        assert turn.turn_number == 1
        assert turn.timestamp == "2025-06-01T12:00:00"
        assert turn.user_audio_path == "/tmp/user.wav"
        assert turn.user_audio_duration == 2.5
        assert turn.transcript == "Hello Friday"
        assert turn.detected_language == "en"
        assert turn.response_text == "Hello Boss, how can I help?"
        assert turn.response_audio_path == "/tmp/response.wav"
        assert turn.response_audio_duration == 1.8
        assert turn.tool_calls == [{"tool": "search", "args": {"q": "test"}}]

    def test_to_dict_roundtrip(self, sample_turn_dict):
        turn = StoredTurn.from_dict(sample_turn_dict)
        result = turn.to_dict()
        assert result == sample_turn_dict

    def test_to_dict_returns_dict(self, sample_turn):
        assert isinstance(sample_turn.to_dict(), dict)

    def test_from_dict_to_dict_identity(self, sample_turn_dict):
        """from_dict -> to_dict should equal the original dict."""
        assert StoredTurn.from_dict(sample_turn_dict).to_dict() == sample_turn_dict

    def test_from_dict_with_none_optionals(self, incomplete_turn_dict):
        turn = StoredTurn.from_dict(incomplete_turn_dict)
        assert turn.transcript is None
        assert turn.response_text is None
        assert turn.user_audio_path is None
        assert turn.detected_language is None

    def test_to_dict_preserves_none_values(self, incomplete_turn_dict):
        turn = StoredTurn.from_dict(incomplete_turn_dict)
        d = turn.to_dict()
        assert d["transcript"] is None
        assert d["response_text"] is None

    def test_from_dict_empty_tool_calls(self, incomplete_turn_dict):
        turn = StoredTurn.from_dict(incomplete_turn_dict)
        assert turn.tool_calls == []

    def test_from_dict_multiple_tool_calls(self, sample_turn_dict):
        sample_turn_dict["tool_calls"] = [{"tool": "a"}, {"tool": "b"}, {"tool": "c"}]
        turn = StoredTurn.from_dict(sample_turn_dict)
        assert len(turn.tool_calls) == 3

    def test_to_dict_type(self, sample_turn):
        d = sample_turn.to_dict()
        assert isinstance(d["turn_number"], int)
        assert isinstance(d["user_audio_duration"], float)
        assert isinstance(d["tool_calls"], list)


# ═════════════════════════════════════════════════════════════════════════════
#  AudioStorage tests
# ═════════════════════════════════════════════════════════════════════════════


class TestAudioStorage:
    """Tests for AudioStorage class."""

    def test_init_creates_base_path(self, tmp_path):
        base = tmp_path / "new_storage"
        config = _make_storage_config()
        AudioStorage(config=config, base_path=base)
        assert base.exists()

    def test_init_with_existing_path(self, tmp_path, storage_config):
        """Should not fail when directory already exists."""
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        assert s.base_path == tmp_path

    def test_init_stores_config(self, storage):
        assert isinstance(storage.config, StorageConfig)

    # ── _get_date_dir ────────────────────────────────────────────────────

    def test_get_date_dir_organize_by_date_true(self, tmp_path):
        config = _make_storage_config(organize_by_date=True)
        s = AudioStorage(config=config, base_path=tmp_path)
        now = datetime(2025, 7, 15, 10, 30)
        result = s._get_date_dir(now)
        assert result == tmp_path / "2025" / "07" / "15"
        assert result.exists()

    def test_get_date_dir_organize_by_date_false(self, tmp_path):
        config = _make_storage_config(organize_by_date=False)
        s = AudioStorage(config=config, base_path=tmp_path)
        result = s._get_date_dir(datetime(2025, 7, 15))
        assert result == tmp_path

    def test_get_date_dir_creates_subdirectory(self, tmp_path):
        config = _make_storage_config(organize_by_date=True)
        s = AudioStorage(config=config, base_path=tmp_path)
        d = s._get_date_dir(datetime(2024, 1, 1))
        assert d.is_dir()

    def test_get_date_dir_default_datetime(self, tmp_path, storage_config):
        """When dt=None, uses datetime.now()."""
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        d = s._get_date_dir()  # no argument
        assert d.is_dir()

    # ── _get_transcript_path ─────────────────────────────────────────────

    def test_get_transcript_path_returns_jsonl(self, tmp_path, storage_config):
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        p = s._get_transcript_path(datetime(2025, 3, 10))
        assert p.name == "transcripts.jsonl"

    def test_get_transcript_path_in_date_dir(self, tmp_path):
        config = _make_storage_config(organize_by_date=True)
        s = AudioStorage(config=config, base_path=tmp_path)
        p = s._get_transcript_path(datetime(2025, 3, 10))
        assert "2025/03/10" in str(p)

    # ── store_user_audio ─────────────────────────────────────────────────

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_returns_stored_turn(self, mock_sf, storage, sample_audio):
        turn = storage.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio, transcript="hi"
        )
        assert isinstance(turn, StoredTurn)

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_turn_fields(self, mock_sf, storage, sample_audio):
        turn = storage.store_user_audio(
            session_id="s1",
            turn_number=2,
            audio=sample_audio,
            transcript="testing",
            language="en",
        )
        assert turn.session_id == "s1"
        assert turn.turn_number == 2
        assert turn.transcript == "testing"
        assert turn.detected_language == "en"
        assert turn.response_text is None
        assert turn.tool_calls == []

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_calls_sf_write(self, mock_sf, storage, sample_audio):
        storage.store_user_audio(session_id="s1", turn_number=1, audio=sample_audio)
        mock_sf.write.assert_called_once()

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_creates_transcript(self, mock_sf, storage, sample_audio):
        storage.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio, transcript="hello"
        )
        # Find transcript file
        jsonl_files = list(storage.base_path.rglob("transcripts.jsonl"))
        assert len(jsonl_files) == 1
        with open(jsonl_files[0]) as f:
            data = json.loads(f.readline())
        assert data["transcript"] == "hello"

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_save_disabled(self, mock_sf, tmp_path, sample_audio):
        config = _make_storage_config(save_user_audio=False)
        s = AudioStorage(config=config, base_path=tmp_path)
        turn = s.store_user_audio(session_id="s1", turn_number=1, audio=sample_audio)
        assert turn.user_audio_path is None
        mock_sf.write.assert_not_called()

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_duration_calculated(self, mock_sf, storage, sample_audio):
        turn = storage.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio, sample_rate=16000
        )
        assert turn.user_audio_duration == pytest.approx(1.0)

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_duration_when_save_disabled(
        self, mock_sf, tmp_path, sample_audio
    ):
        config = _make_storage_config(save_user_audio=False)
        s = AudioStorage(config=config, base_path=tmp_path)
        turn = s.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio, sample_rate=16000
        )
        assert turn.user_audio_duration == pytest.approx(1.0)

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_generates_unique_turn_id(
        self, mock_sf, storage, sample_audio
    ):
        t1 = storage.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio
        )
        t2 = storage.store_user_audio(
            session_id="s1", turn_number=2, audio=sample_audio
        )
        assert t1.turn_id != t2.turn_id

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_turn_id_length(self, mock_sf, storage, sample_audio):
        turn = storage.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio
        )
        assert len(turn.turn_id) == 8

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_audio_path_contains_session(
        self, mock_sf, storage, sample_audio
    ):
        turn = storage.store_user_audio(
            session_id="mysess", turn_number=1, audio=sample_audio
        )
        assert "mysess" in turn.user_audio_path

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_audio_path_wav(self, mock_sf, storage, sample_audio):
        turn = storage.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio
        )
        assert turn.user_audio_path.endswith(".wav")

    @patch("voice.storage.audio_storage.sf")
    def test_store_user_audio_timestamp_iso(self, mock_sf, storage, sample_audio):
        turn = storage.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio
        )
        # Should be parseable ISO format
        datetime.fromisoformat(turn.timestamp)

    # ── store_response_audio ─────────────────────────────────────────────

    @patch("voice.storage.audio_storage.sf")
    def test_store_response_audio_returns_path(
        self, mock_sf, storage, sample_audio, sample_response_audio
    ):
        turn = storage.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio
        )
        result = storage.store_response_audio(
            turn_id=turn.turn_id,
            session_id="s1",
            turn_number=1,
            audio=sample_response_audio,
            text="Hello Boss",
        )
        assert isinstance(result, Path)

    @patch("voice.storage.audio_storage.sf")
    def test_store_response_audio_calls_sf_write(
        self, mock_sf, storage, sample_audio, sample_response_audio
    ):
        turn = storage.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio
        )
        storage.store_response_audio(
            turn_id=turn.turn_id,
            session_id="s1",
            turn_number=1,
            audio=sample_response_audio,
        )
        # sf.write called twice: once for user, once for response
        assert mock_sf.write.call_count == 2

    @patch("voice.storage.audio_storage.sf")
    def test_store_response_audio_updates_transcript(
        self, mock_sf, storage, sample_audio, sample_response_audio
    ):
        turn = storage.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio, transcript="hi"
        )
        storage.store_response_audio(
            turn_id=turn.turn_id,
            session_id="s1",
            turn_number=1,
            audio=sample_response_audio,
            text="Hello Boss",
            tool_calls=[{"tool": "greet"}],
        )
        jsonl_files = list(storage.base_path.rglob("transcripts.jsonl"))
        with open(jsonl_files[0]) as f:
            data = json.loads(f.readline())
        assert data["response_text"] == "Hello Boss"
        assert data["tool_calls"] == [{"tool": "greet"}]

    @patch("voice.storage.audio_storage.sf")
    def test_store_response_audio_save_disabled(
        self, mock_sf, tmp_path, sample_audio, sample_response_audio
    ):
        config = _make_storage_config(save_response_audio=False)
        s = AudioStorage(config=config, base_path=tmp_path)
        turn = s.store_user_audio(session_id="s1", turn_number=1, audio=sample_audio)
        result = s.store_response_audio(
            turn_id=turn.turn_id,
            session_id="s1",
            turn_number=1,
            audio=sample_response_audio,
        )
        # sf.write only called once (user audio), not for response
        assert mock_sf.write.call_count == 1

    @patch("voice.storage.audio_storage.sf")
    def test_store_response_audio_duration(
        self, mock_sf, storage, sample_audio, sample_response_audio
    ):
        turn = storage.store_user_audio(
            session_id="s1", turn_number=1, audio=sample_audio
        )
        storage.store_response_audio(
            turn_id=turn.turn_id,
            session_id="s1",
            turn_number=1,
            audio=sample_response_audio,
            sample_rate=24000,
            text="ok",
        )
        jsonl_files = list(storage.base_path.rglob("transcripts.jsonl"))
        with open(jsonl_files[0]) as f:
            data = json.loads(f.readline())
        assert data["response_audio_duration"] == pytest.approx(1.0)

    # ── _append_transcript ───────────────────────────────────────────────

    def test_append_transcript_writes_jsonl(self, storage, sample_turn):
        now = datetime.now()
        storage._append_transcript(sample_turn, now)
        p = storage._get_transcript_path(now)
        assert p.exists()
        with open(p) as f:
            data = json.loads(f.readline())
        assert data["turn_id"] == sample_turn.turn_id

    def test_append_transcript_multiple_turns(self, storage, sample_turn_dict):
        now = datetime.now()
        for i in range(5):
            d = dict(sample_turn_dict, turn_id=f"t{i}", turn_number=i)
            storage._append_transcript(StoredTurn.from_dict(d), now)

        p = storage._get_transcript_path(now)
        with open(p) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 5

    # ── _update_transcript_response ──────────────────────────────────────

    def test_update_transcript_response_updates_turn(self, storage, sample_turn):
        now = datetime.now()
        storage._append_transcript(sample_turn, now)
        storage._update_transcript_response(
            turn_id=sample_turn.turn_id,
            response_text="Updated response",
            response_audio_path="/tmp/resp.wav",
            response_audio_duration=2.0,
            tool_calls=[{"tool": "x"}],
            dt=now,
        )
        p = storage._get_transcript_path(now)
        with open(p) as f:
            data = json.loads(f.readline())
        assert data["response_text"] == "Updated response"
        assert data["response_audio_path"] == "/tmp/resp.wav"
        assert data["response_audio_duration"] == 2.0
        assert data["tool_calls"] == [{"tool": "x"}]

    def test_update_transcript_response_no_file(self, storage):
        """Should not raise if transcript file does not exist."""
        storage._update_transcript_response(
            turn_id="nonexistent",
            response_text="x",
            response_audio_path=None,
            response_audio_duration=None,
            tool_calls=[],
            dt=datetime(2099, 1, 1),
        )

    def test_update_transcript_response_wrong_turn_id(self, storage, sample_turn):
        now = datetime.now()
        storage._append_transcript(sample_turn, now)
        storage._update_transcript_response(
            turn_id="wrong_id",
            response_text="Should not update",
            response_audio_path=None,
            response_audio_duration=None,
            tool_calls=[],
            dt=now,
        )
        p = storage._get_transcript_path(now)
        with open(p) as f:
            data = json.loads(f.readline())
        # Original response_text should remain
        assert data["response_text"] == sample_turn.response_text

    # ── get_session_turns ────────────────────────────────────────────────

    def test_get_session_turns_empty(self, storage):
        assert storage.get_session_turns("nonexistent") == []

    def test_get_session_turns_finds_turns(self, tmp_path, storage_config):
        turns = [
            {
                "turn_id": "t1",
                "session_id": "sess-A",
                "turn_number": 1,
                "timestamp": "2025-01-01T00:00:00",
                "user_audio_path": None,
                "user_audio_duration": 1.0,
                "transcript": "hi",
                "detected_language": "en",
                "response_text": "hello",
                "response_audio_path": None,
                "response_audio_duration": None,
                "tool_calls": [],
            },
            {
                "turn_id": "t2",
                "session_id": "sess-B",
                "turn_number": 1,
                "timestamp": "2025-01-01T00:01:00",
                "user_audio_path": None,
                "user_audio_duration": 1.0,
                "transcript": "bye",
                "detected_language": "en",
                "response_text": "goodbye",
                "response_audio_path": None,
                "response_audio_duration": None,
                "tool_calls": [],
            },
        ]
        _write_transcript(tmp_path, turns, "2025/01/01")
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        result = s.get_session_turns("sess-A")
        assert len(result) == 1
        assert result[0].turn_id == "t1"

    def test_get_session_turns_sorted_by_turn_number(self, tmp_path, storage_config):
        turns = [
            {
                "turn_id": "t3",
                "session_id": "sess-A",
                "turn_number": 3,
                "timestamp": "2025-01-01T00:03:00",
                "user_audio_path": None,
                "user_audio_duration": 1.0,
                "transcript": "c",
                "detected_language": "en",
                "response_text": "C",
                "response_audio_path": None,
                "response_audio_duration": None,
                "tool_calls": [],
            },
            {
                "turn_id": "t1",
                "session_id": "sess-A",
                "turn_number": 1,
                "timestamp": "2025-01-01T00:01:00",
                "user_audio_path": None,
                "user_audio_duration": 1.0,
                "transcript": "a",
                "detected_language": "en",
                "response_text": "A",
                "response_audio_path": None,
                "response_audio_duration": None,
                "tool_calls": [],
            },
        ]
        _write_transcript(tmp_path, turns, "2025/01/01")
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        result = s.get_session_turns("sess-A")
        assert [t.turn_number for t in result] == [1, 3]

    def test_get_session_turns_searches_recursively(self, tmp_path, storage_config):
        t1 = {
            "turn_id": "t1",
            "session_id": "sess-X",
            "turn_number": 1,
            "timestamp": "2025-01-01T00:00:00",
            "user_audio_path": None,
            "user_audio_duration": 1.0,
            "transcript": "a",
            "detected_language": "en",
            "response_text": "A",
            "response_audio_path": None,
            "response_audio_duration": None,
            "tool_calls": [],
        }
        t2 = {
            "turn_id": "t2",
            "session_id": "sess-X",
            "turn_number": 2,
            "timestamp": "2025-01-02T00:00:00",
            "user_audio_path": None,
            "user_audio_duration": 1.0,
            "transcript": "b",
            "detected_language": "en",
            "response_text": "B",
            "response_audio_path": None,
            "response_audio_duration": None,
            "tool_calls": [],
        }
        _write_transcript(tmp_path, [t1], "2025/01/01")
        _write_transcript(tmp_path, [t2], "2025/01/02")
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        result = s.get_session_turns("sess-X")
        assert len(result) == 2

    # ── get_recent_turns ─────────────────────────────────────────────────

    def test_get_recent_turns_empty(self, storage):
        assert storage.get_recent_turns() == []

    def test_get_recent_turns_returns_turns(self, tmp_path, storage_config):
        turns = [
            {
                "turn_id": "t1",
                "session_id": "s1",
                "turn_number": 1,
                "timestamp": "2025-01-01T00:00:00",
                "user_audio_path": None,
                "user_audio_duration": 1.0,
                "transcript": "hello",
                "detected_language": "en",
                "response_text": "hi",
                "response_audio_path": None,
                "response_audio_duration": None,
                "tool_calls": [],
            },
        ]
        _write_transcript(tmp_path, turns)
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        result = s.get_recent_turns(limit=10)
        assert len(result) == 1

    def test_get_recent_turns_respects_limit(self, tmp_path, storage_config):
        turns = []
        for i in range(20):
            turns.append(
                {
                    "turn_id": f"t{i}",
                    "session_id": "s1",
                    "turn_number": i,
                    "timestamp": f"2025-01-01T00:{i:02d}:00",
                    "user_audio_path": None,
                    "user_audio_duration": 1.0,
                    "transcript": f"msg{i}",
                    "detected_language": "en",
                    "response_text": f"resp{i}",
                    "response_audio_path": None,
                    "response_audio_duration": None,
                    "tool_calls": [],
                }
            )
        _write_transcript(tmp_path, turns)
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        result = s.get_recent_turns(limit=5)
        assert len(result) == 5

    # ── cleanup_old_files ────────────────────────────────────────────────

    def test_cleanup_old_files_removes_old(self, tmp_path, storage_config):
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        # Create an old WAV file
        old_wav = tmp_path / "old.wav"
        old_wav.write_bytes(b"RIFF" + b"\x00" * 100)
        # Set mtime to 200 days ago
        old_time = time.time() - (200 * 86400)
        os.utime(old_wav, (old_time, old_time))
        removed = s.cleanup_old_files(days=90)
        assert removed == 1
        assert not old_wav.exists()

    def test_cleanup_old_files_keeps_recent(self, tmp_path, storage_config):
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        new_wav = tmp_path / "new.wav"
        new_wav.write_bytes(b"RIFF" + b"\x00" * 100)
        removed = s.cleanup_old_files(days=90)
        assert removed == 0
        assert new_wav.exists()

    def test_cleanup_old_files_uses_config_retention(self, tmp_path):
        config = _make_storage_config(retention_days=10)
        s = AudioStorage(config=config, base_path=tmp_path)
        old_wav = tmp_path / "old.wav"
        old_wav.write_bytes(b"RIFF" + b"\x00" * 100)
        old_time = time.time() - (15 * 86400)
        os.utime(old_wav, (old_time, old_time))
        removed = s.cleanup_old_files()
        assert removed == 1

    def test_cleanup_old_files_days_override(self, tmp_path, storage_config):
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        wav = tmp_path / "medium.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)
        old_time = time.time() - (5 * 86400)
        os.utime(wav, (old_time, old_time))
        # Default retention is 90 days, should not remove
        assert s.cleanup_old_files() == 0
        # Override to 3 days, should remove
        assert s.cleanup_old_files(days=3) == 1

    def test_cleanup_old_files_recursive(self, tmp_path, storage_config):
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        subdir = tmp_path / "2025" / "01" / "01"
        subdir.mkdir(parents=True)
        old_wav = subdir / "deep.wav"
        old_wav.write_bytes(b"RIFF" + b"\x00" * 50)
        old_time = time.time() - (200 * 86400)
        os.utime(old_wav, (old_time, old_time))
        removed = s.cleanup_old_files(days=90)
        assert removed == 1

    # ── get_storage_stats ────────────────────────────────────────────────

    def test_get_storage_stats_empty(self, storage):
        stats = storage.get_storage_stats()
        assert stats["audio_files"] == 0
        assert stats["transcript_entries"] == 0
        assert stats["total_size_mb"] == 0.0

    def test_get_storage_stats_counts_wav_files(self, tmp_path, storage_config):
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        for i in range(3):
            (tmp_path / f"file{i}.wav").write_bytes(b"\x00" * 1024)
        stats = s.get_storage_stats()
        assert stats["audio_files"] == 3

    def test_get_storage_stats_counts_transcript_entries(
        self, tmp_path, storage_config
    ):
        turns = [
            {
                "turn_id": f"t{i}",
                "session_id": "s1",
                "turn_number": i,
                "timestamp": "2025-01-01T00:00:00",
                "user_audio_path": None,
                "user_audio_duration": 1.0,
                "transcript": "x",
                "detected_language": "en",
                "response_text": "y",
                "response_audio_path": None,
                "response_audio_duration": None,
                "tool_calls": [],
            }
            for i in range(4)
        ]
        _write_transcript(tmp_path, turns)
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        stats = s.get_storage_stats()
        assert stats["transcript_entries"] == 4

    def test_get_storage_stats_includes_base_path(self, storage):
        stats = storage.get_storage_stats()
        assert stats["base_path"] == str(storage.base_path)

    def test_get_storage_stats_calculates_size(self, tmp_path, storage_config):
        s = AudioStorage(config=storage_config, base_path=tmp_path)
        (tmp_path / "big.wav").write_bytes(b"\x00" * (1024 * 1024))  # 1 MB
        stats = s.get_storage_stats()
        assert stats["total_size_mb"] >= 0.9

    # ── _save_audio ──────────────────────────────────────────────────────

    @patch("voice.storage.audio_storage.sf")
    def test_save_audio_float32(self, mock_sf, storage):
        audio = np.zeros(1000, dtype=np.float32)
        storage._save_audio(Path("/tmp/test.wav"), audio, 16000)
        mock_sf.write.assert_called_once()
        args = mock_sf.write.call_args
        assert args[0][1].dtype == np.float32

    @patch("voice.storage.audio_storage.sf")
    def test_save_audio_float64(self, mock_sf, storage):
        audio = np.zeros(1000, dtype=np.float64)
        storage._save_audio(Path("/tmp/test.wav"), audio, 16000)
        args = mock_sf.write.call_args
        assert args[0][1].dtype == np.float64

    @patch("voice.storage.audio_storage.sf")
    def test_save_audio_int16(self, mock_sf, storage):
        audio = np.zeros(1000, dtype=np.int16)
        storage._save_audio(Path("/tmp/test.wav"), audio, 16000)
        args = mock_sf.write.call_args
        assert args[0][1].dtype == np.int16

    @patch("voice.storage.audio_storage.sf")
    def test_save_audio_other_dtype_converted_to_int16(self, mock_sf, storage):
        audio = np.zeros(1000, dtype=np.int32)
        storage._save_audio(Path("/tmp/test.wav"), audio, 16000)
        args = mock_sf.write.call_args
        assert args[0][1].dtype == np.int16


# ═════════════════════════════════════════════════════════════════════════════
#  TrainingExample tests
# ═════════════════════════════════════════════════════════════════════════════


class TestTrainingExample:
    """Tests for TrainingExample dataclass."""

    def test_to_jsonl_line_valid_json(self):
        example = TrainingExample(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            metadata={"source": "voice"},
        )
        line = example.to_jsonl_line()
        data = json.loads(line)
        assert "messages" in data
        assert data["source"] == "voice"

    def test_to_jsonl_line_includes_messages(self):
        msgs = [{"role": "user", "content": "test"}]
        example = TrainingExample(messages=msgs)
        data = json.loads(example.to_jsonl_line())
        assert data["messages"] == msgs

    def test_to_jsonl_line_includes_metadata_fields(self):
        example = TrainingExample(
            messages=[{"role": "user", "content": "x"}],
            metadata={"turn_id": "t1", "language": "te"},
        )
        data = json.loads(example.to_jsonl_line())
        assert data["turn_id"] == "t1"
        assert data["language"] == "te"

    def test_to_jsonl_line_unicode(self):
        example = TrainingExample(
            messages=[{"role": "user", "content": "నమస్కారం"}],
        )
        line = example.to_jsonl_line()
        data = json.loads(line)
        assert data["messages"][0]["content"] == "నమస్కారం"

    def test_from_turn_complete(self, sample_turn):
        example = TrainingExample.from_turn(
            turn=sample_turn,
            system_prompt="You are Friday",
        )
        assert example is not None
        assert len(example.messages) == 3
        assert example.messages[0]["role"] == "system"
        assert example.messages[0]["content"] == "You are Friday"
        assert example.messages[1]["role"] == "user"
        assert example.messages[1]["content"] == "Hello Friday"
        assert example.messages[2]["role"] == "assistant"
        assert example.messages[2]["content"] == "Hello Boss, how can I help?"

    def test_from_turn_metadata(self, sample_turn):
        example = TrainingExample.from_turn(turn=sample_turn, system_prompt="sys")
        assert example.metadata["source"] == "voice"
        assert example.metadata["turn_id"] == "abc12345"
        assert example.metadata["session_id"] == "sess-001"
        assert example.metadata["language"] == "en"
        assert example.metadata["timestamp"] == "2025-06-01T12:00:00"

    def test_from_turn_incomplete_no_transcript(self, incomplete_turn_dict):
        turn = StoredTurn.from_dict(incomplete_turn_dict)
        result = TrainingExample.from_turn(turn=turn, system_prompt="sys")
        assert result is None

    def test_from_turn_incomplete_no_response(self, sample_turn_dict):
        sample_turn_dict["response_text"] = None
        turn = StoredTurn.from_dict(sample_turn_dict)
        result = TrainingExample.from_turn(turn=turn, system_prompt="sys")
        assert result is None

    def test_from_turn_with_tool_calls(self, sample_turn_dict):
        sample_turn_dict["tool_calls"] = [{"tool": "search", "args": {}}]
        turn = StoredTurn.from_dict(sample_turn_dict)
        example = TrainingExample.from_turn(
            turn=turn, system_prompt="sys", include_tool_calls=True
        )
        assert example is not None
        assert "tool_calls" in example.metadata
        assert len(example.metadata["tool_calls"]) == 1

    def test_from_turn_without_tool_calls_flag(self, sample_turn_dict):
        sample_turn_dict["tool_calls"] = [{"tool": "search"}]
        turn = StoredTurn.from_dict(sample_turn_dict)
        example = TrainingExample.from_turn(
            turn=turn, system_prompt="sys", include_tool_calls=False
        )
        assert example is not None
        # Metadata still includes tool_calls because tool_calls exist on turn
        assert "tool_calls" in example.metadata

    def test_from_turn_no_tool_calls_no_metadata_key(self, sample_turn_dict):
        sample_turn_dict["tool_calls"] = []
        turn = StoredTurn.from_dict(sample_turn_dict)
        example = TrainingExample.from_turn(turn=turn, system_prompt="sys")
        assert "tool_calls" not in example.metadata

    def test_from_turn_unknown_language(self, sample_turn_dict):
        sample_turn_dict["detected_language"] = None
        turn = StoredTurn.from_dict(sample_turn_dict)
        example = TrainingExample.from_turn(turn=turn, system_prompt="sys")
        assert example.metadata["language"] == "unknown"

    def test_default_metadata(self):
        example = TrainingExample(messages=[])
        assert example.metadata == {}


# ═════════════════════════════════════════════════════════════════════════════
#  TrainingDataGenerator tests
# ═════════════════════════════════════════════════════════════════════════════


def _make_turns(n: int, **overrides) -> List[StoredTurn]:
    """Create n complete StoredTurn objects."""
    turns = []
    for i in range(n):
        d = {
            "turn_id": f"t{i:04d}",
            "session_id": "sess-gen",
            "turn_number": i,
            "timestamp": f"2025-01-01T00:{i:02d}:00",
            "user_audio_path": None,
            "user_audio_duration": 1.0,
            "transcript": overrides.get("transcript", f"User message number {i}"),
            "detected_language": overrides.get("detected_language", "en"),
            "response_text": overrides.get(
                "response_text", f"Response message number {i} from Friday"
            ),
            "response_audio_path": None,
            "response_audio_duration": None,
            "tool_calls": overrides.get("tool_calls", []),
        }
        turns.append(StoredTurn.from_dict(d))
    return turns


class TestTrainingDataGenerator:
    """Tests for TrainingDataGenerator class."""

    def test_init_default_system_prompt(self):
        mock_storage = MagicMock(spec=AudioStorage)
        gen = TrainingDataGenerator(storage=mock_storage)
        assert "Friday" in gen.system_prompt
        assert "Boss" in gen.system_prompt

    def test_init_custom_system_prompt(self):
        mock_storage = MagicMock(spec=AudioStorage)
        gen = TrainingDataGenerator(storage=mock_storage, system_prompt="Custom prompt")
        assert gen.system_prompt == "Custom prompt"

    def test_init_stores_storage(self):
        mock_storage = MagicMock(spec=AudioStorage)
        gen = TrainingDataGenerator(storage=mock_storage)
        assert gen.storage is mock_storage

    # ── generate_examples ────────────────────────────────────────────────

    def test_generate_examples_yields_examples(self):
        mock_storage = MagicMock(spec=AudioStorage)
        mock_storage.get_recent_turns.return_value = _make_turns(3)
        gen = TrainingDataGenerator(storage=mock_storage)
        examples = list(gen.generate_examples())
        assert len(examples) == 3
        assert all(isinstance(e, TrainingExample) for e in examples)

    def test_generate_examples_filters_short_transcript(self):
        mock_storage = MagicMock(spec=AudioStorage)
        turns = _make_turns(1, transcript="hi")  # len=2 < 5
        mock_storage.get_recent_turns.return_value = turns
        gen = TrainingDataGenerator(storage=mock_storage)
        examples = list(gen.generate_examples(min_transcript_length=5))
        assert len(examples) == 0

    def test_generate_examples_keeps_long_transcript(self):
        mock_storage = MagicMock(spec=AudioStorage)
        turns = _make_turns(1, transcript="Hello Friday, how are you?")
        mock_storage.get_recent_turns.return_value = turns
        gen = TrainingDataGenerator(storage=mock_storage)
        examples = list(gen.generate_examples(min_transcript_length=5))
        assert len(examples) == 1

    def test_generate_examples_filters_short_response(self):
        mock_storage = MagicMock(spec=AudioStorage)
        turns = _make_turns(1, response_text="ok")  # len=2 < 10
        mock_storage.get_recent_turns.return_value = turns
        gen = TrainingDataGenerator(storage=mock_storage)
        examples = list(gen.generate_examples(min_response_length=10))
        assert len(examples) == 0

    def test_generate_examples_keeps_long_response(self):
        mock_storage = MagicMock(spec=AudioStorage)
        turns = _make_turns(
            1, response_text="Hello Boss, I'm here to help you with that."
        )
        mock_storage.get_recent_turns.return_value = turns
        gen = TrainingDataGenerator(storage=mock_storage)
        examples = list(gen.generate_examples(min_response_length=10))
        assert len(examples) == 1

    def test_generate_examples_filters_by_language(self):
        mock_storage = MagicMock(spec=AudioStorage)
        en_turns = _make_turns(2, detected_language="en")
        te_turns = _make_turns(1, detected_language="te")
        mock_storage.get_recent_turns.return_value = en_turns + te_turns
        gen = TrainingDataGenerator(storage=mock_storage)
        examples = list(gen.generate_examples(languages=["te"]))
        assert len(examples) == 1

    def test_generate_examples_no_language_filter(self):
        mock_storage = MagicMock(spec=AudioStorage)
        turns = _make_turns(3, detected_language="en") + _make_turns(
            2, detected_language="te"
        )
        mock_storage.get_recent_turns.return_value = turns
        gen = TrainingDataGenerator(storage=mock_storage)
        examples = list(gen.generate_examples(languages=None))
        assert len(examples) == 5

    def test_generate_examples_skips_incomplete_transcript(self):
        mock_storage = MagicMock(spec=AudioStorage)
        turns = _make_turns(1)
        turns[0] = StoredTurn(
            turn_id="x",
            session_id="s",
            turn_number=0,
            timestamp="2025-01-01T00:00:00",
            user_audio_path=None,
            user_audio_duration=1.0,
            transcript=None,
            detected_language="en",
            response_text="hello",
            response_audio_path=None,
            response_audio_duration=None,
            tool_calls=[],
        )
        mock_storage.get_recent_turns.return_value = turns
        gen = TrainingDataGenerator(storage=mock_storage)
        examples = list(gen.generate_examples())
        assert len(examples) == 0

    def test_generate_examples_skips_incomplete_response(self):
        mock_storage = MagicMock(spec=AudioStorage)
        turns = _make_turns(1)
        turns[0] = StoredTurn(
            turn_id="x",
            session_id="s",
            turn_number=0,
            timestamp="2025-01-01T00:00:00",
            user_audio_path=None,
            user_audio_duration=1.0,
            transcript="hello there",
            detected_language="en",
            response_text=None,
            response_audio_path=None,
            response_audio_duration=None,
            tool_calls=[],
        )
        mock_storage.get_recent_turns.return_value = turns
        gen = TrainingDataGenerator(storage=mock_storage)
        examples = list(gen.generate_examples())
        assert len(examples) == 0

    def test_generate_examples_multiple_language_filter(self):
        mock_storage = MagicMock(spec=AudioStorage)
        turns = (
            _make_turns(2, detected_language="en")
            + _make_turns(3, detected_language="te")
            + _make_turns(1, detected_language="hi")
        )
        mock_storage.get_recent_turns.return_value = turns
        gen = TrainingDataGenerator(storage=mock_storage)
        examples = list(gen.generate_examples(languages=["en", "te"]))
        assert len(examples) == 5

    # ── export_to_jsonl ──────────────────────────────────────────────────

    def test_export_to_jsonl_creates_file(self, tmp_path):
        mock_storage = MagicMock(spec=AudioStorage)
        mock_storage.get_recent_turns.return_value = _make_turns(5)
        gen = TrainingDataGenerator(storage=mock_storage)
        output = tmp_path / "train.jsonl"
        path, count = gen.export_to_jsonl(output_path=output)
        assert path.exists()
        assert count == 5

    def test_export_to_jsonl_valid_json_lines(self, tmp_path):
        mock_storage = MagicMock(spec=AudioStorage)
        mock_storage.get_recent_turns.return_value = _make_turns(3)
        gen = TrainingDataGenerator(storage=mock_storage)
        output = tmp_path / "train.jsonl"
        gen.export_to_jsonl(output_path=output)
        with open(output) as f:
            for line in f:
                data = json.loads(line)
                assert "messages" in data

    def test_export_to_jsonl_empty(self, tmp_path):
        mock_storage = MagicMock(spec=AudioStorage)
        mock_storage.get_recent_turns.return_value = []
        gen = TrainingDataGenerator(storage=mock_storage)
        output = tmp_path / "empty.jsonl"
        path, count = gen.export_to_jsonl(output_path=output)
        assert count == 0
        assert path.exists()

    def test_export_to_jsonl_default_path(self):
        mock_storage = MagicMock(spec=AudioStorage)
        mock_storage.get_recent_turns.return_value = _make_turns(1)
        gen = TrainingDataGenerator(storage=mock_storage)
        path, count = gen.export_to_jsonl()
        assert count == 1
        assert path.exists()
        # Clean up
        path.unlink(missing_ok=True)

    def test_export_to_jsonl_creates_parent_dirs(self, tmp_path):
        mock_storage = MagicMock(spec=AudioStorage)
        mock_storage.get_recent_turns.return_value = _make_turns(1)
        gen = TrainingDataGenerator(storage=mock_storage)
        output = tmp_path / "deep" / "nested" / "dir" / "train.jsonl"
        path, count = gen.export_to_jsonl(output_path=output)
        assert path.exists()

    def test_export_to_jsonl_with_language_filter(self, tmp_path):
        mock_storage = MagicMock(spec=AudioStorage)
        turns = _make_turns(3, detected_language="en") + _make_turns(
            2, detected_language="te"
        )
        mock_storage.get_recent_turns.return_value = turns
        gen = TrainingDataGenerator(storage=mock_storage)
        output = tmp_path / "filtered.jsonl"
        path, count = gen.export_to_jsonl(output_path=output, languages=["te"])
        assert count == 2

    # ── get_statistics ───────────────────────────────────────────────────

    def test_get_statistics_empty(self):
        mock_storage = MagicMock(spec=AudioStorage)
        mock_storage.get_recent_turns.return_value = []
        gen = TrainingDataGenerator(storage=mock_storage)
        stats = gen.get_statistics()
        assert stats["total_turns"] == 0
        assert stats["complete_turns"] == 0
        assert stats["with_tool_calls"] == 0
        assert stats["completion_rate"] == 0
        assert stats["languages"] == {}

    def test_get_statistics_complete_turns(self):
        mock_storage = MagicMock(spec=AudioStorage)
        turns = _make_turns(5)
        mock_storage.get_recent_turns.return_value = turns
        gen = TrainingDataGenerator(storage=mock_storage)
        stats = gen.get_statistics()
        assert stats["total_turns"] == 5
        assert stats["complete_turns"] == 5
        assert stats["completion_rate"] == 100.0

    def test_get_statistics_incomplete_turns(self):
        mock_storage = MagicMock(spec=AudioStorage)
        complete = _make_turns(3)
        incomplete = [
            StoredTurn(
                turn_id="inc",
                session_id="s",
                turn_number=99,
                timestamp="2025-01-01T00:00:00",
                user_audio_path=None,
                user_audio_duration=1.0,
                transcript=None,
                detected_language="en",
                response_text=None,
                response_audio_path=None,
                response_audio_duration=None,
                tool_calls=[],
            )
        ]
        mock_storage.get_recent_turns.return_value = complete + incomplete
        gen = TrainingDataGenerator(storage=mock_storage)
        stats = gen.get_statistics()
        assert stats["total_turns"] == 4
        assert stats["complete_turns"] == 3
        assert stats["completion_rate"] == 75.0

    def test_get_statistics_with_tool_calls(self):
        mock_storage = MagicMock(spec=AudioStorage)
        turns_no_tools = _make_turns(2, tool_calls=[])
        turns_with_tools = _make_turns(3, tool_calls=[{"tool": "search"}])
        mock_storage.get_recent_turns.return_value = turns_no_tools + turns_with_tools
        gen = TrainingDataGenerator(storage=mock_storage)
        stats = gen.get_statistics()
        assert stats["with_tool_calls"] == 3

    def test_get_statistics_language_counts(self):
        mock_storage = MagicMock(spec=AudioStorage)
        turns = (
            _make_turns(2, detected_language="en")
            + _make_turns(3, detected_language="te")
            + _make_turns(1, detected_language=None)
        )
        mock_storage.get_recent_turns.return_value = turns
        gen = TrainingDataGenerator(storage=mock_storage)
        stats = gen.get_statistics()
        assert stats["languages"]["en"] == 2
        assert stats["languages"]["te"] == 3
        assert stats["languages"]["unknown"] == 1

    # ── export_approved_turns ────────────────────────────────────────────

    def test_export_approved_turns_falls_back_on_import_error(self, tmp_path):
        mock_storage = MagicMock(spec=AudioStorage)
        mock_storage.get_recent_turns.return_value = _make_turns(2)
        gen = TrainingDataGenerator(storage=mock_storage)
        output = tmp_path / "approved.jsonl"
        # sqlalchemy import will fail naturally (or we patch it)
        with patch.dict("sys.modules", {"sqlalchemy": None}):
            path, count = gen.export_approved_turns(output_path=output)
        # Should fall back to export_to_jsonl
        assert path.exists()
        assert count == 2

    def test_export_approved_turns_falls_back_on_db_error(self, tmp_path):
        mock_storage = MagicMock(spec=AudioStorage)
        mock_storage.get_recent_turns.return_value = _make_turns(3)
        gen = TrainingDataGenerator(storage=mock_storage)
        output = tmp_path / "approved_fb.jsonl"
        # Patch sqlalchemy to raise on create_engine
        mock_sa = MagicMock()
        mock_sa.create_engine.side_effect = Exception("DB connection failed")
        with patch.dict(
            "sys.modules",
            {
                "sqlalchemy": mock_sa,
                "sqlalchemy.orm": MagicMock(),
                "db.voice_schema": MagicMock(),
            },
        ):
            path, count = gen.export_approved_turns(output_path=output)
        assert path.exists()
        assert count == 3

    # ── approve_turn / reject_turn ───────────────────────────────────────

    def test_approve_turn_returns_false_on_db_error(self):
        mock_storage = MagicMock(spec=AudioStorage)
        gen = TrainingDataGenerator(storage=mock_storage)
        # Will fail because sqlalchemy/db modules aren't really available
        result = gen.approve_turn("some_turn_id", quality_score=0.9)
        assert result is False

    def test_reject_turn_returns_false_on_db_error(self):
        mock_storage = MagicMock(spec=AudioStorage)
        gen = TrainingDataGenerator(storage=mock_storage)
        result = gen.reject_turn("some_turn_id", reason="bad quality")
        assert result is False


# ═════════════════════════════════════════════════════════════════════════════
#  VoiceChatResponse tests
# ═════════════════════════════════════════════════════════════════════════════


class TestVoiceChatResponse:
    """Tests for VoiceChatResponse dataclass."""

    def test_creation_with_all_fields(self):
        r = VoiceChatResponse(response="Hello Boss", context="general", turn_id=42)
        assert r.response == "Hello Boss"
        assert r.context == "general"
        assert r.turn_id == 42

    def test_response_field(self):
        r = VoiceChatResponse(response="Test", context="c", turn_id=0)
        assert r.response == "Test"

    def test_context_field(self):
        r = VoiceChatResponse(response="r", context="writers_room", turn_id=1)
        assert r.context == "writers_room"

    def test_turn_id_field(self):
        r = VoiceChatResponse(response="r", context="c", turn_id=99)
        assert r.turn_id == 99

    def test_error_context(self):
        r = VoiceChatResponse(response="error msg", context="error", turn_id=0)
        assert r.context == "error"


# ═════════════════════════════════════════════════════════════════════════════
#  OrchestratorClient tests
# ═════════════════════════════════════════════════════════════════════════════


class TestOrchestratorClient:
    """Tests for OrchestratorClient async methods."""

    def test_init_default(self):
        client = OrchestratorClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30.0
        assert client._client is None
        assert client._session_id is None

    def test_init_custom(self):
        client = OrchestratorClient(base_url="http://custom:9000", timeout=60.0)
        assert client.base_url == "http://custom:9000"
        assert client.timeout == 60.0

    @pytest.mark.asyncio
    async def test_get_client_creates_client(self):
        client = OrchestratorClient()
        async_client = await client._get_client()
        assert async_client is not None
        assert client._client is async_client

    @pytest.mark.asyncio
    async def test_get_client_reuses_client(self):
        client = OrchestratorClient()
        c1 = await client._get_client()
        c2 = await client._get_client()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_chat_success(self):
        client = OrchestratorClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Hello Boss!",
            "context": "general",
            "turn_id": 5,
            "session_id": "sess-123",
        }
        mock_response.raise_for_status = MagicMock()

        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_response
        client._client = mock_async_client

        result = await client.chat("Hello Friday")
        assert isinstance(result, VoiceChatResponse)
        assert result.response == "Hello Boss!"
        assert result.context == "general"
        assert result.turn_id == 5

    @pytest.mark.asyncio
    async def test_chat_stores_session_id(self):
        client = OrchestratorClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "ok",
            "context": "general",
            "session_id": "new-sess-456",
        }
        mock_response.raise_for_status = MagicMock()

        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_response
        client._client = mock_async_client

        await client.chat("test")
        assert client._session_id == "new-sess-456"

    @pytest.mark.asyncio
    async def test_chat_uses_stored_session_id(self):
        client = OrchestratorClient()
        client._session_id = "existing-sess"

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok", "context": "general"}
        mock_response.raise_for_status = MagicMock()

        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_response
        client._client = mock_async_client

        await client.chat("test")
        call_kwargs = mock_async_client.post.call_args
        assert call_kwargs[1]["params"]["session_id"] == "existing-sess"

    @pytest.mark.asyncio
    async def test_chat_with_location(self):
        client = OrchestratorClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok", "context": "kitchen"}
        mock_response.raise_for_status = MagicMock()

        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_response
        client._client = mock_async_client

        await client.chat("test", location="kitchen")
        call_kwargs = mock_async_client.post.call_args
        assert call_kwargs[1]["params"]["location"] == "kitchen"

    @pytest.mark.asyncio
    async def test_chat_http_status_error(self):
        client = OrchestratorClient()
        mock_async_client = AsyncMock()
        mock_request = MagicMock()
        mock_response_obj = MagicMock()
        mock_async_client.post.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=mock_request,
            response=mock_response_obj,
        )
        client._client = mock_async_client

        result = await client.chat("test")
        assert isinstance(result, VoiceChatResponse)
        assert result.context == "error"
        assert "trouble processing" in result.response

    @pytest.mark.asyncio
    async def test_chat_request_error(self):
        client = OrchestratorClient()
        mock_async_client = AsyncMock()
        mock_async_client.post.side_effect = httpx.RequestError(
            "Connection refused", request=MagicMock()
        )
        client._client = mock_async_client

        result = await client.chat("test")
        assert isinstance(result, VoiceChatResponse)
        assert result.context == "error"
        assert "can't reach my brain" in result.response

    @pytest.mark.asyncio
    async def test_chat_generic_exception(self):
        client = OrchestratorClient()
        mock_async_client = AsyncMock()
        mock_async_client.post.side_effect = ValueError("unexpected")
        client._client = mock_async_client

        result = await client.chat("test")
        assert isinstance(result, VoiceChatResponse)
        assert result.context == "error"
        assert "something went wrong" in result.response

    @pytest.mark.asyncio
    async def test_chat_default_context_and_turn_id(self):
        client = OrchestratorClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Hello",
            # no context or turn_id
        }
        mock_response.raise_for_status = MagicMock()

        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_response
        client._client = mock_async_client

        result = await client.chat("test")
        assert result.context == "general"
        assert result.turn_id == 0

    @pytest.mark.asyncio
    async def test_chat_without_location(self):
        client = OrchestratorClient()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "ok", "context": "general"}
        mock_response.raise_for_status = MagicMock()

        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_response
        client._client = mock_async_client

        await client.chat("test")
        call_kwargs = mock_async_client.post.call_args
        assert "location" not in call_kwargs[1]["params"]

    # ── health_check ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        client = OrchestratorClient()
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_async_client = AsyncMock()
        mock_async_client.get.return_value = mock_response
        client._client = mock_async_client

        result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_non_200(self):
        client = OrchestratorClient()
        mock_response = MagicMock()
        mock_response.status_code = 503

        mock_async_client = AsyncMock()
        mock_async_client.get.return_value = mock_response
        client._client = mock_async_client

        result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        client = OrchestratorClient()
        mock_async_client = AsyncMock()
        mock_async_client.get.side_effect = ConnectionError("refused")
        client._client = mock_async_client

        result = await client.health_check()
        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        client = OrchestratorClient()
        mock_async_client = AsyncMock()
        mock_async_client.get.side_effect = httpx.TimeoutException("timeout")
        client._client = mock_async_client

        result = await client.health_check()
        assert result is False

    # ── get_context ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_get_context_success(self):
        client = OrchestratorClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"current_context": "writers_room"}

        mock_async_client = AsyncMock()
        mock_async_client.get.return_value = mock_response
        client._client = mock_async_client

        result = await client.get_context()
        assert result == "writers_room"

    @pytest.mark.asyncio
    async def test_get_context_non_200(self):
        client = OrchestratorClient()
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_async_client = AsyncMock()
        mock_async_client.get.return_value = mock_response
        client._client = mock_async_client

        result = await client.get_context()
        assert result == "general"

    @pytest.mark.asyncio
    async def test_get_context_exception(self):
        client = OrchestratorClient()
        mock_async_client = AsyncMock()
        mock_async_client.get.side_effect = Exception("fail")
        client._client = mock_async_client

        result = await client.get_context()
        assert result == "general"

    @pytest.mark.asyncio
    async def test_get_context_missing_key(self):
        client = OrchestratorClient()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # No current_context key

        mock_async_client = AsyncMock()
        mock_async_client.get.return_value = mock_response
        client._client = mock_async_client

        result = await client.get_context()
        assert result == "general"

    # ── close ────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_close_closes_client(self):
        client = OrchestratorClient()
        mock_async_client = AsyncMock()
        client._client = mock_async_client

        await client.close()
        mock_async_client.aclose.assert_awaited_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self):
        client = OrchestratorClient()
        # Should not raise
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_then_get_client_creates_new(self):
        client = OrchestratorClient()
        mock_async_client = AsyncMock()
        client._client = mock_async_client

        await client.close()
        assert client._client is None

        # Getting client again should create a new one
        c2 = await client._get_client()
        assert c2 is not None
        assert c2 is not mock_async_client


# ═════════════════════════════════════════════════════════════════════════════
#  LocalFallbackClient tests
# ═════════════════════════════════════════════════════════════════════════════


class TestLocalFallbackClient:
    """Tests for LocalFallbackClient."""

    @pytest.mark.asyncio
    async def test_chat_echoes_transcript(self):
        client = LocalFallbackClient()
        result = await client.chat("Hello Friday")
        assert isinstance(result, VoiceChatResponse)
        assert "Hello Friday" in result.response

    @pytest.mark.asyncio
    async def test_chat_response_prefix(self):
        client = LocalFallbackClient()
        result = await client.chat("test message")
        assert result.response.startswith("Boss, I heard:")

    @pytest.mark.asyncio
    async def test_chat_with_location(self):
        client = LocalFallbackClient()
        result = await client.chat("hi", location="kitchen")
        assert result.context == "kitchen"

    @pytest.mark.asyncio
    async def test_chat_without_location(self):
        client = LocalFallbackClient()
        result = await client.chat("hi")
        assert result.context == "general"

    @pytest.mark.asyncio
    async def test_chat_turn_id_zero(self):
        client = LocalFallbackClient()
        result = await client.chat("hi")
        assert result.turn_id == 0

    @pytest.mark.asyncio
    async def test_chat_with_session_id(self):
        client = LocalFallbackClient()
        result = await client.chat("hi", session_id="sess-1")
        # Should still work fine; session_id is accepted but unused
        assert "hi" in result.response

    @pytest.mark.asyncio
    async def test_health_check_always_true(self):
        client = LocalFallbackClient()
        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_get_context_returns_general(self):
        client = LocalFallbackClient()
        assert await client.get_context() == "general"

    @pytest.mark.asyncio
    async def test_close_is_noop(self):
        client = LocalFallbackClient()
        # Should not raise
        await client.close()

    @pytest.mark.asyncio
    async def test_close_multiple_times(self):
        client = LocalFallbackClient()
        await client.close()
        await client.close()
        # No error means success

    @pytest.mark.asyncio
    async def test_chat_empty_transcript(self):
        client = LocalFallbackClient()
        result = await client.chat("")
        assert "Boss, I heard:" in result.response

    @pytest.mark.asyncio
    async def test_chat_unicode_transcript(self):
        client = LocalFallbackClient()
        result = await client.chat("నమస్కారం Boss")
        assert "నమస్కారం Boss" in result.response


# ═════════════════════════════════════════════════════════════════════════════
#  Integration-style tests (combining storage + generator)
# ═════════════════════════════════════════════════════════════════════════════


class TestStorageGeneratorIntegration:
    """Tests combining AudioStorage and TrainingDataGenerator."""

    @patch("voice.storage.audio_storage.sf")
    def test_store_and_generate(self, mock_sf, tmp_path):
        config = _make_storage_config(organize_by_date=False)
        storage = AudioStorage(config=config, base_path=tmp_path)
        audio = np.zeros(16000, dtype=np.float32)
        resp_audio = np.zeros(24000, dtype=np.float32)

        turn = storage.store_user_audio(
            session_id="int-sess",
            turn_number=1,
            audio=audio,
            transcript="Hello Friday, what time is it?",
            language="en",
        )
        storage.store_response_audio(
            turn_id=turn.turn_id,
            session_id="int-sess",
            turn_number=1,
            audio=resp_audio,
            text="Boss, it's 3 PM right now.",
        )

        gen = TrainingDataGenerator(storage=storage)
        examples = list(gen.generate_examples())
        assert len(examples) == 1
        assert examples[0].messages[1]["content"] == "Hello Friday, what time is it?"
        assert examples[0].messages[2]["content"] == "Boss, it's 3 PM right now."

    @patch("voice.storage.audio_storage.sf")
    def test_store_multiple_and_export(self, mock_sf, tmp_path):
        config = _make_storage_config(organize_by_date=False)
        storage = AudioStorage(config=config, base_path=tmp_path)
        audio = np.zeros(16000, dtype=np.float32)
        resp_audio = np.zeros(24000, dtype=np.float32)

        for i in range(5):
            turn = storage.store_user_audio(
                session_id="batch-sess",
                turn_number=i,
                audio=audio,
                transcript=f"Question number {i} from the user",
                language="en",
            )
            storage.store_response_audio(
                turn_id=turn.turn_id,
                session_id="batch-sess",
                turn_number=i,
                audio=resp_audio,
                text=f"Answer number {i} from Friday assistant",
            )

        gen = TrainingDataGenerator(storage=storage)
        output = tmp_path / "export.jsonl"
        path, count = gen.export_to_jsonl(output_path=output)
        assert count == 5
        with open(output) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 5

    @patch("voice.storage.audio_storage.sf")
    def test_statistics_after_storage(self, mock_sf, tmp_path):
        config = _make_storage_config(organize_by_date=False)
        storage = AudioStorage(config=config, base_path=tmp_path)
        audio = np.zeros(16000, dtype=np.float32)
        resp = np.zeros(24000, dtype=np.float32)

        for i in range(3):
            turn = storage.store_user_audio(
                session_id="stat-sess",
                turn_number=i,
                audio=audio,
                transcript=f"User message {i} here",
                language="en",
            )
            storage.store_response_audio(
                turn_id=turn.turn_id,
                session_id="stat-sess",
                turn_number=i,
                audio=resp,
                text=f"Response message {i} from Friday",
            )

        gen = TrainingDataGenerator(storage=storage)
        stats = gen.get_statistics()
        assert stats["total_turns"] == 3
        assert stats["complete_turns"] == 3
        assert stats["languages"]["en"] == 3
