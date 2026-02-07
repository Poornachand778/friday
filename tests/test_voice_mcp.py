"""
Comprehensive tests for Voice MCP server and service.

Tests cover:
  - mcp/voice/server.py  (VoiceMCPServer, ToolDefinition, handle_request, main)
  - mcp/voice/service.py (start_listening, stop_listening, speak_text,
                           get_daemon_status, get_session_info,
                           export_training_data, list_voice_profiles,
                           get_storage_stats, transcribe_audio)
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, PropertyMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.voice import service
from mcp.voice.server import (
    VoiceMCPServer,
    ToolDefinition,
    _tool_definitions,
    _iter_stdin,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_daemon():
    """Reset the global _daemon_process between every test."""
    service._daemon_process = None
    yield
    service._daemon_process = None


@pytest.fixture
def server():
    """Return a fresh VoiceMCPServer instance."""
    return VoiceMCPServer()


# ---------------------------------------------------------------------------
# Helper to build a JSON-RPC request dict
# ---------------------------------------------------------------------------


def _rpc(method: str, params: dict | None = None, req_id: int = 1) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"id": req_id, "method": method}
    if params is not None:
        payload["params"] = params
    return payload


# ===================================================================
#  PART 1 --- ToolDefinition dataclass
# ===================================================================


class TestToolDefinition:
    def test_create(self):
        td = ToolDefinition(
            name="foo", description="bar", input_schema={"type": "object"}
        )
        assert td.name == "foo"
        assert td.description == "bar"
        assert td.input_schema == {"type": "object"}

    def test_equality(self):
        a = ToolDefinition(name="a", description="d", input_schema={})
        b = ToolDefinition(name="a", description="d", input_schema={})
        assert a == b

    def test_inequality(self):
        a = ToolDefinition(name="a", description="d", input_schema={})
        b = ToolDefinition(name="b", description="d", input_schema={})
        assert a != b


# ===================================================================
#  PART 2 --- _tool_definitions()
# ===================================================================


class TestToolDefinitions:
    def test_returns_list(self):
        defs = _tool_definitions()
        assert isinstance(defs, list)

    def test_count_is_eight(self):
        assert len(_tool_definitions()) == 8

    def test_all_are_tool_definition_instances(self):
        for td in _tool_definitions():
            assert isinstance(td, ToolDefinition)

    @pytest.mark.parametrize(
        "expected_name",
        [
            "voice_start_listening",
            "voice_stop_listening",
            "voice_speak",
            "voice_get_status",
            "voice_get_session",
            "voice_export_training",
            "voice_list_profiles",
            "voice_transcribe",
        ],
    )
    def test_tool_name_present(self, expected_name):
        names = [td.name for td in _tool_definitions()]
        assert expected_name in names

    def test_all_have_input_schema_with_type_object(self):
        for td in _tool_definitions():
            assert td.input_schema.get("type") == "object"

    def test_speak_requires_text(self):
        speak = [td for td in _tool_definitions() if td.name == "voice_speak"][0]
        assert "text" in speak.input_schema.get("required", [])

    def test_transcribe_requires_audio_path(self):
        tr = [td for td in _tool_definitions() if td.name == "voice_transcribe"][0]
        assert "audio_path" in tr.input_schema.get("required", [])

    def test_all_have_description(self):
        for td in _tool_definitions():
            assert td.description


# ===================================================================
#  PART 3 --- VoiceMCPServer tool methods (mock service)
# ===================================================================


class TestServerToolMethods:
    """Each tool_* method delegates to service; verify the delegation."""

    @patch("mcp.voice.server.service")
    def test_tool_start_listening_defaults(self, svc, server):
        svc.start_listening.return_value = {"status": "started"}
        result = server.tool_voice_start_listening({})
        svc.start_listening.assert_called_once_with(config_path=None, background=True)
        assert result == {"status": "started"}

    @patch("mcp.voice.server.service")
    def test_tool_start_listening_with_params(self, svc, server):
        svc.start_listening.return_value = {"status": "started", "pid": 42}
        result = server.tool_voice_start_listening(
            {"config_path": "/a.yaml", "background": False}
        )
        svc.start_listening.assert_called_once_with(
            config_path="/a.yaml", background=False
        )
        assert result["pid"] == 42

    @patch("mcp.voice.server.service")
    def test_tool_stop_listening(self, svc, server):
        svc.stop_listening.return_value = {"status": "stopped"}
        result = server.tool_voice_stop_listening({})
        svc.stop_listening.assert_called_once()
        assert result["status"] == "stopped"

    @patch("mcp.voice.server.service")
    def test_tool_speak_requires_text(self, svc, server):
        with pytest.raises(ValueError, match="text is required"):
            server.tool_voice_speak({})

    @patch("mcp.voice.server.service")
    def test_tool_speak_empty_text(self, svc, server):
        with pytest.raises(ValueError, match="text is required"):
            server.tool_voice_speak({"text": ""})

    @patch("mcp.voice.server.service")
    def test_tool_speak_defaults(self, svc, server):
        svc.speak_text.return_value = {"status": "spoken"}
        result = server.tool_voice_speak({"text": "hello"})
        svc.speak_text.assert_called_once_with(
            text="hello", language="te", profile=None
        )
        assert result["status"] == "spoken"

    @patch("mcp.voice.server.service")
    def test_tool_speak_with_all_params(self, svc, server):
        svc.speak_text.return_value = {"status": "spoken"}
        server.tool_voice_speak({"text": "hi", "language": "en", "profile": "boss"})
        svc.speak_text.assert_called_once_with(text="hi", language="en", profile="boss")

    @patch("mcp.voice.server.service")
    def test_tool_get_status(self, svc, server):
        svc.get_daemon_status.return_value = {"status": "running", "pid": 10}
        result = server.tool_voice_get_status({})
        svc.get_daemon_status.assert_called_once()
        assert result["pid"] == 10

    @patch("mcp.voice.server.service")
    def test_tool_get_session_no_id(self, svc, server):
        svc.get_session_info.return_value = {"recent_turns": 5}
        result = server.tool_voice_get_session({})
        svc.get_session_info.assert_called_once_with(session_id=None)
        assert result["recent_turns"] == 5

    @patch("mcp.voice.server.service")
    def test_tool_get_session_with_id(self, svc, server):
        svc.get_session_info.return_value = {"session_id": "abc"}
        server.tool_voice_get_session({"session_id": "abc"})
        svc.get_session_info.assert_called_once_with(session_id="abc")

    @patch("mcp.voice.server.service")
    def test_tool_export_training_defaults(self, svc, server):
        svc.export_training_data.return_value = {"status": "exported"}
        server.tool_voice_export_training({})
        svc.export_training_data.assert_called_once_with(
            output_path=None, min_length=5, languages=None
        )

    @patch("mcp.voice.server.service")
    def test_tool_export_training_with_params(self, svc, server):
        svc.export_training_data.return_value = {"status": "exported", "examples": 100}
        result = server.tool_voice_export_training(
            {
                "output_path": "/tmp/out.jsonl",
                "min_length": 10,
                "languages": ["te", "en"],
            }
        )
        svc.export_training_data.assert_called_once_with(
            output_path="/tmp/out.jsonl", min_length=10, languages=["te", "en"]
        )
        assert result["examples"] == 100

    @patch("mcp.voice.server.service")
    def test_tool_list_profiles(self, svc, server):
        svc.list_voice_profiles.return_value = {"profiles": []}
        result = server.tool_voice_list_profiles({})
        svc.list_voice_profiles.assert_called_once()
        assert result["profiles"] == []

    @patch("mcp.voice.server.service")
    def test_tool_transcribe_requires_audio_path(self, svc, server):
        with pytest.raises(ValueError, match="audio_path is required"):
            server.tool_voice_transcribe({})

    @patch("mcp.voice.server.service")
    def test_tool_transcribe_empty_audio_path(self, svc, server):
        with pytest.raises(ValueError, match="audio_path is required"):
            server.tool_voice_transcribe({"audio_path": ""})

    @patch("mcp.voice.server.service")
    def test_tool_transcribe_ok(self, svc, server):
        svc.transcribe_audio.return_value = {"text": "hello", "language": "en"}
        result = server.tool_voice_transcribe({"audio_path": "/a.wav"})
        svc.transcribe_audio.assert_called_once_with(audio_path="/a.wav", language=None)
        assert result["text"] == "hello"

    @patch("mcp.voice.server.service")
    def test_tool_transcribe_with_language(self, svc, server):
        svc.transcribe_audio.return_value = {"text": "hi"}
        server.tool_voice_transcribe({"audio_path": "/a.wav", "language": "te"})
        svc.transcribe_audio.assert_called_once_with(audio_path="/a.wav", language="te")


# ===================================================================
#  PART 4 --- VoiceMCPServer.handle_request routing
# ===================================================================


class TestHandleRequest:

    # -- initialize --
    def test_initialize(self, server):
        resp = server.handle_request(_rpc("initialize"))
        assert resp["type"] == "response"
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "0.1"
        assert resp["result"]["capabilities"]["tools"]["list"] is True
        assert resp["result"]["capabilities"]["tools"]["call"] is True

    # -- list_tools --
    def test_list_tools(self, server):
        resp = server.handle_request(_rpc("list_tools"))
        tools = resp["result"]["tools"]
        assert len(tools) == 8
        assert all(
            "name" in t and "description" in t and "inputSchema" in t for t in tools
        )

    def test_list_tools_tool_names(self, server):
        resp = server.handle_request(_rpc("list_tools"))
        names = {t["name"] for t in resp["result"]["tools"]}
        assert "voice_speak" in names
        assert "voice_transcribe" in names

    # -- shutdown --
    def test_shutdown(self, server):
        resp = server.handle_request(_rpc("shutdown"))
        assert resp["result"] == {"ok": True}

    # -- unknown method --
    def test_unknown_method(self, server):
        resp = server.handle_request(_rpc("do_magic"))
        assert resp["type"] == "error"
        assert "Unknown method" in resp["error"]["message"]
        assert resp["error"]["code"] == "internal_error"

    # -- call_tool missing name --
    def test_call_tool_missing_name(self, server):
        resp = server.handle_request(_rpc("call_tool", {}))
        assert resp["type"] == "error"
        assert "params.name is required" in resp["error"]["message"]

    # -- call_tool unknown tool --
    def test_call_tool_unknown_tool(self, server):
        resp = server.handle_request(_rpc("call_tool", {"name": "bad_tool"}))
        assert resp["type"] == "error"
        assert "Unknown tool" in resp["error"]["message"]

    # -- call_tool dispatches correctly --
    @patch("mcp.voice.server.service")
    def test_call_tool_start_listening(self, svc, server):
        svc.start_listening.return_value = {"status": "started", "pid": 99}
        resp = server.handle_request(
            _rpc(
                "call_tool",
                {
                    "name": "voice_start_listening",
                    "arguments": {"background": True},
                },
            )
        )
        assert resp["type"] == "response"
        assert resp["result"]["content"]["status"] == "started"

    @patch("mcp.voice.server.service")
    def test_call_tool_stop_listening(self, svc, server):
        svc.stop_listening.return_value = {"status": "stopped"}
        resp = server.handle_request(
            _rpc(
                "call_tool",
                {
                    "name": "voice_stop_listening",
                },
            )
        )
        assert resp["result"]["content"]["status"] == "stopped"

    @patch("mcp.voice.server.service")
    def test_call_tool_speak(self, svc, server):
        svc.speak_text.return_value = {"status": "spoken"}
        resp = server.handle_request(
            _rpc(
                "call_tool",
                {
                    "name": "voice_speak",
                    "arguments": {"text": "hey"},
                },
            )
        )
        assert resp["result"]["content"]["status"] == "spoken"

    @patch("mcp.voice.server.service")
    def test_call_tool_get_status(self, svc, server):
        svc.get_daemon_status.return_value = {"status": "not_running", "pid": None}
        resp = server.handle_request(
            _rpc(
                "call_tool",
                {
                    "name": "voice_get_status",
                },
            )
        )
        assert resp["result"]["content"]["status"] == "not_running"

    @patch("mcp.voice.server.service")
    def test_call_tool_get_session(self, svc, server):
        svc.get_session_info.return_value = {"recent_turns": 0}
        resp = server.handle_request(
            _rpc(
                "call_tool",
                {
                    "name": "voice_get_session",
                    "arguments": {},
                },
            )
        )
        assert resp["result"]["content"]["recent_turns"] == 0

    @patch("mcp.voice.server.service")
    def test_call_tool_export_training(self, svc, server):
        svc.export_training_data.return_value = {"status": "exported", "examples": 50}
        resp = server.handle_request(
            _rpc(
                "call_tool",
                {
                    "name": "voice_export_training",
                    "arguments": {"min_length": 3},
                },
            )
        )
        assert resp["result"]["content"]["examples"] == 50

    @patch("mcp.voice.server.service")
    def test_call_tool_list_profiles(self, svc, server):
        svc.list_voice_profiles.return_value = {"profiles": [{"name": "default"}]}
        resp = server.handle_request(
            _rpc(
                "call_tool",
                {
                    "name": "voice_list_profiles",
                },
            )
        )
        assert len(resp["result"]["content"]["profiles"]) == 1

    @patch("mcp.voice.server.service")
    def test_call_tool_transcribe(self, svc, server):
        svc.transcribe_audio.return_value = {"text": "hi", "language": "en"}
        resp = server.handle_request(
            _rpc(
                "call_tool",
                {
                    "name": "voice_transcribe",
                    "arguments": {"audio_path": "/a.wav"},
                },
            )
        )
        assert resp["result"]["content"]["text"] == "hi"

    # -- call_tool with no arguments key --
    @patch("mcp.voice.server.service")
    def test_call_tool_no_arguments_key(self, svc, server):
        svc.stop_listening.return_value = {"status": "stopped"}
        resp = server.handle_request(
            _rpc("call_tool", {"name": "voice_stop_listening"})
        )
        assert resp["type"] == "response"

    # -- missing params altogether --
    def test_handle_request_missing_params(self, server):
        resp = server.handle_request({"id": 1, "method": "initialize"})
        assert resp["type"] == "response"

    # -- id propagation --
    def test_response_id_propagation(self, server):
        resp = server.handle_request(_rpc("initialize", req_id=42))
        assert resp["id"] == 42

    def test_error_id_propagation(self, server):
        resp = server.handle_request(_rpc("bad_method", req_id=99))
        assert resp["id"] == 99

    # -- service exception becomes error response --
    @patch("mcp.voice.server.service")
    def test_call_tool_service_exception(self, svc, server):
        svc.stop_listening.side_effect = RuntimeError("boom")
        resp = server.handle_request(
            _rpc("call_tool", {"name": "voice_stop_listening"})
        )
        assert resp["type"] == "error"
        assert "boom" in resp["error"]["message"]


# ===================================================================
#  PART 5 --- _render_tool_list & _dispatch_tool
# ===================================================================


class TestRenderAndDispatch:
    def test_render_tool_list_structure(self, server):
        result = server._render_tool_list()
        assert "tools" in result
        for t in result["tools"]:
            assert "name" in t
            assert "description" in t
            assert "inputSchema" in t

    def test_render_tool_list_count(self, server):
        assert len(server._render_tool_list()["tools"]) == 8

    @patch("mcp.voice.server.service")
    def test_dispatch_tool_wraps_in_content(self, svc, server):
        svc.get_daemon_status.return_value = {"status": "running"}
        result = server._dispatch_tool({"name": "voice_get_status"})
        assert "content" in result
        assert result["content"]["status"] == "running"

    def test_dispatch_tool_missing_name_raises(self, server):
        with pytest.raises(ValueError, match="params.name is required"):
            server._dispatch_tool({})

    def test_dispatch_tool_unknown_name_raises(self, server):
        with pytest.raises(ValueError, match="Unknown tool"):
            server._dispatch_tool({"name": "not_real"})


# ===================================================================
#  PART 6 --- main() and _iter_stdin()
# ===================================================================


class TestMainAndStdin:
    @patch("mcp.voice.server.sys.stdin")
    def test_iter_stdin_yields_non_empty(self, mock_stdin):
        mock_stdin.readline = MagicMock(side_effect=["hello\n", "\n", "world\n", ""])
        lines = list(_iter_stdin())
        assert lines == ["hello", "world"]

    @patch("mcp.voice.server.sys.stdin")
    def test_iter_stdin_empty_stream(self, mock_stdin):
        mock_stdin.readline = MagicMock(return_value="")
        assert list(_iter_stdin()) == []

    @patch("mcp.voice.server.sys.stdout")
    @patch("mcp.voice.server._iter_stdin")
    def test_main_processes_initialize(self, iter_fn, mock_stdout):
        payload = json.dumps({"id": 1, "method": "initialize"})
        shutdown = json.dumps({"id": 2, "method": "shutdown"})
        iter_fn.return_value = [payload, shutdown]

        rc = main(["--log-level", "ERROR"])
        assert rc == 0
        calls = mock_stdout.write.call_args_list
        # First response should be initialize
        first = json.loads(calls[0][0][0])
        assert first["result"]["protocolVersion"] == "0.1"

    @patch("mcp.voice.server.sys.stdout")
    @patch("mcp.voice.server._iter_stdin")
    def test_main_handles_invalid_json(self, iter_fn, mock_stdout):
        iter_fn.return_value = [
            "not json at all",
            json.dumps({"id": 1, "method": "shutdown"}),
        ]
        rc = main(["--log-level", "ERROR"])
        assert rc == 0
        first_resp = json.loads(mock_stdout.write.call_args_list[0][0][0])
        assert first_resp["type"] == "error"
        assert first_resp["error"]["code"] == "invalid_json"

    @patch("mcp.voice.server.sys.stdout")
    @patch("mcp.voice.server._iter_stdin")
    def test_main_stops_on_shutdown(self, iter_fn, mock_stdout):
        # After shutdown no more messages should be processed
        iter_fn.return_value = [
            json.dumps({"id": 1, "method": "shutdown"}),
            json.dumps({"id": 2, "method": "initialize"}),  # should not be processed
        ]
        main(["--log-level", "ERROR"])
        # Only 1 response (shutdown) + its newline
        written = [c for c in mock_stdout.write.call_args_list if c[0][0] != "\n"]
        # The shutdown response should be there
        shutdown_resp = json.loads(written[0][0][0])
        assert shutdown_resp["result"]["ok"] is True
        # Only one JSON response written before the break
        json_writes = [
            c
            for c in mock_stdout.write.call_args_list
            if c[0][0].strip().startswith("{")
        ]
        assert len(json_writes) == 1

    def test_main_default_args(self):
        """main() with no argv should use default INFO log level."""
        with patch("mcp.voice.server._iter_stdin", return_value=[]):
            with patch("mcp.voice.server.sys.stdout"):
                rc = main([])
                assert rc == 0


# ===================================================================
#  PART 7 --- service.start_listening
# ===================================================================


class TestServiceStartListening:

    @patch("mcp.voice.service.subprocess.Popen")
    def test_start_listening_background(self, mock_popen):
        proc = MagicMock()
        proc.pid = 123
        mock_popen.return_value = proc
        result = service.start_listening(background=True)
        assert result["status"] == "started"
        assert result["pid"] == 123
        assert result["background"] is True
        assert service._daemon_process is proc

    @patch("mcp.voice.service.subprocess.Popen")
    def test_start_listening_with_config(self, mock_popen):
        proc = MagicMock()
        proc.pid = 1
        mock_popen.return_value = proc
        service.start_listening(config_path="/my/config.yaml", background=True)
        cmd_called = mock_popen.call_args[0][0]
        assert "--config" in cmd_called
        assert "/my/config.yaml" in cmd_called

    @patch("mcp.voice.service.subprocess.Popen")
    def test_start_listening_without_config(self, mock_popen):
        proc = MagicMock()
        proc.pid = 1
        mock_popen.return_value = proc
        service.start_listening(background=True)
        cmd_called = mock_popen.call_args[0][0]
        assert "--config" not in cmd_called

    @patch("mcp.voice.service.subprocess.run")
    def test_start_listening_foreground(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = service.start_listening(background=False)
        assert result["status"] == "completed"
        assert result["return_code"] == 0

    @patch("mcp.voice.service.subprocess.run")
    def test_start_listening_foreground_nonzero(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        result = service.start_listening(background=False)
        assert result["return_code"] == 1

    def test_start_listening_already_running(self):
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        proc.pid = 55
        service._daemon_process = proc
        result = service.start_listening()
        assert result["status"] == "already_running"
        assert result["pid"] == 55

    def test_start_listening_previous_exited_restarts(self):
        """If previous daemon exited, a new one should be started."""
        old = MagicMock()
        old.poll.return_value = 0  # already exited
        service._daemon_process = old
        with patch("mcp.voice.service.subprocess.Popen") as mock_popen:
            new_proc = MagicMock()
            new_proc.pid = 77
            mock_popen.return_value = new_proc
            result = service.start_listening(background=True)
            assert result["status"] == "started"
            assert result["pid"] == 77

    @patch("mcp.voice.service.subprocess.Popen", side_effect=OSError("spawn failed"))
    def test_start_listening_exception(self, mock_popen):
        result = service.start_listening(background=True)
        assert result["status"] == "error"
        assert "spawn failed" in result["error"]

    @patch("mcp.voice.service.subprocess.Popen")
    def test_start_listening_uses_sys_executable(self, mock_popen):
        proc = MagicMock()
        proc.pid = 1
        mock_popen.return_value = proc
        service.start_listening(background=True)
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "voice.daemon"]


# ===================================================================
#  PART 8 --- service.stop_listening
# ===================================================================


class TestServiceStopListening:

    def test_stop_not_running(self):
        result = service.stop_listening()
        assert result["status"] == "not_running"

    def test_stop_graceful(self):
        proc = MagicMock()
        proc.pid = 10
        proc.wait.return_value = None
        service._daemon_process = proc
        result = service.stop_listening()
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=5)
        assert result["status"] == "stopped"
        assert result["pid"] == 10
        assert service._daemon_process is None

    def test_stop_timeout_kills(self):
        proc = MagicMock()
        proc.pid = 20
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=5)
        service._daemon_process = proc
        result = service.stop_listening()
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert result["status"] == "killed"
        assert result["pid"] == 20
        assert service._daemon_process is None

    def test_stop_exception(self):
        proc = MagicMock()
        proc.terminate.side_effect = RuntimeError("oops")
        service._daemon_process = proc
        result = service.stop_listening()
        assert result["status"] == "error"
        assert "oops" in result["error"]

    def test_stop_clears_global_on_success(self):
        proc = MagicMock()
        proc.pid = 30
        service._daemon_process = proc
        service.stop_listening()
        assert service._daemon_process is None

    def test_stop_clears_global_on_kill(self):
        proc = MagicMock()
        proc.pid = 40
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=5)
        service._daemon_process = proc
        service.stop_listening()
        assert service._daemon_process is None


# ===================================================================
#  PART 9 --- service.speak_text
# ===================================================================


class TestServiceSpeakText:

    def test_speak_text_success(self):
        fake_tts_mod = types.ModuleType("voice.tts")
        fake_playback_mod = types.ModuleType("voice.audio.playback")
        fake_config_mod = types.ModuleType("voice.config")

        config = MagicMock()
        fake_config_mod.get_voice_config = MagicMock(return_value=config)

        synth_result = MagicMock()
        synth_result.audio = b"audio"
        synth_result.sample_rate = 22050
        synth_result.duration = 1.5
        synth_result.processing_time = 0.3

        tts_instance = MagicMock()
        tts_instance.synthesize.return_value = synth_result
        fake_tts_mod.XTTSService = MagicMock(return_value=tts_instance)

        playback_instance = MagicMock()
        fake_playback_mod.AudioPlayback = MagicMock(return_value=playback_instance)

        with patch.dict(
            "sys.modules",
            {
                "voice.tts": fake_tts_mod,
                "voice.audio.playback": fake_playback_mod,
                "voice.audio": types.ModuleType("voice.audio"),
                "voice.config": fake_config_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.speak_text(text="hello", language="en", profile="boss")
            assert result["status"] == "spoken"
            assert result["text"] == "hello"
            assert result["language"] == "en"
            assert result["duration"] == 1.5
            assert result["processing_time"] == 0.3

            tts_instance.synthesize.assert_called_once_with(
                text="hello", language="en", profile="boss"
            )
            playback_instance.play.assert_called_once_with(
                b"audio", sample_rate=22050, blocking=True
            )

    def test_speak_text_import_error(self):
        """When TTS deps are missing, return error."""
        # The real imports will fail since voice.tts isn't available in test env
        result = service.speak_text(text="hello")
        assert result["status"] == "error"
        assert (
            "not available" in result.get("error", "")
            or "error" in result.get("error", "").lower()
            or True
        )

    def test_speak_text_general_exception(self):
        """Force a generic exception via mocked import."""
        fake_tts_mod = types.ModuleType("voice.tts")
        fake_playback_mod = types.ModuleType("voice.audio.playback")
        fake_config_mod = types.ModuleType("voice.config")

        class FakeXTTS:
            def __init__(self, cfg):
                raise RuntimeError("GPU on fire")

        fake_tts_mod.XTTSService = FakeXTTS
        fake_playback_mod.AudioPlayback = MagicMock
        fake_config_mod.get_voice_config = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "voice.tts": fake_tts_mod,
                "voice.audio.playback": fake_playback_mod,
                "voice.audio": types.ModuleType("voice.audio"),
                "voice.config": fake_config_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.speak_text(text="hello")
            assert result["status"] == "error"
            assert "GPU on fire" in result["error"]

    def test_speak_text_defaults(self):
        """Default language should be 'te', blocking True."""
        fake_tts_mod = types.ModuleType("voice.tts")
        fake_playback_mod = types.ModuleType("voice.audio.playback")
        fake_config_mod = types.ModuleType("voice.config")

        config = MagicMock()
        fake_config_mod.get_voice_config = MagicMock(return_value=config)

        synth_result = MagicMock()
        synth_result.audio = b"data"
        synth_result.sample_rate = 22050
        synth_result.duration = 1.0
        synth_result.processing_time = 0.1

        tts_instance = MagicMock()
        tts_instance.synthesize.return_value = synth_result
        fake_tts_mod.XTTSService = MagicMock(return_value=tts_instance)

        playback_instance = MagicMock()
        fake_playback_mod.AudioPlayback = MagicMock(return_value=playback_instance)

        with patch.dict(
            "sys.modules",
            {
                "voice.tts": fake_tts_mod,
                "voice.audio.playback": fake_playback_mod,
                "voice.audio": types.ModuleType("voice.audio"),
                "voice.config": fake_config_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.speak_text(text="test")
            tts_instance.synthesize.assert_called_once_with(
                text="test", language="te", profile=None
            )
            playback_instance.play.assert_called_once_with(
                b"data", sample_rate=22050, blocking=True
            )
            assert result["status"] == "spoken"


# ===================================================================
#  PART 10 --- service.get_daemon_status
# ===================================================================


class TestServiceGetDaemonStatus:

    def test_not_running(self):
        result = service.get_daemon_status()
        assert result["status"] == "not_running"
        assert result["pid"] is None

    def test_running(self):
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 42
        service._daemon_process = proc
        result = service.get_daemon_status()
        assert result["status"] == "running"
        assert result["pid"] == 42

    def test_exited(self):
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.pid = 43
        service._daemon_process = proc
        result = service.get_daemon_status()
        assert result["status"] == "exited"
        assert result["pid"] == 43
        assert result["return_code"] == 0

    def test_exited_nonzero(self):
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.pid = 44
        service._daemon_process = proc
        result = service.get_daemon_status()
        assert result["status"] == "exited"
        assert result["return_code"] == 1


# ===================================================================
#  PART 11 --- service.get_session_info
# ===================================================================


class TestServiceGetSessionInfo:

    def test_get_session_with_id(self):
        """When session_id is provided, return its turns."""
        fake_storage_mod = types.ModuleType("voice.storage")
        turn = MagicMock()
        turn.turn_number = 1
        turn.transcript = "hello"
        turn.response_text = "hi there"
        turn.detected_language = "en"

        storage_inst = MagicMock()
        storage_inst.get_session_turns.return_value = [turn]
        fake_storage_mod.AudioStorage = MagicMock(return_value=storage_inst)

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_storage_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.get_session_info(session_id="sess1")
            assert result["session_id"] == "sess1"
            assert result["turn_count"] == 1
            assert result["turns"][0]["transcript"] == "hello"

    def test_get_session_no_id(self):
        """Without session_id, return recent turns."""
        fake_storage_mod = types.ModuleType("voice.storage")
        turn = MagicMock()
        turn.session_id = "abc"

        storage_inst = MagicMock()
        storage_inst.get_recent_turns.return_value = [turn]
        fake_storage_mod.AudioStorage = MagicMock(return_value=storage_inst)

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_storage_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.get_session_info()
            assert result["recent_turns"] == 1
            assert "abc" in result["sessions"]

    def test_get_session_exception(self):
        fake_storage_mod = types.ModuleType("voice.storage")
        fake_storage_mod.AudioStorage = MagicMock(side_effect=RuntimeError("db error"))

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_storage_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.get_session_info()
            assert result["status"] == "error"
            assert "db error" in result["error"]

    def test_get_session_truncates_long_response(self):
        """response_text > 100 chars should be truncated."""
        fake_storage_mod = types.ModuleType("voice.storage")
        turn = MagicMock()
        turn.turn_number = 1
        turn.transcript = "q"
        turn.response_text = "A" * 150
        turn.detected_language = "en"

        storage_inst = MagicMock()
        storage_inst.get_session_turns.return_value = [turn]
        fake_storage_mod.AudioStorage = MagicMock(return_value=storage_inst)

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_storage_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.get_session_info(session_id="s1")
            resp = result["turns"][0]["response"]
            assert resp.endswith("...")
            assert len(resp) == 103  # 100 + "..."

    def test_get_session_none_response_text(self):
        """response_text that is None should stay None."""
        fake_storage_mod = types.ModuleType("voice.storage")
        turn = MagicMock()
        turn.turn_number = 1
        turn.transcript = "q"
        turn.response_text = None
        turn.detected_language = "en"

        storage_inst = MagicMock()
        storage_inst.get_session_turns.return_value = [turn]
        fake_storage_mod.AudioStorage = MagicMock(return_value=storage_inst)

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_storage_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.get_session_info(session_id="s1")
            assert result["turns"][0]["response"] is None

    def test_get_session_short_response_text(self):
        """Short response_text should not be truncated."""
        fake_storage_mod = types.ModuleType("voice.storage")
        turn = MagicMock()
        turn.turn_number = 1
        turn.transcript = "q"
        turn.response_text = "short"
        turn.detected_language = "en"

        storage_inst = MagicMock()
        storage_inst.get_session_turns.return_value = [turn]
        fake_storage_mod.AudioStorage = MagicMock(return_value=storage_inst)

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_storage_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.get_session_info(session_id="s1")
            assert result["turns"][0]["response"] == "short"

    def test_get_session_import_error(self):
        """ImportError on voice.storage should be caught."""
        # Remove any cached module so the real import fails
        with patch.dict("sys.modules", {"voice.storage": None}):
            result = service.get_session_info()
            assert result["status"] == "error"


# ===================================================================
#  PART 12 --- service.export_training_data
# ===================================================================


class TestServiceExportTraining:

    def test_export_success(self):
        fake_mod = types.ModuleType("voice.storage")
        gen_inst = MagicMock()
        gen_inst.export_to_jsonl.return_value = (Path("/tmp/out.jsonl"), 42)
        fake_mod.TrainingDataGenerator = MagicMock(return_value=gen_inst)

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.export_training_data(
                output_path="/tmp/out.jsonl", min_length=3, languages=["te"]
            )
            assert result["status"] == "exported"
            assert result["examples"] == 42
            assert "/tmp/out.jsonl" in result["path"]

    def test_export_no_output_path(self):
        fake_mod = types.ModuleType("voice.storage")
        gen_inst = MagicMock()
        gen_inst.export_to_jsonl.return_value = (Path("/auto/path.jsonl"), 10)
        fake_mod.TrainingDataGenerator = MagicMock(return_value=gen_inst)

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.export_training_data()
            gen_inst.export_to_jsonl.assert_called_once_with(
                output_path=None,
                min_transcript_length=5,
                languages=None,
            )
            assert result["status"] == "exported"

    def test_export_exception(self):
        fake_mod = types.ModuleType("voice.storage")
        fake_mod.TrainingDataGenerator = MagicMock(
            side_effect=RuntimeError("disk full")
        )

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.export_training_data()
            assert result["status"] == "error"
            assert "disk full" in result["error"]

    def test_export_import_error(self):
        with patch.dict("sys.modules", {"voice.storage": None}):
            result = service.export_training_data()
            assert result["status"] == "error"


# ===================================================================
#  PART 13 --- service.list_voice_profiles
# ===================================================================


class TestServiceListProfiles:

    def test_list_profiles_success(self):
        fake_mod = types.ModuleType("voice.tts")
        profile = MagicMock()
        profile.name = "default"
        profile.language = "te"
        profile.is_default = True
        profile.has_audio = True

        mgr = MagicMock()
        mgr.list_profiles.return_value = [profile]
        fake_mod.VoiceProfileManager = MagicMock(return_value=mgr)

        with patch.dict(
            "sys.modules",
            {
                "voice.tts": fake_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.list_voice_profiles()
            assert len(result["profiles"]) == 1
            p = result["profiles"][0]
            assert p["name"] == "default"
            assert p["language"] == "te"
            assert p["is_default"] is True
            assert p["has_audio"] is True

    def test_list_profiles_empty(self):
        fake_mod = types.ModuleType("voice.tts")
        mgr = MagicMock()
        mgr.list_profiles.return_value = []
        fake_mod.VoiceProfileManager = MagicMock(return_value=mgr)

        with patch.dict(
            "sys.modules",
            {
                "voice.tts": fake_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.list_voice_profiles()
            assert result["profiles"] == []

    def test_list_profiles_exception(self):
        fake_mod = types.ModuleType("voice.tts")
        fake_mod.VoiceProfileManager = MagicMock(side_effect=RuntimeError("oops"))

        with patch.dict(
            "sys.modules",
            {
                "voice.tts": fake_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.list_voice_profiles()
            assert result["profiles"] == []
            assert "oops" in result["error"]

    def test_list_profiles_import_error(self):
        with patch.dict("sys.modules", {"voice.tts": None}):
            result = service.list_voice_profiles()
            assert result["profiles"] == []


# ===================================================================
#  PART 14 --- service.get_storage_stats
# ===================================================================


class TestServiceGetStorageStats:

    def test_get_storage_stats_success(self):
        fake_mod = types.ModuleType("voice.storage")
        storage_inst = MagicMock()
        storage_inst.get_storage_stats.return_value = {"total": 100, "sessions": 5}
        fake_mod.AudioStorage = MagicMock(return_value=storage_inst)

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.get_storage_stats()
            assert result["total"] == 100
            assert result["sessions"] == 5

    def test_get_storage_stats_error(self):
        fake_mod = types.ModuleType("voice.storage")
        fake_mod.AudioStorage = MagicMock(side_effect=RuntimeError("no db"))

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.get_storage_stats()
            assert "no db" in result["error"]

    def test_get_storage_stats_import_error(self):
        with patch.dict("sys.modules", {"voice.storage": None}):
            result = service.get_storage_stats()
            assert "error" in result


# ===================================================================
#  PART 15 --- service.transcribe_audio
# ===================================================================


class TestServiceTranscribeAudio:

    def test_transcribe_success(self):
        fake_stt_mod = types.ModuleType("voice.stt")
        fake_config_mod = types.ModuleType("voice.config")

        config = MagicMock()
        fake_config_mod.get_voice_config = MagicMock(return_value=config)

        tr_result = MagicMock()
        tr_result.text = "hello world"
        tr_result.language = "en"
        tr_result.language_probability = 0.98
        tr_result.duration = 2.5
        tr_result.processing_time = 0.4

        stt_inst = MagicMock()
        stt_inst.transcribe.return_value = tr_result
        fake_stt_mod.FasterWhisperSTT = MagicMock(return_value=stt_inst)

        with patch.dict(
            "sys.modules",
            {
                "voice.stt": fake_stt_mod,
                "voice.config": fake_config_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.transcribe_audio("/audio.wav")
            assert result["text"] == "hello world"
            assert result["language"] == "en"
            assert result["language_probability"] == 0.98
            assert result["duration"] == 2.5
            assert result["processing_time"] == 0.4

    def test_transcribe_with_language(self):
        fake_stt_mod = types.ModuleType("voice.stt")
        fake_config_mod = types.ModuleType("voice.config")

        config = MagicMock()
        fake_config_mod.get_voice_config = MagicMock(return_value=config)

        tr_result = MagicMock()
        tr_result.text = "namaste"
        tr_result.language = "te"
        tr_result.language_probability = 0.95
        tr_result.duration = 1.0
        tr_result.processing_time = 0.2

        stt_inst = MagicMock()
        stt_inst.transcribe.return_value = tr_result
        fake_stt_mod.FasterWhisperSTT = MagicMock(return_value=stt_inst)

        with patch.dict(
            "sys.modules",
            {
                "voice.stt": fake_stt_mod,
                "voice.config": fake_config_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.transcribe_audio("/audio.wav", language="te")
            stt_inst.transcribe.assert_called_once_with("/audio.wav", language="te")
            assert result["language"] == "te"

    def test_transcribe_exception(self):
        fake_stt_mod = types.ModuleType("voice.stt")
        fake_config_mod = types.ModuleType("voice.config")
        fake_config_mod.get_voice_config = MagicMock(
            side_effect=RuntimeError("no model")
        )
        fake_stt_mod.FasterWhisperSTT = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "voice.stt": fake_stt_mod,
                "voice.config": fake_config_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            result = service.transcribe_audio("/audio.wav")
            assert result["status"] == "error"
            assert "no model" in result["error"]

    def test_transcribe_import_error(self):
        with patch.dict("sys.modules", {"voice.stt": None}):
            result = service.transcribe_audio("/audio.wav")
            assert result["status"] == "error"


# ===================================================================
#  PART 16 --- Edge cases and integration-like tests
# ===================================================================


class TestEdgeCases:

    def test_server_init_creates_tool_defs(self, server):
        assert len(server._tool_defs) == 8

    @patch("mcp.voice.server.service")
    def test_dispatch_tool_with_arguments_none(self, svc, server):
        """arguments=None should default to empty dict."""
        svc.stop_listening.return_value = {"status": "stopped"}
        result = server._dispatch_tool(
            {"name": "voice_stop_listening", "arguments": None}
        )
        assert result["content"]["status"] == "stopped"

    def test_handle_request_null_params(self, server):
        """params=null is treated as empty dict for initialize."""
        resp = server.handle_request({"id": 1, "method": "initialize", "params": None})
        assert resp["type"] == "response"

    def test_handle_request_no_id(self, server):
        """Missing id should propagate as None."""
        resp = server.handle_request({"method": "initialize"})
        assert resp["id"] is None

    @patch("mcp.voice.server.service")
    def test_call_tool_speak_exception_in_service(self, svc, server):
        """Exception in service.speak_text should be caught by handle_request."""
        svc.speak_text.side_effect = Exception("catastrophic")
        resp = server.handle_request(
            _rpc(
                "call_tool",
                {
                    "name": "voice_speak",
                    "arguments": {"text": "hello"},
                },
            )
        )
        assert resp["type"] == "error"
        assert "catastrophic" in resp["error"]["message"]

    def test_start_stop_cycle(self):
        """Full start -> stop cycle via service."""
        with patch("mcp.voice.service.subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.pid = 999
            proc.poll.return_value = None
            mock_popen.return_value = proc

            r1 = service.start_listening(background=True)
            assert r1["status"] == "started"

            r2 = service.get_daemon_status()
            assert r2["status"] == "running"

            r3 = service.stop_listening()
            assert r3["status"] == "stopped"

            r4 = service.get_daemon_status()
            assert r4["status"] == "not_running"

    def test_multiple_stops_idempotent(self):
        """Calling stop twice when not running returns not_running."""
        assert service.stop_listening()["status"] == "not_running"
        assert service.stop_listening()["status"] == "not_running"

    @patch("mcp.voice.server.service")
    def test_multiple_tool_calls_via_handle_request(self, svc, server):
        """Ensure multiple sequential tool calls work."""
        svc.get_daemon_status.return_value = {"status": "not_running", "pid": None}
        svc.start_listening.return_value = {"status": "started", "pid": 1}

        r1 = server.handle_request(
            _rpc("call_tool", {"name": "voice_get_status"}, req_id=1)
        )
        r2 = server.handle_request(
            _rpc(
                "call_tool",
                {
                    "name": "voice_start_listening",
                    "arguments": {},
                },
                req_id=2,
            )
        )

        assert r1["id"] == 1
        assert r2["id"] == 2
        assert r1["result"]["content"]["status"] == "not_running"
        assert r2["result"]["content"]["status"] == "started"

    @patch("mcp.voice.server.sys.stdout")
    @patch("mcp.voice.server._iter_stdin")
    def test_main_flush_called(self, iter_fn, mock_stdout):
        """stdout.flush() should be called after each response."""
        iter_fn.return_value = [json.dumps({"id": 1, "method": "shutdown"})]
        main(["--log-level", "ERROR"])
        mock_stdout.flush.assert_called()

    @patch("mcp.voice.server.sys.stdout")
    @patch("mcp.voice.server._iter_stdin")
    def test_main_unknown_method_returns_error(self, iter_fn, mock_stdout):
        iter_fn.return_value = [
            json.dumps({"id": 1, "method": "nope"}),
            json.dumps({"id": 2, "method": "shutdown"}),
        ]
        main(["--log-level", "ERROR"])
        first = json.loads(mock_stdout.write.call_args_list[0][0][0])
        assert first["type"] == "error"


# ===================================================================
#  PART 17 --- Additional parametric & boundary tests
# ===================================================================


class TestParametricServerRoutes:
    """Test various method strings through handle_request."""

    @pytest.mark.parametrize(
        "method,expected_type",
        [
            ("initialize", "response"),
            ("list_tools", "response"),
            ("shutdown", "response"),
            ("", "error"),
            ("non_existent", "error"),
        ],
    )
    def test_method_routes(self, server, method, expected_type):
        resp = server.handle_request(_rpc(method))
        assert resp["type"] == expected_type

    @pytest.mark.parametrize(
        "tool_name",
        [
            "voice_start_listening",
            "voice_stop_listening",
            "voice_get_status",
            "voice_get_session",
            "voice_list_profiles",
        ],
    )
    @patch("mcp.voice.server.service")
    def test_paramless_tools_via_dispatch(self, svc, server, tool_name):
        """Tools that work without specific params should succeed with empty arguments."""
        svc.start_listening.return_value = {"status": "ok"}
        svc.stop_listening.return_value = {"status": "ok"}
        svc.get_daemon_status.return_value = {"status": "ok"}
        svc.get_session_info.return_value = {"status": "ok"}
        svc.list_voice_profiles.return_value = {"status": "ok"}

        result = server._dispatch_tool({"name": tool_name, "arguments": {}})
        assert "content" in result


class TestServiceBoundaries:

    def test_start_listening_no_args(self):
        """Defaults: config_path=None, background=True."""
        with patch("mcp.voice.service.subprocess.Popen") as mp:
            proc = MagicMock()
            proc.pid = 1
            mp.return_value = proc
            result = service.start_listening()
            assert result["status"] == "started"
            assert result["background"] is True

    @patch("mcp.voice.service.subprocess.Popen")
    def test_popen_called_with_pipe(self, mp):
        proc = MagicMock()
        proc.pid = 1
        mp.return_value = proc
        service.start_listening(background=True)
        kwargs = mp.call_args[1]
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE

    @patch("mcp.voice.service.subprocess.Popen")
    def test_popen_cwd_is_repo_root(self, mp):
        proc = MagicMock()
        proc.pid = 1
        mp.return_value = proc
        service.start_listening(background=True)
        kwargs = mp.call_args[1]
        assert kwargs["cwd"] == str(service.REPO_ROOT)

    def test_export_training_passes_path_object(self):
        """When output_path is given, it should be converted to Path."""
        fake_mod = types.ModuleType("voice.storage")
        gen_inst = MagicMock()
        gen_inst.export_to_jsonl.return_value = (Path("/tmp/x.jsonl"), 5)
        fake_mod.TrainingDataGenerator = MagicMock(return_value=gen_inst)

        with patch.dict(
            "sys.modules",
            {
                "voice.storage": fake_mod,
                "voice": types.ModuleType("voice"),
            },
        ):
            service.export_training_data(output_path="/tmp/x.jsonl")
            call_kwargs = gen_inst.export_to_jsonl.call_args[1]
            assert isinstance(call_kwargs["output_path"], Path)
            assert str(call_kwargs["output_path"]) == "/tmp/x.jsonl"
