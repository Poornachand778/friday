"""
Voice MCP Service for Friday AI
================================

Business logic for voice operations exposed via MCP.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# Daemon process tracking
_daemon_process: Optional[subprocess.Popen] = None


def start_listening(
    config_path: Optional[str] = None,
    background: bool = True,
) -> Dict[str, Any]:
    """
    Start the voice daemon listening for wake words.

    Args:
        config_path: Path to voice config YAML
        background: Run in background process

    Returns:
        dict with status info
    """
    global _daemon_process

    if _daemon_process and _daemon_process.poll() is None:
        return {
            "status": "already_running",
            "pid": _daemon_process.pid,
        }

    try:
        cmd = [sys.executable, "-m", "voice.daemon"]
        if config_path:
            cmd.extend(["--config", config_path])

        if background:
            _daemon_process = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return {
                "status": "started",
                "pid": _daemon_process.pid,
                "background": True,
            }
        else:
            # Foreground mode - blocks
            result = subprocess.run(cmd, cwd=str(REPO_ROOT))
            return {
                "status": "completed",
                "return_code": result.returncode,
            }

    except Exception as e:
        LOGGER.error("Failed to start daemon: %s", e)
        return {
            "status": "error",
            "error": str(e),
        }


def stop_listening() -> Dict[str, Any]:
    """
    Stop the voice daemon.

    Returns:
        dict with status info
    """
    global _daemon_process

    if not _daemon_process:
        return {"status": "not_running"}

    try:
        _daemon_process.terminate()
        _daemon_process.wait(timeout=5)
        pid = _daemon_process.pid
        _daemon_process = None

        return {
            "status": "stopped",
            "pid": pid,
        }

    except subprocess.TimeoutExpired:
        _daemon_process.kill()
        pid = _daemon_process.pid
        _daemon_process = None

        return {
            "status": "killed",
            "pid": pid,
        }

    except Exception as e:
        LOGGER.error("Failed to stop daemon: %s", e)
        return {
            "status": "error",
            "error": str(e),
        }


def speak_text(
    text: str,
    language: str = "te",
    profile: Optional[str] = None,
    blocking: bool = True,
) -> Dict[str, Any]:
    """
    Synthesize and speak text using TTS.

    Args:
        text: Text to speak
        language: Language code (te, en)
        profile: Voice profile name
        blocking: Wait for playback to complete

    Returns:
        dict with synthesis info
    """
    try:
        from voice.tts import XTTSService
        from voice.audio.playback import AudioPlayback
        from voice.config import get_voice_config

        config = get_voice_config()
        tts = XTTSService(config.tts)
        playback = AudioPlayback(config.audio)

        # Synthesize
        result = tts.synthesize(
            text=text,
            language=language,
            profile=profile,
        )

        # Play audio
        playback.play(
            result.audio,
            sample_rate=result.sample_rate,
            blocking=blocking,
        )

        return {
            "status": "spoken",
            "text": text,
            "language": language,
            "duration": result.duration,
            "processing_time": result.processing_time,
        }

    except ImportError as e:
        LOGGER.error("TTS not available: %s", e)
        return {
            "status": "error",
            "error": f"TTS not available: {e}",
        }

    except Exception as e:
        LOGGER.error("Speech failed: %s", e)
        return {
            "status": "error",
            "error": str(e),
        }


def get_daemon_status() -> Dict[str, Any]:
    """
    Get voice daemon status.

    Returns:
        dict with daemon status info
    """
    global _daemon_process

    if not _daemon_process:
        return {
            "status": "not_running",
            "pid": None,
        }

    poll = _daemon_process.poll()
    if poll is None:
        return {
            "status": "running",
            "pid": _daemon_process.pid,
        }
    else:
        return {
            "status": "exited",
            "pid": _daemon_process.pid,
            "return_code": poll,
        }


def get_session_info(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get voice session information.

    Args:
        session_id: Specific session ID (None = current/recent)

    Returns:
        dict with session info
    """
    try:
        from voice.storage import AudioStorage

        storage = AudioStorage()

        if session_id:
            turns = storage.get_session_turns(session_id)
            return {
                "session_id": session_id,
                "turn_count": len(turns),
                "turns": [
                    {
                        "turn_number": t.turn_number,
                        "transcript": t.transcript,
                        "response": (
                            t.response_text[:100] + "..."
                            if t.response_text and len(t.response_text) > 100
                            else t.response_text
                        ),
                        "language": t.detected_language,
                    }
                    for t in turns
                ],
            }
        else:
            # Get recent turns
            turns = storage.get_recent_turns(limit=10)
            return {
                "recent_turns": len(turns),
                "sessions": list(set(t.session_id for t in turns)),
            }

    except Exception as e:
        LOGGER.error("Failed to get session info: %s", e)
        return {
            "status": "error",
            "error": str(e),
        }


def export_training_data(
    output_path: Optional[str] = None,
    min_length: int = 5,
    languages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Export voice data for training.

    Args:
        output_path: Output file path
        min_length: Minimum transcript length
        languages: Filter by languages

    Returns:
        dict with export info
    """
    try:
        from voice.storage import TrainingDataGenerator

        generator = TrainingDataGenerator()
        path, count = generator.export_to_jsonl(
            output_path=Path(output_path) if output_path else None,
            min_transcript_length=min_length,
            languages=languages,
        )

        return {
            "status": "exported",
            "path": str(path),
            "examples": count,
        }

    except Exception as e:
        LOGGER.error("Export failed: %s", e)
        return {
            "status": "error",
            "error": str(e),
        }


def list_voice_profiles() -> Dict[str, Any]:
    """
    List available voice profiles for TTS.

    Returns:
        dict with profile info
    """
    try:
        from voice.tts import VoiceProfileManager

        manager = VoiceProfileManager()
        profiles = manager.list_profiles()

        return {
            "profiles": [
                {
                    "name": p.name,
                    "language": p.language,
                    "is_default": p.is_default,
                    "has_audio": p.has_audio,
                }
                for p in profiles
            ]
        }

    except Exception as e:
        LOGGER.error("Failed to list profiles: %s", e)
        return {
            "profiles": [],
            "error": str(e),
        }


def get_storage_stats() -> Dict[str, Any]:
    """Get voice storage statistics"""
    try:
        from voice.storage import AudioStorage

        storage = AudioStorage()
        return storage.get_storage_stats()

    except Exception as e:
        return {"error": str(e)}


def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe an audio file.

    Args:
        audio_path: Path to audio file
        language: Force language (None = auto-detect)

    Returns:
        dict with transcription result
    """
    try:
        from voice.stt import FasterWhisperSTT
        from voice.config import get_voice_config

        config = get_voice_config()
        stt = FasterWhisperSTT(config.stt)

        result = stt.transcribe(audio_path, language=language)

        return {
            "text": result.text,
            "language": result.language,
            "language_probability": result.language_probability,
            "duration": result.duration,
            "processing_time": result.processing_time,
        }

    except Exception as e:
        LOGGER.error("Transcription failed: %s", e)
        return {
            "status": "error",
            "error": str(e),
        }
