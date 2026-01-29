"""Voice MCP package for Friday AI."""

from .service import (
    start_listening,
    stop_listening,
    speak_text,
    get_daemon_status,
    get_session_info,
    export_training_data,
    list_voice_profiles,
)

__all__ = [
    "start_listening",
    "stop_listening",
    "speak_text",
    "get_daemon_status",
    "get_session_info",
    "export_training_data",
    "list_voice_profiles",
]
