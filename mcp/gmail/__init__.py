"""Gmail MCP package for Friday AI."""

from .service import (
    send_screenplay_email,
    send_simple_email,
    list_labels,
    get_gmail_service,
)

__all__ = [
    "send_screenplay_email",
    "send_simple_email",
    "list_labels",
    "get_gmail_service",
]
