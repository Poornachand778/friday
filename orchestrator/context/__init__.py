"""
Context management for Friday AI Orchestrator
=============================================

Handles room/context detection and configuration.
"""

from .contexts import Context, ContextType, CONTEXTS
from .detector import ContextDetector

__all__ = [
    "Context",
    "ContextType",
    "CONTEXTS",
    "ContextDetector",
]
