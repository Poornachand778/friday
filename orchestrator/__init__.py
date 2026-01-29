"""
Friday AI Orchestrator
======================

Central coordination layer for Friday AI.
Connects voice, LLM, tools, and memory.
"""

__version__ = "0.1.0"

from orchestrator.core import (
    FridayOrchestrator,
    OrchestratorResponse,
    get_orchestrator,
    initialize_orchestrator,
)
from orchestrator.config import OrchestratorConfig, get_config

__all__ = [
    "FridayOrchestrator",
    "OrchestratorResponse",
    "get_orchestrator",
    "initialize_orchestrator",
    "OrchestratorConfig",
    "get_config",
]
