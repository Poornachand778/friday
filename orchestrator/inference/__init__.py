"""
Inference module for Friday AI Orchestrator
===========================================

Handles LLM inference via local (vLLM, llama.cpp) or remote (OpenAI) backends.
"""

from .local_llm import LLMClient, ChatMessage, ChatResponse

__all__ = [
    "LLMClient",
    "ChatMessage",
    "ChatResponse",
]
