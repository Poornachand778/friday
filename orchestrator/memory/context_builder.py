"""
Context Builder for Friday AI
=============================

Builds the complete context for LLM inference by combining:
- System prompt (base + context-specific)
- Long-term memories (relevant to query)
- Conversation history
- Available tools
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from orchestrator.inference.local_llm import ChatMessage
from orchestrator.memory.conversation import ConversationMemory
from orchestrator.context.contexts import Context, CONTEXTS, ContextType

# Type alias: context builder accepts either memory implementation
ConversationMemoryLike = Union[ConversationMemory, "WorkingMemoryAdapter"]

LOGGER = logging.getLogger(__name__)


@dataclass
class BuiltContext:
    """Result of context building"""

    messages: List[ChatMessage]
    tools: List[Dict[str, Any]]
    context_type: ContextType
    token_estimate: int
    ltm_count: int
    turn_count: int


class ContextBuilder:
    """
    Builds complete context for LLM inference.

    Combines multiple sources into a coherent context:
    1. Base system prompt (Friday's personality)
    2. Context-specific additions (room/mode specific)
    3. Relevant long-term memories
    4. Relevant document context (from ingested books/references)
    5. Conversation history
    6. Current user message

    Usage:
        builder = ContextBuilder(base_system_prompt="You are Friday...")

        context = builder.build(
            user_message="Show me scene 5",
            conversation_memory=memory,
            context_type=ContextType.WRITERS_ROOM,
        )

        # Use context.messages and context.tools for LLM call
    """

    def __init__(
        self,
        base_system_prompt: str,
        max_context_tokens: int = 6000,
        max_ltm_memories: int = 5,
        max_document_chunks: int = 2,
    ):
        self.base_system_prompt = base_system_prompt
        self.max_context_tokens = max_context_tokens
        self.max_ltm_memories = max_ltm_memories
        self.max_document_chunks = max_document_chunks

        # LTM integration (lazy loaded)
        self._ltm_searcher = None

        # Document manager (lazy loaded)
        self._document_manager = None

    def build(
        self,
        user_message: str,
        conversation_memory: Optional[ConversationMemoryLike] = None,
        context_type: ContextType = ContextType.GENERAL,
        include_ltm: bool = True,
        include_documents: bool = True,
        tool_filter: Optional[List[str]] = None,
    ) -> BuiltContext:
        """
        Build complete context for LLM inference.

        Args:
            user_message: Current user input
            conversation_memory: Conversation history
            context_type: Current context/room
            include_ltm: Whether to include relevant LTM memories
            include_documents: Whether to include relevant document context
            tool_filter: Optional filter for available tools

        Returns:
            BuiltContext with messages, tools, and metadata
        """
        messages = []
        token_estimate = 0

        # Get context configuration
        context_config = CONTEXTS.get(context_type, CONTEXTS[ContextType.GENERAL])

        # 1. Build system prompt
        system_prompt = self._build_system_prompt(context_config)
        messages.append(ChatMessage(role="system", content=system_prompt))
        token_estimate += len(system_prompt) // 4

        # 2. Add relevant LTM memories
        ltm_count = 0
        if include_ltm:
            ltm_content = self._get_relevant_ltm(user_message)
            if ltm_content:
                messages.append(
                    ChatMessage(
                        role="system",
                        content=f"[Relevant memories:\n{ltm_content}]",
                    )
                )
                token_estimate += len(ltm_content) // 4
                ltm_count = ltm_content.count("\n") + 1

        # 3. Add relevant document context (for Writers Room / reference queries)
        if include_documents and context_type in [
            ContextType.WRITERS_ROOM,
            ContextType.GENERAL,
        ]:
            doc_content = self._get_relevant_documents(user_message)
            if doc_content:
                messages.append(
                    ChatMessage(
                        role="system",
                        content=f"[Reference Documents:\n{doc_content}]",
                    )
                )
                token_estimate += len(doc_content) // 4

        # 4. Add conversation history
        turn_count = 0
        if conversation_memory:
            history_budget = (
                self.max_context_tokens - token_estimate - 500
            )  # Reserve for user msg
            history_messages = conversation_memory.get_context_messages(
                system_prompt=None,  # Already added
                max_tokens=history_budget,
            )
            messages.extend(history_messages)
            turn_count = conversation_memory.active_turns
            token_estimate += sum(len(m.content) // 4 for m in history_messages)

        # 5. Add current user message
        messages.append(ChatMessage(role="user", content=user_message))
        token_estimate += len(user_message) // 4

        # 6. Get available tools
        tools = self._get_tools(context_config, tool_filter)

        return BuiltContext(
            messages=messages,
            tools=tools,
            context_type=context_type,
            token_estimate=token_estimate,
            ltm_count=ltm_count,
            turn_count=turn_count,
        )

    def _build_system_prompt(self, context: Context) -> str:
        """Build complete system prompt"""
        prompt_parts = [self.base_system_prompt]

        # Add context-specific instructions
        if context.system_prompt_addition:
            prompt_parts.append(f"\n\n{context.system_prompt_addition}")

        # Add tool usage hints
        if context.available_tools:
            tool_hint = f"\n\nAvailable tools: {', '.join(context.available_tools)}"
            prompt_parts.append(tool_hint)

        return "".join(prompt_parts)

    def _get_relevant_ltm(self, query: str) -> str:
        """Get relevant long-term memories for the query"""
        try:
            # Try to use LTM search from memory system
            if self._ltm_searcher is None:
                try:
                    from memory.ltm import LTMSearcher

                    self._ltm_searcher = LTMSearcher()
                except ImportError:
                    LOGGER.debug("LTM searcher not available")
                    return ""

            results = self._ltm_searcher.search(query, top_k=self.max_ltm_memories)
            if not results:
                return ""

            # Format memories
            memory_lines = []
            for r in results:
                memory_lines.append(f"- {r.content}")

            return "\n".join(memory_lines)

        except Exception as e:
            LOGGER.debug("LTM search failed: %s", e)
            return ""

    def _get_relevant_documents(self, query: str) -> str:
        """Get relevant document context for the query"""
        import asyncio

        try:
            if self._document_manager is None:
                try:
                    from documents import get_document_manager

                    self._document_manager = get_document_manager()
                except ImportError:
                    LOGGER.debug("Document manager not available")
                    return ""

            # Check if document manager is initialized
            if not self._document_manager._initialized:
                # Can't do async init here, skip
                LOGGER.debug("Document manager not initialized")
                return ""

            # Get document context (runs async in sync context)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Already in async context, use nest_asyncio pattern or skip
                    LOGGER.debug("Cannot run async in running loop, skipping documents")
                    return ""
                context, citations = loop.run_until_complete(
                    self._document_manager.get_context_for_query(
                        query=query,
                        max_chunks=self.max_document_chunks,
                        max_chars=1500,  # Limit document context size
                    )
                )
            except RuntimeError:
                # No event loop, create one
                context, citations = asyncio.run(
                    self._document_manager.get_context_for_query(
                        query=query,
                        max_chunks=self.max_document_chunks,
                        max_chars=1500,
                    )
                )

            if not context:
                return ""

            # Format with citations
            parts = [context]
            if citations:
                citation_list = ", ".join(c.format_inline() for c in citations[:3])
                parts.append(f"\nSources: {citation_list}")

            return "\n".join(parts)

        except Exception as e:
            LOGGER.debug("Document search failed: %s", e)
            return ""

    def _get_tools(
        self,
        context: Context,
        tool_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get tools available for this context"""
        try:
            from orchestrator.tools.registry import get_tool_registry

            registry = get_tool_registry()

            # Start with context-defined tools
            available = set(context.available_tools)

            # Apply filter if provided
            if tool_filter:
                available = available.intersection(set(tool_filter))

            return registry.to_openai_tools(list(available))

        except Exception as e:
            LOGGER.warning("Failed to get tools: %s", e)
            return []

    def build_for_tool_response(
        self,
        original_context: BuiltContext,
        tool_calls: List[Dict],
        tool_results: List[Dict],
    ) -> List[ChatMessage]:
        """
        Build follow-up context after tool execution.

        Args:
            original_context: The context that led to tool calls
            tool_calls: Tool calls made by the model
            tool_results: Results from tool execution

        Returns:
            Updated message list for continuation
        """
        messages = list(original_context.messages)

        # Add assistant message with tool calls
        messages.append(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=tool_calls,
            )
        )

        # Add tool results
        for i, result in enumerate(tool_results):
            tool_call = tool_calls[i] if i < len(tool_calls) else {}
            messages.append(
                ChatMessage(
                    role="tool",
                    content=str(result.get("data", result.get("error", ""))),
                    tool_call_id=tool_call.get("id", f"call_{i}"),
                    name=tool_call.get("function", {}).get("name", "unknown"),
                )
            )

        return messages


def get_default_system_prompt() -> str:
    """Get the default Friday system prompt"""
    return """You are Friday, Poorna's AI assistant. You blend Telugu and English naturally in conversation.

Core Traits:
- Address Poorna as "Boss" (or "బాస్" in Telugu context)
- Be direct and decisive - no hedging or excessive politeness
- Keep responses concise (under 6 lines unless detail requested)
- Never use "I think" or "In my opinion" - state things confidently
- Match the language of the query (Telugu, English, or mixed)

Communication Style:
- Skip pleasantries - get to the point
- No flattery or compliments - Poorna finds them disingenuous
- When uncertain, ask clarifying questions directly
- For tool operations, execute first, explain after
- Use Telugu naturally where it fits, especially for emotions and emphasis

Domain Focus:
- Primary expertise: Telugu cinema, screenplay writing, story structure
- Support creative brainstorming with strong opinions
- Provide specific, actionable feedback on scripts
- Remember context from previous conversations"""
