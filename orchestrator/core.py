"""
Friday AI Orchestrator Core
===========================

The central brain that coordinates all Friday AI components:
- LLM inference (local or cloud)
- Tool execution
- Context management
- Memory (WorkingMemory with capacity zones + poisoning detection)
- Multi-room context switching

Status:
    DONE: Basic orchestrator structure
    DONE: GLM-4 router integration
    DONE: WorkingMemory integration (capacity zones, poisoning, attention)
    DONE: MCP tool routing (scene_manager, documents, book/mentor)
    TODO: Voice pipeline integration (Whisper STT → Friday → XTTS TTS)
    TODO: Camera wake trigger support
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from orchestrator.config import OrchestratorConfig, get_config
from orchestrator.context.contexts import ContextType, CONTEXTS
from orchestrator.context.detector import ContextDetector
from orchestrator.inference.local_llm import LLMClient, ChatResponse
from orchestrator.inference.router import GLMRouter, RouterDecision
from orchestrator.memory.working_memory_adapter import WorkingMemoryAdapter
from orchestrator.memory.context_builder import (
    ContextBuilder,
    get_default_system_prompt,
)
from orchestrator.tools.registry import ToolRegistry, get_tool_registry, ToolResult

from memory.config import get_memory_config

LOGGER = logging.getLogger(__name__)


@dataclass
class OrchestratorResponse:
    """Response from the orchestrator"""

    content: str
    context_type: ContextType
    tool_calls_made: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    turn_id: int
    processing_time_ms: float
    tokens_used: Dict[str, int]


class FridayOrchestrator:
    """
    Central orchestrator for Friday AI.

    Coordinates:
    - LLM inference (vLLM, llama.cpp, or cloud fallback)
    - Tool execution via MCP services
    - Context detection and switching
    - Conversation memory management
    - LoRA adapter selection per context

    Usage:
        orchestrator = FridayOrchestrator()
        await orchestrator.initialize()

        response = await orchestrator.chat(
            message="Boss, show me scene 5 of Gusagusalu",
            location="writers_room",
        )

        print(response.content)
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or get_config()

        # Core components
        self._llm_client: Optional[LLMClient] = None
        self._router: Optional[GLMRouter] = None  # GLM-4.7-Flash router
        self._tool_registry: Optional[ToolRegistry] = None
        self._context_detector: Optional[ContextDetector] = None
        self._context_builder: Optional[ContextBuilder] = None

        # Memory system config
        self._memory_config = get_memory_config()

        # Session state (now using WorkingMemoryAdapter for advanced memory)
        self._sessions: Dict[str, WorkingMemoryAdapter] = {}
        self._current_session_id: Optional[str] = None
        self._current_context: ContextType = ContextType.GENERAL

        # Last routing decision (for debugging/inspection)
        self._last_routing_decision: Optional[RouterDecision] = None

        # Initialization state
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def current_context(self) -> ContextType:
        return self._current_context

    @property
    def current_session(self) -> Optional[WorkingMemoryAdapter]:
        if self._current_session_id:
            return self._sessions.get(self._current_session_id)
        return None

    async def initialize(self) -> None:
        """Initialize all orchestrator components"""
        if self._initialized:
            return

        LOGGER.info("Initializing Friday Orchestrator...")

        # Initialize LLM client (persona model - LLaMA 3.1 8B)
        self._llm_client = LLMClient(self.config.llm)

        # Initialize GLM-4.7-Flash router (if enabled)
        if self.config.router.enabled:
            self._router = GLMRouter(self.config.router)
            LOGGER.info("GLM-4.7-Flash router enabled")
        else:
            LOGGER.info("Router disabled, using keyword-based routing")

        # Initialize tool registry
        self._tool_registry = get_tool_registry()

        # Initialize context detector
        self._context_detector = ContextDetector()

        # Initialize context builder
        system_prompt = self.config.system_prompt_base or get_default_system_prompt()
        self._context_builder = ContextBuilder(
            base_system_prompt=system_prompt,
            max_context_tokens=getattr(self.config.memory, "max_context_tokens", 6000),
        )

        # Create default session
        self._create_session()

        self._initialized = True
        LOGGER.info("Friday Orchestrator initialized")

    async def shutdown(self) -> None:
        """Shutdown orchestrator and cleanup resources"""
        LOGGER.info("Shutting down Friday Orchestrator...")

        if self._llm_client:
            await self._llm_client.close()

        if self._router:
            await self._router.close()

        self._initialized = False
        LOGGER.info("Friday Orchestrator shutdown complete")

    def _create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new conversation session with WorkingMemory."""
        session_id = session_id or str(uuid.uuid4())[:8]

        # Use WorkingMemoryConfig from memory system, with orchestrator overrides
        wm_config = self._memory_config.working
        wm_config.max_turns = self.config.memory.max_history_turns
        wm_config.max_tokens = getattr(self.config.memory, "max_context_tokens", 6000)

        adapter = WorkingMemoryAdapter(config=wm_config)
        adapter.session_id = session_id

        self._sessions[session_id] = adapter
        self._current_session_id = session_id

        LOGGER.info(
            "Created session: %s (WorkingMemory: max_turns=%d, max_tokens=%d)",
            session_id,
            wm_config.max_turns,
            wm_config.max_tokens,
        )
        return session_id

    def switch_session(self, session_id: str) -> bool:
        """Switch to an existing session"""
        if session_id in self._sessions:
            self._current_session_id = session_id
            LOGGER.info("Switched to session: %s", session_id)
            return True
        return False

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        location: Optional[str] = None,
        stream: bool = False,
    ) -> OrchestratorResponse | AsyncIterator[str]:
        """
        Process a chat message through the full pipeline.

        Pipeline:
        1. GLM-4.7-Flash Router (if enabled) → Routing decision
        2. Context detection (informed by router)
        3. Context building (with filtered tools from router)
        4. LLaMA 3.1 8B → Response generation
        5. Tool execution (if needed)
        6. Memory storage

        Args:
            message: User's message
            session_id: Session to use (creates new if not exists)
            location: Physical location hint for context detection
            stream: Whether to stream the response

        Returns:
            OrchestratorResponse with content and metadata
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Get or create session
        if session_id and session_id not in self._sessions:
            self._create_session(session_id)
        elif session_id:
            self.switch_session(session_id)

        memory = self.current_session
        if not memory:
            self._create_session()
            memory = self.current_session

        # Step 1: Router analysis (if enabled)
        routing_decision: Optional[RouterDecision] = None
        tool_filter: Optional[List[str]] = None

        if self._router and self.config.router.enabled:
            try:
                # Get conversation summary for router context
                conversation_context = None
                if memory and memory.turn_count > 0:
                    recent_turns = memory.get_last_n_turns(n=3)
                    if recent_turns:
                        # Build simple summary from recent turns
                        summary_parts = []
                        for turn in recent_turns:
                            summary_parts.append(f"User: {turn.user_message[:100]}")
                            summary_parts.append(
                                f"Friday: {turn.assistant_response[:100]}"
                            )
                        conversation_context = "\n".join(summary_parts)

                routing_decision = await self._router.analyze(
                    message=message,
                    conversation_context=conversation_context,
                    current_context=self._current_context.value,
                )
                self._last_routing_decision = routing_decision

                LOGGER.info(
                    "Router decision: %s/%s, tools=%s, confidence=%.2f",
                    routing_decision.task_type.value,
                    routing_decision.complexity.value,
                    routing_decision.suggested_tools,
                    routing_decision.confidence,
                )

                # Use router's suggested tools as filter
                if routing_decision.suggested_tools:
                    tool_filter = routing_decision.suggested_tools

            except Exception as e:
                LOGGER.warning("Router failed, falling back to default: %s", e)
                routing_decision = None

        # Step 2: Context detection (informed by router if available)
        if routing_decision and routing_decision.confidence >= 0.8:
            # Trust router's context suggestion
            context_map = {
                "writers_room": ContextType.WRITERS_ROOM,
                "kitchen": ContextType.KITCHEN,
                "storyboard": ContextType.STORYBOARD,
                "general": ContextType.GENERAL,
            }
            new_context = context_map.get(
                routing_decision.primary_context,
                self._current_context,
            )
        else:
            # Fall back to keyword-based detection
            new_context = self._context_detector.detect(
                message=message,
                location=location,
                current_context=self._current_context,
            )

        if new_context != self._current_context:
            LOGGER.info(
                "Context switched: %s -> %s",
                self._current_context.value,
                new_context.value,
            )
            self._current_context = new_context
            memory.set_context(new_context.value)

        # Step 3: Build context for LLM (with tool filtering from router)
        context_config = CONTEXTS.get(
            self._current_context, CONTEXTS[ContextType.GENERAL]
        )
        built_context = self._context_builder.build(
            user_message=message,
            conversation_memory=memory,
            context_type=self._current_context,
            tool_filter=tool_filter,  # Pass router's suggested tools
        )

        LOGGER.debug(
            "Built context: %d messages, %d tokens, %d LTM, tools=%s",
            len(built_context.messages),
            built_context.token_estimate,
            built_context.ltm_count,
            (
                [t["function"]["name"] for t in built_context.tools]
                if built_context.tools
                else []
            ),
        )

        # Stream response if requested
        if stream:
            return self._stream_response(built_context, memory, message, start_time)

        # Step 4: Get LLM response (persona model)
        tool_calls_made = []
        tool_results = []

        response = await self._llm_client.chat(
            messages=built_context.messages,
            tools=built_context.tools,
            stream=False,
        )

        # Step 5: Handle tool calls if any
        if response.has_tool_calls:
            # Use router's expected_turns as max_iterations hint
            max_iter = 5
            if routing_decision and routing_decision.agent_mode:
                max_iter = max(routing_decision.expected_turns, 5)

            tool_calls_made, tool_results, response = await self._handle_tool_calls(
                response, built_context, max_iterations=max_iter
            )

        # Step 6: Store in memory
        turn = memory.add_turn(
            user_message=message,
            assistant_response=response.content,
            tool_calls=tool_calls_made,
            tool_results=tool_results,
            context_type=self._current_context.value,
        )

        processing_time = (time.time() - start_time) * 1000

        return OrchestratorResponse(
            content=response.content,
            context_type=self._current_context,
            tool_calls_made=tool_calls_made,
            tool_results=tool_results,
            turn_id=turn.turn_id,
            processing_time_ms=processing_time,
            tokens_used=response.usage,
        )

    async def _stream_response(
        self,
        built_context,
        memory: WorkingMemoryAdapter,
        original_message: str,
        start_time: float,
    ) -> AsyncIterator[str]:
        """Stream response tokens"""
        full_response = ""

        async for token in await self._llm_client.chat(
            messages=built_context.messages,
            tools=built_context.tools,
            stream=True,
        ):
            full_response += token
            yield token

        # Store in memory after streaming completes
        memory.add_turn(
            user_message=original_message,
            assistant_response=full_response,
            context_type=self._current_context.value,
        )

    async def _handle_tool_calls(
        self,
        response: ChatResponse,
        built_context,
        max_iterations: int = 5,
    ) -> tuple[List[Dict], List[Dict], ChatResponse]:
        """
        Handle tool calls from LLM response.

        Executes tools and continues conversation until no more tool calls
        or max iterations reached.
        """
        all_tool_calls = []
        all_tool_results = []
        current_response = response
        messages = list(built_context.messages)
        iteration = 0

        while current_response.has_tool_calls and iteration < max_iterations:
            iteration += 1
            LOGGER.debug("Tool call iteration %d", iteration)

            # Execute each tool call
            tool_calls_this_round = []
            tool_results_this_round = []

            for tc in current_response.tool_calls:
                LOGGER.info("Executing tool: %s", tc.name)

                result = await self._tool_registry.async_execute(tc.name, tc.arguments)

                tool_call_dict = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                tool_calls_this_round.append(tool_call_dict)

                tool_result_dict = {
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "success": result.success,
                    "data": result.data,
                    "error": result.error,
                }
                tool_results_this_round.append(tool_result_dict)

            all_tool_calls.extend(tool_calls_this_round)
            all_tool_results.extend(tool_results_this_round)

            # Build follow-up context
            messages = self._context_builder.build_for_tool_response(
                built_context, tool_calls_this_round, tool_results_this_round
            )

            # Get next response
            current_response = await self._llm_client.chat(
                messages=messages,
                tools=built_context.tools,
                stream=False,
            )

        return all_tool_calls, all_tool_results, current_response

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """Execute a tool directly without LLM"""
        if not self._initialized:
            await self.initialize()

        return await self._tool_registry.async_execute(tool_name, arguments)

    def get_session_info(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a session, including WorkingMemory health."""
        sid = session_id or self._current_session_id
        if not sid or sid not in self._sessions:
            return {"error": "Session not found"}

        memory = self._sessions[sid]
        info = {
            "session_id": sid,
            "turn_count": memory.turn_count,
            "active_turns": memory.active_turns,
            "current_context": memory.current_context,
            "started_at": memory.started_at,
            "total_tokens": memory.total_tokens,
            # WorkingMemory-specific
            "capacity_zone": memory.capacity_zone,
            "capacity_percentage": f"{memory.capacity_percentage:.1%}",
            "tokens_available": memory.tokens_available,
        }
        return info

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [
            {
                "session_id": sid,
                "turn_count": mem.turn_count,
                "current_context": mem.current_context,
                "started_at": mem.started_at,
            }
            for sid, mem in self._sessions.items()
        ]

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        health = {
            "orchestrator": "healthy",
            "initialized": self._initialized,
            "current_context": self._current_context.value,
            "active_sessions": len(self._sessions),
        }

        # Check LLM (persona model)
        if self._llm_client:
            try:
                llm_healthy = await self._llm_client.health_check()
                health["llm"] = "healthy" if llm_healthy else "unhealthy"
            except Exception as e:
                health["llm"] = f"error: {e}"
        else:
            health["llm"] = "not_initialized"

        # Check router (GLM-4.7-Flash)
        if self.config.router.enabled:
            health["router"] = {
                "enabled": True,
                "provider": self.config.router.provider,
                "model": self.config.router.model_name,
                "cache_size": len(self._router._cache) if self._router else 0,
            }
            if self._last_routing_decision:
                health["router"]["last_decision"] = {
                    "task_type": self._last_routing_decision.task_type.value,
                    "complexity": self._last_routing_decision.complexity.value,
                    "confidence": self._last_routing_decision.confidence,
                }
        else:
            health["router"] = {"enabled": False}

        # Check tool registry
        if self._tool_registry:
            health["tools"] = len(self._tool_registry._tools)
        else:
            health["tools"] = 0

        # Check working memory health
        if self.current_session:
            health["memory"] = self.current_session.get_health_status()
        else:
            health["memory"] = {"status": "no_active_session"}

        return health


# Singleton orchestrator instance
_orchestrator: Optional[FridayOrchestrator] = None


def get_orchestrator() -> FridayOrchestrator:
    """Get the singleton orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = FridayOrchestrator()
    return _orchestrator


async def initialize_orchestrator() -> FridayOrchestrator:
    """Initialize and return the orchestrator"""
    orchestrator = get_orchestrator()
    await orchestrator.initialize()
    return orchestrator
