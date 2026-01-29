"""
GLM-4.7-Flash Agentic Router
============================

Intelligent request router that analyzes user messages and determines:
- Task complexity (simple/moderate/complex)
- Required context (writers_room, kitchen, etc.)
- Suggested tools for the task
- Whether to use local LLaMA or cloud model

GLM-4.7-Flash is used as a fast, agentic router because:
- 3B active parameters (MoE) - extremely fast
- 87.4% on tau-Bench (multi-step tool use)
- "Preserved Thinking" - maintains reasoning across turns
- Runs at 120-220 tok/s on RTX 4090

Architecture:
    User Message → GLM Router → Routing Decision → LLaMA Persona Model
                                     ↓
                              [context, tools, complexity]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from orchestrator.config import RouterConfig, get_config

LOGGER = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Task complexity levels"""

    SIMPLE = "simple"  # Direct response, no tools needed
    MODERATE = "moderate"  # Single tool call or simple reasoning
    COMPLEX = "complex"  # Multi-step, multiple tools, chain of thought


class TaskType(str, Enum):
    """High-level task categories"""

    CONVERSATION = "conversation"  # Casual chat, greetings
    SCENE_QUERY = "scene_query"  # Scene search/retrieval
    SCENE_MANAGEMENT = "scene_management"  # Scene update/reorder/link
    EMAIL = "email"  # Email composition
    CREATIVE = "creative"  # Script writing, brainstorming
    TECHNICAL = "technical"  # Code, debugging, analysis
    INFORMATION = "information"  # Factual questions
    MEMORY = "memory"  # Recall past conversations


@dataclass
class RouterDecision:
    """Output from the router analysis"""

    task_type: TaskType
    complexity: TaskComplexity
    primary_context: str  # writers_room, kitchen, general, etc.
    suggested_tools: List[str] = field(default_factory=list)
    tool_order: List[str] = field(default_factory=list)  # Suggested execution order
    requires_tools: bool = False
    route_to_cloud: bool = False  # Use cloud model for complex tasks
    confidence: float = 0.8
    reasoning: str = ""  # Brief explanation

    # For multi-turn agentic flows
    expected_turns: int = 1
    agent_mode: bool = False  # Enable iterative tool execution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "complexity": self.complexity.value,
            "primary_context": self.primary_context,
            "suggested_tools": self.suggested_tools,
            "tool_order": self.tool_order,
            "requires_tools": self.requires_tools,
            "route_to_cloud": self.route_to_cloud,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "expected_turns": self.expected_turns,
            "agent_mode": self.agent_mode,
        }


# Router system prompt for GLM-4.7-Flash
ROUTER_SYSTEM_PROMPT = """You are a request router for Friday AI, an assistant for a Telugu screenwriter.

Your job is to analyze the user's message and determine:
1. Task type: conversation, scene_query, scene_management, email, creative, technical, information, memory
2. Complexity: simple (direct answer), moderate (one tool), complex (multi-step)
3. Context: writers_room (screenplay), kitchen (cooking), general (other)
4. Required tools (if any): scene_search, scene_get, scene_update, scene_reorder, scene_link, send_email, send_screenplay

Respond in JSON format only:
{
    "task_type": "scene_query",
    "complexity": "moderate",
    "context": "writers_room",
    "tools": ["scene_search"],
    "tool_order": ["scene_search"],
    "requires_tools": true,
    "confidence": 0.9,
    "reasoning": "User wants to find scenes with specific character"
}

Available tools:
- scene_search: Search scenes by content, character, or emotion
- scene_get: Get specific scene by code or ID
- scene_update: Modify scene status, text, or metadata
- scene_reorder: Move scene to new position
- scene_link: Create relationship between scenes
- send_email: Send email with content
- send_screenplay: Email screenplay PDF

Context clues:
- Telugu/English film terms, scenes, dialogue → writers_room
- Cooking, recipes, food → kitchen
- General questions, casual chat → general"""


class GLMRouter:
    """
    Intelligent request router using GLM-4.7-Flash.

    Analyzes user messages to determine optimal routing:
    - Which context/room to use
    - Which tools are likely needed
    - Whether to use local or cloud model
    - Expected complexity and turns

    Usage:
        router = GLMRouter()  # Uses config from get_config()
        decision = await router.analyze("Show me scene 5 from Gusagusalu")

        # Use decision to inform orchestrator
        if decision.requires_tools:
            tool_filter = decision.suggested_tools
        if decision.route_to_cloud:
            use_cloud_model = True
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        if config is None:
            config = get_config().router
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: Dict[str, RouterDecision] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client

    async def analyze(
        self,
        message: str,
        conversation_context: Optional[str] = None,
        current_context: Optional[str] = None,
    ) -> RouterDecision:
        """
        Analyze a user message and return routing decision.

        Args:
            message: User's input message
            conversation_context: Recent conversation summary (optional)
            current_context: Current context/room (for sticky behavior)

        Returns:
            RouterDecision with routing information
        """
        if not self.config.enabled:
            return self._default_routing(message, current_context)

        # Check cache for similar messages
        cache_key = self._cache_key(message)
        if self.config.cache_decisions and cache_key in self._cache:
            LOGGER.debug("Router cache hit for: %s", message[:50])
            return self._cache[cache_key]

        try:
            decision = await self._call_glm(
                message, conversation_context, current_context
            )

            # Cache the decision
            if self.config.cache_decisions:
                self._cache[cache_key] = decision
                # Simple cache eviction (keep last 100)
                if len(self._cache) > 100:
                    oldest = list(self._cache.keys())[0]
                    del self._cache[oldest]

            return decision

        except Exception as e:
            LOGGER.warning("Router analysis failed: %s", e)
            if self.config.fallback_on_error:
                return self._default_routing(message, current_context)
            raise

    async def _call_glm(
        self,
        message: str,
        conversation_context: Optional[str],
        current_context: Optional[str],
    ) -> RouterDecision:
        """Call GLM-4.7-Flash for routing decision"""
        client = await self._get_client()

        # Build user prompt
        user_prompt = f'Analyze this message: "{message}"'
        if current_context:
            user_prompt += f"\nCurrent context: {current_context}"
        if conversation_context:
            user_prompt += f"\nRecent conversation: {conversation_context[:500]}"

        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        headers = self._get_headers()
        url = self._get_url()

        LOGGER.debug("Calling GLM router: %s", url)
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        return self._parse_response(data)

    def _get_headers(self) -> Dict[str, str]:
        """Get API headers based on provider"""
        headers = {"Content-Type": "application/json"}

        if self.config.provider == "zhipu":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        elif self.config.provider == "openai":
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        return headers

    def _get_url(self) -> str:
        """Get API URL based on provider"""
        if self.config.provider == "zhipu":
            return f"{self.config.base_url}/chat/completions"
        elif self.config.provider == "openai":
            return "https://api.openai.com/v1/chat/completions"
        return f"{self.config.base_url}/chat/completions"

    def _parse_response(self, data: Dict) -> RouterDecision:
        """Parse GLM response into RouterDecision"""
        try:
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "{}")

            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            parsed = json.loads(content.strip())

            # Map to RouterDecision
            task_type = TaskType(parsed.get("task_type", "conversation"))
            complexity = TaskComplexity(parsed.get("complexity", "simple"))

            return RouterDecision(
                task_type=task_type,
                complexity=complexity,
                primary_context=parsed.get("context", "general"),
                suggested_tools=parsed.get("tools", []),
                tool_order=parsed.get("tool_order", parsed.get("tools", [])),
                requires_tools=parsed.get("requires_tools", False),
                route_to_cloud=complexity == TaskComplexity.COMPLEX,
                confidence=parsed.get("confidence", 0.8),
                reasoning=parsed.get("reasoning", ""),
                expected_turns=self._estimate_turns(
                    complexity, len(parsed.get("tools", []))
                ),
                agent_mode=len(parsed.get("tools", [])) > 1,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            LOGGER.warning("Failed to parse router response: %s", e)
            return self._default_routing("", None)

    def _estimate_turns(self, complexity: TaskComplexity, tool_count: int) -> int:
        """Estimate expected tool execution turns"""
        if complexity == TaskComplexity.SIMPLE:
            return 1
        elif complexity == TaskComplexity.MODERATE:
            return max(1, tool_count)
        else:  # COMPLEX
            return max(2, tool_count + 1)

    def _default_routing(
        self,
        message: str,
        current_context: Optional[str],
    ) -> RouterDecision:
        """Fallback keyword-based routing when GLM is unavailable"""
        message_lower = message.lower()

        # Detect tools needed
        tools = []
        if any(kw in message_lower for kw in ["find", "search", "show me", "look for"]):
            tools.append("scene_search")
        if any(kw in message_lower for kw in ["scene", "get scene"]):
            if "update" in message_lower or "change" in message_lower:
                tools.append("scene_update")
            elif "move" in message_lower or "reorder" in message_lower:
                tools.append("scene_reorder")
            elif "link" in message_lower or "connect" in message_lower:
                tools.append("scene_link")
            elif not tools:
                tools.append("scene_get")
        if any(kw in message_lower for kw in ["email", "send", "mail"]):
            tools.append("send_email")

        # Detect context
        context = current_context or "general"
        if any(
            kw in message_lower
            for kw in ["scene", "script", "dialogue", "story", "film"]
        ):
            context = "writers_room"
        elif any(kw in message_lower for kw in ["cook", "recipe", "food", "kitchen"]):
            context = "kitchen"

        # Detect task type
        if tools:
            if "scene" in " ".join(tools):
                task_type = (
                    TaskType.SCENE_QUERY
                    if "search" in str(tools)
                    else TaskType.SCENE_MANAGEMENT
                )
            elif "email" in str(tools):
                task_type = TaskType.EMAIL
            else:
                task_type = TaskType.INFORMATION
        else:
            task_type = TaskType.CONVERSATION

        # Determine complexity
        complexity = TaskComplexity.SIMPLE
        if len(tools) == 1:
            complexity = TaskComplexity.MODERATE
        elif len(tools) > 1:
            complexity = TaskComplexity.COMPLEX

        return RouterDecision(
            task_type=task_type,
            complexity=complexity,
            primary_context=context,
            suggested_tools=tools,
            tool_order=tools,
            requires_tools=len(tools) > 0,
            route_to_cloud=False,
            confidence=0.6,  # Lower confidence for keyword routing
            reasoning="Keyword-based routing (GLM unavailable)",
            expected_turns=max(1, len(tools)),
            agent_mode=len(tools) > 1,
        )

    def _cache_key(self, message: str) -> str:
        """Generate cache key from message"""
        # Normalize message for caching
        normalized = message.lower().strip()[:100]
        return normalized

    async def close(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def clear_cache(self) -> None:
        """Clear routing cache"""
        self._cache.clear()


# Convenience function for quick routing
async def quick_route(
    message: str,
    api_key: str,
    provider: str = "zhipu",
) -> RouterDecision:
    """
    Quick routing without creating a persistent router.

    Usage:
        decision = await quick_route("Show me scene 5", api_key="...")
    """
    config = RouterConfig(
        enabled=True,
        provider=provider,
        api_key=api_key,
    )
    router = GLMRouter(config)
    try:
        return await router.analyze(message)
    finally:
        await router.close()
