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
    DOCUMENT = "document"  # Document ingestion/search/retrieval
    BOOK_STUDY = "book_study"  # Studying books for knowledge extraction
    MENTORING = "mentoring"  # Applying book knowledge to creative work
    KNOWLEDGE = "knowledge"  # Searching extracted knowledge


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
1. Task type: conversation, scene_query, scene_management, email, creative, technical, information, memory, document, book_study, mentoring, knowledge
2. Complexity: simple (direct answer), moderate (one tool), complex (multi-step)
3. Context: writers_room (screenplay), kitchen (cooking), general (other)
4. Required tools (if any) from the list below

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

Scene tools:
- scene_search: Search scenes by content, character, or emotion
- scene_get: Get specific scene by code or ID
- scene_update: Modify scene status, text, or metadata
- scene_reorder: Move scene to new position
- scene_link: Create relationship between scenes

Document tools:
- document_ingest: Upload and process a PDF document
- document_search: Search across ingested documents with citations
- document_get_context: Get relevant document context for a query
- document_get_chapter: Get full text of a specific chapter
- document_list: List all ingested documents
- document_get: Get specific document details
- document_status: Check processing status of a document
- document_delete: Delete a document and its chunks

Book understanding tools:
- book_study: Study an ingested book to extract concepts, principles, techniques
- book_study_status: Check progress of a book study job
- book_study_jobs: List all study jobs (active and completed)
- book_list_studied: List all studied books

Mentor tools (apply book knowledge to creative work):
- mentor_load_books: Load studied books for a mentoring session
- mentor_analyze: Analyze a scene using book knowledge
- mentor_brainstorm: Brainstorm ideas using book principles
- mentor_check_rules: Check scene against screenwriting rules
- mentor_find_inspiration: Find relevant examples from studied books
- mentor_ask: Ask what books say about a topic
- mentor_compare: Compare what different books say about a topic

Book detail tools:
- book_get_understanding: Get full extracted knowledge from a studied book

Knowledge tools:
- knowledge_search: Search across all extracted book knowledge

Communication tools:
- send_email: Send email with content
- send_screenplay: Email screenplay PDF

Vision tools (placeholder - pending hardware):
- camera_analyze: Analyze current camera feed
- generate_image: Generate an image from a description

Context clues:
- Telugu/English film terms, scenes, dialogue, brainstorm → writers_room
- Cooking, recipes, food → kitchen
- Books, PDFs, documents, references, citations → writers_room or general
- Mentor, analyze scene, what does McKee say → writers_room
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
            choices = data.get("choices", [])
            if not choices:
                return self._default_routing("", None)
            choice = choices[0]
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
        task_type = TaskType.CONVERSATION

        # --- Mentor / book knowledge tools ---
        if any(
            kw in message_lower
            for kw in [
                "mentor",
                "analyze scene",
                "analyze this",
                "what does mckee",
                "what does the book",
                "according to",
                "check rules",
                "rule violation",
                "brainstorm",
                "inspiration",
                "compare books",
                "compare what",
                "violat",
            ]
        ):
            if "analyze" in message_lower:
                tools.append("mentor_analyze")
            elif "brainstorm" in message_lower:
                tools.append("mentor_brainstorm")
            elif "compare" in message_lower:
                tools.append("mentor_compare")
            elif "rule" in message_lower or "violat" in message_lower:
                tools.append("mentor_check_rules")
            elif "inspiration" in message_lower or "example" in message_lower:
                tools.append("mentor_find_inspiration")
            else:
                tools.append("mentor_ask")
            task_type = TaskType.MENTORING

        # --- Book study tools ---
        elif any(
            kw in message_lower
            for kw in [
                "study book",
                "study this",
                "study the",
                "book study",
                "extract knowledge",
            ]
        ):
            tools.append("book_study")
            task_type = TaskType.BOOK_STUDY
        elif any(
            kw in message_lower
            for kw in ["study status", "study progress", "how far", "still studying"]
        ):
            tools.append("book_study_status")
            task_type = TaskType.BOOK_STUDY
        elif any(
            kw in message_lower
            for kw in ["studied books", "list books", "which books", "books studied"]
        ) or (
            "books" in message_lower
            and any(w in message_lower for w in ["studied", "list", "which", "have i"])
        ):
            tools.append("book_list_studied")
            task_type = TaskType.BOOK_STUDY
        elif any(
            kw in message_lower
            for kw in [
                "what did we learn",
                "show understanding",
                "extracted from",
                "book knowledge",
            ]
        ):
            tools.append("book_get_understanding")
            task_type = TaskType.BOOK_STUDY
        elif any(
            kw in message_lower
            for kw in [
                "study jobs",
                "all jobs",
                "job list",
                "active jobs",
                "running jobs",
            ]
        ) or (
            "jobs" in message_lower
            and any(
                w in message_lower for w in ["study", "book", "list", "show", "all"]
            )
        ):
            tools.append("book_study_jobs")
            task_type = TaskType.BOOK_STUDY

        # --- Document tools ---
        elif (
            "ingest" in message_lower
            or "process pdf" in message_lower
            or (
                "upload" in message_lower
                and any(w in message_lower for w in ["pdf", "document", "book"])
            )
        ):
            tools.append("document_ingest")
            task_type = TaskType.DOCUMENT
        elif any(
            kw in message_lower
            for kw in ["document", "pdf", "book", "reference", "citation"]
        ):
            if any(kw in message_lower for kw in ["search", "find", "look"]):
                tools.append("document_search")
            elif any(
                kw in message_lower
                for kw in ["chapter", "read chapter", "get chapter", "show chapter"]
            ):
                tools.append("document_get_chapter")
            elif any(
                kw in message_lower for kw in ["status", "processing", "progress"]
            ):
                tools.append("document_status")
            elif any(kw in message_lower for kw in ["delete", "remove", "discard"]):
                tools.append("document_delete")
            elif any(
                kw in message_lower for kw in ["list", "show all", "all documents"]
            ):
                tools.append("document_list")
            elif any(
                kw in message_lower
                for kw in ["details", "info about", "get document", "show document"]
            ):
                tools.append("document_get")
            elif any(
                kw in message_lower
                for kw in ["context", "relevant context", "get context"]
            ):
                tools.append("document_get_context")
            else:
                tools.append("document_search")
            task_type = TaskType.DOCUMENT

        # --- Knowledge search ---
        elif any(
            kw in message_lower
            for kw in [
                "knowledge",
                "concept",
                "principle",
                "technique",
                "what did I learn",
                "what do the books say",
            ]
        ):
            tools.append("knowledge_search")
            task_type = TaskType.KNOWLEDGE

        # --- Scene tools ---
        elif any(kw in message_lower for kw in ["find", "search", "look for"]):
            if any(kw in message_lower for kw in ["scene", "script", "dialogue"]):
                tools.append("scene_search")
                task_type = TaskType.SCENE_QUERY
        if any(kw in message_lower for kw in ["scene", "get scene"]):
            if "update" in message_lower or "change" in message_lower:
                tools.append("scene_update")
                task_type = TaskType.SCENE_MANAGEMENT
            elif "move" in message_lower or "reorder" in message_lower:
                tools.append("scene_reorder")
                task_type = TaskType.SCENE_MANAGEMENT
            elif "link" in message_lower or "connect" in message_lower:
                tools.append("scene_link")
                task_type = TaskType.SCENE_MANAGEMENT
            elif not tools:
                tools.append("scene_get")
                task_type = TaskType.SCENE_QUERY

        # --- Vision tools (placeholders) ---
        if any(
            kw in message_lower
            for kw in ["camera", "analyze feed", "what's on camera", "check camera"]
        ):
            tools.append("camera_analyze")
        elif any(
            kw in message_lower
            for kw in [
                "generate image",
                "generate a",
                "draw",
                "sketch",
                "storyboard frame",
                "concept art",
                "visualize",
            ]
        ) and any(
            kw in message_lower
            for kw in [
                "image",
                "picture",
                "art",
                "frame",
                "draw",
                "sketch",
                "visualize",
            ]
        ):
            tools.append("generate_image")

        # --- Mentor load books ---
        if any(
            kw in message_lower
            for kw in [
                "load book",
                "load the book",
                "prepare mentor",
                "load for mentor",
            ]
        ):
            if "mentor_analyze" not in tools and "mentor_brainstorm" not in tools:
                tools.append("mentor_load_books")
                task_type = TaskType.MENTORING

        # --- Email tools ---
        if any(kw in message_lower for kw in ["email", "send", "mail"]):
            if "screenplay" in message_lower or "script" in message_lower:
                tools.append("send_screenplay")
            else:
                tools.append("send_email")
            task_type = TaskType.EMAIL

        # Detect context
        context = current_context or "general"
        if any(
            kw in message_lower
            for kw in [
                "scene",
                "script",
                "dialogue",
                "story",
                "film",
                "mentor",
                "brainstorm",
                "mckee",
                "screenplay",
            ]
        ):
            context = "writers_room"
        elif any(kw in message_lower for kw in ["cook", "recipe", "food", "kitchen"]):
            context = "kitchen"
        elif any(kw in message_lower for kw in ["document", "pdf", "ingest", "upload"]):
            context = "general"

        # Fallback task type if no tools detected
        if not tools and task_type == TaskType.CONVERSATION:
            if any(
                kw in message_lower
                for kw in ["write", "draft", "dialogue", "story", "brainstorm"]
            ):
                task_type = TaskType.CREATIVE

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
