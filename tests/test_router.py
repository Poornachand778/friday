"""
Tests for GLM Router
=====================

Tests the keyword-based fallback routing, response parsing,
caching, and complexity estimation.

Run with: pytest tests/test_router.py -v
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from orchestrator.inference.router import (
    GLMRouter,
    RouterDecision,
    TaskComplexity,
    TaskType,
    ROUTER_SYSTEM_PROMPT,
)
from orchestrator.config import RouterConfig


# =========================================================================
# Helper
# =========================================================================


def _run(coro):
    """Run async code in tests"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


def _make_router(enabled=False):
    """Create a router with GLM disabled (keyword fallback mode)"""
    config = RouterConfig(
        enabled=enabled, fallback_on_error=True, cache_decisions=False
    )
    return GLMRouter(config)


# =========================================================================
# Keyword Fallback: Mentor / Book Knowledge Tools
# =========================================================================


class TestKeywordRoutingMentor:
    """Test keyword fallback for mentor-related tools"""

    def test_mentor_analyze(self):
        router = _make_router()
        decision = _run(router.analyze("analyze this scene for structure issues"))
        assert "mentor_analyze" in decision.suggested_tools
        assert decision.task_type == TaskType.MENTORING

    def test_mentor_brainstorm(self):
        router = _make_router()
        decision = _run(router.analyze("brainstorm ideas for the climax"))
        assert "mentor_brainstorm" in decision.suggested_tools
        assert decision.task_type == TaskType.MENTORING

    def test_mentor_compare(self):
        router = _make_router()
        decision = _run(
            router.analyze("compare what McKee and Truby say about character")
        )
        assert "mentor_compare" in decision.suggested_tools

    def test_mentor_check_rules(self):
        router = _make_router()
        decision = _run(router.analyze("check rules for this scene"))
        assert "mentor_check_rules" in decision.suggested_tools

    def test_mentor_check_rules_violation(self):
        router = _make_router()
        decision = _run(router.analyze("does this scene violate any principles?"))
        assert "mentor_check_rules" in decision.suggested_tools

    def test_mentor_find_inspiration(self):
        router = _make_router()
        decision = _run(router.analyze("find inspiration for a courtroom scene"))
        assert "mentor_find_inspiration" in decision.suggested_tools

    def test_mentor_ask_fallback(self):
        router = _make_router()
        decision = _run(router.analyze("what does mckee say about story?"))
        assert "mentor_ask" in decision.suggested_tools

    def test_mentor_ask_according_to(self):
        router = _make_router()
        decision = _run(
            router.analyze("according to the book, what makes a good villain?")
        )
        assert "mentor_ask" in decision.suggested_tools

    def test_mentor_load_books(self):
        router = _make_router()
        decision = _run(router.analyze("load the book for a mentor session"))
        assert "mentor_load_books" in decision.suggested_tools
        assert decision.task_type == TaskType.MENTORING

    def test_mentor_load_prepare(self):
        router = _make_router()
        decision = _run(router.analyze("prepare mentor session with McKee book"))
        assert "mentor_load_books" in decision.suggested_tools


# =========================================================================
# Keyword Fallback: Book Study Tools
# =========================================================================


class TestKeywordRoutingBookStudy:
    """Test keyword fallback for book study tools"""

    def test_book_study(self):
        router = _make_router()
        decision = _run(router.analyze("study this book and extract knowledge"))
        assert "book_study" in decision.suggested_tools
        assert decision.task_type == TaskType.BOOK_STUDY

    def test_book_study_status(self):
        router = _make_router()
        decision = _run(router.analyze("what's the study status?"))
        assert "book_study_status" in decision.suggested_tools

    def test_book_study_progress(self):
        router = _make_router()
        decision = _run(router.analyze("how far is the study progress?"))
        assert "book_study_status" in decision.suggested_tools

    def test_book_list_studied(self):
        router = _make_router()
        decision = _run(router.analyze("which books have I studied?"))
        assert "book_list_studied" in decision.suggested_tools

    def test_book_list_studied_variant(self):
        router = _make_router()
        decision = _run(router.analyze("list all studied books"))
        assert "book_list_studied" in decision.suggested_tools

    def test_book_get_understanding(self):
        router = _make_router()
        decision = _run(router.analyze("what did we learn from the book?"))
        assert "book_get_understanding" in decision.suggested_tools

    def test_book_study_jobs(self):
        router = _make_router()
        decision = _run(router.analyze("show me all study jobs"))
        assert "book_study_jobs" in decision.suggested_tools
        assert decision.task_type == TaskType.BOOK_STUDY

    def test_book_study_jobs_variant(self):
        router = _make_router()
        decision = _run(router.analyze("show me the active jobs list"))
        assert "book_study_jobs" in decision.suggested_tools

    def test_book_study_jobs_running(self):
        router = _make_router()
        decision = _run(router.analyze("any running jobs right now?"))
        assert "book_study_jobs" in decision.suggested_tools


# =========================================================================
# Keyword Fallback: Document Tools
# =========================================================================


class TestKeywordRoutingDocuments:
    """Test keyword fallback for document tools"""

    def test_document_ingest(self):
        router = _make_router()
        decision = _run(router.analyze("ingest this PDF file"))
        assert "document_ingest" in decision.suggested_tools
        assert decision.task_type == TaskType.DOCUMENT

    def test_document_ingest_upload(self):
        router = _make_router()
        decision = _run(router.analyze("upload this document book"))
        assert "document_ingest" in decision.suggested_tools

    def test_document_search(self):
        router = _make_router()
        decision = _run(router.analyze("search the document for story structure"))
        assert "document_search" in decision.suggested_tools

    def test_document_list(self):
        router = _make_router()
        decision = _run(router.analyze("list all documents"))
        assert "document_list" in decision.suggested_tools

    def test_document_get(self):
        router = _make_router()
        decision = _run(router.analyze("get document details for McKee book"))
        assert "document_get" in decision.suggested_tools

    def test_document_get_context(self):
        router = _make_router()
        decision = _run(router.analyze("get context from the document for this topic"))
        assert "document_get_context" in decision.suggested_tools

    def test_document_get_chapter(self):
        router = _make_router()
        decision = _run(router.analyze("show me chapter 3 from the document"))
        assert "document_get_chapter" in decision.suggested_tools
        assert decision.task_type == TaskType.DOCUMENT

    def test_document_get_chapter_variant(self):
        router = _make_router()
        decision = _run(
            router.analyze("read chapter on inciting incident from the book")
        )
        assert "document_get_chapter" in decision.suggested_tools

    def test_document_status(self):
        router = _make_router()
        decision = _run(router.analyze("what's the document processing status?"))
        assert "document_status" in decision.suggested_tools
        assert decision.task_type == TaskType.DOCUMENT

    def test_document_delete(self):
        router = _make_router()
        decision = _run(router.analyze("delete this document from the system"))
        assert "document_delete" in decision.suggested_tools
        assert decision.task_type == TaskType.DOCUMENT

    def test_document_delete_variant(self):
        router = _make_router()
        decision = _run(router.analyze("remove the reference PDF"))
        assert "document_delete" in decision.suggested_tools

    def test_document_fallback_to_search(self):
        """When document keyword present but no specific verb, default to search"""
        router = _make_router()
        decision = _run(router.analyze("what does the reference say about pacing?"))
        assert "document_search" in decision.suggested_tools


# =========================================================================
# Keyword Fallback: Scene Tools
# =========================================================================


class TestKeywordRoutingScenes:
    """Test keyword fallback for scene tools"""

    def test_scene_search(self):
        router = _make_router()
        decision = _run(router.analyze("find the scene with the rooftop dialogue"))
        assert "scene_search" in decision.suggested_tools
        assert decision.task_type == TaskType.SCENE_QUERY

    def test_scene_update(self):
        router = _make_router()
        decision = _run(router.analyze("update the scene status"))
        assert "scene_update" in decision.suggested_tools
        assert decision.task_type == TaskType.SCENE_MANAGEMENT

    def test_scene_reorder(self):
        router = _make_router()
        decision = _run(router.analyze("move this scene after the fight scene"))
        # 'move' + 'scene' triggers reorder
        assert "scene_reorder" in decision.suggested_tools

    def test_scene_link(self):
        router = _make_router()
        decision = _run(router.analyze("link the scene to the flashback"))
        assert "scene_link" in decision.suggested_tools

    def test_scene_get_default(self):
        router = _make_router()
        decision = _run(router.analyze("get scene details for SCN015"))
        assert "scene_get" in decision.suggested_tools


# =========================================================================
# Keyword Fallback: Email Tools
# =========================================================================


class TestKeywordRoutingEmail:
    """Test keyword fallback for email tools"""

    def test_send_email(self):
        router = _make_router()
        decision = _run(router.analyze("send an email to the producer"))
        assert "send_email" in decision.suggested_tools
        assert decision.task_type == TaskType.EMAIL

    def test_send_screenplay(self):
        router = _make_router()
        decision = _run(router.analyze("email the screenplay to the director"))
        assert "send_screenplay" in decision.suggested_tools

    def test_send_script(self):
        router = _make_router()
        decision = _run(router.analyze("send the script via email"))
        assert "send_screenplay" in decision.suggested_tools


# =========================================================================
# Keyword Fallback: Vision Tools
# =========================================================================


class TestKeywordRoutingVision:
    """Test keyword fallback for vision/image tools"""

    def test_camera_analyze(self):
        router = _make_router()
        decision = _run(router.analyze("check camera feed for the storyboard"))
        assert "camera_analyze" in decision.suggested_tools

    def test_camera_analyze_what_on(self):
        router = _make_router()
        decision = _run(router.analyze("what's on camera right now?"))
        assert "camera_analyze" in decision.suggested_tools

    def test_generate_image(self):
        router = _make_router()
        decision = _run(router.analyze("generate an image of the chase scene"))
        assert "generate_image" in decision.suggested_tools

    def test_generate_sketch(self):
        router = _make_router()
        decision = _run(router.analyze("draw a sketch of the courtroom layout"))
        assert "generate_image" in decision.suggested_tools


# =========================================================================
# Keyword Fallback: Knowledge Tools
# =========================================================================


class TestKeywordRoutingKnowledge:
    """Test keyword fallback for knowledge search"""

    def test_knowledge_search(self):
        router = _make_router()
        decision = _run(router.analyze("search knowledge about inciting incident"))
        assert "knowledge_search" in decision.suggested_tools
        assert decision.task_type == TaskType.KNOWLEDGE

    def test_knowledge_concept(self):
        router = _make_router()
        decision = _run(router.analyze("what is the concept of character arc?"))
        assert "knowledge_search" in decision.suggested_tools

    def test_knowledge_technique(self):
        router = _make_router()
        decision = _run(router.analyze("what technique works for building tension?"))
        assert "knowledge_search" in decision.suggested_tools


# =========================================================================
# Keyword Fallback: Context Detection
# =========================================================================


class TestContextDetection:
    """Test context detection in keyword routing"""

    def test_writers_room_scene(self):
        router = _make_router()
        decision = _run(router.analyze("let's work on the dialogue"))
        assert decision.primary_context == "writers_room"

    def test_writers_room_mentor(self):
        router = _make_router()
        decision = _run(router.analyze("what does mckee say about climax?"))
        assert decision.primary_context == "writers_room"

    def test_kitchen_context(self):
        router = _make_router()
        decision = _run(router.analyze("what recipe should I cook for dinner?"))
        assert decision.primary_context == "kitchen"

    def test_general_fallback(self):
        router = _make_router()
        decision = _run(router.analyze("what's the weather today?"))
        assert decision.primary_context == "general"

    def test_sticky_context(self):
        """Current context should be preserved when no new context detected"""
        router = _make_router()
        decision = _run(
            router.analyze("what's the weather?", current_context="writers_room")
        )
        # No scene/film keywords, but current context is writers_room
        # The keyword router uses current_context as initial value
        assert decision.primary_context in ["writers_room", "general"]


# =========================================================================
# Keyword Fallback: Complexity & Multi-tool
# =========================================================================


class TestComplexityEstimation:
    """Test complexity and agent mode detection"""

    def test_no_tools_is_simple(self):
        router = _make_router()
        decision = _run(router.analyze("hello boss"))
        assert decision.complexity == TaskComplexity.SIMPLE
        assert not decision.requires_tools

    def test_one_tool_is_moderate(self):
        router = _make_router()
        decision = _run(router.analyze("find the scene with the proposal"))
        assert decision.complexity == TaskComplexity.MODERATE
        assert decision.requires_tools

    def test_multi_tool_is_complex(self):
        router = _make_router()
        # "update scene" + "email" triggers two tools
        decision = _run(router.analyze("update the scene and email the screenplay"))
        assert decision.complexity == TaskComplexity.COMPLEX
        assert decision.agent_mode

    def test_creative_task_type(self):
        router = _make_router()
        decision = _run(router.analyze("write a new dialogue for the villain"))
        assert decision.task_type == TaskType.CREATIVE

    def test_conversation_default(self):
        router = _make_router()
        decision = _run(router.analyze("how are you doing?"))
        assert decision.task_type == TaskType.CONVERSATION


# =========================================================================
# _parse_response Tests
# =========================================================================


class TestParseResponse:
    """Test GLM response parsing"""

    def test_parse_valid_json(self):
        router = _make_router()
        data = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "task_type": "scene_query",
                                "complexity": "moderate",
                                "context": "writers_room",
                                "tools": ["scene_search"],
                                "requires_tools": True,
                                "confidence": 0.95,
                                "reasoning": "User wants to find a scene",
                            }
                        )
                    }
                }
            ]
        }
        decision = router._parse_response(data)
        assert decision.task_type == TaskType.SCENE_QUERY
        assert decision.complexity == TaskComplexity.MODERATE
        assert decision.suggested_tools == ["scene_search"]
        assert decision.confidence == 0.95
        assert decision.primary_context == "writers_room"

    def test_parse_json_in_code_block(self):
        router = _make_router()
        content = '```json\n{"task_type": "conversation", "complexity": "simple", "tools": []}\n```'
        data = {"choices": [{"message": {"content": content}}]}
        decision = router._parse_response(data)
        assert decision.task_type == TaskType.CONVERSATION
        assert decision.complexity == TaskComplexity.SIMPLE

    def test_parse_json_in_generic_code_block(self):
        router = _make_router()
        content = '```\n{"task_type": "email", "complexity": "moderate", "tools": ["send_email"]}\n```'
        data = {"choices": [{"message": {"content": content}}]}
        decision = router._parse_response(data)
        assert decision.task_type == TaskType.EMAIL

    def test_parse_invalid_json_falls_back(self):
        router = _make_router()
        data = {"choices": [{"message": {"content": "not json at all"}}]}
        decision = router._parse_response(data)
        # Should fall back to default routing
        assert decision.confidence == 0.6  # Default keyword routing confidence

    def test_parse_empty_response(self):
        router = _make_router()
        data = {"choices": []}
        decision = router._parse_response(data)
        assert decision.confidence == 0.6  # Fallback

    def test_parse_multi_tool_sets_agent_mode(self):
        router = _make_router()
        data = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "task_type": "scene_management",
                                "complexity": "complex",
                                "tools": ["scene_search", "scene_update"],
                                "requires_tools": True,
                            }
                        )
                    }
                }
            ]
        }
        decision = router._parse_response(data)
        assert decision.agent_mode is True
        assert decision.expected_turns >= 2


# =========================================================================
# _estimate_turns Tests
# =========================================================================


class TestEstimateTurns:
    """Test turn estimation"""

    def test_simple_is_one_turn(self):
        router = _make_router()
        assert router._estimate_turns(TaskComplexity.SIMPLE, 0) == 1

    def test_moderate_matches_tools(self):
        router = _make_router()
        assert router._estimate_turns(TaskComplexity.MODERATE, 1) == 1
        assert router._estimate_turns(TaskComplexity.MODERATE, 3) == 3

    def test_complex_adds_buffer(self):
        router = _make_router()
        assert router._estimate_turns(TaskComplexity.COMPLEX, 2) == 3
        assert router._estimate_turns(TaskComplexity.COMPLEX, 0) == 2


# =========================================================================
# Cache Tests
# =========================================================================


class TestRouterCache:
    """Test routing cache behavior"""

    def test_cache_only_for_glm_results(self):
        """Cache only stores GLM results, not keyword fallback (by design)"""
        config = RouterConfig(enabled=False, cache_decisions=True)
        router = GLMRouter(config)

        # Keyword fallback does NOT cache (it's cheap)
        _run(router.analyze("find the proposal scene"))
        assert len(router._cache) == 0

    def test_cache_disabled(self):
        config = RouterConfig(enabled=False, cache_decisions=False)
        router = GLMRouter(config)

        _run(router.analyze("test message"))
        assert len(router._cache) == 0

    def test_cache_key_normalization(self):
        router = _make_router()
        k1 = router._cache_key("Hello Boss")
        k2 = router._cache_key("hello boss")
        assert k1 == k2

    def test_cache_stores_glm_results(self):
        """When GLM is enabled and succeeds, results should be cached"""
        config = RouterConfig(enabled=True, cache_decisions=True)
        router = GLMRouter(config)

        mock_decision = RouterDecision(
            task_type=TaskType.SCENE_QUERY,
            complexity=TaskComplexity.MODERATE,
            primary_context="writers_room",
            suggested_tools=["scene_search"],
        )

        with patch.object(router, "_call_glm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_decision
            _run(router.analyze("find scenes"))
            assert len(router._cache) == 1

    def test_clear_cache(self):
        config = RouterConfig(enabled=True, cache_decisions=True)
        router = GLMRouter(config)

        mock_decision = RouterDecision(
            task_type=TaskType.CONVERSATION,
            complexity=TaskComplexity.SIMPLE,
            primary_context="general",
        )

        with patch.object(router, "_call_glm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = mock_decision
            _run(router.analyze("test"))
            assert len(router._cache) > 0

        router.clear_cache()
        assert len(router._cache) == 0


# =========================================================================
# System Prompt Coverage
# =========================================================================


class TestRouterSystemPrompt:
    """Verify system prompt contains all tool categories"""

    def test_contains_scene_tools(self):
        assert "scene_search" in ROUTER_SYSTEM_PROMPT
        assert "scene_get" in ROUTER_SYSTEM_PROMPT
        assert "scene_update" in ROUTER_SYSTEM_PROMPT
        assert "scene_reorder" in ROUTER_SYSTEM_PROMPT
        assert "scene_link" in ROUTER_SYSTEM_PROMPT

    def test_contains_document_tools(self):
        assert "document_ingest" in ROUTER_SYSTEM_PROMPT
        assert "document_search" in ROUTER_SYSTEM_PROMPT
        assert "document_get_context" in ROUTER_SYSTEM_PROMPT
        assert "document_list" in ROUTER_SYSTEM_PROMPT
        assert "document_get" in ROUTER_SYSTEM_PROMPT
        assert "document_get_chapter" in ROUTER_SYSTEM_PROMPT
        assert "document_status" in ROUTER_SYSTEM_PROMPT
        assert "document_delete" in ROUTER_SYSTEM_PROMPT

    def test_contains_book_tools(self):
        assert "book_study" in ROUTER_SYSTEM_PROMPT
        assert "book_study_status" in ROUTER_SYSTEM_PROMPT
        assert "book_study_jobs" in ROUTER_SYSTEM_PROMPT
        assert "book_list_studied" in ROUTER_SYSTEM_PROMPT
        assert "book_get_understanding" in ROUTER_SYSTEM_PROMPT

    def test_contains_mentor_tools(self):
        assert "mentor_load_books" in ROUTER_SYSTEM_PROMPT
        assert "mentor_analyze" in ROUTER_SYSTEM_PROMPT
        assert "mentor_brainstorm" in ROUTER_SYSTEM_PROMPT
        assert "mentor_check_rules" in ROUTER_SYSTEM_PROMPT
        assert "mentor_find_inspiration" in ROUTER_SYSTEM_PROMPT
        assert "mentor_ask" in ROUTER_SYSTEM_PROMPT
        assert "mentor_compare" in ROUTER_SYSTEM_PROMPT

    def test_contains_knowledge_tools(self):
        assert "knowledge_search" in ROUTER_SYSTEM_PROMPT

    def test_contains_communication_tools(self):
        assert "send_email" in ROUTER_SYSTEM_PROMPT
        assert "send_screenplay" in ROUTER_SYSTEM_PROMPT

    def test_contains_vision_tools(self):
        assert "camera_analyze" in ROUTER_SYSTEM_PROMPT
        assert "generate_image" in ROUTER_SYSTEM_PROMPT


# =========================================================================
# GLM Fallback on Error
# =========================================================================


class TestGLMFallback:
    """Test fallback behavior when GLM is unavailable"""

    def test_disabled_router_uses_keyword(self):
        router = _make_router(enabled=False)
        decision = _run(router.analyze("find the rooftop scene"))
        assert decision.confidence == 0.6  # Keyword routing confidence
        assert "scene_search" in decision.suggested_tools

    def test_enabled_router_falls_back_on_error(self):
        config = RouterConfig(enabled=True, fallback_on_error=True)
        router = GLMRouter(config)

        # Patch _call_glm to raise
        with patch.object(router, "_call_glm", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("GLM unavailable")
            decision = _run(router.analyze("find scenes"))
            assert decision.confidence == 0.6  # Fell back to keyword

    def test_enabled_router_raises_when_no_fallback(self):
        config = RouterConfig(enabled=True, fallback_on_error=False)
        router = GLMRouter(config)

        with patch.object(router, "_call_glm", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("GLM unavailable")
            with pytest.raises(Exception, match="GLM unavailable"):
                _run(router.analyze("find scenes"))


# =========================================================================
# RouterDecision
# =========================================================================


class TestRouterDecision:
    """Test RouterDecision serialization"""

    def test_to_dict(self):
        decision = RouterDecision(
            task_type=TaskType.SCENE_QUERY,
            complexity=TaskComplexity.MODERATE,
            primary_context="writers_room",
            suggested_tools=["scene_search"],
            requires_tools=True,
            confidence=0.9,
            reasoning="test",
        )
        d = decision.to_dict()
        assert d["task_type"] == "scene_query"
        assert d["complexity"] == "moderate"
        assert d["suggested_tools"] == ["scene_search"]
        assert d["requires_tools"] is True

    def test_defaults(self):
        decision = RouterDecision(
            task_type=TaskType.CONVERSATION,
            complexity=TaskComplexity.SIMPLE,
            primary_context="general",
        )
        assert decision.suggested_tools == []
        assert decision.requires_tools is False
        assert decision.agent_mode is False
        assert decision.expected_turns == 1
