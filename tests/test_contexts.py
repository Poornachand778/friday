"""
Tests for orchestrator/context/contexts.py
==========================================

Tests for ContextType, Context, CONTEXTS dict, and helper functions.

Tests: 40+
"""

import pytest

from orchestrator.context.contexts import (
    Context,
    ContextType,
    CONTEXTS,
    get_context,
    get_context_by_name,
    list_contexts,
)


# ── ContextType Enum ─────────────────────────────────────────────────────


class TestContextType:
    def test_values(self):
        assert ContextType.WRITERS_ROOM == "writers_room"
        assert ContextType.KITCHEN == "kitchen"
        assert ContextType.STORYBOARD == "storyboard"
        assert ContextType.GENERAL == "general"

    def test_count(self):
        assert len(ContextType) == 4


# ── Context Dataclass ────────────────────────────────────────────────────


class TestContextDataclass:
    def test_context_creation(self):
        ctx = Context(
            name="Test",
            context_type=ContextType.GENERAL,
            description="A test context",
            system_prompt_addition="Be helpful",
        )
        assert ctx.name == "Test"
        assert ctx.context_type == ContextType.GENERAL
        assert ctx.available_tools == []
        assert ctx.lora_adapter is None
        assert ctx.external_apis == []
        assert ctx.detection_keywords == []

    def test_context_with_tools(self):
        ctx = Context(
            name="Room",
            context_type=ContextType.WRITERS_ROOM,
            description="Writing",
            system_prompt_addition="Write stuff",
            available_tools=["scene_search", "scene_get"],
        )
        assert len(ctx.available_tools) == 2

    def test_context_with_lora(self):
        ctx = Context(
            name="Room",
            context_type=ContextType.WRITERS_ROOM,
            description="Writing",
            system_prompt_addition="Write",
            lora_adapter="friday-script",
        )
        assert ctx.lora_adapter == "friday-script"


# ── CONTEXTS Dict ────────────────────────────────────────────────────────


class TestContextsDict:
    def test_all_context_types_defined(self):
        for ct in ContextType:
            assert ct in CONTEXTS

    def test_writers_room(self):
        ctx = CONTEXTS[ContextType.WRITERS_ROOM]
        assert ctx.name == "Writers Room"
        assert "scene_search" in ctx.available_tools
        assert "scene_get" in ctx.available_tools
        assert "scene_update" in ctx.available_tools
        assert ctx.lora_adapter == "friday-script"
        assert "scene" in ctx.detection_keywords

    def test_kitchen(self):
        ctx = CONTEXTS[ContextType.KITCHEN]
        assert ctx.name == "Kitchen"
        assert "camera_analyze" in ctx.available_tools
        assert "cook" in ctx.detection_keywords
        assert ctx.lora_adapter is None

    def test_storyboard(self):
        ctx = CONTEXTS[ContextType.STORYBOARD]
        assert ctx.name == "Storyboard Room"
        assert "generate_image" in ctx.available_tools
        assert "storyboard" in ctx.detection_keywords
        assert "vision" in ctx.external_apis

    def test_general(self):
        ctx = CONTEXTS[ContextType.GENERAL]
        assert ctx.name == "General"
        assert "send_email" in ctx.available_tools
        assert ctx.detection_keywords == []  # Fallback, no keywords
        assert ctx.lora_adapter is None

    def test_writers_room_has_document_tools(self):
        ctx = CONTEXTS[ContextType.WRITERS_ROOM]
        assert "document_search" in ctx.available_tools
        assert "document_get_context" in ctx.available_tools

    def test_writers_room_has_mentor_tools(self):
        ctx = CONTEXTS[ContextType.WRITERS_ROOM]
        assert "mentor_analyze" in ctx.available_tools
        assert "mentor_brainstorm" in ctx.available_tools
        assert "mentor_check_rules" in ctx.available_tools
        assert "book_study" in ctx.available_tools

    def test_general_has_knowledge_tools(self):
        ctx = CONTEXTS[ContextType.GENERAL]
        assert "knowledge_search" in ctx.available_tools
        assert "mentor_ask" in ctx.available_tools
        assert "mentor_compare" in ctx.available_tools

    def test_all_contexts_have_system_prompt(self):
        for ct, ctx in CONTEXTS.items():
            assert ctx.system_prompt_addition, f"{ct} missing system_prompt_addition"
            assert len(ctx.system_prompt_addition) > 20


# ── get_context ──────────────────────────────────────────────────────────


class TestGetContext:
    def test_get_existing(self):
        ctx = get_context(ContextType.WRITERS_ROOM)
        assert ctx.context_type == ContextType.WRITERS_ROOM

    def test_get_general(self):
        ctx = get_context(ContextType.GENERAL)
        assert ctx.name == "General"

    def test_get_all_types(self):
        for ct in ContextType:
            ctx = get_context(ct)
            assert ctx.context_type == ct


# ── get_context_by_name ──────────────────────────────────────────────────


class TestGetContextByName:
    def test_by_value(self):
        ctx = get_context_by_name("writers_room")
        assert ctx is not None
        assert ctx.context_type == ContextType.WRITERS_ROOM

    def test_by_display_name(self):
        ctx = get_context_by_name("Writers Room")
        assert ctx is not None
        assert ctx.context_type == ContextType.WRITERS_ROOM

    def test_by_display_name_lowercase(self):
        ctx = get_context_by_name("writers room")
        assert ctx is not None

    def test_kitchen(self):
        ctx = get_context_by_name("kitchen")
        assert ctx is not None
        assert ctx.context_type == ContextType.KITCHEN

    def test_storyboard(self):
        ctx = get_context_by_name("storyboard")
        assert ctx is not None
        assert ctx.context_type == ContextType.STORYBOARD

    def test_storyboard_room(self):
        ctx = get_context_by_name("Storyboard Room")
        assert ctx is not None

    def test_general(self):
        ctx = get_context_by_name("general")
        assert ctx is not None
        assert ctx.context_type == ContextType.GENERAL

    def test_nonexistent(self):
        ctx = get_context_by_name("nonexistent_room")
        assert ctx is None


# ── list_contexts ────────────────────────────────────────────────────────


class TestListContexts:
    def test_returns_all(self):
        contexts = list_contexts()
        assert len(contexts) == 4

    def test_all_are_context_objects(self):
        for ctx in list_contexts():
            assert isinstance(ctx, Context)

    def test_includes_all_types(self):
        types = {ctx.context_type for ctx in list_contexts()}
        assert types == set(ContextType)
