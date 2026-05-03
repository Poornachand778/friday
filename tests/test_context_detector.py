"""
Tests for Context Detector
============================

Tests explicit switch detection, location-based detection,
keyword detection, sticky context, and state management.

Run with: pytest tests/test_context_detector.py -v
"""

import sys
from pathlib import Path

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from orchestrator.context.detector import ContextDetector
from orchestrator.context.contexts import (
    Context,
    ContextType,
    CONTEXTS,
    get_context,
    get_context_by_name,
    list_contexts,
)


# =========================================================================
# Context Module Helpers
# =========================================================================


class TestContextType:
    """Test ContextType enum values"""

    def test_writers_room_value(self):
        assert ContextType.WRITERS_ROOM.value == "writers_room"

    def test_kitchen_value(self):
        assert ContextType.KITCHEN.value == "kitchen"

    def test_storyboard_value(self):
        assert ContextType.STORYBOARD.value == "storyboard"

    def test_general_value(self):
        assert ContextType.GENERAL.value == "general"

    def test_all_context_types_in_contexts_dict(self):
        for ct in ContextType:
            assert ct in CONTEXTS, f"{ct} missing from CONTEXTS dict"


class TestGetContext:
    """Test get_context() and get_context_by_name() helpers"""

    def test_get_context_writers_room(self):
        ctx = get_context(ContextType.WRITERS_ROOM)
        assert ctx.name == "Writers Room"
        assert ctx.context_type == ContextType.WRITERS_ROOM

    def test_get_context_kitchen(self):
        ctx = get_context(ContextType.KITCHEN)
        assert ctx.name == "Kitchen"

    def test_get_context_storyboard(self):
        ctx = get_context(ContextType.STORYBOARD)
        assert ctx.name == "Storyboard Room"

    def test_get_context_general(self):
        ctx = get_context(ContextType.GENERAL)
        assert ctx.name == "General"

    def test_get_context_by_name_writers_room(self):
        ctx = get_context_by_name("writers_room")
        assert ctx is not None
        assert ctx.context_type == ContextType.WRITERS_ROOM

    def test_get_context_by_name_with_spaces(self):
        ctx = get_context_by_name("Writers Room")
        assert ctx is not None
        assert ctx.context_type == ContextType.WRITERS_ROOM

    def test_get_context_by_name_unknown(self):
        ctx = get_context_by_name("nonexistent_room")
        assert ctx is None

    def test_list_contexts_returns_all(self):
        contexts = list_contexts()
        assert len(contexts) == len(ContextType)


class TestContextHasTools:
    """Verify each context has expected tool lists"""

    def test_writers_room_has_scene_tools(self):
        ctx = get_context(ContextType.WRITERS_ROOM)
        assert "scene_search" in ctx.available_tools
        assert "scene_get" in ctx.available_tools
        assert "scene_update" in ctx.available_tools

    def test_writers_room_has_mentor_tools(self):
        ctx = get_context(ContextType.WRITERS_ROOM)
        assert "mentor_analyze" in ctx.available_tools
        assert "mentor_brainstorm" in ctx.available_tools

    def test_kitchen_has_camera(self):
        ctx = get_context(ContextType.KITCHEN)
        assert "camera_analyze" in ctx.available_tools

    def test_general_has_document_tools(self):
        ctx = get_context(ContextType.GENERAL)
        assert "document_search" in ctx.available_tools
        assert "document_list" in ctx.available_tools

    def test_writers_room_has_detection_keywords(self):
        ctx = get_context(ContextType.WRITERS_ROOM)
        assert len(ctx.detection_keywords) > 0
        assert "scene" in ctx.detection_keywords

    def test_general_has_no_detection_keywords(self):
        ctx = get_context(ContextType.GENERAL)
        assert len(ctx.detection_keywords) == 0


# =========================================================================
# Explicit Switch Detection
# =========================================================================


class TestExplicitSwitchDetection:
    """Test _detect_explicit_switch patterns"""

    def setup_method(self):
        self.detector = ContextDetector()

    def test_switch_to_kitchen(self):
        ctx, confidence = self.detector.detect("switch to kitchen")
        assert ctx.context_type == ContextType.KITCHEN
        assert confidence == 1.0

    def test_switch_to_writing(self):
        ctx, confidence = self.detector.detect("switch to writing")
        assert ctx.context_type == ContextType.WRITERS_ROOM
        assert confidence == 1.0

    def test_switch_to_script(self):
        ctx, confidence = self.detector.detect("switch to script")
        assert ctx.context_type == ContextType.WRITERS_ROOM
        assert confidence == 1.0

    def test_go_to_storyboard(self):
        ctx, confidence = self.detector.detect("go to storyboard")
        assert ctx.context_type == ContextType.STORYBOARD
        assert confidence == 1.0

    def test_lets_go_to_kitchen(self):
        ctx, confidence = self.detector.detect("let's go to kitchen")
        assert ctx.context_type == ContextType.KITCHEN
        assert confidence == 1.0

    def test_cooking_mode(self):
        ctx, confidence = self.detector.detect("cooking mode")
        assert ctx.context_type == ContextType.KITCHEN
        assert confidence == 1.0

    def test_writing_mode(self):
        ctx, confidence = self.detector.detect("writing mode")
        assert ctx.context_type == ContextType.WRITERS_ROOM
        assert confidence == 1.0

    def test_visual_mode(self):
        ctx, confidence = self.detector.detect("visual mode")
        assert ctx.context_type == ContextType.STORYBOARD
        assert confidence == 1.0

    def test_general_mode(self):
        ctx, confidence = self.detector.detect("general mode")
        assert ctx.context_type == ContextType.GENERAL
        assert confidence == 1.0

    def test_in_the_kitchen(self):
        ctx, confidence = self.detector.detect("I'm in the kitchen")
        assert ctx.context_type == ContextType.KITCHEN
        assert confidence == 1.0

    def test_case_insensitive(self):
        ctx, confidence = self.detector.detect("SWITCH TO KITCHEN")
        assert ctx.context_type == ContextType.KITCHEN
        assert confidence == 1.0

    def test_explicit_switch_updates_current(self):
        self.detector.detect("switch to kitchen")
        assert self.detector._current_context == ContextType.KITCHEN

    def test_unrecognized_switch_target(self):
        """Switch to unknown room should not match explicit pattern"""
        ctx, confidence = self.detector.detect("switch to basement")
        # Should fall through to keyword/sticky/default, not match explicit
        assert confidence < 1.0


# =========================================================================
# Location-Based Detection
# =========================================================================


class TestLocationDetection:
    """Test _detect_from_location"""

    def setup_method(self):
        self.detector = ContextDetector()

    def test_kitchen_location(self):
        ctx, confidence = self.detector.detect("hello", location="kitchen")
        assert ctx.context_type == ContextType.KITCHEN
        assert confidence == 0.9

    def test_office_location(self):
        ctx, confidence = self.detector.detect("hello", location="office")
        assert ctx.context_type == ContextType.WRITERS_ROOM
        assert confidence == 0.9

    def test_studio_location(self):
        ctx, confidence = self.detector.detect("hello", location="studio")
        assert ctx.context_type == ContextType.STORYBOARD
        assert confidence == 0.9

    def test_device_id_kitchen(self):
        ctx, confidence = self.detector.detect("hello", device_id="kitchen-speaker")
        assert ctx.context_type == ContextType.KITCHEN
        assert confidence == 0.9

    def test_device_id_writers(self):
        ctx, confidence = self.detector.detect("hello", device_id="writers-desk")
        assert ctx.context_type == ContextType.WRITERS_ROOM
        assert confidence == 0.9

    def test_unknown_location_falls_through(self):
        ctx, confidence = self.detector.detect("hello", location="garage")
        # Should fall to default, not location
        assert confidence != 0.9

    def test_no_location_no_device(self):
        ctx, confidence = self.detector.detect("hello")
        # No location info → falls through to keyword/sticky/default
        assert confidence <= 0.6

    def test_location_case_insensitive(self):
        ctx, confidence = self.detector.detect("hello", location="KITCHEN")
        assert ctx.context_type == ContextType.KITCHEN


# =========================================================================
# Keyword Detection
# =========================================================================


class TestKeywordDetection:
    """Test _detect_from_keywords"""

    def setup_method(self):
        self.detector = ContextDetector()

    def test_screenplay_keyword(self):
        ctx, confidence = self.detector.detect("help me with the screenplay scene")
        assert ctx.context_type == ContextType.WRITERS_ROOM
        assert confidence > 0.5

    def test_kitchen_keywords(self):
        ctx, confidence = self.detector.detect("what recipe should I cook tonight?")
        assert ctx.context_type == ContextType.KITCHEN
        assert confidence > 0.5

    def test_storyboard_keywords(self):
        ctx, confidence = self.detector.detect(
            "I need to visualize this shot with the camera angle"
        )
        assert ctx.context_type == ContextType.STORYBOARD
        assert confidence > 0.5

    def test_multiple_keywords_boost_score(self):
        """More keyword matches should produce higher confidence"""
        ctx_low, conf_low = self.detector.detect("scene")
        self.detector.reset()
        ctx_high, conf_high = self.detector.detect(
            "scene dialogue character beat story"
        )
        # Both should match writers room but multi-keyword should be higher confidence
        assert conf_high >= conf_low

    def test_keyword_below_threshold_doesnt_switch(self):
        """Low-confidence keyword match should not override sticky context"""
        # First, set context to kitchen
        self.detector.set_context(ContextType.KITCHEN)
        # Then send a message with minimal keyword signal
        ctx, confidence = self.detector.detect("something")
        # Should stay in kitchen (sticky)
        assert ctx.context_type == ContextType.KITCHEN


# =========================================================================
# Sticky Context Behavior
# =========================================================================


class TestStickyContext:
    """Test sticky context behavior"""

    def setup_method(self):
        self.detector = ContextDetector(sticky=True)

    def test_context_persists_across_messages(self):
        # Switch to kitchen
        self.detector.detect("switch to kitchen")
        # Next message without context signal should stay in kitchen
        ctx, confidence = self.detector.detect("what should I do?")
        assert ctx.context_type == ContextType.KITCHEN
        assert confidence == 0.6  # Sticky confidence

    def test_explicit_switch_overrides_sticky(self):
        self.detector.detect("switch to kitchen")
        ctx, _ = self.detector.detect("switch to writing")
        assert ctx.context_type == ContextType.WRITERS_ROOM

    def test_high_confidence_keyword_overrides_sticky(self):
        """High confidence keyword detection should override sticky context"""
        self.detector.detect("switch to kitchen")
        # Strong writing keywords should switch context
        ctx, confidence = self.detector.detect(
            "help me revise the screenplay scene dialogue and character arc"
        )
        if confidence > 0.7:
            assert ctx.context_type == ContextType.WRITERS_ROOM

    def test_non_sticky_mode(self):
        detector = ContextDetector(sticky=False)
        detector.detect("switch to kitchen")
        # Without sticky, should fall to default when no signal
        ctx, confidence = detector.detect("what should I do?")
        # Should use default since sticky=False and no keyword match
        assert confidence == 0.5  # Default confidence

    def test_sticky_returns_0_6_confidence(self):
        self.detector.detect("switch to kitchen")
        _, confidence = self.detector.detect("hello")
        assert confidence == 0.6


# =========================================================================
# Default Context
# =========================================================================


class TestDefaultContext:
    """Test default context fallback"""

    def test_default_is_writers_room(self):
        detector = ContextDetector()
        ctx, confidence = detector.detect("hello there")
        assert ctx.context_type == ContextType.WRITERS_ROOM
        assert confidence == 0.5

    def test_custom_default(self):
        detector = ContextDetector(default_context=ContextType.GENERAL)
        ctx, confidence = detector.detect("hello there")
        assert ctx.context_type == ContextType.GENERAL
        assert confidence == 0.5

    def test_default_sets_current_context(self):
        detector = ContextDetector()
        detector.detect("hello there")
        assert detector._current_context == ContextType.WRITERS_ROOM


# =========================================================================
# Detection Priority
# =========================================================================


class TestDetectionPriority:
    """Test that detection priority is: explicit > location > keyword > sticky > default"""

    def setup_method(self):
        self.detector = ContextDetector()

    def test_explicit_beats_location(self):
        """Explicit switch should override location hint"""
        ctx, confidence = self.detector.detect("switch to kitchen", location="studio")
        assert ctx.context_type == ContextType.KITCHEN
        assert confidence == 1.0

    def test_location_beats_keywords(self):
        """Location should override keyword signals"""
        ctx, confidence = self.detector.detect(
            "help with the screenplay", location="kitchen"
        )
        assert ctx.context_type == ContextType.KITCHEN
        assert confidence == 0.9


# =========================================================================
# State Management
# =========================================================================


class TestStateManagement:
    """Test set_context, get_current_context, reset"""

    def setup_method(self):
        self.detector = ContextDetector()

    def test_set_context(self):
        ctx = self.detector.set_context(ContextType.KITCHEN)
        assert ctx.context_type == ContextType.KITCHEN
        assert self.detector._current_context == ContextType.KITCHEN

    def test_get_current_context_initially_none(self):
        assert self.detector.get_current_context() is None

    def test_get_current_context_after_set(self):
        self.detector.set_context(ContextType.STORYBOARD)
        ctx = self.detector.get_current_context()
        assert ctx is not None
        assert ctx.context_type == ContextType.STORYBOARD

    def test_get_current_context_after_detect(self):
        self.detector.detect("switch to kitchen")
        ctx = self.detector.get_current_context()
        assert ctx is not None
        assert ctx.context_type == ContextType.KITCHEN

    def test_reset_clears_context(self):
        self.detector.set_context(ContextType.KITCHEN)
        self.detector.reset()
        assert self.detector._current_context is None
        assert self.detector.get_current_context() is None

    def test_reset_then_detect_uses_default(self):
        self.detector.set_context(ContextType.KITCHEN)
        self.detector.reset()
        ctx, confidence = self.detector.detect("hello")
        assert ctx.context_type == ContextType.WRITERS_ROOM  # default
        assert confidence == 0.5


# =========================================================================
# Edge Cases
# =========================================================================


class TestEdgeCases:
    """Test edge cases"""

    def setup_method(self):
        self.detector = ContextDetector()

    def test_empty_message(self):
        ctx, confidence = self.detector.detect("")
        # Should return default
        assert ctx is not None
        assert confidence <= 0.6

    def test_very_long_message(self):
        msg = "scene " * 500
        ctx, _ = self.detector.detect(msg)
        assert ctx is not None

    def test_none_location_and_device(self):
        ctx, _ = self.detector.detect("hello", device_id=None, location=None)
        assert ctx is not None

    def test_empty_conversation_history(self):
        ctx, _ = self.detector.detect("hello", conversation_history=[])
        assert ctx is not None

    def test_confidence_return_type(self):
        _, confidence = self.detector.detect("switch to kitchen")
        assert isinstance(confidence, float)

    def test_context_return_type(self):
        ctx, _ = self.detector.detect("hello")
        assert isinstance(ctx, Context)
