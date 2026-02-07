"""
Tests for Triplet Extractor
============================

Tests ExtractedTriplet, ExtractionResult, TripletExtractor (init, fallback,
parse, extract, batch), close, and the convenience function.

Run with: pytest tests/test_triplet_extractor.py -v
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from memory.operations.triplet_extractor import (
    ENTITY_TYPES,
    RELATION_TYPES,
    ExtractedTriplet,
    ExtractionResult,
    TripletExtractor,
    extract_triplets,
)


# =========================================================================
# Helpers
# =========================================================================


def _run(coro):
    """Run async coroutine synchronously"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_triplet(**overrides):
    """Create an ExtractedTriplet with sensible defaults"""
    defaults = {
        "subject": "Boss",
        "subject_type": "person",
        "relation": "discusses",
        "object": "climax",
        "object_type": "concept",
        "confidence": 0.9,
        "source_text": "Boss discussed the climax",
    }
    defaults.update(overrides)
    return ExtractedTriplet(**defaults)


def _make_glm_response(triplet_data, total_tokens=100):
    """Build a mock GLM API response dict"""
    content = json.dumps({"triplets": triplet_data})
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"total_tokens": total_tokens},
    }


# =========================================================================
# Constants
# =========================================================================


class TestConstants:
    """Test module-level constants"""

    def test_entity_types_defined(self):
        expected = [
            "character",
            "scene",
            "project",
            "person",
            "concept",
            "location",
            "event",
        ]
        assert ENTITY_TYPES == expected

    def test_relation_types_defined(self):
        expected = [
            "discusses",
            "contains",
            "relates_to",
            "character_in",
            "scene_in",
            "has_relationship",
            "creates",
            "wants",
            "deadline_for",
            "involves",
        ]
        assert RELATION_TYPES == expected


# =========================================================================
# ExtractedTriplet
# =========================================================================


class TestExtractedTriplet:
    """Test ExtractedTriplet dataclass"""

    def test_create_triplet_all_fields(self):
        t = ExtractedTriplet(
            subject="Ravi",
            subject_type="character",
            relation="character_in",
            object="Gusagusalu",
            object_type="project",
            confidence=0.95,
            source_text="Ravi in Gusagusalu",
        )
        assert t.subject == "Ravi"
        assert t.subject_type == "character"
        assert t.relation == "character_in"
        assert t.object == "Gusagusalu"
        assert t.object_type == "project"
        assert t.confidence == 0.95
        assert t.source_text == "Ravi in Gusagusalu"

    def test_default_confidence(self):
        t = ExtractedTriplet(
            subject="Boss",
            subject_type="person",
            relation="discusses",
            object="climax",
            object_type="concept",
        )
        assert t.confidence == 0.8

    def test_default_source_text(self):
        t = ExtractedTriplet(
            subject="Boss",
            subject_type="person",
            relation="discusses",
            object="climax",
            object_type="concept",
        )
        assert t.source_text == ""

    def test_custom_confidence(self):
        t = _make_triplet(confidence=0.42)
        assert t.confidence == 0.42

    def test_as_tuple(self):
        t = _make_triplet(subject="Boss", relation="discusses", object="climax")
        assert t.as_tuple() == ("Boss", "discusses", "climax")

    def test_as_tuple_returns_three_elements(self):
        t = _make_triplet()
        result = t.as_tuple()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_to_dict_all_keys(self):
        t = _make_triplet()
        d = t.to_dict()
        expected_keys = {
            "subject",
            "subject_type",
            "relation",
            "object",
            "object_type",
            "confidence",
            "source_text",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        t = ExtractedTriplet(
            subject="Ravi",
            subject_type="character",
            relation="character_in",
            object="Scene 5",
            object_type="scene",
            confidence=0.75,
            source_text="Ravi enters Scene 5",
        )
        d = t.to_dict()
        assert d["subject"] == "Ravi"
        assert d["subject_type"] == "character"
        assert d["relation"] == "character_in"
        assert d["object"] == "Scene 5"
        assert d["object_type"] == "scene"
        assert d["confidence"] == 0.75
        assert d["source_text"] == "Ravi enters Scene 5"


# =========================================================================
# ExtractionResult
# =========================================================================


class TestExtractionResult:
    """Test ExtractionResult dataclass"""

    def test_empty_result_defaults(self):
        r = ExtractionResult()
        assert r.triplets == []
        assert r.source_text == ""
        assert r.model_used == ""
        assert r.tokens_used == 0

    def test_count_empty(self):
        r = ExtractionResult()
        assert r.count == 0

    def test_count_with_triplets(self):
        r = ExtractionResult(
            triplets=[_make_triplet(), _make_triplet(), _make_triplet()]
        )
        assert r.count == 3

    def test_high_confidence_default_threshold(self):
        t_low = _make_triplet(confidence=0.5)
        t_med = _make_triplet(confidence=0.8)
        t_high = _make_triplet(confidence=0.95)
        r = ExtractionResult(triplets=[t_low, t_med, t_high])
        high = r.high_confidence()
        assert len(high) == 2
        assert t_med in high
        assert t_high in high

    def test_high_confidence_custom_threshold(self):
        t_low = _make_triplet(confidence=0.3)
        t_mid = _make_triplet(confidence=0.6)
        t_high = _make_triplet(confidence=0.9)
        r = ExtractionResult(triplets=[t_low, t_mid, t_high])
        high = r.high_confidence(threshold=0.5)
        assert len(high) == 2
        assert t_mid in high
        assert t_high in high

    def test_high_confidence_none_above(self):
        t1 = _make_triplet(confidence=0.1)
        t2 = _make_triplet(confidence=0.2)
        r = ExtractionResult(triplets=[t1, t2])
        assert r.high_confidence(threshold=0.9) == []

    def test_by_relation_matches(self):
        t1 = _make_triplet(relation="discusses")
        t2 = _make_triplet(relation="character_in")
        t3 = _make_triplet(relation="discusses")
        r = ExtractionResult(triplets=[t1, t2, t3])
        discusses = r.by_relation("discusses")
        assert len(discusses) == 2
        for t in discusses:
            assert t.relation == "discusses"

    def test_by_relation_no_match(self):
        t1 = _make_triplet(relation="discusses")
        r = ExtractionResult(triplets=[t1])
        assert r.by_relation("character_in") == []

    def test_to_tuples(self):
        t1 = _make_triplet(subject="Boss", relation="discusses", object="climax")
        t2 = _make_triplet(subject="Ravi", relation="character_in", object="Scene 5")
        r = ExtractionResult(triplets=[t1, t2])
        tuples = r.to_tuples()
        assert tuples == [
            ("Boss", "discusses", "climax"),
            ("Ravi", "character_in", "Scene 5"),
        ]

    def test_to_tuples_empty(self):
        r = ExtractionResult()
        assert r.to_tuples() == []


# =========================================================================
# TripletExtractor - Init
# =========================================================================


class TestTripletExtractorInit:
    """Test TripletExtractor initialization"""

    def test_default_config(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove env vars so defaults apply
            env = os.environ.copy()
            env.pop("ZHIPU_API_KEY", None)
            env.pop("ZHIPU_BASE_URL", None)
            with patch.dict(os.environ, env, clear=True):
                ext = TripletExtractor()
                assert ext.api_key == ""
                assert ext.base_url == "https://api.z.ai/api/paas/v4"
                assert ext.model_name == "glm-4.7-flash"
                assert ext.timeout == 10.0
                assert ext.min_confidence == 0.6

    def test_custom_config(self):
        ext = TripletExtractor(
            api_key="test-key-123",
            base_url="https://custom.api/v1",
            model_name="glm-custom",
            timeout=30.0,
            min_confidence=0.9,
        )
        assert ext.api_key == "test-key-123"
        assert ext.base_url == "https://custom.api/v1"
        assert ext.model_name == "glm-custom"
        assert ext.timeout == 30.0
        assert ext.min_confidence == 0.9

    def test_is_configured_with_key(self):
        ext = TripletExtractor(api_key="my-key")
        assert ext.is_configured is True

    def test_is_configured_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("ZHIPU_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                ext = TripletExtractor()
                assert ext.is_configured is False

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "env-key-456"}):
            ext = TripletExtractor()
            assert ext.api_key == "env-key-456"
            assert ext.is_configured is True

    def test_base_url_from_env(self):
        with patch.dict(os.environ, {"ZHIPU_BASE_URL": "https://env.api/v2"}):
            ext = TripletExtractor()
            assert ext.base_url == "https://env.api/v2"

    def test_explicit_key_overrides_env(self):
        with patch.dict(os.environ, {"ZHIPU_API_KEY": "env-key"}):
            ext = TripletExtractor(api_key="explicit-key")
            assert ext.api_key == "explicit-key"


# =========================================================================
# TripletExtractor - Fallback Extraction
# =========================================================================


class TestFallbackExtraction:
    """Test rule-based _fallback_extract (no mocking needed)"""

    def setup_method(self):
        self.ext = TripletExtractor(api_key="")

    def test_empty_text_returns_empty(self):
        result = self.ext._fallback_extract("", None)
        assert result.count == 0
        assert result.model_used == "fallback"
        assert result.tokens_used == 0

    def test_character_and_project_match(self):
        result = self.ext._fallback_extract("Ravi is in Gusagusalu", None)
        assert result.count >= 1
        tuples = result.to_tuples()
        assert ("Ravi", "character_in", "Gusagusalu") in tuples

    def test_multiple_characters_same_project(self):
        result = self.ext._fallback_extract(
            "Ravi and Father are in Gusagusalu scene", None
        )
        tuples = result.to_tuples()
        assert ("Ravi", "character_in", "Gusagusalu") in tuples
        assert ("Father", "character_in", "Gusagusalu") in tuples

    def test_boss_discusses_climax(self):
        result = self.ext._fallback_extract("Boss is thinking about the climax", None)
        tuples = result.to_tuples()
        assert ("Boss", "discusses", "climax") in tuples

    def test_boss_discusses_multiple_keywords(self):
        result = self.ext._fallback_extract(
            "Boss wants more emotion in the dialogue", None
        )
        # Should detect both "emotion" and "dialogue"
        relations = result.by_relation("discusses")
        objects = {t.object for t in relations}
        assert "emotion" in objects
        assert "dialogue" in objects

    def test_boss_with_scene_keyword(self):
        result = self.ext._fallback_extract("Boss reviewed the scene flow", None)
        tuples = result.to_tuples()
        assert ("Boss", "discusses", "scene") in tuples

    def test_boss_with_confrontation(self):
        result = self.ext._fallback_extract("Boss wrote the confrontation", None)
        tuples = result.to_tuples()
        assert ("Boss", "discusses", "confrontation") in tuples

    def test_scene_pattern_with_character(self):
        result = self.ext._fallback_extract("Scene 5 has Ravi entering", None)
        tuples = result.to_tuples()
        assert ("Ravi", "character_in", "Scene 5") in tuples

    def test_scene_pattern_with_multiple_characters(self):
        result = self.ext._fallback_extract("In Scene 3 Ravi meets Father", None)
        tuples = result.to_tuples()
        assert ("Ravi", "character_in", "Scene 3") in tuples
        assert ("Father", "character_in", "Scene 3") in tuples

    def test_scene_number_regex_multiple(self):
        result = self.ext._fallback_extract("Ravi appears in scene 2 and scene 7", None)
        tuples = result.to_tuples()
        assert ("Ravi", "character_in", "Scene 2") in tuples
        assert ("Ravi", "character_in", "Scene 7") in tuples

    def test_project_context_adds_associations(self):
        # Character in a scene + project context -> adds project association
        result = self.ext._fallback_extract("Scene 1 has Ravi", "gusagusalu")
        # Original triplet: Ravi character_in Scene 1
        # Project association: Ravi character_in Gusagusalu (from project context)
        tuples = result.to_tuples()
        assert ("Ravi", "character_in", "Scene 1") in tuples
        # The project context association should be added
        found_project_assoc = any(
            t.object == "Gusagusalu" and t.object_type == "project"
            for t in result.triplets
        )
        assert found_project_assoc

    def test_no_matches_returns_empty(self):
        result = self.ext._fallback_extract("Nothing relevant here at all", None)
        assert result.count == 0
        assert result.source_text == "Nothing relevant here at all"

    def test_case_insensitive_character(self):
        result = self.ext._fallback_extract("RAVI in GUSAGUSALU", None)
        tuples = result.to_tuples()
        assert ("Ravi", "character_in", "Gusagusalu") in tuples

    def test_case_insensitive_boss(self):
        result = self.ext._fallback_extract("BOSS discussed the CLIMAX", None)
        tuples = result.to_tuples()
        assert ("Boss", "discusses", "climax") in tuples

    def test_kitchen_project_detected(self):
        result = self.ext._fallback_extract("Ravi in the kitchen drama", None)
        tuples = result.to_tuples()
        assert ("Ravi", "character_in", "Kitchen") in tuples

    def test_amma_character(self):
        result = self.ext._fallback_extract("Amma is in gusagusalu", None)
        tuples = result.to_tuples()
        assert ("Amma", "character_in", "Gusagusalu") in tuples

    def test_nanna_character(self):
        result = self.ext._fallback_extract("Nanna in gusagusalu scene", None)
        tuples = result.to_tuples()
        assert ("Nanna", "character_in", "Gusagusalu") in tuples

    def test_mother_character(self):
        result = self.ext._fallback_extract("Mother appears in kitchen drama", None)
        tuples = result.to_tuples()
        assert ("Mother", "character_in", "Kitchen") in tuples

    def test_fallback_confidence_character_project(self):
        result = self.ext._fallback_extract("Ravi in Gusagusalu", None)
        char_triplets = [
            t
            for t in result.triplets
            if t.subject == "Ravi"
            and t.relation == "character_in"
            and t.object == "Gusagusalu"
        ]
        assert len(char_triplets) == 1
        assert char_triplets[0].confidence == 0.6

    def test_fallback_confidence_boss_discusses(self):
        result = self.ext._fallback_extract("Boss discussed the climax", None)
        boss_triplets = [
            t
            for t in result.triplets
            if t.subject == "Boss" and t.relation == "discusses"
        ]
        assert len(boss_triplets) >= 1
        assert boss_triplets[0].confidence == 0.5

    def test_fallback_confidence_scene_character(self):
        result = self.ext._fallback_extract("Scene 1 has Ravi", None)
        scene_triplets = [
            t
            for t in result.triplets
            if t.object == "Scene 1" and t.object_type == "scene"
        ]
        assert len(scene_triplets) >= 1
        assert scene_triplets[0].confidence == 0.5

    def test_source_text_preserved(self):
        text = "Ravi in Gusagusalu talking to Father"
        result = self.ext._fallback_extract(text, None)
        assert result.source_text == text
        for t in result.triplets:
            assert t.source_text == text

    def test_project_context_no_duplicate_project_relation(self):
        # When character-project pair is already detected from text,
        # project context should not re-add it since the object already
        # matches a known project
        result = self.ext._fallback_extract("Ravi in Gusagusalu", "gusagusalu")
        # The character_in Gusagusalu triplet is from the text itself,
        # the project context loop checks `not any(p in t.object.lower() for p in projects)`
        # Since t.object == "Gusagusalu" and "gusagusalu" is in projects, it should NOT add another
        project_assocs = [
            t
            for t in result.triplets
            if t.object == "Gusagusalu" and t.confidence == 0.4
        ]
        assert len(project_assocs) == 0

    def test_project_context_only_when_triplets_exist(self):
        # If no triplets are found, project context should not add anything
        result = self.ext._fallback_extract("Nothing matches here", "gusagusalu")
        assert result.count == 0


# =========================================================================
# TripletExtractor - Parse Response
# =========================================================================


class TestParseResponse:
    """Test _parse_response with various API responses"""

    def setup_method(self):
        self.ext = TripletExtractor(api_key="test-key", min_confidence=0.6)

    def test_valid_json_response(self):
        data = _make_glm_response(
            [
                {
                    "subject": "Boss",
                    "subject_type": "person",
                    "relation": "discusses",
                    "object": "climax",
                    "object_type": "concept",
                    "confidence": 0.9,
                },
            ]
        )
        result = self.ext._parse_response(data, "Boss discussed the climax")
        assert result.count == 1
        assert result.triplets[0].subject == "Boss"
        assert result.triplets[0].relation == "discusses"
        assert result.triplets[0].object == "climax"
        assert result.triplets[0].confidence == 0.9

    def test_markdown_json_code_block(self):
        content = '```json\n{"triplets": [{"subject": "Ravi", "subject_type": "character", "relation": "character_in", "object": "Scene 1", "object_type": "scene", "confidence": 0.85}]}\n```'
        data = {
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 50},
        }
        result = self.ext._parse_response(data, "Ravi in scene 1")
        assert result.count == 1
        assert result.triplets[0].subject == "Ravi"

    def test_generic_code_block(self):
        content = '```\n{"triplets": [{"subject": "A", "subject_type": "concept", "relation": "relates_to", "object": "B", "object_type": "concept", "confidence": 0.8}]}\n```'
        data = {
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 30},
        }
        result = self.ext._parse_response(data, "A relates to B")
        assert result.count == 1
        assert result.triplets[0].subject == "A"

    def test_confidence_filtering(self):
        data = _make_glm_response(
            [
                {
                    "subject": "A",
                    "subject_type": "concept",
                    "relation": "relates_to",
                    "object": "B",
                    "object_type": "concept",
                    "confidence": 0.9,
                },
                {
                    "subject": "C",
                    "subject_type": "concept",
                    "relation": "relates_to",
                    "object": "D",
                    "object_type": "concept",
                    "confidence": 0.3,
                },
            ]
        )
        result = self.ext._parse_response(data, "test text")
        # min_confidence is 0.6, so only the 0.9 triplet passes
        assert result.count == 1
        assert result.triplets[0].subject == "A"

    def test_malformed_json_falls_back(self):
        data = {
            "choices": [{"message": {"content": "not valid json at all"}}],
            "usage": {},
        }
        result = self.ext._parse_response(data, "Boss discussed the climax")
        # Should fall back to rule-based extraction
        assert result.model_used == "fallback"

    def test_missing_fields_use_defaults(self):
        data = _make_glm_response(
            [
                {"subject": "Boss", "object": "scene"},
            ]
        )
        result = self.ext._parse_response(data, "Boss and scene")
        assert result.count == 1
        t = result.triplets[0]
        assert t.subject == "Boss"
        assert t.subject_type == "concept"  # default
        assert t.relation == "relates_to"  # default
        assert t.object == "scene"
        assert t.object_type == "concept"  # default
        assert t.confidence == 0.8  # default

    def test_token_usage_extracted(self):
        data = _make_glm_response(
            [
                {
                    "subject": "X",
                    "subject_type": "concept",
                    "relation": "relates_to",
                    "object": "Y",
                    "object_type": "concept",
                    "confidence": 0.8,
                }
            ],
            total_tokens=250,
        )
        result = self.ext._parse_response(data, "test")
        assert result.tokens_used == 250

    def test_token_usage_missing_defaults_zero(self):
        data = {
            "choices": [{"message": {"content": '{"triplets": []}'}}],
        }
        result = self.ext._parse_response(data, "test")
        assert result.tokens_used == 0

    def test_model_used_set(self):
        data = _make_glm_response(
            [
                {
                    "subject": "A",
                    "subject_type": "concept",
                    "relation": "relates_to",
                    "object": "B",
                    "object_type": "concept",
                    "confidence": 0.8,
                },
            ]
        )
        result = self.ext._parse_response(data, "test")
        assert result.model_used == "glm-4.7-flash"

    def test_source_text_attached_to_triplets(self):
        data = _make_glm_response(
            [
                {
                    "subject": "Boss",
                    "subject_type": "person",
                    "relation": "discusses",
                    "object": "climax",
                    "object_type": "concept",
                    "confidence": 0.9,
                },
            ]
        )
        source = "Boss discussed the climax"
        result = self.ext._parse_response(data, source)
        assert result.source_text == source
        assert result.triplets[0].source_text == source

    def test_empty_choices_raises_index_error(self):
        data = {"choices": [], "usage": {}}
        # data.get("choices", [{}]) returns [] (the actual empty list),
        # then [0] raises IndexError which is NOT caught by the
        # except (json.JSONDecodeError, KeyError) clause
        with pytest.raises(IndexError):
            self.ext._parse_response(data, "Boss climax")

    def test_missing_choices_key_uses_default(self):
        # When "choices" key is absent, default [{}] is used -> [{}][0] = {}
        # content becomes "{}" -> parsed -> no triplets -> empty result
        data = {"usage": {"total_tokens": 10}}
        result = self.ext._parse_response(data, "test")
        assert result.count == 0
        assert result.tokens_used == 10

    def test_multiple_triplets_parsed(self):
        data = _make_glm_response(
            [
                {
                    "subject": "Boss",
                    "subject_type": "person",
                    "relation": "discusses",
                    "object": "climax",
                    "object_type": "concept",
                    "confidence": 0.9,
                },
                {
                    "subject": "Ravi",
                    "subject_type": "character",
                    "relation": "character_in",
                    "object": "Scene 5",
                    "object_type": "scene",
                    "confidence": 0.85,
                },
                {
                    "subject": "Father",
                    "subject_type": "character",
                    "relation": "involves",
                    "object": "confrontation",
                    "object_type": "concept",
                    "confidence": 0.7,
                },
            ]
        )
        result = self.ext._parse_response(data, "complex scene")
        assert result.count == 3


# =========================================================================
# TripletExtractor - Extract (async)
# =========================================================================


class TestExtract:
    """Test the main extract() method"""

    def test_empty_text_returns_empty(self):
        ext = TripletExtractor(api_key="test-key")
        result = _run(ext.extract(""))
        assert result.count == 0
        assert result.source_text == ""

    def test_whitespace_only_returns_empty(self):
        ext = TripletExtractor(api_key="test-key")
        result = _run(ext.extract("   \n\t  "))
        assert result.count == 0

    def test_not_configured_uses_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("ZHIPU_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                ext = TripletExtractor()
                assert ext.is_configured is False
                result = _run(ext.extract("Ravi in Gusagusalu"))
                assert result.model_used == "fallback"
                assert result.count >= 1

    def test_api_error_falls_back(self):
        ext = TripletExtractor(api_key="test-key")
        with patch.object(ext, "_call_glm", new_callable=AsyncMock) as mock_glm:
            mock_glm.side_effect = Exception("API connection error")
            result = _run(ext.extract("Boss discussed the climax"))
            assert result.model_used == "fallback"
            # Should still extract via fallback rules
            tuples = result.to_tuples()
            assert ("Boss", "discusses", "climax") in tuples

    def test_successful_api_call(self):
        ext = TripletExtractor(api_key="test-key")
        mock_result = ExtractionResult(
            triplets=[_make_triplet()],
            source_text="test",
            model_used="glm-4.7-flash",
            tokens_used=100,
        )
        with patch.object(ext, "_call_glm", new_callable=AsyncMock) as mock_glm:
            mock_glm.return_value = mock_result
            result = _run(ext.extract("test text"))
            assert result.model_used == "glm-4.7-flash"
            assert result.count == 1
            mock_glm.assert_called_once_with("test text", None, None)

    def test_extract_passes_context_and_project(self):
        ext = TripletExtractor(api_key="test-key")
        mock_result = ExtractionResult(
            triplets=[],
            source_text="test",
            model_used="glm-4.7-flash",
        )
        with patch.object(ext, "_call_glm", new_callable=AsyncMock) as mock_glm:
            mock_glm.return_value = mock_result
            _run(ext.extract("text", context="some context", project="gusagusalu"))
            mock_glm.assert_called_once_with("text", "some context", "gusagusalu")


# =========================================================================
# TripletExtractor - _call_glm (async)
# =========================================================================


class TestCallGlm:
    """Test _call_glm HTTP call construction"""

    def test_call_glm_sends_correct_payload(self):
        ext = TripletExtractor(api_key="test-key-abc", model_name="glm-4.7-flash")

        mock_response = MagicMock()
        mock_response.json.return_value = _make_glm_response(
            [
                {
                    "subject": "Boss",
                    "subject_type": "person",
                    "relation": "discusses",
                    "object": "climax",
                    "object_type": "concept",
                    "confidence": 0.9,
                },
            ]
        )
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        ext._client = mock_client

        result = _run(ext._call_glm("Boss discussed the climax", None, None))
        assert result.count == 1

        # Verify the call was made
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        url = call_args[0][0]
        assert url == "https://api.z.ai/api/paas/v4/chat/completions"
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key-abc"
        payload = call_args[1]["json"]
        assert payload["model"] == "glm-4.7-flash"
        assert payload["temperature"] == 0.3
        assert payload["max_tokens"] == 512

    def test_call_glm_includes_project_in_prompt(self):
        ext = TripletExtractor(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = _make_glm_response([])
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        ext._client = mock_client

        _run(ext._call_glm("some text", None, "gusagusalu"))

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        user_msg = payload["messages"][1]["content"]
        assert "gusagusalu" in user_msg

    def test_call_glm_includes_context_in_prompt(self):
        ext = TripletExtractor(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = _make_glm_response([])
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        ext._client = mock_client

        _run(ext._call_glm("some text", "conversation context here", None))

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        user_msg = payload["messages"][1]["content"]
        assert "conversation context here" in user_msg

    def test_call_glm_truncates_long_context(self):
        ext = TripletExtractor(api_key="test-key")

        mock_response = MagicMock()
        mock_response.json.return_value = _make_glm_response([])
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        ext._client = mock_client

        long_context = "x" * 1000
        _run(ext._call_glm("text", long_context, None))

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        user_msg = payload["messages"][1]["content"]
        # Context should be truncated to 300 chars
        assert len(user_msg) < len(long_context)

    def test_call_glm_raises_on_http_error(self):
        ext = TripletExtractor(api_key="test-key")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(),
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        ext._client = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            _run(ext._call_glm("test", None, None))


# =========================================================================
# TripletExtractor - Extract Batch (async)
# =========================================================================


class TestExtractBatch:
    """Test extract_batch"""

    def test_batch_multiple_texts(self):
        ext = TripletExtractor(api_key="")  # Use fallback
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("ZHIPU_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                ext2 = TripletExtractor()
                texts = [
                    "Ravi in Gusagusalu",
                    "Boss discussed the climax",
                    "Nothing relevant here",
                ]
                results = _run(ext2.extract_batch(texts))
                assert len(results) == 3
                # First should have ravi-gusagusalu
                assert results[0].count >= 1
                # Second should have boss-climax
                assert results[1].count >= 1
                # Third should be empty
                assert results[2].count == 0

    def test_batch_empty_list(self):
        ext = TripletExtractor(api_key="test-key")
        results = _run(ext.extract_batch([]))
        assert results == []

    def test_batch_with_project(self):
        ext = TripletExtractor(api_key="")
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("ZHIPU_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                ext2 = TripletExtractor()
                texts = ["Scene 1 has Ravi"]
                results = _run(ext2.extract_batch(texts, project="gusagusalu"))
                assert len(results) == 1
                # Project context should be applied
                assert results[0].count >= 1

    def test_batch_calls_extract_for_each(self):
        ext = TripletExtractor(api_key="test-key")
        call_count = 0

        async def mock_extract(text, context=None, project=None):
            nonlocal call_count
            call_count += 1
            return ExtractionResult(source_text=text)

        with patch.object(ext, "extract", side_effect=mock_extract):
            results = _run(ext.extract_batch(["a", "b", "c"]))
            assert call_count == 3
            assert len(results) == 3


# =========================================================================
# TripletExtractor - Close
# =========================================================================


class TestClose:
    """Test close() method"""

    def test_close_with_client(self):
        ext = TripletExtractor(api_key="test-key")
        mock_client = AsyncMock()
        ext._client = mock_client
        _run(ext.close())
        mock_client.aclose.assert_called_once()
        assert ext._client is None

    def test_close_without_client(self):
        ext = TripletExtractor(api_key="test-key")
        assert ext._client is None
        # Should not raise
        _run(ext.close())
        assert ext._client is None

    def test_close_sets_client_to_none(self):
        ext = TripletExtractor(api_key="test-key")
        mock_client = AsyncMock()
        ext._client = mock_client
        _run(ext.close())
        assert ext._client is None


# =========================================================================
# TripletExtractor - _get_client
# =========================================================================


class TestGetClient:
    """Test lazy client creation"""

    def test_get_client_creates_client(self):
        ext = TripletExtractor(api_key="test-key", timeout=15.0)
        assert ext._client is None
        client = _run(ext._get_client())
        assert client is not None
        assert ext._client is not None
        # Cleanup
        _run(ext.close())

    def test_get_client_reuses_existing(self):
        ext = TripletExtractor(api_key="test-key")
        client1 = _run(ext._get_client())
        client2 = _run(ext._get_client())
        assert client1 is client2
        # Cleanup
        _run(ext.close())


# =========================================================================
# Convenience Function - extract_triplets
# =========================================================================


class TestConvenienceFunction:
    """Test the extract_triplets() convenience function"""

    def test_basic_extraction(self):
        # No API key -> fallback extraction
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("ZHIPU_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                tuples = _run(extract_triplets("Ravi in Gusagusalu"))
                assert ("Ravi", "character_in", "Gusagusalu") in tuples

    def test_returns_list_of_tuples(self):
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("ZHIPU_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                tuples = _run(extract_triplets("Boss discussed the climax"))
                assert isinstance(tuples, list)
                for t in tuples:
                    assert isinstance(t, tuple)
                    assert len(t) == 3

    def test_with_project_param(self):
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("ZHIPU_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                tuples = _run(
                    extract_triplets("Scene 1 has Ravi", project="gusagusalu")
                )
                assert len(tuples) >= 1

    def test_with_api_key_param(self):
        # Provide an API key, but mock the actual HTTP call
        with patch.object(
            TripletExtractor, "_call_glm", new_callable=AsyncMock
        ) as mock_glm:
            mock_glm.return_value = ExtractionResult(
                triplets=[
                    _make_triplet(subject="Boss", relation="wants", object="emotion")
                ],
                source_text="test",
                model_used="glm-4.7-flash",
            )
            tuples = _run(
                extract_triplets("Boss wants more emotion", api_key="key-123")
            )
            assert ("Boss", "wants", "emotion") in tuples

    def test_empty_text(self):
        tuples = _run(extract_triplets(""))
        assert tuples == []

    def test_closes_extractor_on_success(self):
        with patch.object(
            TripletExtractor, "close", new_callable=AsyncMock
        ) as mock_close:
            with patch.dict(os.environ, {}, clear=True):
                env = os.environ.copy()
                env.pop("ZHIPU_API_KEY", None)
                with patch.dict(os.environ, env, clear=True):
                    _run(extract_triplets("test"))
                    mock_close.assert_called_once()

    def test_closes_extractor_on_error(self):
        with patch.object(
            TripletExtractor, "extract", new_callable=AsyncMock
        ) as mock_extract:
            mock_extract.side_effect = RuntimeError("unexpected error")
            with patch.object(
                TripletExtractor, "close", new_callable=AsyncMock
            ) as mock_close:
                with pytest.raises(RuntimeError):
                    _run(extract_triplets("test"))
                # close() should still be called via finally
                mock_close.assert_called_once()


# =========================================================================
# Integration-style: Fallback edge cases
# =========================================================================


class TestFallbackEdgeCases:
    """Additional edge cases for fallback extraction"""

    def setup_method(self):
        self.ext = TripletExtractor(api_key="")

    def test_character_without_project_no_character_in(self):
        # Character name alone without project name -> should NOT produce character_in project
        result = self.ext._fallback_extract("Ravi entered the room", None)
        char_proj = [
            t
            for t in result.triplets
            if t.relation == "character_in" and t.object_type == "project"
        ]
        assert len(char_proj) == 0

    def test_project_without_character(self):
        # Project name alone without character -> no character_in
        result = self.ext._fallback_extract("Gusagusalu is a great project", None)
        assert result.count == 0

    def test_scene_without_character(self):
        # Scene pattern without character -> no triplets
        result = self.ext._fallback_extract("Scene 5 is really long", None)
        assert result.count == 0

    def test_combined_boss_and_character_and_scene(self):
        text = "Boss wants Ravi's scene 3 climax to be more emotional"
        result = self.ext._fallback_extract(text, None)
        tuples = result.to_tuples()
        # Boss discusses climax
        assert ("Boss", "discusses", "climax") in tuples
        # Boss discusses scene
        assert ("Boss", "discusses", "scene") in tuples
        # Boss discusses emotion
        assert ("Boss", "discusses", "emotion") in tuples
        # Ravi character_in Scene 3
        assert ("Ravi", "character_in", "Scene 3") in tuples

    def test_unicode_text_does_not_crash(self):
        result = self.ext._fallback_extract(
            "రవి is in the scene, gusagusalu project", None
        )
        # "ravi" won't match because the text has Telugu "రవి"
        # but "gusagusalu" is present. No character matched -> no triplets
        # unless "scene" pattern matches something else
        # This test just ensures no crash
        assert isinstance(result, ExtractionResult)

    def test_mixed_telugu_english(self):
        result = self.ext._fallback_extract(
            "Boss cheppindi, climax lo ravi undali", None
        )
        tuples = result.to_tuples()
        # boss + climax -> discusses
        assert ("Boss", "discusses", "climax") in tuples
        # No project matched, but ravi is present without project

    def test_scene_number_zero(self):
        result = self.ext._fallback_extract("Scene 0 with Ravi", None)
        tuples = result.to_tuples()
        assert ("Ravi", "character_in", "Scene 0") in tuples

    def test_scene_large_number(self):
        result = self.ext._fallback_extract("Scene 999 includes Ravi", None)
        tuples = result.to_tuples()
        assert ("Ravi", "character_in", "Scene 999") in tuples


# Need httpx import for the HTTPStatusError test
import httpx
