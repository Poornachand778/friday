"""
Triplet Extractor
=================

Extracts knowledge triplets (subject-relation-object) from text
using GLM-4.7-Flash.

Cognee-Inspired:
    - Triplet extraction enables relationship queries
    - Powers "What scenes involve Ravi?" type questions
    - Builds semantic knowledge graph from conversations

Usage:
    extractor = TripletExtractor()
    triplets = await extractor.extract("Boss discussed the climax with Ravi")
    # Returns: [("Boss", "discussed", "climax"), ("Ravi", "involved_in", "climax")]
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

LOGGER = logging.getLogger(__name__)

# Entity types for classification
ENTITY_TYPES = [
    "character",  # Film characters (Ravi, Father, etc.)
    "scene",  # Screenplay scenes
    "project",  # Project names (Gusagusalu, Kitchen)
    "person",  # Real people (Boss, collaborators)
    "concept",  # Abstract concepts (emotion, confrontation)
    "location",  # Places
    "event",  # Events, deadlines
]

# Relation types for classification
RELATION_TYPES = [
    "discusses",  # Boss discusses climax
    "contains",  # Scene contains confrontation
    "relates_to",  # Generic relationship
    "character_in",  # Ravi in Gusagusalu
    "scene_in",  # Scene in project
    "has_relationship",  # Ravi has father
    "creates",  # Boss creates scene
    "wants",  # Boss wants more emotion
    "deadline_for",  # March is deadline for Gusagusalu
    "involves",  # Confrontation involves Ravi
]

# System prompt for triplet extraction
EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extractor for Friday AI, an assistant for a Telugu screenwriter named Poorna (Boss).

Your task: Extract knowledge triplets (subject-relation-object) from conversations.

Entity types: character, scene, project, person, concept, location, event
Relation types: discusses, contains, relates_to, character_in, scene_in, has_relationship, creates, wants, deadline_for, involves

Rules:
1. Extract ONLY meaningful relationships, not every word
2. "Boss" always refers to Poorna (the user), type: person
3. Character names from screenplays → type: character
4. Project names (Gusagusalu, Kitchen drama) → type: project
5. Abstract ideas (emotion, confrontation, arc) → type: concept
6. Handle Telugu-English mixed text naturally

Output JSON format:
{
  "triplets": [
    {
      "subject": "Boss",
      "subject_type": "person",
      "relation": "discusses",
      "object": "climax scene",
      "object_type": "scene",
      "confidence": 0.9
    }
  ]
}

Examples:

Input: "Boss, let's make Ravi's confrontation with his father more emotional"
Output:
{
  "triplets": [
    {"subject": "Boss", "subject_type": "person", "relation": "wants", "object": "more emotional", "object_type": "concept", "confidence": 0.85},
    {"subject": "Ravi", "subject_type": "character", "relation": "has_relationship", "object": "father", "object_type": "character", "confidence": 0.95},
    {"subject": "confrontation", "subject_type": "scene", "relation": "involves", "object": "Ravi", "object_type": "character", "confidence": 0.9}
  ]
}

Input: "Gusagusalu deadline is March"
Output:
{
  "triplets": [
    {"subject": "March", "subject_type": "event", "relation": "deadline_for", "object": "Gusagusalu", "object_type": "project", "confidence": 0.95}
  ]
}

Input: "Scene 5 lo Ravi enters, father waiting"
Output:
{
  "triplets": [
    {"subject": "Ravi", "subject_type": "character", "relation": "character_in", "object": "Scene 5", "object_type": "scene", "confidence": 0.9},
    {"subject": "father", "subject_type": "character", "relation": "character_in", "object": "Scene 5", "object_type": "scene", "confidence": 0.9}
  ]
}"""


@dataclass
class ExtractedTriplet:
    """A single extracted triplet"""

    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    confidence: float = 0.8
    source_text: str = ""

    def as_tuple(self) -> Tuple[str, str, str]:
        """Return as simple (subject, relation, object) tuple"""
        return (self.subject, self.relation, self.object)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "subject_type": self.subject_type,
            "relation": self.relation,
            "object": self.object,
            "object_type": self.object_type,
            "confidence": self.confidence,
            "source_text": self.source_text,
        }


@dataclass
class ExtractionResult:
    """Result from triplet extraction"""

    triplets: List[ExtractedTriplet] = field(default_factory=list)
    source_text: str = ""
    model_used: str = ""
    tokens_used: int = 0

    @property
    def count(self) -> int:
        return len(self.triplets)

    def high_confidence(self, threshold: float = 0.8) -> List[ExtractedTriplet]:
        """Get triplets above confidence threshold"""
        return [t for t in self.triplets if t.confidence >= threshold]

    def by_relation(self, relation: str) -> List[ExtractedTriplet]:
        """Filter triplets by relation type"""
        return [t for t in self.triplets if t.relation == relation]

    def to_tuples(self) -> List[Tuple[str, str, str]]:
        """Convert all triplets to simple tuples"""
        return [t.as_tuple() for t in self.triplets]


class TripletExtractor:
    """
    Extracts knowledge triplets from text using GLM-4.7-Flash.

    Cognee-inspired approach:
        - Extract subject-relation-object triplets
        - Classify entities by type (character, scene, project, etc.)
        - Enable graph-based queries later

    Usage:
        extractor = TripletExtractor()
        result = await extractor.extract("Boss discussed climax with Ravi")

        for triplet in result.triplets:
            print(f"{triplet.subject} -{triplet.relation}-> {triplet.object}")

    Config:
        Set ZHIPU_API_KEY environment variable, or pass api_key to constructor.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "glm-4.7-flash",
        timeout: float = 10.0,
        min_confidence: float = 0.6,
    ):
        self.api_key = api_key or os.environ.get("ZHIPU_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "ZHIPU_BASE_URL", "https://api.z.ai/api/paas/v4"
        )
        self.model_name = model_name
        self.timeout = timeout
        self.min_confidence = min_confidence
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def extract(
        self,
        text: str,
        context: Optional[str] = None,
        project: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extract knowledge triplets from text.

        Args:
            text: Text to extract triplets from
            context: Optional conversation context
            project: Current project (helps classify entities)

        Returns:
            ExtractionResult with list of triplets
        """
        if not text.strip():
            return ExtractionResult(source_text=text)

        if not self.is_configured:
            LOGGER.debug("TripletExtractor not configured, using fallback")
            return self._fallback_extract(text, project)

        try:
            return await self._call_glm(text, context, project)
        except Exception as e:
            LOGGER.warning("Triplet extraction failed: %s", e)
            return self._fallback_extract(text, project)

    async def _call_glm(
        self,
        text: str,
        context: Optional[str],
        project: Optional[str],
    ) -> ExtractionResult:
        """Call GLM-4.7-Flash for triplet extraction"""
        client = await self._get_client()

        # Build user prompt
        user_prompt = f'Extract triplets from: "{text}"'
        if project:
            user_prompt += f"\nProject: {project}"
        if context:
            user_prompt += f"\nContext: {context[:300]}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,  # Low for consistent extraction
            "max_tokens": 512,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"

        LOGGER.debug("Calling GLM for triplet extraction")
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        return self._parse_response(data, text)

    def _parse_response(self, data: Dict, source_text: str) -> ExtractionResult:
        """Parse GLM response into ExtractionResult"""
        try:
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "{}")

            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            parsed = json.loads(content.strip())
            triplet_data = parsed.get("triplets", [])

            triplets = []
            for t in triplet_data:
                confidence = t.get("confidence", 0.8)
                if confidence >= self.min_confidence:
                    triplets.append(
                        ExtractedTriplet(
                            subject=t.get("subject", ""),
                            subject_type=t.get("subject_type", "concept"),
                            relation=t.get("relation", "relates_to"),
                            object=t.get("object", ""),
                            object_type=t.get("object_type", "concept"),
                            confidence=confidence,
                            source_text=source_text,
                        )
                    )

            # Get token usage
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            return ExtractionResult(
                triplets=triplets,
                source_text=source_text,
                model_used=self.model_name,
                tokens_used=tokens_used,
            )

        except (json.JSONDecodeError, KeyError) as e:
            LOGGER.warning("Failed to parse extraction response: %s", e)
            return self._fallback_extract(source_text, None)

    def _fallback_extract(
        self,
        text: str,
        project: Optional[str],
    ) -> ExtractionResult:
        """
        Simple rule-based extraction as fallback.

        Used when GLM is unavailable.
        """
        triplets = []
        text_lower = text.lower()

        # Known characters (expand as needed)
        characters = ["ravi", "father", "mother", "amma", "nanna"]
        # Known projects
        projects = ["gusagusalu", "kitchen"]

        # Extract character-project relationships
        for char in characters:
            if char in text_lower:
                for proj in projects:
                    if proj in text_lower:
                        triplets.append(
                            ExtractedTriplet(
                                subject=char.title(),
                                subject_type="character",
                                relation="character_in",
                                object=proj.title(),
                                object_type="project",
                                confidence=0.6,
                                source_text=text,
                            )
                        )

        # Extract "Boss discusses/wants" patterns
        if "boss" in text_lower:
            for keyword in ["climax", "scene", "emotion", "dialogue", "confrontation"]:
                if keyword in text_lower:
                    triplets.append(
                        ExtractedTriplet(
                            subject="Boss",
                            subject_type="person",
                            relation="discusses",
                            object=keyword,
                            object_type="concept",
                            confidence=0.5,
                            source_text=text,
                        )
                    )

        # Extract scene mentions
        import re

        scene_pattern = r"scene\s*(\d+)"
        scene_matches = re.findall(scene_pattern, text_lower)
        for scene_num in scene_matches:
            for char in characters:
                if char in text_lower:
                    triplets.append(
                        ExtractedTriplet(
                            subject=char.title(),
                            subject_type="character",
                            relation="character_in",
                            object=f"Scene {scene_num}",
                            object_type="scene",
                            confidence=0.5,
                            source_text=text,
                        )
                    )

        # Add project context if provided
        if project and triplets:
            for t in triplets:
                if t.object_type in ("scene", "character") and not any(
                    p in t.object.lower() for p in projects
                ):
                    # Add project association
                    triplets.append(
                        ExtractedTriplet(
                            subject=t.subject,
                            subject_type=t.subject_type,
                            relation=(
                                "scene_in"
                                if t.object_type == "scene"
                                else "character_in"
                            ),
                            object=project.title(),
                            object_type="project",
                            confidence=0.4,
                            source_text=text,
                        )
                    )

        return ExtractionResult(
            triplets=triplets,
            source_text=text,
            model_used="fallback",
            tokens_used=0,
        )

    async def extract_batch(
        self,
        texts: List[str],
        project: Optional[str] = None,
    ) -> List[ExtractionResult]:
        """
        Extract triplets from multiple texts.

        Processes in sequence (could be parallelized).
        """
        results = []
        for text in texts:
            result = await self.extract(text, project=project)
            results.append(result)
        return results

    async def close(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None


# Convenience function
async def extract_triplets(
    text: str,
    api_key: Optional[str] = None,
    project: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """
    Quick triplet extraction.

    Returns list of (subject, relation, object) tuples.

    Usage:
        triplets = await extract_triplets("Boss discussed climax")
        # [("Boss", "discusses", "climax")]
    """
    extractor = TripletExtractor(api_key=api_key)
    try:
        result = await extractor.extract(text, project=project)
        return result.to_tuples()
    finally:
        await extractor.close()
