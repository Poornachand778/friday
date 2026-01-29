"""
Context Detector for Friday AI
===============================

Detects the appropriate context based on:
- Explicit user request
- Message content keywords
- Device/location information
- Conversation history
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from .contexts import Context, ContextType, CONTEXTS, get_context


LOGGER = logging.getLogger(__name__)


class ContextDetector:
    """
    Detects the appropriate context for a conversation.

    Detection priority:
    1. Explicit context switch command
    2. Device/location hint
    3. Keyword analysis
    4. Previous context (sticky)
    5. Default context
    """

    # Patterns for explicit context switching
    SWITCH_PATTERNS = [
        (r"switch to (\w+)", 1),
        (r"go to (\w+)", 1),
        (r"let'?s go to (\w+)", 1),
        (r"(\w+) mode", 1),
        (r"in the (\w+)", 1),
    ]

    def __init__(
        self,
        default_context: ContextType = ContextType.WRITERS_ROOM,
        sticky: bool = True,
    ):
        self.default_context = default_context
        self.sticky = sticky  # Keep context until explicitly changed
        self._current_context: Optional[ContextType] = None

    def detect(
        self,
        message: str,
        device_id: Optional[str] = None,
        location: Optional[str] = None,
        conversation_history: Optional[List[dict]] = None,
    ) -> Tuple[Context, float]:
        """
        Detect the appropriate context.

        Args:
            message: Current user message
            device_id: Device identifier (can hint at location)
            location: Explicit location hint
            conversation_history: Previous messages

        Returns:
            Tuple of (Context, confidence_score)
        """
        # 1. Check for explicit switch command
        explicit_ctx = self._detect_explicit_switch(message)
        if explicit_ctx:
            self._current_context = explicit_ctx.context_type
            LOGGER.info("Explicit context switch to: %s", explicit_ctx.name)
            return explicit_ctx, 1.0

        # 2. Check device/location hints
        location_ctx = self._detect_from_location(device_id, location)
        if location_ctx:
            self._current_context = location_ctx.context_type
            LOGGER.info("Location-based context: %s", location_ctx.name)
            return location_ctx, 0.9

        # 3. Keyword analysis
        keyword_ctx, confidence = self._detect_from_keywords(message)
        if keyword_ctx and confidence > 0.5:
            # Only switch if high confidence or not sticky
            if confidence > 0.7 or not self.sticky or self._current_context is None:
                self._current_context = keyword_ctx.context_type
                LOGGER.info(
                    "Keyword-detected context: %s (%.2f)", keyword_ctx.name, confidence
                )
                return keyword_ctx, confidence

        # 4. Use sticky context if available
        if self.sticky and self._current_context:
            ctx = get_context(self._current_context)
            LOGGER.debug("Using sticky context: %s", ctx.name)
            return ctx, 0.6

        # 5. Default context
        ctx = get_context(self.default_context)
        self._current_context = self.default_context
        LOGGER.debug("Using default context: %s", ctx.name)
        return ctx, 0.5

    def _detect_explicit_switch(self, message: str) -> Optional[Context]:
        """Detect explicit context switch command"""
        message_lower = message.lower()

        for pattern, group in self.SWITCH_PATTERNS:
            match = re.search(pattern, message_lower)
            if match:
                target = match.group(group)

                # Map common names to contexts
                name_mapping = {
                    "kitchen": ContextType.KITCHEN,
                    "cook": ContextType.KITCHEN,
                    "cooking": ContextType.KITCHEN,
                    "writers": ContextType.WRITERS_ROOM,
                    "writing": ContextType.WRITERS_ROOM,
                    "script": ContextType.WRITERS_ROOM,
                    "storyboard": ContextType.STORYBOARD,
                    "visual": ContextType.STORYBOARD,
                    "general": ContextType.GENERAL,
                }

                ctx_type = name_mapping.get(target)
                if ctx_type:
                    return get_context(ctx_type)

        return None

    def _detect_from_location(
        self,
        device_id: Optional[str],
        location: Optional[str],
    ) -> Optional[Context]:
        """Detect context from device/location"""
        if not device_id and not location:
            return None

        location_str = (location or device_id or "").lower()

        location_mapping = {
            "kitchen": ContextType.KITCHEN,
            "cook": ContextType.KITCHEN,
            "writers": ContextType.WRITERS_ROOM,
            "writing": ContextType.WRITERS_ROOM,
            "office": ContextType.WRITERS_ROOM,
            "storyboard": ContextType.STORYBOARD,
            "visual": ContextType.STORYBOARD,
            "studio": ContextType.STORYBOARD,
        }

        for key, ctx_type in location_mapping.items():
            if key in location_str:
                return get_context(ctx_type)

        return None

    def _detect_from_keywords(self, message: str) -> Tuple[Optional[Context], float]:
        """Detect context from message keywords"""
        message_lower = message.lower()
        words = set(re.findall(r"\w+", message_lower))

        best_context = None
        best_score = 0.0

        for ctx_type, ctx in CONTEXTS.items():
            if not ctx.detection_keywords:
                continue

            # Count keyword matches
            matches = sum(1 for kw in ctx.detection_keywords if kw in message_lower)

            # Also check word-level matches
            word_matches = len(words & set(ctx.detection_keywords))

            # Combined score
            total_keywords = len(ctx.detection_keywords)
            if total_keywords > 0:
                score = (matches + word_matches) / (total_keywords * 0.3)  # Normalize
                score = min(score, 1.0)

                if score > best_score:
                    best_score = score
                    best_context = ctx

        return best_context, best_score

    def set_context(self, context_type: ContextType) -> Context:
        """Explicitly set the current context"""
        self._current_context = context_type
        return get_context(context_type)

    def get_current_context(self) -> Optional[Context]:
        """Get the current context"""
        if self._current_context:
            return get_context(self._current_context)
        return None

    def reset(self) -> None:
        """Reset to default context"""
        self._current_context = None
