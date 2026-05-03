"""
Context Definitions for Friday AI
==================================

Defines the different rooms/contexts Friday can operate in.
Each context has:
- System prompt additions
- Available tools
- Optional LoRA adapter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ContextType(str, Enum):
    """Available context types"""

    WRITERS_ROOM = "writers_room"
    KITCHEN = "kitchen"
    STORYBOARD = "storyboard"
    GENERAL = "general"


@dataclass
class Context:
    """A context/room configuration"""

    name: str
    context_type: ContextType
    description: str

    # System prompt additions for this context
    system_prompt_addition: str

    # Tools available in this context
    available_tools: List[str] = field(default_factory=list)

    # Optional LoRA adapter for this context
    lora_adapter: Optional[str] = None

    # External APIs used in this context
    external_apis: List[str] = field(default_factory=list)

    # Keywords that suggest this context
    detection_keywords: List[str] = field(default_factory=list)


# Define available contexts
CONTEXTS = {
    ContextType.WRITERS_ROOM: Context(
        name="Writers Room",
        context_type=ContextType.WRITERS_ROOM,
        description="Screenplay writing and brainstorming",
        system_prompt_addition=(
            "You are in the Writers Room, focused on screenplay development. "
            "Help brainstorm scenes, develop characters, refine dialogue, and manage the script structure. "
            "Use scene tools to search, view, and update screenplay content. "
            "Use document tools to reference craft books and screenwriting guides with citations. "
            "Think like a screenwriter - focus on visual storytelling, subtext, and emotional beats."
        ),
        available_tools=[
            "scene_search",
            "scene_get",
            "scene_update",
            "scene_reorder",
            "scene_link",
            "send_screenplay",
            "send_email",
            # Document tools for reference
            "document_search",
            "document_get_context",
            "document_get_chapter",
            "document_list",
            "document_get",
            "document_status",
            "document_delete",
            "document_ingest",
            # Book understanding & mentor tools
            "book_study",
            "book_study_status",
            "book_study_jobs",
            "book_list_studied",
            "book_get_understanding",
            "mentor_load_books",
            "mentor_analyze",
            "mentor_brainstorm",
            "mentor_check_rules",
            "mentor_find_inspiration",
            "mentor_ask",
            "mentor_compare",
            "knowledge_search",
        ],
        lora_adapter="friday-script",
        external_apis=[],
        detection_keywords=[
            "scene",
            "script",
            "screenplay",
            "dialogue",
            "character",
            "act",
            "beat",
            "story",
            "write",
            "draft",
            "revision",
            "neelima",
            "arjun",  # Character names from scripts
            # Document-related keywords
            "book",
            "mckee",
            "reference",
            "according to",
            "what does",
        ],
    ),
    ContextType.KITCHEN: Context(
        name="Kitchen",
        context_type=ContextType.KITCHEN,
        description="Cooking assistance with camera",
        system_prompt_addition=(
            "You are in the Kitchen, helping with cooking. "
            "Guide through recipes step by step, suggest ingredients, and provide cooking tips. "
            "When camera is active, analyze what you see and provide real-time guidance. "
            "Be precise with measurements and timing. Warn about food safety."
        ),
        available_tools=[
            "camera_analyze",
            "send_email",
            # Document tools for recipe references
            "document_search",
            "document_get_context",
            "knowledge_search",
        ],
        lora_adapter=None,  # Use base model with tools
        external_apis=["vision"],  # Claude Vision for camera
        detection_keywords=[
            "cook",
            "recipe",
            "ingredient",
            "kitchen",
            "food",
            "bake",
            "fry",
            "boil",
            "chop",
            "dice",
            "mix",
            "temperature",
            "timer",
            "oven",
            "stove",
        ],
    ),
    ContextType.STORYBOARD: Context(
        name="Storyboard Room",
        context_type=ContextType.STORYBOARD,
        description="Visual storyboarding and video generation",
        system_prompt_addition=(
            "You are in the Storyboard Room, focused on visualizing the screenplay. "
            "Help create storyboards, generate reference images, and plan shots. "
            "Think visually - describe camera angles, lighting, composition, and mood. "
            "Use video and image generation tools to bring scenes to life."
        ),
        available_tools=[
            "scene_search",
            "scene_get",
            "generate_image",
            "camera_analyze",
            "send_email",
            # Document tools for visual references
            "document_search",
            "document_get_context",
            "knowledge_search",
        ],
        lora_adapter=None,  # Use base model with visual tools
        external_apis=["vision", "video_gen", "image_gen"],
        detection_keywords=[
            "storyboard",
            "visual",
            "shot",
            "frame",
            "camera",
            "angle",
            "lighting",
            "composition",
            "generate",
            "image",
            "video",
            "visualize",
            "render",
        ],
    ),
    ContextType.GENERAL: Context(
        name="General",
        context_type=ContextType.GENERAL,
        description="General assistant mode",
        system_prompt_addition=(
            "You are in general assistant mode. "
            "Help with any task - questions, planning, research, or conversation. "
            "You can ingest documents (PDFs, books) and search across them. "
            "If a task seems related to a specific room (writing, cooking, storyboarding), "
            "mention that you can switch to that context for better tools."
        ),
        available_tools=[
            "send_email",
            # Document tools available everywhere
            "document_ingest",
            "document_search",
            "document_get_context",
            "document_get_chapter",
            "document_list",
            "document_get",
            "document_status",
            "document_delete",
            # Book knowledge search
            "knowledge_search",
            "book_list_studied",
            "book_study_jobs",
            "book_get_understanding",
            "mentor_ask",
            "mentor_compare",
        ],
        lora_adapter=None,
        external_apis=[],
        detection_keywords=[],  # Default fallback
    ),
}


def get_context(context_type: ContextType) -> Context:
    """Get context configuration by type"""
    return CONTEXTS.get(context_type, CONTEXTS[ContextType.GENERAL])


def get_context_by_name(name: str) -> Optional[Context]:
    """Get context by name string"""
    name_lower = name.lower().replace(" ", "_")

    for ctx_type, ctx in CONTEXTS.items():
        if (
            ctx_type.value == name_lower
            or ctx.name.lower().replace(" ", "_") == name_lower
        ):
            return ctx

    return None


def list_contexts() -> List[Context]:
    """List all available contexts"""
    return list(CONTEXTS.values())
