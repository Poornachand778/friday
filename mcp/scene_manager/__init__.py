"""
MCP Scene Manager - Screenplay scene operations for Friday AI.
"""

from mcp.scene_manager.service import (
    search_scenes,
    get_scene_detail,
    update_scene,
    create_relation,
    add_scene_element,
    fetch_scene,
    fetch_scene_by_number,
    get_scene_text,
    generate_scene_embedding,
    generate_project_embeddings,
    list_projects,
    get_project_scenes,
)

__all__ = [
    "search_scenes",
    "get_scene_detail",
    "update_scene",
    "create_relation",
    "add_scene_element",
    "fetch_scene",
    "fetch_scene_by_number",
    "get_scene_text",
    "generate_scene_embedding",
    "generate_project_embeddings",
    "list_projects",
    "get_project_scenes",
]
