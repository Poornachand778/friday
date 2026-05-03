"""
MCP Scene Manager Service for Friday AI
========================================

Provides screenplay scene operations for FastAPI and MCP servers.
Uses the new structured screenplay schema with proper scene elements.
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional, Any, Dict

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select

from db import get_engine
from db.screenplay_schema import (
    ScreenplayProject,
    ScreenplayScene,
    SceneElement,
    DialogueLine,
    SceneEmbedding,
    SceneRelation,
    SceneRevision,
)

LOGGER = logging.getLogger(__name__)

MODEL_NAME = os.getenv(
    "SCENE_EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)

_engine = None
_model: Any = None


def get_engine_instance():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for scene embedding operations."
            ) from exc
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def fetch_scene(session: Session, scene_id: int) -> ScreenplayScene:
    """Fetch a scene by ID"""
    scene = session.get(ScreenplayScene, scene_id)
    if scene is None:
        raise ValueError(f"Scene not found: {scene_id}")
    return scene


def fetch_scene_by_number(
    session: Session, project_slug: str, scene_number: int
) -> ScreenplayScene:
    """Fetch a scene by project slug and scene number"""
    project = session.query(ScreenplayProject).filter_by(slug=project_slug).first()
    if not project:
        raise ValueError(f"Project not found: {project_slug}")

    scene = (
        session.query(ScreenplayScene)
        .filter_by(project_id=project.id, scene_number=scene_number)
        .first()
    )
    if not scene:
        raise ValueError(f"Scene {scene_number} not found in {project_slug}")
    return scene


def get_scene_text(scene: ScreenplayScene) -> str:
    """Get full text content of a scene for search/embedding"""
    parts = []

    # Scene heading
    heading = f"{scene.int_ext}. {scene.location}"
    if scene.sub_location:
        heading += f" - {scene.sub_location}"
    if scene.time_of_day:
        heading += f" - {scene.time_of_day}"
    parts.append(heading)

    # Scene title/summary
    if scene.title:
        parts.append(scene.title)
    if scene.summary:
        parts.append(scene.summary)

    # Elements
    for element in scene.elements:
        if element.element_type == "action":
            parts.append(element.content.get("text", ""))
        elif element.element_type == "dialogue":
            char_name = element.content.get("character", "")
            parts.append(f"{char_name}:")
            for line in element.content.get("lines", []):
                parts.append(line.get("text", ""))
                if line.get("translation"):
                    parts.append(f"({line.get('translation')})")
        elif element.element_type == "transition":
            parts.append(element.content.get("text", ""))

    return "\n".join(parts)


def search_scenes(
    query: str,
    project_slug: Optional[str] = None,
    top_k: int = 5,
) -> List[dict]:
    """Search scenes by semantic similarity or substring matching"""
    engine = get_engine_instance()

    try:
        model = get_model()
    except RuntimeError as exc:
        LOGGER.warning(
            "Embedding model unavailable; falling back to substring search: %s", exc
        )
        model = None

    with Session(engine) as session:
        # Build query
        stmt = select(ScreenplayScene)
        if project_slug:
            stmt = stmt.join(ScreenplayScene.project).filter(
                ScreenplayProject.slug == project_slug
            )
        scenes = session.execute(stmt).scalars().all()

        if model is None:
            # Fallback to substring/fuzzy matching
            from difflib import SequenceMatcher

            scored = []
            for scene in scenes:
                scene_text = get_scene_text(scene)
                ratio = SequenceMatcher(None, query.lower(), scene_text.lower()).ratio()
                scored.append((ratio, scene, scene_text))

            scored.sort(key=lambda x: x[0], reverse=True)

            return [
                {
                    "scene_id": scene.id,
                    "scene_number": scene.scene_number,
                    "heading": f"{scene.int_ext}. {scene.location}",
                    "title": scene.title,
                    "narrative_order": scene.narrative_order,
                    "status": scene.status,
                    "score": float(ratio),
                    "preview": (
                        scene_text[:200] + "..."
                        if len(scene_text) > 200
                        else scene_text
                    ),
                }
                for ratio, scene, scene_text in scored[:top_k]
            ]

        # Vector search with embeddings
        query_vec = model.encode(query, normalize_embeddings=True)

        # Get all embeddings
        stmt = select(SceneEmbedding, ScreenplayScene).join(
            ScreenplayScene, ScreenplayScene.id == SceneEmbedding.scene_id
        )
        if project_slug:
            stmt = stmt.join(ScreenplayScene.project).filter(
                ScreenplayProject.slug == project_slug
            )
        rows = session.execute(stmt).all()

        if not rows:
            # No embeddings yet, fall back to fuzzy search
            LOGGER.warning("No embeddings found; falling back to fuzzy search")
            from difflib import SequenceMatcher

            scored = []
            for scene in scenes:
                scene_text = get_scene_text(scene)
                ratio = SequenceMatcher(None, query.lower(), scene_text.lower()).ratio()
                scored.append((ratio, scene, scene_text))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {
                    "scene_id": scene.id,
                    "scene_number": scene.scene_number,
                    "heading": f"{scene.int_ext}. {scene.location}",
                    "title": scene.title,
                    "narrative_order": scene.narrative_order,
                    "status": scene.status,
                    "score": float(ratio),
                    "preview": (
                        scene_text[:200] + "..."
                        if len(scene_text) > 200
                        else scene_text
                    ),
                }
                for ratio, scene, scene_text in scored[:top_k]
            ]

        # Calculate similarity scores
        results = []
        for emb, scene in rows:
            vec = np.array(emb.vector)
            denom = np.linalg.norm(query_vec) * np.linalg.norm(vec)
            score = float(np.dot(query_vec, vec) / denom) if denom else 0.0
            results.append((score, scene))

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:top_k]

        output = []
        for score, scene in top:
            scene_text = get_scene_text(scene)
            output.append(
                {
                    "scene_id": scene.id,
                    "scene_number": scene.scene_number,
                    "heading": f"{scene.int_ext}. {scene.location}",
                    "title": scene.title,
                    "narrative_order": scene.narrative_order,
                    "status": scene.status,
                    "score": score,
                    "preview": (
                        scene_text[:200] + "..."
                        if len(scene_text) > 200
                        else scene_text
                    ),
                }
            )
        return output


def get_scene_detail(
    scene_id: Optional[int] = None,
    scene_number: Optional[int] = None,
    project_slug: Optional[str] = None,
) -> dict:
    """Get full scene details including all elements"""
    engine = get_engine_instance()

    with Session(engine) as session:
        if scene_id:
            scene = fetch_scene(session, scene_id)
        elif scene_number and project_slug:
            scene = fetch_scene_by_number(session, project_slug, scene_number)
        else:
            raise ValueError("Must provide scene_id or (scene_number and project_slug)")

        # Get related scenes
        relations = (
            session.query(SceneRelation)
            .filter(SceneRelation.from_scene_id == scene.id)
            .all()
        )

        # Build elements list
        elements = []
        for elem in scene.elements:
            elem_data = {
                "type": elem.element_type,
                "order": elem.order_index,
                "content": elem.content,
            }
            elements.append(elem_data)

        return {
            "scene_id": scene.id,
            "scene_number": scene.scene_number,
            "int_ext": scene.int_ext,
            "location": scene.location,
            "sub_location": scene.sub_location,
            "time_of_day": scene.time_of_day,
            "title": scene.title,
            "summary": scene.summary,
            "narrative_order": scene.narrative_order,
            "status": scene.status,
            "tags": scene.tags,
            "elements": elements,
            "related_scenes": [
                {"scene_id": r.to_scene_id, "relation": r.relation_type}
                for r in relations
            ],
            "project": {
                "id": scene.project.id,
                "title": scene.project.title,
                "slug": scene.project.slug,
            },
        }


def update_scene(
    scene_id: int,
    status: Optional[str] = None,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    tags: Optional[List[str]] = None,
    narrative_order: Optional[float] = None,
) -> bool:
    """Update scene metadata"""
    engine = get_engine_instance()
    changed = False

    with Session(engine) as session:
        scene = fetch_scene(session, scene_id)

        if status is not None:
            scene.status = status
            changed = True
        if title is not None:
            scene.title = title
            changed = True
        if summary is not None:
            scene.summary = summary
            changed = True
        if tags is not None:
            scene.tags = tags
            changed = True
        if narrative_order is not None:
            scene.narrative_order = narrative_order
            changed = True

        if changed:
            session.add(scene)
            session.commit()
        else:
            session.rollback()

    return changed


def add_scene_element(
    scene_id: int,
    element_type: str,
    content: Dict,
    order_index: Optional[int] = None,
) -> int:
    """Add a new element to a scene"""
    engine = get_engine_instance()

    with Session(engine) as session:
        scene = fetch_scene(session, scene_id)

        # Auto-calculate order if not provided
        if order_index is None:
            max_order = max([e.order_index for e in scene.elements], default=-1)
            order_index = max_order + 1

        element = SceneElement(
            scene_id=scene.id,
            element_type=element_type,
            order_index=order_index,
            content=content,
        )
        session.add(element)
        session.commit()

        return element.id


def create_relation(
    from_scene_id: int,
    to_scene_id: int,
    relation_type: str = "sequence",
    notes: Optional[str] = None,
) -> bool:
    """Create a relationship between two scenes"""
    engine = get_engine_instance()

    with Session(engine) as session:
        source = fetch_scene(session, from_scene_id)
        target = fetch_scene(session, to_scene_id)

        relation = SceneRelation(
            project_id=source.project_id,
            from_scene_id=source.id,
            to_scene_id=target.id,
            relation_type=relation_type,
            notes=notes,
        )
        session.add(relation)
        session.commit()

    return True


def generate_scene_embedding(scene_id: int) -> bool:
    """Generate and store embedding for a scene"""
    engine = get_engine_instance()
    model = get_model()

    with Session(engine) as session:
        scene = fetch_scene(session, scene_id)
        scene_text = get_scene_text(scene)

        # Generate hash for content
        import hashlib

        content_hash = hashlib.sha256(scene_text.encode()).hexdigest()

        # Check if embedding already exists
        existing = (
            session.query(SceneEmbedding)
            .filter_by(scene_id=scene.id, content_hash=content_hash)
            .first()
        )

        if existing:
            LOGGER.info(f"Embedding already exists for scene {scene_id}")
            return False

        # Generate embedding
        vector = model.encode(scene_text, normalize_embeddings=True).tolist()

        embedding = SceneEmbedding(
            scene_id=scene.id,
            content_type="full",
            content_hash=content_hash,
            model_name=MODEL_NAME,
            vector=vector,
        )
        session.add(embedding)
        session.commit()

    return True


def generate_project_embeddings(project_slug: str) -> int:
    """Generate embeddings for all scenes in a project"""
    engine = get_engine_instance()
    count = 0

    with Session(engine) as session:
        project = session.query(ScreenplayProject).filter_by(slug=project_slug).first()
        if not project:
            raise ValueError(f"Project not found: {project_slug}")

        for scene in project.scenes:
            try:
                if generate_scene_embedding(scene.id):
                    count += 1
            except Exception as e:
                LOGGER.error(f"Error generating embedding for scene {scene.id}: {e}")

    return count


def list_projects() -> List[dict]:
    """List all screenplay projects"""
    engine = get_engine_instance()

    with Session(engine) as session:
        projects = session.query(ScreenplayProject).all()
        return [
            {
                "id": p.id,
                "title": p.title,
                "slug": p.slug,
                "status": p.status,
                "scene_count": len(p.scenes),
                "primary_language": p.primary_language,
            }
            for p in projects
        ]


def get_project_scenes(project_slug: str) -> List[dict]:
    """Get all scenes for a project"""
    engine = get_engine_instance()

    with Session(engine) as session:
        project = session.query(ScreenplayProject).filter_by(slug=project_slug).first()
        if not project:
            raise ValueError(f"Project not found: {project_slug}")

        return [
            {
                "scene_id": scene.id,
                "scene_number": scene.scene_number,
                "heading": f"{scene.int_ext}. {scene.location}",
                "title": scene.title,
                "status": scene.status,
                "narrative_order": scene.narrative_order,
            }
            for scene in project.scenes
        ]
