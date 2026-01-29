#!/usr/bin/env python3
"""Generate sentence embeddings for screenplay scenes and store them in Postgres."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

from db import get_engine
from db.schema import ScriptScene, ScriptSceneEmbedding


# Lightweight multilingual sentence encoder
DEFAULT_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update scene embeddings.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Sentence transformer model to use"
    )
    parser.add_argument("--project", help="Optional project slug to filter scenes")
    parser.add_argument("--scene", help="Optional scene code to filter (e.g., SCN005)")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Delete existing embeddings for the selected scope before recomputing",
    )
    return parser


def fetch_scenes(
    session: Session, project_slug: str | None, scene_code: str | None
) -> list[ScriptScene]:
    query = session.query(ScriptScene)
    if project_slug:
        query = query.join(ScriptScene.project).filter_by(slug=project_slug)
    if scene_code:
        query = query.filter(ScriptScene.scene_code == scene_code)
    return query.order_by(ScriptScene.narrative_order, ScriptScene.id).all()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    engine = get_engine()

    with Session(engine) as session:
        scenes = fetch_scenes(session, args.project, args.scene)
        if not scenes:
            print("No scenes matched the criteria.")
            return 0

        if args.recompute:
            stmt = session.query(ScriptSceneEmbedding).filter(
                ScriptSceneEmbedding.model_name == args.model
            )
            if args.project:
                stmt = (
                    stmt.join(ScriptSceneEmbedding.scene)
                    .join(ScriptScene.project)
                    .filter_by(slug=args.project)
                )
            if args.scene:
                stmt = stmt.join(ScriptSceneEmbedding.scene).filter(
                    ScriptScene.scene_code == args.scene
                )
            deleted = stmt.delete(synchronize_session=False)
            print(f"Deleted {deleted} existing embeddings for model {args.model}.")
            session.commit()

    processed = 0
    with Session(engine) as session:
        scenes = fetch_scenes(session, args.project, args.scene)
        for scene in scenes:
            text = scene.canonical_text or scene.summary or scene.title or ""
            text = text.strip()
            if not text:
                continue
            vector = model.encode(text, normalize_embeddings=True).tolist()
            embedding = ScriptSceneEmbedding(
                scene_id=scene.id,
                revision_id=scene.current_revision_id,
                model_name=args.model,
                vector=vector,
            )
            session.add(embedding)
            processed += 1
        session.commit()

    print(f"Stored {processed} embeddings using model {args.model}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
