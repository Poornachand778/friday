#!/usr/bin/env python3
"""
Migration: Create Screenplay Schema Tables
==========================================

Creates the new structured screenplay tables:
- screenplay_projects
- screenplay_characters
- screenplay_scenes
- scene_elements
- dialogue_lines
- scene_embeddings
- scene_relations
- scene_revisions
- export_configs

Usage:
    python db/migrations/001_screenplay_schema.py
    python db/migrations/001_screenplay_schema.py --rollback
"""

import argparse
import os
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Database connection
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "vectordb")
DB_USER = os.getenv("DB_USER", "vectoruser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "friday")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_engine():
    return create_engine(DATABASE_URL, echo=True)


def create_tables(engine):
    """Create all new screenplay tables"""
    print("Creating screenplay schema tables...")

    # Import models to register them with SQLAlchemy
    from db.screenplay_schema import (
        Base,
        ScreenplayProject,
        ScreenplayCharacter,
        ScreenplayScene,
        SceneElement,
        DialogueLine,
        SceneEmbedding,
        SceneRelation,
        SceneRevision,
        ExportConfig,
    )

    # Create all tables
    Base.metadata.create_all(engine)
    print("Tables created successfully!")


def insert_default_config(engine):
    """Insert default export configuration"""
    with engine.connect() as conn:
        # Check if default config exists
        result = conn.execute(
            text("SELECT id FROM export_configs WHERE name = 'celtx_default'")
        )
        if result.fetchone() is None:
            conn.execute(
                text(
                    """
                INSERT INTO export_configs (
                    name, font_family, font_size,
                    page_width, page_height,
                    margin_top, margin_bottom, margin_left, margin_right,
                    scene_heading_bg_color, scene_heading_bold,
                    character_name_caps, parenthetical_italics,
                    show_translations, translation_in_parentheses,
                    created_at
                ) VALUES (
                    'celtx_default', 'Courier Prime', 12,
                    8.5, 11.0,
                    1.0, 1.0, 1.5, 1.0,
                    '#CCCCCC', true,
                    true, false,
                    true, true,
                    NOW()
                )
            """
                )
            )
            conn.commit()
            print("Default export config inserted.")


def rollback(engine):
    """Drop all screenplay tables (use with caution!)"""
    print("Rolling back screenplay schema...")

    tables = [
        "scene_revisions",
        "scene_relations",
        "scene_embeddings",
        "dialogue_lines",
        "scene_elements",
        "screenplay_scenes",
        "screenplay_characters",
        "screenplay_projects",
        "export_configs",
    ]

    with engine.connect() as conn:
        for table in tables:
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                print(f"  Dropped: {table}")
            except Exception as e:
                print(f"  Error dropping {table}: {e}")
        conn.commit()

    print("Rollback complete.")


def main():
    parser = argparse.ArgumentParser(description="Screenplay schema migration")
    parser.add_argument(
        "--rollback", action="store_true", help="Rollback (drop tables)"
    )
    args = parser.parse_args()

    engine = get_engine()

    if args.rollback:
        confirm = input("This will DROP all screenplay tables. Type 'yes' to confirm: ")
        if confirm.lower() == "yes":
            rollback(engine)
        else:
            print("Rollback cancelled.")
    else:
        create_tables(engine)
        insert_default_config(engine)
        print("\nMigration complete!")
        print("Next: Run the script parser to populate data.")


if __name__ == "__main__":
    main()
