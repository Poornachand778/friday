#!/usr/bin/env python3
"""
Voice Schema Migration for Friday AI
=====================================

Creates voice pipeline database tables:
- voice_sessions
- voice_turns
- voice_profiles
- voice_training_examples

Usage:
    python scripts/migrate_voice_schema.py [--drop-existing]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import create_engine, text, inspect
from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from db.voice_schema import (
    Base,
    VoiceSession,
    VoiceTurn,
    VoiceProfile,
    VoiceTrainingExample,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def get_database_url() -> str:
    """Get database URL from environment"""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        # Build from components
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "friday")
        db_user = os.environ.get("DB_USER", "friday")
        db_pass = os.environ.get("DB_PASSWORD", "friday")
        db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    return db_url


def migrate_voice_schema(drop_existing: bool = False) -> None:
    """Create voice schema tables"""
    db_url = get_database_url()
    engine = create_engine(db_url)

    VOICE_TABLES = [
        "voice_training_examples",
        "voice_turns",
        "voice_sessions",
        "voice_profiles",
    ]

    with engine.connect() as conn:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        if drop_existing:
            LOGGER.warning("Dropping existing voice tables...")
            for table in VOICE_TABLES:
                if table in existing_tables:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                    LOGGER.info("  Dropped: %s", table)
            conn.commit()

        # Check for conflicts
        for table in VOICE_TABLES:
            if table in existing_tables and not drop_existing:
                LOGGER.info("Table %s already exists, skipping", table)

    # Create tables
    LOGGER.info("Creating voice schema tables...")

    # Create only voice-related tables
    voice_tables = [
        VoiceSession.__table__,
        VoiceTurn.__table__,
        VoiceProfile.__table__,
        VoiceTrainingExample.__table__,
    ]

    for table in voice_tables:
        table.create(engine, checkfirst=True)
        LOGGER.info("  Created: %s", table.name)

    LOGGER.info("Voice schema migration complete!")

    # Show table counts
    with engine.connect() as conn:
        for table in VOICE_TABLES:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            LOGGER.info("  %s: %d rows", table, count)


def create_default_voice_profile(engine) -> None:
    """Create default Friday Telugu voice profile"""
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        # Check if default profile exists
        existing = session.query(VoiceProfile).filter_by(name="friday_telugu").first()
        if existing:
            LOGGER.info("Default voice profile 'friday_telugu' already exists")
            return

        # Create default profile
        profile = VoiceProfile(
            name="friday_telugu",
            description="Friday AI Telugu voice - warm, helpful assistant",
            language="te",
            tts_engine="xtts_v2",
            speaking_rate=1.0,
            is_active=True,
            is_default=True,
            reference_audio_paths=[],  # Will be populated with voice samples
        )
        session.add(profile)
        session.commit()
        LOGGER.info("Created default voice profile: friday_telugu")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate voice schema")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing voice tables before creating",
    )
    parser.add_argument(
        "--create-default-profile",
        action="store_true",
        help="Create default Friday Telugu voice profile",
    )
    args = parser.parse_args(argv)

    try:
        migrate_voice_schema(drop_existing=args.drop_existing)

        if args.create_default_profile:
            db_url = get_database_url()
            engine = create_engine(db_url)
            create_default_voice_profile(engine)

        return 0
    except Exception as e:
        LOGGER.error("Migration failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
