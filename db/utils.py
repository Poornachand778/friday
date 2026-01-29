"""Utility helpers for interacting with the database."""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from .config import get_engine
from .schema import Base


def create_all(engine: Engine | None = None) -> None:
    """Create all tables defined in the ORM schema."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)


def get_schema_snapshot(engine: Engine | None = None) -> dict[str, list[str]]:
    """Return a mapping of table names to column names."""
    if engine is None:
        engine = get_engine()
    inspector = inspect(engine)
    snapshot: dict[str, list[str]] = {}
    for table_name in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns(table_name)]
        snapshot[table_name] = columns
    return snapshot
