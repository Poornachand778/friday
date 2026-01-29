"""Database configuration utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    name: str
    user: str
    password: str
    options: Dict[str, Any]

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        host = os.getenv("DB_HOST", "localhost")
        port = int(os.getenv("DB_PORT", "5432"))
        name = os.getenv("DB_NAME", "vectordb")
        user = os.getenv("DB_USER", "vectoruser")
        password = os.getenv("DB_PASSWORD", "friday")
        options: Dict[str, Any] = {}
        sslmode = os.getenv("DB_SSLMODE")
        if sslmode:
            options["sslmode"] = sslmode
        return cls(host, port, name, user, password, options)

    def sqlalchemy_url(self) -> str:
        params = ""
        if self.options:
            opts = "&".join(f"{key}={value}" for key, value in self.options.items())
            params = f"?{opts}"
        return f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}{params}"


@lru_cache(maxsize=1)
def get_engine(settings: DatabaseSettings | None = None) -> Engine:
    """Return a cached SQLAlchemy engine configured from environment variables."""
    if settings is None:
        settings = DatabaseSettings.from_env()
    return create_engine(settings.sqlalchemy_url(), future=True)
