"""Engine and session setup. ``init_db`` brings the schema to the latest migration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from fnd_api.config import Settings

# backend/alembic, next to backend/src. The service runs from a source checkout (also in Docker).
ALEMBIC_DIR = Path(__file__).resolve().parents[3] / "alembic"


def _on_connect(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")  # wait for a concurrent writer instead of failing
    cursor.close()


def create_engine(database_url: str) -> AsyncEngine:
    url = make_url(database_url)
    options: dict[str, Any] = {}
    if url.get_backend_name() == "sqlite":
        if url.database in (None, "", ":memory:"):
            options["poolclass"] = StaticPool  # an in-memory database lives in one connection
        else:
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(url, **options)
    if url.get_backend_name() == "sqlite":
        event.listen(engine.sync_engine, "connect", _on_connect)
    return engine


def _upgrade(connection: Connection) -> None:
    if not ALEMBIC_DIR.is_dir():
        raise RuntimeError(f"migrations not found at {ALEMBIC_DIR}")
    config = Config()
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    config.attributes["connection"] = connection
    command.upgrade(config, "head")


async def init_db(settings: Settings) -> AsyncEngine:
    """Create the engine and run ``alembic upgrade head`` on it."""
    engine = create_engine(settings.database_url)
    async with engine.begin() as connection:
        await connection.run_sync(_upgrade)
    return engine


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
