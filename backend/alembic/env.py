"""Alembic environment.

The service passes its own connection (``fnd_api.db.session.init_db``). From the command line the
database URL is taken from the service settings.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection

from fnd_api.config import Settings
from fnd_api.db.models import Base
from fnd_api.db.session import create_engine

config = context.config
target_metadata = Base.metadata


def run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_from_command_line() -> None:
    engine = create_engine(Settings().database_url)
    async with engine.begin() as connection:
        await connection.run_sync(run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    raise SystemExit("offline (--sql) migrations are not supported; run against the database")

provided = config.attributes.get("connection")
if provided is not None:
    run_migrations(provided)
else:
    if config.config_file_name is not None:
        fileConfig(config.config_file_name)
    asyncio.run(run_from_command_line())
