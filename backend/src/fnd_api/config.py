"""Service settings, read from ``FND_*`` environment variables and an optional ``.env`` file."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from limits import parse as parse_rate_limit
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FND_", env_file=".env", extra="ignore")

    # Storage. ``database_url`` and ``cache_dir`` default to locations inside ``var_dir``.
    var_dir: Path = Path("var")
    database_url: str = ""
    cache_dir: Path | None = None
    redis_url: str | None = None  # when set, the scan cache uses Redis instead of the disk cache
    artifacts_dir: Path = Path("ml/artifacts")
    lexicon_dir: Path = Path("data/lexicons")

    # Limits
    rate_limit_analyze: str = "30/minute"
    rate_limit_read: str = "120/minute"
    max_text_chars: int = 100_000
    max_fetch_bytes: int = 5_000_000
    fetch_timeout_s: float = 8.0
    allow_private_urls: bool = False  # leave off outside tests: it disables the SSRF guard

    # Privacy and logging
    store_input_text: bool = True
    log_json: bool = True
    log_level: str = "INFO"

    @field_validator("rate_limit_analyze", "rate_limit_read")
    @classmethod
    def _valid_rate_limit(cls, value: str) -> str:
        parse_rate_limit(value)  # raises ValueError for anything but "<count>/<period>"
        return value

    @model_validator(mode="after")
    def _derive_storage_paths(self) -> Self:
        if self.cache_dir is None:
            self.cache_dir = self.var_dir / "cache"
        if not self.database_url:
            self.database_url = f"sqlite+aiosqlite:///{(self.var_dir / 'fnd.db').as_posix()}"
        return self
