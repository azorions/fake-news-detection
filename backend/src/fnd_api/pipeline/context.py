from __future__ import annotations

from dataclasses import dataclass

import httpx

from fnd_api.cache import Cache
from fnd_api.config import Settings
from fnd_api.db.repositories import ScanRepository
from fnd_api.ingest.url_guard import Resolver
from fnd_api.models_registry import ModelRegistry
from fnd_api.signals.rules import RuleEngine


@dataclass(frozen=True)
class PipelineContext:
    """Everything one analysis needs from the running application."""

    settings: Settings
    models: ModelRegistry
    cache: Cache
    http: httpx.AsyncClient
    rules: RuleEngine
    scans: ScanRepository
    resolver: Resolver
