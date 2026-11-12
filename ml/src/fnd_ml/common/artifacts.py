"""Registry of trained model artifacts: ``<artifacts_dir>/manifest.json`` names the active versions."""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_NAME = "manifest.json"


def read_manifest(artifacts_dir: Path) -> dict[str, dict[str, str]]:
    path = artifacts_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    manifest: dict[str, dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
    return manifest


def register(artifacts_dir: Path, kind: str, version: str, relative_path: str) -> None:
    """Make ``relative_path`` the active artifact of ``kind`` (for example ``"baseline"``)."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(artifacts_dir)
    manifest[kind] = {"version": version, "path": relative_path}
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (artifacts_dir / MANIFEST_NAME).write_text(text, encoding="utf-8")


def resolve(artifacts_dir: Path, kind: str) -> Path | None:
    """Directory of the active artifact of ``kind``, or ``None`` if none is registered or it is gone."""
    entry = read_manifest(artifacts_dir).get(kind)
    if entry is None:
        return None
    path = artifacts_dir / entry["path"]
    return path if path.is_dir() else None
