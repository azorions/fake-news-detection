"""Dataset registry and checksum-verified downloads.

The registry (``data/registry/datasets.yaml``) says where each dataset comes from and under which
license. The first successful download of a file records its sha256 next to the registry in
``checksums.json``; every later download must match, so a silently changed upstream file is an error
instead of a different experiment.

A file that is already present is never downloaded again. That also covers datasets that need a login:
put the file in ``data/raw/<dataset>/`` by hand and it is accepted and checksummed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

_USER_AGENT = "fnd-ml-dataset-fetch/0.1"
_CHUNK_SIZE = 1 << 20


class DownloadError(RuntimeError):
    """A dataset file could not be downloaded."""


class ChecksumMismatch(RuntimeError):
    """A downloaded file does not match the checksum recorded for it."""


@dataclass(frozen=True)
class RemoteFile:
    url: str
    filename: str


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    license: str
    homepage: str
    files: tuple[RemoteFile, ...]
    notes: str = ""


def load_registry(path: Path) -> dict[str, DatasetEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        name: DatasetEntry(
            name=name,
            license=str(spec["license"]),
            homepage=str(spec["homepage"]),
            files=tuple(RemoteFile(url=f["url"], filename=f["filename"]) for f in spec["files"]),
            notes=str(spec.get("notes", "")).strip(),
        )
        for name, spec in raw["datasets"].items()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checksums(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    data: dict[str, dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _write_checksums(path: Path, checksums: dict[str, dict[str, str]]) -> None:
    path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _download(client: httpx.Client, url: str, target: Path) -> str:
    """Stream ``url`` into ``target`` and return the sha256 of what was written."""
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise DownloadError(f"{url} returned HTTP {response.status_code}")
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(_CHUNK_SIZE):
                    handle.write(chunk)
                    digest.update(chunk)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"{url} failed: {exc}") from exc
    except DownloadError:
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, target)
    return digest.hexdigest()


def fetch_dataset(
    name: str, raw_dir: Path, registry: Path, *, timeout_s: float = 120.0
) -> list[Path]:
    """Make every file of dataset ``name`` available under ``raw_dir/name`` and return the paths."""
    entries = load_registry(registry)
    if name not in entries:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(entries)}")

    checksums_path = registry.parent / "checksums.json"
    checksums = _read_checksums(checksums_path)
    recorded = checksums.setdefault(name, {})
    target_dir = raw_dir / name
    target_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    headers = {"User-Agent": _USER_AGENT}
    with httpx.Client(follow_redirects=True, timeout=timeout_s, headers=headers) as client:
        for file in entries[name].files:
            target = target_dir / file.filename
            expected = recorded.get(file.filename)

            if target.exists():
                actual = sha256_file(target)
                if expected is None:
                    recorded[file.filename] = actual
                    _write_checksums(checksums_path, checksums)
                if expected in (None, actual):
                    paths.append(target)
                    continue

            actual = _download(client, file.url, target)
            if expected is not None and actual != expected:
                target.unlink()
                raise ChecksumMismatch(
                    f"{name}/{file.filename}: downloaded sha256 {actual} does not match the "
                    f"recorded {expected}. If the upstream file changed on purpose, remove its "
                    f"entry from {checksums_path.name} and download again."
                )
            recorded[file.filename] = actual
            _write_checksums(checksums_path, checksums)
            paths.append(target)
    return paths
