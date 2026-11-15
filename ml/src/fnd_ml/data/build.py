"""``fnd data build``: raw datasets -> cleaned, deduplicated, grouped splits plus a manifest.

Order: load, clean, exact duplicates, near-duplicates, group by publisher, split. The evaluation-only
datasets are cleaned and written next to the splits; they never enter the corpus.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import fnd_ml
from fnd_ml.data import loaders
from fnd_ml.data.clean import add_year, clean_frame
from fnd_ml.data.dedup import (
    count_cross_split_near_duplicates,
    drop_exact_duplicates,
    drop_near_duplicates,
)
from fnd_ml.data.download import sha256_file
from fnd_ml.data.schema import validate_frame
from fnd_ml.data.splits import SPLITS, assign_group, assign_split

WIRE_CHECKED_DATASETS = ("welfake", "isot")
MIN_WIRE_RELIABLE_SHARE = 0.95
SPLITS_VERSION = 2
MANIFEST_NAME = "splits_manifest.json"
DEFAULT_LOCK = Path("ml/configs/data_v2.lock")


class ManifestMismatch(RuntimeError):
    """The processed data on disk is not the frozen version that models are compared on."""


def _check_wire_labels(name: str, frame: pd.DataFrame) -> None:
    share = loaders.wire_label_share(frame)
    if not math.isnan(share) and share < MIN_WIRE_RELIABLE_SHARE:
        raise ValueError(
            f"{name}: only {share:.1%} of wire-service rows are labelled reliable; "
            "the loader's label map looks inverted"
        )


def _balance(frame: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    grouped = frame.groupby("dataset")["label"]
    return {
        str(name): {"rows": int(count), "unreliable_share": round(float(mean), 4)}
        for name, count, mean in zip(
            grouped.count().index, grouped.count(), grouped.mean(), strict=True
        )
    }


def _shared(frame: pd.DataFrame, column: str) -> int:
    per_value = frame.groupby(column)["split"].nunique()
    return int((per_value > 1).sum())


def _dates(frame: pd.DataFrame) -> dict[str, int | None]:
    years = frame["year"].dropna()
    return {
        "rows_with_year": len(years),
        "min_year": int(years.min()) if len(years) else None,
        "max_year": int(years.max()) if len(years) else None,
    }


def _external_sets(
    raw_dir: Path, corpus_hashes: set[str]
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float | int]], list[str]]:
    """The evaluation-only datasets that are on disk, cleaned, without any text that is also in
    the corpus. Returns the frames, their manifest entries and the names that were not found."""
    frames: dict[str, pd.DataFrame] = {}
    summary: dict[str, dict[str, float | int]] = {}
    missing: list[str] = []
    for name in loaders.EXTERNAL_LOADERS:
        if not (raw_dir / name).is_dir():
            missing.append(name)
            continue
        cleaned = add_year(clean_frame(loaders.load_external(name, raw_dir)))
        overlap = cleaned["text_hash"].isin(corpus_hashes)
        frame = cleaned[~overlap].reset_index(drop=True)
        frames[name] = frame
        summary[name] = {
            "rows": len(frame),
            "unreliable_share": round(float(frame["label"].mean()), 4) if len(frame) else 0.0,
            "dropped_overlap_with_corpus": int(overlap.sum()),
        }
    return frames, summary, missing


def build_corpus(raw_dir: Path, out_dir: Path, seed: int = 13) -> dict[str, Any]:
    loaded: list[pd.DataFrame] = []
    sources: dict[str, int] = {}
    for name in loaders.LOADERS:
        frame = loaders.load(name, raw_dir)
        validate_frame(frame)
        if name in WIRE_CHECKED_DATASETS:
            _check_wire_labels(name, frame)
        sources[name] = len(frame)
        loaded.append(frame)

    cleaned = clean_frame(pd.concat(loaded, ignore_index=True))
    exact, dedup_stats = drop_exact_duplicates(cleaned)
    corpus, near_stats = drop_near_duplicates(exact, seed=seed)
    corpus = add_year(corpus)
    corpus["group"] = assign_group(corpus)
    corpus["split"] = assign_split(corpus["group"], seed=seed)

    middle = add_year(clean_frame(loaders.load_liar_middle(raw_dir)))
    middle["group"] = assign_group(middle)
    middle["split"] = assign_split(middle["group"], seed=seed)

    external, external_summary, external_missing = _external_sets(raw_dir, set(corpus["text_hash"]))

    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    (out_dir / "eval").mkdir(parents=True, exist_ok=True)
    written = {"corpus.parquet": corpus, "eval/liar_middle.parquet": middle}
    written |= {f"eval/{name}.parquet": frame for name, frame in external.items()}
    written |= {f"splits/{split}.parquet": corpus[corpus["split"] == split] for split in SPLITS}
    for relative, frame in written.items():
        frame.to_parquet(out_dir / relative, index=False)

    manifest: dict[str, Any] = {
        "fnd_ml_version": fnd_ml.__version__,
        "splits_version": SPLITS_VERSION,
        "seed": seed,
        "sources": sources,
        "dedup": dedup_stats,
        "near_dedup": near_stats,
        "rows": {"total": len(corpus)} | {s: int((corpus["split"] == s).sum()) for s in SPLITS},
        "balance": {s: _balance(corpus[corpus["split"] == s]) for s in SPLITS},
        "length_buckets": {
            str(k): int(v) for k, v in corpus["length_bucket"].value_counts().items()
        },
        "groups": int(corpus["group"].nunique()),
        "shared_groups_between_splits": _shared(corpus, "group"),
        "shared_hashes_between_splits": _shared(corpus, "text_hash"),
        "shared_near_duplicate_clusters_between_splits": count_cross_split_near_duplicates(
            corpus, seed=seed
        ),
        "dates": _dates(corpus),
        "liar_middle_rows": len(middle),
        "external": external_summary,
        "external_missing": external_missing,
        "files": {relative: sha256_file(out_dir / relative) for relative in sorted(written)},
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_lock(processed_dir: Path, lock_path: Path = DEFAULT_LOCK) -> str:
    """Freeze the processed data: record the manifest's sha256 in ``lock_path`` and return it."""
    manifest = json.loads((processed_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    digest = sha256_file(processed_dir / MANIFEST_NAME)
    lock = {
        "manifest_sha256": digest,
        "splits_version": manifest.get("splits_version"),
        "rows": manifest["rows"],
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return digest


def assert_frozen(processed_dir: Path, lock_path: Path = DEFAULT_LOCK) -> str:
    """Return the manifest's sha256 if it is the frozen one; raise ``ManifestMismatch`` if not.

    Every model in a comparison must be trained and scored on the same splits, so training and
    evaluation call this first.
    """
    if not lock_path.exists():
        raise ManifestMismatch(
            f"{lock_path} does not exist. Build the data, then run `fnd data freeze`."
        )
    expected = json.loads(lock_path.read_text(encoding="utf-8"))["manifest_sha256"]
    actual = sha256_file(processed_dir / MANIFEST_NAME)
    if actual != expected:
        raise ManifestMismatch(
            f"{processed_dir / MANIFEST_NAME} (sha256 {actual[:12]}) does not match the frozen "
            f"version in {lock_path} ({expected[:12]}). Rebuild the same data, or freeze the new "
            "build with `fnd data freeze` and retrain every model that is compared."
        )
    return actual
