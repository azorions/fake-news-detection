"""``fnd data build``: raw datasets -> cleaned, deduplicated, grouped splits plus a manifest."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import fnd_ml
from fnd_ml.data import loaders
from fnd_ml.data.clean import clean_frame
from fnd_ml.data.dedup import drop_exact_duplicates
from fnd_ml.data.download import sha256_file
from fnd_ml.data.schema import validate_frame
from fnd_ml.data.splits import SPLITS, assign_group, assign_split

WIRE_CHECKED_DATASETS = ("welfake", "isot")
MIN_WIRE_RELIABLE_SHARE = 0.95


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
    corpus, dedup_stats = drop_exact_duplicates(cleaned)
    corpus["group"] = assign_group(corpus)
    corpus["split"] = assign_split(corpus["group"], seed=seed)

    middle = clean_frame(loaders.load_liar_middle(raw_dir))
    middle["group"] = assign_group(middle)
    middle["split"] = assign_split(middle["group"], seed=seed)

    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    (out_dir / "eval").mkdir(parents=True, exist_ok=True)
    written = {"corpus.parquet": corpus, "eval/liar_middle.parquet": middle}
    written |= {f"splits/{split}.parquet": corpus[corpus["split"] == split] for split in SPLITS}
    for relative, frame in written.items():
        frame.to_parquet(out_dir / relative, index=False)

    manifest: dict[str, Any] = {
        "fnd_ml_version": fnd_ml.__version__,
        "seed": seed,
        "sources": sources,
        "dedup": dedup_stats,
        "rows": {"total": len(corpus)} | {s: int((corpus["split"] == s).sum()) for s in SPLITS},
        "balance": {s: _balance(corpus[corpus["split"] == s]) for s in SPLITS},
        "length_buckets": {
            str(k): int(v) for k, v in corpus["length_bucket"].value_counts().items()
        },
        "groups": int(corpus["group"].nunique()),
        "shared_groups_between_splits": _shared(corpus, "group"),
        "shared_hashes_between_splits": _shared(corpus, "text_hash"),
        "liar_middle_rows": len(middle),
        "files": {relative: sha256_file(out_dir / relative) for relative in sorted(written)},
    }
    (out_dir / "splits_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
