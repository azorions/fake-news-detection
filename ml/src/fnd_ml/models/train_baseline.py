"""``fnd train baseline``: fit, calibrate, pick thresholds, save the artifact and write a report."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fnd_ml.calibration.thresholds import Thresholds, pick_thresholds
from fnd_ml.common.artifacts import register
from fnd_ml.data.download import sha256_file
from fnd_ml.eval.metrics import classification_report
from fnd_ml.eval.report import write_report
from fnd_ml.models.baseline import BaselineConfig, BaselineModel

_EVAL_SPLITS = ("val", "calib", "test")
_FEATURE_NAME_SAMPLE = 2000


def _iter_texts(path: Path) -> Iterator[str]:
    """Stream the ``model_text`` column so the whole split is never held in memory as strings."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["model_text"]):
        yield from batch.column(0).to_pylist()


def _bands(y: np.ndarray, p: np.ndarray, thresholds: Thresholds) -> dict[str, dict[str, float]]:
    masks = {
        "reliable": p < thresholds.t_low,
        "potentially_misleading": (p >= thresholds.t_low) & (p < thresholds.t_high),
        "high_risk": p >= thresholds.t_high,
    }
    return {
        band: {
            "share": round(float(mask.mean()), 4),
            "unreliable_rate": round(float(y[mask].mean()), 4) if mask.any() else float("nan"),
        }
        for band, mask in masks.items()
    }


def _by_group(frame: pd.DataFrame, p: np.ndarray, column: str) -> dict[str, dict[str, Any]]:
    labels = frame["label"].to_numpy()
    return {
        str(value): classification_report(labels[mask], p[mask])
        for value in sorted(frame[column].unique())
        for mask in [(frame[column] == value).to_numpy()]
    }


def train_baseline(
    processed_dir: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    config: BaselineConfig | None = None,
) -> dict[str, Any]:
    config = config or BaselineConfig()
    splits = processed_dir / "splits"
    columns = ["label", "dataset", "length_bucket"]
    meta = {
        s: pd.read_parquet(splits / f"{s}.parquet", columns=columns)
        for s in ("train", *_EVAL_SPLITS)
    }
    labels = {split: frame["label"].to_numpy() for split, frame in meta.items()}

    model = BaselineModel(config).fit(_iter_texts(splits / "train.parquet"), labels["train"])
    raw = {s: model.predict_raw(_iter_texts(splits / f"{s}.parquet")) for s in _EVAL_SPLITS}
    model.calibrate_scores(raw["calib"], labels["calib"])
    calibrated = {split: model.to_calibrated(scores) for split, scores in raw.items()}
    thresholds = pick_thresholds(labels["calib"], calibrated["calib"])

    data_sha = sha256_file(processed_dir / "splits_manifest.json")
    fingerprint = json.dumps(asdict(config), sort_keys=True) + data_sha
    version = "v1-" + hashlib.sha256(fingerprint.encode()).hexdigest()[:8]

    metrics: dict[str, Any] = {
        split: {
            "raw": classification_report(labels[split], raw[split]),
            "calibrated": classification_report(labels[split], calibrated[split]),
        }
        for split in _EVAL_SPLITS
    }
    metrics["test_by_dataset"] = _by_group(meta["test"], calibrated["test"], "dataset")
    metrics["test_by_length"] = _by_group(meta["test"], calibrated["test"], "length_bucket")

    sample = itertools.islice(_iter_texts(splits / "train.parquet"), _FEATURE_NAME_SAMPLE)
    report: dict[str, Any] = {
        "model": "baseline",
        "version": version,
        "config": asdict(config),
        "data_manifest_sha256": data_sha,
        "rows": {split: len(frame) for split, frame in meta.items()},
        "metrics": metrics,
        "thresholds": thresholds.to_dict(),
        "bands_on_test": _bands(labels["test"], calibrated["test"], thresholds),
        "top_features": model.top_features(sample),
    }

    artifact = artifacts_dir / "baseline" / version
    model.save(
        artifact,
        {
            "version": version,
            "data_manifest_sha256": data_sha,
            "test_metrics": metrics["test"]["calibrated"],
        },
    )
    thresholds_text = json.dumps(thresholds.to_dict(), indent=2) + "\n"
    (artifact / "thresholds.json").write_text(thresholds_text, encoding="utf-8")
    register(artifacts_dir, "baseline", version, f"baseline/{version}")
    write_report(report, reports_dir, "baseline_v1")
    return report
