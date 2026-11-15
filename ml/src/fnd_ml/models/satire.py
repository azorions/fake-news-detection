"""``fnd train satire``: a small model that scores how much a headline reads like satire.

Satire is not misinformation, but a classifier trained on news can mistake it for a fabricated
story. The score is reported next to the verdict so that a reader can tell the two apart.

Same model family as the baseline (hashed TF-IDF, logistic regression, isotonic calibration), on
headlines only. The satire headlines are an evaluation set: this is the only model trained on
them, and its test tenth is written out for evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from fnd_ml.common.artifacts import register
from fnd_ml.data.build import DEFAULT_LOCK, assert_frozen
from fnd_ml.models.baseline import BaselineConfig, BaselineModel
from fnd_ml.models.train_baseline import REPORT_SUFFIX, model_version, split_metrics

SATIRE_FILE = "satire_headlines.parquet"
SLICE_FILE = "satire_slice.parquet"  # the test tenth: nothing is fit on it
DEFAULT_CONFIG = BaselineConfig(n_features=2**18, max_tokens=64, min_df=2, C=2.0)
_METRICS = ("n", "unreliable_share", "macro_f1", "roc_auc", "pr_auc", "brier", "ece")
_HEADERS = ("rows", "satire share", "macro-F1", "ROC-AUC", "PR-AUC", "Brier", "ECE")


def satire_split(text_hashes: pd.Series) -> pd.Series:
    """``train`` / ``calib`` / ``test`` (80/10/10) from the text hash alone, so that equal texts
    always land in the same part."""
    bucket = text_hashes.map(lambda digest: int(digest[:8], 16) % 10)
    return bucket.map(lambda b: "train" if b < 8 else "calib" if b == 8 else "test")


def train_satire(
    processed_dir: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    config: BaselineConfig | None = None,
    *,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    config = config or DEFAULT_CONFIG
    data_sha = assert_frozen(processed_dir, lock_path)
    frame = pd.read_parquet(processed_dir / "eval" / SATIRE_FILE)
    split = satire_split(frame["text_hash"])
    parts = {name: frame[split == name] for name in ("train", "calib", "test")}
    labels = {name: part["label"].to_numpy() for name, part in parts.items()}

    model = BaselineModel(config).fit(parts["train"]["model_text"], labels["train"])
    raw = {name: model.predict_raw(parts[name]["model_text"]) for name in ("calib", "test")}
    model.calibrate_scores(raw["calib"], labels["calib"])
    version = model_version(config, data_sha)

    top = model.top_features(parts["train"]["model_text"], k=25)
    by_publisher = frame.groupby("publisher")["label"].agg(["size", "mean"])
    report: dict[str, Any] = {
        "model": "satire",
        "version": version,
        "config": asdict(config),
        "data_manifest_sha256": data_sha,
        "rows": {name: len(part) for name, part in parts.items()},
        "publishers": {
            str(publisher): {"rows": int(row["size"]), "satire_share": round(float(row["mean"]), 4)}
            for publisher, row in by_publisher.iterrows()
        },
        "metrics": {
            name: split_metrics(labels[name], raw[name], model.to_calibrated(raw[name]))
            for name in ("calib", "test")
        },
        "top_features": {"satire": top["unreliable"], "not_satire": top["reliable"]},
    }

    parts["test"].to_parquet(processed_dir / "eval" / SLICE_FILE, index=False)
    artifact = artifacts_dir / "satire" / version
    model.save(
        artifact,
        {
            "version": version,
            "data_manifest_sha256": data_sha,
            "test_metrics": report["metrics"]["test"]["calibrated"],
        },
        kind="satire",
    )
    register(artifacts_dir, "satire", version, f"satire/{version}")
    reports_dir.mkdir(parents=True, exist_ok=True)
    name = f"satire_{REPORT_SUFFIX}"
    (reports_dir / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (reports_dir / f"{name}.md").write_text(_render(report), encoding="utf-8")
    return report


def _render(report: dict[str, Any]) -> str:
    config, rows = report["config"], report["rows"]
    lines = [
        f"# Satire headline scorer: {report['version']}",
        "",
        f"TF-IDF (hashed word 1-2-grams of the first {config['max_tokens']} words) + logistic "
        "regression, isotonic calibration, trained on headlines only.",
        f"Data manifest sha256: `{report['data_manifest_sha256']}`",
        "",
        "## What the score means",
        "",
        "Every satire headline in the data comes from one publisher and every other headline from "
        "another, so the score partly reflects the house style of these two publishers, not "
        "satire as such. It is a hint shown next to the verdict, not a verdict.",
        "",
        "| publisher | rows | satire share |",
        "|---|---:|---:|",
    ]
    for publisher, stats in sorted(report["publishers"].items()):
        lines.append(f"| {publisher} | {stats['rows']} | {stats['satire_share']} |")
    lines += [
        "",
        "## Metrics",
        "",
        f"The rows are split 80/10/10 by text hash: {rows['train']} train, {rows['calib']} calib, "
        f"{rows['test']} test. The calibrator is fit on the calib tenth, so its calibrated numbers "
        f"are in-sample. Nothing is fit on the test tenth (`eval/{SLICE_FILE}`).",
        "",
        "| | " + " | ".join(_HEADERS) + " |",
        "|---|" + "---:|" * len(_HEADERS),
    ]
    for split in ("calib", "test"):
        for kind in ("raw", "calibrated"):
            metrics = report["metrics"][split][kind]
            cells = ["-" if metrics[key] is None else str(metrics[key]) for key in _METRICS]
            lines.append(f"| {split} ({kind}) | " + " | ".join(cells) + " |")
    lines += ["", "## Strongest features", ""]
    for side, title in (("satire", "satire"), ("not_satire", "not satire")):
        features = ", ".join(f"`{feature}`" for feature in report["top_features"][side])
        lines += [f"**Toward {title}:** {features}", ""]
    return "\n".join(lines)
