"""``fnd train baseline``: fit, calibrate, pick thresholds, save the artifact and write a report.

Also holds what every model family's training shares (the version name, the report tables) and the
character n-gram ablation.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import NDArray

from fnd_ml.calibration.thresholds import Thresholds, pick_thresholds
from fnd_ml.common.artifacts import register
from fnd_ml.common.resources import PeakMemory
from fnd_ml.data.build import DEFAULT_LOCK, assert_frozen
from fnd_ml.data.leak_probe import (
    DEFAULT_IDENTIFIERS,
    count_unnamed,
    label_probe,
    load_identifiers,
    load_probe_sample,
    origin_probe,
)
from fnd_ml.eval.metrics import classification_report, report_by_group
from fnd_ml.eval.report import write_report
from fnd_ml.models.baseline import BaselineConfig, BaselineModel

META_COLUMNS = ["label", "dataset", "length_bucket"]
REPORT_SUFFIX = "v2"  # reports are named <model>_v2: they describe models trained on splits v2
_FEATURE_NAME_SAMPLE = 10_000
ADOPT_MIN_GAIN = 0.01  # character n-grams must add this much val macro-F1 ...
ADOPT_MAX_GB = 6.0  # ... and fit the whole train split into this much memory


def iter_texts(path: Path) -> Iterator[str]:
    """Stream the ``model_text`` column so the whole split is never held in memory as strings."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["model_text"]):
        yield from batch.column(0).to_pylist()


def load_meta(processed_dir: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(processed_dir / "splits" / f"{split}.parquet", columns=META_COLUMNS)


def model_version(config: Any, data_sha: str) -> str:
    """``v2-<hash>`` of a config dataclass and the frozen data manifest."""
    fingerprint = json.dumps(asdict(config), sort_keys=True) + data_sha
    return "v2-" + hashlib.sha256(fingerprint.encode()).hexdigest()[:8]


def split_metrics(
    y: NDArray[np.int_], raw: NDArray[np.float64], calibrated: NDArray[np.float64]
) -> dict[str, Any]:
    return {
        "raw": classification_report(y, raw),
        "calibrated": classification_report(y, calibrated),
    }


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


def add_test_results(
    report: dict[str, Any],
    meta: pd.DataFrame,
    raw: NDArray[np.float64],
    calibrated: NDArray[np.float64],
    thresholds: Thresholds,
) -> None:
    """Put the test split's numbers into ``report``: metrics, by dataset, by length, bands."""
    labels = meta["label"].to_numpy()
    report["rows"]["test"] = len(meta)
    report["metrics"]["test"] = split_metrics(labels, raw, calibrated)
    report["metrics"]["test_by_dataset"] = report_by_group(meta, calibrated, "dataset")
    report["metrics"]["test_by_length"] = report_by_group(meta, calibrated, "length_bucket")
    report["bands_on_test"] = _bands(labels, calibrated, thresholds)


def save_thresholds(artifact: Path, thresholds: Thresholds) -> None:
    text = json.dumps(thresholds.to_dict(), indent=2) + "\n"
    (artifact / "thresholds.json").write_text(text, encoding="utf-8")


def _origin(processed_dir: Path) -> dict[str, Any] | None:
    """The dataset-of-origin probe, or ``None`` when a split holds a single dataset."""
    sample = load_probe_sample(processed_dir)
    if sample.groupby("split")["dataset"].nunique().min() < 2:
        return None
    return origin_probe(sample)


def train_baseline(
    processed_dir: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    config: BaselineConfig | None = None,
    *,
    lock_path: Path = DEFAULT_LOCK,
    identifiers_path: Path = DEFAULT_IDENTIFIERS,
) -> dict[str, Any]:
    """Train on the frozen splits.

    When one of the strongest features names a source, the report is written without test numbers
    and nothing is saved: the test split is not opened, so it stays unseen until the leak is fixed.
    """
    config = config or BaselineConfig()
    data_sha = assert_frozen(processed_dir, lock_path)
    splits = processed_dir / "splits"
    meta = {split: load_meta(processed_dir, split) for split in ("train", "val", "calib")}
    labels = {split: frame["label"].to_numpy() for split, frame in meta.items()}

    model = BaselineModel(config).fit(iter_texts(splits / "train.parquet"), labels["train"])
    raw = {s: model.predict_raw(iter_texts(splits / f"{s}.parquet")) for s in ("val", "calib")}
    model.calibrate_scores(raw["calib"], labels["calib"])
    calibrated = {split: model.to_calibrated(scores) for split, scores in raw.items()}
    thresholds = pick_thresholds(labels["calib"], calibrated["calib"])
    version = model_version(config, data_sha)

    # Every k-th train row: the split file is ordered by dataset, so its head is one dataset only.
    step = max(1, len(meta["train"]) // _FEATURE_NAME_SAMPLE)
    sample = itertools.islice(iter_texts(splits / "train.parquet"), 0, None, step)
    top_features = model.top_features(sample)
    flagged = label_probe(top_features, load_identifiers(identifiers_path))

    report: dict[str, Any] = {
        "model": "baseline",
        "version": version,
        "config": asdict(config),
        "data_manifest_sha256": data_sha,
        "rows": {split: len(frame) for split, frame in meta.items()},
        "metrics": {s: split_metrics(labels[s], raw[s], calibrated[s]) for s in ("val", "calib")},
        "thresholds": thresholds.to_dict(),
        "top_features": top_features,
        "leak_probe": {"flagged": flagged, "unnamed_buckets": count_unnamed(top_features)},
        "origin_probe": _origin(processed_dir),
    }
    if flagged:
        write_report(report, reports_dir, f"baseline_{REPORT_SUFFIX}")
        return report

    test_meta = load_meta(processed_dir, "test")
    test_raw = model.predict_raw(iter_texts(splits / "test.parquet"))
    add_test_results(report, test_meta, test_raw, model.to_calibrated(test_raw), thresholds)

    artifact = artifacts_dir / "baseline" / version
    model.save(
        artifact,
        {
            "version": version,
            "data_manifest_sha256": data_sha,
            "test_metrics": report["metrics"]["test"]["calibrated"],
        },
    )
    save_thresholds(artifact, thresholds)
    register(artifacts_dir, "baseline", version, f"baseline/{version}")
    write_report(report, reports_dir, f"baseline_{REPORT_SUFFIX}")
    return report


# ---------------------------------------------------------------- character n-gram ablation


def stratified_positions(meta: pd.DataFrame, n: int, seed: int) -> NDArray[np.int_]:
    """``n`` row positions of ``meta`` (fewer if it is smaller), keeping its dataset x label mix."""
    if n >= len(meta):
        return np.arange(len(meta))
    rng = np.random.default_rng(seed)
    groups = list(meta.groupby(["dataset", "label"], sort=True).indices.values())
    exact = np.array([len(group) for group in groups]) * n / len(meta)
    quota = np.floor(exact).astype(int)
    # Hand the rows lost to rounding down to the groups with the largest remainders.
    for index in np.argsort(quota - exact, kind="stable")[: n - quota.sum()]:
        quota[index] += 1
    chosen = [
        rng.choice(group, size=size, replace=False)
        for group, size in zip(groups, quota, strict=True)
    ]
    return np.sort(np.concatenate(chosen))


def _texts_at(path: Path, positions: NDArray[np.int_]) -> list[str]:
    wanted = set(positions.tolist())
    return [text for position, text in enumerate(iter_texts(path)) if position in wanted]


def ablation_char_ngrams(
    processed_dir: Path,
    reports_dir: Path,
    config: BaselineConfig | None = None,
    *,
    sample_rows: int = 20_000,
    seed: int = 13,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    """Word 1-2-grams against word 1-2-grams + ``char_wb`` 3-5-grams, on a stratified train sample.

    Scored on val, uncalibrated. The decision rule is fixed in advance: adopt character n-grams
    only if val macro-F1 rises by ``ADOPT_MIN_GAIN`` and the projected memory for the whole train
    split stays under ``ADOPT_MAX_GB``.
    """
    config = config or BaselineConfig()
    data_sha = assert_frozen(processed_dir, lock_path)
    splits = processed_dir / "splits"
    train_meta, val_meta = load_meta(processed_dir, "train"), load_meta(processed_dir, "val")
    positions = stratified_positions(train_meta, sample_rows, seed)
    train_texts = _texts_at(splits / "train.parquet", positions)
    train_y = train_meta["label"].to_numpy()[positions]
    val_texts = list(iter_texts(splits / "val.parquet"))
    val_y = val_meta["label"].to_numpy()
    scale = len(train_meta) / len(positions)

    variants: list[dict[str, Any]] = []
    for name, char_ngrams in (("word 1-2", False), ("word 1-2 + char_wb 3-5", True)):
        gc.collect()
        with PeakMemory() as memory:
            model = BaselineModel(replace(config, char_ngrams=char_ngrams))
            model.fit(train_texts, train_y)
            scores = model.predict_raw(val_texts)
        del model
        metrics = classification_report(val_y, scores)
        variants.append(
            {
                "name": name,
                "val": {key: metrics[key] for key in ("macro_f1", "roc_auc")},
                "peak_memory_mb": round(memory.peak_mb, 1),
                "projected_full_train_gb": round(memory.peak_mb * scale / 1024, 2),
            }
        )

    words, chars = variants
    gain = round(chars["val"]["macro_f1"] - words["val"]["macro_f1"], 4)
    result: dict[str, Any] = {
        "data_manifest_sha256": data_sha,
        "config": asdict(config),
        "rows": {"train_sample": len(positions), "train": len(train_meta), "val": len(val_meta)},
        "variants": variants,
        "macro_f1_gain": gain,
        "adopt_char_ngrams": bool(
            gain >= ADOPT_MIN_GAIN and chars["projected_full_train_gb"] < ADOPT_MAX_GB
        ),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "ablation_char_ngrams.md").write_text(_render_ablation(result), encoding="utf-8")
    return result


def _render_ablation(result: dict[str, Any]) -> str:
    rows = result["rows"]
    lines = [
        "# Ablation: character n-grams in the baseline",
        "",
        f"Both variants are fit on the same stratified sample of {rows['train_sample']} train rows "
        f"(of {rows['train']}) and scored on the whole val split ({rows['val']} rows), "
        "uncalibrated.",
        f"Data manifest sha256: `{result['data_manifest_sha256']}`",
        "",
        "Peak memory is the growth of the process's resident memory while fitting and scoring. "
        "The projection for the whole train split scales it by the row ratio.",
        "",
        "| features | val macro-F1 | val ROC-AUC | peak memory (MB) | projected, full train (GB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in result["variants"]:
        lines.append(
            f"| {variant['name']} | {variant['val']['macro_f1']} | {variant['val']['roc_auc']} "
            f"| {variant['peak_memory_mb']} | {variant['projected_full_train_gb']} |"
        )
    verdict = "adopted" if result["adopt_char_ngrams"] else "not adopted"
    lines += [
        "",
        "## Decision",
        "",
        f"Character n-grams are **{verdict}**: val macro-F1 changes by "
        f"{result['macro_f1_gain']:+.4f}. The rule, fixed before the run, is to adopt them only "
        f"for a gain of at least {ADOPT_MIN_GAIN} with projected memory under {ADOPT_MAX_GB:g} GB.",
        "",
    ]
    return "\n".join(lines)
