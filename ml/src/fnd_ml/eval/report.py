"""Writing evaluation reports as JSON (for tools) and Markdown (for people)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_METRICS = ("n", "unreliable_share", "macro_f1", "roc_auc", "pr_auc", "brier", "ece")
_HEADERS = ("rows", "unreliable share", "macro-F1", "ROC-AUC", "PR-AUC", "Brier", "ECE")


def _row(label: str, metrics: dict[str, Any]) -> str:
    cells = ["-" if metrics[key] is None else str(metrics[key]) for key in _METRICS]
    return f"| {label} | " + " | ".join(cells) + " |"


def _table(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    header = "| | " + " | ".join(_HEADERS) + " |"
    divider = "|---|" + "---:|" * len(_HEADERS)
    return [header, divider, *(_row(label, metrics) for label, metrics in rows)]


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    thresholds = report["thresholds"]
    lines = [
        f"# Baseline model report: {report['version']}",
        "",
        "TF-IDF (hashed word 1-2-grams) + logistic regression, isotonic calibration.",
        f"Data manifest sha256: `{report['data_manifest_sha256']}`",
        "",
        "## Metrics by split",
        "",
        "The calibrator and thresholds are fit on the calib split, so its calibrated numbers are",
        "in-sample. The test split is the honest estimate.",
        "",
    ]
    by_split = []
    for split in ("val", "calib", "test"):
        by_split.append((f"{split} (raw)", metrics[split]["raw"]))
        by_split.append((f"{split} (calibrated)", metrics[split]["calibrated"]))
    lines += _table(by_split)
    lines += ["", "## Test split by dataset (calibrated)", ""]
    lines += _table(sorted(metrics["test_by_dataset"].items()))
    lines += ["", "## Test split by length (calibrated)", ""]
    lines += _table(sorted(metrics["test_by_length"].items()))
    lines += [
        "",
        "## Verdict thresholds",
        "",
        f"`t_low = {thresholds['t_low']}`, `t_high = {thresholds['t_high']}` "
        f"(band targets met on calib: {thresholds['met_targets']}).",
        "",
        "| band on test | share of rows | unreliable rate |",
        "|---|---:|---:|",
    ]
    for band, stats in report["bands_on_test"].items():
        lines.append(f"| {band} | {stats['share']} | {stats['unreliable_rate']} |")
    lines += ["", "## Strongest features", ""]
    for side in ("unreliable", "reliable"):
        lines.append(
            f"**Toward {side}:** " + ", ".join(f"`{f}`" for f in report["top_features"][side])
        )
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], reports_dir: Path, name: str) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = reports_dir / f"{name}.json", reports_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
