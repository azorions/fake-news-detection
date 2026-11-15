"""Writing evaluation reports as JSON (for tools) and Markdown (for people)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_METRICS = ("n", "unreliable_share", "macro_f1", "roc_auc", "pr_auc", "brier", "ece")
_HEADERS = ("rows", "unreliable share", "macro-F1", "ROC-AUC", "PR-AUC", "Brier", "ECE")
_TITLES = {"baseline": "Baseline", "xgb": "XGBoost"}


def _row(label: str, metrics: dict[str, Any]) -> str:
    cells = ["-" if metrics[key] is None else str(metrics[key]) for key in _METRICS]
    return f"| {label} | " + " | ".join(cells) + " |"


def _table(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    header = "| | " + " | ".join(_HEADERS) + " |"
    divider = "|---|" + "---:|" * len(_HEADERS)
    return [header, divider, *(_row(label, metrics) for label, metrics in rows)]


def _description(report: dict[str, Any]) -> str:
    config = report["config"]
    if report["model"] == "xgb":
        return (
            "XGBoost over 12 stylometric features and a "
            f"{config['svd_components']}-component SVD of the hashed TF-IDF, isotonic calibration."
        )
    grams = "word 1-2-grams"
    if config.get("char_ngrams"):
        grams += " and character 3-5-grams"
    return f"TF-IDF (hashed {grams}) + logistic regression, isotonic calibration."


def _code_list(items: list[str]) -> str:
    return ", ".join(f"`{item}`" for item in items)


def _leak_probes(report: dict[str, Any]) -> list[str]:
    label, origin = report["leak_probe"], report["origin_probe"]
    checked = sum(len(features) for features in report["top_features"].values())
    lines = ["", "## Leak probes", ""]
    if label["flagged"]:
        lines.append(
            f"Label probe: **failed.** These of the {checked} strongest features name a source: "
            + _code_list(label["flagged"])
            + "."
        )
    else:
        lines.append(f"Label probe: none of the {checked} strongest features names a source.")
    if label["unnamed_buckets"]:
        lines.append(
            f"{label['unnamed_buckets']} of them are hash buckets that no n-gram of the naming "
            "sample falls into, so they could not be checked."
        )
    lines.append("")
    if origin is None:
        lines.append("Origin probe: not run, the splits hold a single dataset.")
    else:
        within = ", ".join(
            f"within label {name}: {'-' if probe is None else probe['accuracy']}"
            for name, probe in sorted(origin["per_label"].items())
        )
        lines.append(
            "Origin probe: a linear model names the dataset of a scrubbed val text with accuracy "
            f"{origin['accuracy']} (always answering the largest dataset gives "
            f"{origin['majority_baseline']}); {within}. Reported, not gated."
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    thresholds = report["thresholds"]
    scored = "test" in metrics
    lines = [
        f"# {_TITLES[report['model']]} model report: {report['version']}",
        "",
        _description(report),
        f"Data manifest sha256: `{report['data_manifest_sha256']}`",
        "",
    ]
    if not scored:
        lines += [
            "**The test split was not scored and no artifact was saved:** one of the strongest "
            "features names a source (see Leak probes). Add it to the scrub list, rebuild and "
            "freeze the data, then train again.",
            "",
        ]
    lines += [
        "## Metrics by split",
        "",
        "The calibrator and thresholds are fit on the calib split, so its calibrated numbers are",
        "in-sample. The test split is the honest estimate.",
        "",
    ]
    by_split = []
    for split in ("val", "calib", "test") if scored else ("val", "calib"):
        by_split.append((f"{split} (raw)", metrics[split]["raw"]))
        by_split.append((f"{split} (calibrated)", metrics[split]["calibrated"]))
    lines += _table(by_split)
    if scored:
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
    ]
    if scored:
        lines += ["", "| band on test | share of rows | unreliable rate |", "|---|---:|---:|"]
        for band, stats in report["bands_on_test"].items():
            lines.append(f"| {band} | {stats['share']} | {stats['unreliable_rate']} |")
    if "leak_probe" in report:
        lines += _leak_probes(report)
    if "top_features" in report:
        lines += ["", "## Strongest features", ""]
        for side in ("unreliable", "reliable"):
            lines += [f"**Toward {side}:** " + _code_list(report["top_features"][side]), ""]
    if "feature_importance" in report:
        lines += [
            "",
            "## Feature importance",
            "",
            f"{report['trees']} trees after early stopping on val. Mean gain of the splits that "
            "use a feature; `svd_N` is the N-th SVD component of the TF-IDF.",
            "",
            "| feature | gain |",
            "|---|---:|",
        ]
        lines += [f"| `{row['feature']}` | {row['gain']} |" for row in report["feature_importance"]]
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], reports_dir: Path, name: str) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = reports_dir / f"{name}.json", reports_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
