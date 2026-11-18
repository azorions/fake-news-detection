"""Writing evaluation reports as JSON (for tools) and Markdown (for people)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fnd_ml.eval.gates import GateResult, measured_text

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


# ------------------------------------------------------------ the model comparison (`fnd eval`)

_MODEL_TITLES = {"baseline": "Baseline", "xgb": "XGBoost", "transformer": "Transformer"}
_SET_TITLES = {"test": "the test split"}


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else str(value)


def _grid(
    headers: Sequence[str], rows: Sequence[Sequence[object]], text_columns: int = 1
) -> list[str]:
    """A markdown table; the first ``text_columns`` columns are left-aligned, the rest are numbers."""
    divider = "|" + "---|" * text_columns + "---:|" * (len(headers) - text_columns)
    body = ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return ["| " + " | ".join(headers) + " |", divider, *body]


def _numbers(metrics: Mapping[str, Any]) -> list[object]:
    return [metrics[key] for key in _METRICS]


def _gate_section(gates: Sequence[GateResult]) -> list[str]:
    lines = ["", "## Gates", ""]
    if not gates:
        return [*lines, "No gate was applied."]
    failed = [gate.name for gate in gates if not gate.passed]
    lines.append(
        f"{len(failed)} of {len(gates)} gates fail: " + _code_list(failed) + "."
        if failed
        else f"All {len(gates)} gates pass."
    )
    rows = [
        (
            f"`{gate.name}`",
            "pass" if gate.passed else "**FAIL**",
            measured_text(gate),
            gate.threshold,
            gate.note,
        )
        for gate in gates
    ]
    return [*lines, "", *_grid(("gate", "result", "measured", "limit", "note"), rows, 5)]


def _split_by(models: Mapping[str, Any], key: str, title: str) -> list[str]:
    """Macro-F1 of every model in each group (a dataset, a length bucket), with the group's rows."""
    first = next(iter(models.values()))[key]
    headers = ["model", *(f"{group} ({metrics['n']:,} rows)" for group, metrics in first.items())]
    rows = [
        [name, *(model[key][group]["macro_f1"] for group in first)]
        for name, model in models.items()
    ]
    return ["", f"### Macro-F1 by {title}", "", *_grid(headers, rows)]


def _in_distribution(models: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "## In distribution",
        "",
        "The grouped test split: publishers and near-duplicate clusters that no model saw in",
        "training. Calibrated probabilities; macro-F1 at 0.5.",
        "",
        *_grid(
            ("model", *_HEADERS), [[name, *_numbers(m["overall"])] for name, m in models.items()]
        ),
        *_split_by(models, "by_dataset", "dataset"),
        *_split_by(models, "by_length", "length"),
    ]
    compared = {name: model for name, model in models.items() if "overall_raw" in model}
    if compared:
        rows = [
            [
                name,
                model["overall_raw"]["ece"],
                model["overall"]["ece"],
                model["calibration"]["brier_uncalibrated"],
                model["calibration"]["brier"],
            ]
            for name, model in compared.items()
        ]
        headers = ("model", "ECE before", "ECE after", "Brier before", "Brier after")
        lines += ["", "### Before and after calibration", "", *_grid(headers, rows)]
    return lines


def _leave_one_out(folds_by_model: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Leave one dataset out",
        "",
        "Each model is trained again without one dataset (train, val and calib rows of the",
        "others) and scored on that dataset's test rows.",
        "",
    ]
    held_out = sorted({name for folds in folds_by_model.values() for name in folds})
    rows = [
        [dataset, model, folds[dataset]["train_rows"], *_numbers(folds[dataset])]
        for dataset in held_out
        for model, folds in folds_by_model.items()
        if dataset in folds
    ]
    if not rows:
        return [*lines, "Not run."]
    return [*lines, *_grid(("held out", "model", "training rows", *_HEADERS), rows, 2)]


def _external(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "## External sets",
        "",
        "Never used for training, calibration or thresholds.",
        "",
    ]
    rows = [
        [name, model, *_numbers(metrics)]
        for name, models in result["sets"].items()
        for model, metrics in models.items()
    ]
    lines += _grid(("set", "model", *_HEADERS), rows, 2)
    means = ", ".join(f"{model} {mean:g}" for model, mean in result["mean_macro_f1"].items())
    return [*lines, "", f"Mean macro-F1 over the sets: {means}."]


def _figures(figures: Mapping[str, Mapping[str, str]]) -> list[str]:
    lines = [
        "",
        "## Reliability diagrams",
        "",
        "15 bins. A point is a bin: its mean predicted probability against the share of its rows",
        "that are unreliable. A calibrated model lies on the diagonal.",
    ]
    for model, by_set in figures.items():
        for scored, path in by_set.items():
            where = _SET_TITLES.get(scored, scored)
            lines += ["", f"![{_MODEL_TITLES.get(model, model)} on {where}]({path})"]
    return lines


def _export_section(artifact: Mapping[str, Any]) -> list[str]:
    latency, parity, val = (artifact.get(key) for key in ("latency_ms", "parity", "val_metrics"))
    if not (latency and parity and val):
        return []
    lines = [
        "",
        "## Transformer export",
        "",
        f"Shipped file: **{artifact['variant']}** of `{artifact['base_model']}`, version "
        f"`{artifact['version']}`. fp32 against PyTorch: largest probability difference "
        f"{parity['max_abs_probability_diff']:g} on {int(parity['n'])} val rows. Val macro-F1 on "
        f"{val['sample_rows']:,} rows: fp32 {val['macro_f1_fp32']:g}, int8 {val['macro_f1_int8']:g}.",
        "",
        f"Latency of one article on the CPU, {latency['threads']} threads, {latency['runs']} runs:",
        "",
    ]
    rows = [
        [name, tokens, timing["p50_ms"], timing["p95_ms"]]
        for name in ("fp32", "int8")
        for tokens, timing in latency.get(name, {}).items()
    ]
    lines += _grid(("file", "tokens", "p50 ms", "p95 ms"), rows, 1)
    for exception in artifact.get("gate_exceptions", []):
        lines += ["", f"Gate exception recorded at export: {exception}."]
    return lines


def _touches(touches: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["", "## Test split", "", "Scored once per model version (`test_touch_log.json`).", ""]
    for touch in touches:
        how = "scored by this run" if touch["scored"] else "scores read from an earlier run"
        line = f"- {touch['model']} `{touch['version']}`: {how}."
        if touch["reason"]:
            line += f" Scored more than once; the reason given: {touch['reason']}"
        lines.append(line)
    return lines


def render_eval_markdown(results: Mapping[str, Any], gates: Sequence[GateResult]) -> str:
    """The comparison report: models, gates, one table per protocol, diagrams, test-split log."""
    models = [
        (name, f"`{model['version']}`", model["description"])
        for name, model in results["models"].items()
    ]
    lines = [
        f"# Model comparison: {results['name']}",
        "",
        f"Data manifest sha256: `{results['data_manifest_sha256']}`",
        "",
        *_grid(("model", "version", "what it is"), models, 3),
        *_gate_section(gates),
        *_in_distribution(results["in_distribution"]),
        *_leave_one_out(results["leave_one_dataset_out"]),
        *_external(results["external"]),
        *_figures(results.get("figures") or {}),
        *_export_section(results.get("transformer") or {}),
        *_touches(results.get("test_touches") or []),
    ]
    if results.get("notes"):
        lines += ["", "## Notes", "", *(f"- {note}" for note in results["notes"])]
    return "\n".join(lines) + "\n"


def write_eval_report(
    results: Mapping[str, Any], gates: Sequence[GateResult], out: Path
) -> tuple[Path, Path]:
    """Write ``<out>.json`` (results and gates) and ``<out>.md``; ``out`` has no extension."""
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out.parent / f"{out.name}.json", out.parent / f"{out.name}.md"
    document = {**results, "gates": [asdict(gate) for gate in gates]}
    json_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_eval_markdown(results, gates), encoding="utf-8")
    return json_path, md_path


def write_report(report: dict[str, Any], reports_dir: Path, name: str) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = reports_dir / f"{name}.json", reports_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
