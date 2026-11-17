"""The quality gates of the design (section 4): limits from ``eval_gates.yaml``, one verdict each.

``evaluate`` reads the results that ``fnd eval`` collects:

* ``data``: the split-hygiene counts of the data manifest;
* ``leak_probe``: the label probe of the baseline's training report, or ``None``;
* ``in_distribution``: per model ``overall`` and ``calibration`` (see ``protocols.in_distribution``);
* ``external``: ``mean_macro_f1`` per model;
* ``transformer``: the served artifact's ``metadata.json``, or ``None`` when none is registered.

A gate that cannot be measured fails, and says what is missing. Differences of rounded metrics
are rounded again before they are compared, so a value exactly on its limit passes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

NOT_MEASURED = "not measured"
NO_TRANSFORMER = "no transformer artifact is registered"
NO_EXPORT = "the artifact's metadata has no such measurement"
_EPS = 1e-9


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    measured: float | str
    threshold: str
    note: str = ""


def load_gates(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _missing(name: str, threshold: str, note: str) -> GateResult:
    return GateResult(name, False, NOT_MEASURED, threshold, note)


def _split_hygiene(data: Mapping[str, Any], limits: Mapping[str, Any]) -> GateResult:
    name = "split_hygiene"
    most = (limits["max_shared_groups"], limits["max_shared_near_duplicate_clusters"])
    threshold = (
        f"{most[0]} publisher groups and {most[1]} near-duplicate clusters shared between splits"
    )
    groups = data.get("shared_groups_between_splits")
    clusters = data.get("shared_near_duplicate_clusters_between_splits")
    if groups is None or clusters is None:
        return _missing(name, threshold, "the data manifest does not count them")
    passed = groups <= most[0] and clusters <= most[1]
    return GateResult(name, passed, f"{groups} groups, {clusters} clusters", threshold)


def _leak_probe(probe: Mapping[str, Any] | None, limits: Mapping[str, Any]) -> GateResult:
    name = "leak_probe"
    threshold = "no source identifier among the top 50 baseline features"
    if probe is None:
        return _missing(name, threshold, "no training report for the registered baseline")
    flagged = list(probe["flagged"])
    passed = len(flagged) <= limits["max_flagged_features"]
    return GateResult(name, passed, f"{len(flagged)} flagged", threshold, ", ".join(flagged))


def _baseline_floor(models: Mapping[str, Any], limits: Mapping[str, Any]) -> GateResult:
    name = "baseline_floor"
    floor = float(limits["min_macro_f1"])
    threshold = f"macro-F1 >= {floor:g} on the grouped test split"
    if "baseline" not in models:
        return _missing(name, threshold, "no baseline artifact is registered")
    measured = float(models["baseline"]["overall"]["macro_f1"])
    return GateResult(name, measured >= floor - _EPS, measured, threshold)


def _gain(name: str, scores: Mapping[str, float], least: float, threshold: str) -> GateResult:
    """The transformer's score minus the baseline's must be at least ``least``."""
    if "transformer" not in scores or "baseline" not in scores:
        missing = NO_TRANSFORMER if "transformer" not in scores else "no baseline was scored"
        return _missing(name, threshold, missing)
    ours, theirs = float(scores["transformer"]), float(scores["baseline"])
    gain = round(ours - theirs, 6)
    return GateResult(name, gain >= least - _EPS, gain, threshold, f"{ours:g} against {theirs:g}")


def _calibration(name: str, model: Mapping[str, Any], limits: Mapping[str, Any]) -> GateResult:
    gate = f"calibration_{name}"
    most = float(limits["max_ece"])
    threshold = f"ECE <= {most:g} and Brier below the uncalibrated score"
    figures = model.get("calibration")
    if figures is None or figures.get("brier_uncalibrated") is None:
        return _missing(gate, threshold, "the uncalibrated scores were not kept")
    ece, after, before = (float(figures[key]) for key in ("ece", "brier", "brier_uncalibrated"))
    passed = ece <= most + _EPS and after < before
    return GateResult(
        gate, passed, ece, threshold, f"Brier {after:g} against {before:g} uncalibrated"
    )


def _parity(artifact: Mapping[str, Any], limits: Mapping[str, Any]) -> GateResult:
    name = "onnx_parity"
    most = float(limits["max_abs_probability_diff"])
    threshold = f"fp32 max probability difference <= {most:g}"
    check = artifact.get("parity")
    if check is None:
        return _missing(name, threshold, NO_EXPORT)
    measured = float(check["max_abs_probability_diff"])
    return GateResult(
        name, measured <= most + 1e-12, measured, threshold, f"{int(check['n'])} val rows"
    )


def _int8(artifact: Mapping[str, Any], limits: Mapping[str, Any]) -> GateResult:
    name = "int8_quantization"
    most = float(limits["max_int8_macro_f1_drop"])
    threshold = f"int8 macro-F1 drop <= {most:g} on val, else ship fp32"
    val = artifact.get("val_metrics") or {}
    if "macro_f1_fp32" not in val or "macro_f1_int8" not in val:
        return _missing(name, threshold, NO_EXPORT)
    drop = round(float(val["macro_f1_fp32"]) - float(val["macro_f1_int8"]), 6)
    variant = str(artifact["variant"])
    if drop <= most + _EPS:
        return GateResult(name, True, drop, threshold, f"shipped: {variant}")
    note = f"int8 loses more than {most:g}: ship fp32 (shipped: {variant})"
    return GateResult(name, variant == "fp32", drop, threshold, note)


def _latency(artifact: Mapping[str, Any], limits: Mapping[str, Any]) -> GateResult:
    name = "latency_p95"
    most = float(limits["max_p95_ms"])
    threshold = f"p95 <= {most:g} ms on the CPU at 512 tokens"
    latency = artifact.get("latency_ms") or {}
    variant, window = str(artifact["variant"]), artifact["max_length"]
    timing = (latency.get(variant) or {}).get(str(window))
    if timing is None:
        return _missing(name, threshold, NO_EXPORT)
    measured = float(timing["p95_ms"])
    note = f"{variant}, {latency.get('threads', '?')} threads, {window} tokens"
    return GateResult(name, measured <= most + _EPS, measured, threshold, note)


def _export(artifact: Mapping[str, Any] | None, limits: Mapping[str, Any]) -> list[GateResult]:
    checks = (_parity, _int8, _latency)
    if artifact is None:
        placeholder: Mapping[str, Any] = {"variant": "fp32", "max_length": 512}
        return [
            _missing(gate.name, gate.threshold, NO_TRANSFORMER)
            for gate in (check(placeholder, limits) for check in checks)
        ]
    return [check(artifact, limits) for check in checks]


def evaluate(results: Mapping[str, Any], gates_path: Path) -> list[GateResult]:
    """Every gate of the design with its measured value; a gate without a measurement fails."""
    limits = load_gates(gates_path)
    models: Mapping[str, Any] = results["in_distribution"]
    versus = limits["transformer_vs_baseline"]
    least = float(versus["min_gain_macro_f1"])
    in_distribution = {name: model["overall"]["macro_f1"] for name, model in models.items()}
    return [
        _split_hygiene(results.get("data") or {}, limits["split_hygiene"]),
        _leak_probe(results.get("leak_probe"), limits["leak_probe"]),
        _baseline_floor(models, limits["baseline_floor"]),
        _gain(
            "transformer_gain",
            in_distribution,
            least,
            f">= +{least:g} macro-F1 over the baseline, in distribution",
        ),
        _gain(
            "transformer_external",
            results["external"]["mean_macro_f1"],
            float(versus["min_external_mean_gain"]),
            "mean macro-F1 on the external sets not below the baseline's",
        ),
        *(_calibration(name, model, limits["calibration"]) for name, model in models.items()),
        *_export(results.get("transformer"), {**limits["onnx_parity"], **limits["latency"]}),
    ]


def measured_text(gate: GateResult) -> str:
    return f"{gate.measured:g}" if isinstance(gate.measured, float) else str(gate.measured)


def format_gates(gates: Sequence[GateResult]) -> str:
    """The gate table as plain text, for the terminal."""
    rows = [("gate", "result", "measured", "threshold")]
    rows += [
        (
            gate.name,
            "pass" if gate.passed else "FAIL",
            measured_text(gate),
            gate.threshold + (f"  ({gate.note})" if gate.note else ""),
        )
        for gate in gates
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(3)]
    return "\n".join(
        "  ".join([*(cell.ljust(width) for cell, width in zip(row, widths, strict=False)), row[3]])
        for row in rows
    )
