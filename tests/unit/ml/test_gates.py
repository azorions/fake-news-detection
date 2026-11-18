from pathlib import Path
from typing import Any

import pytest

from fnd_ml.eval.gates import GateResult, evaluate

GATES = Path(__file__).parents[3] / "ml" / "configs" / "eval_gates.yaml"
TRANSFORMER_GATES = {
    "transformer_gain",
    "transformer_external",
    "calibration_transformer",
    "onnx_parity",
    "int8_quantization",
    "latency_p95",
}


def _model(
    macro_f1: float, ece: float = 0.02, brier: float = 0.10, uncalibrated: float = 0.11
) -> dict[str, Any]:
    return {
        "overall": {"n": 1000, "macro_f1": macro_f1, "brier": brier, "ece": ece},
        "calibration": {"ece": ece, "brier": brier, "brier_uncalibrated": uncalibrated},
    }


def _results() -> dict[str, Any]:
    """Results in which every gate sits exactly on its threshold."""
    return {
        "data": {
            "shared_groups_between_splits": 0,
            "shared_near_duplicate_clusters_between_splits": 0,
        },
        "leak_probe": {"flagged": []},
        "in_distribution": {
            "baseline": _model(0.80),
            "xgb": _model(0.82),
            "transformer": _model(0.83, ece=0.05),
        },
        "external": {"mean_macro_f1": {"baseline": 0.6, "xgb": 0.7, "transformer": 0.6}},
        "transformer": {
            "variant": "int8",
            "max_length": 512,
            "parity": {"max_abs_probability_diff": 0.001, "n": 500},
            "val_metrics": {"macro_f1_fp32": 0.90, "macro_f1_int8": 0.89},
            "latency_ms": {
                "threads": 4,
                "runs": 100,
                "fp32": {"512": {"p50_ms": 640.0, "p95_ms": 700.0}},
                "int8": {"512": {"p50_ms": 410.0, "p95_ms": 500.0}},
            },
        },
    }


def _gates(results: dict[str, Any]) -> dict[str, GateResult]:
    return {gate.name: gate for gate in evaluate(results, GATES)}


def test_every_gate_passes_at_its_exact_threshold() -> None:
    gates = _gates(_results())

    assert set(gates) == TRANSFORMER_GATES | {
        "split_hygiene",
        "leak_probe",
        "baseline_floor",
        "calibration_baseline",
        "calibration_xgb",
    }
    assert [name for name, gate in gates.items() if not gate.passed] == []
    assert gates["baseline_floor"].measured == 0.80
    assert gates["transformer_gain"].measured == pytest.approx(0.03)
    assert gates["latency_p95"].measured == 500.0


def test_a_gate_reports_its_threshold_in_words() -> None:
    gate = _gates(_results())["transformer_gain"]

    assert "0.03" in gate.threshold
    assert "0.83" in gate.note and "0.8" in gate.note


def test_a_baseline_under_the_floor_fails() -> None:
    results = _results()
    results["in_distribution"]["baseline"] = _model(0.7999)

    assert not _gates(results)["baseline_floor"].passed


def test_a_gain_of_0_029_over_the_baseline_fails() -> None:
    results = _results()
    results["in_distribution"]["transformer"] = _model(0.829)

    gate = _gates(results)["transformer_gain"]

    assert not gate.passed
    assert gate.measured == pytest.approx(0.029)


def test_an_external_mean_below_the_baseline_fails() -> None:
    results = _results()
    results["external"]["mean_macro_f1"]["transformer"] = 0.5999

    gate = _gates(results)["transformer_external"]

    assert not gate.passed
    assert gate.measured == pytest.approx(-0.0001)


def test_an_ece_above_0_05_fails() -> None:
    results = _results()
    results["in_distribution"]["xgb"] = _model(0.82, ece=0.0501)

    gates = _gates(results)

    assert not gates["calibration_xgb"].passed
    assert gates["calibration_baseline"].passed


def test_calibration_that_does_not_lower_the_brier_score_fails() -> None:
    results = _results()
    results["in_distribution"]["baseline"] = _model(0.80, brier=0.11, uncalibrated=0.11)

    gate = _gates(results)["calibration_baseline"]

    assert not gate.passed
    assert "0.11" in gate.note


def test_a_parity_difference_above_0_001_fails() -> None:
    results = _results()
    results["transformer"]["parity"]["max_abs_probability_diff"] = 0.0011

    assert not _gates(results)["onnx_parity"].passed


def test_an_int8_drop_of_0_011_says_ship_fp32() -> None:
    results = _results()
    results["transformer"]["val_metrics"]["macro_f1_int8"] = 0.889

    gate = _gates(results)["int8_quantization"]

    assert "ship fp32" in gate.note
    assert not gate.passed  # int8 was shipped all the same


def test_an_int8_drop_of_0_011_passes_when_fp32_was_shipped() -> None:
    results = _results()
    results["transformer"]["val_metrics"]["macro_f1_int8"] = 0.889
    results["transformer"]["variant"] = "fp32"
    results["transformer"]["latency_ms"]["fp32"]["512"]["p95_ms"] = 480.0

    gates = _gates(results)

    assert gates["int8_quantization"].passed
    assert "ship fp32" in gates["int8_quantization"].note
    assert gates["latency_p95"].measured == 480.0  # the shipped file's latency is the one gated


def test_a_p95_above_500_ms_fails() -> None:
    results = _results()
    results["transformer"]["latency_ms"]["int8"]["512"]["p95_ms"] = 500.1

    assert not _gates(results)["latency_p95"].passed


def test_a_shared_publisher_group_fails_split_hygiene() -> None:
    results = _results()
    results["data"]["shared_groups_between_splits"] = 1

    assert not _gates(results)["split_hygiene"].passed


def test_a_source_name_among_the_top_features_fails_and_is_named() -> None:
    results = _results()
    results["leak_probe"] = {"flagged": ["reuters"]}

    gate = _gates(results)["leak_probe"]

    assert not gate.passed
    assert "reuters" in gate.note


def test_without_a_transformer_its_gates_fail_as_not_measured() -> None:
    results = _results()
    results["transformer"] = None
    del results["in_distribution"]["transformer"]
    del results["external"]["mean_macro_f1"]["transformer"]

    gates = _gates(results)

    for name in TRANSFORMER_GATES - {"calibration_transformer"}:
        assert not gates[name].passed
        assert gates[name].measured == "not measured"
    assert "calibration_transformer" not in gates
    assert gates["baseline_floor"].passed


def test_an_artifact_without_export_measurements_fails_those_gates() -> None:
    results = _results()
    results["transformer"] = {"variant": "fp32", "max_length": 512}

    gates = _gates(results)

    for name in ("onnx_parity", "int8_quantization", "latency_p95"):
        assert not gates[name].passed
        assert gates[name].measured == "not measured"
    assert gates["transformer_gain"].passed
