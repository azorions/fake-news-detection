from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from fnd_ml.eval.gates import GateResult
from fnd_ml.eval.metrics import classification_report
from fnd_ml.eval.plots import reliability_diagram
from fnd_ml.eval.protocols import (
    RetestError,
    TouchLog,
    external,
    in_distribution,
    leave_one_dataset_out,
)
from fnd_ml.eval.report import render_eval_markdown
from fnd_ml.models.scorer import Scorer
from tests.fixtures.synthetic import make_corpus

DATASETS = np.array(["alpha", "beta", "gamma"])
SPLITS = np.array(["train", "train", "train", "val", "calib", "test"])


def _corpus(n: int = 360) -> pd.DataFrame:
    """Three datasets, each with both labels in every split; ``p`` is a noisy score of the label."""
    frame = make_corpus(n, seed=3)
    index = np.arange(n)
    noise = np.random.default_rng(4).uniform(-0.45, 0.45, n)
    return frame.assign(
        dataset=DATASETS[index % 3],
        split=SPLITS[(index // 6) % 6],
        length_bucket=np.where(index % 5 == 0, "article", "headline"),
        p=np.clip(0.3 + 0.4 * frame["label"].to_numpy() + noise, 0.01, 0.99),
    )


def _column(name: str) -> Scorer:
    def score(frame: pd.DataFrame) -> NDArray[np.float64]:
        return frame[name].to_numpy(dtype=np.float64)

    return score


def test_in_distribution_reports_each_model_overall_by_dataset_and_by_length() -> None:
    test = _corpus()
    test = test[test["split"] == "test"].assign(q=0.5)

    result = in_distribution({"good": _column("p"), "blind": _column("q")}, test)

    assert set(result) == {"good", "blind"}
    assert result["good"]["overall"] == classification_report(test["label"], test["p"])
    assert set(result["good"]["by_dataset"]) == {"alpha", "beta", "gamma"}
    assert set(result["good"]["by_length"]) == {"article", "headline"}
    assert result["good"]["overall"]["macro_f1"] > result["blind"]["overall"]["macro_f1"]


def test_rows_by_dataset_and_by_length_add_up_to_the_total() -> None:
    test = _corpus()
    test = test[test["split"] == "test"]

    result = in_distribution({"good": _column("p")}, test)["good"]

    assert result["overall"]["n"] == len(test)
    assert sum(group["n"] for group in result["by_dataset"].values()) == len(test)
    assert sum(group["n"] for group in result["by_length"].values()) == len(test)


def test_in_distribution_compares_with_the_uncalibrated_scores_when_given() -> None:
    test = _corpus()
    test = test[test["split"] == "test"]
    overconfident = test.assign(r=np.where(test["p"] > 0.5, 0.99, 0.01))

    result = in_distribution({"good": _column("p")}, overconfident, raw={"good": _column("r")})

    calibration = result["good"]["calibration"]
    assert result["good"]["overall_raw"] == classification_report(test["label"], overconfident["r"])
    assert calibration["brier"] < calibration["brier_uncalibrated"]
    assert calibration["ece"] == pytest.approx(result["good"]["overall"]["ece"], abs=1e-4)


class _RecordingFit:
    """A ``Fit`` that notes what it was trained on and what its scorers were asked to score."""

    def __init__(self) -> None:
        self.fitted: list[dict[str, pd.DataFrame]] = []
        self.scored: list[pd.DataFrame] = []

    def __call__(self, train: pd.DataFrame, val: pd.DataFrame, calib: pd.DataFrame) -> Scorer:
        self.fitted.append({"train": train, "val": val, "calib": calib})

        def score(frame: pd.DataFrame) -> NDArray[np.float64]:
            self.scored.append(frame)
            return frame["p"].to_numpy(dtype=np.float64)

        return score


def test_leave_one_dataset_out_never_trains_on_the_held_out_dataset() -> None:
    corpus = _corpus()
    fit = _RecordingFit()

    result = leave_one_dataset_out(fit, corpus)

    assert list(result) == ["alpha", "beta", "gamma"]
    for held_out, frames, scored in zip(result, fit.fitted, fit.scored, strict=True):
        for role, frame in frames.items():
            assert held_out not in set(frame["dataset"]), role
            assert set(frame["split"]) == {role}
            assert set(frame["dataset"]) == set(DATASETS) - {held_out}
        assert set(scored["dataset"]) == {held_out}
        assert set(scored["split"]) == {"test"}
        assert result[held_out]["n"] == len(scored)
        assert result[held_out]["train_rows"] == len(frames["train"])
        assert result[held_out]["macro_f1"] > 0.5


def test_leave_one_dataset_out_can_run_a_single_fold() -> None:
    fit = _RecordingFit()

    result = leave_one_dataset_out(fit, _corpus(), datasets=["beta"])

    assert list(result) == ["beta"]
    assert len(fit.fitted) == 1


def test_external_scores_every_set_with_every_model_and_averages_macro_f1() -> None:
    corpus = _corpus().assign(q=0.5)
    sets = {"one": corpus.iloc[:120], "two": corpus.iloc[120:]}

    result = external({"good": _column("p"), "blind": _column("q")}, sets)

    assert set(result["sets"]) == {"one", "two"}
    assert result["sets"]["one"]["good"] == classification_report(
        sets["one"]["label"], sets["one"]["p"]
    )
    good = [result["sets"][name]["good"]["macro_f1"] for name in ("one", "two")]
    assert result["mean_macro_f1"]["good"] == pytest.approx(sum(good) / 2, abs=1e-4)
    assert result["mean_macro_f1"]["good"] > result["mean_macro_f1"]["blind"]


def test_a_second_scoring_of_the_same_version_needs_a_reason(tmp_path: Path) -> None:
    log = TouchLog(tmp_path / "reports" / "test_touch_log.json")
    log.check("baseline", "v2-aaaa", None)
    log.record("baseline", "v2-aaaa", report="eval_v1", reason=None)

    with pytest.raises(RetestError, match="v2-aaaa"):
        TouchLog(log.path).check("baseline", "v2-aaaa", None)

    again = TouchLog(log.path)
    again.check("baseline", "v2-aaaa", "the score cache was deleted")
    again.record("baseline", "v2-aaaa", report="eval_v1", reason="the score cache was deleted")
    entries = TouchLog(log.path).entries("baseline", "v2-aaaa")
    assert [entry["reason"] for entry in entries] == [None, "the score cache was deleted"]
    assert all(entry["report"] == "eval_v1" and entry["at"] for entry in entries)


def test_another_version_of_the_model_is_a_first_scoring(tmp_path: Path) -> None:
    log = TouchLog(tmp_path / "test_touch_log.json")
    log.record("baseline", "v2-aaaa", report="eval_v1", reason=None)

    log.check("baseline", "v2-bbbb", None)
    log.check("xgb", "v2-aaaa", None)

    assert log.entries("baseline", "v2-bbbb") == []


def test_a_blank_reason_does_not_allow_a_retest(tmp_path: Path) -> None:
    log = TouchLog(tmp_path / "test_touch_log.json")
    log.record("baseline", "v2-aaaa", report="eval_v1", reason=None)

    with pytest.raises(RetestError):
        log.check("baseline", "v2-aaaa", "   ")


def test_reliability_diagram_writes_a_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    sharpened = np.clip(0.5 + 1.8 * (p - 0.5), 0.0, 1.0)
    path = tmp_path / "figures" / "reliability_model_test.png"

    reliability_diagram(y, sharpened, p, path, "model on the test split")

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 5000


def test_reliability_diagram_draws_scores_that_fill_one_bin_only(tmp_path: Path) -> None:
    y = np.array([0, 1] * 50)
    constant = np.full(100, 0.5)
    path = tmp_path / "reliability_flat.png"

    reliability_diagram(y, constant, constant, path, "a constant score")

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _metrics(macro_f1: float, n: int = 100) -> dict[str, Any]:
    return {
        "n": n,
        "unreliable_share": 0.4,
        "macro_f1": macro_f1,
        "roc_auc": 0.9,
        "pr_auc": 0.88,
        "brier": 0.1,
        "ece": 0.02,
    }


def _model(macro_f1: float) -> dict[str, Any]:
    return {
        "overall": _metrics(macro_f1),
        "overall_raw": _metrics(macro_f1 - 0.01),
        "calibration": {"ece": 0.02, "brier": 0.1, "brier_uncalibrated": 0.12},
        "by_dataset": {"alpha": _metrics(macro_f1, 60), "beta": _metrics(macro_f1, 40)},
        "by_length": {"article": _metrics(macro_f1, 70), "headline": _metrics(macro_f1, 30)},
    }


def _report() -> dict[str, Any]:
    return {
        "name": "eval_v1",
        "data_manifest_sha256": "ab" * 32,
        "models": {
            "baseline": {"version": "v2-aaaa", "description": "TF-IDF + logistic regression"},
            "transformer": {"version": "v1-bbbb", "description": "deberta, int8"},
        },
        "in_distribution": {"baseline": _model(0.85), "transformer": _model(0.91)},
        "leave_one_dataset_out": {
            "baseline": {"alpha": {**_metrics(0.61), "train_rows": 500}},
            "transformer": {"alpha": {**_metrics(0.66), "train_rows": 400}},
        },
        "external": {
            "sets": {"constraint": {"baseline": _metrics(0.55), "transformer": _metrics(0.58)}},
            "mean_macro_f1": {"baseline": 0.55, "transformer": 0.58},
        },
        "figures": {"transformer": {"test": "figures/reliability_transformer_test.png"}},
        "test_touches": [
            {"model": "baseline", "version": "v2-aaaa", "scored": True, "reason": "cache lost"},
            {"model": "transformer", "version": "v1-bbbb", "scored": False, "reason": None},
        ],
        "transformer": None,
        "notes": ["a note for the reader"],
    }


def _tables(markdown: str) -> dict[str, list[str]]:
    """The table rows under each second-level heading."""
    tables: dict[str, list[str]] = {}
    heading = ""
    for line in markdown.splitlines():
        if line.startswith("## "):
            heading = line[3:]
        elif line.startswith("|") and not line.startswith("|---"):
            tables.setdefault(heading, []).append(line)
    return tables


def test_the_markdown_has_a_table_per_protocol_and_the_gate_table() -> None:
    gates = [
        GateResult("baseline_floor", True, 0.85, "macro-F1 >= 0.8", ""),
        GateResult("latency_p95", False, "not measured", "p95 <= 500 ms", "no export"),
    ]

    tables = _tables(render_eval_markdown(_report(), gates))

    assert {"Gates", "In distribution", "Leave one dataset out", "External sets"} <= set(tables)
    assert any("baseline_floor" in row and "pass" in row for row in tables["Gates"])
    assert any("latency_p95" in row and "FAIL" in row for row in tables["Gates"])
    for protocol in ("In distribution", "Leave one dataset out", "External sets"):
        rows = "\n".join(tables[protocol])
        assert "baseline" in rows and "transformer" in rows, protocol
    assert any("0.91" in row for row in tables["In distribution"])
    assert any("0.66" in row for row in tables["Leave one dataset out"])
    assert any("0.58" in row for row in tables["External sets"])


def test_the_markdown_shows_diagrams_retest_reasons_and_notes() -> None:
    markdown = render_eval_markdown(_report(), [])

    assert "![" in markdown and "(figures/reliability_transformer_test.png)" in markdown
    assert "cache lost" in markdown
    assert "a note for the reader" in markdown
    assert "ab" * 32 in markdown


def test_the_markdown_groups_thousands_in_row_counts() -> None:
    report = _report()
    report["in_distribution"]["baseline"]["overall"] = _metrics(0.85, n=8704)
    report["leave_one_dataset_out"]["baseline"]["alpha"]["train_rows"] = 48638

    tables = _tables(render_eval_markdown(report, []))

    assert any("| baseline | 8,704 |" in row for row in tables["In distribution"])
    assert any("| 48,638 |" in row for row in tables["Leave one dataset out"])


def test_a_diagram_is_captioned_with_its_model_and_set() -> None:
    report = _report()
    report["figures"] = {"xgb": {"test": "figures/a.png", "constraint": "figures/b.png"}}

    markdown = render_eval_markdown(report, [])

    assert "![XGBoost on the test split](figures/a.png)" in markdown
    assert "![XGBoost on constraint](figures/b.png)" in markdown
