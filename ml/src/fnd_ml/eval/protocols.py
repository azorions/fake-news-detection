"""The evaluation protocols of M2 over the common ``Scorer`` interface, and the test-split log.

* in distribution: the grouped test split, overall, by dataset and by length;
* leave one dataset out: train on the other datasets, score the held-out one's test rows;
* external: sets that no model has seen in any form.

Nothing here knows a model family: the caller passes scorers and a ``Fit``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fnd_ml.calibration.metrics import brier, expected_calibration_error
from fnd_ml.eval.metrics import classification_report, report_by_group
from fnd_ml.models.scorer import Fit, Scorer

TRAINING_ROLES = ("train", "val", "calib")


def in_distribution(
    scorers: Mapping[str, Scorer],
    test: pd.DataFrame,
    *,
    raw: Mapping[str, Scorer] | None = None,
) -> dict[str, Any]:
    """Per model: ``overall``, ``by_dataset`` and ``by_length`` on the test split.

    ``raw`` holds the same models' uncalibrated scorers. With it a model also gets ``overall_raw``
    and ``calibration``: ECE and Brier at six decimals, and the Brier score before calibration.
    """
    labels = test["label"].to_numpy(dtype=int)
    results: dict[str, Any] = {}
    for name, scorer in scorers.items():
        p = scorer(test)
        entry: dict[str, Any] = {
            "overall": classification_report(labels, p),
            "by_dataset": report_by_group(test, p, "dataset"),
            "by_length": report_by_group(test, p, "length_bucket"),
        }
        if raw is not None and name in raw:
            before = raw[name](test)
            entry["overall_raw"] = classification_report(labels, before)
            entry["calibration"] = {
                "ece": round(expected_calibration_error(labels, p), 6),
                "brier": round(brier(labels, p), 6),
                "brier_uncalibrated": round(brier(labels, before), 6),
            }
        results[name] = entry
    return results


def leave_one_dataset_out(
    fit: Fit, corpus: pd.DataFrame, *, datasets: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Per held-out dataset: a model fitted on the others, scored on the held-out test rows.

    ``corpus`` holds every split (columns ``dataset``, ``split``, ``label`` and whatever ``fit``
    reads). The model gets the train, val and calib rows of the other datasets in their usual
    roles and never a row of the held-out one. ``datasets`` limits the folds that are run.
    """
    folds = sorted(corpus["dataset"].unique()) if datasets is None else datasets
    results: dict[str, dict[str, Any]] = {}
    for held_out in folds:
        inside = corpus["dataset"] == held_out
        train, val, calib = (corpus[~inside & (corpus["split"] == role)] for role in TRAINING_ROLES)
        test = corpus[inside & (corpus["split"] == "test")]
        scorer = fit(train, val, calib)
        report = classification_report(test["label"], scorer(test))
        results[str(held_out)] = {**report, "train_rows": len(train)}
    return results


def external(scorers: Mapping[str, Scorer], sets: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """``sets``: per set and model the metrics; ``mean_macro_f1``: per model, over the sets."""
    by_set = {
        name: {
            model: classification_report(frame["label"], scorer(frame))
            for model, scorer in scorers.items()
        }
        for name, frame in sets.items()
    }
    mean = {
        model: round(float(np.mean([models[model]["macro_f1"] for models in by_set.values()])), 4)
        for model in (scorers if by_set else ())
    }
    return {"sets": by_set, "mean_macro_f1": mean}


class RetestError(RuntimeError):
    """A model version would score the test split again without a stated reason."""


class TouchLog:
    """Which model versions have scored the test split: when, for which report, and why again.

    The test split is touched once per model version. A second scoring is possible, but only
    with a reason, and the reason is kept here and shown in the report.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> dict[str, list[dict[str, Any]]]:
        if not self.path.exists():
            return {}
        log: dict[str, list[dict[str, Any]]] = json.loads(self.path.read_text(encoding="utf-8"))
        return log

    def entries(self, model: str, version: str) -> list[dict[str, Any]]:
        return self._read().get(f"{model}/{version}", [])

    def check(self, model: str, version: str, reason: str | None) -> None:
        """Raise ``RetestError`` when ``version`` has scored the test split and no reason is given."""
        earlier = self.entries(model, version)
        if earlier and not (reason and reason.strip()):
            first = earlier[0]
            raise RetestError(
                f"{model} {version} has already scored the test split ({first['at']}, report "
                f"{first['report']}). The test split is touched once per model version. To score "
                'it again, give the reason with --allow-retest "..."; it is written into the report.'
            )

    def record(self, model: str, version: str, *, report: str, reason: str | None) -> None:
        log = self._read()
        entry = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "report": report,
            "reason": reason,
        }
        log.setdefault(f"{model}/{version}", []).append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
