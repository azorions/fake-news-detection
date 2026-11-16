"""``fnd calibrate``: fit the temperature and the verdict thresholds of a fine-tuned run.

Both come from the calib split only. The val split chose the model, the test split measures it;
neither may also set its confidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fnd_ml.calibration.metrics import brier, expected_calibration_error
from fnd_ml.calibration.temperature import apply_temperature, fit_temperature, sigmoid
from fnd_ml.calibration.thresholds import pick_thresholds
from fnd_ml.data.build import DEFAULT_LOCK, ManifestMismatch, assert_frozen
from fnd_ml.models.transformer import BEST_DIR, CALIBRATION_NAME, SUMMARY_NAME, predict_logits

THRESHOLDS_NAME = "thresholds.json"
CALIB_LOGITS_NAME = "calib_logits.npy"


def calibrate_run(
    run_dir: Path,
    processed_dir: Path,
    *,
    lock_path: Path = DEFAULT_LOCK,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Write ``calibration.json`` and ``thresholds.json`` into ``run_dir`` and return both.

    The run must have been trained on the frozen build: a temperature fitted on other rows than
    the model's own calib split would not describe it.
    """
    data_sha = assert_frozen(processed_dir, lock_path)
    summary = json.loads((run_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    trained_on = str(summary.get("data_manifest_sha256"))
    if trained_on != data_sha:
        raise ManifestMismatch(
            f"{run_dir} was trained on data manifest {trained_on[:12]}, but the frozen build is "
            f"{data_sha[:12]}. Train the model on the frozen build before calibrating it."
        )

    calib = pd.read_parquet(
        processed_dir / "splits" / "calib.parquet", columns=["title", "text", "label"]
    )
    labels = calib["label"].to_numpy()
    logits = predict_logits(run_dir / BEST_DIR, calib, batch_size=batch_size)
    np.save(run_dir / CALIB_LOGITS_NAME, logits)

    temperature = fit_temperature(logits, labels)
    before, after = sigmoid(logits), apply_temperature(logits, temperature)
    calibration = {
        "method": "temperature",
        "temperature": round(temperature, 6),
        "ece_before": round(expected_calibration_error(labels, before), 4),
        "ece_after": round(expected_calibration_error(labels, after), 4),
        "brier_before": round(brier(labels, before), 4),
        "brier_after": round(brier(labels, after), 4),
        "rows": len(calib),
        "data_manifest_sha256": data_sha,
    }
    thresholds = pick_thresholds(labels, after).to_dict()
    for name, document in ((CALIBRATION_NAME, calibration), (THRESHOLDS_NAME, thresholds)):
        (run_dir / name).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return {"calibration": calibration, "thresholds": thresholds}
