"""Classification metrics for a binary "unreliable" probability."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from fnd_ml.calibration.metrics import brier, expected_calibration_error


def classification_report(y: ArrayLike, p: ArrayLike, threshold: float = 0.5) -> dict[str, Any]:
    """Headline metrics. ``roc_auc`` and ``pr_auc`` are ``None`` when only one class is present."""
    y_arr, p_arr = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    both_classes = len(np.unique(y_arr)) == 2
    predictions = (p_arr >= threshold).astype(int)
    return {
        "n": len(y_arr),
        "unreliable_share": round(float(y_arr.mean()), 4),
        "macro_f1": round(float(f1_score(y_arr, predictions, average="macro")), 4),
        "roc_auc": round(float(roc_auc_score(y_arr, p_arr)), 4) if both_classes else None,
        "pr_auc": round(float(average_precision_score(y_arr, p_arr)), 4) if both_classes else None,
        "brier": round(brier(y_arr, p_arr), 4),
        "ece": round(expected_calibration_error(y_arr, p_arr), 4),
    }
