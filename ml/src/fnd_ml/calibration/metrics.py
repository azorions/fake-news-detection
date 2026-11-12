"""Calibration metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def brier(y: ArrayLike, p: ArrayLike) -> float:
    """Mean squared error between the predicted probability and the 0/1 outcome. Lower is better."""
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p_arr - y_arr) ** 2))


def expected_calibration_error(y: ArrayLike, p: ArrayLike, n_bins: int = 15) -> float:
    """Average gap between predicted probability and observed frequency, over equal-width bins."""
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.minimum((p_arr * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for index in range(n_bins):
        in_bin = bins == index
        if in_bin.any():
            error += in_bin.mean() * abs(y_arr[in_bin].mean() - p_arr[in_bin].mean())
    return float(error)
