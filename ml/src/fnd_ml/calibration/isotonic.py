"""Isotonic calibration kept as two arrays (the knots of the fitted step function), so a calibrated
model can be saved without pickling a scikit-learn object."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression

PROBABILITY_FLOOR = 0.01  # calibrated probabilities are clipped to [0.01, 0.99]

Knots = tuple[NDArray[np.float64], NDArray[np.float64]]


def fit_isotonic(raw: ArrayLike, y: ArrayLike) -> Knots:
    """Fit raw probabilities to labels; returns the knots ``(x, y)`` to interpolate between."""
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(np.asarray(raw, dtype=float), np.asarray(y, dtype=float))
    return (
        np.asarray(isotonic.X_thresholds_, dtype=np.float64),
        np.asarray(isotonic.y_thresholds_, dtype=np.float64),
    )


def apply_isotonic(raw: ArrayLike, knots: Knots | None) -> NDArray[np.float64]:
    """Calibrated probabilities, clipped; without knots the raw values are only clipped."""
    values = np.asarray(raw, dtype=float)
    if knots is not None:
        values = np.interp(values, *knots)
    return np.clip(values, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
