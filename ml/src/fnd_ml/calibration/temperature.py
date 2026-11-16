"""Temperature scaling: one number that makes a neural classifier's confidence honest.

A fine-tuned transformer is usually too sure of itself. Dividing its logit by a temperature ``T``
fitted on held-out data (the calib split) fixes the confidence without changing which class wins.
This module needs only numpy and scipy, so the API can apply the temperature without PyTorch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

from fnd_ml.calibration.isotonic import PROBABILITY_FLOOR

MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0


def fit_temperature(logits: ArrayLike, y: ArrayLike) -> float:
    """Return the ``T`` in [0.05, 20] that minimizes the negative log-likelihood of ``sigmoid(z / T)``.

    ``logits`` are margins (the logit of class 1 minus the logit of class 0), ``y`` the 0/1 labels.
    """
    z = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)

    def loss(log_t: float) -> float:
        scaled = z / np.exp(log_t)
        # log(1 + exp(s)) - y * s, written so that a large |s| cannot overflow
        return float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled))

    result = minimize_scalar(
        loss, bounds=(np.log(MIN_TEMPERATURE), np.log(MAX_TEMPERATURE)), method="bounded"
    )
    return float(np.clip(np.exp(result.x), MIN_TEMPERATURE, MAX_TEMPERATURE))


def sigmoid(logits: ArrayLike) -> NDArray[np.float64]:
    """Uncalibrated probability of class 1 for each margin: no temperature and no clipping."""
    z = np.asarray(logits, dtype=np.float64)
    probability: NDArray[np.float64] = np.exp(-np.logaddexp(0.0, -z))  # cannot overflow
    return probability


def apply_temperature(logits: ArrayLike, t: float) -> NDArray[np.float64]:
    """Calibrated probability of class 1 for each margin, clipped to [0.01, 0.99]."""
    probability = sigmoid(np.asarray(logits, dtype=np.float64) / t)
    clipped: NDArray[np.float64] = np.clip(probability, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    return clipped
