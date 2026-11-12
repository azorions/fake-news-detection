"""Choosing the two risk thresholds that split scores into three verdict bands.

``risk < t_low`` is Reliable, ``risk >= t_high`` is High Risk, everything between is Potentially
Misleading. The thresholds are never looser than ``(0.35, 0.65)``, so a middle band always exists and no
item is called High Risk while its own calibrated risk is close to a coin flip. They are tightened
further until the High Risk band reaches the precision target and the Reliable band reaches the
negative-predictive-value target on the data given (the calibration split).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

DEFAULT_LOW = 0.35
DEFAULT_HIGH = 0.65
MIN_BAND_ROWS = 30
_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


@dataclass(frozen=True)
class Thresholds:
    t_low: float
    t_high: float
    met_targets: bool
    precision_target: float
    npv_target: float

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def pick_thresholds(
    y: ArrayLike, p: ArrayLike, precision_target: float = 0.90, npv_target: float = 0.90
) -> Thresholds:
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)

    t_high: float | None = None
    for candidate in _GRID[_GRID >= DEFAULT_HIGH]:  # ascending: the first hit is the loosest
        band = p_arr >= candidate
        if band.sum() >= MIN_BAND_ROWS and y_arr[band].mean() >= precision_target:
            t_high = float(candidate)
            break

    t_low: float | None = None
    for candidate in _GRID[_GRID <= DEFAULT_LOW][::-1]:  # descending: the first hit is the loosest
        band = p_arr < candidate
        if band.sum() >= MIN_BAND_ROWS and (1.0 - y_arr[band]).mean() >= npv_target:
            t_low = float(candidate)
            break

    return Thresholds(
        t_low=DEFAULT_LOW if t_low is None else t_low,
        t_high=DEFAULT_HIGH if t_high is None else t_high,
        met_targets=t_low is not None and t_high is not None,
        precision_target=precision_target,
        npv_target=npv_target,
    )
