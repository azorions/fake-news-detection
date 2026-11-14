"""From a risk score to a verdict band."""

from __future__ import annotations

import math

from fnd_api.schemas import Verdict

_EPSILON = 1e-6


def logit(probability: float) -> float:
    """Log-odds of ``probability``, finite at 0 and 1."""
    clipped = min(max(probability, _EPSILON), 1.0 - _EPSILON)
    return math.log(clipped / (1.0 - clipped))


def band(risk: float, t_low: float, t_high: float) -> Verdict:
    """Below ``t_low`` reliable, from ``t_high`` up high risk, in between potentially misleading."""
    if risk >= t_high:
        return Verdict.HIGH_RISK
    if risk >= t_low:
        return Verdict.MISLEADING
    return Verdict.RELIABLE
