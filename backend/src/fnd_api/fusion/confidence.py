"""How much weight a verdict deserves.

In this version confidence is the calibrated probability of the more likely side, capped when the
input is too thin to judge well. The measured per-band accuracy described in the design replaces
it when evidence fusion arrives.
"""

from __future__ import annotations

from fnd_api.schemas import ConfidenceLevel

THIN_INPUT_CAP = 0.6
MIN_WORDS = 40
HIGH_FROM = 0.85
MEDIUM_FROM = 0.65


def compute(risk: float, *, partial_text: bool, word_count: int) -> tuple[float, ConfidenceLevel]:
    value = max(risk, 1.0 - risk)
    if partial_text or word_count < MIN_WORDS:
        value = min(value, THIN_INPUT_CAP)
    if value >= HIGH_FROM:
        return value, ConfidenceLevel.HIGH
    if value >= MEDIUM_FROM:
        return value, ConfidenceLevel.MEDIUM
    return value, ConfidenceLevel.LOW
