import math
import re

import pytest

from fnd_api.fusion import confidence, scorer, summary
from fnd_api.schemas import ConfidenceLevel, Verdict

T_LOW, T_HIGH = 0.35, 0.65


@pytest.mark.parametrize(
    ("risk", "verdict"),
    [
        (0.0, Verdict.RELIABLE),
        (0.3499, Verdict.RELIABLE),
        (0.35, Verdict.MISLEADING),  # t_low itself is already "potentially misleading"
        (0.5, Verdict.MISLEADING),
        (0.6499, Verdict.MISLEADING),
        (0.65, Verdict.HIGH_RISK),  # t_high itself is already "high risk"
        (1.0, Verdict.HIGH_RISK),
    ],
)
def test_band_boundaries(risk: float, verdict: Verdict) -> None:
    assert scorer.band(risk, T_LOW, T_HIGH) == verdict


def test_logit_is_finite_at_the_extremes() -> None:
    assert scorer.logit(0.5) == 0.0
    assert scorer.logit(0.9) == pytest.approx(math.log(9))
    assert math.isfinite(scorer.logit(0.0))
    assert math.isfinite(scorer.logit(1.0))
    assert scorer.logit(0.0) < 0 < scorer.logit(1.0)


@pytest.mark.parametrize(
    ("risk", "partial", "words", "expected"),
    [
        (0.95, True, 500, (0.6, ConfidenceLevel.LOW)),
        (0.95, False, 39, (0.6, ConfidenceLevel.LOW)),
        (0.95, False, 40, (0.95, ConfidenceLevel.HIGH)),
        (0.05, False, 500, (0.95, ConfidenceLevel.HIGH)),
        (0.85, False, 500, (0.85, ConfidenceLevel.HIGH)),
        (0.84, False, 500, (0.84, ConfidenceLevel.MEDIUM)),
        (0.65, False, 500, (0.65, ConfidenceLevel.MEDIUM)),
        (0.64, False, 500, (0.64, ConfidenceLevel.LOW)),
        (0.5, False, 500, (0.5, ConfidenceLevel.LOW)),
        (0.55, True, 10, (0.55, ConfidenceLevel.LOW)),
    ],
)
def test_confidence(
    risk: float, partial: bool, words: int, expected: tuple[float, ConfidenceLevel]
) -> None:
    value, level = confidence.compute(risk, partial_text=partial, word_count=words)
    assert value == pytest.approx(expected[0])
    assert level == expected[1]


@pytest.mark.parametrize("verdict", list(Verdict))
def test_summary_is_deterministic_and_never_calls_a_text_fake(verdict: Verdict) -> None:
    arguments = {
        "verdict": verdict,
        "risk": 0.82,
        "confidence_level": ConfidenceLevel.MEDIUM,
        "signal_labels": ["Sensational wording", "Repeated punctuation", "Sensational wording"],
        "partial_text": verdict == Verdict.UNVERIFIED,
        "word_count": 12 if verdict == Verdict.UNVERIFIED else 400,
    }
    first = summary.build(**arguments)
    assert first == summary.build(**arguments)
    assert first.strip() == first
    assert len(first) > 40
    for accusation in ("fake", "false", "lie", "lies", "hoax", "true", "untrue"):
        assert not re.search(rf"\b{accusation}\b", first.lower())


def test_summary_names_the_signals_and_the_risk() -> None:
    text = summary.build(
        verdict=Verdict.HIGH_RISK,
        risk=0.914,
        confidence_level=ConfidenceLevel.HIGH,
        signal_labels=["Sensational wording", "Curiosity-gap headline", "Sensational wording"],
        partial_text=False,
        word_count=300,
    )
    assert "91%" in text
    assert "3 " in text
    assert "sensational wording" in text.lower()
    assert "high" in text.lower()


def test_summary_without_signals_says_so() -> None:
    text = summary.build(
        verdict=Verdict.RELIABLE,
        risk=0.08,
        confidence_level=ConfidenceLevel.HIGH,
        signal_labels=[],
        partial_text=False,
        word_count=300,
    )
    assert "no rhetoric signals" in text.lower()
    assert "8%" in text


def test_summary_for_unreadable_pages_asks_for_the_text() -> None:
    text = summary.build(
        verdict=Verdict.UNVERIFIED,
        risk=0.5,
        confidence_level=ConfidenceLevel.LOW,
        signal_labels=[],
        partial_text=True,
        word_count=6,
    )
    assert "paste" in text.lower()
    assert "6 words" in text
