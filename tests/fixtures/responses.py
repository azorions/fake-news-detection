"""A complete, valid ``AnalysisResponse`` for tests that need one to store or serialize."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ulid import ULID

from fnd_api.schemas import AnalysisResponse

SAMPLE_TEXT = "SHOCKING: the government is hiding the truth about vaccines! Wake up, people!!!"


def make_response(**overrides: Any) -> AnalysisResponse:
    document: dict[str, Any] = {
        "scan_id": str(ULID()),
        "created_at": datetime.now(UTC),
        "verdict": "high_risk",
        "risk_score": 0.91,
        "confidence": 0.91,
        "confidence_level": "high",
        "summary": "The wording of this text is typical of unreliable articles.",
        "tags": [],
        "article": {
            "input_type": "text",
            "url": None,
            "canonical_url": None,
            "text": SAMPLE_TEXT,
            "word_count": 13,
            "language": "en",
            "extraction": "user_text",
        },
        "classifier": {
            "model": "baseline",
            "version": "v-test",
            "p_unreliable": 0.91,
            "p_unreliable_raw": 0.88,
            "calibration": "isotonic",
            "truncated": False,
        },
        "source": {
            "domain": None,
            "known": False,
            "category": "unknown",
            "domain_age_days": None,
            "flags": [],
            "provenance": [],
        },
        "metadata": {
            "title": None,
            "authors": [],
            "site_name": None,
            "published_at": None,
            "modified_at": None,
            "date_source": "none",
            "anomalies": [],
        },
        "claims": [],
        "highlights": [
            {
                "id": "s1",
                "start": 0,
                "end": 8,
                "kind": "sensational",
                "label": "Alarm word",
                "explanation": "Words like this are used to provoke rather than inform.",
                "score": 0.8,
                "origin": "rule",
            }
        ],
        "attributions": [
            {
                "method": "lexical_linear",
                "model": "baseline",
                "tokens": [{"start": 0, "end": 8, "token": "shocking", "weight": 1.25}],
            }
        ],
        "score_breakdown": [{"name": "classifier", "log_odds": 2.3, "detail": "baseline"}],
        "stages": [{"name": "classifier", "status": "ok", "duration_ms": 4}],
        "models": {"baseline": "v-test"},
        "cached": False,
        "latency_ms": 12,
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(document.get(key), dict):
            document[key] = {**document[key], **value}
        else:
            document[key] = value
    return AnalysisResponse.model_validate(document)
