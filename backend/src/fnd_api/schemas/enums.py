from __future__ import annotations

from enum import StrEnum


class InputType(StrEnum):
    URL = "url"
    TEXT = "text"
    HEADLINE = "headline"


class Depth(StrEnum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class Verdict(StrEnum):
    RELIABLE = "reliable"
    MISLEADING = "potentially_misleading"
    HIGH_RISK = "high_risk"
    UNVERIFIED = "unverified"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    MIXED = "mixed"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"
    NOT_CHECKED = "not_checked"


class Stance(StrEnum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


class SpanKind(StrEnum):
    CLICKBAIT = "clickbait"
    SENSATIONAL = "sensational"
    LOADED_LANGUAGE = "loaded_language"
    VAGUE_ATTRIBUTION = "vague_attribution"
    FALLACY = "fallacy"
    CLAIM = "claim"


class StageStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
