"""The analysis result. All ``start``/``end`` values are code point offsets into ``article.text``,
end exclusive."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from fnd_api.schemas.enums import (
    ClaimStatus,
    ConfidenceLevel,
    InputType,
    SpanKind,
    StageStatus,
    Stance,
    Verdict,
)

Tag = Literal[
    "satire",
    "opinion",
    "partial_text",
    "no_evidence_found",
    "evidence_conflict",
    "old_article",
    "external_disabled",
]


class Flag(BaseModel):
    code: str  # e.g. "future_publish_date", "lookalike_domain"
    severity: Literal["info", "warn", "alert"]
    message: str


class Span(BaseModel):
    id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    kind: SpanKind
    label: str  # e.g. "False dilemma", "Curiosity-gap headline"
    explanation: str
    score: float = Field(ge=0, le=1)
    origin: Literal["rule", "llm", "model"]
    claim_id: str | None = None


class TokenAttribution(BaseModel):
    start: int
    end: int
    token: str
    weight: float  # signed; positive pushes toward high risk


class SentenceAttribution(BaseModel):
    index: int
    start: int
    end: int
    weight: float  # change in log-odds when the sentence is removed


class Attribution(BaseModel):
    method: Literal["lexical_linear", "sentence_occlusion", "integrated_gradients"]
    model: str
    sentences: list[SentenceAttribution] = Field(default_factory=list)
    tokens: list[TokenAttribution] = Field(default_factory=list)


class ClassifierResult(BaseModel):
    model: str
    version: str
    p_unreliable: float = Field(ge=0, le=1)  # calibrated
    p_unreliable_raw: float = Field(ge=0, le=1)
    calibration: Literal["temperature", "isotonic", "none"]
    satire_score: float | None = Field(None, ge=0, le=1)  # None until a satire scorer exists (M2)
    truncated: bool


class SourceInfo(BaseModel):
    domain: str | None
    known: bool
    credibility_score: float | None = Field(None, ge=0, le=1)
    category: Literal["reliable", "mixed", "unreliable", "satire", "unknown"]
    domain_age_days: int | None
    flags: list[Flag]
    provenance: list[str]  # which reputation lists the rating came from


class MetadataInfo(BaseModel):
    title: str | None
    authors: list[str]
    site_name: str | None
    published_at: datetime | None
    modified_at: datetime | None
    date_source: Literal["jsonld", "meta", "url", "text", "none"]
    anomalies: list[Flag]


class Evidence(BaseModel):
    id: str
    provider: Literal["google_factcheck", "web_search", "local_index"]
    title: str
    url: str | None
    publisher: str | None
    published_at: datetime | None
    snippet: str
    rating: str | None  # fact-checker's own rating text, e.g. "False"
    stance: Stance
    stance_score: float = Field(ge=0, le=1)
    trust_tier: Literal["fact_checker", "high", "reference", "unknown"]
    snapshot_date: date | None  # set for local_index items


class Claim(BaseModel):
    id: str
    text: str
    start: int | None  # location in article.text when found verbatim
    end: int | None
    checkworthiness: float = Field(ge=0, le=1)
    status: ClaimStatus
    confidence: float = Field(ge=0, le=1)
    rationale: str
    method: Literal["llm", "nli", "none"]
    evidence: list[Evidence]


class ScoreComponent(BaseModel):
    name: Literal["classifier", "source", "evidence", "guard"]
    log_odds: float
    detail: str


class StageReport(BaseModel):
    name: str
    status: StageStatus
    duration_ms: int
    provider: str | None = None
    detail: str | None = None


class ArticleInfo(BaseModel):
    input_type: InputType
    url: str | None
    canonical_url: str | None
    text: str  # the analysed text; all offsets refer to this string
    word_count: int
    language: str
    extraction: Literal["full", "partial", "headline_only", "user_text"]


class AnalysisResponse(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    scan_id: str
    created_at: datetime
    verdict: Verdict
    risk_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    confidence_level: ConfidenceLevel
    summary: str  # deterministic, template-built explanation
    tags: list[Tag]
    article: ArticleInfo
    classifier: ClassifierResult
    source: SourceInfo
    metadata: MetadataInfo
    claims: list[Claim]
    highlights: list[Span]
    attributions: list[Attribution]
    score_breakdown: list[ScoreComponent]
    stages: list[StageReport]
    models: dict[str, str]
    cached: bool
    latency_ms: int
