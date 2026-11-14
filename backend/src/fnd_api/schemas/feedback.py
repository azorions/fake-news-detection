from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from fnd_api.schemas.enums import InputType, Verdict


class FeedbackCreate(BaseModel):
    rating: Literal["agree", "disagree"]
    corrected_verdict: Verdict | None = None
    target_type: Literal["verdict", "claim", "highlight", "source"] = "verdict"
    target_id: str | None = None  # claim or span id inside the stored result
    comment: str | None = Field(None, max_length=2000)


class FeedbackOut(FeedbackCreate):
    id: str
    scan_id: str
    created_at: datetime


class ScanSummary(BaseModel):
    scan_id: str
    created_at: datetime
    input_type: InputType
    title: str | None
    domain: str | None
    verdict: Verdict
    risk_score: float
    confidence: float
    feedback_count: int


class ScanPage(BaseModel):
    items: list[ScanSummary]
    next_cursor: str | None
