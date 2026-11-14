"""Events of the analysis pipeline. Over SSE: ``stage`` events, then one ``result`` or ``error``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from fnd_api.schemas.enums import StageStatus

StageName = Literal[
    "ingest",
    "source",
    "classifier",
    "signals",
    "preliminary_verdict",
    "claims",
    "evidence",
    "explanations",
    "final_verdict",
]


class StageEvent(BaseModel):
    stage: StageName
    status: StageStatus
    data: dict[str, Any]  # the part of AnalysisResponse that this stage produced
