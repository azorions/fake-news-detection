"""``POST /api/v1/analyze``: analyse pasted text or the article at a URL."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from fnd_api.api.deps import PipelineDep
from fnd_api.pipeline import orchestrator
from fnd_api.ratelimit import limit_analyze
from fnd_api.schemas import AnalysisResponse, AnalyzeRequest

router = APIRouter(tags=["analysis"])


@router.post("/analyze", dependencies=[Depends(limit_analyze)])
async def analyze(body: AnalyzeRequest, context: PipelineDep) -> AnalysisResponse:
    return await orchestrator.analyze(body, context)
