"""Feedback on a stored scan: agree or disagree with the verdict, a claim, a highlight or the
source rating."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from fnd_api.api.deps import FeedbackDep
from fnd_api.api.v1.scans import scan_not_found
from fnd_api.db.repositories import ScanNotFoundError
from fnd_api.ratelimit import limit_read
from fnd_api.schemas import FeedbackCreate, FeedbackOut

router = APIRouter(tags=["feedback"], dependencies=[Depends(limit_read)])


@router.post("/scans/{scan_id}/feedback", status_code=201)
async def add_feedback(scan_id: str, body: FeedbackCreate, feedback: FeedbackDep) -> FeedbackOut:
    try:
        return await feedback.add(scan_id, body)
    except ScanNotFoundError as error:
        raise scan_not_found() from error
