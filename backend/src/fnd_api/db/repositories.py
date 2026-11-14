"""Storage of scans and feedback. The full result is kept as a JSON snapshot; a few columns are
copied out of it for listing and filtering."""

from __future__ import annotations

import base64
import binascii
import json
from datetime import UTC, datetime

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from ulid import ULID

from fnd_api.db.models import Feedback, Scan
from fnd_api.schemas import (
    AnalysisResponse,
    FeedbackCreate,
    FeedbackOut,
    InputType,
    ScanSummary,
    Verdict,
)

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"  # fixed width, so text order is time order
MAX_PAGE_SIZE = 100


class ScanNotFoundError(LookupError):
    pass


def _to_text(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(_TIME_FORMAT)


def _from_text(value: str) -> datetime:
    return datetime.strptime(value, _TIME_FORMAT).replace(tzinfo=UTC)


def _encode_cursor(created_at: str, scan_id: str) -> str:
    return base64.urlsafe_b64encode(f"{created_at}|{scan_id}".encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[str, str]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        created_at, scan_id = raw.split("|", 1)
        _from_text(created_at)
    except (ValueError, binascii.Error) as error:
        raise ValueError("invalid cursor") from error
    return created_at, scan_id


def _escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _without_text(response: AnalysisResponse) -> AnalysisResponse:
    """The snapshot to store when input text must not be kept: no text, and nothing that quotes it
    or points into it."""
    return response.model_copy(
        update={
            "article": response.article.model_copy(update={"text": ""}),
            "highlights": [],
            "attributions": [],
            "claims": [c.model_copy(update={"start": None, "end": None}) for c in response.claims],
        }
    )


class ScanRepository:
    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], store_input_text: bool = True
    ) -> None:
        self._sessions = sessions
        self._store_input_text = store_input_text

    async def add(
        self,
        response: AnalysisResponse,
        *,
        input_hash: str,
        depth: str,
        external_used: bool,
        status: str,
    ) -> None:
        snapshot = response if self._store_input_text else _without_text(response)
        row = Scan(
            id=response.scan_id,
            created_at=_to_text(response.created_at),
            input_type=response.article.input_type.value,
            input_hash=input_hash,
            url=response.article.url,
            canonical_url=response.article.canonical_url,
            domain=response.source.domain,
            title=response.metadata.title,
            input_text=response.article.text if self._store_input_text else None,
            word_count=response.article.word_count,
            verdict=response.verdict.value,
            risk_score=response.risk_score,
            confidence=response.confidence,
            classifier_prob=response.classifier.p_unreliable,
            depth=depth,
            external_used=int(external_used),
            status=status,
            latency_ms=response.latency_ms,
            model_versions=json.dumps(response.models, sort_keys=True),
            schema_version=response.schema_version,
            result_json=snapshot.model_dump_json(),
        )
        async with self._sessions.begin() as session:
            session.add(row)

    async def exists(self, scan_id: str) -> bool:
        async with self._sessions() as session:
            return await session.scalar(select(Scan.id).where(Scan.id == scan_id)) is not None

    async def get(self, scan_id: str) -> AnalysisResponse | None:
        async with self._sessions() as session:
            snapshot = await session.scalar(select(Scan.result_json).where(Scan.id == scan_id))
        return None if snapshot is None else AnalysisResponse.model_validate_json(snapshot)

    async def list(
        self,
        cursor: str | None = None,
        limit: int = 20,
        verdict: Verdict | None = None,
        domain: str | None = None,
        q: str | None = None,
    ) -> tuple[list[ScanSummary], str | None]:
        """One page of scans, newest first, and the cursor of the next page (``None`` at the end).

        ``q`` is a case-insensitive substring of the title, URL or domain. Raises ``ValueError``
        for a malformed cursor.
        """
        limit = max(1, min(limit, MAX_PAGE_SIZE))
        feedback_count = (
            select(func.count())
            .select_from(Feedback)
            .where(Feedback.scan_id == Scan.id)
            .scalar_subquery()
        )
        query = select(
            Scan.id,
            Scan.created_at,
            Scan.input_type,
            Scan.title,
            Scan.domain,
            Scan.verdict,
            Scan.risk_score,
            Scan.confidence,
            feedback_count,
        )
        if cursor:
            created_at, scan_id = _decode_cursor(cursor)
            query = query.where(
                or_(
                    Scan.created_at < created_at,
                    and_(Scan.created_at == created_at, Scan.id < scan_id),
                )
            )
        if verdict is not None:
            query = query.where(Scan.verdict == verdict.value)
        if domain:
            query = query.where(Scan.domain == domain.lower())
        if q:
            pattern = f"%{_escape_like(q)}%"
            query = query.where(
                or_(
                    Scan.title.ilike(pattern, escape="\\"),
                    Scan.url.ilike(pattern, escape="\\"),
                    Scan.domain.ilike(pattern, escape="\\"),
                )
            )
        query = query.order_by(Scan.created_at.desc(), Scan.id.desc()).limit(limit + 1)

        async with self._sessions() as session:
            rows = (await session.execute(query)).all()
        page = rows[:limit]
        items = [
            ScanSummary(
                scan_id=row[0],
                created_at=_from_text(row[1]),
                input_type=InputType(row[2]),
                title=row[3],
                domain=row[4],
                verdict=Verdict(row[5]),
                risk_score=row[6],
                confidence=row[7],
                feedback_count=row[8],
            )
            for row in page
        ]
        more = len(rows) > limit
        return items, _encode_cursor(page[-1][1], page[-1][0]) if more else None

    async def delete(self, scan_id: str) -> bool:
        """Delete a scan and, through the foreign key, its feedback. ``False`` if it did not exist."""
        async with self._sessions.begin() as session:
            deleted = await session.scalar(
                delete(Scan).where(Scan.id == scan_id).returning(Scan.id)
            )
        return deleted is not None


class FeedbackRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add(self, scan_id: str, data: FeedbackCreate) -> FeedbackOut:
        """Store feedback on a scan. Raises ``ScanNotFoundError`` when the scan does not exist."""
        created_at = datetime.now(UTC)
        row = Feedback(
            id=str(ULID()),
            scan_id=scan_id,
            created_at=_to_text(created_at),
            rating=data.rating,
            corrected_verdict=data.corrected_verdict.value if data.corrected_verdict else None,
            target_type=data.target_type,
            target_id=data.target_id,
            comment=data.comment,
        )
        async with self._sessions.begin() as session:
            if await session.scalar(select(Scan.id).where(Scan.id == scan_id)) is None:
                raise ScanNotFoundError(scan_id)
            session.add(row)
        return FeedbackOut(
            **data.model_dump(), id=row.id, scan_id=scan_id, created_at=_from_text(row.created_at)
        )

    async def count(self, scan_id: str) -> int:
        query = select(func.count()).select_from(Feedback).where(Feedback.scan_id == scan_id)
        async with self._sessions() as session:
            return int(await session.scalar(query) or 0)
