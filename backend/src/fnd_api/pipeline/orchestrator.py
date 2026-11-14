"""The analysis pipeline: one async generator of stage events that ends with the result.

The JSON endpoint collects it (``analyze``); the streaming endpoint will forward the events.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from ulid import ULID

from fnd_api.api.errors import ApiError
from fnd_api.cache.keys import SCAN_TTL_S, scan_key
from fnd_api.fusion import confidence, scorer, summary
from fnd_api.logging import get_logger
from fnd_api.pipeline import stages
from fnd_api.pipeline.context import PipelineContext
from fnd_api.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    ScoreComponent,
    StageEvent,
    StageReport,
    StageStatus,
    Tag,
    Verdict,
)

log = get_logger("fnd_api.pipeline")

SCAN_REF_PREFIX = "scanref:"  # scan id -> cache key of its result, so a deleted scan can be purged


class _Clock:
    """Times each stage and collects the stage reports."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.reports: list[StageReport] = []

    def elapsed_ms(self) -> int:
        return round((time.perf_counter() - self.started) * 1000)

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        outcome: dict[str, Any] = {"status": StageStatus.OK, "detail": None}
        start = time.perf_counter()
        yield outcome
        self.reports.append(
            StageReport(
                name=name,
                status=outcome["status"],
                duration_ms=round((time.perf_counter() - start) * 1000),
                detail=outcome["detail"],
            )
        )


async def _cached(ctx: PipelineContext, key: str) -> AnalysisResponse | None:
    try:
        raw = await ctx.cache.get(key)
        return AnalysisResponse.model_validate_json(raw) if raw else None
    except Exception:  # the cache is an optimisation: a broken one must not fail the request
        log.warning("cache_read_failed", exc_info=True)
        return None


async def _remember(ctx: PipelineContext, key: str, response: AnalysisResponse) -> None:
    try:
        await ctx.cache.set(key, response.model_dump_json().encode(), SCAN_TTL_S)
        await ctx.cache.set(SCAN_REF_PREFIX + response.scan_id, key.encode(), SCAN_TTL_S)
    except Exception:
        log.warning("cache_write_failed", exc_info=True)


async def _store(
    ctx: PipelineContext, response: AnalysisResponse, input_hash: str, depth: str
) -> None:
    """Write the scan to history. A failure is logged; the caller still gets the result."""
    try:
        if await ctx.scans.exists(response.scan_id):
            return
        await ctx.scans.add(
            response,
            input_hash=input_hash,
            depth=depth,
            external_used=False,
            status="partial" if "partial_text" in response.tags else "complete",
        )
    except SQLAlchemyError:
        log.error("scan_store_failed", scan_id=response.scan_id, exc_info=True)


async def run(
    req: AnalyzeRequest, ctx: PipelineContext
) -> AsyncIterator[StageEvent | AnalysisResponse]:
    """Yield a ``StageEvent`` per finished stage, then the ``AnalysisResponse``.

    Raises ``ApiError`` for input that cannot be analysed.
    """
    clock = _Clock()
    clf = ctx.models.baseline
    if clf is None:
        raise ApiError("internal", 503, "No model is loaded, so nothing can be analysed yet.")

    if req.url is not None:
        url = str(req.url).split("#", 1)[0]
        text = None
        canonical_input = f"url\n{url}"
    else:
        url = None
        text = stages.prepare_text(req.text or "", ctx.settings.max_text_chars)
        canonical_input = f"text\n{req.title or ''}\n{text}"
    input_hash = hashlib.sha256(canonical_input.encode("utf-8")).hexdigest()

    # Results hold the analysed text, so they are cached only when storing text is allowed.
    caching = ctx.settings.store_input_text
    key = scan_key(canonical_input, req.options, ctx.models.versions)
    if caching and req.options.use_cache:
        hit = await _cached(ctx, key)
        if hit is not None:
            response = hit.model_copy(update={"cached": True, "latency_ms": clock.elapsed_ms()})
            if req.options.store:
                await _store(ctx, response, input_hash, req.options.depth.value)
            yield StageEvent(stage="final_verdict", status=StageStatus.OK, data={"cached": True})
            yield response
            return

    with clock.stage("ingest") as outcome:
        if url is not None:
            ingested = await stages.ingest_url(url, ctx)
        else:
            ingested = stages.ingest_text(text or "", req.title)
        if ingested.partial:
            outcome["status"] = StageStatus.DEGRADED
            outcome["detail"] = "only part of the article could be read"
    article, metadata = ingested.article, ingested.metadata
    yield StageEvent(
        stage="ingest",
        status=clock.reports[-1].status,
        data={
            "article": article.model_dump(mode="json"),
            "metadata": metadata.model_dump(mode="json"),
        },
    )

    with clock.stage("source") as outcome:
        source = stages.source_info(ingested.site_url)
        if source.domain is None:
            outcome["status"] = StageStatus.SKIPPED
            outcome["detail"] = "no URL was given"
    yield StageEvent(
        stage="source",
        status=clock.reports[-1].status,
        data={"source": source.model_dump(mode="json")},
    )

    with clock.stage("classifier"):
        classifier = await stages.classify(clf, metadata.title, article.text)
    yield StageEvent(
        stage="classifier",
        status=StageStatus.OK,
        data={"classifier": classifier.model_dump(mode="json")},
    )

    with clock.stage("signals"):
        highlights = await stages.rhetoric_signals(ctx, article.text)
    yield StageEvent(
        stage="signals",
        status=StageStatus.OK,
        data={"highlights": [span.model_dump(mode="json") for span in highlights]},
    )

    with clock.stage("explanations"):
        attributions = await stages.explain(clf, metadata.title, article.text)
    yield StageEvent(
        stage="explanations",
        status=StageStatus.OK,
        data={"attributions": [item.model_dump(mode="json") for item in attributions]},
    )

    with clock.stage("final_verdict"):
        risk = classifier.p_unreliable
        t_low, t_high = clf.thresholds
        breakdown = [
            ScoreComponent(
                name="classifier",
                log_odds=scorer.logit(risk),
                detail="Calibrated probability from the baseline text classifier.",
            )
        ]
        tags: list[Tag] = ["partial_text"] if ingested.partial else []
        if ingested.thin:
            verdict = Verdict.UNVERIFIED
            breakdown.append(
                ScoreComponent(
                    name="guard",
                    log_odds=0.0,
                    detail=(
                        f"Fewer than {stages.THIN_PAGE_WORDS} words could be read, so no "
                        "verdict is given."
                    ),
                )
            )
        else:
            verdict = scorer.band(risk, t_low, t_high)
        confidence_value, confidence_level = confidence.compute(
            risk, partial_text=ingested.partial, word_count=article.word_count
        )
        summary_text = summary.build(
            verdict=verdict,
            risk=risk,
            confidence_level=confidence_level,
            signal_labels=[span.label for span in highlights],
            partial_text=ingested.partial,
            word_count=article.word_count,
        )
    verdict_data = {
        "verdict": verdict.value,
        "risk_score": risk,
        "confidence": confidence_value,
        "confidence_level": confidence_level.value,
        "summary": summary_text,
        "tags": list(tags),
        "score_breakdown": [component.model_dump(mode="json") for component in breakdown],
    }
    yield StageEvent(stage="final_verdict", status=StageStatus.OK, data=verdict_data)

    response = AnalysisResponse(
        scan_id=str(ULID()),
        created_at=datetime.now(UTC),
        verdict=verdict,
        risk_score=risk,
        confidence=confidence_value,
        confidence_level=confidence_level,
        summary=summary_text,
        tags=tags,
        article=article,
        classifier=classifier,
        source=source,
        metadata=metadata,
        claims=[],
        highlights=highlights,
        attributions=attributions,
        score_breakdown=breakdown,
        stages=clock.reports,
        models=ctx.models.versions,
        cached=False,
        latency_ms=clock.elapsed_ms(),
    )
    if caching:
        await _remember(ctx, key, response)
    if req.options.store:
        await _store(ctx, response, input_hash, req.options.depth.value)
    yield response


async def analyze(req: AnalyzeRequest, ctx: PipelineContext) -> AnalysisResponse:
    """Run the pipeline to its end and return the result."""
    result: AnalysisResponse | None = None
    async for item in run(req, ctx):
        if isinstance(item, AnalysisResponse):
            result = item
    if result is None:  # run() always ends with a response or raises
        raise ApiError("internal", 500, "The analysis did not produce a result.")
    return result
