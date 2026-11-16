"""The individual steps of an analysis. Each returns the part of the response it produces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import anyio
import anyio.to_thread
import tldextract

from fnd_api.api.errors import ApiError
from fnd_api.ingest import IngestError
from fnd_api.ingest.extractor import extract_article
from fnd_api.ingest.fetcher import fetch_html
from fnd_api.ingest.language import detect_language
from fnd_api.logging import get_logger
from fnd_api.models_registry import ModelRegistry
from fnd_api.pipeline.context import PipelineContext
from fnd_api.schemas import (
    ArticleInfo,
    Attribution,
    ClassifierResult,
    Flag,
    InputType,
    MetadataInfo,
    SourceInfo,
    Span,
    StageStatus,
    TokenAttribution,
)
from fnd_api.signals.spans import finalize
from fnd_ml.common.text import normalize_text
from fnd_ml.explain import lexical
from fnd_ml.inference.baseline_classifier import BaselineClassifier

log = get_logger("fnd_api.pipeline")

MIN_WORDS = 3  # pasted text shorter than this cannot be analysed
THIN_PAGE_WORDS = 40  # a fetched page with fewer words is not judged
HEADLINE_MAX_WORDS = 40
MAX_HIGHLIGHTS = 200
TOP_TOKENS = 40

_WORD = re.compile(r"[^\W_]+")
# The bundled public suffix list is used as is; nothing is downloaded.
_extract_domain = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def count_words(text: str) -> int:
    return len(_WORD.findall(text))


def prepare_text(text: str, max_chars: int) -> str:
    """Normalize pasted text and enforce the configured size limit."""
    normalized = normalize_text(text)
    if len(normalized) > max_chars:
        raise ApiError(
            "payload_too_large", 413, f"The text is longer than {max_chars:,} characters."
        )
    return normalized


@dataclass(frozen=True)
class Ingested:
    article: ArticleInfo
    metadata: MetadataInfo
    partial: bool  # only part of the article is available
    thin: bool  # too little text from a page to judge at all
    site_url: str | None  # where the text came from, for the source stage


def _require_english(text: str) -> None:
    language = detect_language(text)
    if language not in (None, "en"):
        raise ApiError(
            "unsupported_language",
            422,
            f"Only English text can be analysed (this looks like '{language}').",
        )


def ingest_text(text: str, title: str | None) -> Ingested:
    """``text`` is already normalized (see ``prepare_text``)."""
    words = count_words(text)
    if words < MIN_WORDS:
        raise ApiError(
            "content_too_short", 422, f"At least {MIN_WORDS} words are needed for an analysis."
        )
    _require_english(text)
    input_type = InputType.HEADLINE if words <= HEADLINE_MAX_WORDS else InputType.TEXT
    return Ingested(
        article=ArticleInfo(
            input_type=input_type,
            url=None,
            canonical_url=None,
            text=text,
            word_count=words,
            language="en",
            extraction="user_text",
        ),
        metadata=MetadataInfo(
            title=normalize_text(title) if title else None,
            authors=[],
            site_name=None,
            published_at=None,
            modified_at=None,
            date_source="none",
            anomalies=[],
        ),
        partial=False,
        thin=False,
        site_url=None,
    )


async def ingest_url(url: str, ctx: PipelineContext) -> Ingested:
    try:
        page = await fetch_html(url, ctx.http, ctx.settings, ctx.resolver)
    except IngestError as error:
        raise ApiError(error.code, error.status, error.detail) from error
    extracted = await anyio.to_thread.run_sync(extract_article, page.html, page.final_url)

    text = normalize_text(extracted.text)[: ctx.settings.max_text_chars]
    words = count_words(text)
    _require_english(text)
    thin = words < THIN_PAGE_WORDS
    partial = thin or extracted.paywall_hint

    anomalies = []
    if extracted.published_at and extracted.published_at > datetime.now(UTC) + timedelta(days=1):
        anomalies.append(
            Flag(
                code="future_publish_date",
                severity="warn",
                message="The page gives a publication date in the future.",
            )
        )
    return Ingested(
        article=ArticleInfo(
            input_type=InputType.URL,
            url=url,
            canonical_url=extracted.canonical_url or page.final_url,
            text=text,
            word_count=words,
            language="en",
            extraction="partial" if partial else "full",
        ),
        metadata=MetadataInfo(
            title=normalize_text(extracted.title) if extracted.title else None,
            authors=extracted.authors,
            site_name=extracted.site_name,
            published_at=extracted.published_at,
            modified_at=extracted.modified_at,
            date_source=extracted.date_source,
            anomalies=anomalies,
        ),
        partial=partial,
        thin=thin,
        site_url=page.final_url,
    )


def source_info(site_url: str | None) -> SourceInfo:
    """The registrable domain. Reputation data is not part of this version."""
    domain = None
    if site_url:
        host = (urlsplit(site_url).hostname or "").lower()
        # Hosts outside the public suffix list (IP addresses, private TLDs) are used as they are.
        domain = (
            _extract_domain(host).top_domain_under_public_suffix
            or host.removeprefix("www.")
            or None
        )
    return SourceInfo(
        domain=domain,
        known=False,
        credibility_score=None,
        category="unknown",
        domain_age_days=None,
        flags=[],
        provenance=[],
    )


async def _satire_score(
    models: ModelRegistry, title: str | None, text: str, limiter: anyio.CapacityLimiter
) -> float | None:
    """How much the headline reads like satire; ``None`` without the scorer or when it fails."""
    if models.satire is None:
        return None
    try:
        return await anyio.to_thread.run_sync(models.satire.score, title, text, limiter=limiter)
    except Exception:  # the satire tag is an extra: its failure must not cost the verdict
        log.error("satire_scorer_failed", version=models.satire.version, exc_info=True)
        return None


async def classify(
    models: ModelRegistry, title: str | None, text: str, limiter: anyio.CapacityLimiter
) -> tuple[ClassifierResult, StageStatus, str | None]:
    """Score the text with the transformer, or with the baseline when the transformer cannot.

    Returns the result, the stage status and its detail. The status is ``DEGRADED`` whenever the
    baseline had to answer in the transformer's place.
    """
    satire_score = await _satire_score(models, title, text, limiter)
    status, detail = StageStatus.OK, None
    transformer, baseline = models.transformer, models.baseline
    if transformer is None:
        status, detail = StageStatus.DEGRADED, "transformer model not available; baseline used"
    else:
        try:
            output = await anyio.to_thread.run_sync(
                transformer.predict, title, text, limiter=limiter
            )
        except Exception:
            log.error("transformer_failed", version=transformer.version, exc_info=True)
            status, detail = StageStatus.DEGRADED, "transformer model failed; baseline used"
        else:
            result = ClassifierResult(
                model=transformer.name,
                version=output.version,
                p_unreliable=output.p_unreliable,
                p_unreliable_raw=output.p_raw,
                calibration="temperature",
                satire_score=satire_score,
                truncated=output.truncated,
            )
            return result, status, detail

    if baseline is None:
        raise ApiError("internal", 503, "The model failed and no other model is loaded.")
    output = await anyio.to_thread.run_sync(baseline.predict, title, text, limiter=limiter)
    result = ClassifierResult(
        model="baseline",
        version=output.version,
        p_unreliable=output.p_unreliable,
        p_unreliable_raw=output.p_raw,
        calibration="isotonic" if baseline.model.calib_x_ is not None else "none",
        satire_score=satire_score,
        truncated=output.truncated,
    )
    return result, status, detail


async def rhetoric_signals(ctx: PipelineContext, text: str) -> list[Span]:
    raw = await anyio.to_thread.run_sync(ctx.rules.find, text)
    return finalize(raw, text)[:MAX_HIGHLIGHTS]


async def explain(clf: BaselineClassifier, title: str | None, text: str) -> list[Attribution]:
    weights = await anyio.to_thread.run_sync(lexical.attribute, clf, title, text, TOP_TOKENS)
    tokens = [
        TokenAttribution(start=w.start, end=w.end, token=w.token, weight=w.weight) for w in weights
    ]
    return [Attribution(method="lexical_linear", model="baseline", tokens=tokens)]
