"""Turn raw detector output into the highlight list of a response."""

from __future__ import annotations

from fnd_api.schemas import Span
from fnd_api.signals.rules import RawSpan


def finalize(raw: list[RawSpan], text: str) -> list[Span]:
    """Valid, tidy highlights for ``text``.

    Drops spans that are empty or reach outside the text, and spans that lie inside another span
    of the same kind (of two identical spans the stronger one stays). The result is sorted by
    position and numbered ``s1``, ``s2``, …
    """
    valid = [span for span in raw if 0 <= span.start < span.end <= len(text)]
    # Widest first, so that whatever could contain a span has been decided before it.
    valid.sort(key=lambda span: (span.start, -span.end, -span.score))
    kept: list[RawSpan] = []
    for span in valid:
        nested = any(
            other.kind == span.kind and other.start <= span.start and span.end <= other.end
            for other in kept
        )
        if not nested:
            kept.append(span)
    kept.sort(key=lambda span: (span.start, span.end, span.kind.value))
    return [
        Span(
            id=f"s{number}",
            start=span.start,
            end=span.end,
            kind=span.kind,
            label=span.label,
            explanation=span.explanation,
            score=min(max(span.score, 0.0), 1.0),
            origin="rule",
        )
        for number, span in enumerate(kept, start=1)
    ]
