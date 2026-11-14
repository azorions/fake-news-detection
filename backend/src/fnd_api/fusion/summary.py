"""The one-paragraph explanation of a result. Built from templates, so the same result always
reads the same, and worded as risk, never as an accusation."""

from __future__ import annotations

from collections import Counter

from fnd_api.schemas import ConfidenceLevel, Verdict

_OPENING = {
    Verdict.RELIABLE: "Low risk ({risk}): the wording resembles reporting from reliable sources.",
    Verdict.MISLEADING: (
        "Mixed signals ({risk} risk): the wording shares features with both reliable and "
        "unreliable articles, so read it with care."
    ),
    Verdict.HIGH_RISK: (
        "High risk ({risk}): the wording closely resembles articles from unreliable sources."
    ),
}
_MAX_LABELS = 3


def _signals_sentence(signal_labels: list[str]) -> str:
    if not signal_labels:
        return "No rhetoric signals were found."
    # Most frequent first; ties in alphabetical order, so the sentence is stable.
    ranked = sorted(Counter(signal_labels).items(), key=lambda item: (-item[1], item[0]))
    names = ", ".join(label.lower() for label, _ in ranked[:_MAX_LABELS])
    count = len(signal_labels)
    noun = "rhetoric signal was" if count == 1 else "rhetoric signals were"
    return f"{count} {noun} found, including {names}."


def build(
    *,
    verdict: Verdict,
    risk: float,
    confidence_level: ConfidenceLevel,
    signal_labels: list[str],
    partial_text: bool,
    word_count: int,
) -> str:
    if verdict == Verdict.UNVERIFIED:
        return (
            f"Only {word_count} words could be read from this page, which is too little to "
            "assess. Paste the article text to get a result."
        )
    sentences = [_OPENING[verdict].format(risk=f"{round(risk * 100)}%")]
    sentences.append(_signals_sentence(signal_labels))
    if partial_text:
        sentences.append("Only part of the article could be read.")
    sentences.append(f"Confidence is {confidence_level.value}.")
    sentences.append("This is an estimate from writing style, not a check of the facts.")
    return " ".join(sentences)
