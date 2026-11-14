"""Language detection. The service analyses English only; this tells it when to say so."""

from __future__ import annotations

import functools
import re

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

MIN_WORDS = 5
SHORT_TEXT_WORDS = 20
MIN_CONFIDENCE = 0.75
MIN_CONFIDENCE_SHORT = 0.9  # short texts are easy to misjudge, so a rejection needs more certainty
_SAMPLE_CHARS = 4000

_LANGUAGES = (
    Language.ENGLISH,
    Language.SPANISH,
    Language.FRENCH,
    Language.GERMAN,
    Language.ITALIAN,
    Language.PORTUGUESE,
    Language.DUTCH,
    Language.SWEDISH,
    Language.POLISH,
    Language.ROMANIAN,
    Language.TURKISH,
    Language.INDONESIAN,
    Language.VIETNAMESE,
    Language.RUSSIAN,
    Language.UKRAINIAN,
    Language.ARABIC,
    Language.HINDI,
    Language.CHINESE,
    Language.JAPANESE,
    Language.KOREAN,
)
_WORD = re.compile(r"[^\W\d_]{2,}")


@functools.cache
def _detector() -> LanguageDetector:
    return LanguageDetectorBuilder.from_languages(*_LANGUAGES).build()


def detect_language(text: str) -> str | None:
    """ISO 639-1 code of the text's language, or ``None`` when it cannot be told reliably.

    ``None`` (to be treated as English) is returned for texts under five words and for uncertain
    detections, so a short English headline is never rejected as foreign.
    """
    sample = text[:_SAMPLE_CHARS]
    n_words = len(_WORD.findall(sample))
    if n_words < MIN_WORDS:
        return None
    values = _detector().compute_language_confidence_values(sample)
    if not values:
        return None
    best = values[0]
    needed = MIN_CONFIDENCE_SHORT if n_words < SHORT_TEXT_WORDS else MIN_CONFIDENCE
    if best.value < needed:
        return None
    return best.language.iso_code_639_1.name.lower()
