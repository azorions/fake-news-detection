"""Row cleaning: normalized text for storage, scrubbed text for the models, size features, hashes."""

from __future__ import annotations

import re

import pandas as pd

from fnd_ml.common.text import model_input, normalize_text, scrub_text, text_hash

HEADLINE_MAX_WORDS = 40
SHORT_MAX_WORDS = 150
MIN_WORDS = 3

_WORD_RE = re.compile(r"\w+")
_LETTER_RE = re.compile(r"[^\W\d_]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-zÀ-ɏ]")
_ENGLISH_STOPWORDS = frozenset(
    [
        "the",
        "of",
        "and",
        "to",
        "in",
        "is",
        "that",
        "for",
        "on",
        "with",
        "as",
        "was",
        "by",
        "at",
        "it",
        "be",
        "this",
        "are",
        "from",
    ]
)
_LANGUAGE_PREFIX_CHARS = 2000
_MIN_WORDS_FOR_STOPWORD_CHECK = 30


def looks_english(text: str) -> bool:
    """Cheap English check for training data (the API uses a real language detector).

    Rejects text that is mostly in a non-Latin script, and longer text in which common English
    function words are rare. Short Latin-script text is accepted, because a headline has too few
    words to judge.
    """
    prefix = text[:_LANGUAGE_PREFIX_CHARS]
    letters = len(_LETTER_RE.findall(prefix))
    if letters and len(_LATIN_LETTER_RE.findall(prefix)) / letters < 0.8:
        return False
    words = [word.lower() for word in _WORD_RE.findall(prefix)]
    if len(words) < _MIN_WORDS_FOR_STOPWORD_CHECK:
        return True
    return sum(word in _ENGLISH_STOPWORDS for word in words) / len(words) >= 0.10


def length_bucket(n_words: int) -> str:
    if n_words <= HEADLINE_MAX_WORDS:
        return "headline"
    if n_words <= SHORT_MAX_WORDS:
        return "short"
    return "article"


def add_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``year`` (nullable ``Int16``) read from ``date``; empty where the date
    is missing or unreadable. Call it after duplicate removal, which fills in missing dates."""
    out = frame.copy()
    dates = pd.to_datetime(out["date"], errors="coerce", format="%Y-%m-%d")
    out["year"] = dates.dt.year.astype("Int16")
    return out


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of ``frame`` with these extra columns:

    ``model_text`` (what the classifiers read: scrubbed title + scrubbed body), ``n_words`` (words in
    the scrubbed body), ``length_bucket`` and ``text_hash`` (of the scrubbed body). ``title`` and
    ``text`` are normalized but keep their artifacts. Rows with fewer than three words of scrubbed body
    are dropped, and so are rows that do not look English.
    """
    out = frame.copy()
    out["text"] = out["text"].map(normalize_text)
    # Missing titles arrive as None or NaN depending on how the frame was built; make them None.
    titles = [normalize_text(t) if isinstance(t, str) else "" for t in out["title"]]
    out["title"] = pd.Series([t or None for t in titles], index=out.index, dtype="object")
    bodies = out["text"].map(scrub_text)
    out["model_text"] = [
        model_input(title or None, text)[0] for title, text in zip(titles, out["text"], strict=True)
    ]
    out["n_words"] = bodies.map(lambda body: len(_WORD_RE.findall(body))).astype("int64")
    out["length_bucket"] = out["n_words"].map(length_bucket)
    out["text_hash"] = bodies.map(text_hash)
    keep = (out["n_words"] >= MIN_WORDS) & out["text"].map(looks_english)
    return out[keep].reset_index(drop=True)
