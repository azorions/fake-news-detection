"""Row cleaning: normalized text for storage, scrubbed text for the models, size features, hashes."""

from __future__ import annotations

import re

import pandas as pd

from fnd_ml.common.text import model_input, normalize_text, scrub_text, text_hash

HEADLINE_MAX_WORDS = 40
SHORT_MAX_WORDS = 150
MIN_WORDS = 3

_WORD_RE = re.compile(r"\w+")


def length_bucket(n_words: int) -> str:
    if n_words <= HEADLINE_MAX_WORDS:
        return "headline"
    if n_words <= SHORT_MAX_WORDS:
        return "short"
    return "article"


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of ``frame`` with these extra columns:

    ``model_text`` (what the classifiers read: scrubbed title + scrubbed body), ``n_words`` (words in
    the scrubbed body), ``length_bucket`` and ``text_hash`` (of the scrubbed body). ``title`` and
    ``text`` are normalized but keep their artifacts. Rows with fewer than three words of scrubbed body
    are dropped.
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
    return out[out["n_words"] >= MIN_WORDS].reset_index(drop=True)
