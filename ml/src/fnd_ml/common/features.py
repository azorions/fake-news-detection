"""Stylometric features: how a text is written, not what it is about.

Twelve numbers per text (length, word shape, punctuation, sentence rhythm, direct address). They
are computed from the text as written, so capitals and punctuation count; the word n-gram models
never see either.
"""

from __future__ import annotations

import math
import re

import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES: tuple[str, ...] = (
    "log_words",
    "mean_word_len",
    "type_token_ratio",
    "caps_share",
    "title_case_share",
    "exclaim_per_100w",
    "question_per_100w",
    "quote_per_100w",
    "digit_share",
    "mean_sentence_words",
    "std_sentence_words",
    "second_person_share",
)

TYPE_TOKEN_WORDS = 600  # the type-token ratio falls with length, so it is read on a fixed prefix

# A word: a number with its separators ("3.5", "10,000") or letters and digits with inner apostrophes.
_WORD_RE = re.compile(r"\d+(?:[.,]\d+)+|[^\W_]+(?:['’][^\W_]+)*")
# Sentence end: closing punctuation, optional closing quotes or brackets, then a space or the end.
# "3.5" and "example.com" therefore do not end a sentence.
_SENTENCE_END_RE = re.compile(r"[.!?]+[\"'”’)\]]*(?=\s|$)")
_QUOTES = frozenset('"“”«»')
_SECOND_PERSON = frozenset({"you", "your", "yours", "yourself", "yourselves"})


def _is_second_person(word: str) -> bool:
    return re.split("['’]", word.lower(), maxsplit=1)[0] in _SECOND_PERSON


def stylometric(text: str) -> NDArray[np.float32]:
    """The features named in ``FEATURE_NAMES``, in that order. A text without words gives zeros."""
    words = _WORD_RE.findall(text)
    n_words = len(words)
    per_word = 1.0 / max(n_words, 1)
    per_100_words = 100.0 * per_word

    sentences = [len(_WORD_RE.findall(part)) for part in _SENTENCE_END_RE.split(text)]
    sentences = [count for count in sentences if count] or [0]
    visible = sum(not char.isspace() for char in text)
    prefix = [word.lower() for word in words[:TYPE_TOKEN_WORDS]]

    values = (
        math.log1p(n_words),
        sum(map(len, words)) * per_word,
        len(set(prefix)) / max(len(prefix), 1),
        sum(len(word) >= 2 and word.isupper() for word in words) * per_word,
        sum(word[0].isupper() and word[1:].islower() for word in words) * per_word,
        text.count("!") * per_100_words,
        text.count("?") * per_100_words,
        sum(char in _QUOTES for char in text) * per_100_words,
        sum(char.isdigit() for char in text) / max(visible, 1),
        float(np.mean(sentences)),
        float(np.std(sentences)),
        sum(map(_is_second_person, words)) * per_word,
    )
    return np.asarray(values, dtype=np.float32)
