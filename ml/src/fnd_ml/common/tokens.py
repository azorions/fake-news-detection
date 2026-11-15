"""Word tokenization shared by baseline training, inference and lexical attribution."""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence

TOKEN_RE = re.compile(r"\b\w\w+\b")
MAX_TOKENS = 600


def word_tokens(text: str, max_tokens: int | None = None) -> list[tuple[str, int, int]]:
    """Return ``(lowercase token, start, end)`` for each word of two or more characters.

    Offsets refer to ``text`` itself; each token is lowercased on its own so that characters whose
    lowercase form has a different length cannot shift the offsets.
    """
    matches = itertools.islice(TOKEN_RE.finditer(text), max_tokens)
    return [(match.group().lower(), match.start(), match.end()) for match in matches]


def ngrams(tokens: Sequence[str]) -> list[str]:
    """All unigrams in order, followed by all bigrams in order (bigram ``i`` joins tokens ``i, i+1``)."""
    return [*tokens, *(f"{first} {second}" for first, second in itertools.pairwise(tokens))]


def analyze(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """Features of ``text``: unigrams and bigrams over its first ``max_tokens`` words."""
    return ngrams([token for token, _, _ in word_tokens(text, max_tokens)])


def analyze_chars(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """Character 3-5-grams inside word boundaries, over the first ``max_tokens`` words.

    Each word is padded with one space on both sides, as scikit-learn's ``char_wb`` analyzer does;
    a padded word shorter than the n-gram is given once, whole.
    """
    grams: list[str] = []
    for token, _, _ in word_tokens(text, max_tokens):
        padded = f" {token} "
        for n in (3, 4, 5):
            if len(padded) <= n:
                grams.append(padded)
                break
            grams.extend(padded[start : start + n] for start in range(len(padded) - n + 1))
    return grams
