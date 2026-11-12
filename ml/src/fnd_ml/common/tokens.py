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
