"""Exact per-word contributions to the baseline's score.

The baseline is linear in its TF-IDF features, so its logit is a sum over the feature buckets of a
text. Each bucket's share is divided equally among the n-grams of the text that fall into it, and a
bigram's share equally between its two words. The word weights therefore add up, with the
intercept, to exactly the model's decision function. A positive weight pushes toward "unreliable".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fnd_ml.common.text import model_input
from fnd_ml.common.tokens import ngrams, word_tokens
from fnd_ml.inference.baseline_classifier import BaselineClassifier


@dataclass(frozen=True)
class TokenWeight:
    start: int  # offsets into the original text
    end: int
    token: str
    weight: float


def _weights(
    clf: BaselineClassifier, title: str | None, text: str
) -> tuple[list[tuple[str, int, int]], NDArray[np.float64], list[int]]:
    """Tokens of the model input, the weight of each, and the model-input-to-text offset mapping."""
    model = clf.model
    if model.coef_ is None:
        raise RuntimeError("model is not fitted")
    model_text, mapping = model_input(title, text)
    tokens = word_tokens(model_text, model.config.max_tokens)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return tokens, np.zeros(0), mapping

    # The model's own feature row, so the weights cannot drift from what it predicts with.
    row = model.transform([model_text])
    buckets, feature_bucket, occurrences = np.unique(
        model.feature_indices(ngrams([token for token, _, _ in tokens])),
        return_inverse=True,
        return_counts=True,
    )
    bucket_logit = np.zeros(len(buckets))
    bucket_logit[np.searchsorted(buckets, row.indices)] = row.data * model.coef_[row.indices]

    # Features are the unigrams in order, then the bigrams in order (bigram i joins words i, i+1).
    per_feature = (bucket_logit / occurrences)[feature_bucket]
    weights = per_feature[:n_tokens].copy()
    half = per_feature[n_tokens:] / 2.0
    weights[:-1] += half
    weights[1:] += half
    return tokens, weights, mapping


def total_logit(clf: BaselineClassifier, title: str | None, text: str) -> float:
    """Intercept plus every word weight, title included. Equals the model's decision function."""
    _, weights, _ = _weights(clf, title, text)
    return float(weights.sum()) + clf.model.intercept_


def attribute(
    clf: BaselineClassifier, title: str | None, text: str, top_k: int | None = 40
) -> list[TokenWeight]:
    """The ``top_k`` strongest words of ``text`` (all of them for ``None``), in reading order.

    Left out: words of the title, words the model gives no weight, and the rare word that only
    exists because an artifact was removed from its middle (it has no single span in ``text``).
    """
    tokens, weights, mapping = _weights(clf, title, text)
    found: list[TokenWeight] = []
    for (token, start, end), weight in zip(tokens, weights.tolist(), strict=True):
        first, last = mapping[start], mapping[end - 1]
        if weight == 0.0 or first < 0 or last - first != end - 1 - start:
            continue
        found.append(TokenWeight(first, last + 1, token, weight))
    if top_k is not None and len(found) > top_k:
        strongest = sorted(found, key=lambda item: abs(item.weight), reverse=True)[:top_k]
        found = sorted(strongest, key=lambda item: item.start)
    return found
