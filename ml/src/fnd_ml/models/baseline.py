"""Baseline classifier: hashed TF-IDF over word 1-2-grams + logistic regression + isotonic calibration.

Features are hashed into a fixed number of buckets instead of building a vocabulary. A vocabulary of
every bigram in ~100k news articles needs gigabytes of RAM; hashing needs none, and the model stays
exactly linear, so each word's contribution to a prediction can still be computed exactly
(``fnd_ml.explain.lexical``).

The saved artifact is two plain files, ``model.npz`` and ``metadata.json``. Nothing is pickled, so
loading it cannot execute code and does not depend on the scikit-learn version used for training.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from sklearn.feature_extraction import FeatureHasher
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from fnd_ml.common.tokens import MAX_TOKENS, analyze

PROBABILITY_FLOOR = 0.01  # calibrated probabilities are clipped to [0.01, 0.99]
_BATCH = 2000


@dataclass(frozen=True)
class BaselineConfig:
    n_features: int = 2**20
    max_tokens: int = MAX_TOKENS
    min_df: int = 3
    C: float = 4.0
    max_iter: int = 300


class BaselineModel:
    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self.metadata: dict[str, Any] = {}
        self.idf_: NDArray[np.float32] | None = None
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float = 0.0
        self.calib_x_: NDArray[np.float64] | None = None
        self.calib_y_: NDArray[np.float64] | None = None
        self._hasher = FeatureHasher(
            n_features=self.config.n_features,
            input_type="string",
            alternate_sign=False,
            dtype=np.float32,
        )

    # ------------------------------------------------------------ features

    def feature_indices(self, features: list[str]) -> NDArray[np.int32]:
        """The bucket of each feature string, in order."""
        if not features:
            return np.empty(0, dtype=np.int32)
        indices: NDArray[np.int32] = self._hasher.transform([[f] for f in features]).indices
        return indices

    def _counts(self, texts: Iterable[str]) -> sp.csr_matrix:
        max_tokens = self.config.max_tokens
        chunks = [
            self._hasher.transform([analyze(text, max_tokens) for text in batch])
            for batch in itertools.batched(texts, _BATCH)
        ]
        if not chunks:
            return sp.csr_matrix((0, self.config.n_features), dtype=np.float32)
        return sp.vstack(chunks, format="csr")

    def _weight(self, counts: sp.csr_matrix) -> sp.csr_matrix:
        """Turn counts into sublinear TF-IDF rows of unit length (in place)."""
        if self.idf_ is None:
            raise RuntimeError("model is not fitted")
        np.log(counts.data, out=counts.data)
        counts.data += 1.0
        counts.data *= self.idf_[counts.indices]
        counts.eliminate_zeros()
        normalize(counts, norm="l2", copy=False)
        return counts

    def transform(self, texts: Iterable[str]) -> sp.csr_matrix:
        return self._weight(self._counts(texts))

    # ------------------------------------------------------------ training

    def fit(self, texts: Iterable[str], y: ArrayLike) -> Self:
        labels = np.asarray(y, dtype=int)
        counts = self._counts(texts)
        if counts.shape[0] != len(labels):
            raise ValueError(f"got {counts.shape[0]} texts but {len(labels)} labels")

        n_docs = counts.shape[0]
        document_frequency = np.bincount(counts.indices, minlength=self.config.n_features)
        idf = np.log((1 + n_docs) / (1 + document_frequency)) + 1.0
        idf[document_frequency < self.config.min_df] = 0.0
        self.idf_ = idf.astype(np.float32)

        classifier = LogisticRegression(
            C=self.config.C, class_weight="balanced", max_iter=self.config.max_iter
        )
        classifier.fit(self._weight(counts), labels)
        self.coef_ = classifier.coef_.ravel().astype(np.float64)
        self.intercept_ = float(classifier.intercept_[0])
        self.calib_x_ = self.calib_y_ = None
        return self

    def calibrate_scores(self, raw: ArrayLike, y: ArrayLike) -> Self:
        """Fit the isotonic calibrator on raw probabilities from ``predict_raw``."""
        isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        isotonic.fit(np.asarray(raw, dtype=float), np.asarray(y, dtype=float))
        self.calib_x_ = np.asarray(isotonic.X_thresholds_, dtype=np.float64)
        self.calib_y_ = np.asarray(isotonic.y_thresholds_, dtype=np.float64)
        return self

    def calibrate(self, texts: Iterable[str], y: ArrayLike) -> Self:
        return self.calibrate_scores(self.predict_raw(texts), y)

    # ------------------------------------------------------------ prediction

    def decision_function(self, texts: Iterable[str]) -> NDArray[np.float64]:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        scores: NDArray[np.float64] = self.transform(texts) @ self.coef_ + self.intercept_
        return scores

    def predict_raw(self, texts: Iterable[str]) -> NDArray[np.float64]:
        """Uncalibrated P(unreliable)."""
        raw: NDArray[np.float64] = expit(self.decision_function(texts))
        return raw

    def to_calibrated(self, raw: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(raw, dtype=float)
        if self.calib_x_ is not None and self.calib_y_ is not None:
            values = np.interp(values, self.calib_x_, self.calib_y_)
        return np.clip(values, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)

    def predict_proba(self, texts: Iterable[str]) -> NDArray[np.float64]:
        """Calibrated P(unreliable), clipped to [0.01, 0.99]."""
        return self.to_calibrated(self.predict_raw(texts))

    # ------------------------------------------------------------ inspection

    def top_features(self, sample_texts: Iterable[str], k: int = 50) -> dict[str, list[str]]:
        """Names of the ``k`` strongest buckets per class, named from ``sample_texts``.

        A bucket is named after the most frequent n-gram in the sample that hashes into it.
        """
        if self.coef_ is None or self.idf_ is None:
            raise RuntimeError("model is not fitted")
        strength = self.coef_ * self.idf_
        order = np.argsort(strength)
        wanted = {"unreliable": order[::-1][:k], "reliable": order[:k]}

        counter: Counter[str] = Counter()
        for text in sample_texts:
            counter.update(analyze(text, self.config.max_tokens))
        names = [ngram for ngram, count in counter.items() if count >= 2]
        best: dict[int, tuple[str, int]] = {}
        for ngram, index in zip(names, self.feature_indices(names), strict=True):
            if counter[ngram] > best.get(int(index), ("", 0))[1]:
                best[int(index)] = (ngram, counter[ngram])
        return {
            side: [best[int(i)][0] if int(i) in best else f"<bucket {int(i)}>" for i in indices]
            for side, indices in wanted.items()
        }

    # ------------------------------------------------------------ persistence

    def save(self, directory: Path, metadata: dict[str, Any]) -> None:
        if self.coef_ is None or self.idf_ is None:
            raise RuntimeError("model is not fitted")
        directory.mkdir(parents=True, exist_ok=True)
        empty = np.empty(0, dtype=np.float64)
        np.savez_compressed(
            directory / "model.npz",
            idf=self.idf_,
            coef=self.coef_,
            calib_x=empty if self.calib_x_ is None else self.calib_x_,
            calib_y=empty if self.calib_y_ is None else self.calib_y_,
        )
        document = {
            "kind": "baseline",
            "config": asdict(self.config),
            "intercept": self.intercept_,
            "calibrated": self.calib_x_ is not None,
            "metadata": metadata,
        }
        text = json.dumps(document, indent=2) + "\n"
        (directory / "metadata.json").write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Self:
        document = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        model = cls(BaselineConfig(**document["config"]))
        with np.load(directory / "model.npz", allow_pickle=False) as arrays:
            model.idf_ = arrays["idf"]
            model.coef_ = arrays["coef"]
            if document["calibrated"]:
                model.calib_x_ = arrays["calib_x"]
                model.calib_y_ = arrays["calib_y"]
        model.intercept_ = float(document["intercept"])
        model.metadata = dict(document["metadata"])
        return model
