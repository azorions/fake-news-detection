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
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from fnd_ml.calibration.isotonic import apply_isotonic, fit_isotonic
from fnd_ml.common.tokens import MAX_TOKENS, analyze, analyze_chars

if TYPE_CHECKING:
    import pandas as pd

    from fnd_ml.models.scorer import Scorer

_BATCH = 2000


@dataclass(frozen=True)
class BaselineConfig:
    n_features: int = 2**20
    max_tokens: int = MAX_TOKENS
    min_df: int = 3
    C: float = 4.0
    max_iter: int = 300
    char_ngrams: bool = False  # also hash character 3-5-grams inside word boundaries


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

    def _analyze(self, text: str) -> list[str]:
        """The feature strings of ``text``. Character n-grams carry a ``#`` so that the 3-gram
        "now" inside "known" does not share a bucket with the word "now"."""
        features = analyze(text, self.config.max_tokens)
        if self.config.char_ngrams:
            features += [f"#{gram}" for gram in analyze_chars(text, self.config.max_tokens)]
        return features

    def _counts(self, texts: Iterable[str]) -> sp.csr_matrix:
        chunks = [
            self._hasher.transform([self._analyze(text) for text in batch])
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

    def fit_features(self, texts: Iterable[str]) -> sp.csr_matrix:
        """Learn the IDF weights from ``texts`` and return their weighted rows."""
        counts = self._counts(texts)
        n_docs = counts.shape[0]
        document_frequency = np.bincount(counts.indices, minlength=self.config.n_features)
        idf = np.log((1 + n_docs) / (1 + document_frequency)) + 1.0
        idf[document_frequency < self.config.min_df] = 0.0
        self.idf_ = idf.astype(np.float32)
        return self._weight(counts)

    def fit(self, texts: Iterable[str], y: ArrayLike) -> Self:
        labels = np.asarray(y, dtype=int)
        features = self.fit_features(texts)
        if features.shape[0] != len(labels):
            raise ValueError(f"got {features.shape[0]} texts but {len(labels)} labels")

        classifier = LogisticRegression(
            C=self.config.C, class_weight="balanced", max_iter=self.config.max_iter
        )
        classifier.fit(features, labels)
        self.coef_ = classifier.coef_.ravel().astype(np.float64)
        self.intercept_ = float(classifier.intercept_[0])
        self.calib_x_ = self.calib_y_ = None
        return self

    def calibrate_scores(self, raw: ArrayLike, y: ArrayLike) -> Self:
        """Fit the isotonic calibrator on raw probabilities from ``predict_raw``."""
        self.calib_x_, self.calib_y_ = fit_isotonic(raw, y)
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
        if self.calib_x_ is None or self.calib_y_ is None:
            return apply_isotonic(raw, None)
        return apply_isotonic(raw, (self.calib_x_, self.calib_y_))

    def predict_proba(self, texts: Iterable[str]) -> NDArray[np.float64]:
        """Calibrated P(unreliable), clipped to [0.01, 0.99]."""
        return self.to_calibrated(self.predict_raw(texts))

    # ------------------------------------------------------------ inspection

    def top_features(self, sample_texts: Iterable[str], k: int = 50) -> dict[str, list[str]]:
        """Names of the ``k`` strongest buckets per class, named from ``sample_texts``.

        A bucket is named after the most frequent n-gram in the sample that hashes into it; a
        bucket no sample n-gram hashes into is shown as ``<bucket N>``.
        """
        if self.coef_ is None or self.idf_ is None:
            raise RuntimeError("model is not fitted")
        strength = self.coef_ * self.idf_
        order = np.argsort(strength)
        wanted = {"unreliable": order[::-1][:k], "reliable": order[:k]}
        to_name = {int(i) for indices in wanted.values() for i in indices}

        counter: Counter[str] = Counter()
        for text in sample_texts:
            counter.update(self._analyze(text))
        best: dict[int, tuple[str, int]] = {}
        for names in itertools.batched(counter, 100_000):
            for ngram, index in zip(names, self.feature_indices(list(names)), strict=True):
                bucket = int(index)
                if bucket in to_name and counter[ngram] > best.get(bucket, ("", 0))[1]:
                    best[bucket] = (ngram, counter[ngram])
        return {
            side: [best[int(i)][0] if int(i) in best else f"<bucket {int(i)}>" for i in indices]
            for side, indices in wanted.items()
        }

    # ------------------------------------------------------------ persistence

    def save(self, directory: Path, metadata: dict[str, Any], kind: str = "baseline") -> None:
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
            "kind": kind,
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


# ------------------------------------------------------------ the common scorer interface


def _as_scorer(model: BaselineModel) -> Scorer:
    def score(frame: pd.DataFrame) -> NDArray[np.float64]:
        return model.predict_proba(frame["model_text"])

    return score


def fit_scorer(
    train: pd.DataFrame,
    val: pd.DataFrame,
    calib: pd.DataFrame,
    config: BaselineConfig | None = None,
) -> Scorer:
    """Fit on ``train`` and calibrate on ``calib``. ``val`` is not used: nothing is selected."""
    model = BaselineModel(config).fit(train["model_text"], train["label"])
    model.calibrate(calib["model_text"], calib["label"])
    return _as_scorer(model)


def load_scorer(artifact_dir: Path) -> Scorer:
    return _as_scorer(BaselineModel.load(artifact_dir))
