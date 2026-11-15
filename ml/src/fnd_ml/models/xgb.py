"""Comparison model: boosted trees over stylometric features and an SVD of the hashed TF-IDF.

It answers one question in the model comparison: how far do writing style and coarse topic get
without reading the words in order? It is never served, so the API loads without ``xgboost``.

The TF-IDF here uses 2**17 hash buckets, not the baseline's 2**20: the SVD projection is a dense
``components x buckets`` matrix, which at 2**20 buckets would need several gigabytes to fit.

The artifact is ``booster.json`` (XGBoost's own format), ``features.npz`` and ``metadata.json``.
Nothing is pickled.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier

from fnd_ml.calibration.isotonic import Knots, apply_isotonic, fit_isotonic
from fnd_ml.calibration.thresholds import pick_thresholds
from fnd_ml.common.artifacts import register
from fnd_ml.common.features import FEATURE_NAMES, stylometric
from fnd_ml.common.tokens import MAX_TOKENS
from fnd_ml.data.build import DEFAULT_LOCK, assert_frozen
from fnd_ml.eval.report import write_report
from fnd_ml.models.baseline import BaselineConfig, BaselineModel
from fnd_ml.models.scorer import Scorer
from fnd_ml.models.train_baseline import (
    REPORT_SUFFIX,
    add_test_results,
    iter_texts,
    load_meta,
    model_version,
    save_thresholds,
    split_metrics,
)

_BATCH = 4000
_TOP_IMPORTANCES = 20


@dataclass(frozen=True)
class XgbConfig:
    n_features: int = 2**17
    max_tokens: int = MAX_TOKENS
    min_df: int = 3
    svd_components: int = 256
    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.08
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 30
    seed: int = 13


class XgbModel:
    def __init__(self, config: XgbConfig | None = None) -> None:
        self.config = config or XgbConfig()
        self.metadata: dict[str, Any] = {}
        self._tfidf = BaselineModel(
            BaselineConfig(
                n_features=self.config.n_features,
                max_tokens=self.config.max_tokens,
                min_df=self.config.min_df,
            )
        )
        self.components_: NDArray[np.float32] | None = None
        self.classifier_: XGBClassifier | None = None
        self.calibration_: Knots | None = None

    # ------------------------------------------------------------ features

    @property
    def feature_names(self) -> list[str]:
        n_components = 0 if self.components_ is None else len(self.components_)
        return [*FEATURE_NAMES, *(f"svd_{i}" for i in range(n_components))]

    def _stack(self, texts: Sequence[str], weighted: sp.csr_matrix) -> NDArray[np.float32]:
        """Stylometric columns, then the TF-IDF rows projected onto the SVD components."""
        if self.components_ is None:
            raise RuntimeError("model is not fitted")
        style = np.array([stylometric(text) for text in texts], dtype=np.float32)
        topics = np.asarray(weighted @ self.components_.T, dtype=np.float32)
        return np.hstack([style.reshape(len(texts), len(FEATURE_NAMES)), topics])

    def features(self, texts: Sequence[str]) -> NDArray[np.float32]:
        parts = [
            self._stack(batch, self._tfidf.transform(batch))
            for batch in itertools.batched(texts, _BATCH)
        ]
        if not parts:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        return np.vstack(parts)

    # ------------------------------------------------------------ training

    def fit(
        self, texts: Sequence[str], y: ArrayLike, val_texts: Sequence[str], val_y: ArrayLike
    ) -> Self:
        """Fit on ``texts``; the number of trees is chosen by early stopping on the val texts."""
        labels = np.asarray(y, dtype=int)
        if len(texts) != len(labels):
            raise ValueError(f"got {len(texts)} texts but {len(labels)} labels")

        weighted = self._tfidf.fit_features(texts)
        n_components = min(self.config.svd_components, min(weighted.shape) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=self.config.seed).fit(weighted)
        self.components_ = svd.components_.astype(np.float32)

        classifier = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            tree_method="hist",
            early_stopping_rounds=self.config.early_stopping_rounds,
            eval_metric="logloss",
            random_state=self.config.seed,
        )
        classifier.fit(
            self._stack(texts, weighted),
            labels,
            eval_set=[(self.features(val_texts), np.asarray(val_y, dtype=int))],
            verbose=False,
        )
        self.classifier_ = classifier
        self.calibration_ = None
        return self

    def calibrate_scores(self, raw: ArrayLike, y: ArrayLike) -> Self:
        """Fit the isotonic calibrator on raw probabilities from ``predict_raw``."""
        self.calibration_ = fit_isotonic(raw, y)
        return self

    def calibrate(self, texts: Sequence[str], y: ArrayLike) -> Self:
        return self.calibrate_scores(self.predict_raw(texts), y)

    # ------------------------------------------------------------ prediction

    def predict_raw(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Uncalibrated P(unreliable), from the trees up to the early-stopping round."""
        if self.classifier_ is None:
            raise RuntimeError("model is not fitted")
        probabilities = self.classifier_.predict_proba(self.features(texts))
        raw: NDArray[np.float64] = probabilities[:, 1].astype(np.float64)
        return raw

    def to_calibrated(self, raw: ArrayLike) -> NDArray[np.float64]:
        return apply_isotonic(raw, self.calibration_)

    def predict_proba(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Calibrated P(unreliable), clipped to [0.01, 0.99]."""
        return self.to_calibrated(self.predict_raw(texts))

    def importances(self, k: int = _TOP_IMPORTANCES) -> list[dict[str, Any]]:
        """The ``k`` features with the largest mean gain over the splits that use them."""
        if self.classifier_ is None:
            raise RuntimeError("model is not fitted")
        names = self.feature_names
        booster = self.classifier_.get_booster()
        gains = cast(dict[str, float], booster.get_score(importance_type="gain"))
        ranked = sorted(gains.items(), key=lambda item: item[1], reverse=True)[:k]
        return [{"feature": names[int(key[1:])], "gain": round(gain, 4)} for key, gain in ranked]

    # ------------------------------------------------------------ persistence

    def save(self, directory: Path, metadata: dict[str, Any]) -> None:
        if self.classifier_ is None or self.components_ is None or self._tfidf.idf_ is None:
            raise RuntimeError("model is not fitted")
        directory.mkdir(parents=True, exist_ok=True)
        self.classifier_.save_model(directory / "booster.json")
        empty = np.empty(0, dtype=np.float64)
        calib_x, calib_y = self.calibration_ or (empty, empty)
        np.savez_compressed(
            directory / "features.npz",
            idf=self._tfidf.idf_,
            components=self.components_,
            calib_x=calib_x,
            calib_y=calib_y,
        )
        document = {
            "kind": "xgb",
            "config": asdict(self.config),
            "calibrated": self.calibration_ is not None,
            "metadata": metadata,
        }
        text = json.dumps(document, indent=2) + "\n"
        (directory / "metadata.json").write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> Self:
        document = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        model = cls(XgbConfig(**document["config"]))
        with np.load(directory / "features.npz", allow_pickle=False) as arrays:
            model._tfidf.idf_ = arrays["idf"]
            model.components_ = arrays["components"]
            if document["calibrated"]:
                model.calibration_ = (arrays["calib_x"], arrays["calib_y"])
        classifier = XGBClassifier()
        classifier.load_model(directory / "booster.json")
        model.classifier_ = classifier
        model.metadata = dict(document["metadata"])
        return model


# ------------------------------------------------------------ the common scorer interface


def _as_scorer(model: XgbModel) -> Scorer:
    def score(frame: pd.DataFrame) -> NDArray[np.float64]:
        return model.predict_proba(frame["model_text"].tolist())

    return score


def fit_scorer(
    train: pd.DataFrame,
    val: pd.DataFrame,
    calib: pd.DataFrame,
    config: XgbConfig | None = None,
) -> Scorer:
    """Fit on ``train``, stop early on ``val``, calibrate on ``calib``."""
    model = XgbModel(config).fit(
        train["model_text"].tolist(), train["label"], val["model_text"].tolist(), val["label"]
    )
    model.calibrate(calib["model_text"].tolist(), calib["label"])
    return _as_scorer(model)


def load_scorer(artifact_dir: Path) -> Scorer:
    return _as_scorer(XgbModel.load(artifact_dir))


# ------------------------------------------------------------ training command


def train_xgb(
    processed_dir: Path,
    artifacts_dir: Path,
    reports_dir: Path,
    config: XgbConfig | None = None,
    *,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    """Train on the frozen splits, calibrate, save the artifact and write the report."""
    config = config or XgbConfig()
    data_sha = assert_frozen(processed_dir, lock_path)
    splits = processed_dir / "splits"
    meta = {s: load_meta(processed_dir, s) for s in ("train", "val", "calib", "test")}
    labels = {split: frame["label"].to_numpy() for split, frame in meta.items()}

    def texts(split: str) -> list[str]:
        return list(iter_texts(splits / f"{split}.parquet"))

    val_texts = texts("val")
    model = XgbModel(config).fit(texts("train"), labels["train"], val_texts, labels["val"])
    raw = {"val": model.predict_raw(val_texts)}
    raw |= {split: model.predict_raw(texts(split)) for split in ("calib", "test")}
    model.calibrate_scores(raw["calib"], labels["calib"])
    calibrated = {split: model.to_calibrated(scores) for split, scores in raw.items()}
    thresholds = pick_thresholds(labels["calib"], calibrated["calib"])
    version = model_version(config, data_sha)

    assert model.classifier_ is not None
    report: dict[str, Any] = {
        "model": "xgb",
        "version": version,
        "config": asdict(config),
        "data_manifest_sha256": data_sha,
        "rows": {split: len(frame) for split, frame in meta.items()},
        "metrics": {s: split_metrics(labels[s], raw[s], calibrated[s]) for s in ("val", "calib")},
        "thresholds": thresholds.to_dict(),
        "trees": int(model.classifier_.best_iteration) + 1,
        "feature_importance": model.importances(),
    }
    add_test_results(report, meta["test"], raw["test"], calibrated["test"], thresholds)

    artifact = artifacts_dir / "xgb" / version
    model.save(
        artifact,
        {
            "version": version,
            "data_manifest_sha256": data_sha,
            "test_metrics": report["metrics"]["test"]["calibrated"],
        },
    )
    save_thresholds(artifact, thresholds)
    register(artifacts_dir, "xgb", version, f"xgb/{version}")
    write_report(report, reports_dir, f"xgb_{REPORT_SUFFIX}")
    return report
