"""Serving wrapper around a saved baseline artifact: one ``(title, text)`` in, one probability out."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from fnd_ml.calibration.thresholds import DEFAULT_HIGH, DEFAULT_LOW
from fnd_ml.common.text import model_input
from fnd_ml.common.tokens import word_tokens
from fnd_ml.models.baseline import BaselineModel

THRESHOLDS_NAME = "thresholds.json"


@dataclass(frozen=True)
class ClassifierOutput:
    p_unreliable: float  # calibrated
    p_raw: float  # before calibration
    version: str
    truncated: bool  # the text has more words than the model reads


class BaselineClassifier:
    def __init__(self, model: BaselineModel, version: str, thresholds: tuple[float, float]) -> None:
        self.model = model
        self.version = version
        self._thresholds = thresholds

    @classmethod
    def load(cls, artifact_dir: Path) -> Self:
        """Load ``model.npz``, ``metadata.json`` and, when present, ``thresholds.json``."""
        model = BaselineModel.load(artifact_dir)
        version = str(model.metadata.get("version") or artifact_dir.name)
        thresholds = (DEFAULT_LOW, DEFAULT_HIGH)
        path = artifact_dir / THRESHOLDS_NAME
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            thresholds = (float(document["t_low"]), float(document["t_high"]))
        return cls(model, version, thresholds)

    @property
    def thresholds(self) -> tuple[float, float]:
        """``(t_low, t_high)``: below ``t_low`` reads as reliable, from ``t_high`` up as high risk."""
        return self._thresholds

    def predict(self, title: str | None, text: str) -> ClassifierOutput:
        model_text, _ = model_input(title, text)
        raw = self.model.predict_raw([model_text])
        limit = self.model.config.max_tokens
        return ClassifierOutput(
            p_unreliable=float(self.model.to_calibrated(raw)[0]),
            p_raw=float(raw[0]),
            version=self.version,
            truncated=len(word_tokens(model_text, limit + 1)) > limit,
        )
