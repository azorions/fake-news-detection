"""Serving wrapper around a saved satire artifact: how much a headline reads like satire."""

from __future__ import annotations

from pathlib import Path
from typing import Self

from fnd_ml.common.text import model_input
from fnd_ml.models.baseline import BaselineModel

HEADLINE_WORDS = 40  # a text without a title is scored on this many leading words


class SatireScorer:
    def __init__(self, model: BaselineModel, version: str) -> None:
        self.model = model
        self.version = version

    @classmethod
    def load(cls, artifact_dir: Path) -> Self:
        model = BaselineModel.load(artifact_dir)
        return cls(model, str(model.metadata.get("version") or artifact_dir.name))

    def score(self, title: str | None, text: str) -> float:
        """Calibrated P(satire) of the title, or of the first words of ``text`` without one."""
        headline = (title or "").strip() or " ".join(text.split()[:HEADLINE_WORDS])
        model_text, _ = model_input(None, headline)
        return float(self.model.predict_proba([model_text])[0])
