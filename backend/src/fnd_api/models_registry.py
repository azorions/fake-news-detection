"""The models the service has loaded, resolved through the artifacts manifest.

The transformer gives the verdict when its artifact loads; the baseline answers otherwise and
supplies the word weights of the explanation; the satire scorer only adds a tag. An artifact that
is missing or cannot be loaded leaves its slot empty: the service starts with what it has.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from fnd_api.logging import get_logger
from fnd_ml.common.artifacts import resolve
from fnd_ml.inference.baseline_classifier import BaselineClassifier
from fnd_ml.inference.onnx_classifier import OnnxClassifier
from fnd_ml.inference.satire_scorer import SatireScorer

log = get_logger("fnd_api.models")


def _load[Model](artifacts_dir: Path, kind: str, load: Callable[[Path], Model]) -> Model | None:
    directory = resolve(artifacts_dir, kind)
    if directory is None:
        return None
    try:
        return load(directory)
    except Exception:  # a damaged artifact must not keep the service from starting
        log.error("model_load_failed", kind=kind, path=str(directory), exc_info=True)
        return None


@dataclass(frozen=True)
class ModelRegistry:
    baseline: BaselineClassifier | None = None
    transformer: OnnxClassifier | None = None
    satire: SatireScorer | None = None

    @classmethod
    def load(cls, artifacts_dir: Path, *, onnx_threads: int = 4) -> Self:
        """Load the active artifacts. One that is missing or fails to load stays ``None``."""
        return cls(
            baseline=_load(artifacts_dir, "baseline", BaselineClassifier.load),
            transformer=_load(
                artifacts_dir,
                "transformer",
                lambda directory: OnnxClassifier.load(directory, threads=onnx_threads),
            ),
            satire=_load(artifacts_dir, "satire", SatireScorer.load),
        )

    @property
    def primary(self) -> BaselineClassifier | OnnxClassifier | None:
        """The model that gives the verdict: the transformer, or the baseline without it."""
        return self.transformer or self.baseline

    def thresholds(self, version: str) -> tuple[float, float]:
        """The verdict thresholds of the loaded classifier with this version."""
        for model in (self.transformer, self.baseline):
            if model is not None and model.version == version:
                return model.thresholds
        raise KeyError(version)

    @property
    def ready(self) -> bool:
        return self.primary is not None

    @property
    def versions(self) -> dict[str, str]:
        """Kind -> version of every loaded model."""
        loaded = {"baseline": self.baseline, "transformer": self.transformer, "satire": self.satire}
        return {kind: model.version for kind, model in loaded.items() if model is not None}
