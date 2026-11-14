"""The models the service has loaded, resolved through the artifacts manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from fnd_ml.common.artifacts import resolve
from fnd_ml.inference.baseline_classifier import BaselineClassifier


@dataclass(frozen=True)
class ModelRegistry:
    baseline: BaselineClassifier | None = None

    @classmethod
    def load(cls, artifacts_dir: Path) -> Self:
        """Load the active artifacts. A missing artifact leaves its slot ``None`` (not ready)."""
        directory = resolve(artifacts_dir, "baseline")
        return cls(baseline=BaselineClassifier.load(directory) if directory else None)

    @property
    def ready(self) -> bool:
        return self.baseline is not None

    @property
    def versions(self) -> dict[str, str]:
        return {"baseline": self.baseline.version} if self.baseline else {}
