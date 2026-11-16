"""Serving the fine-tuned transformer with ONNX Runtime and the tokenizers library. No PyTorch.

An artifact folder holds ``model.onnx``, ``tokenizer/tokenizer.json``, ``calibration.json`` (the
temperature), ``thresholds.json`` and ``metadata.json`` (version, base model and the settings that
turn an article into input ids).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Self

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from tokenizers import Tokenizer

from fnd_ml.calibration.temperature import apply_temperature, sigmoid
from fnd_ml.common.text import transformer_parts
from fnd_ml.common.truncation import InputSpec, pad_batch
from fnd_ml.inference.baseline_classifier import THRESHOLDS_NAME, ClassifierOutput

MODEL_NAME = "model.onnx"
TOKENIZER_NAME = "tokenizer.json"
TOKENIZER_DIR = "tokenizer"
CALIBRATION_NAME = "calibration.json"
METADATA_NAME = "metadata.json"
DEFAULT_THREADS = 4
INPUT_KEYS = ("max_length", "max_title_tokens", "head_share", "cls_id", "sep_id", "pad_id")


def load_tokenizer(path: Path) -> Tokenizer:
    """A ``tokenizer.json`` as training saved it. Cutting and padding are done by this package."""
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def open_session(model_path: Path, threads: int | None = DEFAULT_THREADS) -> ort.InferenceSession:
    """An ONNX Runtime session on the CPU; ``threads=None`` leaves the thread count to the runtime."""
    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def encode_article(
    tokenizer: Tokenizer, spec: InputSpec, title: str | None, text: str
) -> tuple[list[int], bool]:
    """Input ids of one article and whether it was cut: the same steps as in training."""
    clean_title, body = transformer_parts(title if isinstance(title, str) else None, text)
    title_ids = tokenizer.encode(clean_title, add_special_tokens=False).ids
    body_ids = tokenizer.encode(body, add_special_tokens=False).ids
    return spec.build(title_ids, body_ids)


def run_margins(
    session: ort.InferenceSession,
    rows: Sequence[Sequence[int] | NDArray[np.int32]],
    pad_id: int,
    batch_size: int = 8,
) -> NDArray[np.float64]:
    """``z1 - z0`` for every row of input ids. Rows are scored longest first, to pad little."""
    order = np.argsort([-len(row) for row in rows], kind="stable")
    margins = np.empty(len(rows), dtype=np.float64)
    for start in range(0, len(order), batch_size):
        chosen = order[start : start + batch_size]
        input_ids, mask = pad_batch([rows[int(index)] for index in chosen], pad_id)
        (logits,) = session.run(["logits"], {"input_ids": input_ids, "attention_mask": mask})
        margins[chosen] = logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64)
    return margins


class OnnxClassifier:
    def __init__(
        self,
        session: ort.InferenceSession,
        tokenizer: Tokenizer,
        spec: InputSpec,
        temperature: float,
        thresholds: tuple[float, float],
        metadata: dict[str, Any],
    ) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._thresholds = thresholds
        self.spec = spec
        self.temperature = temperature
        self.metadata = metadata
        self.version = str(metadata["version"])
        self.name = str(metadata["base_model"])

    @classmethod
    def load(cls, artifact_dir: Path, *, threads: int = DEFAULT_THREADS) -> Self:
        """Load an artifact folder. Raises when a file is missing or the model cannot be read."""

        def document(name: str) -> dict[str, Any]:
            loaded: dict[str, Any] = json.loads((artifact_dir / name).read_text(encoding="utf-8"))
            return loaded

        metadata = document(METADATA_NAME)
        thresholds = document(THRESHOLDS_NAME)
        return cls(
            session=open_session(artifact_dir / MODEL_NAME, threads),
            tokenizer=load_tokenizer(artifact_dir / TOKENIZER_DIR / TOKENIZER_NAME),
            spec=InputSpec(**{key: metadata[key] for key in INPUT_KEYS}),
            temperature=float(document(CALIBRATION_NAME)["temperature"]),
            thresholds=(float(thresholds["t_low"]), float(thresholds["t_high"])),
            metadata=metadata,
        )

    @property
    def thresholds(self) -> tuple[float, float]:
        """``(t_low, t_high)``: below ``t_low`` reads as reliable, from ``t_high`` up as high risk."""
        return self._thresholds

    def margins(
        self, articles: Iterable[tuple[str | None, str]], batch_size: int = 8
    ) -> NDArray[np.float64]:
        """``z1 - z0`` for each ``(title, text)``."""
        rows = [
            encode_article(self._tokenizer, self.spec, title, text)[0] for title, text in articles
        ]
        return run_margins(self._session, rows, self.spec.pad_id, batch_size)

    def probabilities(
        self, articles: Iterable[tuple[str | None, str]], batch_size: int = 8
    ) -> NDArray[np.float64]:
        """Calibrated ``P(unreliable)`` for each ``(title, text)``."""
        return apply_temperature(self.margins(articles, batch_size), self.temperature)

    def predict(self, title: str | None, text: str) -> ClassifierOutput:
        ids, cut = encode_article(self._tokenizer, self.spec, title, text)
        margin = run_margins(self._session, [ids], self.spec.pad_id)
        return ClassifierOutput(
            p_unreliable=float(apply_temperature(margin, self.temperature)[0]),
            p_raw=float(sigmoid(margin)[0]),
            version=self.version,
            truncated=cut,
        )
