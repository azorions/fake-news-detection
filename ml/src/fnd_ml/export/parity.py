"""Does the exported model say what the PyTorch model says?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fnd_ml.calibration.temperature import sigmoid
from fnd_ml.common.truncation import InputSpec
from fnd_ml.inference.onnx_classifier import (
    TOKENIZER_NAME,
    encode_article,
    load_tokenizer,
    open_session,
    run_margins,
)
from fnd_ml.models.transformer import INPUT_CONFIG_NAME, predict_logits


def load_spec(model_dir: Path) -> InputSpec:
    """The input settings a ``best/`` folder was trained with."""
    return InputSpec(**json.loads((model_dir / INPUT_CONFIG_NAME).read_text(encoding="utf-8")))


def encode_frame(model_dir: Path, frame: pd.DataFrame) -> list[list[int]]:
    """Input ids of every row of ``frame``, built the way the API builds them."""
    spec, tokenizer = load_spec(model_dir), load_tokenizer(model_dir / TOKENIZER_NAME)
    articles = zip(frame["title"].tolist(), frame["text"].tolist(), strict=True)
    return [encode_article(tokenizer, spec, title, text)[0] for title, text in articles]


def parity(
    model_dir: Path, onnx_path: Path, frame: pd.DataFrame, *, n: int = 500, seed: int = 13
) -> dict[str, float]:
    """The largest difference in ``P(unreliable)`` between PyTorch and ONNX Runtime on ``n`` rows.

    The two sides also build their inputs separately (training code against serving code), so a
    difference in tokenizing or cutting shows up here as well.
    """
    sample = frame.sample(n=min(n, len(frame)), random_state=seed)
    rows = encode_frame(model_dir, sample)
    exported = sigmoid(run_margins(open_session(onnx_path), rows, load_spec(model_dir).pad_id))
    original = sigmoid(predict_logits(model_dir, sample))
    return {
        "max_abs_probability_diff": float(np.abs(exported - original).max()),
        "n": float(len(sample)),
    }
