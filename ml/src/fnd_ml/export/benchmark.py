"""How long the exported model takes to classify one article on the CPU."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from fnd_ml.common.truncation import InputSpec
from fnd_ml.inference.onnx_classifier import (
    TOKENIZER_NAME,
    encode_article,
    load_tokenizer,
    open_session,
    run_margins,
)

_SENTENCE = (
    "The central bank said on Tuesday that quarterly inflation remained unchanged from the "
    "previous quarter, according to official figures published after the committee's review. "
)


def _bert_style_spec(tokenizer: Tokenizer) -> InputSpec:
    """The usual window, with the special tokens looked up by their BERT names."""
    ids = {}
    for token in ("[CLS]", "[SEP]", "[PAD]"):
        found = tokenizer.token_to_id(token)
        if found is None:
            raise ValueError(f"the tokenizer has no {token} token; pass the model's InputSpec")
        ids[token] = found
    return InputSpec(
        max_length=512,
        max_title_tokens=64,
        head_share=0.75,
        cls_id=ids["[CLS]"],
        sep_id=ids["[SEP]"],
        pad_id=ids["[PAD]"],
    )


def benchmark(
    onnx_path: Path,
    tokenizer_dir: Path,
    *,
    spec: InputSpec | None = None,
    lengths: tuple[int, ...] = (128, 256, 512),
    runs: int = 100,
    warmup: int = 10,
    threads: int | None = None,
) -> dict[int, dict[str, float]]:
    """``p50_ms`` and ``p95_ms`` of one article of exactly ``length`` tokens, for each length.

    A run is what the API's classifier stage does: scrub and tokenize the text, cut it to the
    window, run the model. Without ``spec`` the special tokens are looked up by their BERT names.
    """
    session = open_session(onnx_path, threads)
    tokenizer = load_tokenizer(tokenizer_dir / TOKENIZER_NAME)
    if spec is None:
        spec = _bert_style_spec(tokenizer)

    results: dict[int, dict[str, float]] = {}
    for length in lengths:
        window = replace(spec, max_length=length)
        text = _SENTENCE
        while len(encode_article(tokenizer, window, None, text)[0]) < length:
            text += _SENTENCE
        timings = []
        for run in range(warmup + runs):
            started = time.perf_counter()
            ids, _ = encode_article(tokenizer, window, None, text)
            run_margins(session, [ids], window.pad_id)
            if run >= warmup:
                timings.append((time.perf_counter() - started) * 1000.0)
        results[length] = {
            "p50_ms": round(float(np.percentile(timings, 50)), 2),
            "p95_ms": round(float(np.percentile(timings, 95)), 2),
        }
    return results
