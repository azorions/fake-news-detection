"""Write ``tests/fixtures/tiny_transformer``: an artifact folder the API tests can serve.

The "model" is an embedding table, a masked mean and a linear layer, built by hand: sensational
words push the logit towards "unreliable", reporting words towards "reliable", every other word
counts for nothing. It has the files and the input and output names of a real exported
transformer at a few kilobytes, so the serving code is tested without PyTorch or a download.

Run from the repository root (needs the ``train`` extra for the onnx package)::

    uv run --no-sync python scripts/make_tiny_onnx_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

DEFAULT_OUT = Path("tests/fixtures/tiny_transformer")
VERSION = "v1-tinyfix"
SPECIAL = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
SENSATIONAL = (
    *("shocking", "breaking", "exposed", "bombshell", "believe", "hiding", "truth", "rigged"),
    *("miracle", "cure", "control", "mind", "destroys", "evidence", "hoax", "secret", "wake"),
    *("deleted", "shock", "won", "want", "know", "share", "people"),
)
REPORTING = (
    *("said", "according", "ministry", "spokesperson", "bank", "agency", "monday", "tuesday"),
    *("wednesday", "thursday", "friday", "budget", "inflation", "plan", "funding", "percent"),
    *("approved", "committee", "review", "discussed", "session", "unchanged", "quarter"),
    *("statement", "figures", "minutes", "briefing", "official", "published"),
)
SCALE = 20.0  # logit margin = SCALE * (share of sensational tokens - share of reporting tokens)
TEMPERATURE = 2.0
OPSET = 17


def _tokenizer() -> Tokenizer:
    vocab = {token: index for index, token in enumerate([*SPECIAL, *SENSATIONAL, *REPORTING])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


def _model() -> onnx.ModelProto:
    embedding = np.zeros((len(SPECIAL) + len(SENSATIONAL) + len(REPORTING), 2), dtype=np.float32)
    first_reporting = len(SPECIAL) + len(SENSATIONAL)
    embedding[len(SPECIAL) : first_reporting, 1] = 1.0  # column 1 counts sensational tokens
    embedding[first_reporting:, 0] = 1.0  # column 0 counts reporting tokens
    weight = (SCALE * np.eye(2)).astype(np.float32)
    bias = np.zeros(2, dtype=np.float32)

    shape = ["batch", "sequence"]
    nodes = [
        helper.make_node("Gather", ["embedding", "input_ids"], ["embedded"]),
        helper.make_node("Cast", ["attention_mask"], ["mask"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask", "last_axis"], ["mask_3d"]),
        helper.make_node("Mul", ["embedded", "mask_3d"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "sequence_axis"], ["summed"], keepdims=0),
        helper.make_node("ReduceSum", ["mask_3d", "sequence_axis"], ["count"], keepdims=0),
        helper.make_node("Div", ["summed", "count"], ["mean"]),
        helper.make_node("MatMul", ["mean", "weight"], ["projected"]),
        helper.make_node("Add", ["projected", "bias"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_fixture",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, shape),
            helper.make_tensor_value_info("attention_mask", TensorProto.INT64, shape),
        ],
        outputs=[helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
        initializer=[
            numpy_helper.from_array(embedding, "embedding"),
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(bias, "bias"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), "last_axis"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "sequence_axis"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="fnd-tiny-fixture",
        opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    return model


def write_fixture(out: Path = DEFAULT_OUT) -> Path:
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    onnx.save(_model(), str(out / "model.onnx"))
    _tokenizer().save(str(out / "tokenizer" / "tokenizer.json"))
    documents = {
        "calibration.json": {"method": "temperature", "temperature": TEMPERATURE},
        "thresholds.json": {"t_low": 0.35, "t_high": 0.65},
        "metadata.json": {
            "version": VERSION,
            "kind": "transformer",
            "base_model": "tiny-fixture",
            "variant": "fp32",
            "max_length": 512,
            "max_title_tokens": 64,
            "head_share": 0.75,
            "cls_id": SPECIAL.index("[CLS]"),
            "sep_id": SPECIAL.index("[SEP]"),
            "pad_id": SPECIAL.index("[PAD]"),
        },
    }
    for name, document in documents.items():
        (out / name).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    target = write_fixture(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
    size = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
    print(f"wrote {target} ({size} bytes)")
