"""Dynamic int8 quantization of an exported model. Needs the onnx package."""

from __future__ import annotations

from pathlib import Path


def quantize_int8(fp32_path: Path, out_path: Path) -> Path:
    """Store the weights of the matrix products as int8; activations are quantized while running.

    The file gets about four times smaller and usually faster on a CPU. What it costs in accuracy
    has to be measured: see the shipping rule in ``fnd_ml.export.pipeline``.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(fp32_path), str(out_path), weight_type=QuantType.QInt8)
    return out_path
