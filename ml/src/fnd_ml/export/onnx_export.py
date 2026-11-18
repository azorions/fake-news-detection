"""Write a fine-tuned model as an ONNX file. Needs PyTorch, transformers and the onnx packages."""

from __future__ import annotations

from pathlib import Path


def export_onnx(model_dir: Path, out_path: Path, *, opset: int = 17) -> Path:
    """Export the model in ``model_dir`` (a ``best/`` folder) to one file with its weights inside.

    Inputs ``input_ids`` and ``attention_mask`` (int64, any batch size and sequence length);
    output ``logits``. PyTorch's exporter builds the graph in a newer operator set and converts
    it down when ``opset`` is older.
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    class LogitsOnly(torch.nn.Module):
        def __init__(self, classifier: torch.nn.Module) -> None:
            super().__init__()
            self.classifier = classifier

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            logits: torch.Tensor = self.classifier(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            return logits

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float32)
    # Two rows, one of them padded: neither axis and no mask value is the same for every row.
    input_ids = torch.ones((2, 16), dtype=torch.long)
    attention_mask = torch.ones((2, 16), dtype=torch.long)
    attention_mask[1, 8:] = 0
    batch, sequence = torch.export.Dim("batch"), torch.export.Dim("sequence")
    program = torch.onnx.export(
        LogitsOnly(model).eval(),
        (input_ids, attention_mask),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={
            "input_ids": {0: batch, 1: sequence},
            "attention_mask": {0: batch, 1: sequence},
        },
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    if program is None:  # only the old exporter returns nothing
        raise RuntimeError("torch.onnx.export returned no program")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(out_path), external_data=False)
    return out_path
