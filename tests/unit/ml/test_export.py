"""Export of a fine-tuned model: ONNX file, int8 file, parity, latency and the served artifact.

The slow tests train the toy model once and export it. PyTorch is imported inside them.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from fnd_ml.calibration.run import calibrate_run
from fnd_ml.common.artifacts import resolve
from fnd_ml.data.build import MANIFEST_NAME
from fnd_ml.data.download import sha256_file
from fnd_ml.export import pipeline
from fnd_ml.export.benchmark import benchmark
from fnd_ml.export.onnx_export import export_onnx
from fnd_ml.export.parity import encode_frame, load_spec, parity
from fnd_ml.export.pipeline import ExportError, choose_variant, export_run
from fnd_ml.export.quantize import quantize_int8
from fnd_ml.inference.onnx_classifier import OnnxClassifier, open_session, run_margins
from fnd_ml.models import transformer
from fnd_ml.models.transformer import load_scorer, predict_logits, train
from tests.fixtures.processed import write_processed
from tests.fixtures.synthetic import make_corpus
from tests.fixtures.tiny_deberta import split, tiny_config

# ---------------------------------------------------------------- the shipping rule


def test_int8_ships_when_it_is_as_good_and_faster() -> None:
    assert choose_variant({"fp32": 0.900, "int8": 0.890}, {"fp32": 400.0, "int8": 250.0}) == "int8"


def test_fp32_ships_when_int8_costs_more_than_a_point() -> None:
    assert choose_variant({"fp32": 0.900, "int8": 0.889}, {"fp32": 400.0, "int8": 250.0}) == "fp32"


def test_fp32_ships_when_int8_is_not_faster() -> None:
    assert choose_variant({"fp32": 0.900, "int8": 0.900}, {"fp32": 400.0, "int8": 400.0}) == "fp32"


def test_export_refuses_a_run_that_was_not_calibrated(tmp_path: Path) -> None:
    processed, lock = write_processed(tmp_path, make_corpus(400))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {"data_manifest_sha256": sha256_file(processed / MANIFEST_NAME)}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ExportError, match="fnd calibrate"):
        export_run(run_dir, processed, tmp_path / "artifacts", lock_path=lock)

    assert not (tmp_path / "artifacts").exists()


# ---------------------------------------------------------------- a trained toy model


@pytest.fixture(scope="module")
def cpu() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(transformer, "_device", lambda requested=None: requested or "cpu")
        yield


@pytest.fixture(scope="module")
def run_dir(
    cpu: None,
    tiny_model: Path,
    tiny_data: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A trained and calibrated run of the toy model."""
    processed, lock = tiny_data
    directory = tmp_path_factory.mktemp("export") / "run"
    train(tiny_config(tiny_model, epochs=2), processed, directory, lock_path=lock)
    calibrate_run(directory, processed, lock_path=lock)
    return directory


@pytest.fixture(scope="module")
def fp32(run_dir: Path) -> Path:
    return export_onnx(run_dir / "best", run_dir / "files" / "fp32.onnx")


@pytest.mark.slow
def test_onnx_runtime_gives_pytorch_logits_for_any_batch_and_length(
    run_dir: Path, fp32: Path
) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification

    best = run_dir / "best"
    model = AutoModelForSequenceClassification.from_pretrained(best).eval()
    words = Tokenizer.from_file(str(best / "tokenizer.json")).get_vocab_size()
    session = open_session(fp32)
    rng = np.random.default_rng(0)

    for batch, length in ((1, 5), (3, 5), (1, 40), (3, 40)):
        input_ids = rng.integers(4, words, size=(batch, length))
        mask = np.ones((batch, length), dtype=np.int64)
        if batch > 1:  # the last row is shorter than the others
            input_ids[-1, length // 2 :] = 0
            mask[-1, length // 2 :] = 0
        with torch.inference_mode():
            expected = model(
                input_ids=torch.from_numpy(input_ids), attention_mask=torch.from_numpy(mask)
            ).logits.numpy()

        (logits,) = session.run(["logits"], {"input_ids": input_ids, "attention_mask": mask})

        np.testing.assert_allclose(logits, expected, atol=1e-4)

    assert [(i.name, i.type) for i in session.get_inputs()] == [
        ("input_ids", "tensor(int64)"),
        ("attention_mask", "tensor(int64)"),
    ]
    assert not fp32.with_name(fp32.name + ".data").exists()  # the weights are in the file


@pytest.mark.slow
def test_the_int8_file_is_smaller_and_still_answers(run_dir: Path, fp32: Path) -> None:
    int8 = quantize_int8(fp32, run_dir / "files" / "int8.onnx")

    assert int8.stat().st_size < fp32.stat().st_size
    rows = encode_frame(run_dir / "best", split("val"))
    margins = run_margins(open_session(int8), rows, load_spec(run_dir / "best").pad_id)
    assert margins.shape == (len(rows),)
    assert np.isfinite(margins).all()


@pytest.mark.slow
def test_the_exported_model_gives_pytorch_probabilities(run_dir: Path, fp32: Path) -> None:
    result = parity(run_dir / "best", fp32, split("val"), n=40)

    assert result["n"] == 40
    assert result["max_abs_probability_diff"] <= 0.001


@pytest.mark.slow
def test_serving_builds_the_same_inputs_as_training(run_dir: Path, fp32: Path) -> None:
    best, val = run_dir / "best", split("val")

    served = run_margins(open_session(fp32), encode_frame(best, val), load_spec(best).pad_id)

    np.testing.assert_allclose(served, predict_logits(best, val), atol=1e-4)


@pytest.mark.slow
def test_the_benchmark_reports_p50_and_p95_for_each_length(run_dir: Path, fp32: Path) -> None:
    best = run_dir / "best"

    result = benchmark(fp32, best, spec=load_spec(best), lengths=(16, 48), runs=5, warmup=1)

    assert sorted(result) == [16, 48]
    for timing in result.values():
        assert timing["p95_ms"] >= timing["p50_ms"] > 0


@pytest.mark.slow
def test_export_run_writes_and_registers_the_served_artifact(
    run_dir: Path, tiny_model: Path, tiny_data: tuple[Path, Path], tmp_path: Path
) -> None:
    processed, lock = tiny_data
    artifacts = tmp_path / "artifacts"

    metadata = export_run(run_dir, processed, artifacts, lock_path=lock, val_rows=40, runs=5)

    artifact = artifacts / "transformer" / metadata["version"]
    assert resolve(artifacts, "transformer") == artifact
    assert sorted(path.relative_to(artifact).as_posix() for path in artifact.rglob("*.*")) == [
        "calibration.json",
        "metadata.json",
        "model.onnx",
        "thresholds.json",
        "tokenizer/tokenizer.json",
    ]
    assert json.loads((artifact / "metadata.json").read_text(encoding="utf-8")) == metadata
    assert metadata["version"] == "v1-" + sha256_file(artifact / "model.onnx")[:8]
    assert metadata["variant"] in ("fp32", "int8")
    assert metadata["base_model"] == str(tiny_model)
    assert metadata["max_length"] == 48
    assert metadata["data_manifest_sha256"] == sha256_file(processed / MANIFEST_NAME)
    assert metadata["parity"]["max_abs_probability_diff"] <= 0.001
    assert set(metadata["latency_ms"]["fp32"]) == {"48"}  # the toy model reads 48 tokens
    assert metadata["latency_ms"]["threads"] == 4
    assert metadata["gate_exceptions"] == []

    classifier = OnnxClassifier.load(artifact)
    assert classifier.version == metadata["version"]
    test = split("test")
    served = classifier.probabilities(zip(test["title"], test["text"], strict=True))
    assert served.min() >= 0.01
    assert served.max() <= 0.99
    labels = test["label"].to_numpy()
    assert served[labels == 1].mean() > served[labels == 0].mean() + 0.5
    if metadata["variant"] == "fp32":
        np.testing.assert_allclose(served, load_scorer(run_dir)(test), atol=1e-3)


@pytest.mark.slow
def test_export_stops_when_the_file_does_not_match_pytorch(
    run_dir: Path,
    tiny_data: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed, lock = tiny_data
    monkeypatch.setattr(
        pipeline, "parity", lambda *_, **__: {"max_abs_probability_diff": 0.2, "n": 40.0}
    )

    with pytest.raises(ExportError, match=r"differs from PyTorch by 0\.2000"):
        export_run(run_dir, processed, tmp_path / "artifacts", lock_path=lock, val_rows=40, runs=2)

    assert not (tmp_path / "artifacts").exists()
