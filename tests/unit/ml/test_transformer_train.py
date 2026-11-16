"""Fine-tuning end to end on the CPU: a tiny random DeBERTa and a word-level tokenizer.

PyTorch is imported inside the fixtures, so collecting this file does not load it.
"""

import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from fnd_ml.calibration.run import calibrate_run
from fnd_ml.calibration.temperature import apply_temperature
from fnd_ml.models import transformer
from fnd_ml.models.transformer import (
    TrainConfig,
    benchmark_precision,
    fit_scorer,
    load_scorer,
    predict_logits,
    train,
)
from tests.fixtures.processed import BOUNDS, two_dataset_corpus, write_processed

pytestmark = pytest.mark.slow

STEPS_PER_EPOCH = 30  # 240 train rows in batches of 8


def _corpus() -> pd.DataFrame:
    """The synthetic two-dataset corpus. Every third row has a title, the others have none."""
    frame = two_dataset_corpus()
    titles = [
        " ".join(text.split()[:3]) if index % 3 == 0 else None
        for index, text in enumerate(frame["text"])
    ]
    return frame.assign(title=titles)


def _split(name: str) -> pd.DataFrame:
    start, end = BOUNDS[name]
    return _corpus().iloc[start:end]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder that looks like a pretrained checkpoint: encoder weights and a tokenizer."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    directory = tmp_path_factory.mktemp("tiny-deberta")

    split_words = pre_tokenizers.Whitespace()
    words = sorted(
        {
            word
            for text in _corpus()["text"]
            for word, _ in split_words.pre_tokenize_str(text.lower())
        }
    )
    vocab = {
        token: index for index, token in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", *words])
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = split_words
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    ).save_pretrained(directory)

    config = transformers.DebertaV2Config(  # the DeBERTa-v3 layout, at toy size
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=16,
        norm_rel_ebd="layer_norm",
        share_att_key=True,
        pos_att_type=["p2c", "c2p"],
        position_biased_input=False,
        pooler_hidden_size=32,
    )
    torch.manual_seed(0)
    transformers.DebertaV2Model(config).save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The processed folder and its lock."""
    return write_processed(tmp_path_factory.mktemp("data"), _corpus())


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transformer, "_device", lambda requested=None: requested or "cpu")


@pytest.fixture
def config(tiny_model: Path) -> TrainConfig:
    return TrainConfig(
        model_name=str(tiny_model),
        max_length=48,
        max_title_tokens=8,
        lr=3e-3,
        weight_decay=0.0,
        epochs=3,
        batch_size=8,
        eval_every_steps=30,
        checkpoint_every_steps=20,
        log_every_steps=5,
    )


def _log(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _losses(run_dir: Path) -> list[float]:
    return [entry["loss"] for entry in _log(run_dir) if "loss" in entry]


def test_training_lowers_the_loss_and_keeps_the_best_model(
    config: TrainConfig, data: tuple[Path, Path], tmp_path: Path
) -> None:
    processed, lock = data
    run_dir = tmp_path / "run"

    summary = train(config, processed, run_dir, lock_path=lock)

    losses = _losses(run_dir)
    assert losses[-1] < 0.5 * losses[0]
    assert summary["steps"] == summary["total_steps"] == 3 * STEPS_PER_EPOCH
    assert summary["finished"]
    assert not summary["stopped_early"]
    assert summary["best_val_macro_f1"] > 0.9
    assert summary["seconds_per_step"] > 0
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) == summary
    assert TrainConfig.from_yaml(run_dir / "config.yaml") == config

    best = run_dir / "best"
    assert (best / "tokenizer.json").exists()
    assert json.loads((best / "input_config.json").read_text())["max_length"] == 48
    # The saved model, loaded again and fed from a frame, repeats the score it was chosen for.
    val = _split("val")
    margins = predict_logits(best, val)
    assert margins.shape == (len(val),)
    assert margins.dtype == np.float64
    reloaded = f1_score(val["label"], margins > 0, average="macro")
    assert reloaded == pytest.approx(summary["best_val_macro_f1"])


def test_resume_matches_an_uninterrupted_run(
    config: TrainConfig, data: tuple[Path, Path], tmp_path: Path
) -> None:
    import torch

    processed, lock = data
    whole, parts = tmp_path / "whole", tmp_path / "parts"

    train(config, processed, whole, max_steps=40, lock_path=lock)
    first = train(config, processed, parts, max_steps=20, lock_path=lock)
    second = train(config, processed, parts, resume=True, max_steps=40, lock_path=lock)

    assert (first["steps"], second["steps"]) == (20, 40)
    assert not second["finished"]
    assert [entry["step"] for entry in _log(parts)] == [entry["step"] for entry in _log(whole)]
    assert _losses(parts)[-1] == pytest.approx(_losses(whole)[-1], abs=1e-4)
    expected = torch.load(whole / "checkpoint.pt", weights_only=True)["model"]
    resumed = torch.load(parts / "checkpoint.pt", weights_only=True)["model"]
    for name, tensor in expected.items():
        torch.testing.assert_close(resumed[name], tensor, rtol=0, atol=1e-5, msg=name)


def test_resume_ignores_a_half_written_checkpoint(
    config: TrainConfig, data: tuple[Path, Path], tmp_path: Path
) -> None:
    processed, lock = data
    run_dir = tmp_path / "run"
    train(config, processed, run_dir, max_steps=20, lock_path=lock)
    (run_dir / "checkpoint.pt.tmp").write_bytes(b"killed while this was being written")

    summary = train(config, processed, run_dir, resume=True, max_steps=30, lock_path=lock)

    assert summary["steps"] == 30
    assert not (run_dir / "checkpoint.pt.tmp").exists()
    steps = [entry["step"] for entry in _log(run_dir) if "loss" in entry]
    assert steps == [5, 10, 15, 20, 25, 30]


def test_resume_before_the_first_checkpoint_starts_from_the_beginning(
    config: TrainConfig, data: tuple[Path, Path], tmp_path: Path
) -> None:
    processed, lock = data
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(config.to_yaml(), encoding="utf-8")
    (run_dir / "log.jsonl").write_text(json.dumps({"step": 5, "loss": 0.7}) + "\n")

    summary = train(config, processed, run_dir, resume=True, max_steps=5, lock_path=lock)

    assert summary["steps"] == 5
    assert [entry["step"] for entry in _log(run_dir)] == [5]


def test_training_stops_when_val_stops_improving(
    config: TrainConfig, data: tuple[Path, Path], tmp_path: Path
) -> None:
    processed, lock = data
    run_dir = tmp_path / "run"
    frozen = replace(config, lr=0.0, eval_every_steps=5, patience=2)  # nothing can improve

    summary = train(frozen, processed, run_dir, lock_path=lock)

    assert summary["stopped_early"]
    assert summary["finished"]
    assert summary["steps"] == 15  # the first evaluation sets the mark, two more do not beat it
    assert summary["best_step"] == 5
    again = train(frozen, processed, run_dir, resume=True, lock_path=lock)
    assert again["steps"] == 15


def test_without_balanced_sampling_the_loss_is_class_weighted(
    config: TrainConfig, data: tuple[Path, Path], tmp_path: Path
) -> None:
    processed, lock = data
    run_dir = tmp_path / "run"

    summary = train(replace(config, balance="none"), processed, run_dir, lock_path=lock)

    assert summary["best_val_macro_f1"] > 0.9
    assert _losses(run_dir)[-1] < 0.5 * _losses(run_dir)[0]


def test_gradient_accumulation_counts_optimizer_steps(
    config: TrainConfig, data: tuple[Path, Path], tmp_path: Path
) -> None:
    processed, lock = data
    accumulated = replace(config, batch_size=4, grad_accum=2, epochs=1)

    summary = train(accumulated, processed, tmp_path / "run", lock_path=lock)

    assert summary["total_steps"] == STEPS_PER_EPOCH  # 240 rows / (4 x 2)
    assert summary["steps"] == STEPS_PER_EPOCH


def test_a_row_cap_trains_on_a_stratified_part_of_the_split(
    config: TrainConfig, data: tuple[Path, Path], tmp_path: Path
) -> None:
    processed, lock = data
    capped = replace(config, max_train_rows=80, epochs=1)

    summary = train(capped, processed, tmp_path / "run", lock_path=lock)

    assert summary["train_rows"] == 80
    assert summary["total_steps"] == 10


def test_fit_scorer_gives_calibrated_probabilities_and_cleans_up(
    config: TrainConfig, tmp_path: Path
) -> None:
    work_dir = tmp_path / "fits"
    test = _split("test")

    scorer = fit_scorer(config, work_dir=work_dir)(_split("train"), _split("val"), _split("calib"))
    scores = scorer(test)

    labels = test["label"].to_numpy()
    assert scores.shape == (len(test),)
    assert scores.dtype == np.float64
    assert scores.min() >= 0.01
    assert scores.max() <= 0.99
    assert scores[labels == 1].mean() > scores[labels == 0].mean() + 0.5
    (fold,) = list(work_dir.iterdir())
    assert not (fold / "checkpoint.pt").exists()
    del scorer
    gc.collect()
    assert not fold.exists()


def test_load_scorer_applies_the_temperature_of_the_run(
    config: TrainConfig, data: tuple[Path, Path], tmp_path: Path
) -> None:
    processed, lock = data
    run_dir = tmp_path / "run"
    train(replace(config, epochs=1), processed, run_dir, lock_path=lock)
    (run_dir / "calibration.json").write_text(json.dumps({"temperature": 2.5}))
    test = _split("test")

    scores = load_scorer(run_dir)(test)

    expected = apply_temperature(predict_logits(run_dir / "best", test), 2.5)
    np.testing.assert_allclose(scores, expected)


def test_the_benchmark_times_full_precision_only_on_a_cpu(
    config: TrainConfig, data: tuple[Path, Path]
) -> None:
    processed, lock = data

    result = benchmark_precision(config, processed, steps=6, lock_path=lock)

    assert result["device"] == "cpu"
    assert result["amp"] is None
    assert not result["use_amp"]
    assert result["fp32"]["fits"]
    assert result["fp32"]["loss_finite"]
    assert result["fp32"]["s_per_step"] > 0


def test_a_trained_run_is_calibrated_on_its_calib_split(
    config: TrainConfig, data: tuple[Path, Path], tmp_path: Path
) -> None:
    processed, lock = data
    run_dir = tmp_path / "run"
    train(replace(config, epochs=1), processed, run_dir, lock_path=lock)

    result = calibrate_run(run_dir, processed, lock_path=lock)

    calibration = result["calibration"]
    assert calibration["rows"] == len(_split("calib"))
    assert 0.05 <= calibration["temperature"] <= 20.0
    assert calibration["ece_after"] <= calibration["ece_before"] + 0.02
    logits = np.load(run_dir / "calib_logits.npy")
    np.testing.assert_allclose(logits, predict_logits(run_dir / "best", _split("calib")))
    test = _split("test")
    expected = apply_temperature(predict_logits(run_dir / "best", test), calibration["temperature"])
    np.testing.assert_allclose(load_scorer(run_dir)(test), expected)
