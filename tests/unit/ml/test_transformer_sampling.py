"""The parts of the transformer trainer that need no PyTorch: settings, sampling, input building
and the run folder's rules."""

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from typer.testing import CliRunner

from fnd_ml.cli import app
from fnd_ml.common import resources
from fnd_ml.data.build import MANIFEST_NAME, ManifestMismatch
from fnd_ml.models import transformer
from fnd_ml.models.transformer import (
    InputSpec,
    RunFolderError,
    Tokens,
    TrainConfig,
    choose_amp,
    encode_articles,
    epoch_order,
    sampling_weights,
    train,
)
from tests.fixtures.processed import two_dataset_corpus, write_processed

# ---------------------------------------------------------------- sampling


def test_weights_balance_labels_inside_each_dataset() -> None:
    w = sampling_weights(["a"] * 4 + ["b"] * 6, [1, 0, 0, 0, 1, 1, 1, 1, 1, 0])

    assert w[0] == pytest.approx(w[1:4].sum())
    assert w[4:9].sum() == pytest.approx(w[9])
    assert w[:4].sum() / w.sum() == pytest.approx(0.4)


def test_a_dataset_with_one_label_keeps_its_share() -> None:
    w = sampling_weights(["a"] * 2 + ["b"] * 8, [1, 1, 0, 0, 0, 0, 1, 1, 1, 1])

    assert w[0] == pytest.approx(w[1])
    assert w[:2].sum() / w.sum() == pytest.approx(0.2)


def test_epoch_order_is_repeatable_and_changes_per_epoch() -> None:
    weights = sampling_weights(["a"] * 50 + ["b"] * 50, [0, 1] * 50)

    first = epoch_order(weights, 100, seed=13, epoch=0)

    assert first.dtype == np.int64
    assert first.shape == (100,)
    assert first.min() >= 0
    assert first.max() < 100
    np.testing.assert_array_equal(first, epoch_order(weights, 100, seed=13, epoch=0))
    assert not np.array_equal(first, epoch_order(weights, 100, seed=13, epoch=1))
    assert not np.array_equal(first, epoch_order(weights, 100, seed=14, epoch=0))


def test_epoch_order_draws_rows_in_proportion_to_their_weight() -> None:
    # One row of label 1 among ten: balanced sampling draws it about as often as the other nine.
    weights = sampling_weights(["a"] * 10, [1] + [0] * 9)

    order = epoch_order(weights, 20_000, seed=13, epoch=0)

    assert (order == 0).mean() == pytest.approx(0.5, abs=0.02)


def test_without_weights_an_epoch_reads_every_row_once() -> None:
    order = epoch_order(None, 50, seed=13, epoch=0)

    assert sorted(order.tolist()) == list(range(50))
    assert order.tolist() != list(range(50))


# ---------------------------------------------------------------- mixed precision choice


def _timing(
    s_per_step: float | None, *, fits: bool = True, finite: bool = True
) -> dict[str, object]:
    return {"s_per_step": s_per_step, "peak_mem_mb": 1000.0, "loss_finite": finite, "fits": fits}


def test_mixed_precision_is_used_only_when_it_is_clearly_faster() -> None:
    assert choose_amp(_timing(1.0), _timing(0.85))
    assert not choose_amp(_timing(1.0), _timing(0.86))


def test_mixed_precision_is_used_when_full_precision_does_not_fit() -> None:
    assert choose_amp(_timing(None, fits=False), _timing(2.0))


def test_mixed_precision_is_refused_when_its_loss_is_not_finite_or_it_cannot_run() -> None:
    assert not choose_amp(_timing(1.0), _timing(0.5, finite=False))
    assert not choose_amp(_timing(1.0), _timing(None, fits=False))
    assert not choose_amp(_timing(1.0), None)


# ---------------------------------------------------------------- settings


def test_config_reads_yaml_and_accepts_scientific_notation(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model_name: some/model\nlr: 2e-5\nbatch_size: 16\n", encoding="utf-8")

    config = TrainConfig.from_yaml(path)

    assert config.model_name == "some/model"
    assert config.lr == pytest.approx(2e-5)
    assert config.batch_size == 16
    assert (config.max_title_tokens, config.head_share, config.patience) == (64, 0.75, 3)


def test_config_reads_yaml_on_and_off_as_the_amp_modes(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model_name: m\namp: off\n", encoding="utf-8")  # YAML reads `off` as false

    assert TrainConfig.from_yaml(path).amp == "off"


def test_config_survives_a_round_trip_through_yaml(tmp_path: Path) -> None:
    config = TrainConfig(model_name="m", lr=2e-5, max_train_rows=20_000, balance="none")
    path = tmp_path / "config.yaml"
    path.write_text(config.to_yaml(), encoding="utf-8")

    assert TrainConfig.from_yaml(path) == config
    assert TrainConfig.from_dict({"model_name": "m"}).max_train_rows is None


def test_config_rejects_unknown_modes_and_settings() -> None:
    with pytest.raises(ValueError, match="amp"):
        TrainConfig(model_name="m", amp="sometimes")
    with pytest.raises(ValueError, match="balance"):
        TrainConfig(model_name="m", balance="labels")
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig.from_dict({"model_name": "m", "learning_rate": 1e-5})


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("deberta_small", ("microsoft/deberta-v3-small", 256, 16, 1, 2, 500, 500)),
        ("deberta_base", ("microsoft/deberta-v3-base", 512, 8, 2, 3, 1000, 500)),
    ],
)
def test_the_shipped_configs_hold_the_planned_settings(
    name: str, expected: tuple[str, int, int, int, int, int, int]
) -> None:
    config = TrainConfig.from_yaml(Path("ml/configs") / f"{name}.yaml")

    assert (
        config.model_name,
        config.max_length,
        config.batch_size,
        config.grad_accum,
        config.epochs,
        config.eval_every_steps,
        config.checkpoint_every_steps,
    ) == expected
    assert (config.lr, config.weight_decay, config.warmup_share) == (2e-5, 0.01, 0.10)
    assert (config.balance, config.amp, config.seed) == ("dataset_label", "auto", 13)


# ---------------------------------------------------------------- articles to input ids

PAD, UNK, CLS, SEP = 0, 1, 2, 3
SPEC = InputSpec(
    max_length=16, max_title_tokens=4, head_share=0.75, cls_id=CLS, sep_id=SEP, pad_id=PAD
)


def _word_tokenizer(words: Sequence[str]) -> SimpleNamespace:
    """Stands in for the transformers tokenizer: only its ``backend_tokenizer`` is used."""
    vocab = {"[PAD]": PAD, "[UNK]": UNK, "[CLS]": CLS, "[SEP]": SEP}
    vocab |= {word: index for index, word in enumerate(words, start=4)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return SimpleNamespace(backend_tokenizer=backend)


def test_three_words_without_a_title_become_cls_body_sep() -> None:
    tokenizer = _word_tokenizer(["markets", "fell", "today"])

    tokens = encode_articles(
        tokenizer, [(None, "markets fell today"), (float("nan"), "Today markets fell")], SPEC
    )

    assert len(tokens) == 2
    assert tokens.row(0).tolist() == [CLS, 4, 5, 6, SEP]
    assert tokens.row(1).tolist() == [CLS, 6, 4, 5, SEP]
    assert tokens.cut.tolist() == [False, False]


def test_a_long_article_is_cut_to_the_window_and_flagged() -> None:
    words = [f"w{i}" for i in range(200)]
    tokenizer = _word_tokenizer(words)

    tokens = encode_articles(tokenizer, [("w0 w1", " ".join(words)), (None, "w5")], SPEC)

    assert tokens.lengths().tolist() == [16, 3]
    assert tokens.cut.tolist() == [True, False]
    # [CLS] w0 w1 [SEP], then 11 body tokens: the first 8 and the last 3, then [SEP]
    assert tokens.row(0).tolist() == [CLS, 4, 5, SEP, *range(4, 12), 201, 202, 203, SEP]


def test_articles_are_scrubbed_before_they_are_tokenized() -> None:
    tokenizer = _word_tokenizer(["markets", "fell", "shares", "rose"])

    tokens = encode_articles(
        tokenizer, [("Shares rose - Reuters", "LONDON (Reuters) - Markets fell")], SPEC
    )

    assert tokens.row(0).tolist() == [CLS, 6, 7, SEP, 4, 5, SEP]


def test_no_articles_give_an_empty_token_set() -> None:
    tokens = encode_articles(_word_tokenizer([]), [], SPEC)

    assert len(tokens) == 0
    assert tokens.ids.shape == (0,)


def test_tokens_are_cached_for_the_same_stamp_only(tmp_path: Path) -> None:
    tokenizer = _word_tokenizer(["markets", "fell"])
    calls: list[int] = []

    def make() -> Tokens:
        calls.append(1)
        return encode_articles(tokenizer, [(None, "markets fell"), (None, "fell")], SPEC)

    first = transformer._cached_tokens(tmp_path, "train", {"rows": 2}, make)
    again = transformer._cached_tokens(tmp_path, "train", {"rows": 2}, make)
    transformer._cached_tokens(tmp_path, "train", {"rows": 3}, make)

    assert len(calls) == 2
    np.testing.assert_array_equal(again.ids, first.ids)
    np.testing.assert_array_equal(again.offsets, first.offsets)
    np.testing.assert_array_equal(again.cut, first.cut)
    assert again.row(1).tolist() == [CLS, 5, SEP]


# ---------------------------------------------------------------- run folder


def test_training_refuses_a_changed_manifest(tmp_path: Path) -> None:
    processed, lock = write_processed(tmp_path, two_dataset_corpus())
    manifest = processed / MANIFEST_NAME
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ManifestMismatch, match="does not match the frozen"):
        train(TrainConfig(model_name="unused"), processed, tmp_path / "run", lock_path=lock)

    assert not (tmp_path / "run").exists()


def test_a_run_folder_is_not_started_over_without_resume(tmp_path: Path) -> None:
    processed, lock = write_processed(tmp_path, two_dataset_corpus())
    config = TrainConfig(model_name="unused")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(config.to_yaml(), encoding="utf-8")

    with pytest.raises(RunFolderError, match="--resume"):
        train(config, processed, run_dir, lock_path=lock)


def test_a_run_is_not_resumed_with_other_settings(tmp_path: Path) -> None:
    processed, lock = write_processed(tmp_path, two_dataset_corpus())
    config = TrainConfig(model_name="unused")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(config.to_yaml(), encoding="utf-8")

    with pytest.raises(RunFolderError, match="other settings"):
        train(replace(config, lr=1e-4), processed, run_dir, resume=True, lock_path=lock)

    assert TrainConfig.from_yaml(run_dir / "config.yaml") == config


def _best_folder(path: Path, step: int, score: float) -> None:
    path.mkdir()
    (path / "metrics.json").write_text(json.dumps({"step": step, "val_macro_f1": score}))


def test_a_complete_best_folder_left_by_a_kill_is_put_in_place(tmp_path: Path) -> None:
    _best_folder(tmp_path / "best.tmp", step=40, score=0.9)
    progress = transformer._Progress(step=20, best_step=20, best_val_macro_f1=0.8)

    transformer._recover_best(tmp_path, progress)

    assert (tmp_path / "best" / "metrics.json").exists()
    assert not (tmp_path / "best.tmp").exists()
    assert (progress.best_step, progress.best_val_macro_f1) == (40, 0.9)


def test_a_half_written_best_folder_is_thrown_away(tmp_path: Path) -> None:
    _best_folder(tmp_path / "best", step=20, score=0.8)
    (tmp_path / "best.tmp").mkdir()
    (tmp_path / "best.tmp" / "model.safetensors").write_bytes(b"killed while writing")
    progress = transformer._Progress(step=30)

    transformer._recover_best(tmp_path, progress)

    assert not (tmp_path / "best.tmp").exists()
    assert (progress.best_step, progress.best_val_macro_f1) == (20, 0.8)


def test_the_log_loses_the_lines_after_the_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text("".join(json.dumps({"step": step}) + "\n" for step in (10, 20, 20, 30)))

    transformer._trim_log(path, 20)

    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [10, 20, 20]


# ---------------------------------------------------------------- command line


def test_the_command_stops_with_a_clear_message_when_memory_is_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "free_ram_gb", lambda: 0.4)
    run_dir = tmp_path / "run"
    arguments = ["train", "transformer", "--config", "ml/configs/deberta_small.yaml"]

    result = CliRunner().invoke(app, [*arguments, "--run-dir", str(run_dir)])

    assert result.exit_code == 2
    assert "needs about 2.5 GB" in result.output
    assert not run_dir.exists()
