"""``fnd train transformer``: fine-tune a DeBERTa-v3 classifier on the frozen splits.

A plain PyTorch loop, written so that a long run on one consumer GPU can be stopped and continued:

* The articles are tokenized once; the token ids are cached in the run folder.
* The rows of an epoch come from ``epoch_order``, which depends only on the seed and the epoch
  number, so a resumed run reads the same rows in the same order.
* ``checkpoint.pt`` holds everything the next step depends on (model, optimizer, schedule, scaler,
  step counters and random-generator states). It is written to a temporary file and then renamed,
  so a run killed while writing still has its previous checkpoint.
* ``best/`` holds the model with the highest macro-F1 on the val split so far, with its tokenizer
  and the settings that turn an article into input ids.

PyTorch and transformers are imported inside the functions that need them. Importing this module,
its configuration and its sampling code works without them.
"""

from __future__ import annotations

import gc
import json
import math
import os
import shutil
import tempfile
import time
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from numpy.typing import NDArray
from sklearn.metrics import f1_score

from fnd_ml.calibration.temperature import apply_temperature, fit_temperature
from fnd_ml.common.text import transformer_parts
from fnd_ml.common.truncation import build_input
from fnd_ml.data.build import DEFAULT_LOCK, assert_frozen
from fnd_ml.models.scorer import Fit, Scorer
from fnd_ml.models.train_baseline import stratified_positions

BALANCE_MODES = ("dataset_label", "none")
AMP_MODES = ("auto", "on", "off")
AMP_MAX_TIME_RATIO = 0.85  # mixed precision is worth its risk only when it is at least 15% faster
MAX_GRAD_NORM = 1.0
BENCHMARK_WARMUP_STEPS = 3  # the first steps include one-time set-up and are not timed
AUTO_AMP_STEPS = 20  # timed steps per precision when a run resolves `amp: auto` by itself

BEST_DIR = "best"
CHECKPOINT_NAME = "checkpoint.pt"
LOG_NAME = "log.jsonl"
CONFIG_NAME = "config.yaml"
SUMMARY_NAME = "summary.json"
PRECISION_NAME = "precision.json"
CALIBRATION_NAME = "calibration.json"
INPUT_CONFIG_NAME = "input_config.json"
BEST_METRICS_NAME = "metrics.json"
DEFAULT_FIT_DIR = Path("ml/runs/fit")


class RunFolderError(RuntimeError):
    """The run folder holds another run, or one that was started with other settings."""


# ---------------------------------------------------------------- configuration


def _coerce(kind: str, value: Any) -> Any:
    """Turn a YAML value into the field's type: PyYAML reads ``2e-5`` as text and ``on`` as true."""
    if value is None:
        return None
    base = kind.split(" | ")[0]
    if base == "str":
        return {True: "on", False: "off"}[value] if isinstance(value, bool) else value
    return {"int": int, "float": float}[base](value)


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    max_length: int = 512
    max_title_tokens: int = 64
    head_share: float = 0.75
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 1  # batches per optimizer step; the effective batch is their product
    warmup_share: float = 0.10
    eval_every_steps: int = 1000
    checkpoint_every_steps: int = 500
    log_every_steps: int = 10
    patience: int = 3  # evaluations without a better val macro-F1 before the run stops
    balance: str = "dataset_label"
    amp: str = "auto"
    seed: int = 13
    max_train_rows: int | None = None

    def __post_init__(self) -> None:
        if self.amp not in AMP_MODES:
            raise ValueError(f"amp must be one of {AMP_MODES}, not {self.amp!r}")
        if self.balance not in BALANCE_MODES:
            raise ValueError(f"balance must be one of {BALANCE_MODES}, not {self.balance!r}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Self:
        kinds = {field.name: str(field.type) for field in fields(cls)}
        unknown = sorted(set(raw) - set(kinds))
        if unknown:
            raise ValueError(f"unknown training setting(s): {', '.join(unknown)}")
        return cls(**{name: _coerce(kinds[name], value) for name, value in raw.items()})

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False)

    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum


# ---------------------------------------------------------------- sampling


def sampling_weights(dataset: Sequence[str], label: Sequence[int]) -> NDArray[np.float64]:
    """Row weights that draw both labels equally often inside each dataset.

    Each dataset keeps its share of the draws: ``w = n_dataset / (labels_in_dataset *
    n_dataset,label)``, which is ``0.5 * n_dataset / n_dataset,label`` when it has both labels.
    """
    frame = pd.DataFrame({"dataset": list(dataset), "label": list(label)})
    by_dataset = frame.groupby("dataset")["label"]
    in_cell = frame.groupby(["dataset", "label"])["label"].transform("size")
    weights = by_dataset.transform("size") / (by_dataset.transform("nunique") * in_cell)
    return weights.to_numpy(dtype=np.float64)


def epoch_order(
    weights: NDArray[np.float64] | None, n: int, seed: int, epoch: int
) -> NDArray[np.int64]:
    """The ``n`` row positions one epoch reads. The same arguments always give the same order.

    With weights, rows are drawn with replacement in proportion to their weight. Without, the
    epoch reads every row of ``range(n)`` once, shuffled.
    """
    rng = np.random.default_rng([seed, epoch])
    if weights is None:
        return rng.permutation(n).astype(np.int64)
    share = np.asarray(weights, dtype=np.float64)
    return rng.choice(len(share), size=n, replace=True, p=share / share.sum()).astype(np.int64)


def choose_amp(fp32: Mapping[str, Any], amp: Mapping[str, Any] | None) -> bool:
    """Whether to train in mixed precision, given the two timings of ``benchmark_precision``.

    Only when it runs and its loss stays finite, and then only when full precision does not fit in
    memory or mixed precision is at least 15% faster.
    """
    if amp is None or not amp["fits"] or not amp["loss_finite"] or amp["s_per_step"] is None:
        return False
    if not fp32["fits"] or fp32["s_per_step"] is None:
        return True
    return bool(amp["s_per_step"] <= AMP_MAX_TIME_RATIO * fp32["s_per_step"])


# ---------------------------------------------------------------- articles to input ids


@dataclass(frozen=True)
class InputSpec:
    """How an article becomes input ids. Saved with the model, so serving builds the same input."""

    max_length: int
    max_title_tokens: int
    head_share: float
    cls_id: int
    sep_id: int
    pad_id: int


@dataclass(frozen=True)
class Tokens:
    """Input ids of many articles in one flat array: row ``i`` is ``ids[offsets[i]:offsets[i+1]]``."""

    ids: NDArray[np.int32]
    offsets: NDArray[np.int64]
    cut: NDArray[np.bool_]  # whether the row lost tokens to truncation

    @classmethod
    def from_rows(cls, rows: Sequence[NDArray[np.int32]], cut: Sequence[bool]) -> Self:
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum([len(row) for row in rows], out=offsets[1:])
        ids = np.concatenate(rows) if len(rows) else np.empty(0, dtype=np.int32)
        return cls(ids=ids, offsets=offsets, cut=np.asarray(cut, dtype=np.bool_))

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def row(self, index: int) -> NDArray[np.int32]:
        return self.ids[self.offsets[index] : self.offsets[index + 1]]

    def lengths(self) -> NDArray[np.int64]:
        return np.diff(self.offsets)

    def take(self, rows: NDArray[np.int64]) -> Self:
        return self.from_rows([self.row(int(row)) for row in rows], self.cut[rows].tolist())


@dataclass(frozen=True)
class _Split:
    tokens: Tokens
    labels: NDArray[np.int64]
    datasets: list[str]

    def take(self, rows: NDArray[np.int64]) -> _Split:
        datasets = [self.datasets[int(row)] for row in rows]
        return _Split(self.tokens.take(rows), self.labels[rows], datasets)


def _load_tokenizer(name_or_dir: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_dir)
    # Articles are cut by ``build_input``, not by the tokenizer; the saved tokenizer.json must not
    # truncate or pad either, because serving loads that file.
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    return tokenizer


def _input_spec(tokenizer: Any, config: TrainConfig) -> InputSpec:
    return InputSpec(
        max_length=config.max_length,
        max_title_tokens=config.max_title_tokens,
        head_share=config.head_share,
        cls_id=int(tokenizer.cls_token_id),
        sep_id=int(tokenizer.sep_token_id),
        pad_id=int(tokenizer.pad_token_id),
    )


def encode_articles(
    tokenizer: Any,
    articles: Iterable[tuple[str | None, str]],
    spec: InputSpec,
    *,
    chunk: int = 256,
) -> Tokens:
    """Scrub, tokenize and cut ``(title, text)`` pairs into model inputs."""
    backend = tokenizer.backend_tokenizer
    rows: list[NDArray[np.int32]] = []
    cut: list[bool] = []
    pending: list[tuple[str, str]] = []

    def flush() -> None:
        titles = backend.encode_batch([title for title, _ in pending], add_special_tokens=False)
        bodies = backend.encode_batch([body for _, body in pending], add_special_tokens=False)
        for title, body in zip(titles, bodies, strict=True):
            ids, was_cut = build_input(
                title.ids,
                body.ids,
                cls_id=spec.cls_id,
                sep_id=spec.sep_id,
                max_length=spec.max_length,
                max_title=spec.max_title_tokens,
                head_share=spec.head_share,
            )
            rows.append(np.asarray(ids, dtype=np.int32))
            cut.append(was_cut)
        pending.clear()

    for title, text in articles:
        pending.append(transformer_parts(title if isinstance(title, str) else None, text))
        if len(pending) == chunk:
            flush()
    if pending:
        flush()
    return Tokens.from_rows(rows, cut)


def _frame_articles(frame: pd.DataFrame) -> Iterator[tuple[str | None, str]]:
    return zip(frame["title"].tolist(), frame["text"].tolist(), strict=True)


def _file_articles(
    path: Path, positions: NDArray[np.int_] | None = None
) -> Iterator[tuple[str | None, str]]:
    """Stream ``(title, text)`` from a split file: all rows, or the rows at the sorted positions."""
    wanted = None if positions is None else set(positions.tolist())
    position = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=["title", "text"]):
        for article in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            if wanted is None or position in wanted:
                yield article
            position += 1


def _cached_tokens(
    run_dir: Path, name: str, stamp: dict[str, Any], make: Callable[[], Tokens]
) -> Tokens:
    """Tokens from the run folder if they were made from the same data and settings, else new."""
    path, stamp_path = run_dir / f"tokens_{name}.npz", run_dir / f"tokens_{name}.json"
    cached = path.exists() and stamp_path.exists()
    if cached and json.loads(stamp_path.read_text(encoding="utf-8")) == stamp:
        with np.load(path) as data:
            return Tokens(ids=data["ids"], offsets=data["offsets"], cut=data["cut"])
    tokens = make()
    stamp_path.unlink(missing_ok=True)  # the stamp is written last: it vouches for a complete file
    np.savez(path, ids=tokens.ids, offsets=tokens.offsets, cut=tokens.cut)
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    return tokens


# ---------------------------------------------------------------- model steps


@dataclass
class _Learner:
    """The model and everything that changes with it while it trains."""

    model: Any
    optimizer: Any
    scheduler: Any
    scaler: Any
    device: str
    use_amp: bool


class _WarmupThenDecay:
    """Learning-rate factor: linear from 0 to 1 over the warmup steps, then linear down to 0."""

    def __init__(self, total_steps: int, warmup_steps: int) -> None:
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        return max(0.0, (self.total_steps - step) / max(1, self.total_steps - self.warmup_steps))


def _device(requested: str | None = None) -> str:
    import torch

    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def _new_learner(
    config: TrainConfig, total_steps: int, warmup_steps: int, device: str, use_amp: bool
) -> _Learner:
    import torch
    from transformers import AutoModelForSequenceClassification

    torch.manual_seed(config.seed)  # the classification head starts from random weights
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label={0: "reliable", 1: "unreliable"},
        label2id={"reliable": 0, "unreliable": 1},
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight")
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    decayed = [p for name, p in named if not name.endswith(no_decay)]
    plain = [p for name, p in named if name.endswith(no_decay)]
    optimizer = torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": config.weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=config.lr,
    )
    schedule = _WarmupThenDecay(total_steps, warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler(device.split(":")[0], enabled=use_amp)
    return _Learner(model, optimizer, scheduler, scaler, device, use_amp)


def _batch(tokens: Tokens, rows: NDArray[np.int64], pad_id: int, device: str) -> tuple[Any, Any]:
    """``input_ids`` and ``attention_mask`` of the rows, padded to the longest of them."""
    import torch

    pieces = [tokens.row(int(row)) for row in rows]
    width = max(len(piece) for piece in pieces)
    input_ids = np.full((len(pieces), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(pieces), width), dtype=np.int64)
    for index, piece in enumerate(pieces):
        input_ids[index, : len(piece)] = piece
        mask[index, : len(piece)] = 1
    return torch.from_numpy(input_ids).to(device), torch.from_numpy(mask).to(device)


def _margins(
    model: Any, tokens: Tokens, pad_id: int, device: str, batch_size: int, use_amp: bool
) -> NDArray[np.float64]:
    """``z1 - z0`` for every row.

    Rows are scored longest first: batches need little padding, and a batch that does not fit in
    memory fails at the start instead of near the end.
    """
    import torch

    order = np.argsort(-tokens.lengths(), kind="stable")
    margins = np.empty(len(tokens), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            input_ids, mask = _batch(tokens, rows, pad_id, device)
            with torch.autocast(device.split(":")[0], dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=input_ids, attention_mask=mask).logits
            margins[rows] = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
    return margins


def _loss_weights(labels: NDArray[np.int64], config: TrainConfig) -> NDArray[np.float32]:
    """Per-row loss weights: 1 under balanced sampling, inverse class frequency without it."""
    if config.balance == "dataset_label":
        return np.ones(len(labels), dtype=np.float32)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    per_class = len(labels) / (2.0 * np.maximum(counts, 1.0))
    return per_class[labels].astype(np.float32)


def _train_step(
    learner: _Learner,
    split: _Split,
    loss_weights: NDArray[np.float32],
    rows: NDArray[np.int64],
    batch_size: int,
    pad_id: int,
) -> float:
    """One optimizer step over ``rows``, in batches of ``batch_size``; returns the mean loss."""
    import torch

    device_type = learner.device.split(":")[0]
    learner.model.train()
    learner.optimizer.zero_grad(set_to_none=True)
    total_weight = float(loss_weights[rows].sum())
    step_loss = 0.0
    for start in range(0, len(rows), batch_size):
        part = rows[start : start + batch_size]
        input_ids, mask = _batch(split.tokens, part, pad_id, learner.device)
        target = torch.from_numpy(split.labels[part]).to(learner.device)
        weight = torch.from_numpy(loss_weights[part]).to(learner.device)
        with torch.autocast(device_type, dtype=torch.float16, enabled=learner.use_amp):
            logits = learner.model(input_ids=input_ids, attention_mask=mask).logits
        losses = torch.nn.functional.cross_entropy(logits.float(), target, reduction="none")
        loss = (losses * weight).sum() / total_weight
        learner.scaler.scale(loss).backward()
        step_loss += float(loss.detach())
    if not math.isfinite(step_loss):
        raise FloatingPointError(
            "the training loss is not finite; with mixed precision, set `amp: off` and start again"
        )
    learner.scaler.unscale_(learner.optimizer)
    torch.nn.utils.clip_grad_norm_(learner.model.parameters(), MAX_GRAD_NORM)
    learner.scaler.step(learner.optimizer)
    learner.scaler.update()
    learner.scheduler.step()
    return step_loss


# ---------------------------------------------------------------- precision benchmark


def _time_steps(
    config: TrainConfig, pad_id: int, split: _Split, steps: int, use_amp: bool, device: str
) -> dict[str, Any]:
    """Run ``steps`` training steps on a fresh model and report time, memory and loss health."""
    import torch

    cuda = device.startswith("cuda")
    effective = config.effective_batch
    # The longest rows go first: if a batch of full-length articles does not fit, it shows at once.
    by_length = np.argsort(-split.tokens.lengths(), kind="stable")
    rest = np.random.default_rng(config.seed).permutation(by_length[effective:])
    order = np.concatenate([by_length[:effective], rest])
    loss_weights = _loss_weights(split.labels, config)
    result: dict[str, Any] = {
        "s_per_step": None,
        "peak_mem_mb": None,
        "loss_finite": True,
        "fits": True,
    }
    seconds: list[float] = []
    try:
        learner = _new_learner(config, 10 * steps, 0, device, use_amp)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        for index in range(steps):
            rows = order[index * effective : (index + 1) * effective]
            if len(rows) == 0:
                break
            started = time.perf_counter()
            _train_step(learner, split, loss_weights, rows, config.batch_size, pad_id)
            if cuda:
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
    except FloatingPointError:
        result["loss_finite"] = False
    except torch.OutOfMemoryError:
        result["fits"] = False
    if cuda:
        result["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    timed = seconds[BENCHMARK_WARMUP_STEPS:] or seconds
    if timed and result["fits"]:
        result["s_per_step"] = round(float(np.mean(timed)), 4)
    return result


def _benchmark(config: TrainConfig, pad_id: int, split: _Split, steps: int) -> dict[str, Any]:
    import torch

    device = _device()
    timings: dict[str, dict[str, Any] | None] = {"fp32": None, "amp": None}
    for name, use_amp in (("fp32", False), ("amp", True)):
        if use_amp and device == "cpu":
            continue
        timings[name] = _time_steps(config, pad_id, split, steps, use_amp, device)
        gc.collect()  # the model of this timing must be gone before the next one is loaded
        if device != "cpu":
            torch.cuda.empty_cache()
    fp32 = timings["fp32"]
    assert fp32 is not None
    use_amp = choose_amp(fp32, timings["amp"])
    return {"device": device, "steps": steps, **timings, "use_amp": use_amp}


def benchmark_precision(
    config: TrainConfig,
    processed_dir: Path,
    steps: int = 50,
    *,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    """Time ``steps`` training steps in full (``fp32``) and in mixed precision (``amp``).

    Each entry has ``s_per_step``, ``peak_mem_mb``, ``loss_finite`` and ``fits``; ``use_amp`` is
    the choice ``amp: auto`` makes from them. On a machine without a GPU ``amp`` is ``None``.
    """
    assert_frozen(processed_dir, lock_path)
    tokenizer = _load_tokenizer(config.model_name)
    spec = _input_spec(tokenizer, config)
    path = processed_dir / "splits" / "train.parquet"
    meta = pd.read_parquet(path, columns=["label", "dataset"])
    positions = stratified_positions(meta, steps * config.effective_batch, config.seed)
    chosen = meta.iloc[positions]
    split = _Split(
        tokens=encode_articles(tokenizer, _file_articles(path, positions), spec),
        labels=chosen["label"].to_numpy(dtype=np.int64),
        datasets=chosen["dataset"].tolist(),
    )
    return _benchmark(config, spec.pad_id, split, steps)


def _use_amp(config: TrainConfig, run_dir: Path, pad_id: int, train: _Split) -> bool:
    """Resolve ``amp``; ``auto`` is measured once per run folder and then read from its file."""
    if _device() == "cpu" or config.amp == "off":
        return False
    if config.amp == "on":
        return True
    path = run_dir / PRECISION_NAME
    if not path.exists():
        steps = AUTO_AMP_STEPS + BENCHMARK_WARMUP_STEPS
        # Rows spread over the whole split: the split file is ordered by dataset.
        wanted = min(len(train.tokens), steps * config.effective_batch)
        rows = np.unique(np.linspace(0, len(train.tokens) - 1, num=wanted).astype(np.int64))
        result = _benchmark(config, pad_id, train.take(rows), steps)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return bool(json.loads(path.read_text(encoding="utf-8"))["use_amp"])


# ---------------------------------------------------------------- run folder


@dataclass
class _Progress:
    step: int = 0  # optimizer steps done
    evals_without_gain: int = 0
    train_seconds: float = 0.0  # time spent in training steps, over all restarts
    best_step: int | None = None
    best_val_macro_f1: float | None = None


def _retry(action: Callable[[], object], attempts: int = 5, wait: float = 2.0) -> None:
    """Run a file operation again when Windows refuses it because a scanner holds the file."""
    for attempt in range(attempts):
        try:
            action()
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(wait)


def _prepare_run_dir(run_dir: Path, config: TrainConfig, resume: bool) -> None:
    config_path = run_dir / CONFIG_NAME
    if config_path.exists():
        if not resume:
            raise RunFolderError(
                f"{run_dir} already holds a run. Continue it with --resume, or choose another "
                "run folder."
            )
        if TrainConfig.from_yaml(config_path) != config:
            raise RunFolderError(
                f"{run_dir} was started with other settings ({config_path}). A run can only be "
                "resumed with the settings it began with."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config.to_yaml(), encoding="utf-8")


def _save_checkpoint(run_dir: Path, learner: _Learner, progress: _Progress) -> None:
    import torch

    cuda = learner.device.startswith("cuda")
    state = {
        "model": learner.model.state_dict(),
        "optimizer": learner.optimizer.state_dict(),
        "scheduler": learner.scheduler.state_dict(),
        "scaler": learner.scaler.state_dict(),
        "progress": asdict(progress),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if cuda else [],
    }
    path = run_dir / CHECKPOINT_NAME
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    _retry(lambda: os.replace(temporary, path))


def _load_checkpoint(run_dir: Path, learner: _Learner) -> _Progress:
    import torch

    path = run_dir / CHECKPOINT_NAME
    state = torch.load(path, map_location=learner.device, weights_only=True)
    learner.model.load_state_dict(state["model"])
    learner.optimizer.load_state_dict(state["optimizer"])
    learner.scheduler.load_state_dict(state["scheduler"])
    learner.scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"] and learner.device.startswith("cuda"):
        torch.cuda.set_rng_state_all([generator.cpu() for generator in state["cuda_rng"]])
    return _Progress(**state["progress"])


def _save_best(
    run_dir: Path, model: Any, tokenizer: Any, spec: InputSpec, metrics: dict[str, Any]
) -> None:
    """Replace ``best/``. The new folder is complete before the old one is removed."""
    temporary, final = run_dir / (BEST_DIR + ".tmp"), run_dir / BEST_DIR
    shutil.rmtree(temporary, ignore_errors=True)
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    spec_text = json.dumps(asdict(spec), indent=2) + "\n"
    (temporary / INPUT_CONFIG_NAME).write_text(spec_text, encoding="utf-8")
    # Written last: its presence says the folder is complete.
    metrics_text = json.dumps(metrics, indent=2) + "\n"
    (temporary / BEST_METRICS_NAME).write_text(metrics_text, encoding="utf-8")
    if final.exists():
        _retry(lambda: shutil.rmtree(final))
    _retry(lambda: temporary.rename(final))


def _recover_best(run_dir: Path, progress: _Progress) -> None:
    """Tidy up after a kill during ``_save_best``; ``best/`` then says what the best model is."""
    temporary, final = run_dir / (BEST_DIR + ".tmp"), run_dir / BEST_DIR
    if not final.exists() and (temporary / BEST_METRICS_NAME).exists():
        temporary.rename(final)
    shutil.rmtree(temporary, ignore_errors=True)
    metrics_path = final / BEST_METRICS_NAME
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        progress.best_step = int(metrics["step"])
        progress.best_val_macro_f1 = float(metrics["val_macro_f1"])
    else:
        progress.best_step = progress.best_val_macro_f1 = None


def _trim_log(path: Path, step: int) -> None:
    """Drop log lines written after the checkpoint a resumed run starts from."""
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if line.strip() and json.loads(line)["step"] <= step]
    path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def _log(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------- training


def _fit(
    config: TrainConfig,
    spec: InputSpec,
    tokenizer: Any,
    train: _Split,
    val: _Split,
    run_dir: Path,
    *,
    resume: bool,
    max_steps: int | None,
    details: dict[str, Any],
) -> dict[str, Any]:
    n = len(train.tokens)
    effective = config.effective_batch
    steps_per_epoch = math.ceil(n / effective)
    total_steps = steps_per_epoch * config.epochs
    last_step = total_steps if max_steps is None else min(total_steps, max_steps)
    use_amp = _use_amp(config, run_dir, spec.pad_id, train)
    warmup_steps = int(config.warmup_share * total_steps)
    learner = _new_learner(config, total_steps, warmup_steps, _device(), use_amp)

    progress = _Progress()
    (run_dir / (CHECKPOINT_NAME + ".tmp")).unlink(missing_ok=True)  # left by a kill while writing
    if resume and (run_dir / CHECKPOINT_NAME).exists():
        progress = _load_checkpoint(run_dir, learner)
    _recover_best(run_dir, progress)
    log_path = run_dir / LOG_NAME
    _trim_log(log_path, progress.step)

    balanced = config.balance == "dataset_label"
    weights = sampling_weights(train.datasets, train.labels.tolist()) if balanced else None
    loss_weights = _loss_weights(train.labels, config)

    def evaluate() -> None:
        started = time.perf_counter()
        margins = _margins(
            learner.model, val.tokens, spec.pad_id, learner.device, 2 * config.batch_size, use_amp
        )
        macro_f1 = float(f1_score(val.labels, margins > 0, average="macro"))
        loss = float(np.mean(np.logaddexp(0.0, margins) - val.labels * margins))
        best_so_far = progress.best_val_macro_f1
        improved = best_so_far is None or macro_f1 > best_so_far
        if improved:
            progress.best_step, progress.best_val_macro_f1 = progress.step, macro_f1
            progress.evals_without_gain = 0
            metrics = {"step": progress.step, "val_macro_f1": macro_f1, "val_loss": loss}
            _save_best(run_dir, learner.model, tokenizer, spec, metrics)
        else:
            progress.evals_without_gain += 1
        entry = {
            "step": progress.step,
            "val_macro_f1": round(macro_f1, 4),
            "val_loss": round(loss, 4),
            "best": improved,
            "eval_seconds": round(time.perf_counter() - started, 1),
        }
        _log(log_path, entry)

    order: NDArray[np.int64] | None = None
    order_epoch = -1
    window_loss = window_seconds = 0.0
    window_steps = 0
    stopped_early = progress.evals_without_gain >= config.patience
    while progress.step < last_step and not stopped_early:
        epoch, index = divmod(progress.step, steps_per_epoch)
        if order is None or order_epoch != epoch:
            order, order_epoch = epoch_order(weights, n, config.seed, epoch), epoch
        rows = order[index * effective : (index + 1) * effective]
        started = time.perf_counter()
        loss = _train_step(learner, train, loss_weights, rows, config.batch_size, spec.pad_id)
        elapsed = time.perf_counter() - started
        progress.step += 1
        progress.train_seconds += elapsed
        window_loss += loss
        window_seconds += elapsed
        window_steps += 1
        if progress.step % config.log_every_steps == 0 or progress.step == last_step:
            entry = {
                "step": progress.step,
                "epoch": epoch,
                "loss": round(window_loss / window_steps, 6),
                "lr": learner.scheduler.get_last_lr()[0],
                "s_per_step": round(window_seconds / window_steps, 4),
            }
            _log(log_path, entry)
            window_loss = window_seconds = 0.0
            window_steps = 0

        evaluated = progress.step % config.eval_every_steps == 0 or progress.step == total_steps
        if evaluated:
            evaluate()
            stopped_early = progress.evals_without_gain >= config.patience
        due = progress.step % config.checkpoint_every_steps == 0 or progress.step == last_step
        if evaluated or due:
            _save_checkpoint(run_dir, learner, progress)

    per_step = round(progress.train_seconds / progress.step, 4) if progress.step else None
    summary = {
        "best_step": progress.best_step,
        "best_val_macro_f1": progress.best_val_macro_f1,
        "steps": progress.step,
        "seconds_per_step": per_step,
        "stopped_early": stopped_early,
        "finished": stopped_early or progress.step >= total_steps,
        "total_steps": total_steps,
        "train_seconds": round(progress.train_seconds, 1),
        "device": learner.device,
        "amp": use_amp,
        "train_rows": n,
        "val_rows": len(val.tokens),
        "truncated_share": {
            "train": round(float(train.tokens.cut.mean()), 4),
            "val": round(float(val.tokens.cut.mean()), 4),
        },
        **details,
    }
    (run_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _read_split(
    processed_dir: Path,
    split: str,
    tokenizer: Any,
    spec: InputSpec,
    run_dir: Path,
    stamp: dict[str, Any],
    *,
    max_rows: int | None = None,
    seed: int = 13,
) -> _Split:
    path = processed_dir / "splits" / f"{split}.parquet"
    meta = pd.read_parquet(path, columns=["label", "dataset"])
    positions = None
    if max_rows is not None and max_rows < len(meta):
        positions = stratified_positions(meta, max_rows, seed)
        meta = meta.iloc[positions]
    stamp = {**stamp, "rows": len(meta), "max_rows": max_rows, "seed": seed}

    def make() -> Tokens:
        return encode_articles(tokenizer, _file_articles(path, positions), spec)

    return _Split(
        tokens=_cached_tokens(run_dir, split, stamp, make),
        labels=meta["label"].to_numpy(dtype=np.int64),
        datasets=meta["dataset"].tolist(),
    )


def train(
    config: TrainConfig,
    processed_dir: Path,
    run_dir: Path,
    *,
    resume: bool = False,
    max_steps: int | None = None,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    """Fine-tune on the frozen train split, selecting the best step on val.

    ``resume`` continues from the run folder's last complete checkpoint (or from the start when
    there is none yet). ``max_steps`` stops early without changing the learning-rate schedule.
    Returns ``best_step``, ``best_val_macro_f1``, ``steps``, ``seconds_per_step`` and
    ``stopped_early`` among others; the same is written to ``summary.json``.
    """
    data_sha = assert_frozen(processed_dir, lock_path)
    _prepare_run_dir(run_dir, config, resume)
    tokenizer = _load_tokenizer(config.model_name)
    spec = _input_spec(tokenizer, config)
    stamp = {"data_manifest_sha256": data_sha, "model_name": config.model_name, **asdict(spec)}
    train_split = _read_split(
        processed_dir,
        "train",
        tokenizer,
        spec,
        run_dir,
        stamp,
        max_rows=config.max_train_rows,
        seed=config.seed,
    )
    val_split = _read_split(processed_dir, "val", tokenizer, spec, run_dir, stamp)
    details = {"model_name": config.model_name, "data_manifest_sha256": data_sha}
    return _fit(
        config,
        spec,
        tokenizer,
        train_split,
        val_split,
        run_dir,
        resume=resume,
        max_steps=max_steps,
        details=details,
    )


# ---------------------------------------------------------------- scoring


def predict_logits(
    model_dir: Path, frame: pd.DataFrame, *, batch_size: int = 32, device: str | None = None
) -> NDArray[np.float64]:
    """The margin ``z1 - z0`` of each row of ``frame`` (columns ``title`` and ``text``).

    ``model_dir`` is a ``best/`` folder: model, tokenizer and ``input_config.json``.
    """
    from transformers import AutoModelForSequenceClassification

    spec = InputSpec(**json.loads((model_dir / INPUT_CONFIG_NAME).read_text(encoding="utf-8")))
    tokens = encode_articles(_load_tokenizer(str(model_dir)), _frame_articles(frame), spec)
    target = _device(device)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(target)
    return _margins(model, tokens, spec.pad_id, target, batch_size, use_amp=False)


def _as_scorer(model_dir: Path, temperature: float) -> Scorer:
    def score(frame: pd.DataFrame) -> NDArray[np.float64]:
        return apply_temperature(predict_logits(model_dir, frame), temperature)

    return score


def load_scorer(run_dir: Path) -> Scorer:
    """Calibrated scorer of a finished run: ``best/`` and the temperature from ``fnd calibrate``."""
    calibration = json.loads((run_dir / CALIBRATION_NAME).read_text(encoding="utf-8"))
    return _as_scorer(run_dir / BEST_DIR, float(calibration["temperature"]))


def fit_scorer(config: TrainConfig, *, work_dir: Path = DEFAULT_FIT_DIR) -> Fit:
    """A ``Fit`` that fine-tunes a new model on the frames it is given (leave-one-dataset-out).

    Each call trains in a new folder under ``work_dir``; the folder is removed when the returned
    scorer is no longer used.
    """

    def fit(train_frame: pd.DataFrame, val: pd.DataFrame, calib: pd.DataFrame) -> Scorer:
        if config.max_train_rows is not None and config.max_train_rows < len(train_frame):
            positions = stratified_positions(train_frame, config.max_train_rows, config.seed)
            train_frame = train_frame.iloc[positions]
        work_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="fold-", dir=work_dir))
        (run_dir / CONFIG_NAME).write_text(config.to_yaml(), encoding="utf-8")
        tokenizer = _load_tokenizer(config.model_name)
        spec = _input_spec(tokenizer, config)

        def split(frame: pd.DataFrame) -> _Split:
            return _Split(
                tokens=encode_articles(tokenizer, _frame_articles(frame), spec),
                labels=frame["label"].to_numpy(dtype=np.int64),
                datasets=frame["dataset"].tolist(),
            )

        details = {"model_name": config.model_name}
        _fit(
            config,
            spec,
            tokenizer,
            split(train_frame),
            split(val),
            run_dir,
            resume=False,
            max_steps=None,
            details=details,
        )
        (run_dir / CHECKPOINT_NAME).unlink(missing_ok=True)  # only best/ is needed from here on
        best = run_dir / BEST_DIR
        temperature = fit_temperature(predict_logits(best, calib), calib["label"])
        scorer = _as_scorer(best, temperature)
        weakref.finalize(scorer, shutil.rmtree, str(run_dir), True)
        return scorer

    return fit
