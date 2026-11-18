"""``fnd eval``: score the registered models on every protocol, apply the gates, write the report.

The models compared are the registered artifacts: the baseline, XGBoost and, when one is
exported, the transformer, scored through its ONNX file as the API serves it. Every (model, set)
is scored once and its raw and calibrated scores are kept in ``cache_dir``; the folds of leave
one dataset out are kept there too. A run that is interrupted, or repeated for the report's
sake, reads them back, so the test split is scored once per model version.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fnd_ml.calibration.temperature import apply_temperature, sigmoid
from fnd_ml.common.artifacts import read_manifest, resolve
from fnd_ml.data.build import DEFAULT_LOCK, MANIFEST_NAME, assert_frozen
from fnd_ml.eval.gates import GateResult, evaluate, load_gates
from fnd_ml.eval.plots import reliability_diagram
from fnd_ml.eval.protocols import TouchLog, external, in_distribution, leave_one_dataset_out
from fnd_ml.eval.report import write_eval_report
from fnd_ml.models import baseline
from fnd_ml.models.scorer import Fit, Scorer
from fnd_ml.models.train_baseline import REPORT_SUFFIX

DEFAULT_GATES = Path("ml/configs/eval_gates.yaml")
DEFAULT_CACHE = Path("ml/runs/eval_scores")
DEFAULT_LODO_CONFIG = Path("ml/configs/deberta_small.yaml")
DEFAULT_THREADS = 4
LODO_TRAIN_ROWS = 20_000  # per fold, when the transformer's config sets no cap of its own
TOUCH_LOG_NAME = "test_touch_log.json"
TEST = "test"
SCORED_COLUMNS = ["label", "dataset", "length_bucket", "title", "text", "model_text"]
HYGIENE_KEYS = ("shared_groups_between_splits", "shared_near_duplicate_clusters_between_splits")

Pair = tuple[NDArray[np.float64], NDArray[np.float64]]


class EvalError(RuntimeError):
    """The comparison cannot be made: no model is registered."""


@dataclass(frozen=True)
class Model:
    """A registered artifact: how it scores a frame and how its family is trained again."""

    name: str
    version: str
    description: str
    both: Callable[[pd.DataFrame], Pair]  # uncalibrated and calibrated P(unreliable)
    fit: Fit | None = None
    metadata: dict[str, Any] | None = None


def _baseline(directory: Path, version: str) -> Model:
    model = baseline.BaselineModel.load(directory)

    def both(frame: pd.DataFrame) -> Pair:
        raw = model.predict_raw(frame["model_text"])
        return raw, model.to_calibrated(raw)

    description = "TF-IDF (hashed word n-grams) + logistic regression, isotonic calibration"
    fit = partial(baseline.fit_scorer, config=model.config)
    return Model("baseline", version, description, both, fit)


def _xgb(directory: Path, version: str) -> Model:
    from fnd_ml.models import xgb

    model = xgb.XgbModel.load(directory)

    def both(frame: pd.DataFrame) -> Pair:
        raw = model.predict_raw(frame["model_text"].tolist())
        return raw, model.to_calibrated(raw)

    description = "XGBoost over stylometric features and an SVD of the TF-IDF, isotonic calibration"
    return Model("xgb", version, description, both, partial(xgb.fit_scorer, config=model.config))


def _transformer(directory: Path, version: str, threads: int) -> Model:
    from fnd_ml.inference.onnx_classifier import OnnxClassifier

    classifier = OnnxClassifier.load(directory, threads=threads)

    def both(frame: pd.DataFrame) -> Pair:
        articles = zip(frame["title"].tolist(), frame["text"].tolist(), strict=True)
        margins = classifier.margins(articles)
        return sigmoid(margins), apply_temperature(margins, classifier.temperature)

    variant = classifier.metadata.get("variant", "fp32")
    description = f"{classifier.name} through ONNX Runtime ({variant}), temperature scaling"
    return Model("transformer", version, description, both, None, classifier.metadata)


def load_models(artifacts_dir: Path, *, threads: int = DEFAULT_THREADS) -> list[Model]:
    """The registered baseline, XGBoost and transformer artifacts, in that order."""
    registry = read_manifest(artifacts_dir)
    loaders: dict[str, Callable[[Path, str], Model]] = {
        "baseline": _baseline,
        "xgb": _xgb,
        "transformer": partial(_transformer, threads=threads),
    }
    models = []
    for kind, load in loaders.items():
        directory = resolve(artifacts_dir, kind)
        if directory is not None:
            models.append(load(directory, registry[kind]["version"]))
    return models


def _released(frame: pd.DataFrame) -> Pair:
    raise RuntimeError("the artifacts are released once every set is scored; read the cache")


class _Scores:
    """Raw and calibrated scores of every (model, set), kept on disk; a set is a frame by name."""

    def __init__(self, cache_dir: Path, data_sha: str) -> None:
        self._dir = cache_dir
        self._sha = data_sha[:8]
        self._frames: dict[str, pd.DataFrame] = {}

    def add(self, name: str, frame: pd.DataFrame) -> None:
        self._frames[name] = frame

    def _path(self, model: Model, name: str) -> Path:
        return self._dir / f"{model.name}_{model.version}_{self._sha}_{name}.npz"

    def cached(self, model: Model, name: str) -> bool:
        return self._path(model, name).exists()

    def get(self, model: Model, name: str) -> Pair:
        path, frame = self._path(model, name), self._frames[name]
        if path.exists():
            with np.load(path) as arrays:
                if len(arrays["raw"]) == len(frame):
                    return arrays["raw"], arrays["calibrated"]
        raw, calibrated = model.both(frame)
        self._dir.mkdir(parents=True, exist_ok=True)
        np.savez(path, raw=raw, calibrated=calibrated)
        return raw, calibrated

    def scorers(self, models: list[Model], which: int) -> dict[str, Scorer]:
        """Scorers for the protocols: ``which`` is 0 for uncalibrated, 1 for calibrated."""

        def scorer(model: Model) -> Scorer:
            def score(frame: pd.DataFrame) -> NDArray[np.float64]:
                name = next(name for name, known in self._frames.items() if known is frame)
                return self.get(model, name)[which]

            return score

        return {model.name: scorer(model) for model in models}


def _transformer_fit(config_path: Path, work_dir: Path) -> tuple[Fit, dict[str, Any], str]:
    """The ``Fit`` of the capped transformer folds, its settings for the report, and a cache tag."""
    from fnd_ml.models import transformer

    config = transformer.TrainConfig.from_yaml(config_path)
    if config.max_train_rows is None:
        config = replace(config, max_train_rows=LODO_TRAIN_ROWS)
    settings = {
        "model_name": config.model_name,
        "max_train_rows": config.max_train_rows,
        "epochs": config.epochs,
        "max_length": config.max_length,
    }
    tag = hashlib.sha256(config.to_yaml().encode()).hexdigest()[:8]
    return transformer.fit_scorer(config, work_dir=work_dir), settings, tag


def _corpus(processed_dir: Path, columns: list[str]) -> pd.DataFrame:
    """Every split in one frame, with the split's name in ``split``."""
    wanted = ["dataset", "label", *columns]
    parts = [
        pd.read_parquet(processed_dir / "splits" / f"{split}.parquet", columns=wanted).assign(
            split=split
        )
        for split in ("train", "val", "calib", TEST)
    ]
    return pd.concat(parts, ignore_index=True)


def _folds(
    fit: Fit, corpus: pd.DataFrame, cache_dir: Path, key: str, cap: int | None
) -> dict[str, dict[str, Any]]:
    """Leave one dataset out, one fold at a time; a finished fold is read back from the cache."""
    folds: dict[str, dict[str, Any]] = {}
    for dataset in sorted(corpus["dataset"].unique()):
        path = cache_dir / f"lodo_{key}_{dataset}.json"
        if not path.exists():
            fold = leave_one_dataset_out(fit, corpus, datasets=[dataset])[dataset]
            if cap is not None:
                fold["train_rows"] = min(fold["train_rows"], cap)
            cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(fold) + "\n", encoding="utf-8")
        folds[dataset] = json.loads(path.read_text(encoding="utf-8"))
    return folds


def _leak_probe(reports_dir: Path, models: list[Model]) -> dict[str, Any] | None:
    """The label probe of the registered baseline's training report, if that report is there."""
    path = reports_dir / f"baseline_{REPORT_SUFFIX}.json"
    version = next((model.version for model in models if model.name == "baseline"), None)
    if version is None or not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    probe: dict[str, Any] | None = report.get("leak_probe")
    return probe if report.get("version") == version else None


def run_eval(
    out: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    *,
    gates_path: Path = DEFAULT_GATES,
    lock_path: Path = DEFAULT_LOCK,
    cache_dir: Path = DEFAULT_CACHE,
    with_transformer_lodo: bool = False,
    lodo_config: Path = DEFAULT_LODO_CONFIG,
    allow_retest: str | None = None,
    threads: int = DEFAULT_THREADS,
) -> tuple[dict[str, Any], list[GateResult]]:
    """Compare the registered models and write ``<out>.json``, ``<out>.md`` and the figures.

    ``out`` is the report's path without an extension; its folder also holds the training
    reports and the test-split log. Raises ``RetestError`` before anything is scored when a model
    version would score the test split a second time and ``allow_retest`` gives no reason.
    """
    data_sha = assert_frozen(processed_dir, lock_path)
    limits = load_gates(gates_path)
    reports_dir, name = out.parent, out.name
    models = load_models(artifacts_dir, threads=threads)
    if not models:
        raise EvalError(f"No model is registered in {artifacts_dir}. Train the baseline first.")

    scores = _Scores(cache_dir, data_sha)
    touch_log = TouchLog(reports_dir / TOUCH_LOG_NAME)
    fresh = [model for model in models if not scores.cached(model, TEST)]
    for model in fresh:
        touch_log.check(model.name, model.version, allow_retest)

    sets = {
        TEST: pd.read_parquet(processed_dir / "splits" / "test.parquet", columns=SCORED_COLUMNS)
    }
    for external_set in limits["external_sets"]:
        path = processed_dir / "eval" / f"{external_set}.parquet"
        sets[external_set] = pd.read_parquet(path, columns=SCORED_COLUMNS)
    for set_name, frame in sets.items():
        scores.add(set_name, frame)

    touches = []
    for model in models:
        again = bool(touch_log.entries(model.name, model.version))
        scores.get(model, TEST)
        if model in fresh:
            reason = allow_retest if again else None
            touch_log.record(model.name, model.version, report=name, reason=reason)
        reasons = [entry["reason"] for entry in touch_log.entries(model.name, model.version)]
        touches.append(
            {
                "model": model.name,
                "version": model.version,
                "scored": model in fresh,
                "reason": next((reason for reason in reversed(reasons) if reason), None),
            }
        )

    # Everything an artifact is needed for happens here. From now on its scores are read from
    # the cache, and the served model is not held in memory while the folds below are trained.
    for model in models:
        for set_name in sets:
            scores.get(model, set_name)
    models = [replace(model, both=_released) for model in models]
    gc.collect()

    calibrated, raw = scores.scorers(models, 1), scores.scorers(models, 0)
    outside = {set_name: frame for set_name, frame in sets.items() if set_name != TEST}
    notes = [
        "The baseline and XGBoost score the test split once when a version is trained; this "
        "report scores it again with the same files and logs that in `test_touch_log.json`.",
        "The transformer is scored through its exported ONNX file, as the API serves it.",
    ]

    lodo: dict[str, dict[str, dict[str, Any]]] = {}
    classic = [model for model in models if model.fit is not None]
    if classic:
        corpus = _corpus(processed_dir, ["model_text"])
        for model in classic:
            assert model.fit is not None
            key = f"{model.name}_{model.version}_{data_sha[:8]}"
            lodo[model.name] = _folds(model.fit, corpus, cache_dir, key, None)
        del corpus
    if with_transformer_lodo:
        fit, settings, tag = _transformer_fit(lodo_config, cache_dir / "lodo_fit")
        corpus = _corpus(processed_dir, ["title", "text"])
        key = f"transformer_{tag}_{data_sha[:8]}"
        lodo["transformer"] = _folds(fit, corpus, cache_dir, key, settings["max_train_rows"])
        notes.append(
            f"Leave one dataset out, transformer: `{settings['model_name']}` fine-tuned per fold "
            f"on at most {settings['max_train_rows']:,} training rows, {settings['epochs']} "
            f"epochs, {settings['max_length']} tokens, scored with PyTorch."
        )
    else:
        notes.append(
            "The transformer was not part of leave one dataset out in this run "
            "(`--with-transformer-lodo`)."
        )

    figures: dict[str, dict[str, str]] = {}
    for model in models:
        figures[model.name] = {}
        for set_name, frame in sets.items():
            before, after = scores.get(model, set_name)
            relative = f"figures/reliability_{model.name}_{set_name}.png"
            title = f"{model.name} {model.version} on {set_name} ({len(frame):,} rows)"
            reliability_diagram(frame["label"], before, after, reports_dir / relative, title)
            figures[model.name][set_name] = relative

    manifest = json.loads((processed_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    served = next((model.metadata for model in models if model.name == "transformer"), None)
    results: dict[str, Any] = {
        "name": name,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_manifest_sha256": data_sha,
        "models": {
            model.name: {"version": model.version, "description": model.description}
            for model in models
        },
        "data": {key: manifest.get(key) for key in HYGIENE_KEYS},
        "leak_probe": _leak_probe(reports_dir, models),
        "in_distribution": in_distribution(calibrated, sets[TEST], raw=raw),
        "leave_one_dataset_out": lodo,
        "external": external(calibrated, outside),
        "transformer": served,
        "figures": figures,
        "test_touches": touches,
        "notes": notes,
    }
    gates = evaluate(results, gates_path)
    write_eval_report(results, gates, out)
    return results, gates
