"""``fnd export``: from a calibrated run to the artifact the API serves.

The model is exported to ONNX in full precision and quantized to int8. The full-precision file
must give PyTorch's probabilities; both files are scored on a sample of the val split and timed;
one of them is chosen and written, with tokenizer, calibration and metadata, as
``<artifacts>/transformer/<version>/``. The test split is not read here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score

from fnd_ml.calibration.run import (
    THRESHOLDS_NAME,
    calibration_documents,
    require_trained_on,
    write_documents,
)
from fnd_ml.common.artifacts import register
from fnd_ml.data.build import DEFAULT_LOCK, assert_frozen
from fnd_ml.data.download import sha256_file
from fnd_ml.export.benchmark import benchmark
from fnd_ml.export.onnx_export import export_onnx
from fnd_ml.export.parity import encode_frame, load_spec, parity
from fnd_ml.export.quantize import quantize_int8
from fnd_ml.inference.onnx_classifier import (
    DEFAULT_THREADS,
    METADATA_NAME,
    MODEL_NAME,
    TOKENIZER_DIR,
    TOKENIZER_NAME,
    open_session,
    run_margins,
)
from fnd_ml.models.train_baseline import stratified_positions
from fnd_ml.models.transformer import BEST_DIR, CALIBRATION_NAME

KIND = "transformer"
MAX_PARITY_DIFF = 0.001  # fp32 ONNX against PyTorch, in probability
MAX_INT8_F1_DROP = 0.01  # val macro-F1 that int8 may cost
LATENCY_GATE_MS = 500.0  # p95 of one article at the full window, on the CPU
LATENCY_LENGTHS = (128, 256, 512)


class ExportError(RuntimeError):
    """The exported model does not reproduce the PyTorch model, or the run is not ready."""


def choose_variant(macro_f1: Mapping[str, float], p95_ms: Mapping[str, float]) -> str:
    """``int8`` when it costs at most 0.01 val macro-F1 and is faster; otherwise ``fp32``."""
    drop = macro_f1["fp32"] - macro_f1["int8"]
    faster = p95_ms["int8"] < p95_ms["fp32"]
    return "int8" if drop <= MAX_INT8_F1_DROP + 1e-9 and faster else "fp32"


def _text_keys(latency: Mapping[str, Mapping[int, Any]]) -> dict[str, dict[str, Any]]:
    """Token counts as text: what a JSON file gives back when it is read again."""
    return {
        name: {str(length): timing for length, timing in by_length.items()}
        for name, by_length in latency.items()
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _read(processed_dir: Path, split: str, rows: int | None, seed: int) -> pd.DataFrame:
    """A split, or a sample of ``rows`` of it that keeps its mix of datasets and labels."""
    frame = pd.read_parquet(
        processed_dir / "splits" / f"{split}.parquet", columns=["title", "text", "label", "dataset"]
    )
    if rows is None or rows >= len(frame):
        return frame
    return frame.iloc[stratified_positions(frame, rows, seed)]


def export_run(
    run_dir: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    *,
    lock_path: Path = DEFAULT_LOCK,
    val_rows: int = 2000,
    threads: int = DEFAULT_THREADS,
    runs: int = 100,
    seed: int = 13,
) -> dict[str, Any]:
    """Export, check and register the run's best model; return the artifact's metadata."""
    data_sha = assert_frozen(processed_dir, lock_path)
    summary = require_trained_on(run_dir, data_sha)
    for name in (CALIBRATION_NAME, THRESHOLDS_NAME):
        if not (run_dir / name).exists():
            raise ExportError(
                f"{run_dir / name} is missing. Run `fnd calibrate --run {run_dir}` first."
            )
    best = run_dir / BEST_DIR
    spec = load_spec(best)

    work = run_dir / "export"
    files = {"fp32": export_onnx(best, work / "model_fp32.onnx")}
    files["int8"] = quantize_int8(files["fp32"], work / "model_int8.onnx")

    val = _read(processed_dir, "val", val_rows, seed)
    check = parity(best, files["fp32"], val, seed=seed)
    if check["max_abs_probability_diff"] > MAX_PARITY_DIFF:
        raise ExportError(
            f"The exported model differs from PyTorch by {check['max_abs_probability_diff']:.4f} "
            f"in probability on {int(check['n'])} val rows (limit {MAX_PARITY_DIFF}). "
            "Nothing was registered."
        )

    rows = encode_frame(best, val)
    # The model's own window is always timed: it is the length the latency gate is about.
    shorter = {length for length in LATENCY_LENGTHS if length < spec.max_length}
    lengths = tuple(sorted(shorter | {spec.max_length}))
    macro_f1: dict[str, float] = {}
    latency: dict[str, dict[int, dict[str, float]]] = {}
    for name, path in files.items():
        margins = run_margins(open_session(path, threads), rows, spec.pad_id)
        macro_f1[name] = float(f1_score(val["label"], margins > 0, average="macro"))
        latency[name] = benchmark(
            path, best, spec=spec, lengths=lengths, runs=runs, threads=threads
        )
    gate_length = lengths[-1]
    p95 = {name: timing[gate_length]["p95_ms"] for name, timing in latency.items()}
    variant = choose_variant(macro_f1, p95)

    if variant == "int8":
        # The temperature was fitted on PyTorch's logits; quantization moves them, so the shipped
        # file gets its own temperature and thresholds, again from the calib split only.
        calib = _read(processed_dir, "calib", None, seed)
        session = open_session(files["int8"], threads)
        logits = run_margins(session, encode_frame(best, calib), spec.pad_id)
        calibration, thresholds = calibration_documents(logits, calib["label"], data_sha)
    else:
        calibration = json.loads((run_dir / CALIBRATION_NAME).read_text(encoding="utf-8"))
        thresholds = json.loads((run_dir / THRESHOLDS_NAME).read_text(encoding="utf-8"))

    gate_exceptions = []
    if p95[variant] > LATENCY_GATE_MS:
        gate_exceptions.append(
            f"classifier p95 at {gate_length} tokens is {p95[variant]:.0f} ms on {threads} "
            f"threads (limit {LATENCY_GATE_MS:.0f} ms)"
        )

    model_sha = sha256_file(files[variant])
    version = "v1-" + model_sha[:8]
    artifact = artifacts_dir / KIND / version
    (artifact / TOKENIZER_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(files[variant], artifact / MODEL_NAME)
    shutil.copyfile(best / TOKENIZER_NAME, artifact / TOKENIZER_DIR / TOKENIZER_NAME)
    write_documents(artifact, calibration, thresholds)
    metadata: dict[str, Any] = {
        "version": version,
        "kind": KIND,
        "base_model": summary["model_name"],
        "variant": variant,
        **asdict(spec),
        "git_commit": _git_commit(),
        "data_manifest_sha256": data_sha,
        "val_metrics": {
            "best_step": summary["best_step"],
            "macro_f1_pytorch_all_rows": summary["best_val_macro_f1"],
            "sample_rows": len(val),
            "macro_f1_fp32": round(macro_f1["fp32"], 4),
            "macro_f1_int8": round(macro_f1["int8"], 4),
        },
        "parity": check,
        "latency_ms": {"threads": threads, "runs": runs, **_text_keys(latency)},
        "file_bytes": {name: path.stat().st_size for name, path in files.items()},
        "model_sha256": model_sha,
        "gate_exceptions": gate_exceptions,
    }
    text = json.dumps(metadata, indent=2) + "\n"
    (artifact / METADATA_NAME).write_text(text, encoding="utf-8")
    register(artifacts_dir, KIND, version, f"{KIND}/{version}")
    return metadata
