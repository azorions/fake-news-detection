"""``fnd`` command line: data and model commands.

Run from the repository root, for example ``uv run fnd data download``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from fnd_ml.data.download import ChecksumMismatch, DownloadError, fetch_dataset, load_registry

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Download and prepare datasets.")
app.add_typer(data_app, name="data")
train_app = typer.Typer(no_args_is_help=True, help="Train models.")
app.add_typer(train_app, name="train")

RegistryOption = Annotated[Path, typer.Option(help="Dataset registry file.")]
RawDirOption = Annotated[Path, typer.Option(help="Where raw dataset files are stored.")]

DEFAULT_REGISTRY = Path("data/registry/datasets.yaml")
DEFAULT_RAW_DIR = Path("data/raw")
BUILD_RAM_GB = 3.0  # the whole corpus is held in memory several times over while it is cleaned


def _check_memory(gb: float, what: str, force: bool) -> None:
    """Stop early, with the reason, instead of being killed half-way for lack of memory."""
    from fnd_ml.common import resources

    if force:
        return
    try:
        resources.require_free_ram(gb, what)
    except resources.InsufficientMemory as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error


@data_app.command("download")
def data_download(
    only: Annotated[
        list[str] | None, typer.Option(help="Dataset name; repeat to fetch several.")
    ] = None,
    registry: RegistryOption = DEFAULT_REGISTRY,
    raw_dir: RawDirOption = DEFAULT_RAW_DIR,
) -> None:
    """Download every registered dataset (or only the named ones) and verify checksums."""
    names = only or list(load_registry(registry))
    failed: list[str] = []
    for name in names:
        try:
            paths = fetch_dataset(name, raw_dir, registry)
        except (DownloadError, ChecksumMismatch, KeyError) as exc:
            failed.append(name)
            typer.echo(f"FAILED  {name}: {exc}")
            continue
        size_mb = sum(path.stat().st_size for path in paths) / 1e6
        typer.echo(f"ok      {name}: {len(paths)} file(s), {size_mb:.1f} MB")
    if failed:
        raise typer.Exit(code=1)


@data_app.command("build")
def data_build(
    raw_dir: RawDirOption = DEFAULT_RAW_DIR,
    out_dir: Annotated[Path, typer.Option(help="Output directory.")] = Path("data/processed"),
    seed: Annotated[int, typer.Option(help="Split seed.")] = 13,
    force: Annotated[bool, typer.Option(help="Run even when free memory looks too low.")] = False,
) -> None:
    """Clean, deduplicate and split the raw datasets; write parquet files and a manifest."""
    from fnd_ml.data.build import build_corpus

    _check_memory(BUILD_RAM_GB, "The data build", force)
    manifest = build_corpus(raw_dir, out_dir, seed=seed)
    shown = ("sources", "dedup", "near_dedup", "rows", "dates", "external")
    typer.echo(json.dumps({key: manifest[key] for key in shown}, indent=2))
    shared = (
        manifest["shared_groups_between_splits"],
        manifest["shared_hashes_between_splits"],
        manifest["shared_near_duplicate_clusters_between_splits"],
    )
    typer.echo(
        "shared groups / hashes / near-duplicate clusters between splits: "
        f"{shared[0]} / {shared[1]} / {shared[2]}"
    )
    if manifest["external_missing"]:
        missing = ", ".join(manifest["external_missing"])
        typer.echo(f"evaluation sets not found (run `fnd data download`): {missing}")


@data_app.command("freeze")
def data_freeze(
    processed_dir: Annotated[Path, typer.Option(help="Output of `fnd data build`.")] = Path(
        "data/processed"
    ),
    lock: Annotated[Path, typer.Option(help="Lock file to write.")] = Path(
        "ml/configs/data_v2.lock"
    ),
) -> None:
    """Freeze the current build: models may only be trained and compared on this manifest."""
    from fnd_ml.data.build import write_lock

    digest = write_lock(processed_dir, lock)
    typer.echo(f"frozen: manifest sha256 {digest} -> {lock}")


@data_app.command("probe")
def data_probe(
    processed_dir: Annotated[Path, typer.Option(help="Output of `fnd data build`.")] = Path(
        "data/processed"
    ),
    report: Annotated[
        Path | None, typer.Option(help="Baseline report (JSON) whose top features are checked.")
    ] = None,
    identifiers: Annotated[Path, typer.Option(help="Source identifier list.")] = Path(
        "data/registry/source_identifiers.txt"
    ),
    max_rows_per_dataset: Annotated[int, typer.Option(help="Sample size per dataset.")] = 8000,
) -> None:
    """Leak probes: can the dataset be told from the text, and do top features name a source?"""
    from fnd_ml.data.leak_probe import (
        count_unnamed,
        label_probe,
        load_identifiers,
        load_probe_sample,
        origin_probe,
    )

    sample = load_probe_sample(processed_dir, max_rows_per_dataset=max_rows_per_dataset)
    result: dict[str, object] = {"origin": origin_probe(sample)}
    flagged: list[str] = []
    if report is not None:
        top_features = json.loads(report.read_text(encoding="utf-8"))["top_features"]
        flagged = label_probe(top_features, load_identifiers(identifiers))
        result["label"] = {"flagged": flagged, "unnamed_buckets": count_unnamed(top_features)}
    typer.echo(json.dumps(result, indent=2))
    if flagged:
        raise typer.Exit(code=1)


@train_app.command("baseline")
def train_baseline_command(
    processed_dir: Annotated[Path, typer.Option(help="Output of `fnd data build`.")] = Path(
        "data/processed"
    ),
    artifacts_dir: Annotated[Path, typer.Option(help="Model artifact root.")] = Path(
        "ml/artifacts"
    ),
    reports_dir: Annotated[Path, typer.Option(help="Where reports are written.")] = Path(
        "ml/reports"
    ),
    config: Annotated[Path, typer.Option(help="Baseline config file.")] = Path(
        "ml/configs/baseline.yaml"
    ),
) -> None:
    """Train, calibrate and register the TF-IDF + logistic regression baseline."""
    import yaml

    from fnd_ml.models.baseline import BaselineConfig
    from fnd_ml.models.train_baseline import train_baseline

    settings = BaselineConfig(**yaml.safe_load(config.read_text(encoding="utf-8")))
    report = train_baseline(processed_dir, artifacts_dir, reports_dir, settings)
    typer.echo(f"version {report['version']}")
    typer.echo(json.dumps(report["metrics"]["test"], indent=2))
    typer.echo(json.dumps(report["thresholds"]))
