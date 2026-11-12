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
) -> None:
    """Clean, deduplicate and split the raw datasets; write parquet files and a manifest."""
    from fnd_ml.data.build import build_corpus

    manifest = build_corpus(raw_dir, out_dir, seed=seed)
    typer.echo(json.dumps({k: manifest[k] for k in ("sources", "dedup", "rows")}, indent=2))
    shared = manifest["shared_groups_between_splits"], manifest["shared_hashes_between_splits"]
    typer.echo(f"shared groups / hashes between splits: {shared[0]} / {shared[1]}")


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
