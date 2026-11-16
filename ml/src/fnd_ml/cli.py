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
TRAIN_RAM_GB = 3.0  # the TF-IDF matrix of the train split, plus the solver's working copies
SATIRE_RAM_GB = 1.0  # headlines only: about 28,000 short texts
TOPICS_RAM_GB = 3.0  # PyTorch, the embedding model and the corpus texts


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


@data_app.command("topics")
def data_topics(
    processed_dir: Annotated[Path, typer.Option(help="Output of `fnd data build`.")] = Path(
        "data/processed"
    ),
    k: Annotated[int, typer.Option(help="Number of topic groups.")] = 50,
    seed: Annotated[int, typer.Option(help="Clustering seed.")] = 13,
    force: Annotated[bool, typer.Option(help="Run even when free memory looks too low.")] = False,
) -> None:
    """Give every corpus text a topic group (k-means over sentence embeddings; needs PyTorch)."""
    from fnd_ml.data.topics import write_topics

    _check_memory(TOPICS_RAM_GB, "Embedding the corpus", force)
    summary = write_topics(processed_dir, k=k, seed=seed)
    sizes = sorted((topic["size"] for topic in summary["topics"]), reverse=True)
    typer.echo(
        f"{summary['k']} topics over {summary['rows']} texts; largest "
        f"{summary['largest_topic_share']:.1%}, smallest {sizes[-1]} texts"
    )


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


ProcessedOption = Annotated[Path, typer.Option(help="Output of `fnd data build`.")]
ArtifactsOption = Annotated[Path, typer.Option(help="Model artifact root.")]
ReportsOption = Annotated[Path, typer.Option(help="Where reports are written.")]
ForceOption = Annotated[bool, typer.Option(help="Run even when free memory looks too low.")]


def _frozen_or_exit(error: Exception) -> typer.Exit:
    typer.echo(str(error))
    return typer.Exit(code=2)


@train_app.command("baseline")
def train_baseline_command(
    processed_dir: ProcessedOption = Path("data/processed"),
    artifacts_dir: ArtifactsOption = Path("ml/artifacts"),
    reports_dir: ReportsOption = Path("ml/reports"),
    config: Annotated[Path, typer.Option(help="Baseline config file.")] = Path(
        "ml/configs/baseline.yaml"
    ),
    ablation: Annotated[
        str | None,
        typer.Option(help="Run an ablation instead of training. `char`: character n-grams."),
    ] = None,
    force: ForceOption = False,
) -> None:
    """Train, calibrate and register the TF-IDF + logistic regression baseline."""
    import yaml

    from fnd_ml.data.build import ManifestMismatch
    from fnd_ml.models.baseline import BaselineConfig
    from fnd_ml.models.train_baseline import ablation_char_ngrams, train_baseline

    if ablation not in (None, "char"):
        raise typer.BadParameter(f"unknown ablation {ablation!r}; the only one is `char`")
    settings = BaselineConfig(**yaml.safe_load(config.read_text(encoding="utf-8")))
    _check_memory(TRAIN_RAM_GB, "Training the baseline", force)
    try:
        if ablation == "char":
            result = ablation_char_ngrams(processed_dir, reports_dir, settings)
            typer.echo(json.dumps(result["variants"], indent=2))
            verdict = "adopt" if result["adopt_char_ngrams"] else "do not adopt"
            typer.echo(f"macro-F1 gain {result['macro_f1_gain']:+.4f}: {verdict} char n-grams")
            return
        report = train_baseline(processed_dir, artifacts_dir, reports_dir, settings)
    except ManifestMismatch as error:
        raise _frozen_or_exit(error) from error
    typer.echo(f"version {report['version']}")
    typer.echo(json.dumps({"leak_probe": report["leak_probe"]}))
    if report["leak_probe"]["flagged"]:
        typer.echo("A strongest feature names a source: the test split was not scored.")
        raise typer.Exit(code=1)
    typer.echo(json.dumps(report["metrics"]["test"], indent=2))
    typer.echo(json.dumps(report["thresholds"]))


@train_app.command("xgb")
def train_xgb_command(
    processed_dir: ProcessedOption = Path("data/processed"),
    artifacts_dir: ArtifactsOption = Path("ml/artifacts"),
    reports_dir: ReportsOption = Path("ml/reports"),
    config: Annotated[Path, typer.Option(help="XGBoost config file.")] = Path(
        "ml/configs/xgb.yaml"
    ),
    force: ForceOption = False,
) -> None:
    """Train the comparison model: XGBoost over stylometric features and a TF-IDF SVD."""
    import yaml

    from fnd_ml.data.build import ManifestMismatch
    from fnd_ml.models.xgb import XgbConfig, train_xgb

    settings = XgbConfig(**yaml.safe_load(config.read_text(encoding="utf-8")))
    _check_memory(TRAIN_RAM_GB, "Training the XGBoost model", force)
    try:
        report = train_xgb(processed_dir, artifacts_dir, reports_dir, settings)
    except ManifestMismatch as error:
        raise _frozen_or_exit(error) from error
    typer.echo(f"version {report['version']} ({report['trees']} trees)")
    typer.echo(json.dumps(report["metrics"]["test"], indent=2))
    typer.echo(json.dumps(report["thresholds"]))


@train_app.command("satire")
def train_satire_command(
    processed_dir: ProcessedOption = Path("data/processed"),
    artifacts_dir: ArtifactsOption = Path("ml/artifacts"),
    reports_dir: ReportsOption = Path("ml/reports"),
    config: Annotated[Path, typer.Option(help="Satire scorer config file.")] = Path(
        "ml/configs/satire.yaml"
    ),
    force: ForceOption = False,
) -> None:
    """Train the satire headline scorer on the satire evaluation set (80/10/10 by text hash)."""
    import yaml

    from fnd_ml.data.build import ManifestMismatch
    from fnd_ml.models.baseline import BaselineConfig
    from fnd_ml.models.satire import train_satire

    settings = BaselineConfig(**yaml.safe_load(config.read_text(encoding="utf-8")))
    _check_memory(SATIRE_RAM_GB, "Training the satire scorer", force)
    try:
        report = train_satire(processed_dir, artifacts_dir, reports_dir, settings)
    except ManifestMismatch as error:
        raise _frozen_or_exit(error) from error
    typer.echo(f"version {report['version']}")
    typer.echo(json.dumps(report["metrics"]["test"], indent=2))
