"""``fnd`` command line: data and model commands.

Run from the repository root, for example ``uv run fnd data download``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from fnd_ml.data.download import ChecksumMismatch, DownloadError, fetch_dataset, load_registry

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Download and prepare datasets.")
app.add_typer(data_app, name="data")

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
