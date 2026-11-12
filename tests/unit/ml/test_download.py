import hashlib
import json
from pathlib import Path

import httpx
import pytest
import respx
from click.testing import Result
from typer.testing import CliRunner

from fnd_ml.cli import app
from fnd_ml.data.download import (
    ChecksumMismatch,
    DownloadError,
    fetch_dataset,
    load_registry,
)

REGISTRY_YAML = """
datasets:
  toy:
    license: CC0
    homepage: https://example.org/toy
    files:
      - url: https://example.org/toy/a.csv
        filename: a.csv
      - url: https://example.org/toy/b.zip?download=1
        filename: b.zip
"""


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    path = tmp_path / "registry" / "datasets.yaml"
    path.parent.mkdir()
    path.write_text(REGISTRY_YAML, encoding="utf-8")
    return path


def _mock_toy(
    a: bytes = b"id,text\n1,hello\n", b: bytes = b"PK-zip-bytes"
) -> dict[str, respx.Route]:
    return {
        "a": respx.get("https://example.org/toy/a.csv").mock(
            return_value=httpx.Response(200, content=a)
        ),
        "b": respx.get("https://example.org/toy/b.zip?download=1").mock(
            return_value=httpx.Response(200, content=b)
        ),
    }


def test_registry_lists_datasets_with_their_files(registry: Path) -> None:
    entries = load_registry(registry)
    assert list(entries) == ["toy"]
    assert entries["toy"].license == "CC0"
    assert [f.filename for f in entries["toy"].files] == ["a.csv", "b.zip"]


@respx.mock
def test_fetch_downloads_files_and_records_their_checksums(registry: Path, tmp_path: Path) -> None:
    _mock_toy()
    raw_dir = tmp_path / "raw"

    paths = fetch_dataset("toy", raw_dir, registry)

    assert paths == [raw_dir / "toy" / "a.csv", raw_dir / "toy" / "b.zip"]
    assert paths[0].read_bytes() == b"id,text\n1,hello\n"
    recorded = json.loads((registry.parent / "checksums.json").read_text(encoding="utf-8"))
    assert recorded["toy"]["a.csv"] == hashlib.sha256(b"id,text\n1,hello\n").hexdigest()
    assert recorded["toy"]["b.zip"] == hashlib.sha256(b"PK-zip-bytes").hexdigest()


@respx.mock
def test_fetch_does_not_download_again_when_the_file_matches(
    registry: Path, tmp_path: Path
) -> None:
    routes = _mock_toy()
    raw_dir = tmp_path / "raw"

    fetch_dataset("toy", raw_dir, registry)
    fetch_dataset("toy", raw_dir, registry)

    assert routes["a"].call_count == 1
    assert routes["b"].call_count == 1


@respx.mock
def test_fetch_rejects_a_file_whose_content_changed_upstream(
    registry: Path, tmp_path: Path
) -> None:
    _mock_toy()
    raw_dir = tmp_path / "raw"
    fetch_dataset("toy", raw_dir, registry)
    (raw_dir / "toy" / "a.csv").unlink()
    respx.get("https://example.org/toy/a.csv").mock(
        return_value=httpx.Response(200, content=b"id,text\n1,TAMPERED\n")
    )

    with pytest.raises(ChecksumMismatch, match=r"a\.csv"):
        fetch_dataset("toy", raw_dir, registry)

    assert not (raw_dir / "toy" / "a.csv").exists()


@respx.mock
def test_fetch_reports_http_failures(registry: Path, tmp_path: Path) -> None:
    respx.get("https://example.org/toy/a.csv").mock(return_value=httpx.Response(404))

    with pytest.raises(DownloadError, match="404"):
        fetch_dataset("toy", tmp_path / "raw", registry)

    assert not (tmp_path / "raw" / "toy" / "a.csv").exists()


def test_fetch_of_an_unknown_dataset_names_the_known_ones(registry: Path, tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="toy"):
        fetch_dataset("nope", tmp_path / "raw", registry)


def test_a_manually_placed_file_is_accepted_and_its_checksum_recorded(
    registry: Path, tmp_path: Path
) -> None:
    target = tmp_path / "raw" / "toy"
    target.mkdir(parents=True)
    (target / "a.csv").write_bytes(b"manual a")
    (target / "b.zip").write_bytes(b"manual b")

    paths = fetch_dataset(
        "toy", tmp_path / "raw", registry
    )  # no HTTP mock: must not hit the network

    assert [p.name for p in paths] == ["a.csv", "b.zip"]
    recorded = json.loads((registry.parent / "checksums.json").read_text(encoding="utf-8"))
    assert recorded["toy"]["a.csv"] == hashlib.sha256(b"manual a").hexdigest()


# ---------------------------------------------------------------- CLI


def _invoke_download(registry: Path, raw_dir: Path) -> Result:
    args = ["data", "download", "--registry", str(registry), "--raw-dir", str(raw_dir)]
    return CliRunner().invoke(app, args)


@respx.mock
def test_cli_download_fetches_every_registered_dataset(registry: Path, tmp_path: Path) -> None:
    _mock_toy()

    result = _invoke_download(registry, tmp_path / "raw")

    assert result.exit_code == 0, result.output
    assert "toy" in result.output
    assert (tmp_path / "raw" / "toy" / "b.zip").exists()


@respx.mock
def test_cli_download_exits_nonzero_and_names_the_failed_dataset(
    registry: Path, tmp_path: Path
) -> None:
    respx.get("https://example.org/toy/a.csv").mock(return_value=httpx.Response(500))

    result = _invoke_download(registry, tmp_path / "raw")

    assert result.exit_code == 1
    assert "toy" in result.output
    assert "500" in result.output
