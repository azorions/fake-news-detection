import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fnd_ml.cli import app
from fnd_ml.common import resources
from tests.fixtures.raw_datasets import make_raw_datasets

runner = CliRunner()


def test_build_stops_with_a_clear_message_when_memory_is_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "free_ram_gb", lambda: 0.4)
    raw = make_raw_datasets(tmp_path)
    out = tmp_path / "out"

    result = runner.invoke(app, ["data", "build", "--raw-dir", str(raw), "--out-dir", str(out)])

    assert result.exit_code == 2
    assert "needs about 3.0 GB" in result.output
    assert "0.4 GB" in result.output
    assert not out.exists()


def test_build_with_force_runs_and_prints_the_three_hygiene_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "free_ram_gb", lambda: 0.4)
    raw = make_raw_datasets(tmp_path)
    out = tmp_path / "out"

    result = runner.invoke(
        app, ["data", "build", "--raw-dir", str(raw), "--out-dir", str(out), "--force"]
    )

    assert result.exit_code == 0, result.output
    assert (
        "shared groups / hashes / near-duplicate clusters between splits: 0 / 0 / 0"
        in result.output
    )
    assert "evaluation sets not found" in result.output  # the tiny raw folder has none


def test_freeze_records_the_manifest_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(resources, "free_ram_gb", lambda: 64.0)
    raw = make_raw_datasets(tmp_path)
    out = tmp_path / "out"
    lock = tmp_path / "data.lock"
    built = runner.invoke(app, ["data", "build", "--raw-dir", str(raw), "--out-dir", str(out)])
    assert built.exit_code == 0, built.output

    result = runner.invoke(
        app, ["data", "freeze", "--processed-dir", str(out), "--lock", str(lock)]
    )

    assert result.exit_code == 0, result.output
    digest = json.loads(lock.read_text(encoding="utf-8"))["manifest_sha256"]
    assert len(digest) == 64 and digest in result.output
