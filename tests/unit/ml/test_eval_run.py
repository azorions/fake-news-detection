import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from fnd_ml.cli import app
from fnd_ml.common import resources
from fnd_ml.common.artifacts import register
from fnd_ml.data.build import MANIFEST_NAME, ManifestMismatch, write_lock
from fnd_ml.eval import run as eval_run
from fnd_ml.eval.gates import GateResult
from fnd_ml.eval.protocols import RetestError, TouchLog
from fnd_ml.eval.run import run_eval
from fnd_ml.models.baseline import BaselineConfig
from fnd_ml.models.train_baseline import train_baseline
from tests.fixtures.processed import two_dataset_corpus, write_processed
from tests.fixtures.synthetic import make_corpus

ROOT = Path(__file__).parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "tiny_transformer"
GATES = ROOT / "ml" / "configs" / "eval_gates.yaml"


@dataclass(frozen=True)
class World:
    processed: Path
    lock: Path
    artifacts: Path
    reports: Path
    cache: Path
    baseline_version: str

    @property
    def out(self) -> Path:
        return self.reports / "eval_v1"

    def run(self, **options: Any) -> tuple[dict[str, Any], dict[str, GateResult]]:
        results, gates = run_eval(
            self.out,
            self.processed,
            self.artifacts,
            gates_path=GATES,
            lock_path=self.lock,
            cache_dir=self.cache,
            **options,
        )
        return results, {gate.name: gate for gate in gates}

    def touches(self, model: str, version: str) -> list[dict[str, Any]]:
        return TouchLog(self.reports / "test_touch_log.json").entries(model, version)


@pytest.fixture
def world(tmp_path: Path) -> World:
    """Frozen toy data with two external sets, a trained baseline and the tiny ONNX transformer."""
    processed, lock = write_processed(tmp_path, two_dataset_corpus(), freeze=False)
    manifest = json.loads((processed / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["shared_groups_between_splits"] = 0
    manifest["shared_near_duplicate_clusters_between_splits"] = 0
    (processed / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    write_lock(processed, lock)
    (processed / "eval").mkdir()
    for name, seed in (("constraint", 5), ("buzzfeed_webis", 6)):
        frame = make_corpus(80, seed=seed).assign(dataset=name)
        frame.to_parquet(processed / "eval" / f"{name}.parquet", index=False)

    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"
    identifiers = tmp_path / "identifiers.txt"
    identifiers.write_text("reuters\n", encoding="utf-8")
    report = train_baseline(
        processed,
        artifacts,
        reports,
        BaselineConfig(n_features=2**16),
        lock_path=lock,
        identifiers_path=identifiers,
    )
    shutil.copytree(FIXTURE, artifacts / "transformer" / "v1-tinyfix")
    register(artifacts, "transformer", "v1-tinyfix", "transformer/v1-tinyfix")
    return World(processed, lock, artifacts, reports, tmp_path / "cache", report["version"])


def test_eval_compares_the_registered_models_on_every_protocol(world: World) -> None:
    results, gates = world.run()

    versions = {name: model["version"] for name, model in results["models"].items()}
    assert versions == {"baseline": world.baseline_version, "transformer": "v1-tinyfix"}
    for model in ("baseline", "transformer"):
        assert results["in_distribution"][model]["overall"]["n"] == 50
        assert set(results["in_distribution"][model]["by_dataset"]) == {"alpha", "beta"}
        assert "overall_raw" in results["in_distribution"][model]
        assert results["external"]["sets"]["constraint"][model]["n"] == 80
        assert results["external"]["sets"]["buzzfeed_webis"][model]["n"] == 80
    assert set(results["leave_one_dataset_out"]["baseline"]) == {"alpha", "beta"}
    assert "transformer" not in results["leave_one_dataset_out"]
    assert gates["split_hygiene"].passed and gates["leak_probe"].passed
    assert gates["baseline_floor"].passed
    assert not gates["onnx_parity"].passed  # the fixture carries no export measurements


def test_eval_writes_the_report_the_figures_and_the_touch_log(world: World) -> None:
    results, gates = world.run()

    written = json.loads((world.reports / "eval_v1.json").read_text(encoding="utf-8"))
    markdown = (world.reports / "eval_v1.md").read_text(encoding="utf-8")
    assert written["in_distribution"] == results["in_distribution"]
    assert {gate["name"] for gate in written["gates"]} == set(gates)
    assert world.baseline_version in markdown and "v1-tinyfix" in markdown
    for model in ("baseline", "transformer"):
        for scored_set in ("test", "constraint", "buzzfeed_webis"):
            figure = f"figures/reliability_{model}_{scored_set}.png"
            assert (world.reports / figure).stat().st_size > 5000
            assert f"({figure})" in markdown
    assert len(world.touches("baseline", world.baseline_version)) == 1
    assert len(world.touches("transformer", "v1-tinyfix")) == 1
    assert all(touch["scored"] for touch in results["test_touches"])


def test_a_second_run_reads_its_cached_scores_and_leaves_the_test_split_alone(
    world: World,
) -> None:
    first, _ = world.run()

    second, _ = world.run()

    assert second["in_distribution"] == first["in_distribution"]
    assert second["leave_one_dataset_out"] == first["leave_one_dataset_out"]
    assert [touch["scored"] for touch in second["test_touches"]] == [False, False]
    assert len(world.touches("baseline", world.baseline_version)) == 1


def test_scoring_the_test_split_again_needs_a_reason(world: World) -> None:
    world.run()
    shutil.rmtree(world.cache)

    with pytest.raises(RetestError, match=world.baseline_version):
        world.run()
    assert not list(world.cache.glob("*test*"))  # refused before anything was scored

    world.run(allow_retest="the score cache was deleted")

    reasons = [touch["reason"] for touch in world.touches("transformer", "v1-tinyfix")]
    assert reasons == [None, "the score cache was deleted"]
    markdown = (world.reports / "eval_v1.md").read_text(encoding="utf-8")
    assert "the score cache was deleted" in markdown


def test_eval_refuses_data_that_is_not_the_frozen_version(world: World) -> None:
    manifest = world.processed / MANIFEST_NAME
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ManifestMismatch):
        world.run()

    assert not (world.reports / "eval_v1.md").exists()


def test_every_set_is_scored_before_the_folds_are_trained(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The artifacts are done, and can be released, before the long part of the run begins."""
    cached_at_each_fold: list[int] = []
    real = eval_run.leave_one_dataset_out

    def spy(*args: Any, **options: Any) -> Any:
        cached_at_each_fold.append(len(list(world.cache.glob("*.npz"))))
        return real(*args, **options)

    monkeypatch.setattr(eval_run, "leave_one_dataset_out", spy)

    world.run()

    assert cached_at_each_fold == [6, 6]  # two models on three sets, before each of two folds


def test_without_a_transformer_the_classic_models_are_still_compared(world: World) -> None:
    shutil.rmtree(world.artifacts / "transformer")

    results, gates = world.run()

    assert set(results["in_distribution"]) == {"baseline"}
    assert results["transformer"] is None
    assert gates["transformer_gain"].measured == "not measured"
    assert (world.reports / "eval_v1.md").exists()


@pytest.mark.slow
def test_the_transformer_joins_leave_one_dataset_out_when_asked(
    world: World, tiny_model: Path, tmp_path: Path
) -> None:
    config = tmp_path / "lodo.yaml"
    config.write_text(
        f"model_name: {tiny_model.as_posix()}\nmax_length: 32\nepochs: 1\nbatch_size: 8\n"
        "eval_every_steps: 5\ncheckpoint_every_steps: 5\nmax_train_rows: 40\namp: off\n",
        encoding="utf-8",
    )

    results, _ = world.run(with_transformer_lodo=True, lodo_config=config)

    folds = results["leave_one_dataset_out"]["transformer"]
    assert set(folds) == {"alpha", "beta"}
    assert folds["alpha"]["train_rows"] == 40
    assert folds["alpha"]["n"] == 24  # the alpha rows of the 50-row test split


def _fake_run(gates: list[GateResult], calls: list[dict[str, Any]]) -> Any:
    def run(out: Path, processed_dir: Path, artifacts_dir: Path, **options: Any) -> Any:
        calls.append({"out": out, **options})
        return {"name": out.name}, gates

    return run


def test_the_eval_command_prints_the_gates_and_fails_when_one_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "free_ram_gb", lambda: 64.0)
    gates = [
        GateResult("baseline_floor", True, 0.8535, "macro-F1 >= 0.8", ""),
        GateResult("latency_p95", False, 612.0, "p95 <= 500 ms", "fp32, 4 threads"),
    ]
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(eval_run, "run_eval", _fake_run(gates, calls))

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--out",
            str(tmp_path / "eval_v1"),
            "--with-transformer-lodo",
            "--allow-retest",
            "a reason",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "baseline_floor" in result.output and "0.8535" in result.output
    assert "latency_p95" in result.output and "FAIL" in result.output and "612" in result.output
    assert calls[0]["out"] == tmp_path / "eval_v1"
    assert calls[0]["with_transformer_lodo"] is True
    assert calls[0]["allow_retest"] == "a reason"


def test_the_eval_command_succeeds_when_every_gate_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "free_ram_gb", lambda: 64.0)
    gates = [GateResult("baseline_floor", True, 0.8535, "macro-F1 >= 0.8", "")]
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(eval_run, "run_eval", _fake_run(gates, calls))

    result = CliRunner().invoke(app, ["eval", "--out", str(tmp_path / "eval_v1")])

    assert result.exit_code == 0, result.output
    assert calls[0]["with_transformer_lodo"] is False
    assert calls[0]["allow_retest"] is None


def test_the_eval_command_explains_a_refused_retest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(resources, "free_ram_gb", lambda: 64.0)

    def refuse(*args: Any, **options: Any) -> Any:
        raise RetestError("baseline v2-aaaa has already scored the test split")

    monkeypatch.setattr(eval_run, "run_eval", refuse)

    result = CliRunner().invoke(app, ["eval", "--out", str(tmp_path / "eval_v1")])

    assert result.exit_code == 2
    assert "already scored the test split" in result.output
