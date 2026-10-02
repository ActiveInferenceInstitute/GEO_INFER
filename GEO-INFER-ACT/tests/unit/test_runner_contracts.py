"""Contract tests for Active Inference scenario runners and script outputs."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest

from geo_infer_act.runners import (
    SCENARIO_NAMES,
    RunConfig,
    load_run_config,
    run_all_scenarios,
    run_scenario,
)


@pytest.mark.slow
@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_each_scenario_writes_manifest_schema_data_and_visualization(
    scenario: str, tmp_path: Path
) -> None:
    config = RunConfig(
        scenario=scenario,
        output_dir=tmp_path / scenario,
        seed=17,
        deterministic=True,
        timesteps=3,
        visualizations=True,
        h3_resolution=8,
        h3_ring_size=1,
    )

    result = run_scenario(config)

    assert result.output_dir == config.output_dir
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["schema_version"] == "geo-infer-act-run-manifest/v1"
    assert manifest["scenario"] == scenario
    assert manifest["validation"]["status"] == "passed"

    generated = {item["path"] for item in manifest["generated_files"]}
    assert "data/full_history.json" in generated
    assert "data/step_metrics.csv" in generated
    assert any(path.startswith("visualizations/") for path in generated)

    for item in manifest["generated_files"]:
        path = result.output_dir / item["path"]
        assert path.exists(), item["path"]
        assert path.stat().st_size > 0, item["path"]
        assert item["artifact_type"]
        assert item["mime_type"]
        assert len(item["sha256"]) == 64
        if item["artifact_type"] == "visualization":
            metadata_path = result.output_dir / item["figure_metadata_path"]
            data_path = result.output_dir / item["figure_data_path"]
            assert metadata_path.exists() and metadata_path.stat().st_size > 0
            assert data_path.exists() and data_path.stat().st_size > 0
            assert item["description"]
            assert item["alt_text"]
            assert item["data_sources"]
            assert item["plotted_metrics"]


def test_load_run_config_applies_yaml_and_overrides(tmp_path: Path) -> None:
    config_file = tmp_path / "run.yaml"
    config_file.write_text(
        "\n".join(
            [
                "schema_version: geo-infer-act-run-config/v1",
                "scenario: simple",
                "seed: 5",
                "timesteps: 8",
                "visualizations: false",
            ]
        )
    )

    config = load_run_config(
        config_file,
        overrides={"timesteps": 2, "output_dir": tmp_path / "out"},
    )

    assert config.scenario == "simple"
    assert config.seed == 5
    assert config.timesteps == 2
    assert config.visualizations is False
    assert config.output_dir == tmp_path / "out"


def test_seeded_scenario_outputs_are_deterministic(tmp_path: Path) -> None:
    first = run_scenario(
        RunConfig(
            scenario="simple",
            output_dir=tmp_path / "first",
            seed=101,
            deterministic=True,
            timesteps=4,
            visualizations=True,
        )
    )
    second = run_scenario(
        RunConfig(
            scenario="simple",
            output_dir=tmp_path / "second",
            seed=101,
            deterministic=True,
            timesteps=4,
            visualizations=True,
        )
    )

    first_steps = json.loads(
        (first.output_dir / "data" / "full_history.json").read_text()
    )
    second_steps = json.loads(
        (second.output_dir / "data" / "full_history.json").read_text()
    )

    assert [step["action"] for step in first_steps] == [
        step["action"] for step in second_steps
    ]
    assert [step["free_energy"] for step in first_steps] == [
        step["free_energy"] for step in second_steps
    ]


def test_run_all_scenarios_writes_suite_manifest(tmp_path: Path) -> None:
    results = run_all_scenarios(
        output_dir=tmp_path / "suite",
        scenarios=["simple", "h3"],
        seed=29,
        timesteps=2,
        deterministic=True,
        visualizations=True,
    )

    assert len(results.scenario_results) == 2
    assert results.manifest_path.exists()
    manifest = json.loads(results.manifest_path.read_text())
    assert manifest["schema_version"] == "geo-infer-act-suite-manifest/v1"
    assert manifest["validation"]["status"] == "passed"
    assert {item["scenario"] for item in manifest["scenarios"]} == {"simple", "h3"}
    moved = tmp_path / "relocated"
    shutil.copytree(results.output_dir, moved)
    for scenario in manifest["scenarios"]:
        assert not Path(scenario["output_dir"]).is_absolute()
        assert not Path(scenario["manifest"]).is_absolute()
        child = json.loads((moved / scenario["manifest"]).read_text())
        assert child["validation"]["output_dir"] == "."
        for artifact in child["generated_files"]:
            copied = moved / scenario["output_dir"] / artifact["path"]
            assert hashlib.sha256(copied.read_bytes()).hexdigest() == artifact["sha256"]


def test_package_version_fallback_uses_public_runtime_version(monkeypatch):
    from importlib import metadata
    from geo_infer_act import __version__
    from geo_infer_act.runners import io

    def absent_distribution(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(io.metadata, "version", absent_distribution)
    assert io.package_version() == __version__ == "0.4.0"
