"""Static custody controls use real files; they never claim runtime replay."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError


@pytest.fixture(scope="module")
def modules():
    example = Path(__file__).resolve().parents[2] / "examples/cascadia"
    loaded = []
    for name in ("study_contract", "study_custody", "validate_study"):
        spec = importlib.util.spec_from_file_location(name, example / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    return tuple(loaded)


@pytest.fixture
def static_study(tmp_path, modules):
    contract, _, validator = modules
    measures = contract.RunMeasures(
        steps=4,
        reached_home_step=None,
        entered_natal_step=None,
        home_lower_bound_steps=13,
        final_posterior_home=0.0,
        max_pymdp_identity_gap=0.0,
        min_decision_margin=0.1,
        prior_light_cone_violations=[],
        posterior_support_violations=[],
        home_prior_mass_before_bound=0.0,
        mean_free_energy=1.0,
    )
    records = (
        contract.SeedOutcome(
            condition=contract.Condition.REFERENCE,
            process_seed=7,
            measures=measures,
            run_directory="reference/seed_7",
            qualified=True,
            problems=(),
        ),
    )
    condition = records[0].condition.value
    run = records[0].run_directory
    (tmp_path / run).mkdir(parents=True)
    from geo_infer_act.core.gnn_contract import GNNArtifact

    fixtures = Path(__file__).resolve().parents[2] / "examples/cascadia/data"
    artifact_path = fixtures / "smith_river_homing.geo-infer.json"
    artifact = GNNArtifact.load(artifact_path)
    state_ids = artifact.to_dict()["space"]["state_ids"]
    retained = {
        "process_seed": 7,
        "agent_seed": 0,
        "true_states": [0] * 4,
        "observations": [0] * 4,
        "trace": {
            "schema_version": "gnn-geo-infer/1",
            "artifact_sha256": artifact.digest,
            "random_seed": 0,
            "state_ids": state_ids,
            "steps": [{"observation": 0} for _ in range(4)],
        },
    }
    for filename, payload in (
        ("closed_loop.json", retained),
        ("diagnostics.json", []),
        ("summary.json", measures.model_dump(mode="json")),
        ("outcome.json", records[0].model_dump(mode="json")),
    ):
        (tmp_path / run / filename).write_text(json.dumps(payload), encoding="utf-8")
    for suffix in (
        "gnn.md",
        "state_ids.json",
        "domain.json",
        "geo-infer.json",
        "export.json",
    ):
        (tmp_path / condition / f"smith_river_homing.{suffix}").write_text(
            "fixture\n", encoding="utf-8"
        )
    (tmp_path / condition / artifact_path.name).write_bytes(artifact_path.read_bytes())
    (tmp_path / condition / "smith_river_homing.state_ids.json").write_text(
        json.dumps(state_ids)
    )
    (tmp_path / "reference_config.yaml").write_text("fixture: true\n", encoding="utf-8")
    (tmp_path / "seed_outcomes.png").write_bytes(b"fixture")
    design = contract.StudyDesign(
        seeds=(7,), conditions=(records[0].condition,), steps=4
    )
    (tmp_path / "study_design.json").write_text(
        design.model_dump_json(), encoding="utf-8"
    )
    (tmp_path / "outcomes.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in records]), encoding="utf-8"
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in contract.summarize(records)]
        ),
        encoding="utf-8",
    )
    owner = "GEO-INFER-ACT/examples/cascadia/study_contract.py"
    frozen = tmp_path / "source_owners" / owner
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b"fixture source\n")
    custody = {
        "geo_infer_revision": "a" * 40,
        "workspace_dirty": False,
        "source_owner_sha256": {owner: hashlib.sha256(frozen.read_bytes()).hexdigest()},
    }
    files = {
        str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    manifest = {
        "schema_version": "geo-infer-act/cascadia-seed-study/1",
        "complete": True,
        "qualified": True,
        "native_claim_ready": False,
        "design": design.model_dump(mode="json"),
        "versions": {"python": "fixture"},
        "files": files,
        "source_custody": custody,
        "source_drift": [],
    }
    receipt = tmp_path / "study_manifest.json"
    receipt.write_text(json.dumps(manifest), encoding="utf-8")
    assert validator.validate_study(tmp_path) == ()
    return manifest


def test_static_validator_rejects_digest_and_grid_corruption(
    tmp_path, modules, static_study
):
    _, _, validator = modules
    manifest = static_study
    run = "reference/seed_7"
    receipt = tmp_path / "study_manifest.json"
    target = tmp_path / run / "closed_loop.json"
    target.write_text(target.read_text() + "\n", encoding="utf-8")
    assert any(
        "digest mismatch" in problem for problem in validator.validate_study(tmp_path)
    )
    manifest["files"][str(target.relative_to(tmp_path))] = hashlib.sha256(
        target.read_bytes()
    ).hexdigest()
    manifest["design"]["seeds"] = [6]
    receipt.write_text(json.dumps(manifest), encoding="utf-8")
    assert any(
        "outcome grid" in problem for problem in validator.validate_study(tmp_path)
    )
    manifest["files"]["../outside"] = "0" * 64
    receipt.write_text(json.dumps(manifest), encoding="utf-8")
    assert any(
        "unsafe manifest path" in problem
        for problem in validator.validate_study(tmp_path)
    )


@pytest.mark.parametrize(
    ("filename", "field", "value", "message"),
    [
        ("summary.json", "mean_free_energy", 2.0, "run summary differs"),
        ("outcome.json", "qualified", False, "run outcome differs"),
        ("closed_loop.json", "process_seed", 8, "process seed differs"),
        ("closed_loop.json", "agent_seed", 1, "agent seed differs"),
        ("closed_loop.json", "true_states", [0], "closed-loop horizon differs"),
        ("closed_loop.json", "observations", [1] * 4, "trace observations differ"),
    ],
)
def test_rehashed_run_contradictions_fail(
    tmp_path, modules, static_study, filename, field, value, message
):
    _, _, validator = modules
    path = tmp_path / "reference/seed_7" / filename
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    static_study["files"][str(path.relative_to(tmp_path))] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    (tmp_path / "study_manifest.json").write_text(json.dumps(static_study))
    assert any(message in problem for problem in validator.validate_study(tmp_path))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_sha256", "0" * 64, "trace artifact digest differs"),
        ("state_ids", ["wrong"], "trace state order differs"),
        ("random_seed", 1, "agent seed differs"),
        ("steps", [], "closed-loop horizon differs"),
    ],
)
def test_rehashed_trace_contradictions_fail(
    tmp_path, modules, static_study, field, value, message
):
    _, _, validator = modules
    path = tmp_path / "reference/seed_7/closed_loop.json"
    payload = json.loads(path.read_text())
    payload["trace"][field] = value
    path.write_text(json.dumps(payload))
    static_study["files"][str(path.relative_to(tmp_path))] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    (tmp_path / "study_manifest.json").write_text(json.dumps(static_study))
    assert any(message in problem for problem in validator.validate_study(tmp_path))


def test_source_snapshot_detects_real_owner_drift(tmp_path, modules):
    _, custody, _ = modules
    root = tmp_path / "repo"
    owner = root / "GEO-INFER-ACT/examples/cascadia/experiment.py"
    owner.parent.mkdir(parents=True)
    owner.write_bytes(b"value = 1\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "config", "core.fsmonitor", "false"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    snapshot = custody.freeze_sources(root, tmp_path / "edition")
    assert snapshot.workspace_dirty is False
    assert custody.source_problems(root, snapshot) == ()
    owner.write_bytes(b"value = 2\n")
    assert custody.source_problems(root, snapshot) == (
        "source owner differs: GEO-INFER-ACT/examples/cascadia/experiment.py",
    )
    assert (
        tmp_path / "edition/source_owners/GEO-INFER-ACT/examples/cascadia/experiment.py"
    ).read_bytes() == b"value = 1\n"
    with pytest.raises(ValidationError):
        custody.SourceCustody(
            geo_infer_revision="a" * 40,
            workspace_dirty=False,
            source_owner_sha256={"../outside.py": "0" * 64},
        )
