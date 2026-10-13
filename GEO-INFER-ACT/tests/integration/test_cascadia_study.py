"""Real seed replay and non-vacuous controls for the Cascadia study contract."""

from __future__ import annotations

from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

from geo_infer_act.core.gnn_contract import GNNArtifact
from geo_infer_act.models.river_homing import compile_river_homing
from geo_infer_act.runners.closed_loop import run_closed_loop, trace_diagnostics

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/cascadia"


@pytest.fixture(scope="module")
def modules():
    loaded = []
    for name in ("cascadia_homing", "study_contract", "study_figure"):
        spec = importlib.util.spec_from_file_location(name, EXAMPLE / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    return tuple(loaded)


def test_design_rejects_bad_grids_before_creating_output(tmp_path, modules):
    _, contract, _ = modules
    for fields in (
        {"seeds": ()},
        {"seeds": (1, 1)},
        {"seeds": (-1,)},
        {"seeds": (True,)},
        {"seeds": (2**32,)},
        {"steps": 0},
        {"steps": True},
        {"conditions": ()},
        {"seeds": tuple(range(65))},
        {"seeds": tuple(range(64)), "steps": 512},
    ):
        with pytest.raises(ValidationError):
            contract.StudyDesign(**fields)
    output = tmp_path / "invalid"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "cascadia_study.py"),
            "--output",
            str(output),
            "--gnn-repo",
            str(tmp_path),
            "--gnn-python",
            sys.executable,
            "--seeds",
            "7",
            "7",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "seeds must be unique" in completed.stderr
    assert not output.exists()


@pytest.fixture(scope="module")
def outcomes(modules):
    homing, contract, _ = modules
    config = homing.load_config()
    reference, _, _ = homing.build_model(config)
    artifact = GNNArtifact.load(
        config["data"]["fixtures"] / homing.FIXTURE_NAMES["artifact"]
    )
    outcomes = []
    for condition in contract.Condition:
        model = compile_river_homing(
            reference.network,
            contract.condition_parameters(reference.parameters, condition),
        )
        data = artifact.to_dict()
        data["matrices"] = model.gnn_spec(
            model_name="test", annotation="test"
        ).float_matrices()
        test_artifact = GNNArtifact.from_dict(data)
        result = run_closed_loop(
            test_artifact, initial_state=0, steps=4, process_seed=7
        )
        measures = contract.RunMeasures.model_validate(
            homing.analyze(model, result, trace_diagnostics(test_artifact, result))
        )
        problems = measures.qualification_problems()
        outcomes.append(
            contract.SeedOutcome(
                condition=condition,
                process_seed=7,
                measures=measures,
                run_directory=f"{condition}/seed_7",
                qualified=not problems,
                problems=problems,
            )
        )
    return tuple(outcomes)


def test_condition_changes_are_exact_and_isolated(modules):
    homing, contract, _ = modules
    reference, _, _ = homing.build_model(homing.load_config())
    assert (
        contract.condition_parameters(
            reference.parameters, contract.Condition.REFERENCE
        )
        is reference.parameters
    )
    uniform = contract.condition_parameters(
        reference.parameters, contract.Condition.UNIFORM_HABIT
    )
    assert uniform.habit == (Fraction(1, 4),) * 4
    assert uniform.odor_likelihood == reference.parameters.odor_likelihood
    odor = contract.condition_parameters(
        reference.parameters, contract.Condition.UNINFORMATIVE_ODOR
    )
    assert len(set(odor.odor_likelihood)) == 1
    assert odor.habit == reference.parameters.habit
    for condition in contract.Condition:
        model = compile_river_homing(
            reference.network,
            contract.condition_parameters(reference.parameters, condition),
        )
        assert all(
            sum(row[state] for row in model.A) == 1 for state in range(len(model.D))
        )
        assert sum(model.E) == 1


def test_censoring_preserves_every_seed_and_last_step_arrivals(modules, outcomes):
    _, contract, _ = modules
    summaries = contract.summarize(outcomes)
    assert all(
        item.home_censored == 1 and item.home_arrivals == 0 for item in summaries
    )
    assert all(item.restricted_mean_first_home_steps == 3 for item in summaries)
    arrived = outcomes[0].measures.model_copy(
        update={"steps": 48, "reached_home_step": 47}
    )
    records = (outcomes[0].model_copy(update={"measures": arrived}),)
    assert contract.summarize(records)[0].home_arrivals == 1
    assert arrived.qualification_problems() == ()


def test_numerical_gate_rejects_real_corruptions(outcomes):
    measures = outcomes[0].measures
    assert measures.qualification_problems() == ()
    for update in (
        {"prior_light_cone_violations": [1]},
        {"posterior_support_violations": [1]},
        {"home_prior_mass_before_bound": 0.01},
        {"max_pymdp_identity_gap": 1e-5},
        {"reached_home_step": 1},
        {"reached_home_step": 4},
    ):
        corrupted = measures.model_copy(update=update)
        assert corrupted.qualification_problems()
    payload = measures.model_dump()
    payload["max_pymdp_identity_gap"] = float("nan")
    with pytest.raises(ValidationError):
        type(measures).model_validate(payload)


def test_figure_and_json_preserve_censored_outcomes(tmp_path, modules, outcomes):
    _, _, figure = modules
    path = figure.save_study_figure(tmp_path / "seeds.png", outcomes)
    assert path.stat().st_size > 1000
    payload = json.loads(outcomes[0].model_dump_json())
    assert payload["measures"]["reached_home_step"] is None
    assert payload["qualified"] is True
