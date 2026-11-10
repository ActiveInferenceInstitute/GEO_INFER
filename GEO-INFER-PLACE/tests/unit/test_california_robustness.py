"""Real inference regressions for a separate synthetic observation generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geo_infer_act.core.gnn_factored_contract import FactoredGNNArtifact
from geo_infer_place.locations.california.contracts import (
    CaliforniaError,
    Hazard,
    ObservatoryConfig,
    Strategy,
    digest,
)
from geo_infer_place.locations.california.geometry import Landscape
from geo_infer_place.locations.california.inference import COSTS, infer_field
from geo_infer_place.locations.california.monitoring import (
    qualify_models,
    run_scenario,
    simulated_report,
    validate_generator_model,
)
from geo_infer_place.locations.california.robustness import (
    generator_conditions,
    run_tiny_robustness,
    synthetic_generator_models,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "research/measurement-quality/fixture.json"
MODELS = ROOT / "GEO-INFER-PLACE/locations/california/models"


def fixture_models() -> tuple[Landscape, tuple[FactoredGNNArtifact, ...]]:
    landscape = Landscape.model_validate_json(FIXTURE.read_bytes())
    return landscape, qualify_models(landscape.config, MODELS)[0]


@pytest.mark.parametrize("strategy", tuple(Strategy))
def test_matched_generator_preserves_pinned_baseline(strategy: Strategy) -> None:
    landscape, planners = fixture_models()
    baseline = json.loads(FIXTURE.with_name("baseline-scenarios.json").read_bytes())
    for seed in landscape.config.seeds:
        default = run_scenario(landscape, planners, strategy, seed)
        matched = run_scenario(
            landscape, planners, strategy, seed, generator_artifacts=planners
        )
        assert default == matched
        expected = next(
            item
            for item in baseline
            if item["strategy"] == strategy and item["seed"] == seed
        )
        assert default.model_dump(mode="json") == expected


def test_generator_changes_readings_but_returns_planner_likelihood() -> None:
    landscape, planners = fixture_models()
    planner = planners[0]
    args = (1, 2, 0, Hazard.FIRE, landscape.cells[0].id, 0)
    original, _ = simulated_report(planner, *args)
    forced = 2 if original == 1 else 1
    data = planner.to_dict()
    likelihood = np.asarray(data["modalities"][0]["likelihood"])
    likelihood[:, :, 1:] = 0
    likelihood[forced, :, 1:] = 1
    data["modalities"][0]["likelihood"] = likelihood.tolist()
    generator = FactoredGNNArtifact.from_dict(data)
    outcome, assumed = simulated_report(planner, *args, generator_artifact=generator)
    repeated, repeated_assumed = simulated_report(
        planner, *args, generator_artifact=generator
    )
    assert outcome == repeated == forced != original
    np.testing.assert_array_equal(
        assumed,
        np.asarray(planner.to_dict()["modalities"][0]["likelihood"])[outcome, :, 2],
    )
    np.testing.assert_array_equal(assumed, repeated_assumed)


@pytest.mark.parametrize("strategy", tuple(Strategy))
def test_first_allocation_independent_of_generator_and_hidden_labels(
    strategy: Strategy,
) -> None:
    landscape, planners = fixture_models()
    raw = landscape.model_dump(mode="json")
    for indicator in raw["indicators"]:
        flipped = [1 - value for value in indicator["positive"][-1]]
        indicator["positive"][-1] = flipped
        indicator["values"][-1] = [float(value) for value in flipped]
    flipped_landscape = Landscape.model_validate_json(json.dumps(raw))
    matched = run_scenario(landscape, planners, strategy, 0).frames[1]
    for condition in ("overoptimistic", "pessimistic"):
        generators = synthetic_generator_models(
            generator_conditions(landscape.config)[condition], planners
        )
        for source in (landscape, flipped_landscape):
            first = run_scenario(
                source, planners, strategy, 0, generator_artifacts=generators
            ).frames[1]
            assert first.modes == matched.modes
            assert first.expected_free_energy == matched.expected_free_energy
            assert first.information_gain == matched.information_gain
            assert first.expected_preference_cost == matched.expected_preference_cost


def test_mismatch_assimilation_uses_planner_likelihood() -> None:
    landscape, planners = fixture_models()
    generators = synthetic_generator_models(
        generator_conditions(landscape.config)["overoptimistic"], planners
    )
    scenario = run_scenario(
        landscape, planners, Strategy.ACTIVE, 0, generator_artifacts=generators
    )
    count = len(landscape.cells)
    cumulative = [np.zeros((count, 2)) for _ in Hazard]
    for frame in scenario.frames[1:]:
        for site, (mode, observation) in enumerate(
            zip(frame.modes, frame.observations, strict=True)
        ):
            hazard, cell = divmod(site, count)
            if mode:
                assumed = np.asarray(
                    planners[hazard].to_dict()["modalities"][0]["likelihood"]
                )
                cumulative[hazard][cell] += np.log(assumed[observation, :, mode])
        for hazard, indicator in enumerate(landscape.indicators):
            expected = infer_field(
                np.asarray(indicator.prior),
                cumulative[hazard],
                landscape.edges,
                landscape.config.coupling,
            )
            np.testing.assert_array_equal(
                frame.fields[hazard].posterior, expected.posterior
            )


@pytest.mark.parametrize(
    "mutation",
    (
        "preferences",
        "resources",
        "transitions",
        "policies",
        "time",
        "name",
        "passive",
        "axes",
    ),
)
def test_generator_refuses_other_model_changes(mutation: str) -> None:
    _, planners = fixture_models()
    data = planners[0].to_dict()
    if mutation == "preferences":
        data["modalities"][1]["preferences"][1] = -0.5
    elif mutation == "resources":
        data["modalities"][1]["likelihood"].reverse()
    elif mutation == "transitions":
        data["transitions"][0]["probabilities"].reverse()
    elif mutation == "policies":
        data["policy_prior"] = [0.5, 0.25, 0.25]
    elif mutation == "time":
        data["time"]["step_seconds"] = 1
    elif mutation == "name":
        data["model_name"] = "different hazard"
    elif mutation == "passive":
        likelihood = np.asarray(data["modalities"][0]["likelihood"])
        likelihood[:, :, 0] = np.array([[0.5], [0.5], [0]])
        data["modalities"][0]["likelihood"] = likelihood.tolist()
    else:
        data["modalities"][0]["outcomes"] = ["none", "positive", "negative"]
    generator = FactoredGNNArtifact.from_dict(data)
    with pytest.raises(CaliforniaError):
        validate_generator_model(planners[0], generator)


def test_generator_roster_and_hazard_order_are_validated() -> None:
    landscape, planners = fixture_models()
    for generators in (planners[:2], tuple(reversed(planners))):
        with pytest.raises(CaliforniaError):
            run_scenario(
                landscape, planners, Strategy.ACTIVE, 0, generator_artifacts=generators
            )


def test_generator_source_cannot_silently_declare_different_costs() -> None:
    landscape, planners = fixture_models()
    config = ObservatoryConfig.model_validate(
        {**landscape.config.model_dump(), "sensor_cost": 0.5}
    )
    with pytest.raises(CaliforniaError):
        synthetic_generator_models(config, planners)


def test_harness_refuses_alternative_years(tmp_path: Path) -> None:
    landscape, _ = fixture_models()
    raw = landscape.model_dump(mode="json")
    raw["config"].update(first_year=2017, target_year=2019)
    for indicator in raw["indicators"]:
        indicator["years"] = [2017, 2018, 2019]
        indicator["dates"] = ["2017", "2018", "2019"]
    alternate = tmp_path / "alternate.json"
    alternate.write_text(json.dumps(raw))
    output = tmp_path / "refused"
    with pytest.raises(CaliforniaError, match="tiny synthetic design"):
        run_tiny_robustness(alternate, MODELS, output)
    assert not output.exists()


def test_tiny_harness_retains_assumptions_scores_and_separate_runtime(
    tmp_path: Path,
) -> None:
    output = tmp_path / "probe"
    result = run_tiny_robustness(FIXTURE, MODELS, output)
    data = json.loads(result.read_bytes())
    assert len(data["runs"]) == 18
    assert data["cost_mismatch_tested"] is False
    assert set(data["assumptions"]) == {"matched", "overoptimistic", "pessimistic"}
    assert len(json.loads((output / "timings.json").read_bytes())) == 18
    for item in data["runs"]:
        assert "scenario_wall_seconds" not in item
        for frame in item["scenario"]["frames"]:
            assert frame["spent"] == sum(COSTS[mode] for mode in frame["modes"])
            assert frame["spent"] <= 6
            assert all(field["scores"]["log_score"] >= 0 for field in frame["fields"])
    for name, checksum in json.loads(
        (output / "files.sha256.json").read_bytes()
    ).items():
        assert digest(output / name) == checksum
    before = result.read_bytes()
    with pytest.raises(FileExistsError):
        run_tiny_robustness(FIXTURE, MODELS, output)
    assert result.read_bytes() == before
