"""Independent numerical and boundary oracles from the complete ACT method scan."""

import copy

import numpy as np
import pytest

from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.core.dynamic_causal_model import DynamicCausalModel
from geo_infer_act.core.factored_runtime import build_runtime_artifact
from geo_infer_act.core.free_energy import FreeEnergyCalculator
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.core.markov_decision_process import MarkovDecisionProcess
from geo_infer_act.core.types import ActiveInferenceStepResult
from geo_infer_act.core.variational_inference import VariationalInference
from geo_infer_act.models.base import GaussianModel
from geo_infer_act.models.climate import ClimateModel
from geo_infer_act.models.ecological import EcologicalModel
from geo_infer_act.models.resource import ResourceModel
from geo_infer_act.models.multi_agent import MultiAgentModel
from geo_infer_act.models.urban import UrbanModel
from geo_infer_act.runners.contracts import RunConfig, normalize_scenario_list
from geo_infer_act.utils.config import load_config, merge_configs
from geo_infer_act.utils.h3_adapter import get_h3_adapter
from geo_infer_act.core.spatial_agent import SpatialActiveInferenceAgent
from geo_infer_act.utils.integration import ModernToolsIntegration, integrate_rxinfer
from geo_infer_act.api.interface import ActiveInferenceInterface


def test_gaussian_free_energy_matches_closed_form_rectangular_oracle():
    """KL(q||p) + expected normalized measurement NLL, calculated by scalars."""
    mean, prior_mean = np.array([1.0, -0.5]), np.array([0.25, 0.5])
    covariance = np.diag([0.5, 2.0])
    prior_variance = np.array([2.0, 0.25])
    measurement, noise_variance = 2.0, 0.4
    mapping = np.array([[2.0, -1.0]])
    kl = 0.5 * sum(
        variance / prior_var
        + (mu - initial) ** 2 / prior_var
        - 1
        + np.log(prior_var / variance)
        for variance, prior_var, mu, initial in zip(
            np.diag(covariance), prior_variance, mean, prior_mean
        )
    )
    nll = 0.5 * (
        np.log(2 * np.pi * noise_variance)
        + ((measurement - 2.5) ** 2 + 4 * 0.5 + 2) / noise_variance
    )
    actual = FreeEnergyCalculator().compute_gaussian_free_energy(
        mean,
        np.diag(1 / np.diag(covariance)),
        np.array([measurement]),
        prior_mean,
        np.diag(1 / prior_variance),
        observation_matrix=mapping,
        observation_precision=np.array([[1 / noise_variance]]),
    )
    assert actual == pytest.approx(kl + nll, abs=1e-12)
    vi = VariationalInference()
    assert vi.compute_elbo(
        {"mean": mean, "precision": np.diag(1 / np.diag(covariance))},
        {"mean": prior_mean, "precision": np.diag(1 / prior_variance)},
        {"observation_matrix": mapping, "precision": np.array([[1 / noise_variance]])},
        np.array([measurement]),
    ) == pytest.approx(-actual)


@pytest.mark.parametrize("state_dim, obs_dim", [(3, 1), (1, 3)])
def test_rectangular_gaussian_model_matches_independent_scalar_filter(
    state_dim, obs_dim
):
    model = GaussianModel(state_dim=state_dim, obs_dim=obs_dim)
    assert model.C.shape == (obs_dim, state_dim)
    model.set_observation_model(model.C, R=np.eye(obs_dim))
    model.set_transition_model(np.eye(state_dim), Q=np.zeros((state_dim, state_dim)))
    result = model.update_beliefs(np.ones(obs_dim))
    expected_mean, expected_covariance = np.zeros(state_dim), np.eye(state_dim)
    expected_mean[: min(state_dim, obs_dim)] = 0.5
    expected_covariance[
        np.arange(min(state_dim, obs_dim)), np.arange(min(state_dim, obs_dim))
    ] = 0.5
    np.testing.assert_allclose(result["mean"], expected_mean)
    np.testing.assert_allclose(result["cov"], expected_covariance)
    result["mean"][:] = 999
    assert not np.any(model.belief_mean == 999)
    original = model.A.copy()
    with pytest.raises(ValueError):
        model.set_transition_model(np.eye(state_dim) * 2, Q=-np.eye(state_dim))
    np.testing.assert_array_equal(model.A, original)


@pytest.mark.parametrize(
    "factory, observation", [(ClimateModel, [1, 1]), (EcologicalModel, [2, 1])]
)
def test_domain_factored_posterior_policy_and_next_prior_match_enumeration(
    factory, observation
):
    model = factory(config={"policy_selection_mode": "deterministic"}, random_seed=9)
    generative = model.generative_model
    priors = generative.parameters["D"]
    prior = np.kron(*priors)
    likelihoods = [
        np.asarray(matrix).reshape(matrix.shape[0], -1)
        for matrix in generative.observation_model
    ]
    observed = np.prod(
        [matrix[index] for matrix, index in zip(likelihoods, observation)], axis=0
    )
    evidence = observed @ prior
    posterior = observed * prior / evidence
    result = model.step(observation, return_result=True)
    assert isinstance(result, ActiveInferenceStepResult)
    np.testing.assert_allclose(result.beliefs["joint"], posterior, atol=1e-12)
    assert result.free_energy == pytest.approx(-np.log(evidence))
    controls = [result.action] if isinstance(result.action, int) else result.action
    transitions = generative.transition_model
    next_prior = (
        np.kron(transitions[0][:, :, controls[0]], transitions[1][:, :, controls[-1]])
        @ posterior
    )
    observation2 = [0, 0]
    likelihood2 = np.prod(
        [matrix[index] for matrix, index in zip(likelihoods, observation2)], axis=0
    )
    expected_second = next_prior * likelihood2
    expected_second /= expected_second.sum()
    second = model.step(observation2, return_result=True)
    np.testing.assert_allclose(second.beliefs["joint"], expected_second, atol=1e-12)
    assert len(model.get_history()) == 2
    model.reset()
    reset = model.step(observation, return_result=True)
    np.testing.assert_allclose(reset.beliefs["joint"], posterior)
    assert reset.action == result.action


def test_correlated_joint_prior_survives_filtering_and_input_ownership():
    initial = np.array([0.4, 0.1, 0, 0.1, 0.3, 0, 0, 0, 0.1])
    model = ClimateModel(
        config={
            "initial_joint": initial.tolist(),
            "policy_selection_mode": "deterministic",
        }
    )
    before = copy.deepcopy(model.generative_model.parameters)
    result = model.step([0, 0], return_result=True)
    joint = result.beliefs["joint"].reshape(3, 3)
    assert not np.allclose(joint, np.outer(joint.sum(1), joint.sum(0)))
    result.beliefs["joint"][:] = 0
    assert model.current_beliefs["joint"].sum() == pytest.approx(1)
    assert before["initial_joint"] == model.generative_model.parameters["initial_joint"]


@pytest.mark.parametrize(
    "factory, observation", [(ClimateModel, [0, 0]), (EcologicalModel, [2, 1])]
)
@pytest.mark.parametrize("conditioned", [False, True])
def test_factored_summary_entropy_is_the_declared_joint_with_ragged_factor_support(
    factory, observation, conditioned
):
    model = factory(random_seed=9)
    if factory is ClimateModel:
        model.generative_model.parameters["initial_joint"] = [
            0.4,
            0.1,
            0,
            0.1,
            0.3,
            0,
            0,
            0,
            0.1,
        ]
    prior = np.asarray(
        build_runtime_artifact(model.generative_model).to_dict()["initial_joint"]
    )
    if conditioned:
        observed = np.prod(
            [
                matrix[index].reshape(-1)
                for matrix, index in zip(
                    model.generative_model.observation_model, observation
                )
            ],
            axis=0,
        )
        expected_joint = prior * observed
        expected_joint /= expected_joint.sum()
        model.step(observation)
    else:
        expected_joint = prior
    positive = expected_joint > 0
    expected_entropy = -np.sum(
        expected_joint[positive] * np.log(expected_joint[positive])
    )
    summary = model.generative_model.get_model_summary()
    assert summary["belief_entropy"] == pytest.approx(expected_entropy, abs=1e-10)
    assert summary["free_energy"] == pytest.approx(
        np.log(expected_joint.size) - expected_entropy, abs=1e-10
    )
    assert summary["free_energy_definition"] == "reference_kl_to_uniform_joint"


@pytest.mark.parametrize(
    "observation", [[True, 0], [0.5, 0], [-1, 0], [3, 0], [np.nan, 0], [0]]
)
def test_factored_invalid_observations_preserve_state(observation):
    model = ClimateModel()
    initial = copy.deepcopy(model.current_beliefs)
    with pytest.raises(ValueError):
        model.step(observation)
    for before, after in zip(initial["states"], model.current_beliefs["states"]):
        np.testing.assert_array_equal(before, after)
    assert model.history == []


def test_factored_temperature_and_zero_policy_prior_are_effective():
    model = EcologicalModel(config={"E": [0, 1, 0], "policy_temperature": 4.0})
    result = model.step([1, 0], return_result=True)
    assert result.action == 1
    assert result.metadata["policy_selection"]["all_probabilities"].tolist() == [
        0,
        1,
        0,
    ]
    models = [
        EcologicalModel(
            config={
                "policy_temperature": value,
                "policy_selection_mode": "deterministic",
            }
        )
        for value in (0.5, 4.0)
    ]
    probabilities = [
        value.step([1, 0], return_result=True).metadata["policy_selection"][
            "all_probabilities"
        ]
        for value in models
    ]
    assert probabilities[0].max() > probabilities[1].max()


def test_urban_horizon_is_used_and_allocation_is_bounded():
    model = UrbanModel(
        n_agents=1, n_resources=4, n_locations=3, planning_horizon=2, random_seed=8
    )
    artifact = build_runtime_artifact(
        model.agents[0]["model"].generative_model
    ).to_dict()
    assert all(len(policy) == 2 for policy in artifact["policies"])
    assert all(
        len(modality["preferences"]) in (3, 4) for modality in artifact["modalities"]
    )
    first = model.run_simulation(2)
    model.reset()
    second = model.run_simulation(2)
    assert [state["states"][0]["location"] for state in first] == [
        state["states"][0]["location"] for state in second
    ]
    with pytest.raises(ValueError, match="work budget"):
        UrbanModel(n_agents=1, planning_horizon=8)


def test_resource_transport_conserves_mass_and_invalid_action_is_atomic():
    model = ResourceModel(
        n_resources=1, n_locations=3, replenishment_rate=0, random_seed=2
    )
    model.resource_distribution[:] = [[0.1, 0.2, 0.3]]
    model.location_demand[:] = 0
    model.location_connectivity = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    before = model.resource_distribution.copy()
    with pytest.raises(ValueError):
        model.step({"allocations": [[-1, 0, 0]]})
    np.testing.assert_array_equal(model.resource_distribution, before)
    assert model.step_count == 0
    result, _ = model.step()
    np.testing.assert_allclose(result["resource_distribution"], [[0.12, 0.19, 0.29]])
    assert result["total_resources"] == pytest.approx(0.6)
    model.location_demand[:] = 2
    depleted, _ = model.step([[1, 1, 1]])
    assert np.all(depleted["harvest_yield"] == 0)


def test_dcm_irregular_time_filter_matches_scalar_oracle():
    model = DynamicCausalModel(1, 1, 1, random_seed=1)
    model.set_parameters(np.array([[-0.5]]), np.array([[2.0]]), np.array([[1.0]]))
    model.set_noise_parameters(np.array([[0.2]]), np.array([[0.5]]))
    observations, inputs, axis = (
        np.array([[1], [2], [-1.0]]),
        np.array([[0.3], [-0.2]]),
        np.array([0, 0.2, 1.0]),
    )
    actual = model._estimate_states(observations, inputs, axis, np.array([0.0]))
    mean, variance, oracle = 0.0, 1.0, []
    for index, observation in enumerate(observations[:, 0]):
        if index:
            dt = axis[index] - axis[index - 1]
            mean = (1 - 0.5 * dt) * mean + dt * 2 * inputs[index - 1, 0]
            variance = (1 - 0.5 * dt) ** 2 * variance + 0.2 * dt
        gain = variance / (variance + 0.5)
        mean += gain * (observation - mean)
        variance *= 1 - gain
        oracle.append(mean)
    np.testing.assert_allclose(actual[:, 0], oracle, atol=1e-12)


def test_dcm_parameter_fit_returns_continuous_generators(monkeypatch):
    model = DynamicCausalModel(1, 1, 1)
    axis = np.array([0, 0.2, 0.7, 1.1])
    inputs = np.array([[1], [-1], [0.5]])
    states = [2.0]
    for dt, control in zip(np.diff(axis), inputs[:, 0]):
        states.append(states[-1] + dt * (-0.3 * states[-1] + 0.8 * control))
    states = np.asarray(states).reshape(-1, 1)
    monkeypatch.setattr(model, "_estimate_states", lambda *args: states.copy())
    result = model.estimate_parameters(states * 1.2, inputs, axis)
    np.testing.assert_allclose(result["A"], [[-0.3]], atol=1e-12)
    np.testing.assert_allclose(result["B"], [[0.8]], atol=1e-12)
    np.testing.assert_allclose(result["C"], [[1.2]], atol=1e-12)


@pytest.mark.parametrize(
    "axis, inputs", [([0, 0], [[0]]), ([0, np.nan], [[0]]), ([0, 1], [])]
)
def test_dcm_rejects_missing_or_invalid_model_intervals(axis, inputs):
    model = DynamicCausalModel(1, 1, 1)
    with pytest.raises(ValueError):
        model.integrate_dynamics(
            np.zeros(1), np.asarray(inputs).reshape(-1, 1), np.asarray(axis)
        )
    with pytest.raises(ValueError):
        model.estimate_parameters(
            np.zeros((2, 1)), np.asarray(inputs).reshape(-1, 1), np.asarray(axis)
        )


def test_mdp_deterministic_initial_observation_and_owned_distributions():
    transition = np.zeros((2, 2, 1))
    transition[:, :, 0] = [[0.8, 0.2], [0.1, 0.9]]  # MDP uses (current, next, action).
    model = MarkovDecisionProcess(
        2, 2, 1, transition, np.array([[0.9, 0.4], [0.1, 0.6]]), random_seed=7
    )
    for _ in range(10):
        assert model.simulate(0, [0], stochastic=False) == ([0, 0], [0, 0])
    np.testing.assert_allclose(
        model.get_predictive_state(np.array([0.25, 0.75]), 0), [0.275, 0.725]
    )
    model.get_transition_prob(0, 0)[:] = 0
    assert model.transition_prob[0, :, 0].sum() == pytest.approx(1)
    old = model.transition_prob.copy()
    with pytest.raises(ValueError):
        model.set_transition_matrix(0, 0, np.array([-0.1, 1.1]))
    np.testing.assert_array_equal(model.transition_prob, old)
    with pytest.raises(ValueError):
        model.get_transition_prob(-1, 0)


def test_importance_sampling_normalization_and_zero_support():
    vi = VariationalInference(random_seed=17)
    prior = {"mean": np.zeros(1), "covariance": np.eye(1)}
    result = vi.importance_sampling_update(
        prior, lambda sample, observation: 1e-200, np.ones(1), n_samples=8
    )
    np.testing.assert_allclose(result["weights"], [0.125] * 8)
    np.testing.assert_allclose(result["mean"], result["samples"].mean(0))
    with pytest.raises(ValueError, match="positive support"):
        vi.importance_sampling_update(prior, lambda *args: 0, np.ones(1), n_samples=8)


def test_contradictory_factor_support_is_rejected():
    graph = {
        "variables": {"x": {"dimension": 2, "prior": [1, 0]}, "y": {"dimension": 2}},
        "factors": {
            "relation": {"variables": ["x", "y"], "potential": [[0, 1], [1, 0]]}
        },
    }
    with pytest.raises(ValueError, match="zero.*support"):
        VariationalInference().structured_update(graph, {"y": np.array([1, 0])})


def test_config_merge_and_load_are_owned_and_fail_on_scalar_yaml(tmp_path):
    base, override = {"nested": {"list": [1]}}, {"nested": {"extra": [2]}}
    merged = merge_configs(base, override)
    merged["nested"]["list"].append(3)
    merged["nested"]["extra"].append(4)
    assert base == {"nested": {"list": [1]}}
    assert override == {"nested": {"extra": [2]}}
    path = tmp_path / "invalid.yaml"
    path.write_text("42")
    with pytest.raises(ValueError, match="mapping"):
        load_config(str(path))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timesteps": 0},
        {"h3_resolution": True},
        {"seed": -1},
        {"visualizations": 1},
        {"output_formats": ["json"]},
        {"schema_version": "unknown"},
    ],
)
def test_runner_rejects_unsupported_configuration(kwargs):
    with pytest.raises(ValueError):
        RunConfig(**kwargs)


def test_empty_selection_and_unconfigured_policy_fail():
    for names in ([], ["simple", "simple"]):
        with pytest.raises(ValueError):
            normalize_scenario_list(names)
    with pytest.raises(ValueError, match="beliefs"):
        ActiveInferenceModel().select_policy([{"action": "stay"}])


def test_spatial_cell_ownership_pentagon_and_allocation_budget():
    adapter = get_h3_adapter()
    import h3

    cells = h3.get_pentagons(0)[:1]
    agent = SpatialActiveInferenceAgent(initial_cells=cells, h3_resolution=0)
    assert agent.cells == cells
    cells.clear()
    assert len(agent.cells) == 1
    with pytest.raises(ValueError, match="nonempty"):
        SpatialActiveInferenceAgent(initial_cells=[])
    with pytest.raises(ValueError, match="budget"):
        SpatialActiveInferenceAgent(max_entries=10)
    cell = agent.cells[0]
    assert len(adapter.grid_ring(cell, 1)) == 5
    with pytest.raises(ValueError, match="Duplicate"):
        SpatialActiveInferenceAgent(initial_cells=[cell, cell], h3_resolution=0)


def test_spatial_navigation_transition_column_orientation():
    model = GenerativeModel("categorical", {"state_dim": 2, "obs_dim": 2})
    model.enable_spatial_navigation(grid_size=2)
    assert model.transition_model.shape == (4, 4, 4)
    np.testing.assert_allclose(model.transition_model.sum(axis=0), 1)
    assert model.transition_model[1, 0, 3] == 1  # east from top left
    with pytest.raises(ValueError, match="budget"):
        model.enable_spatial_navigation(grid_size=100)
    assert model.grid_size == 2


def test_explicit_local_gaussian_is_a_conjugate_posterior():
    model = GenerativeModel("categorical", {"state_dim": 2})
    result = model.integrate_rxinfer(
        "",
        {
            "observations": [1, 2, 3],
            "prior_mean": 4,
            "prior_precision": 2,
            "measurement_precision": 0.5,
        },
        backend="local_gaussian",
    )
    assert result["backend"] == "local_gaussian"
    assert result["posterior_marginals"]["mean"] == pytest.approx(11 / 3.5)
    assert result["posterior_marginals"]["variance"] == pytest.approx(1 / 3.5)


def test_numpy_metropolis_does_not_hide_density_failures_or_invent_ess():
    model = GenerativeModel("categorical", {"state_dim": 2, "random_seed": 3})
    result = model.integrate_bayeux(
        lambda x: -float(x**2), {"x": np.array(0.0)}, n_samples=20, warmup=3
    )
    assert result["backend"] == "numpy_metropolis"
    assert result["posterior_samples"]["x"].shape == (20,)
    assert result["log_marginal_likelihood"] is None
    assert "effective_sample_size" not in result["diagnostics"]
    calls = 0

    def failed_density(x):
        nonlocal calls
        calls += 1
        if calls > 2:
            raise RuntimeError("density evaluation failed")
        return -float(x**2)

    with pytest.raises(RuntimeError, match="density evaluation failed"):
        model.integrate_bayeux(
            failed_density, {"x": np.array(0.0)}, n_samples=10, warmup=0
        )


def test_integration_helpers_require_data_and_explicit_local_backend():
    with pytest.raises(ValueError, match="explicit data"):
        integrate_rxinfer({}, {})
    with pytest.raises(ValueError, match="model_specification"):
        integrate_rxinfer({}, {"data": {"observations": [1]}})
    hub = ModernToolsIntegration({"allow_local_fallback": True})
    with pytest.raises(ValueError, match="was removed"):
        hub.create_rxinfer_model("", {"observations": [1]})
    hub = ModernToolsIntegration()
    with pytest.raises(RuntimeError, match="allow_dynamic_code"):
        hub.create_bayeux_model("def log_density(point): return 0", {"x": [0]})


def test_spatial_diffusion_and_parent_aggregation_match_real_h3_oracles():
    adapter = get_h3_adapter()
    pentagon = adapter.h3.get_pentagons(2)[0]
    cells = [pentagon, *adapter.grid_ring(pentagon, 1)]
    assert len(cells) == 6
    model = GenerativeModel("categorical", {"state_dim": 2})
    model.spatial_mode = True
    model.spatial_graph = {
        cell: set(adapter.grid_ring(cell, 1)) & set(cells) for cell in cells
    }
    beliefs = {cell: np.array([1.0, 0.0]) for cell in reversed(cells)}
    beliefs[pentagon] = np.array([0.0, 1.0])
    result = model.diffuse_beliefs(beliefs, diffusion_rate=0.25)
    assert list(result) == list(beliefs)
    np.testing.assert_allclose(result[pentagon], [0.25, 0.75])
    for cell in cells[1:]:
        neighbors = model.spatial_graph[cell]
        expected = [1 - 0.25 / len(neighbors), 0.25 / len(neighbors)]
        np.testing.assert_allclose(result[cell], expected)
    np.testing.assert_allclose(beliefs[pentagon], [0, 1])
    aggregate = model.aggregate_beliefs_to_resolution(beliefs, 0)
    parent = adapter.cell_to_parent(pentagon, 0)
    np.testing.assert_allclose(aggregate[parent], [5 / 6, 1 / 6])
    for rate in [-0.1, 1.1, np.nan, True]:
        with pytest.raises(ValueError, match="diffusion_rate"):
            model.diffuse_beliefs(beliefs, rate)
    for values in [[0, 0], [-1, 2], [np.inf, 1]]:
        with pytest.raises(ValueError, match="positive mass"):
            model.diffuse_beliefs({pentagon: values})
    with pytest.raises(ValueError, match="Invalid H3"):
        model.aggregate_beliefs_to_resolution({"invalid": [0.5, 0.5]}, 0)
    with pytest.raises(ValueError, match="refine"):
        model.aggregate_beliefs_to_resolution(beliefs, 3)
    with pytest.raises(ValueError, match="single input"):
        model.aggregate_beliefs_to_resolution(
            {pentagon: [0.5, 0.5], parent: [0.5, 0.5]}, 0
        )


def test_multi_agent_actions_are_validated_atomically_and_results_owned():
    model = MultiAgentModel(n_agents=2, n_resources=1, n_locations=2, random_seed=3)
    resources = model.resource_distribution.copy()
    beliefs = [agent.beliefs.copy() for agent in model.agent_models]
    with pytest.raises(ValueError, match="location"):
        model.step(
            [
                {"agent_id": 0, "resource": 0, "amount": 0.2},
                {"agent_id": 1, "location": 99},
            ]
        )
    np.testing.assert_array_equal(model.resource_distribution, resources)
    for before, agent in zip(beliefs, model.agent_models):
        np.testing.assert_array_equal(agent.beliefs, before)
    assert model.step_count == 0
    with pytest.raises(ValueError, match="Duplicate"):
        model.step([{"agent_id": 0}, {"agent_id": 0}])
    state, _ = model.step()
    state["resource_distribution"][:] = 99
    assert not np.any(model.history[0]["resource_distribution"] == 99)


def test_nested_multi_agent_reconfiguration_preserves_resource_axes():
    adapter = get_h3_adapter()
    cell = adapter.latlng_to_cell(37.7, -122.4, 9)
    model = MultiAgentModel(n_agents=2, n_resources=1, n_locations=2, random_seed=3)
    model.enable_nested_h3_spatial([8, 9], cells=[cell])
    assert model.resource_distribution.shape == (1, 1)
    assert model.agent_preferences.shape == (1, 1)
    model.step([{"agent_id": 0, "resource": 0, "amount": 0.1}])
    model.reset()
    assert model.resource_distribution.shape == (1, 1)


def test_flat_model_explicit_prediction_conserves_probability_across_missing_data():
    generative = GenerativeModel(
        "categorical",
        {
            "state_dim": 3,
            "obs_dim": 3,
            "A": np.eye(3),
            "D": np.array([0.2, 0.5, 0.3]),
            "B": np.stack([np.eye(3), np.roll(np.eye(3), 1, axis=0)], axis=2),
        },
    )
    model = ActiveInferenceModel()
    model.set_generative_model(generative)
    with pytest.raises(ValueError, match="action_index"):
        model.predict_beliefs()
    first = model.predict_beliefs(1)
    np.testing.assert_allclose(first["states"], [0.3, 0.2, 0.5])
    first["states"][:] = 0
    second = model.predict_beliefs(1)
    np.testing.assert_allclose(second["states"], [0.5, 0.3, 0.2])
    assert second["states"].sum() == pytest.approx(1)
    assert model.current_observations is None
    assert model.get_history() == []


def test_public_observation_and_outcome_surfaces_own_data_and_reject_before_history():
    generative = GenerativeModel(
        "categorical", {"state_dim": 2, "obs_dim": 2, "A": np.eye(2)}
    )
    model = ActiveInferenceModel()
    model.set_generative_model(generative)
    observation = np.array([1.0, 0.0])
    model.perceive(observation)
    observation[:] = 999
    np.testing.assert_array_equal(model.current_observations, [1, 0])
    with pytest.raises(ValueError):
        model.perceive(np.ones(3))
    np.testing.assert_array_equal(model.current_observations, [1, 0])
    with pytest.raises(ValueError):
        model.update_with_outcome({"action": "sample"}, {"observation": [-1, 2]})
    assert model.history == []
    decision = {"action": {"name": "wait"}}
    outcome = {"reward": [1]}
    model.update_with_outcome(decision, outcome)
    decision["action"]["name"] = "changed"
    outcome["reward"][0] = 999
    assert model.get_history()[0]["decision"]["action"]["name"] == "wait"
    assert model.get_history()[0]["outcome"]["reward"] == [1]
    supplied = {"context": {"sensor": [2]}}
    model.update_observations(supplied)
    supplied["context"]["sensor"][0] = 999
    state = model.get_current_state()
    assert state["observations"]["context"]["sensor"] == [2]
    state["observations"]["context"]["sensor"][0] = 999
    assert model.current_observations["context"]["sensor"] == [2]


def test_api_does_not_guess_observation_order_or_overwrite_configured_model():
    interface = ActiveInferenceInterface()
    interface.create_model("ordered", "categorical", {"state_dim": 2, "obs_dim": 2})
    model = interface.models["ordered"]
    with pytest.raises(ValueError, match="already exists"):
        interface.create_model("ordered", "categorical", {"state_dim": 3})
    assert interface.models["ordered"] is model
    with pytest.raises(ValueError, match="explicit"):
        interface.update_beliefs("ordered", {"z_sensor": 0, "a_sensor": 1})
    assert model.current_observations is None
    with pytest.raises(ValueError, match="non-empty"):
        interface.create_model("", "categorical", {"state_dim": 2})


def test_rectangular_gaussian_core_vfe_retains_declared_observation_map():
    generative = GenerativeModel("gaussian", {"state_dim": 2, "obs_dim": 1})
    model = ActiveInferenceModel("gaussian")
    model.set_generative_model(generative)
    prior_variance, noise_variance, measurement = 1.01, 0.01, 2.0
    variance = prior_variance * noise_variance / (prior_variance + noise_variance)
    mean = prior_variance * measurement / (prior_variance + noise_variance)
    kl = 0.5 * (
        variance / prior_variance
        + mean**2 / prior_variance
        - 1
        + np.log(prior_variance / variance)
    )
    nll = 0.5 * (
        np.log(2 * np.pi * noise_variance)
        + ((measurement - mean) ** 2 + variance) / noise_variance
    )
    beliefs = model.perceive(np.array([measurement]))
    np.testing.assert_allclose(beliefs["mean"], [mean, 0])
    assert model.compute_free_energy() == pytest.approx(kl + nll, abs=1e-12)
    with pytest.raises(ValueError, match="explicit control dynamics"):
        model.act(["move"])
    np.testing.assert_allclose(model.current_beliefs["mean"], [mean, 0])
    assert model.get_history() == []
