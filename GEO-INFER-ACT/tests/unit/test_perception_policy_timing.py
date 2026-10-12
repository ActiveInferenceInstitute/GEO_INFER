"""Analytic regressions for one observation update followed by policy selection."""

import numpy as np
import pytest

from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.utils.pymdp_adapter import run_model_step


def _agent(**options):
    likelihood = np.array([[0.8, 0.15, 0.4], [0.2, 0.85, 0.6]])
    transitions = np.stack([np.roll(np.eye(3), 1, axis=0), np.eye(3)], axis=2)
    model = GenerativeModel(
        "categorical",
        {
            "state_dim": 3,
            "obs_dim": 2,
            "A": likelihood,
            "B": transitions,
            "D": np.array([0.2, 0.5, 0.3]),
            "C": np.array([1.0, -1.0]),
            "num_controls": [2],
        },
    )
    agent = ActiveInferenceModel(num_controls=[2], random_seed=7, **options)
    agent.set_generative_model(model)
    return agent, likelihood, np.array([0.2, 0.5, 0.3])


@pytest.mark.parametrize("observation", [[1.0, 0.0], [8.0, 2.0]])
def test_step_uses_one_posterior_for_beliefs_policies_and_evidence(observation):
    """Nonidentity B affects prospective policies, without reconditioning q(s)."""
    agent, likelihood, prior = _agent()
    frequencies = np.asarray(observation) / np.sum(observation)
    unnormalized = prior * np.exp(frequencies @ np.log(likelihood))
    posterior = unnormalized / unnormalized.sum()
    expected = run_model_step(agent.generative_model, observation, prior=prior)

    result = agent.step(np.asarray(observation), return_result=True)

    np.testing.assert_allclose(result.beliefs["states"], posterior, atol=1e-6)
    np.testing.assert_allclose(agent.latest_pymdp_result.beliefs, posterior, atol=1e-6)
    np.testing.assert_allclose(
        agent.generative_model.beliefs["states"], posterior, atol=1e-6
    )
    np.testing.assert_allclose(
        agent.latest_pymdp_result.policy_posterior, expected.policy_posterior, atol=1e-6
    )
    np.testing.assert_allclose(
        agent.latest_pymdp_result.negative_expected_free_energy,
        expected.negative_expected_free_energy,
        atol=1e-6,
    )
    assert result.free_energy == pytest.approx(expected.free_energy, abs=1e-6)
    assert result.metadata["pymdp"]["inference_mode"] == "policy_only"


def test_repeated_act_never_calls_state_inference_or_advances_beliefs(monkeypatch):
    from pymdp.agent import Agent

    agent, _, _ = _agent()
    posterior = agent.perceive(np.array([1.0, 0.0]))["states"]
    free_energy = agent.latest_pymdp_result.free_energy

    def unexpected_perception(*args, **kwargs):
        raise AssertionError("Action selection must not infer states again")

    monkeypatch.setattr(Agent, "infer_states", unexpected_perception)
    for candidates in (["left", "right"], ["hold", "move"], ["left", "right"]):
        assert agent.act(candidates) in candidates
        np.testing.assert_allclose(agent.current_beliefs["states"], posterior)
        np.testing.assert_allclose(
            agent.latest_pymdp_result.beliefs, posterior, atol=1e-7
        )
        assert agent.latest_pymdp_result.free_energy == free_energy
    with pytest.raises(ValueError, match="Action count"):
        agent.act(["hold"])
    assert agent.history == []


def test_separate_perceptions_each_condition_once_and_preserve_tuple_api():
    agent, likelihood, prior = _agent()
    for observation_index in (0, 1):
        expected = prior * likelihood[observation_index]
        expected /= expected.sum()
        beliefs, action = agent.step(np.eye(2)[observation_index])
        np.testing.assert_allclose(beliefs["states"], expected, atol=1e-6)
        assert action in (0, 1)
        prior = expected
    assert len(agent.history) == 2


def test_replacement_and_reset_clear_perception_diagnostics():
    agent, _, prior = _agent()
    agent.step(np.array([1.0, 0.0]))
    replacement, _, _ = _agent()
    agent.set_generative_model(replacement.generative_model)
    assert agent.latest_pymdp_result is None
    assert agent.current_observations is None
    assert agent.current_actions is None
    assert agent._perception_free_energy is None
    np.testing.assert_allclose(agent.current_beliefs["states"], prior)
    agent.step(np.array([0.0, 1.0]))
    agent.reset()
    assert agent._perception_free_energy is None
    assert agent.latest_pymdp_result is None
    assert agent.history == []


@pytest.mark.parametrize("observation", [[1.0, 0.0], [8.0, 2.0]])
@pytest.mark.parametrize("clear_beliefs", [False, True])
def test_backend_recovery_after_local_perception_does_not_recondition(
    monkeypatch, observation, clear_beliefs
):
    import geo_infer_act.core.active_inference as runtime

    agent, likelihood, prior = _agent(allow_local_pymdp_fallback=True)
    if clear_beliefs:
        agent.current_beliefs = None
    original = runtime.run_model_step

    def unavailable(*args, **kwargs):
        raise RuntimeError("Backend temporarily unavailable")

    monkeypatch.setattr(runtime, "run_model_step", unavailable)
    posterior = agent.perceive(np.array(observation))["states"]
    assert agent.latest_pymdp_result is None
    monkeypatch.setattr(runtime, "run_model_step", original)
    agent.act()
    np.testing.assert_allclose(agent.latest_pymdp_result.beliefs, posterior, atol=1e-7)
    np.testing.assert_allclose(agent.generative_model.beliefs["states"], posterior)
    # The opt-in local updater retains hard/soft count semantics (no frequency
    # normalization). Its preserved diagnostic must use that actual likelihood.
    evidence = np.sum(prior * np.exp(np.asarray(observation) @ np.log(likelihood)))
    assert agent.latest_pymdp_result.free_energy == pytest.approx(-np.log(evidence))


@pytest.mark.parametrize("posterior", [[0.4, 0.6], [-0.1, 0.6, 0.5], [0.2, 0.2, 0.2]])
def test_policy_only_adapter_rejects_invalid_posteriors(posterior):
    agent, _, prior = _agent()
    with pytest.raises(ValueError, match="normalized state vector"):
        run_model_step(
            agent.generative_model,
            [1.0, 0.0],
            prior=prior,
            posterior=posterior,
            perception_free_energy=0.1,
        )


@pytest.mark.parametrize("free_energy", [None, float("inf"), float("nan")])
def test_policy_only_adapter_requires_finite_perception_diagnostic(free_energy):
    agent, _, prior = _agent()
    with pytest.raises(ValueError, match="finite perception_free_energy"):
        run_model_step(
            agent.generative_model,
            [1.0, 0.0],
            prior=prior,
            posterior=prior,
            perception_free_energy=free_energy,
        )
