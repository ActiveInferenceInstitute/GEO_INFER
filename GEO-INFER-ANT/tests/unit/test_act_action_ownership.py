"""Real categorical inference binds physical execution to an owned action."""

import asyncio

import numpy as np
import pytest

from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_ant.core.agent_base import SwarmAgent, SensoryInput


@pytest.fixture
def configured_agent():
    model = ActiveInferenceModel(model_type="categorical", random_seed=42)
    model.set_generative_model(
        GenerativeModel(
            model_type="categorical",
            parameters={
                "state_dim": 2,
                "obs_dim": 2,
                "A": np.array([[0.9, 0.2], [0.1, 0.8]]),
                "B": np.stack([np.eye(2), np.array([[0, 1], [1, 0]])], axis=2),
                "C": np.array([0.0, 4.0]),
                "D": np.array([0.6, 0.4]),
            },
        )
    )
    agent = SwarmAgent(
        "owned-act",
        np.zeros(2),
        movement_speed=1.0,
        active_inference_enabled=True,
        active_inference_model=model,
        act_observation_encoder=lambda context: np.array([1.0, 0.0]),
        act_actions=[
            {
                "action_type": "move_toward_resource",
                "parameters": {"target": np.array([-3.0, 0.0])},
            },
            {
                "action_type": "move_toward_resource",
                "parameters": {"target": np.array([3.0, 0.0])},
            },
        ],
    )
    decision = agent.make_decision(SensoryInput())
    assert model.latest_policy_evaluation.index == 1
    return agent, model, decision


def test_public_decision_mutation_cannot_retarget_selected_transition(configured_agent):
    agent, model, decision = configured_agent
    decision.action_type = "rest"
    decision.parameters["target"][:] = -100
    result = asyncio.run(agent.execute_action(decision))
    assert result["action_type"] == "move_toward_resource"
    np.testing.assert_array_equal(agent.position, [1, 0])
    expected = np.array([0.08, 0.54]) / 0.62
    np.testing.assert_allclose(model.current_beliefs["states"], expected, atol=2e-6)
    recorded = agent.act_prediction_history[0]["action"]
    np.testing.assert_array_equal(recorded["parameters"]["target"], [3, 0])
    assert len(decision.alternative_actions) == 1
    # Snapshot serialization cannot mutate the retained physical record.
    snapshot = agent._pending_act_decision
    assert snapshot is None
    exported = decision.to_dict()
    exported["parameters"]["target"][:] = 20
    np.testing.assert_array_equal(decision.parameters["target"], [-100, -100])


def test_prediction_failure_retry_does_not_repeat_physical_execution(
    configured_agent, monkeypatch
):
    agent, model, decision = configured_agent
    real_predict = model.predict_beliefs

    def failed_predict(action):
        raise RuntimeError("model propagation failed")

    monkeypatch.setattr(model, "predict_beliefs", failed_predict)
    with pytest.raises(RuntimeError, match="model propagation failed"):
        asyncio.run(agent.execute_action(decision))
    np.testing.assert_array_equal(agent.position, [1, 0])
    energy = agent.energy_level
    assert len(agent.performance_history) == 1
    monkeypatch.setattr(model, "predict_beliefs", real_predict)
    result = asyncio.run(agent.execute_action(decision))
    assert result["success"]
    np.testing.assert_array_equal(agent.position, [1, 0])
    assert agent.energy_level == energy
    assert len(agent.performance_history) == 1
    assert len(agent.act_prediction_history) == 1


@pytest.mark.asyncio
async def test_concurrent_execution_cannot_duplicate_selected_action(
    configured_agent, monkeypatch
):
    agent, model, decision = configured_agent
    entered = asyncio.Event()
    release = asyncio.Event()
    real_movement = agent._execute_movement_action

    async def bounded_movement(selected):
        entered.set()
        await asyncio.wait_for(release.wait(), timeout=5)
        return await real_movement(selected)

    monkeypatch.setattr(agent, "_execute_movement_action", bounded_movement)
    task = asyncio.create_task(agent.execute_action(decision))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        with pytest.raises(ValueError, match="already executing"):
            await agent.execute_action(decision)
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=5)
    np.testing.assert_array_equal(agent.position, [1, 0])
    assert len(agent.performance_history) == len(agent.act_prediction_history) == 1
