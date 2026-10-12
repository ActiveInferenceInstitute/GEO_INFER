#!/usr/bin/env python3

"""
Smoke tests for the torch-based GenerativeModel in
geo_infer_agent.core.active_inference.

Uses tiny dimensions and few optimisation steps so the suite stays fast
while still exercising encode / plan_actions / expected_free_energy /
update end-to-end.
"""

import math

import numpy as np
import torch

from geo_infer_agent.core.active_inference import (
    ActiveInferenceConfig,
    ActiveInferenceAgent,
    GenerativeModel,
)

TINY_CONFIG = ActiveInferenceConfig(
    planning_horizon=2,
    optimization_steps=2,
    hidden_size=4,
    n_hidden_layers=0,
    n_samples=2,
    learning_rate=0.05,
    random_seed=0,
)


def _tiny_model() -> GenerativeModel:
    torch.manual_seed(0)
    return GenerativeModel(state_dim=2, obs_dim=2, action_dim=2, config=TINY_CONFIG)


class TestGenerativeModel:
    """End-to-end smoke tests for the neural generative model."""

    def test_distributions_have_correct_shapes(self) -> None:
        model = _tiny_model()
        state = torch.zeros(3, 2)
        action = torch.zeros(3, 2)

        obs_dist = model.likelihood(state)
        next_state_dist = model.transition(state, action)
        prior_dist = model.prior(batch_size=3)
        policy_dist = model.policy(state)

        assert obs_dist.mean.shape == (3, 2)
        assert next_state_dist.mean.shape == (3, 2)
        assert prior_dist.mean.shape == (3, 2)
        assert policy_dist.probs.shape == (3, 2)

    def test_encode_returns_finite_state_distribution(self) -> None:
        model = _tiny_model()
        observation = torch.zeros(1, 2)

        q_s = model.encode(observation, steps=2)

        assert q_s.mean.shape == (1, 2)
        assert torch.isfinite(q_s.mean).all()
        assert (q_s.stddev > 0).all()

    def test_expected_free_energy_is_finite_scalar_per_sample(self) -> None:
        model = _tiny_model()
        state = torch.zeros(1, 2)
        action = torch.zeros(1, 2)

        g = model.expected_free_energy(state, action)

        assert g.shape == (1,)
        assert torch.isfinite(g).all()

    def test_plan_actions_returns_horizon_of_actions(self) -> None:
        model = _tiny_model()
        state = torch.zeros(1, 2)

        plan = model.plan_actions(state)

        assert len(plan) == TINY_CONFIG.planning_horizon
        for action in plan:
            assert action.shape == (1, model.action_dim)
            assert torch.isfinite(action).all()

    def test_update_returns_finite_losses_and_changes_parameters(self) -> None:
        model = _tiny_model()
        batch = 2
        states = torch.randn(batch, 2)
        actions = torch.randn(batch, 2)
        next_states = torch.randn(batch, 2)
        observations = torch.randn(batch, 2)

        params_before = [p.detach().clone() for p in model.parameters()]
        losses = model.update(states, actions, next_states, observations)

        assert set(losses) == {
            "likelihood_loss",
            "transition_loss",
            "prior_loss",
            "policy_loss",
            "total_loss",
        }
        assert all(math.isfinite(v) for v in losses.values())
        changed = any(
            not torch.equal(before, after.detach())
            for before, after in zip(params_before, model.parameters())
        )
        assert changed, "update() must train the model parameters"


class TestActiveInferenceAgentSmoke:
    """Minimal numpy-level smoke test for the high-level agent wrapper."""

    def test_perceive_plan_act_learn_cycle(self) -> None:
        agent = ActiveInferenceAgent(
            state_dim=2,
            obs_dim=2,
            action_dim=2,
            config={
                "planning_horizon": 2,
                "iterations": 2,
                "hidden_size": 4,
                "n_hidden_layers": 0,
                "n_samples": 2,
                "random_seed": 0,
            },
        )

        observation = np.zeros(2, dtype=np.float32)
        state = agent.perceive(observation)
        assert state.shape == (2,)
        assert np.isfinite(state).all()

        plan = agent.plan(state)
        assert len(plan) == 2
        assert all(a.shape == (2,) for a in plan)

        action = agent.act(state)
        assert action.shape == (2,)
        assert np.isfinite(action).all()

        agent.add_experience(state, action, state, observation)
        losses = agent.learn()
        assert losses
        assert all(math.isfinite(v) for v in losses.values())
