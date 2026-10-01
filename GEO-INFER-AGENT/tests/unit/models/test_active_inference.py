#!/usr/bin/env python3

"""
Tests for the Active Inference agent module.

This module contains unit tests for:
- GenerativeModel
- ActiveInferenceState
- ActiveInferenceAgent
"""

import unittest
import numpy as np

from copy import deepcopy

from geo_infer_agent.models.active_inference import (
    GenerativeModel,
    ActiveInferenceState,
    ActiveInferenceAgent,
)


class TestGenerativeModel(unittest.TestCase):
    """Test cases for the GenerativeModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dims = 3
        self.obs_dims = 4
        self.control_dims = 2
        self.model = GenerativeModel(
            state_dimensions=self.state_dims,
            observation_dimensions=self.obs_dims,
            control_dimensions=self.control_dims,
            learning_rate=0.1,
        )

    def test_initialization(self):
        """Test if the model initializes with correct dimensions."""
        self.assertEqual(self.model.state_dimensions, self.state_dims)
        self.assertEqual(self.model.observation_dimensions, self.obs_dims)
        self.assertEqual(self.model.control_dimensions, self.control_dims)

        # Check that matrices are initialized with correct shapes
        self.assertEqual(self.model.A.shape, (self.obs_dims, self.state_dims))
        self.assertEqual(
            self.model.B.shape, (self.state_dims, self.state_dims, self.control_dims)
        )
        self.assertEqual(self.model.C.shape, (self.obs_dims,))
        self.assertEqual(self.model.D.shape, (self.state_dims,))

        # Check if beliefs are initialized
        np.testing.assert_array_equal(self.model.current_state_beliefs, self.model.D)

    def test_update_likelihood(self):
        """Test updating the likelihood mapping."""
        observation = np.array([0.1, 0.2, 0.3, 0.4])
        state = np.array([0.5, 0.3, 0.2])

        # Save original A matrix
        original_A = deepcopy(self.model.A)

        # Update likelihood
        self.model.update_likelihood(observation, state)

        # Ensure A matrix changed
        self.assertFalse(np.array_equal(self.model.A, original_A))

        # Check that A is still a valid probability distribution
        np.testing.assert_almost_equal(
            np.sum(self.model.A, axis=0), np.ones(self.state_dims)
        )

    def test_update_transition(self):
        """Test updating the transition probabilities."""
        prev_state = np.array([0.2, 0.3, 0.5])
        current_state = np.array([0.4, 0.5, 0.1])
        action = 1

        # Save original B matrix for the given action
        original_B = deepcopy(self.model.B[:, :, action])

        # Update transition
        self.model.update_transition(prev_state, current_state, action)

        # Ensure B matrix changed for the given action
        self.assertFalse(np.array_equal(self.model.B[:, :, action], original_B))

        # Check that B is still a valid probability distribution
        np.testing.assert_almost_equal(
            np.sum(self.model.B[:, :, action], axis=0), np.ones(self.state_dims)
        )

    def test_infer_state(self):
        """Test state inference from an observation."""
        # Create a simplified test case
        self.model.A = np.array(
            [[0.8, 0.2, 0.3], [0.1, 0.7, 0.2], [0.05, 0.05, 0.4], [0.05, 0.05, 0.1]]
        )
        self.model.current_state_beliefs = np.array([0.3, 0.4, 0.3])

        # Test with a simple observation
        observation = np.array([1, 0, 0, 0])
        posterior = self.model.infer_state(observation)

        # Posterior should be normalized
        self.assertAlmostEqual(np.sum(posterior), 1.0)

        # Highest probability should be for state 0 given our A matrix
        self.assertEqual(np.argmax(posterior), 0)

    def test_predict_next_state(self):
        """Test next state prediction."""
        current_state = np.array([0.2, 0.7, 0.1])
        action = 0

        # Set B to identity matrix for action 0
        self.model.B[:, :, action] = np.eye(self.state_dims)

        predicted = self.model.predict_next_state(current_state, action)

        # With identity transition, predicted should equal current
        np.testing.assert_array_almost_equal(predicted, current_state)

    def test_select_action(self):
        """Test action selection through expected free energy minimization."""
        # Simplify for deterministic testing
        self.model.B[:, :, 0] = np.array(
            [[0.9, 0.1, 0.1], [0.05, 0.8, 0.1], [0.05, 0.1, 0.8]]
        )
        self.model.B[:, :, 1] = np.array(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
        )

        # Set a preference for observation 0
        self.model.C = np.array([2.0, 0.0, 0.0, 0.0])

        # Ensure action selection runs without error
        current_belief = np.array([0.33, 0.33, 0.34])
        action = self.model.select_action(current_belief, planning_horizon=1)

        # Action should be 0 or 1
        self.assertIn(action, [0, 1])

    def test_to_from_dict(self):
        """Test serialization to and from dictionary."""
        model_dict = self.model.to_dict()

        # Check dictionary contains expected keys
        expected_keys = [
            "state_dimensions",
            "observation_dimensions",
            "control_dimensions",
            "learning_rate",
            "A",
            "B",
            "C",
            "D",
            "current_state_beliefs",
        ]
        for key in expected_keys:
            self.assertIn(key, model_dict)

        # Create a new model from the dictionary
        new_model = GenerativeModel.from_dict(model_dict)

        # Check dimensions match
        self.assertEqual(new_model.state_dimensions, self.model.state_dimensions)
        self.assertEqual(
            new_model.observation_dimensions, self.model.observation_dimensions
        )
        self.assertEqual(new_model.control_dimensions, self.model.control_dimensions)

        # Check matrices match
        np.testing.assert_array_equal(new_model.A, self.model.A)
        np.testing.assert_array_equal(new_model.B, self.model.B)
        np.testing.assert_array_equal(new_model.C, self.model.C)
        np.testing.assert_array_equal(new_model.D, self.model.D)


class TestActiveInferenceState(unittest.TestCase):
    """Test cases for the ActiveInferenceState class."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dims = 3
        self.obs_dims = 4
        self.control_dims = 2
        self.state = ActiveInferenceState(
            state_dimensions=self.state_dims,
            observation_dimensions=self.obs_dims,
            control_dimensions=self.control_dims,
        )

    def test_initialization(self):
        """Test if the state initializes correctly."""
        self.assertEqual(self.state.state_dimensions, self.state_dims)
        self.assertEqual(self.state.observation_dimensions, self.obs_dims)
        self.assertEqual(self.state.control_dimensions, self.control_dims)

        # Check generative model initialized
        self.assertIsInstance(self.state.generative_model, GenerativeModel)

        # Check history initialized
        self.assertEqual(len(self.state.observation_history), 0)
        self.assertEqual(len(self.state.action_history), 0)

    def test_update_with_observation(self):
        """Test updating state with new observation."""
        observation = np.array([0.1, 0.2, 0.3, 0.4])

        # Pre-update state belief
        pre_belief = deepcopy(self.state.generative_model.current_state_beliefs)

        # Update with observation
        posterior = self.state.update_with_observation(observation)

        # Check posterior returned
        self.assertEqual(len(posterior), self.state_dims)

        # Check observation history updated
        self.assertEqual(len(self.state.observation_history), 1)
        self.assertEqual(
            len(self.state.observation_history[0]["observation"]), self.obs_dims
        )

        # Current belief should be updated
        self.assertFalse(
            np.array_equal(
                self.state.generative_model.current_state_beliefs, pre_belief
            )
        )

    def test_record_action(self):
        """Test recording an action."""
        action = 1
        reward = 0.5

        # Record action
        self.state.record_action(action, reward)

        # Check action history updated
        self.assertEqual(len(self.state.action_history), 1)
        self.assertEqual(self.state.action_history[0]["action"], action)
        self.assertEqual(self.state.action_history[0]["reward"], reward)

    def test_update_preferences(self):
        """Test updating preferences."""
        preferences = np.array([1.0, 0.0, 0.0, 0.0])

        # Save original preferences
        original_prefs = deepcopy(self.state.generative_model.C)

        # Update preferences
        self.state.update_preferences(preferences)

        # Preferences should be updated
        self.assertFalse(np.array_equal(self.state.generative_model.C, original_prefs))
        np.testing.assert_array_almost_equal(
            self.state.generative_model.C,
            (1 - self.state.generative_model.learning_rate) * original_prefs
            + self.state.generative_model.learning_rate * preferences,
        )

    def test_select_action(self):
        """Test action selection."""
        # Ensure action selection runs without error
        action = self.state.select_action(planning_horizon=1)

        # Action should be valid
        self.assertIn(action, range(self.control_dims))

    def test_to_from_dict(self):
        """Test serialization to and from dictionary."""
        # Add some history data
        self.state.update_with_observation(np.array([0.1, 0.2, 0.3, 0.4]))
        self.state.record_action(1, 0.5)

        state_dict = self.state.to_dict()

        # Check dictionary contains expected keys
        expected_keys = [
            "state_dimensions",
            "observation_dimensions",
            "control_dimensions",
            "generative_model",
            "observation_history",
            "action_history",
        ]
        for key in expected_keys:
            self.assertIn(key, state_dict)

        # Create a new state from the dictionary
        new_state = ActiveInferenceState.from_dict(state_dict)

        # Check dimensions match
        self.assertEqual(new_state.state_dimensions, self.state.state_dimensions)
        self.assertEqual(
            new_state.observation_dimensions, self.state.observation_dimensions
        )
        self.assertEqual(new_state.control_dimensions, self.state.control_dimensions)

        # Check history lengths match
        self.assertEqual(
            len(new_state.observation_history), len(self.state.observation_history)
        )
        self.assertEqual(len(new_state.action_history), len(self.state.action_history))


class TestActiveInferenceAgent:
    """Test cases for the ActiveInferenceAgent class."""

    AGENT_ID = "test_agent"
    CONFIG = {
        "state_dimensions": 3,
        "observation_dimensions": 4,
        "control_dimensions": 2,
        "learning_rate": 0.1,
    }

    def setup_method(self):
        """Set up a fresh agent before each test."""
        self.agent = ActiveInferenceAgent(
            agent_id=self.AGENT_ID, config=dict(self.CONFIG)
        )

    def test_initialization(self):
        """Agent initializes with correct parameters; state is None until initialize()."""
        assert self.agent.agent_id == self.AGENT_ID
        assert self.agent.config["state_dimensions"] == self.CONFIG["state_dimensions"]
        assert (
            self.agent.config["observation_dimensions"]
            == self.CONFIG["observation_dimensions"]
        )
        assert (
            self.agent.config["control_dimensions"] == self.CONFIG["control_dimensions"]
        )
        assert self.agent.state is None

    async def test_initialize(self):
        """After initialize(), state is created and handlers are registered."""
        await self.agent.initialize()

        assert self.agent.state is not None
        assert isinstance(self.agent.state, ActiveInferenceState)
        assert len(self.agent._action_handlers) > 0
        assert len(self.agent._perception_handlers) > 0

    async def test_perception_cycle(self):
        """perceive() returns a dict (empty when no sensors registered)."""
        await self.agent.initialize()
        observations = await self.agent.perceive()
        assert isinstance(observations, dict)

    async def test_decision_cycle(self):
        """decide() returns an action dict with an action_type key."""
        await self.agent.initialize()
        obs = np.array([0.1, 0.2, 0.3, 0.4])
        self.agent.state.update_with_observation(obs)

        action = await self.agent.decide()
        assert action is not None
        assert "action_type" in action

    async def test_action_cycle(self):
        """act() dispatches a wait action and returns a result with status."""
        await self.agent.initialize()
        # Use the action format produced by decide() / expected by act()
        test_action = {
            "action_type": "wait",
            "action_id": "test_wait",
            "parameters": {"duration": 0.01},
        }
        result = await self.agent.act(test_action)
        assert result is not None
        assert "status" in result

    async def test_full_cycle(self):
        """Full perception→decision→action cycle completes without error."""
        await self.agent.initialize()

        observations = await self.agent.perceive()
        assert isinstance(observations, dict)

        obs = np.array([0.1, 0.2, 0.3, 0.4])
        self.agent.state.update_with_observation(obs)

        action = await self.agent.decide()
        assert action is not None
        assert "action_type" in action

        result = await self.agent.act(action)
        assert result is not None
        assert "status" in result

    async def test_action_handlers(self):
        """_handle_wait_action returns status='success'."""
        await self.agent.initialize()
        wait_action = {
            "action_type": "wait",
            "action_id": "test",
            "parameters": {"duration": 0.01},
        }
        result = await self.agent._handle_wait_action(self.agent, wait_action)
        assert "status" in result
        assert result["status"] == "success"

    async def test_shutdown(self):
        """shutdown() completes without error after initialization."""
        await self.agent.initialize()
        await self.agent.shutdown()


if __name__ == "__main__":
    unittest.main()
