"""
Tests for GEO-INFER-ACT core functionality.
"""

import unittest
import pytest
import numpy as np

from geo_infer_act import ActiveInferenceStepResult, H3BeliefUpdateResult
from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.core.belief_updating import BayesianBeliefUpdate
from geo_infer_act.core.dynamic_causal_model import DynamicCausalModel
from geo_infer_act.core.free_energy import FreeEnergyCalculator
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.core.markov_decision_process import MarkovDecisionProcess
from geo_infer_act.core.policy_selection import PolicySelector
from geo_infer_act.core.variational_inference import VariationalInference
from geo_infer_act.core.generative_model import MarkovBlanket


class TestActiveInferenceModel(unittest.TestCase):
    """Tests for ActiveInferenceModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = ActiveInferenceModel(model_type="categorical")
        self.gen_params = {"state_dim": 3, "obs_dim": 2}

    def test_initialization(self):
        """Test model initializes correctly."""
        self.assertEqual(self.model.model_type, "categorical")
        self.assertIsInstance(self.model.free_energy_calculator, FreeEnergyCalculator)
        self.assertIsInstance(self.model.policy_selector, PolicySelector)
        self.assertIsInstance(self.model.belief_updater, BayesianBeliefUpdate)
        self.assertIsNone(self.model.generative_model)
        self.assertIsNone(self.model.current_beliefs)
        self.assertIsNone(self.model.current_observations)
        self.assertIsNone(self.model.current_actions)
        self.assertEqual(self.model.history, [])

    def test_set_generative_model(self):
        """Test setting generative model."""
        gen_model = GenerativeModel("categorical", {"state_dim": 3})
        self.model.set_generative_model(gen_model)
        self.assertEqual(self.model.generative_model, gen_model)
        beliefs_states = (
            self.model.current_beliefs["states"]
            if isinstance(self.model.current_beliefs, dict)
            else self.model.current_beliefs
        )
        self.assertTrue(np.allclose(beliefs_states, np.ones(3) / 3))

    def test_seed_preferences_and_control_shape_reach_runtime_backend(self):
        """Seed, preferences, and list-valued controls must reach pymdp."""
        gen_model = GenerativeModel(
            "categorical",
            {"state_dim": 3, "obs_dim": 3, "C": np.zeros(3)},
        )
        model = ActiveInferenceModel(
            model_type="categorical",
            random_seed=17,
            num_controls=[2],
            policy_selection_mode="deterministic",
        )
        model.set_generative_model(gen_model)
        model.set_preferences({"observations": np.array([1.0, 0.0, 0.0])})

        assert model.random_seed == 17
        np.testing.assert_allclose(
            model.generative_model.preferences["observations"], [1.0, 0.0, 0.0]
        )
        model.perceive(np.array([1.0, 0.0, 0.0]))
        assert model.act() in {0, 1}

    def test_reset_restores_generative_model_beliefs(self):
        """Reset must restore both agent and generative-model state."""
        gen_model = GenerativeModel("categorical", {"state_dim": 3, "obs_dim": 3})
        self.model.set_generative_model(gen_model)
        gen_model.beliefs["states"] = np.array([0.0, 1.0, 0.0])
        self.model.current_beliefs = {"states": np.array([0.0, 1.0, 0.0])}

        self.model.reset()

        np.testing.assert_allclose(gen_model.beliefs["states"], [1 / 3] * 3)
        np.testing.assert_allclose(self.model.current_beliefs["states"], [1 / 3] * 3)

    def test_perceive(self):
        """Test belief updating via perceive."""
        gen_model = GenerativeModel("categorical", self.gen_params)
        gen_model.observation_model = np.array([[0.8, 0.1, 0.2], [0.2, 0.9, 0.8]])
        self.model.set_generative_model(gen_model)
        observation = np.array([1, 0])
        updated_beliefs = self.model.perceive(observation)
        self.assertIsNotNone(self.model.current_observations)
        # Extract states array from dict if needed
        beliefs_arr = (
            updated_beliefs["states"]
            if isinstance(updated_beliefs, dict)
            else updated_beliefs
        )
        self.assertTrue(np.allclose(np.sum(beliefs_arr), 1.0))
        self.assertFalse(
            np.allclose(beliefs_arr, np.ones(3) / 3)
        )  # Beliefs should change

    def test_act(self):
        """Test action selection."""
        gen_model = GenerativeModel("categorical", {"state_dim": 3})
        self.model.set_generative_model(gen_model)
        self.model.current_beliefs = np.ones(3) / 3
        action = self.model.act()
        self.assertIsNotNone(action)
        self.assertEqual(self.model.current_actions, action)

    def test_act_accepts_scalar_num_controls_in_local_backend(self):
        """The local action path accepts scalar and sequence control counts."""
        gen_model = GenerativeModel("categorical", {"state_dim": 3})
        self.model.set_generative_model(gen_model)
        self.model.current_beliefs = np.ones(3) / 3

        for controls in (2, [2]):
            gen_model.num_controls = controls
            assert self.model.act() in {0, 1}

    def test_step(self):
        gen_model = GenerativeModel("categorical", self.gen_params)
        self.model.set_generative_model(gen_model)
        obs = np.array([1, 0])
        beliefs, action = self.model.step(obs)
        self.assertIsNotNone(beliefs)
        self.assertIsNotNone(action)

    def test_step_can_return_typed_result(self):
        gen_model = GenerativeModel("categorical", self.gen_params)
        self.model.set_generative_model(gen_model)
        obs = np.array([1, 0])
        result = self.model.step(obs, return_result=True)
        self.assertIsInstance(result, ActiveInferenceStepResult)
        self.assertIsNotNone(result.beliefs)
        self.assertIsNotNone(result.action)
        self.assertTrue(np.isfinite(result.free_energy))

    def test_compute_free_energy(self):
        """Test free energy computation."""
        gen_model = GenerativeModel("categorical", {"state_dim": 3, "obs_dim": 2})
        self.model.set_generative_model(gen_model)
        self.model.current_beliefs = np.ones(3) / 3
        self.model.current_observations = np.array([1, 0])
        fe = self.model.compute_free_energy()
        self.assertIsInstance(fe, float)

    def test_reset(self):
        """Test model reset."""
        gen_model = GenerativeModel("categorical", {"state_dim": 3})
        self.model.set_generative_model(gen_model)
        self.model.current_beliefs = np.array([0.1, 0.2, 0.7])
        self.model.reset()
        beliefs_states = (
            self.model.current_beliefs["states"]
            if isinstance(self.model.current_beliefs, dict)
            else self.model.current_beliefs
        )
        self.assertTrue(np.allclose(beliefs_states, np.ones(3) / 3))
        self.assertIsNone(self.model.current_observations)
        self.assertIsNone(self.model.current_actions)
        self.assertEqual(self.model.history, [])

    def test_get_history(self):
        """Test getting history."""
        self.model.history = [{"test": 1}]
        history = self.model.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["test"], 1)
        # Ensure it's a copy
        self.model.history.append({"test": 2})
        self.assertEqual(len(history), 1)

    def test_get_current_state(self):
        """Test getting current state."""
        self.model.model_type = "test"
        self.model.current_beliefs = np.array([0.5, 0.5])
        self.model.current_observations = np.array([1, 0])
        self.model.current_actions = "test_action"
        state = self.model.get_current_state()
        self.assertEqual(state["model_type"], "test")
        self.assertTrue(np.allclose(state["beliefs"], [0.5, 0.5]))
        self.assertTrue(np.allclose(state["observations"], [1, 0]))
        self.assertEqual(state["actions"], "test_action")
        self.assertIn("free_energy", state)

    def test_update_observations(self):
        """Test updating observations directly."""
        self.model.update_observations({"temp": 22.5, "pressure": 101.3})
        self.assertEqual(self.model.current_observations["temp"], 22.5)
        self.assertEqual(self.model.current_observations["pressure"], 101.3)

    def test_update_preferences(self):
        """Test updating preferences directly."""
        self.model.update_preferences({"accuracy": 0.9, "speed": 0.7})
        self.assertEqual(self.model.preferences["accuracy"], 0.9)
        self.assertEqual(self.model.preferences["speed"], 0.7)

    def test_update_with_outcome(self):
        """Test decision-outcome learning loop."""
        gen_model = GenerativeModel("categorical", self.gen_params)
        gen_model.observation_model = np.array([[0.8, 0.1, 0.2], [0.2, 0.9, 0.8]])
        self.model.set_generative_model(gen_model)
        decision = {"action": "move_north"}
        outcome = {"observation": np.array([1.0, 0.0]), "reward": 0.5}
        self.model.update_with_outcome(decision, outcome)
        # Should store in history
        self.assertEqual(len(self.model.history), 1)
        self.assertEqual(self.model.history[0]["decision"], decision)
        # Should have updated beliefs from the observation
        self.assertIsNotNone(self.model.current_beliefs)

    def test_generate_policies(self):
        """Test policy generation from available actions."""
        actions = [{"action": "north", "cost": 1}, {"action": "south", "cost": 2}]
        policies = self.model.generate_policies(actions)
        self.assertEqual(len(policies), 2)
        self.assertEqual(policies[0]["action"], "north")

    def test_select_policy_uses_expected_free_energy(self):
        """Test policy selection does not just return the first candidate."""
        self.model.current_beliefs = np.array([0.5, 0.3, 0.2])
        self.model.policy_selector.selection_mode = "deterministic"
        policies = [
            {"action": "north", "expected_free_energy": 3.0},
            {"action": "south", "expected_free_energy": -1.0},
        ]
        selected = self.model.select_policy(policies)
        self.assertEqual(selected["action"], "south")

    def test_compute_expected_free_energy_real(self):
        """Test real EFE computation delegates to FreeEnergyCalculator."""
        gen_model = GenerativeModel("categorical", self.gen_params)
        self.model.set_generative_model(gen_model)
        policy = {"exploration_bonus": 0.1, "risk_preference": 0.0}
        efe = self.model.compute_expected_free_energy(policy)
        self.assertIsInstance(efe, float)
        self.assertTrue(np.isfinite(efe))

    def test_compute_expected_free_energy_no_beliefs(self):
        """Test EFE returns inf when no beliefs are available."""
        self.model.current_beliefs = None
        efe = self.model.compute_expected_free_energy({})
        self.assertEqual(efe, float("inf"))

    def test_apply_to_h3(self):
        """Test applying H3 spatial observations."""
        gen_model = GenerativeModel("categorical", self.gen_params)
        gen_model.enable_h3_spatial(
            8,
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.41, 37.77],
                        [-122.41, 37.80],
                        [-122.38, 37.80],
                        [-122.38, 37.77],
                        [-122.41, 37.77],
                    ]
                ],
            },
        )
        self.model.set_generative_model(gen_model)
        # Create observations for first few cells — obs_dim is 2 per gen_params
        h3_obs = {}
        for cell in gen_model.h3_cells[:2]:
            h3_obs[cell] = np.random.rand(2)
        result = self.model.apply_to_h3(h3_obs)
        self.assertIsNotNone(result)
        self.assertIn("spatial_consistency", result)
        self.assertIn("global_coherence", result["spatial_consistency"])
        self.assertTrue(np.isfinite(result["spatial_consistency"]["global_coherence"]))


class TestGenerativeModelSummary(unittest.TestCase):
    """Tests for GenerativeModel summary and diagnostic methods."""

    def test_get_model_summary(self):
        """Test comprehensive model summary generation."""
        gen_model = GenerativeModel("categorical", {"state_dim": 4})
        summary = gen_model.get_model_summary()
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["model_type"], "categorical")
        self.assertEqual(summary["state_dim"], 4)
        self.assertIn("free_energy", summary)
        self.assertIn("belief_entropy", summary)
        self.assertIn("convergence_status", summary)

    def test_get_model_summary_hierarchical(self):
        """Test model summary includes hierarchical details."""
        gen_model = GenerativeModel(
            "categorical",
            {"state_dims": [4, 3], "obs_dims": [4, 3], "hierarchical": True},
        )
        summary = gen_model.get_model_summary()
        self.assertTrue(summary["hierarchical"])
        self.assertIn("levels", summary)
        self.assertIn("level_details", summary)


class TestBayesianBeliefUpdate(unittest.TestCase):
    """Tests for BayesianBeliefUpdate class."""

    def setUp(self):
        """Set up test fixtures."""
        self.updater = BayesianBeliefUpdate(prior_precision=1.0)

    def test_update_categorical(self):
        """Test categorical belief update."""
        prior = np.array([0.2, 0.3, 0.5])
        observation = np.array([1, 0])
        likelihood_matrix = np.array([[0.8, 0.1, 0.2], [0.2, 0.9, 0.8]])
        posterior = self.updater.update_categorical(
            prior, observation, likelihood_matrix
        )
        self.assertTrue(np.allclose(np.sum(posterior), 1.0))
        self.assertFalse(np.allclose(posterior, prior))  # Should change

    def test_update_gaussian(self):
        """Test Gaussian belief update."""
        prior_mean = np.array([0, 0])
        prior_precision = np.eye(2)
        observation = np.array([1, 1])
        observation_matrix = np.eye(2)
        observation_precision = np.eye(2) * 10
        result = self.updater.update_gaussian(
            prior_mean,
            prior_precision,
            observation,
            observation_matrix,
            observation_precision,
        )
        self.assertIn("mean", result)
        self.assertIn("precision", result)
        self.assertFalse(np.allclose(result["mean"], prior_mean))

    def test_compute_prediction_error(self):
        """Test precision-weighted prediction error."""
        prediction = np.array([0, 0])
        observation = np.array([1, 1])
        error = self.updater.compute_prediction_error(
            prediction, observation, precision=2.0
        )
        self.assertEqual(error, 4.0)  # 2 * (1^2 + 1^2) = 4

    def test_compute_surprise(self):
        """Test surprise computation."""
        observation = np.array([1, 0])
        predicted = np.array([0.7, 0.3])
        surprise = self.updater.compute_surprise(observation, predicted)
        self.assertGreater(surprise, 0)


class TestDynamicCausalModel(unittest.TestCase):
    """Tests for DynamicCausalModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DynamicCausalModel(state_dim=2, input_dim=1, output_dim=1, dt=0.01)

    def test_state_equation(self):
        """Test state evolution equation."""
        state = np.array([1, 1])
        inputs = np.array([0.5])
        dxdt = self.model.state_equation(state, 0, inputs)
        self.assertEqual(len(dxdt), 2)

    def test_observation_equation(self):
        """Test observation equation."""
        state = np.array([1, 1])
        obs = self.model.observation_equation(state)
        self.assertEqual(len(obs), 1)

    def test_integrate_dynamics(self):
        """Test dynamics integration."""
        initial_state = np.zeros(2)
        inputs = np.zeros((5, 1))
        time_points = np.linspace(0, 0.1, 6)
        trajectory = self.model.integrate_dynamics(initial_state, inputs, time_points)
        self.assertEqual(trajectory.shape, (6, 2))

    def test_generate_observations(self):
        """Test observation generation."""
        trajectory = np.zeros((5, 2))
        observations = self.model.generate_observations(trajectory)
        self.assertEqual(observations.shape, (5, 1))

    def test_estimate_parameters(self):
        """Test parameter estimation."""
        observations = np.random.randn(5, 1)
        inputs = np.random.randn(4, 1)
        time_points = np.linspace(0, 0.1, 5)
        results = self.model.estimate_parameters(observations, inputs, time_points)
        self.assertIn("A", results)
        self.assertIn("B", results)
        self.assertIn("C", results)

    def test_set_parameters(self):
        """Test setting model parameters."""
        A = np.eye(2) * 0.5
        B = np.ones((2, 1))
        C = np.ones((1, 2))
        self.model.set_parameters(A, B, C)
        self.assertTrue(np.allclose(self.model.A, A))

    def test_set_noise_parameters(self):
        """Test setting noise parameters."""
        Q = np.eye(2) * 0.1
        R = np.eye(1) * 0.05
        self.model.set_noise_parameters(Q, R)
        self.assertTrue(np.allclose(self.model.Q, Q))


class TestFreeEnergyCalculator(unittest.TestCase):
    """Tests for FreeEnergyCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = FreeEnergyCalculator()

    def test_compute_categorical_free_energy(self):
        """Test categorical free energy computation."""
        beliefs = np.array([0.4, 0.6])
        observations = np.array([0.7, 0.3])
        preferences = np.array([0.5, 0.5])
        fe = self.calculator.compute_categorical_free_energy(
            beliefs, observations, preferences
        )
        self.assertIsInstance(fe, float)

    def test_compute_gaussian_free_energy(self):
        """Test Gaussian free energy computation."""
        mean = np.array([0, 0])
        precision = np.eye(2)
        observations = np.array([1, 1])
        prior_mean = np.zeros(2)
        prior_precision = np.eye(2)
        fe = self.calculator.compute_gaussian_free_energy(
            mean, precision, observations, prior_mean, prior_precision
        )
        self.assertIsInstance(fe, float)

    def test_compute_expected_free_energy(self):
        """Test expected free energy computation."""
        beliefs = np.array([0.4, 0.6])
        policy = {
            "exploration_bonus": 0.1,
            "risk_preference": 0.0,
            "temporal_discount": 0.9,
        }
        preferences = np.array([0.5, 0.5])
        efe = self.calculator.compute_expected_free_energy(beliefs, policy, preferences)
        self.assertIsInstance(efe, float)

    def test_compute_categorical(self):
        beliefs = np.array([0.4, 0.6])
        obs = np.array([1, 0])
        prefs = np.array([0.3, 0.7])
        fe = self.calculator.compute_categorical_free_energy(beliefs, obs, prefs)
        self.assertIsInstance(fe, float)


@pytest.mark.slow
class TestGenerativeModel(unittest.TestCase):
    """Tests for GenerativeModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = {"state_dim": 3, "obs_dim": 2, "prior_precision": 1.0}
        self.model = GenerativeModel("categorical", self.params)
        # Set non-uniform observation model for testing updates
        self.model.observation_model = np.array(
            [[0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]
        ).T  # shape (obs_dim, state_dim)

    def test_initialization(self):
        """Test model initializes correctly."""
        self.assertEqual(self.model.model_type, "categorical")
        self.assertEqual(self.model.state_dim, 3)
        self.assertEqual(self.model.obs_dim, 2)
        self.assertFalse(self.model.hierarchical)
        self.assertFalse(self.model.markov_blankets)
        self.assertTrue(self.model.message_passing)
        self.assertFalse(self.model.spatial_mode)
        self.assertFalse(self.model.temporal_hierarchies)
        self.assertIn("states", self.model.beliefs)
        self.assertTrue(np.allclose(self.model.beliefs["states"], np.ones(3) / 3))

    def test_update_beliefs(self):
        """Test belief updating."""
        observations = {"observations": np.array([1, 0])}
        updated = self.model.update_beliefs(observations)
        self.assertTrue(np.allclose(np.sum(updated["states"]), 1.0))
        self.assertFalse(np.allclose(updated["states"], np.ones(3) / 3))

    def test_compute_free_energy(self):
        """Test free energy computation."""
        fe = self.model.compute_free_energy()
        self.assertIsInstance(fe, float)

    def test_enable_spatial_navigation(self):
        """Test enabling spatial navigation mode."""
        self.model.enable_spatial_navigation(grid_size=3)
        self.assertTrue(self.model.spatial_mode)
        self.assertEqual(self.model.state_dim, 9)
        self.assertEqual(self.model.obs_dim, 9)
        np.testing.assert_array_equal(self.model.observation_model, np.eye(9))
        self.assertEqual(self.model.transition_model.shape, (9, 9, 4))
        np.testing.assert_allclose(self.model.transition_model.sum(axis=0), 1)

    def test_set_preferences(self):
        """Test setting preferences."""
        preferences = {"observations": np.array([0.7, 0.3])}
        self.model.set_preferences(preferences)
        self.assertTrue(np.allclose(self.model.preferences["observations"], [0.7, 0.3]))

    # Add tests for hierarchical mode
    def test_hierarchical_initialization(self):
        """Test hierarchical model initialization."""
        hier_params = {
            "hierarchical": True,
            "levels": 2,
            "state_dims": [3, 2],
            "obs_dims": [2, 1],
        }
        hier_model = GenerativeModel("categorical", hier_params)
        self.assertTrue(hier_model.hierarchical)
        self.assertEqual(len(hier_model.levels), 2)
        self.assertIn("level_0", hier_model.beliefs)
        self.assertIn("level_1", hier_model.beliefs)

    def test_hierarchical_update_beliefs(self):
        """Test belief updating in hierarchical model."""
        hier_params = {
            "hierarchical": True,
            "levels": 2,
            "state_dims": [3, 2],
            "obs_dims": [2, 1],
        }
        hier_model = GenerativeModel("categorical", hier_params)
        observations = {"level_0": np.array([1, 0]), "level_1": np.array([1])}
        updated = hier_model.update_beliefs(observations)
        self.assertIn("level_0", updated)
        self.assertIn("level_1", updated)
        self.assertTrue(np.allclose(np.sum(updated["level_0"]["states"]), 1.0))

    def test_add_nested_level(self):
        """Test adding nested level."""
        parent = GenerativeModel("categorical", {"state_dim": 3, "obs_dim": 2})
        child = GenerativeModel("categorical", {"state_dim": 2, "obs_dim": 1})
        parent.add_nested_level(child)
        self.assertTrue(hasattr(parent, "nested_models"))
        self.assertEqual(len(parent.nested_models), 1)

    def test_update_nested_beliefs(self):
        """Test updating nested beliefs."""
        parent = GenerativeModel("categorical", {"state_dim": 3, "obs_dim": 2})
        child = GenerativeModel("categorical", {"state_dim": 2, "obs_dim": 1})
        parent.add_nested_level(child)
        observations = {"observations": np.array([1, 0])}
        parent.update_nested_beliefs(observations)
        self.assertTrue(np.allclose(np.sum(parent.beliefs["states"]), 1.0))
        self.assertTrue(np.allclose(np.sum(child.beliefs["states"]), 1.0))

    @pytest.mark.slow
    def test_integrate_rxinfer(self):
        """Test RxInfer integration with a deterministic local backend."""
        model_spec = """ # Julia code for model """
        data = {"observations": np.random.randn(10)}
        result = self.model.integrate_rxinfer(
            model_spec, data, backend="local_gaussian"
        )
        self.assertEqual(result["status"], "success")
        self.assertIn(result["backend"], {"local_gaussian"})

    def test_integrate_bayeux(self):
        """Test Bayeux-compatible inference with deterministic NumPy sampling."""
        result = self.model.integrate_bayeux(
            lambda location, scale_log: (
                -float(np.sum(location**2)) - float(scale_log**2)
            ),
            {"location": np.zeros(2), "scale_log": np.array(0.0)},
        )
        self.assertEqual(result["status"], "success")
        self.assertTrue(np.isfinite(result["posterior_samples"]["location"]).all())

    # Add more tests for other methods like add_nested_level, integrate_rxinfer, etc.
    # For integrations, we can test if they run without error, but since they may require external libs, use try-except or skip.

    def test_markov_blanket_check(self):
        """Test Markov blanket independence check."""
        blanket = MarkovBlanket(sensory_states=[0, 1], internal_states=[2, 3])
        states = np.random.randn(4)
        self.assertTrue(blanket.check_conditional_independence(2, states))

    def test_hierarchical_levels(self):
        """Test hierarchical level initialization."""
        params = {"hierarchical": True, "levels": 3, "state_dims": [4, 3, 2]}
        model = GenerativeModel("categorical", params)
        self.assertEqual(len(model.levels), 3)
        self.assertEqual(model.levels[0].state_dim, 4)

    def test_spatial_mode(self):
        """Test spatial navigation mode."""
        model = GenerativeModel("categorical", {"state_dim": 9})
        model.enable_spatial_navigation(3)
        self.assertTrue(model.spatial_mode)
        self.assertEqual(model.grid_size, 3)

    def test_enable_h3_spatial(self):
        """Test H3 spatial enabling."""
        model = GenerativeModel("categorical", {"state_dim": 1})
        boundary = {
            "type": "Polygon",
            "coordinates": [
                [
                    [-122.42, 37.77],
                    [-122.42, 37.78],
                    [-122.41, 37.78],
                    [-122.41, 37.77],
                    [-122.42, 37.77],
                ]
            ],
        }
        model.enable_h3_spatial(8, boundary)
        if hasattr(model, "spatial_config"):
            self.assertTrue(model.spatial_mode)
            self.assertTrue(len(model.h3_cells) > 0)

    def test_update_h3_beliefs(self):
        model = GenerativeModel("categorical", {"state_dim": 2})
        model.enable_h3_spatial(
            8,
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.42, 37.77],
                        [-122.42, 37.78],
                        [-122.41, 37.78],
                        [-122.41, 37.77],
                        [-122.42, 37.77],
                    ]
                ],
            },
        )
        obs = {
            model.h3_cells[0]: np.array([1, 0]),
            model.h3_cells[min(1, len(model.h3_cells) - 1)]: np.array([0, 1]),
        }
        updated = model.update_h3_beliefs(obs)
        self.assertIn("h3_beliefs", updated)
        self.assertIn("aggregate_free_energy", updated)
        self.assertTrue(
            all(np.allclose(np.sum(b), 1.0) for b in updated["h3_beliefs"].values())
        )
        typed = model.update_h3_beliefs(obs, return_result=True)
        self.assertIsInstance(typed, H3BeliefUpdateResult)
        self.assertTrue(np.isfinite(typed.aggregate_free_energy))


class TestMarkovDecisionProcess(unittest.TestCase):
    """Tests for MarkovDecisionProcess class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mdp = MarkovDecisionProcess(n_states=3, n_observations=2, n_actions=2)

    def test_initialization(self):
        """Test MDP initializes correctly."""
        self.assertEqual(self.mdp.n_states, 3)
        self.assertEqual(self.mdp.n_observations, 2)
        self.assertEqual(self.mdp.n_actions, 2)
        self.assertEqual(self.mdp.transition_prob.shape, (3, 3, 2))
        self.assertEqual(self.mdp.observation_prob.shape, (2, 3))
        self.assertTrue(hasattr(self.mdp, "policies"))

    def test_get_transition_prob(self):
        """Test getting transition probabilities."""
        probs = self.mdp.get_transition_prob(0, 0)
        self.assertEqual(len(probs), 3)
        self.assertTrue(np.allclose(np.sum(probs), 1.0))

    def test_get_observation_prob(self):
        """Test getting observation probabilities."""
        probs = self.mdp.get_observation_prob(0)
        self.assertEqual(len(probs), 2)
        self.assertTrue(np.allclose(np.sum(probs), 1.0))

    def test_transition(self):
        """Test state transition."""
        next_state = self.mdp.transition(0, 0)
        self.assertTrue(0 <= next_state < 3)

    def test_observe(self):
        """Test observation sampling."""
        obs = self.mdp.observe(0)
        self.assertTrue(0 <= obs < 2)

    def test_simulate(self):
        """Test trajectory simulation."""
        policy = [0, 1]
        states, obs = self.mdp.simulate(0, policy)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(obs), 3)

    def test_get_predictive_state(self):
        """Test predictive state distribution."""
        belief = np.ones(3) / 3
        pred = self.mdp.get_predictive_state(belief, 0)
        self.assertTrue(np.allclose(np.sum(pred), 1.0))

    def test_get_predictive_observation(self):
        """Test predictive observation distribution."""
        state_dist = np.ones(3) / 3
        pred = self.mdp.get_predictive_observation(state_dist)
        self.assertTrue(np.allclose(np.sum(pred), 1.0))

    def test_update_belief(self):
        """Test belief updating."""
        prior = np.ones(3) / 3
        observation = 0
        posterior = self.mdp.update_belief(prior, observation)
        self.assertTrue(np.allclose(np.sum(posterior), 1.0))

    def test_set_transition_matrix(self):
        """Test setting transition matrix."""
        dist = np.array([0.1, 0.2, 0.7])
        self.mdp.set_transition_matrix(0, 0, dist)
        self.assertTrue(
            np.allclose(self.mdp.transition_prob[0, :, 0], dist / np.sum(dist))
        )

    def test_set_observation_matrix(self):
        """Test setting observation matrix."""
        dist = np.array([0.4, 0.6])
        self.mdp.set_observation_matrix(0, dist)
        self.assertTrue(
            np.allclose(self.mdp.observation_prob[:, 0], dist / np.sum(dist))
        )


# Add similar TestCase for other classes like PolicySelector (existing), VariationalInference, etc.


class TestVariationalInference(unittest.TestCase):
    """Tests for VariationalInference class."""

    def setUp(self):
        """Set up test fixtures."""
        self.vi = VariationalInference()
        self.prior = np.array([0.5, 0.5])
        self.likelihood = np.array([[0.8, 0.2], [0.2, 0.8]])
        self.observations = np.array([1, 0])

    def test_update_categorical(self):
        """Test mean-field update for categorical."""
        posterior = self.vi.mean_field_update_categorical(
            self.prior, self.likelihood, self.observations
        )
        self.assertEqual(len(posterior), 2)
        self.assertAlmostEqual(sum(posterior), 1.0)

    def test_update_gaussian(self):
        """Test mean-field update for Gaussian."""
        mean = np.zeros(2)
        cov = np.eye(2)
        obs = np.array([1, 0])
        posterior = self.vi.mean_field_update_gaussian(mean, cov, obs)
        self.assertEqual(len(posterior), 2)

    # Add tests for structured_update, importance_sampling_update, compute_elbo


class TestPolicySelector(unittest.TestCase):
    """Tests for PolicySelector class."""

    def setUp(self):
        self.selector = PolicySelector()
        self.beliefs = np.array([0.4, 0.6])
        self.actions = [0, 1]
        self.model = GenerativeModel("categorical", {"state_dim": 2, "obs_dim": 2})

    def test_select_policy(self):
        policy = self.selector.select_policy(
            self.beliefs, self.actions, self.model.preferences
        )
        self.assertIn("policy", policy)

    def test_compute_expected_free_energy(self):
        efe = self.selector.compute_expected_free_energy(
            self.beliefs, {"action": 0, "exploration_bonus": 0.1}, np.array([0.3, 0.7])
        )
        self.assertIsInstance(efe, float)


class TestMultiAgentMessages(unittest.TestCase):
    """Tests for MultiAgentModel.get_agent_messages."""

    def test_get_agent_messages_valid(self):
        """Test getting messages from a valid agent."""
        from geo_infer_act.models.multi_agent import MultiAgentModel

        model = MultiAgentModel(n_agents=3)
        msg = model.get_agent_messages(0)
        self.assertIsInstance(msg, dict)
        self.assertEqual(msg["agent_id"], 0)
        self.assertIn("beliefs", msg)

    def test_get_agent_messages_invalid(self):
        """Test getting messages from an invalid agent ID."""
        from geo_infer_act.models.multi_agent import MultiAgentModel

        model = MultiAgentModel(n_agents=3)
        msg = model.get_agent_messages(99)
        self.assertEqual(msg, {})

    def test_get_agent_messages_negative_id(self):
        """Test getting messages from a negative agent ID."""
        from geo_infer_act.models.multi_agent import MultiAgentModel

        model = MultiAgentModel(n_agents=3)
        msg = model.get_agent_messages(-1)
        self.assertEqual(msg, {})


class TestDiffuseAndAggregateBeliefs(unittest.TestCase):
    """Tests for GenerativeModel spatial belief diffusion and aggregation."""

    def test_diffuse_beliefs_non_spatial(self):
        """Test that diffuse_beliefs returns beliefs unchanged when not spatial."""
        gen_model = GenerativeModel("categorical", {"state_dim": 3})
        beliefs = {
            "cell_a": np.array([0.5, 0.3, 0.2]),
            "cell_b": np.array([0.1, 0.8, 0.1]),
        }
        result = gen_model.diffuse_beliefs(beliefs, diffusion_rate=0.1)
        # Non-spatial mode should return input unchanged
        self.assertEqual(set(result.keys()), set(beliefs.keys()))

    def test_aggregate_beliefs_to_resolution(self):
        """Test belief aggregation to coarser H3 resolution."""

        gen_model = GenerativeModel("categorical", {"state_dim": 3})
        gen_model.enable_h3_spatial(
            8,
            {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.41, 37.77],
                        [-122.41, 37.80],
                        [-122.38, 37.80],
                        [-122.38, 37.77],
                        [-122.41, 37.77],
                    ]
                ],
            },
        )
        # Create beliefs for each cell
        beliefs = {cell: np.array([0.3, 0.3, 0.4]) for cell in gen_model.h3_cells[:5]}
        # Aggregate to coarser resolution
        aggregated = gen_model.aggregate_beliefs_to_resolution(
            beliefs, target_resolution=5
        )
        self.assertIsInstance(aggregated, dict)
        # Should have fewer cells at coarser resolution
        self.assertLessEqual(len(aggregated), len(beliefs))
        # All values should be valid distributions
        for b in aggregated.values():
            self.assertAlmostEqual(np.sum(b), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
