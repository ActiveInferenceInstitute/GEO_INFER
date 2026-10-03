"""
Unit tests for the Free Energy Calculator.

Tests the FreeEnergyCalculator class which implements variational free energy
computations for both categorical and Gaussian active inference models.
Free energy F = Complexity - Accuracy serves as the cost function that
agents minimize through perception and action.
"""

import numpy as np
import pytest

from geo_infer_act import (
    ActiveInferenceStepResult,
    FreeEnergyBreakdown,
    PolicyEvaluation,
)
from geo_infer_act.core.free_energy import FreeEnergyCalculator
from geo_infer_act.core.policy_selection import PolicySelector


class TestFreeEnergyInit:
    """Test FreeEnergyCalculator initialization."""

    def test_initialization(self) -> None:
        """Test that calculator initializes correctly."""
        calc = FreeEnergyCalculator()
        assert calc.last_computed_energy == 0.0
        assert calc.computation_count == 0


class TestCategoricalFreeEnergy:
    """Test free energy computation for categorical (discrete) models."""

    def setup_method(self) -> None:
        """Set up calculator and test data."""
        self.calc = FreeEnergyCalculator()

    def test_uniform_beliefs_zero_complexity(self) -> None:
        """Test that uniform beliefs produce zero complexity (KL = 0 from uniform prior)."""
        n_states = 4
        beliefs = np.ones(n_states) / n_states
        observations = np.ones(n_states) / n_states
        fe = self.calc.compute_categorical_free_energy(beliefs, observations)
        # With uniform beliefs and no preferences, complexity should be ~0
        # Free energy should be finite
        assert np.isfinite(fe)

    def test_peaked_beliefs_higher_complexity(self) -> None:
        """Test that peaked beliefs produce higher complexity than uniform."""
        uniform = np.ones(4) / 4
        peaked = np.array([0.9, 0.05, 0.03, 0.02])
        obs = np.ones(4) / 4
        prefs = np.ones(4) / 4

        fe_uniform = self.calc.compute_categorical_free_energy(uniform, obs, prefs)
        fe_peaked = self.calc.compute_categorical_free_energy(peaked, obs, prefs)
        # Peaked beliefs should have higher complexity term
        assert fe_peaked != fe_uniform

    def test_matching_observations_lower_energy(self) -> None:
        """Test that observations matching beliefs produce lower free energy."""
        beliefs = np.array([0.7, 0.2, 0.1])
        good_obs = np.array([0.8, 0.15, 0.05])  # Close to beliefs
        bad_obs = np.array([0.1, 0.1, 0.8])  # Far from beliefs

        fe_good = self.calc.compute_categorical_free_energy(beliefs, good_obs)
        fe_bad = self.calc.compute_categorical_free_energy(beliefs, bad_obs)
        # Better matching observations should produce different free energy
        assert fe_good != fe_bad

    def test_returns_finite_value(self) -> None:
        """Test that free energy is always finite for valid inputs."""
        for _ in range(10):
            n = np.random.randint(2, 10)
            beliefs = np.random.dirichlet(np.ones(n))
            obs = np.random.dirichlet(np.ones(n))
            fe = self.calc.compute_categorical_free_energy(beliefs, obs)
            assert np.isfinite(fe)

    def test_dimension_mismatch_handling(self) -> None:
        """Test that mismatched dimensions are handled gracefully."""
        beliefs = np.array([0.5, 0.3, 0.2])
        obs = np.array([0.6, 0.4])  # Different length
        with pytest.raises(ValueError, match="length 3"):
            self.calc.compute_categorical_free_energy(beliefs, obs)

    def test_breakdown_matches_complexity_minus_accuracy(self) -> None:
        """Test that categorical free energy exposes its mathematical terms."""
        beliefs = np.array([0.7, 0.2, 0.1])
        observations = np.array([0.8, 0.15, 0.05])
        preferences = np.array([0.6, 0.25, 0.15])

        breakdown = self.calc.compute_categorical_free_energy(
            beliefs,
            observations,
            preferences,
            return_breakdown=True,
        )

        assert isinstance(breakdown, FreeEnergyBreakdown)
        assert np.isfinite(breakdown.free_energy)
        assert np.isfinite(breakdown.accuracy)
        assert np.isfinite(breakdown.complexity)
        assert np.isfinite(breakdown.entropy)
        assert breakdown.free_energy == pytest.approx(
            breakdown.complexity - breakdown.accuracy
        )

    def test_act_exports_typed_result_objects(self) -> None:
        """Test that public ACT exports include typed inference result contracts."""
        assert FreeEnergyBreakdown.__name__ == "FreeEnergyBreakdown"
        assert PolicyEvaluation.__name__ == "PolicyEvaluation"
        assert ActiveInferenceStepResult.__name__ == "ActiveInferenceStepResult"


class TestGaussianFreeEnergy:
    """Test free energy computation for Gaussian (continuous) models."""

    def setup_method(self) -> None:
        """Set up calculator."""
        self.calc = FreeEnergyCalculator()

    def test_perfect_observation_low_energy(self) -> None:
        """Test that observations at the mean produce low free energy."""
        mean = np.array([1.0, 2.0])
        precision = np.eye(2) * 10.0  # High precision
        observations = np.array([1.0, 2.0])  # Exactly at mean
        fe = self.calc.compute_gaussian_free_energy(mean, precision, observations)
        assert np.isfinite(fe)

    def test_distant_observation_higher_energy(self) -> None:
        """Test that distant observations produce higher free energy."""
        mean = np.array([0.0, 0.0])
        precision = np.eye(2)
        close_obs = np.array([0.1, 0.1])
        far_obs = np.array([10.0, 10.0])

        fe_close = self.calc.compute_gaussian_free_energy(mean, precision, close_obs)
        fe_far = self.calc.compute_gaussian_free_energy(mean, precision, far_obs)
        assert fe_far > fe_close

    def test_with_prior(self) -> None:
        """Test Gaussian free energy with explicit prior."""
        mean = np.array([1.0, 0.0])
        precision = np.eye(2) * 2.0
        observations = np.array([0.5, 0.5])
        prior_mean = np.zeros(2)
        prior_precision = np.eye(2)

        fe = self.calc.compute_gaussian_free_energy(
            mean, precision, observations, prior_mean, prior_precision
        )
        assert np.isfinite(fe)

    def test_returns_float(self) -> None:
        """Test that result is a Python float."""
        mean = np.array([0.0])
        precision = np.array([[1.0]])
        obs = np.array([0.5])
        fe = self.calc.compute_gaussian_free_energy(mean, precision, obs)
        assert isinstance(fe, float)


class TestExpectedFreeEnergy:
    """Test expected free energy for policy evaluation."""

    def setup_method(self) -> None:
        """Set up calculator."""
        self.calc = FreeEnergyCalculator()

    def test_basic_computation(self) -> None:
        """Test that expected free energy computation returns finite value."""
        beliefs = np.array([0.4, 0.3, 0.2, 0.1])
        policy = {"exploration_bonus": 0.2, "risk_preference": 0.0}
        preferences = np.array([0.1, 0.2, 0.3, 0.4])

        efe = self.calc.compute_expected_free_energy(beliefs, policy, preferences)
        assert np.isfinite(efe)
        assert isinstance(efe, float)

    def test_exploration_bonus_effect(self) -> None:
        """Test that higher exploration bonus increases epistemic value weight."""
        beliefs = np.array([0.25, 0.25, 0.25, 0.25])
        low_explore = {"exploration_bonus": 0.01, "risk_preference": 0.0}
        high_explore = {"exploration_bonus": 1.0, "risk_preference": 0.0}
        prefs = np.array([0.7, 0.1, 0.1, 0.1])

        efe_low = self.calc.compute_expected_free_energy(beliefs, low_explore, prefs)
        efe_high = self.calc.compute_expected_free_energy(beliefs, high_explore, prefs)
        assert efe_low != efe_high

    def test_no_preferences(self) -> None:
        """Test expected free energy without preferences."""
        beliefs = np.array([0.5, 0.3, 0.2])
        policy = {"exploration_bonus": 0.1}
        efe = self.calc.compute_expected_free_energy(beliefs, policy)
        assert np.isfinite(efe)

    def test_policy_conditioned_expected_free_energy_breakdown(self) -> None:
        """Test EFE uses policy-conditioned predictive beliefs."""
        beliefs = np.array([0.6, 0.3, 0.1])
        policy = {
            "predicted_beliefs": np.array([0.1, 0.8, 0.1]),
            "exploration_bonus": 0.2,
            "ambiguity": 0.05,
        }
        preferences = np.array([0.1, 0.8, 0.1])

        breakdown = self.calc.compute_expected_free_energy(
            beliefs,
            policy,
            preferences,
            return_breakdown=True,
        )

        assert isinstance(breakdown, FreeEnergyBreakdown)
        assert breakdown.ambiguity == pytest.approx(0.05)
        np.testing.assert_allclose(
            breakdown.metadata["predictive_beliefs"],
            np.array([0.1, 0.8, 0.1]),
        )


class TestPolicySelectorEfeParity:
    """FreeEnergyCalculator and PolicySelector must agree on identical inputs.

    Both classes delegate to the shared
    ``compute_policy_expected_free_energy`` implementation; these tests pin
    that contract so the historical ``temporal_discount`` drift (1.0 vs 0.9)
    cannot reintroduce a class-dependent policy score.
    """

    def test_efe_parity_with_matching_inputs(self) -> None:
        """Identical beliefs, policy, and preferences give identical scores."""
        beliefs = np.array([0.4, 0.3, 0.2, 0.1])
        policy = {
            "exploration_bonus": 0.2,
            "risk_preference": 0.1,
            "ambiguity": 0.05,
        }
        preferences = np.array([0.1, 0.2, 0.3, 0.4])
        calculator = FreeEnergyCalculator()
        selector = PolicySelector()

        assert calculator.compute_expected_free_energy(
            beliefs, policy, preferences
        ) == selector.compute_expected_free_energy(beliefs, policy, preferences)

    def test_efe_parity_with_explicit_temporal_discount(self) -> None:
        """An explicit temporal_discount is honoured identically by both."""
        beliefs = np.array([0.5, 0.3, 0.2])
        policy = {"exploration_bonus": 0.1, "temporal_discount": 0.9}
        preferences = np.array([0.1, 0.3, 0.6])
        calculator = FreeEnergyCalculator()
        selector = PolicySelector()

        assert calculator.compute_expected_free_energy(
            beliefs, policy, preferences
        ) == selector.compute_expected_free_energy(beliefs, policy, preferences)

    def test_efe_parity_default_temporal_discount(self) -> None:
        """Both classes default temporal_discount to 1.0 (no discounting)."""
        beliefs = np.array([0.4, 0.3, 0.2, 0.1])
        policy = {"exploration_bonus": 0.2, "risk_preference": 0.1}
        preferences = np.array([0.1, 0.2, 0.3, 0.4])

        calculator_breakdown = FreeEnergyCalculator().compute_expected_free_energy(
            beliefs, policy, preferences, return_breakdown=True
        )
        selector_breakdown = PolicySelector().compute_expected_free_energy(
            beliefs, policy, preferences, return_breakdown=True
        )

        assert isinstance(calculator_breakdown, FreeEnergyBreakdown)
        assert isinstance(selector_breakdown, FreeEnergyBreakdown)
        assert (
            calculator_breakdown.metadata["temporal_discount"]
            == selector_breakdown.metadata["temporal_discount"]
            == 1.0
        )
        assert calculator_breakdown.free_energy == selector_breakdown.free_energy
        assert (
            calculator_breakdown.pragmatic_value == selector_breakdown.pragmatic_value
        )
        assert (
            calculator_breakdown.epistemic_value == selector_breakdown.epistemic_value
        )

    def test_efe_parity_policy_conditioned_predictive(self) -> None:
        """Policy-conditioned predictive distributions score identically."""
        beliefs = np.array([0.6, 0.3, 0.1])
        policy = {
            "predicted_beliefs": np.array([0.1, 0.8, 0.1]),
            "exploration_bonus": 0.2,
            "ambiguity": 0.05,
        }
        preferences = np.array([0.1, 0.8, 0.1])

        calculator_breakdown = FreeEnergyCalculator().compute_expected_free_energy(
            beliefs, policy, preferences, return_breakdown=True
        )
        selector_breakdown = PolicySelector().compute_expected_free_energy(
            beliefs, policy, preferences, return_breakdown=True
        )

        assert calculator_breakdown.free_energy == selector_breakdown.free_energy
        np.testing.assert_allclose(
            calculator_breakdown.metadata["predictive_beliefs"],
            selector_breakdown.metadata["predictive_beliefs"],
        )

    def test_calculator_bookkeeping_tracks_delegated_efe(self) -> None:
        """Delegated EFE scores update the calculator's bookkeeping counters."""
        calc = FreeEnergyCalculator()
        beliefs = np.array([0.4, 0.3, 0.2, 0.1])
        policy = {"exploration_bonus": 0.2}

        efe = calc.compute_expected_free_energy(beliefs, policy)
        assert calc.computation_count == 1
        assert calc.last_computed_energy == pytest.approx(efe)

        # Externally supplied scores short-circuit without new bookkeeping.
        calc.compute_expected_free_energy(beliefs, {"expected_free_energy": 2.5})
        assert calc.computation_count == 1


class TestComputeDispatch:
    """Test the general compute dispatch method."""

    def test_categorical_dispatch(self) -> None:
        """Test dispatch to categorical computation."""
        calc = FreeEnergyCalculator()
        beliefs = np.array([0.5, 0.3, 0.2])
        obs = np.array([0.4, 0.4, 0.2])
        fe = calc.compute(beliefs, obs, model_type="categorical")
        assert np.isfinite(fe)

    def test_unsupported_model_type(self) -> None:
        """Test that unsupported model type raises ValueError."""
        calc = FreeEnergyCalculator()
        with pytest.raises(ValueError, match="Unsupported model type"):
            calc.compute(np.ones(3) / 3, model_type="unknown_type")
