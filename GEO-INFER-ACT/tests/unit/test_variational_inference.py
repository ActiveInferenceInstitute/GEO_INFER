"""
Unit tests for the Variational Inference module.

Tests the VariationalInference class which implements variational inference
algorithms for belief updating in active inference models, including
mean-field updates, structured inference, and ELBO computation.
"""

import numpy as np
import pytest

from geo_infer_act.core.variational_inference import VariationalInference


class TestVariationalInferenceInit:
    """Test VariationalInference initialization."""

    def test_default_parameters(self) -> None:
        """Test default initialization parameters."""
        vi = VariationalInference()
        assert vi.max_iterations == 100
        assert vi.tolerance == 1e-6

    def test_custom_parameters(self) -> None:
        """Test custom initialization parameters."""
        vi = VariationalInference(max_iterations=50, tolerance=1e-4)
        assert vi.max_iterations == 50
        assert vi.tolerance == 1e-4


class TestMeanFieldUpdate:
    """Test mean-field variational inference updates."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.vi = VariationalInference()

    def test_dirichlet_categorical_update(self) -> None:
        """Test Dirichlet-categorical conjugate update."""
        prior = {"concentration": np.array([1.0, 1.0, 1.0])}
        likelihood = {}
        observations = np.array([3.0, 1.0, 0.0])
        result = self.vi.mean_field_update(prior, likelihood, observations)
        assert "concentration" in result
        assert "mean" in result
        assert "precision" in result
        # Concentration should be prior + observations
        np.testing.assert_array_equal(
            result["concentration"], np.array([4.0, 2.0, 1.0])
        )
        # Mean should be normalized
        np.testing.assert_allclose(result["mean"].sum(), 1.0, atol=1e-10)

    def test_gaussian_update(self) -> None:
        """Test Gaussian conjugate update."""
        dim = 3
        prior = {"mean": np.zeros(dim), "precision": np.eye(dim)}
        likelihood = {
            "precision": np.eye(dim) * 10.0  # High-precision observation
        }
        observations = np.array([1.0, 2.0, 3.0])
        result = self.vi.mean_field_update(prior, likelihood, observations)
        assert "mean" in result
        assert "precision" in result
        assert "covariance" in result
        # Posterior mean should shift toward observations
        assert np.linalg.norm(result["mean"] - observations) < np.linalg.norm(
            observations
        )

    def test_categorical_shorthand(self) -> None:
        """Test the categorical update shorthand method."""
        prior = np.array([1.0, 1.0, 1.0])
        likelihood = np.ones((3, 3))
        obs = np.array([2.0, 1.0, 0.0])
        result = self.vi.mean_field_update_categorical(prior, likelihood, obs)
        assert result.shape == (3,)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-10)

    def test_gaussian_shorthand(self) -> None:
        """Test the Gaussian update shorthand method."""
        mean = np.array([0.0, 0.0])
        cov = np.eye(2)
        obs = np.array([1.0, 2.0])
        result = self.vi.mean_field_update_gaussian(mean, cov, obs)
        assert result.shape == (2,)

    def test_unsupported_prior_is_rejected(self) -> None:
        """Unknown prior families must not fabricate a posterior update."""
        with pytest.raises(ValueError, match="Unsupported prior"):
            self.vi.mean_field_update(
                {"some_key": np.array([1.0, 2.0])}, {}, np.zeros(2)
            )


class TestStructuredInference:
    """Test structured variational inference with factor graphs."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.vi = VariationalInference(max_iterations=50)

    def test_belief_propagation(self) -> None:
        """Test belief propagation on a simple factor graph."""
        factor_graph = {
            "variables": {
                "x": {"dimension": 3},
                "y": {"dimension": 3},
            },
            "factors": {},
        }
        observations = {"y": np.array([0.0, 1.0, 0.0])}
        result = self.vi.structured_update(
            factor_graph, observations, method="belief_propagation"
        )
        assert "x" in result
        assert "y" in result
        # Observed variable should be clamped
        np.testing.assert_array_equal(result["y"], observations["y"])
        # Unobserved variable should be normalized
        np.testing.assert_allclose(result["x"].sum(), 1.0, atol=1e-6)

    def test_structured_mean_field(self) -> None:
        """Test structured mean-field variational inference."""
        factor_graph = {
            "variables": {
                "z1": {"dimension": 4},
                "z2": {"dimension": 4},
            },
            "factors": {},
        }
        observations = {}
        result = self.vi.structured_update(
            factor_graph, observations, method="mean_field"
        )
        assert "z1" in result
        assert "z2" in result
        # Both should be valid probability distributions
        np.testing.assert_allclose(result["z1"].sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(result["z2"].sum(), 1.0, atol=1e-6)

    def test_invalid_method_raises(self) -> None:
        """Test that invalid inference method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown inference method"):
            self.vi.structured_update({}, {}, method="gibbs")


class TestImportanceSampling:
    """Test importance sampling for posterior approximation."""

    def test_importance_sampling_gaussian(self) -> None:
        """Test importance sampling with Gaussian prior."""
        vi = VariationalInference()
        prior = {"mean": np.zeros(2), "covariance": np.eye(2)}

        def likelihood_fn(sample: np.ndarray, obs: np.ndarray) -> float:
            return float(np.exp(-0.5 * np.sum((sample - obs) ** 2)))

        observations = np.array([1.0, 1.0])
        result = vi.importance_sampling_update(
            prior, likelihood_fn, observations, n_samples=500
        )
        assert "mean" in result
        assert "covariance" in result
        assert "precision" in result
        assert "samples" in result
        assert "weights" in result
        assert result["samples"].shape == (500, 2)
        # Posterior mean should be shifted toward observations
        assert np.linalg.norm(result["mean"] - observations) < np.linalg.norm(
            observations
        )


class TestELBO:
    """Test Evidence Lower Bound computation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.vi = VariationalInference()

    def test_gaussian_elbo(self) -> None:
        """Test ELBO computation for Gaussian distributions."""
        posterior = {"mean": np.array([1.0, 0.5]), "precision": np.eye(2) * 2.0}
        prior = {"mean": np.zeros(2), "precision": np.eye(2)}
        likelihood = {"precision": np.eye(2) * 10.0}
        observations = np.array([1.0, 0.5])
        elbo = self.vi.compute_elbo(posterior, prior, likelihood, observations)
        assert isinstance(elbo, float)
        assert np.isfinite(elbo)

    def test_categorical_elbo(self) -> None:
        """Test ELBO computation for categorical distributions."""
        posterior = {"mean": np.array([0.6, 0.3, 0.1])}
        prior = {"mean": np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])}
        likelihood = {}
        observations = np.array([0.5, 0.3, 0.2])
        elbo = self.vi.compute_elbo(posterior, prior, likelihood, observations)
        assert isinstance(elbo, float)
        assert np.isfinite(elbo)

    def test_perfect_match_high_elbo(self) -> None:
        """Test that perfect observation-posterior match gives higher ELBO."""
        posterior_good = {"mean": np.array([1.0, 0.0]), "precision": np.eye(2) * 10.0}
        posterior_bad = {"mean": np.array([5.0, 5.0]), "precision": np.eye(2) * 10.0}
        prior = {"mean": np.zeros(2), "precision": np.eye(2)}
        likelihood = {"precision": np.eye(2) * 10.0}
        observations = np.array([1.0, 0.0])

        elbo_good = self.vi.compute_elbo(
            posterior_good, prior, likelihood, observations
        )
        elbo_bad = self.vi.compute_elbo(posterior_bad, prior, likelihood, observations)
        assert elbo_good > elbo_bad
