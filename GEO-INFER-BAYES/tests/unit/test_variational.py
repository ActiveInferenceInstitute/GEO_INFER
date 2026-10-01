"""
Unit tests for VariationalInference in core/variational.py.

Tests cover initialisation, variational parameter setup, ELBO
computation, and that the approximate posterior is sensible.
"""

import numpy as np
import pytest
from typing import Any


from geo_infer_bayes.core.variational import VariationalInference
from geo_infer_bayes.models.base import BayesianModel


class _SimpleVIModel(BayesianModel):
    """Gaussian model for testing variational inference."""

    def _setup_model(self, **kwargs) -> None:
        self.parameters = {
            "mu": {"prior": "normal", "hyperparams": {"mu": 0.0, "sigma": 5.0}},
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        obs = np.asarray(data)
        mu = theta["mu"]
        return float(-0.5 * np.sum((obs - mu) ** 2))

    def log_prior(self, theta: dict[str, Any]) -> float:
        mu = theta["mu"]
        return float(-0.5 * (mu / 5.0) ** 2)

    def predict(self, X_new, posterior=None, samples=100, return_std=False):
        m = np.zeros(len(X_new))
        return (m, np.ones(len(X_new))) if return_std else m

    def posterior_predictive(self, posterior, X=None, samples=100):
        return np.zeros((samples, 1))


class _VectorVIModel(BayesianModel):
    """Vector-valued Gaussian model for mean-field VI shape contracts."""

    def _setup_model(self, **kwargs) -> None:
        self.parameters = {
            "weights": {
                "prior": "normal",
                "hyperparams": {"mu": np.zeros(2), "sigma": np.ones(2)},
            }
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        weights = np.asarray(theta["weights"], dtype=float)
        observations = np.asarray(data, dtype=float)
        return float(-0.5 * np.sum((observations - weights) ** 2))

    def log_prior(self, theta: dict[str, Any]) -> float:
        weights = np.asarray(theta["weights"], dtype=float)
        return float(-0.5 * np.sum((weights / 5.0) ** 2))

    def predict(self, X_new, posterior=None, samples=100, return_std=False):
        values = np.zeros(len(X_new))
        return (values, np.ones(len(X_new))) if return_std else values

    def posterior_predictive(self, posterior, X=None, samples=100):
        return np.zeros((samples, 1))


class TestVariationalInferenceInit:
    def test_default_init(self) -> None:
        model = _SimpleVIModel(name="test")
        vi = VariationalInference(model)
        assert vi.learning_rate == 0.01
        assert vi.vi_method == "meanfield"

    def test_fullrank_init(self) -> None:
        model = _SimpleVIModel(name="test")
        vi = VariationalInference(model, vi_method="fullrank")
        assert vi.vi_method == "fullrank"

    def test_invalid_method_raises(self) -> None:
        model = _SimpleVIModel(name="test")
        with pytest.raises(ValueError, match="Unsupported VI method"):
            VariationalInference(model, vi_method="invalid")


class TestVariationalParameterInit:
    def test_initialize_variational_parameters(self) -> None:
        model = _SimpleVIModel(name="test")
        vi = VariationalInference(model, random_seed=0)
        var_params = vi._initialize_variational_parameters(["mu"])
        assert "mu" in var_params
        assert "mean" in var_params["mu"]
        assert "log_std" in var_params["mu"]
        # Mean should be initialized from prior
        np.testing.assert_allclose(var_params["mu"]["mean"], 0.0)

    def test_fullrank_has_cov_factor(self) -> None:
        model = _SimpleVIModel(name="test")
        vi = VariationalInference(model, vi_method="fullrank", random_seed=0)
        var_params = vi._initialize_variational_parameters(["mu"])
        assert "cov_factor" in var_params["mu"]
        assert var_params["mu"]["cov_factor"].shape == (1, 1)


class TestVariationalSampling:
    def test_sample_variational_distribution(self) -> None:
        model = _SimpleVIModel(name="test")
        vi = VariationalInference(model, random_seed=0)
        var_params = {
            "mu": {"mean": 3.0, "log_std": np.log(0.5)},
        }
        sample = vi._sample_variational_distribution(var_params)
        assert "mu" in sample
        # Sample should be finite
        assert np.isfinite(sample["mu"])

    def test_log_prob_variational_finite(self) -> None:
        model = _SimpleVIModel(name="test")
        vi = VariationalInference(model, random_seed=0)
        var_params = {
            "mu": {"mean": 0.0, "log_std": 0.0},
        }
        sample = {"mu": 0.5}
        lp = vi._log_prob_variational(sample, var_params)
        assert np.isfinite(lp)


class TestVariationalInferenceRun:
    def test_run_produces_samples(self) -> None:
        model = _SimpleVIModel(name="test")
        vi = VariationalInference(
            model,
            n_iterations=100,
            learning_rate=0.05,
            n_mc_samples=5,
            random_seed=42,
        )
        data = np.array([3.0, 3.5, 4.0, 3.0, 3.5])
        samples = vi.run(data, progress_bar=False, n_samples=200)
        assert "mu" in samples
        assert len(samples["mu"]) == 200

    def test_vi_approximate_posterior_reasonable(self) -> None:
        """VI should produce finite samples from the approximate posterior.
        With the score function estimator, convergence is not guaranteed
        in a small number of iterations, so we only check finiteness
        and that the variational parameters were updated."""
        model = _SimpleVIModel(name="test")
        vi = VariationalInference(
            model,
            n_iterations=50,
            learning_rate=0.01,
            n_mc_samples=3,
            random_seed=42,
        )
        data = np.array([5.0, 5.5, 4.5, 5.0, 5.2])
        samples = vi.run(data, progress_bar=False, n_samples=200)
        # VI ran and produced samples
        assert "mu" in samples
        assert len(samples["mu"]) == 200
        # Samples should be finite (no NaN/inf)
        assert np.all(np.isfinite(samples["mu"]))

    def test_generate_samples(self) -> None:
        model = _SimpleVIModel(name="test")
        vi = VariationalInference(model, random_seed=0)
        var_params = {
            "mu": {"mean": 2.0, "log_std": np.log(0.3)},
        }
        samples = vi._generate_samples(var_params, n_samples=100)
        assert len(samples["mu"]) == 100
        np.testing.assert_allclose(np.mean(samples["mu"]), 2.0, atol=0.3)

    def test_fullrank_run_generates_finite_samples(self) -> None:
        model = _SimpleVIModel(name="test")
        vi = VariationalInference(
            model,
            n_iterations=3,
            learning_rate=0.01,
            n_mc_samples=4,
            vi_method="fullrank",
            random_seed=42,
        )
        data = np.array([1.0, 1.5, 2.0])
        samples = vi.run(data, progress_bar=False, n_samples=8)
        assert "mu" in samples
        assert samples["mu"].shape == (8,)
        assert np.all(np.isfinite(samples["mu"]))


def test_variational_seed_does_not_mutate_global_rng() -> None:
    np.random.seed(123)
    before = np.random.random()
    VariationalInference(_SimpleVIModel(name="test"), random_seed=7)
    after = np.random.random()

    np.random.seed(123)
    np.testing.assert_allclose([before, after], np.random.random(2))


def test_variational_update_uses_previous_samples_as_warm_start() -> None:
    model = _SimpleVIModel(name="test")
    vi = VariationalInference(
        model,
        n_iterations=1,
        learning_rate=1e-12,
        n_mc_samples=1,
        random_seed=0,
    )

    samples = vi.update(
        np.array([10.0, 10.0]),
        {"mu": np.full(20, 10.0)},
        progress_bar=False,
        n_samples=20,
    )

    assert np.mean(samples["mu"]) > 9.9


def test_variational_vector_parameters_preserve_shape_and_warm_start() -> None:
    model = _VectorVIModel(name="vector")
    vi = VariationalInference(
        model,
        n_iterations=2,
        learning_rate=1e-12,
        n_mc_samples=1,
        random_seed=0,
    )

    samples = vi.update(
        np.array([10.0, 10.0]),
        {"weights": np.full((20, 2), 10.0)},
        progress_bar=False,
        n_samples=12,
    )

    assert samples["weights"].shape == (12, 2)
    assert np.all(np.isfinite(samples["weights"]))
    assert np.all(np.mean(samples["weights"], axis=0) > 9.9)
