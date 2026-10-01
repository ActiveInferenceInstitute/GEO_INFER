"""
Unit tests for the BayesianModel abstract base class in models/base.py.

Verifies that the ABC interface is correct, that log_posterior combines
log_likelihood and log_prior, and that concrete subclasses work as expected.
"""

import numpy as np
import pytest
from typing import Any

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from geo_infer_bayes.models.base import BayesianModel


class SimpleBayesianModel(BayesianModel):
    """Minimal concrete subclass for testing the base class."""

    def _setup_model(self, **kwargs) -> None:
        self.parameters = {
            "mu": {"prior": "normal", "hyperparams": {"mu": 0.0, "sigma": 10.0}},
            "sigma": {"prior": "log_normal", "hyperparams": {"mu": 0.0, "sigma": 1.0}},
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        mu = theta["mu"]
        sigma = max(theta["sigma"], 1e-10)
        observations = np.asarray(data)
        residuals = observations - mu
        n = len(observations)
        return -0.5 * np.sum(residuals**2 / sigma**2) - n * np.log(
            sigma * np.sqrt(2 * np.pi)
        )

    def log_prior(self, theta: dict[str, Any]) -> float:
        mu = theta["mu"]
        sigma = theta["sigma"]
        # Normal(0, 10) prior on mu
        lp = -0.5 * (mu / 10.0) ** 2 - np.log(10.0 * np.sqrt(2 * np.pi))
        # Log-normal(0, 1) prior on sigma
        if sigma <= 0:
            return -np.inf
        lp += -0.5 * np.log(sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        return lp

    def predict(
        self,
        X_new: np.ndarray,
        posterior: Any = None,
        samples: int = 100,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        mean = np.full(len(X_new), 0.0)
        if return_std:
            return mean, np.ones(len(X_new))
        return mean

    def posterior_predictive(
        self,
        posterior: Any,
        X: np.ndarray | None = None,
        samples: int = 100,
    ) -> np.ndarray:
        n = len(X) if X is not None else 10
        return np.random.default_rng(0).standard_normal((samples, n))


class TestBayesianModelInterface:
    """Tests for the abstract base class interface."""

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            BayesianModel(name="should_fail")

    def test_concrete_subclass_instantiation(self) -> None:
        model = SimpleBayesianModel(name="test_model")
        assert model.name == "test_model"
        assert "mu" in model.parameters
        assert "sigma" in model.parameters

    def test_log_posterior_combines_likelihood_and_prior(self) -> None:
        model = SimpleBayesianModel(name="test")
        data = np.array([1.0, 2.0, 3.0])
        theta = {"mu": 2.0, "sigma": 1.0}

        ll = model.log_likelihood(theta, data)
        lp = model.log_prior(theta)
        log_post = model.log_posterior(theta, data)

        np.testing.assert_allclose(log_post, ll + lp, atol=1e-12)

    def test_log_likelihood_finite(self) -> None:
        model = SimpleBayesianModel(name="test")
        data = np.array([0.0, 0.5, 1.0])
        theta = {"mu": 0.5, "sigma": 1.0}
        ll = model.log_likelihood(theta, data)
        assert np.isfinite(ll)

    def test_log_prior_finite_for_valid_params(self) -> None:
        model = SimpleBayesianModel(name="test")
        theta = {"mu": 0.0, "sigma": 1.0}
        lp = model.log_prior(theta)
        assert np.isfinite(lp)

    def test_log_prior_negative_inf_for_invalid_sigma(self) -> None:
        model = SimpleBayesianModel(name="test")
        theta = {"mu": 0.0, "sigma": -1.0}
        lp = model.log_prior(theta)
        assert lp == -np.inf

    def test_prepare_data_default_passthrough(self) -> None:
        model = SimpleBayesianModel(name="test")
        data = np.array([1.0, 2.0])
        result = model.prepare_data(data)
        np.testing.assert_array_equal(result, data)

    def test_predict_returns_correct_shape(self) -> None:
        model = SimpleBayesianModel(name="test")
        X_new = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        pred = model.predict(X_new)
        assert pred.shape == (3,)

    def test_predict_with_std(self) -> None:
        model = SimpleBayesianModel(name="test")
        X_new = np.array([[0.0], [1.0]])
        mean, std = model.predict(X_new, return_std=True)
        assert mean.shape == (2,)
        assert std.shape == (2,)
        assert np.all(std > 0)
