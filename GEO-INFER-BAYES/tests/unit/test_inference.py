"""
Unit tests for the BayesianInference engine in core/inference.py.

Tests verify that the inference engine correctly initialises backends,
routes to the right sampler, and that run() produces sample dictionaries.
"""

import numpy as np
import pytest
from typing import Any


from geo_infer_bayes.models.base import BayesianModel


class _SimpleModel(BayesianModel):
    """Minimal model for testing the inference engine."""

    def _setup_model(self, **kwargs) -> None:
        self.parameters = {
            "mu": {"prior": "normal", "hyperparams": {"mu": 0.0, "sigma": 1.0}},
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        obs = np.asarray(data)
        mu = theta["mu"]
        return float(-0.5 * np.sum((obs - mu) ** 2))

    def log_prior(self, theta: dict[str, Any]) -> float:
        mu = theta["mu"]
        return float(-0.5 * mu**2)

    def predict(self, X_new, posterior=None, samples=100, return_std=False):
        m = np.zeros(len(X_new))
        if return_std:
            return m, np.ones(len(X_new))
        return m

    def posterior_predictive(self, posterior, X=None, samples=100):
        n = 10 if X is None else len(X)
        return np.random.default_rng(0).standard_normal((samples, n))


class TestBayesianInferenceInit:
    def test_init_with_mcmc(self) -> None:
        from geo_infer_bayes.core.inference import BayesianInference

        model = _SimpleModel(name="test")
        bi = BayesianInference(model, method="mcmc")
        assert bi.method == "mcmc"
        assert bi.backend is not None

    def test_init_with_vi(self) -> None:
        from geo_infer_bayes.core.inference import BayesianInference

        model = _SimpleModel(name="test")
        bi = BayesianInference(model, method="vi")
        assert bi.method == "vi"

    def test_init_with_hmc(self) -> None:
        from geo_infer_bayes.core.inference import BayesianInference

        model = _SimpleModel(name="test")
        bi = BayesianInference(model, method="hmc")
        assert bi.method == "hmc"

    def test_init_with_smc(self) -> None:
        from geo_infer_bayes.core.inference import BayesianInference

        model = _SimpleModel(name="test")
        bi = BayesianInference(model, method="smc")
        assert bi.method == "smc"

    def test_init_with_abc(self) -> None:
        from geo_infer_bayes.core.inference import BayesianInference

        model = _SimpleModel(name="test")
        bi = BayesianInference(model, method="abc")
        assert bi.method == "abc"

    def test_invalid_method_raises(self) -> None:
        from geo_infer_bayes.core.inference import BayesianInference

        model = _SimpleModel(name="test")
        with pytest.raises(ValueError, match="Unsupported inference method"):
            BayesianInference(model, method="invalid")


class TestBayesianInferenceRun:
    def test_mcmc_run_produces_samples(self) -> None:
        from geo_infer_bayes.core.inference import BayesianInference

        model = _SimpleModel(name="test")
        bi = BayesianInference(
            model, method="mcmc", sampler_config={"n_chains": 1, "random_seed": 42}
        )
        data = np.array([1.0, 1.5, 2.0])
        posterior = bi.run(data, n_samples=50, n_warmup=20, progress_bar=False)
        assert hasattr(posterior, "samples")
        assert "mu" in posterior.samples

    def test_smc_run_produces_samples(self) -> None:
        from geo_infer_bayes.core.inference import BayesianInference

        model = _SimpleModel(name="test")
        bi = BayesianInference(
            model, method="smc", sampler_config={"n_particles": 50, "random_seed": 42}
        )
        data = np.array([1.0, 2.0, 3.0])
        posterior = bi.run(data, n_steps=10, progress_bar=False)
        assert hasattr(posterior, "samples")
