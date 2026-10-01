"""Regression tests for the Hamiltonian and NUTS samplers."""

from typing import Any

import numpy as np

from geo_infer_bayes.core.hmc import HMC
from geo_infer_bayes.models.base import BayesianModel


class _GaussianHMCModel(BayesianModel):
    def _setup_model(self, **kwargs) -> None:
        self.parameters = {
            "mu": {"prior": "normal", "hyperparams": {"mu": 0.0, "sigma": 5.0}}
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        return float(-0.5 * np.sum((np.asarray(data) - theta["mu"]) ** 2))

    def log_prior(self, theta: dict[str, Any]) -> float:
        return float(-0.5 * (theta["mu"] / 5.0) ** 2)

    def predict(self, X_new, posterior=None, samples=100, return_std=False):
        values = np.zeros(len(X_new))
        return (values, np.ones(len(X_new))) if return_std else values

    def posterior_predictive(self, posterior, X=None, samples=100):
        return np.zeros((samples, 1))


class _VectorGaussianHMCModel(BayesianModel):
    """Vector-valued Gaussian model for HMC/NUTS layout regression tests."""

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


def test_nuts_uses_dynamic_trajectory_and_returns_finite_samples() -> None:
    sampler = HMC(
        _GaussianHMCModel(name="test"),
        n_chains=1,
        step_size=0.05,
        max_tree_depth=4,
        random_seed=11,
    )
    samples = sampler.run(
        np.array([2.0, 2.5, 3.0]),
        n_samples=20,
        n_warmup=10,
        use_nuts=True,
        progress_bar=False,
    )

    assert samples["mu"].shape == (20,)
    assert np.all(np.isfinite(samples["mu"]))


def test_hmc_seed_does_not_mutate_global_rng() -> None:
    np.random.seed(123)
    before = np.random.random()
    HMC(_GaussianHMCModel(name="test"), random_seed=7)
    after = np.random.random()

    np.random.seed(123)
    np.testing.assert_allclose([before, after], np.random.random(2))


def test_nuts_restores_vector_parameter_shapes() -> None:
    sampler = HMC(
        _VectorGaussianHMCModel(name="vector"),
        n_chains=1,
        step_size=0.03,
        max_tree_depth=3,
        random_seed=12,
    )
    samples = sampler.run(
        np.array([1.0, -1.0]),
        n_samples=8,
        n_warmup=4,
        init_strategy="custom",
        custom_init=[{"weights": np.zeros(2)}],
        use_nuts=True,
        progress_bar=False,
    )

    assert samples["weights"].shape == (8, 2)
    assert np.all(np.isfinite(samples["weights"]))


def test_step_size_adaptation_converges_to_target_acceptance() -> None:
    """Windowed dual-averaging adaptation drives post-warmup acceptance
    toward ``target_accept`` on a 2D Gaussian, even from a badly oversized
    initial step size."""
    data = np.array([1.0, -1.0])
    sampler = HMC(
        _VectorGaussianHMCModel(name="adapt"),
        n_chains=2,
        step_size=1.0,
        target_accept=0.8,
        max_tree_depth=5,
        random_seed=21,
    )
    samples = sampler.run(
        data,
        n_samples=200,
        n_warmup=300,
        use_nuts=True,
        progress_bar=False,
    )

    assert samples["weights"].shape == (400, 2)
    assert np.all(np.isfinite(samples["weights"]))
    assert sampler.sampling_acceptance_rates is not None
    assert np.all(np.abs(sampler.sampling_acceptance_rates - 0.8) <= 0.15), (
        f"sampling acceptance {sampler.sampling_acceptance_rates} off target"
    )
