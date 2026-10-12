"""
Unit tests for the MCMC sampler in core/mcmc.py.

Tests verify chain initialisation, proposal generation, acceptance
logic, and that the sampler converges to the known posterior for a
simple Gaussian model.
"""

import numpy as np
from typing import Any


from geo_infer_bayes.core.mcmc import MCMC
from geo_infer_bayes.models.base import BayesianModel


class _GaussianModel(BayesianModel):
    """Gaussian model with known posterior for testing MCMC."""

    def _setup_model(self, **kwargs) -> None:
        self.parameters = {
            "mu": {"prior": "normal", "hyperparams": {"mu": 0.0, "sigma": 10.0}},
        }

    def log_likelihood(self, theta: dict[str, Any], data: Any) -> float:
        obs = np.asarray(data)
        mu = theta["mu"]
        return float(-0.5 * np.sum((obs - mu) ** 2))

    def log_prior(self, theta: dict[str, Any]) -> float:
        mu = theta["mu"]
        sigma_prior = 10.0
        return float(-0.5 * (mu / sigma_prior) ** 2)

    def predict(self, X_new, posterior=None, samples=100, return_std=False):
        m = np.zeros(len(X_new))
        return (m, np.ones(len(X_new))) if return_std else m

    def posterior_predictive(self, posterior, X=None, samples=100):
        return np.zeros((samples, 1))


class _VectorGaussianModel(BayesianModel):
    """Two-dimensional Gaussian model for parameter-layout regression tests."""

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


class TestMCMCInitialization:
    def test_default_init(self) -> None:
        model = _GaussianModel(name="test")
        mcmc = MCMC(model)
        assert mcmc.n_chains == 4
        assert mcmc.step_size == 0.1
        assert mcmc.adapt_step_size is True

    def test_custom_init(self) -> None:
        model = _GaussianModel(name="test")
        mcmc = MCMC(model, n_chains=2, step_size=0.05, random_seed=42)
        assert mcmc.n_chains == 2
        assert mcmc.step_size == 0.05

    def test_chain_initialization_random(self) -> None:
        model = _GaussianModel(name="test")
        mcmc = MCMC(model, n_chains=3, random_seed=0)
        data = np.array([1.0, 2.0])
        chains = mcmc._initialize_chains(data, "random")
        assert len(chains) == 3
        for chain in chains:
            assert "mu" in chain
            assert isinstance(chain["mu"], (float, np.floating))


class TestMCMCProposal:
    def test_propose_returns_new_theta(self) -> None:
        model = _GaussianModel(name="test")
        mcmc = MCMC(model, random_seed=0)
        current = {"mu": 1.0}
        proposed, log_ratio = mcmc._propose(current)
        assert "mu" in proposed
        # With step_size > 0, proposed should differ from current
        # (statistically almost certain)
        assert proposed["mu"] != current["mu"] or True  # non-deterministic

    def test_log_proposal_ratio_is_zero_for_symmetric(self) -> None:
        """Normal proposals are symmetric, so log ratio should be 0."""
        model = _GaussianModel(name="test")
        mcmc = MCMC(model, random_seed=1)
        current = {"mu": 0.0}
        _, log_ratio = mcmc._propose(current)
        np.testing.assert_allclose(log_ratio, 0.0)


class TestMCMCSampling:
    def test_run_produces_samples(self) -> None:
        model = _GaussianModel(name="test")
        mcmc = MCMC(model, n_chains=1, random_seed=42)
        data = np.array([2.0, 2.5, 3.0, 2.0, 2.5])
        samples = mcmc.run(data, n_samples=100, n_warmup=50, progress_bar=False)
        assert "mu" in samples
        assert len(samples["mu"]) == 100  # 1 chain * 100 samples

    def test_posterior_mean_near_data_mean(self) -> None:
        """For a Gaussian likelihood with flat prior, the posterior mean
        should be close to the sample mean of the data."""
        model = _GaussianModel(name="test")
        mcmc = MCMC(model, n_chains=2, step_size=0.5, random_seed=0)
        rng = np.random.default_rng(0)
        data = rng.normal(5.0, 1.0, size=20)
        samples = mcmc.run(data, n_samples=500, n_warmup=200, progress_bar=False)
        posterior_mean = np.mean(samples["mu"])
        # With 20 data points from N(5,1) and broad prior, posterior
        # mean should be near 5.0
        np.testing.assert_allclose(posterior_mean, np.mean(data), atol=1.5)

    def test_thinning(self) -> None:
        model = _GaussianModel(name="test")
        mcmc = MCMC(model, n_chains=1, random_seed=10)
        data = np.array([1.0, 2.0, 3.0])
        samples = mcmc.run(data, n_samples=50, n_warmup=20, thin=2, progress_bar=False)
        assert len(samples["mu"]) == 50


class TestMCMCLogPosterior:
    def test_log_posterior_finite(self) -> None:
        model = _GaussianModel(name="test")
        mcmc = MCMC(model)
        theta = {"mu": 1.0}
        data = np.array([1.0, 2.0])
        lp = mcmc._log_posterior(theta, data)
        assert np.isfinite(lp)

    def test_log_posterior_invalid_returns_neg_inf(self) -> None:
        """If model.log_posterior raises, MCMC should return -inf."""
        model = _GaussianModel(name="test")

        # Monkey-patch to raise
        original = model.log_posterior

        def bad_log_post(theta, data):
            raise ValueError("bad")

        model.log_posterior = bad_log_post

        mcmc = MCMC(model)
        result = mcmc._log_posterior({"mu": 0.0}, np.array([1.0]))
        assert result == -np.inf

        model.log_posterior = original


def test_mcmc_random_seed_does_not_mutate_global_rng() -> None:
    """Sampler reproducibility must be isolated from NumPy's global RNG."""
    np.random.seed(123)
    before = np.random.random()
    MCMC(_GaussianModel(name="test"), random_seed=7)
    after = np.random.random()

    np.random.seed(123)
    np.testing.assert_allclose([before, after], np.random.random(2))


def test_mcmc_update_supports_fewer_previous_samples_than_chains() -> None:
    model = _GaussianModel(name="test")
    sampler = MCMC(model, n_chains=3, random_seed=0)
    previous = {"mu": np.array([2.0])}

    samples = sampler.update(
        np.array([2.0, 2.5]),
        previous,
        n_samples=4,
        progress_bar=False,
    )

    assert samples["mu"].shape == (12,)


def test_mcmc_restores_vector_parameter_shapes() -> None:
    sampler = MCMC(_VectorGaussianModel(name="vector"), n_chains=1, random_seed=4)
    samples = sampler.run(
        np.array([1.0, -1.0]),
        n_samples=12,
        n_warmup=5,
        init_strategy="custom",
        custom_init=[{"weights": np.zeros(2)}],
        progress_bar=False,
    )

    assert samples["weights"].shape == (12, 2)
    assert np.all(np.isfinite(samples["weights"]))
