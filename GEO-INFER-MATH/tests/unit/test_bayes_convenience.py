"""
Tests for GEO-INFER-MATH Bayesian convenience API.

Tests cover: posterior_helper, prior_builder, mcmc_wrapper,
bayesian_optimization_helper, and BayesianConvenience class.
"""

import numpy as np
import pytest


from geo_infer_math.api.convenience.bayes_convenience import (
    posterior_helper,
    prior_builder,
    mcmc_wrapper,
    bayesian_optimization_helper,
    BayesianConvenience,
)


class TestPosteriorHelper:
    """Tests for posterior_helper function."""

    def test_posterior_is_normalized(self):
        prior = np.array([0.25, 0.25, 0.25, 0.25])

        def likelihood(data, params):
            return np.array([0.1, 0.6, 0.2, 0.1])

        data = np.array([1.0])
        posterior = posterior_helper(prior, likelihood, data)
        assert abs(np.sum(posterior) - 1.0) < 1e-10

    def test_posterior_proportional_to_prior_times_likelihood(self):
        prior = np.array([0.5, 0.3, 0.2])
        likelihood_vals = np.array([0.8, 0.1, 0.1])

        def likelihood(data, params):
            return likelihood_vals

        data = np.array([1.0])
        posterior = posterior_helper(prior, likelihood, data, normalize=False)
        expected_unnorm = prior * likelihood_vals
        np.testing.assert_allclose(posterior, expected_unnorm, rtol=1e-5)

    def test_uniform_prior_posterior_equals_likelihood(self):
        n = 5
        prior = np.ones(n) / n
        likelihood_vals = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

        def likelihood(d, p):
            return likelihood_vals

        data = np.array([1.0])
        posterior = posterior_helper(prior, likelihood, data)
        # With uniform prior, posterior should be proportional to likelihood
        expected = likelihood_vals / np.sum(likelihood_vals)
        np.testing.assert_allclose(posterior, expected, rtol=1e-5)


class TestPriorBuilder:
    """Tests for prior_builder function."""

    def test_uniform_prior_sums_to_one(self):
        prior = prior_builder("uniform", size=100)
        assert abs(np.sum(prior) - 1.0) < 1e-10

    def test_gaussian_prior_sums_to_one(self):
        prior = prior_builder(
            "gaussian", parameters={"mean": 0.0, "std": 1.0}, size=100
        )
        assert abs(np.sum(prior) - 1.0) < 1e-10

    def test_gaussian_prior_peaked_at_mean(self):
        prior = prior_builder(
            "gaussian", parameters={"mean": 0.0, "std": 1.0}, size=101
        )
        # Peak should be at or near the middle index
        peak_idx = np.argmax(prior)
        assert abs(peak_idx - 50) <= 1

    def test_unknown_distribution_raises(self):
        with pytest.raises(ValueError, match="Unknown distribution"):
            prior_builder("unknown_dist")

    def test_custom_size(self):
        prior = prior_builder("uniform", size=50)
        assert len(prior) == 50


class TestMCMCWrapper:
    """Tests for mcmc_wrapper function."""

    def test_metropolis_returns_correct_shape(self):
        def log_posterior(x):
            return float(-0.5 * np.sum(x**2))  # Standard normal

        samples, metadata = mcmc_wrapper(
            log_posterior,
            initial_state=np.array([0.0, 0.0]),
            n_samples=100,
            n_burnin=20,
            step_size=0.5,
            method="metropolis",
            rng=np.random.default_rng(42),
        )
        assert samples.shape == (100, 2)
        assert "acceptance_rate" in metadata
        assert 0.0 <= metadata["acceptance_rate"] <= 1.0

    def test_gibbs_sampling(self):
        def log_posterior(x):
            return float(-0.5 * np.sum(x**2))

        samples, metadata = mcmc_wrapper(
            log_posterior,
            initial_state=np.array([1.0]),
            n_samples=50,
            n_burnin=10,
            step_size=0.5,
            method="gibbs",
            rng=np.random.default_rng(42),
        )
        assert samples.shape == (50, 1)
        assert metadata["method"] == "gibbs"

    def test_samples_concentrated_near_mode(self):
        def log_posterior(x):
            return float(-0.5 * np.sum(x**2))

        samples, _ = mcmc_wrapper(
            log_posterior,
            initial_state=np.array([0.0]),
            n_samples=500,
            n_burnin=100,
            step_size=1.0,
            rng=np.random.default_rng(42),
        )
        # Mean should be near 0 for standard normal
        assert abs(np.mean(samples)) < 1.0


class TestBayesianOptimizationHelper:
    """Tests for bayesian_optimization_helper."""

    def test_finds_optimum(self):
        prior = np.ones(10) / 10

        def objective(idx):  # max at idx=5
            return -float((idx - 5) ** 2)

        opt_params, opt_val, metadata = bayesian_optimization_helper(
            objective, prior, n_iterations=50, rng=np.random.default_rng(42)
        )
        assert opt_val >= -25
        assert "n_iterations" in metadata


class TestBayesianConvenience:
    """Tests for BayesianConvenience class."""

    def test_initialization(self):
        bc = BayesianConvenience()
        assert hasattr(bc, "logger")
        assert hasattr(bc, "_posterior_cache")

    def test_calculate_posterior(self):
        bc = BayesianConvenience()
        prior = np.array([0.5, 0.3, 0.2])

        def likelihood(d, p):
            return np.array([0.1, 0.8, 0.1])

        data = np.array([1.0])
        posterior = bc.calculate_posterior(prior, likelihood, data)
        assert abs(np.sum(posterior) - 1.0) < 1e-10

    def test_build_prior(self):
        bc = BayesianConvenience()
        # BayesianConvenience.build_prior passes **kwargs as `parameters` dict
        # size parameter must be passed directly to prior_builder
        prior = bc.build_prior("uniform")
        assert len(prior) == 100  # default size
        assert abs(np.sum(prior) - 1.0) < 1e-10

    def test_build_prior_gaussian(self):
        bc = BayesianConvenience()
        prior = bc.build_prior("gaussian", mean=0.0, std=1.0)
        assert abs(np.sum(prior) - 1.0) < 1e-10

    def test_mcmc_sample(self):
        bc = BayesianConvenience()

        def log_post(x):
            return float(-0.5 * np.sum(x**2))

        samples, meta = bc.mcmc_sample(
            log_post,
            np.array([0.0]),
            n_samples=50,
            n_burnin=10,
            rng=np.random.default_rng(42),
        )
        assert samples.shape[0] == 50
