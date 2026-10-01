"""
Unit tests for diagnostics in utils/diagnostics.py.

Tests cover MCMC diagnostics (ESS, R-hat), convergence
metrics (Geweke Z-score, Monte Carlo standard error), and edge cases.
"""

import numpy as np


from geo_infer_bayes.utils.diagnostics import (
    mcmc_diagnostics,
    convergence_metrics,
)


class TestMCMCDiagnostics:
    def test_basic_diagnostics(self) -> None:
        rng = np.random.default_rng(0)
        samples = {"mu": rng.normal(3.0, 0.5, size=1000)}
        diag = mcmc_diagnostics(samples)
        assert "mu" in diag
        np.testing.assert_allclose(diag["mu"]["mean"], 3.0, atol=0.1)
        assert diag["mu"]["std"] > 0
        assert diag["mu"]["min"] < diag["mu"]["max"]

    def test_ess_positive(self) -> None:
        rng = np.random.default_rng(1)
        samples = {"theta": rng.standard_normal(500)}
        diag = mcmc_diagnostics(samples)
        assert diag["theta"]["ess"] > 0

    def test_r_hat_is_undefined_without_multiple_chains(self) -> None:
        """A single chain cannot identify between-chain scale differences."""
        rng = np.random.default_rng(2)
        samples = {"x": rng.standard_normal(200)}
        diag = mcmc_diagnostics(samples)
        assert np.isnan(diag["x"]["r_hat"])

    def test_r_hat_detects_multiple_chain_scale_difference(self) -> None:
        samples = {"x": np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])}
        assert mcmc_diagnostics(samples)["x"]["r_hat"] > 1.0

    def test_multiple_parameters(self) -> None:
        rng = np.random.default_rng(3)
        samples = {
            "alpha": rng.standard_normal(300),
            "beta": rng.standard_normal(300) + 5.0,
        }
        diag = mcmc_diagnostics(samples)
        assert "alpha" in diag
        assert "beta" in diag
        np.testing.assert_allclose(diag["beta"]["mean"], 5.0, atol=0.3)

    def test_2d_samples_flattened(self) -> None:
        rng = np.random.default_rng(4)
        samples = {"mu": rng.standard_normal((4, 100))}
        diag = mcmc_diagnostics(samples)
        assert np.isfinite(diag["mu"]["mean"])


class TestConvergenceMetrics:
    def test_geweke_z_near_zero_for_stationary_chain(self) -> None:
        """A stationary chain should have Geweke Z near zero."""
        rng = np.random.default_rng(0)
        samples = {"mu": rng.normal(0.0, 1.0, size=2000)}
        metrics = convergence_metrics(samples)
        assert "mu" in metrics
        # For a truly stationary chain, |z| should typically be < 2
        assert abs(metrics["mu"]["geweke_z"]) < 3.0

    def test_geweke_z_large_for_nonstationary_chain(self) -> None:
        """A trending chain should have large Geweke Z."""
        trend = np.linspace(0, 10, 1000)
        samples = {"mu": trend}
        metrics = convergence_metrics(samples)
        assert abs(metrics["mu"]["geweke_z"]) > 2.0

    def test_monte_carlo_se_positive(self) -> None:
        rng = np.random.default_rng(5)
        samples = {"x": rng.standard_normal(500)}
        metrics = convergence_metrics(samples)
        assert metrics["x"]["monte_carlo_se"] > 0

    def test_monte_carlo_se_decreases_with_more_samples(self) -> None:
        rng = np.random.default_rng(6)
        small = {"x": rng.standard_normal(100)}
        large = {"x": rng.standard_normal(10000)}
        mcse_small = convergence_metrics(small)["x"]["monte_carlo_se"]
        mcse_large = convergence_metrics(large)["x"]["monte_carlo_se"]
        assert mcse_large < mcse_small

    def test_constant_samples_zero_variance(self) -> None:
        samples = {"c": np.ones(100)}
        metrics = convergence_metrics(samples)
        assert metrics["c"]["geweke_z"] == 0.0
        assert metrics["c"]["monte_carlo_se"] == 0.0
