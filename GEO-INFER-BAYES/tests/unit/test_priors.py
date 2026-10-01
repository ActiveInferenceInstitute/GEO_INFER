"""
Unit tests for prior distributions in utils/priors.py.

Tests cover SpatialPrior (ICAR, BYM, Leroux), TemporalPrior
(AR1, RW1, RW2), and GaussianProcessPrior.
"""

import numpy as np
import pytest


from geo_infer_bayes.utils.priors import (
    SpatialPrior,
    TemporalPrior,
    GaussianProcessPrior,
)


class TestSpatialPrior:
    @pytest.fixture
    def adjacency_matrix(self):
        """Simple 4-node adjacency matrix (line graph)."""
        W = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )
        return W

    def test_icar_prior_returns_scalar(self, adjacency_matrix) -> None:
        prior = SpatialPrior(prior_type="icar", tau=1.0)
        phi = np.array([0.1, 0.2, 0.3, 0.4])
        result = prior.log_prior(phi, adjacency_matrix)
        assert np.isscalar(result)
        assert np.isfinite(result)

    def test_icar_prior_zero_field(self, adjacency_matrix) -> None:
        """A zero field should give the maximum prior value for ICAR."""
        prior = SpatialPrior(prior_type="icar", tau=1.0)
        phi_zero = np.zeros(4)
        phi_nonzero = np.array([1.0, -1.0, 2.0, -2.0])
        lp_zero = prior.log_prior(phi_zero, adjacency_matrix)
        lp_nonzero = prior.log_prior(phi_nonzero, adjacency_matrix)
        assert lp_zero >= lp_nonzero

    def test_bym_prior_returns_scalar(self, adjacency_matrix) -> None:
        prior = SpatialPrior(prior_type="bym", tau=1.0, alpha=0.5)
        phi = np.array([0.5, 0.3, 0.1, 0.2])
        result = prior.log_prior(phi, adjacency_matrix)
        assert np.isfinite(result)

    def test_leroux_prior_returns_scalar(self, adjacency_matrix) -> None:
        prior = SpatialPrior(prior_type="leroux", tau=1.0, rho=0.5)
        phi = np.array([0.1, 0.2, 0.1, 0.3])
        result = prior.log_prior(phi, adjacency_matrix)
        assert np.isfinite(result)

    def test_unknown_prior_raises(self) -> None:
        prior = SpatialPrior(prior_type="unknown")
        with pytest.raises(ValueError, match="Unknown spatial prior"):
            prior.log_prior(np.zeros(3), np.eye(3))


class TestTemporalPrior:
    def test_ar1_prior_finite(self) -> None:
        prior = TemporalPrior(prior_type="ar1", phi=0.8, tau=1.0)
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = prior.log_prior(x)
        assert np.isfinite(result)

    def test_rw1_prior_finite(self) -> None:
        prior = TemporalPrior(prior_type="rw1", tau=1.0)
        x = np.array([1.0, 1.1, 1.2, 1.3])
        result = prior.log_prior(x)
        assert np.isfinite(result)

    def test_rw1_constant_field_maximises(self) -> None:
        """A constant field should maximise the RW1 prior."""
        prior = TemporalPrior(prior_type="rw1", tau=1.0)
        x_const = np.ones(10)
        x_noisy = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        assert prior.log_prior(x_const) > prior.log_prior(x_noisy)

    def test_rw2_prior_finite(self) -> None:
        prior = TemporalPrior(prior_type="rw2", tau=1.0)
        x = np.linspace(0, 1, 10)
        result = prior.log_prior(x)
        assert np.isfinite(result)

    def test_rw2_linear_field_maximises(self) -> None:
        """A linear field has zero second differences, maximising RW2."""
        prior = TemporalPrior(prior_type="rw2", tau=1.0)
        x_linear = np.linspace(0, 5, 20)
        x_wiggly = np.sin(np.linspace(0, 10, 20))
        assert prior.log_prior(x_linear) > prior.log_prior(x_wiggly)

    def test_unknown_temporal_prior_raises(self) -> None:
        prior = TemporalPrior(prior_type="invalid")
        with pytest.raises(ValueError, match="Unknown temporal prior"):
            prior.log_prior(np.zeros(5))


class TestGaussianProcessPrior:
    def test_log_prior_finite_for_valid_params(self) -> None:
        gp_prior = GaussianProcessPrior(kernel="rbf")
        result = gp_prior.log_prior(lengthscale=1.0, variance=1.0)
        assert np.isfinite(result)

    def test_log_prior_negative(self) -> None:
        """Log density should always be negative for continuous distributions."""
        gp_prior = GaussianProcessPrior(kernel="matern")
        result = gp_prior.log_prior(lengthscale=2.0, variance=0.5)
        assert result < 0

    def test_mode_at_exp_mu(self) -> None:
        """For log-normal with mu=0, sigma=1, mode is at 1/e.
        Values near the mode should have higher log-prior."""
        gp_prior = GaussianProcessPrior(
            kernel="rbf",
            lengthscale_mu=0.0,
            lengthscale_sigma=1.0,
            variance_mu=0.0,
            variance_sigma=1.0,
        )
        lp_near_mode = gp_prior.log_prior(lengthscale=np.exp(-1), variance=np.exp(-1))
        lp_far = gp_prior.log_prior(lengthscale=100.0, variance=100.0)
        assert lp_near_mode > lp_far
