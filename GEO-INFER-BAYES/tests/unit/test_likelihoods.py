"""
Unit tests for likelihood functions in utils/likelihoods.py.

Tests cover SpatialLikelihood (Gaussian, Poisson, Binomial),
PoissonProcess, and GaussianLikelihood.
"""

import numpy as np
import pytest


from geo_infer_bayes.utils.likelihoods import (
    SpatialLikelihood,
    PoissonProcess,
    GaussianLikelihood,
)


class TestSpatialLikelihoodGaussian:
    def test_gaussian_perfect_prediction(self) -> None:
        ll = SpatialLikelihood(likelihood_type="gaussian", sigma=1.0)
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        result = ll.log_likelihood(pred, obs)
        # Perfect prediction: residuals = 0
        expected = -len(obs) * np.log(1.0 * np.sqrt(2 * np.pi))
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_gaussian_worse_prediction_lower_ll(self) -> None:
        ll = SpatialLikelihood(likelihood_type="gaussian", sigma=1.0)
        obs = np.array([0.0, 0.0, 0.0])
        pred_good = np.array([0.0, 0.0, 0.0])
        pred_bad = np.array([5.0, 5.0, 5.0])
        assert ll.log_likelihood(pred_good, obs) > ll.log_likelihood(pred_bad, obs)

    def test_gaussian_with_weights(self) -> None:
        ll = SpatialLikelihood(likelihood_type="gaussian", sigma=1.0)
        obs = np.array([1.0, 2.0])
        pred = np.array([1.5, 2.5])
        weights = np.array([1.0, 1.0])
        result = ll.log_likelihood(pred, obs, spatial_weights=weights)
        assert np.isfinite(result)

    def test_smaller_sigma_penalises_more(self) -> None:
        obs = np.array([0.0])
        pred = np.array([1.0])
        ll_wide = SpatialLikelihood(likelihood_type="gaussian", sigma=10.0)
        ll_narrow = SpatialLikelihood(likelihood_type="gaussian", sigma=0.1)
        assert ll_wide.log_likelihood(pred, obs) > ll_narrow.log_likelihood(pred, obs)


class TestSpatialLikelihoodPoisson:
    def test_poisson_returns_finite(self) -> None:
        ll = SpatialLikelihood(likelihood_type="poisson")
        obs = np.array([2, 3, 5], dtype=float)
        pred = np.array([2.5, 3.0, 4.5])
        result = ll.log_likelihood(pred, obs)
        assert np.isfinite(result)

    def test_poisson_factorial_term_per_observation(self) -> None:
        """The factorial penalty must be sum(log(obs_i!)), not log(max(obs)!)."""
        ll = SpatialLikelihood(likelihood_type="poisson")
        pred = np.array([2.0, 2.0])
        obs = np.array([1, 5])
        expected = float(np.sum(obs * np.log(pred) - pred))
        from scipy.special import gammaln

        expected -= float(np.sum(gammaln(obs + 1)))
        assert ll.log_likelihood(pred, obs) == pytest.approx(expected)

    def test_poisson_empty_obs_rejected(self) -> None:
        ll = SpatialLikelihood(likelihood_type="poisson")
        with pytest.raises(ValueError):
            ll.log_likelihood(np.array([1.0]), np.array([]))


class TestSpatialLikelihoodBinomial:
    def test_binomial_returns_finite(self) -> None:
        ll = SpatialLikelihood(likelihood_type="binomial", n=1)
        obs = np.array([1, 0, 1], dtype=float)
        pred = np.array([2.0, -2.0, 1.0])  # logit scale
        result = ll.log_likelihood(pred, obs)
        assert np.isfinite(result)

    def test_unknown_type_raises(self) -> None:
        ll = SpatialLikelihood(likelihood_type="unknown_type")
        with pytest.raises(ValueError, match="Unknown likelihood"):
            ll.log_likelihood(np.zeros(1), np.zeros(1))


class TestPoissonProcess:
    def test_log_likelihood_finite(self) -> None:
        pp = PoissonProcess()
        intensity = np.array([2.0, 3.0, 4.0])
        points = np.array([[0.1, 0.2], [0.5, 0.5]])
        window = {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0}
        result = pp.log_likelihood(intensity, points, window)
        assert np.isfinite(result)

    def test_integrate_intensity(self) -> None:
        pp = PoissonProcess()
        intensity = np.array([5.0, 5.0, 5.0])
        window = {"xmin": 0.0, "xmax": 2.0, "ymin": 0.0, "ymax": 3.0}
        integral = pp._integrate_intensity(intensity, window)
        # Area = 2*3 = 6, mean intensity = 5, integral = 30
        np.testing.assert_allclose(integral, 30.0)


class TestGaussianLikelihood:
    def test_gaussian_likelihood_finite(self) -> None:
        gl = GaussianLikelihood(sigma=1.0)
        pred = np.array([0.0, 1.0])
        obs = np.array([0.1, 0.9])
        result = gl.log_likelihood(pred, obs)
        assert np.isfinite(result)

    def test_perfect_match(self) -> None:
        gl = GaussianLikelihood(sigma=1.0)
        obs = np.array([1.0, 2.0, 3.0])
        result = gl.log_likelihood(obs, obs)
        expected = -len(obs) * np.log(np.sqrt(2 * np.pi))
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_larger_residual_lower_likelihood(self) -> None:
        gl = GaussianLikelihood(sigma=1.0)
        obs = np.zeros(5)
        ll_close = gl.log_likelihood(np.full(5, 0.1), obs)
        ll_far = gl.log_likelihood(np.full(5, 10.0), obs)
        assert ll_close > ll_far
