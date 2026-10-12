"""
Unit tests for the high-level GaussianProcess class in __init__.py.

Tests cover kernel computation, Cholesky-based fitting, prediction
with mean and standard deviation, and the log marginal likelihood.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


from geo_infer_bayes import GaussianProcess


class TestGaussianProcessInit:
    """Tests for GaussianProcess initialisation and kernel selection."""

    def test_default_construction(self) -> None:
        gp = GaussianProcess()
        assert gp.kernel_type == "rbf"
        assert gp.length_scale == 1.0
        assert gp.signal_variance == 1.0
        assert gp.noise_variance == 1e-2
        assert gp.X_train is None
        assert gp._alpha is None

    def test_custom_construction(self) -> None:
        gp = GaussianProcess(
            kernel_type="matern32",
            length_scale=0.5,
            signal_variance=2.0,
            noise_variance=0.05,
        )
        assert gp.kernel_type == "matern32"
        assert gp.length_scale == 0.5
        assert gp.signal_variance == 2.0

    def test_unsupported_kernel_raises(self) -> None:
        gp = GaussianProcess(kernel_type="invalid_kernel")
        X = np.array([[0.0], [1.0]])
        with pytest.raises(ValueError, match="Unsupported kernel"):
            gp._compute_kernel(X, X)


class TestGaussianProcessKernels:
    """Tests for kernel matrix computations."""

    def test_rbf_kernel_symmetric(self) -> None:
        gp = GaussianProcess(kernel_type="rbf", length_scale=1.0, signal_variance=1.0)
        X = np.random.default_rng(42).standard_normal((10, 2))
        K = gp._compute_kernel(X, X)
        assert K.shape == (10, 10)
        assert_allclose(K, K.T, atol=1e-12)

    def test_rbf_kernel_diagonal(self) -> None:
        gp = GaussianProcess(kernel_type="rbf", signal_variance=3.0)
        X = np.random.default_rng(7).standard_normal((5, 1))
        K = gp._compute_kernel(X, X)
        assert_allclose(np.diag(K), 3.0 * np.ones(5), atol=1e-12)

    def test_matern32_kernel_positive_definite(self) -> None:
        gp = GaussianProcess(kernel_type="matern32", length_scale=0.5)
        X = np.random.default_rng(99).standard_normal((8, 2))
        K = gp._compute_kernel(X, X) + 1e-8 * np.eye(8)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > 0)

    def test_exponential_kernel(self) -> None:
        gp = GaussianProcess(
            kernel_type="exponential", length_scale=2.0, signal_variance=1.5
        )
        X = np.array([[0.0], [1.0], [2.0]])
        K = gp._compute_kernel(X, X)
        assert K.shape == (3, 3)
        # k(x,x) = signal_variance for exponential kernel
        assert_allclose(np.diag(K), 1.5 * np.ones(3), atol=1e-12)


class TestGaussianProcessFitPredict:
    """Tests for fitting and prediction."""

    @pytest.fixture
    def simple_data(self):
        rng = np.random.default_rng(0)
        X = np.linspace(0, 5, 20).reshape(-1, 1)
        y = np.sin(X).ravel() + 0.05 * rng.standard_normal(20)
        return X, y

    def test_fit_returns_self(self, simple_data) -> None:
        X, y = simple_data
        gp = GaussianProcess(noise_variance=0.01)
        result = gp.fit(X, y)
        assert result is gp

    def test_fit_stores_training_data(self, simple_data) -> None:
        X, y = simple_data
        gp = GaussianProcess()
        gp.fit(X, y)
        assert gp.X_train is not None
        assert gp.y_train is not None
        assert gp._L is not None
        assert gp._alpha is not None
        assert gp.X_train.shape == X.shape

    def test_predict_without_fit_raises(self) -> None:
        gp = GaussianProcess()
        with pytest.raises(RuntimeError, match="not been fitted"):
            gp.predict(np.array([[0.0]]))

    def test_predict_mean_at_training_points(self, simple_data) -> None:
        X, y = simple_data
        gp = GaussianProcess(noise_variance=1e-4, jitter=1e-8)
        gp.fit(X, y)
        mean, std = gp.predict(X, return_std=True)
        # At training points the mean should closely interpolate the data
        assert_allclose(mean, y, atol=0.15)
        # Standard deviation at training points should be small
        assert np.all(std < 0.5)

    def test_predict_shape(self, simple_data) -> None:
        X, y = simple_data
        gp = GaussianProcess()
        gp.fit(X, y)
        X_new = np.linspace(-1, 6, 30).reshape(-1, 1)
        mean, std = gp.predict(X_new, return_std=True)
        assert mean.shape == (30,)
        assert std.shape == (30,)

    def test_predict_mean_only(self, simple_data) -> None:
        X, y = simple_data
        gp = GaussianProcess()
        gp.fit(X, y)
        result = gp.predict(X, return_std=False)
        # Should return just the mean array, not a tuple
        assert isinstance(result, np.ndarray)
        assert result.shape == (20,)

    def test_1d_input_auto_reshape(self) -> None:
        """1-D array input should be auto-reshaped to column vector."""
        X = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 0.0])
        gp = GaussianProcess()
        gp.fit(X, y)
        mean = gp.predict(np.array([0.5, 1.5]), return_std=False)
        assert mean.shape == (2,)


class TestGaussianProcessLogMarginalLikelihood:
    """Tests for the log marginal likelihood."""

    def test_lml_is_finite(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.standard_normal((15, 1))
        y = np.sin(X).ravel()
        gp = GaussianProcess()
        gp.fit(X, y)
        lml = gp.log_marginal_likelihood()
        assert np.isfinite(lml)

    def test_lml_without_fit_raises(self) -> None:
        gp = GaussianProcess()
        with pytest.raises(RuntimeError, match="not been fitted"):
            gp.log_marginal_likelihood()

    def test_better_params_higher_lml(self) -> None:
        """A GP with appropriate length scale should have higher LML
        than one with a grossly wrong length scale."""
        rng = np.random.default_rng(3)
        X = np.linspace(0, 4, 25).reshape(-1, 1)
        y = np.sin(X).ravel() + 0.01 * rng.standard_normal(25)

        gp_good = GaussianProcess(length_scale=1.0, noise_variance=0.01)
        gp_good.fit(X, y)

        gp_bad = GaussianProcess(length_scale=0.001, noise_variance=0.01)
        gp_bad.fit(X, y)

        assert gp_good.log_marginal_likelihood() > gp_bad.log_marginal_likelihood()


class TestPackageSurface:
    """Regression pins for the GS-120 package __init__ cleanup."""

    def test_gaussian_process_defined_in_models_module(self) -> None:
        """GaussianProcess/SpatialCovariance live in models/, re-exported at root."""
        import geo_infer_bayes as pkg
        from geo_infer_bayes.models import gaussian_process as gp_mod

        assert GaussianProcess is gp_mod.GaussianProcess
        assert gp_mod.GaussianProcess.__module__ == (
            "geo_infer_bayes.models.gaussian_process"
        )
        assert pkg.SpatialCovariance is gp_mod.SpatialCovariance

    def test_submodule_aliases_are_real_modules(self) -> None:
        """Dead None-fallbacks removed: submodule aliases always import."""
        import geo_infer_bayes as pkg
        import geo_infer_bayes.api as api_mod
        import geo_infer_bayes.core as core_mod
        import geo_infer_bayes.models as models_mod
        import geo_infer_bayes.utils as utils_mod

        assert pkg.api is api_mod
        assert pkg.core is core_mod
        assert pkg.models is models_mod
        assert pkg.utils is utils_mod
