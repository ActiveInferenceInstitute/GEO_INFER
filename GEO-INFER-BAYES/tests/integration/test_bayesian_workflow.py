"""
Integration test: end-to-end Bayesian inference workflow.

Tests a complete pipeline from data preparation through model
fitting to posterior analysis, using the high-level GaussianProcess
class and lower-level components.
"""

import numpy as np


class TestGaussianProcessEndToEnd:
    """End-to-end test of the GaussianProcess convenience class."""

    def test_fit_predict_1d_sine(self) -> None:
        """Fit a GP to noisy sine data and verify predictions."""
        from geo_infer_bayes import GaussianProcess

        rng = np.random.default_rng(42)
        X_train = np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)
        y_train = np.sin(X_train).ravel() + 0.05 * rng.standard_normal(30)

        gp = GaussianProcess(
            kernel_type="rbf",
            length_scale=1.0,
            signal_variance=1.0,
            noise_variance=0.01,
        )
        gp.fit(X_train, y_train)

        X_test = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
        mean, std = gp.predict(X_test, return_std=True)

        assert mean.shape == (50,)
        assert std.shape == (50,)
        assert np.all(std > 0)
        # Mean predictions should roughly follow sin(x)
        residual = np.abs(mean - np.sin(X_test).ravel())
        assert np.mean(residual) < 0.3

    def test_fit_predict_2d_data(self) -> None:
        """Fit a GP to 2-D spatial data."""
        from geo_infer_bayes import GaussianProcess

        rng = np.random.default_rng(7)
        n = 40
        X_train = rng.random((n, 2)) * 5.0
        y_train = (
            np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + 0.1 * rng.standard_normal(n)
        )

        gp = GaussianProcess(
            kernel_type="rbf",
            length_scale=1.5,
            noise_variance=0.02,
        )
        gp.fit(X_train, y_train)

        X_test = rng.random((20, 2)) * 5.0
        mean = gp.predict(X_test, return_std=False)
        assert mean.shape == (20,)
        assert np.all(np.isfinite(mean))

    def test_log_marginal_likelihood(self) -> None:
        """LML should be computable after fitting."""
        from geo_infer_bayes import GaussianProcess

        rng = np.random.default_rng(3)
        X = np.linspace(0, 3, 15).reshape(-1, 1)
        y = np.cos(X).ravel() + 0.05 * rng.standard_normal(15)

        gp = GaussianProcess(length_scale=0.8, noise_variance=0.01)
        gp.fit(X, y)

        lml = gp.log_marginal_likelihood()
        assert np.isfinite(lml)


class TestSpatialGPEndToEnd:
    """End-to-end test using the SpatialGP model from models/spatial_gp.py."""

    def test_spatial_gp_fit_predict(self) -> None:
        from geo_infer_bayes.models.spatial_gp import SpatialGP

        rng = np.random.default_rng(10)
        X = rng.random((25, 2))
        y = np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1]) + 0.05 * rng.standard_normal(25)

        model = SpatialGP(kernel="rbf", lengthscale=0.3, variance=1.0, noise=0.05)
        model.fit(X, y)

        X_new = rng.random((10, 2))
        y_pred, y_std = model.predict(X_new, return_std=True)
        assert y_pred.shape == (10,)
        assert y_std.shape == (10,)
        assert np.all(y_std > 0)

    def test_spatial_gp_log_likelihood(self) -> None:
        from geo_infer_bayes.models.spatial_gp import SpatialGP

        rng = np.random.default_rng(11)
        X = rng.random((20, 2))
        y = X[:, 0] + 0.1 * rng.standard_normal(20)

        model = SpatialGP(kernel="rbf", lengthscale=0.5, variance=1.0, noise=0.1)
        model.fit(X, y)

        theta = {"lengthscale": 0.5, "variance": 1.0, "noise": 0.1}
        data = {"X": X, "y": y}
        ll = model.log_likelihood(theta, data)
        assert np.isfinite(ll)


class TestDataProcessingWorkflow:
    """Integration test for the data processing pipeline."""

    def test_prepare_validate_grid_workflow(self) -> None:
        import pandas as pd
        from geo_infer_bayes.utils.data_processing import (
            prepare_spatial_data,
            validate_spatial_data,
            create_spatial_grid,
        )

        rng = np.random.default_rng(31)
        df = pd.DataFrame(
            {
                "lat": rng.uniform(30, 40, 50),
                "lon": rng.uniform(-80, -70, 50),
                "measurement": rng.standard_normal(50),
            }
        )

        coords, values, temporal, metadata = prepare_spatial_data(
            df, value_col="measurement"
        )
        assert coords.shape == (50, 2)

        validation = validate_spatial_data(coords, values)
        assert validation["is_valid"] is True

        grid, grid_meta = create_spatial_grid(
            metadata["spatial_bounds"], resolution=1.0
        )
        assert grid.shape[1] == 2
        assert grid_meta["n_points"] > 0


class TestModelComparisonWorkflow:
    """Integration test for model comparison."""

    def test_compare_two_gp_models(self) -> None:
        from geo_infer_bayes.core.model_comparison import ModelComparison
        from geo_infer_bayes.models.spatial_gp import SpatialGP

        rng = np.random.default_rng(0)

        model_rbf = SpatialGP(kernel="rbf", lengthscale=0.5)
        model_rbf.name = "SpatialGP_RBF"
        model_exp = SpatialGP(kernel="exponential", lengthscale=0.5)
        model_exp.name = "SpatialGP_Exp"

        # Use pre-computed log-likelihood matrix for speed
        ll_matrix = rng.normal(-2.0, 0.3, size=(30, 15))
        data = {
            "observations": rng.standard_normal(15),
            "log_likelihood_matrix": ll_matrix,
        }

        mc = ModelComparison(models=[model_rbf, model_exp])
        results = mc.compare_models(data, method="waic")
        assert "ranking" in results
        assert len(results["ranking"]) == 2

    def test_aic_bic_comparison(self) -> None:
        from geo_infer_bayes.core.model_comparison import ModelComparison

        mc = ModelComparison()
        aic_simple = mc.compute_aic(log_likelihood=-100, n_params=2)
        aic_complex = mc.compute_aic(log_likelihood=-98, n_params=10)

        bic_simple = mc.compute_bic(log_likelihood=-100, n_params=2, n_obs=100)
        bic_complex = mc.compute_bic(log_likelihood=-98, n_params=10, n_obs=100)

        # Complex model has higher likelihood but more params
        # BIC should penalise the complex model more heavily
        assert bic_complex > bic_simple or aic_complex > aic_simple
