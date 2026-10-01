"""
Unit tests for Bayesian analysis functionality
"""

import numpy as np
import pytest
from types import SimpleNamespace

from geo_infer_spm.models.data_models import SPMData, DesignMatrix
from geo_infer_spm.core.bayesian import BayesianSPM, gelman_rubin_r_hat


class TestBayesianSPM:
    """Test BayesianSPM class functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_points = 50
        n_regressors = 3

        # Create test data
        self.coordinates = np.column_stack(
            [
                np.random.uniform(-180, 180, n_points),
                np.random.uniform(-90, 90, n_points),
            ]
        )
        self.X = np.random.randn(n_points, n_regressors)
        self.beta_true = np.array([1.0, 2.0, -1.5])
        self.y = self.X @ self.beta_true + 0.1 * np.random.randn(n_points)

        self.spm_data = SPMData(
            data=self.y, coordinates=self.coordinates, crs="EPSG:4326"
        )
        self.design_matrix = DesignMatrix(
            matrix=self.X, names=["intercept", "x1", "x2"]
        )

        self.bayesian_spm = BayesianSPM()

    def test_initialization(self):
        """Test BayesianSPM initialization."""
        assert self.bayesian_spm.model_type == "hierarchical_glm"
        assert self.bayesian_spm.model_spec is None
        assert isinstance(self.bayesian_spm.priors, dict)

    def test_default_priors(self):
        """Test default prior specification."""
        priors = self.bayesian_spm._default_priors(3)

        required_keys = ["beta", "sigma", "beta_intercept"]
        for key in required_keys:
            assert key in priors
            assert "type" in priors[key]

    def test_empirical_bayes_glm(self):
        """Test empirical Bayes GLM fitting."""
        # Force empirical Bayes mode
        self.bayesian_spm.model_type = "empirical_bayes"

        result = self.bayesian_spm.fit_bayesian_glm(
            self.spm_data, self.design_matrix.matrix, n_samples=100, n_tune=50
        )

        assert result.beta_coefficients.shape == (3,)
        assert result.residuals.shape == (50,)
        assert "method" in result.model_diagnostics
        assert result.model_diagnostics["method"] == "Empirical_Bayes_GLM"

    def test_fit_works_with_or_without_pymc(self):
        """Fitting succeeds via PyMC >= 5 when installed, else empirical Bayes."""
        result = self.bayesian_spm.fit_bayesian_glm(
            self.spm_data, self.design_matrix.matrix, n_samples=50, n_tune=25
        )
        assert result is not None

    def test_posterior_probability_map(self):
        """Test posterior probability map computation."""
        # First fit a model
        self.bayesian_spm.model_type = "empirical_bayes"
        _result = self.bayesian_spm.fit_bayesian_glm(
            self.spm_data, self.design_matrix.matrix
        )

        # Create mock statistical map
        stat_map = np.random.randn(50)

        # This should work with empirical Bayes
        posterior_prob = self.bayesian_spm.posterior_probability_map(
            stat_map, threshold=0.95
        )

        assert posterior_prob.shape == stat_map.shape
        assert np.all((posterior_prob >= 0) & (posterior_prob <= 1))

    def test_bayesian_model_comparison(self):
        """Test Bayesian model comparison."""
        # Create two simple models
        self.bayesian_spm.model_type = "empirical_bayes"

        model1 = self.bayesian_spm.fit_bayesian_glm(
            self.spm_data,
            self.design_matrix.matrix[:, :2],  # Fewer parameters
        )

        model2 = self.bayesian_spm.fit_bayesian_glm(
            self.spm_data,
            self.design_matrix.matrix,  # All parameters
        )

        comparison = self.bayesian_spm.bayesian_model_comparison([model1, model2])

        assert "method" in comparison
        assert "best_model_index" in comparison
        assert comparison["best_model_index"] in [0, 1]

    def test_spatial_hierarchical_model(self):
        """Test spatial hierarchical model."""
        spatial_structure = {"n_basis": 5, "scale": 10.0}

        result = self.bayesian_spm.spatial_hierarchical_model(
            self.spm_data, self.design_matrix.matrix, spatial_structure
        )

        assert result.beta_coefficients.shape[0] > 3  # Should include spatial basis
        assert result.model_diagnostics.get("spatial_hierarchical")

    def test_variational_inference(self):
        """Test variational inference approximation."""
        result = self.bayesian_spm.variational_inference(
            self.spm_data, self.design_matrix.matrix, n_iterations=20
        )

        assert result.beta_coefficients.shape == (3,)
        assert "method" in result.model_diagnostics
        assert result.model_diagnostics["method"] == "Variational_Inference"


class TestBayesianModelTypes:
    """Test different Bayesian model types."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n_points = 30

        coordinates = np.column_stack(
            [
                np.random.uniform(-180, 180, n_points),
                np.random.uniform(-90, 90, n_points),
            ]
        )
        X = np.random.randn(n_points, 2)
        y = X @ np.array([1.5, -0.5]) + 0.2 * np.random.randn(n_points)

        self.spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        self.design_matrix = DesignMatrix(matrix=X, names=["intercept", "slope"])

    def test_hierarchical_glm_model(self):
        """Test hierarchical GLM model specification."""
        bayesian_spm = BayesianSPM(model_type="hierarchical_glm")

        # Should initialize with hierarchical model type
        assert bayesian_spm.model_type == "hierarchical_glm"

        # Test fitting (falls back to empirical Bayes when PyMC is unavailable)
        result = bayesian_spm.fit_bayesian_glm(
            self.spm_data, self.design_matrix.matrix, n_samples=50, n_tune=25
        )

        assert result is not None

    def test_spatial_hierarchical_model(self):
        """Test spatial hierarchical model."""
        bayesian_spm = BayesianSPM(model_type="spatial_hierarchical")

        spatial_structure = {"n_basis": 3, "scale": 5.0}

        result = bayesian_spm.spatial_hierarchical_model(
            self.spm_data, self.design_matrix.matrix, spatial_structure
        )

        # Should include spatial basis functions
        assert result.beta_coefficients.shape[0] >= 2  # At least original parameters
        assert result.model_diagnostics.get("spatial_hierarchical")


class TestBayesianDiagnostics:
    """Test Bayesian diagnostic and utility functions."""

    def test_ess_computation_placeholder(self):
        """Test ESS computation (placeholder)."""
        bayesian_spm = BayesianSPM()

        # Mock trace object
        class MockTrace:
            def __init__(self):
                self.posterior = type(
                    "obj", (object,), {"dims": {"chain": 2, "draw": 200}}
                )()

        mock_trace = MockTrace()
        ess = bayesian_spm._compute_ess(mock_trace)

        assert isinstance(ess, np.ndarray)
        assert len(ess) == 1  # Single ESS value

    def test_spatial_basis_creation(self):
        """Test spatial basis function creation for hierarchical models."""
        bayesian_spm = BayesianSPM()

        coordinates = np.column_stack(
            [np.random.uniform(-179, 179, 20), np.random.uniform(-89, 89, 20)]
        )
        spatial_structure = {"n_basis": 5, "scale": 10.0}

        basis = bayesian_spm._create_spatial_basis(coordinates, spatial_structure)

        assert basis.shape == (20, 5)
        assert np.all(basis >= 0)  # Gaussian basis should be non-negative


class TestSplitRHat:
    """Test the split Gelman-Rubin R-hat implementation."""

    def _trace(self, posterior):
        """Wrap a posterior in a minimal trace-like namespace."""
        return SimpleNamespace(posterior=posterior)

    def test_converged_chains_have_r_hat_near_one(self):
        """I.i.d. chains from the same distribution give R-hat near 1."""
        rng = np.random.default_rng(42)
        chains = rng.normal(0.0, 1.0, size=(4, 2000))

        r_hat = gelman_rubin_r_hat(chains)

        assert r_hat.shape == (1,)
        assert 0.99 < float(r_hat[0]) < 1.05

    def test_non_converged_chains_have_r_hat_far_above_one(self):
        """Chains sampling separated distributions give R-hat well above 1."""
        rng = np.random.default_rng(42)
        means = np.array([[0.0], [5.0], [-5.0], [10.0]])
        chains = rng.normal(means, 1.0, size=(4, 2000))

        r_hat = gelman_rubin_r_hat(chains)

        assert float(r_hat[0]) > 1.5

    def test_vector_parameter_gives_per_component_values(self):
        """A (chains, draws, components) input yields one R-hat per component."""
        rng = np.random.default_rng(7)
        chains = rng.normal(0.0, 1.0, size=(3, 1500, 2))

        r_hat = gelman_rubin_r_hat(chains)

        assert r_hat.shape == (2,)
        assert np.all((r_hat > 0.9) & (r_hat < 1.1))

    def test_equal_constant_chains_are_converged(self):
        """Identical constant chains are trivially converged (R-hat 1)."""
        chains = np.ones((4, 100))

        r_hat = gelman_rubin_r_hat(chains)

        assert float(r_hat[0]) == 1.0

    def test_unequal_constant_chains_never_converge(self):
        """Constant chains that differ between chains give R-hat inf."""
        chains = np.array([[0.0] * 100, [1.0] * 100, [2.0] * 100])

        r_hat = gelman_rubin_r_hat(chains)

        assert np.isinf(float(r_hat[0]))

    def test_rejects_single_chain(self):
        """R-hat is undefined for a single chain."""
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError, match="at least 2 chains"):
            gelman_rubin_r_hat(rng.normal(size=(1, 100)))

    def test_rejects_too_few_draws(self):
        """Split-R-hat needs at least two draws per half-chain."""
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError, match="at least 4 draws"):
            gelman_rubin_r_hat(rng.normal(size=(4, 3)))

    def test_compute_r_hat_extracts_posterior_parameters(self):
        """_compute_r_hat reports one (worst-component) value per parameter."""
        import xarray as xr

        rng = np.random.default_rng(42)
        converged = rng.normal(0.0, 1.0, size=(4, 1000))
        divergent = rng.normal(
            np.array([[0.0], [8.0], [0.0], [8.0]]), 1.0, size=(4, 1000)
        )

        posterior = xr.Dataset(
            {
                "beta_intercept": (("chain", "draw"), converged),
                "beta_wandering": (("chain", "draw"), divergent),
            }
        )
        bayesian_spm = BayesianSPM()

        r_hat = bayesian_spm._compute_r_hat(self._trace(posterior))

        assert isinstance(r_hat, np.ndarray)
        assert r_hat.shape == (2,)
        assert 0.99 < float(r_hat[0]) < 1.05
        assert float(r_hat[1]) > 1.5

    def test_compute_r_hat_vector_parameter_uses_worst_component(self):
        """A vector posterior parameter contributes its maximum component."""
        import xarray as xr

        rng = np.random.default_rng(3)
        scalar = rng.normal(0.0, 1.0, size=(4, 800))
        vector = rng.normal(
            np.array([[[0.0]], [[6.0]], [[0.0]], [[6.0]]]), 1.0, size=(4, 800, 1)
        )

        posterior = xr.Dataset(
            {
                "beta_intercept": (("chain", "draw"), scalar),
                "beta_other": (("chain", "draw", "beta_other_dim_0"), vector),
            }
        )
        bayesian_spm = BayesianSPM()

        r_hat = bayesian_spm._compute_r_hat(self._trace(posterior))

        assert r_hat.shape == (2,)
        assert 0.99 < float(r_hat[0]) < 1.05
        assert float(r_hat[1]) > 1.5

    def test_compute_r_hat_reports_nan_without_samples(self):
        """A trace without posterior samples yields NaN, not a fake 1.0."""
        bayesian_spm = BayesianSPM()

        class EmptyTrace:
            pass

        r_hat = bayesian_spm._compute_r_hat(EmptyTrace())

        assert isinstance(r_hat, np.ndarray)
        assert r_hat.shape == (1,)
        assert np.isnan(float(r_hat[0]))


class TestBayesianEdgeCases:
    """Test edge cases in Bayesian analysis."""

    def test_insufficient_data(self):
        """Test behavior with very small datasets."""
        coordinates = np.array([[0.0, 0.0], [1.0, 1.0]])
        X = np.array([[1.0, 0.5], [1.0, 1.5]])
        y = np.array([1.0, 2.0])

        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(matrix=X, names=["intercept", "slope"])

        bayesian_spm = BayesianSPM()

        # Should handle small datasets gracefully
        result = bayesian_spm.fit_bayesian_glm(spm_data, design_matrix.matrix)

        assert result.beta_coefficients.shape == (2,)

    def test_rank_deficient_design(self):
        """Test with rank deficient design matrix."""
        coordinates = np.column_stack(
            [np.random.uniform(-179, 179, 10), np.random.uniform(-89, 89, 10)]
        )
        X = np.ones((10, 3))  # Rank deficient
        X[:, 1] = np.random.randn(10) * 0.01  # Nearly constant
        y = np.random.randn(10)

        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(matrix=X, names=["int", "x1", "x2"])

        bayesian_spm = BayesianSPM()

        # Should handle rank deficiency
        result = bayesian_spm.fit_bayesian_glm(spm_data, design_matrix.matrix)

        assert result.beta_coefficients.shape == (3,)

    def test_extreme_parameter_values(self):
        """Test with extreme parameter values."""
        coordinates = np.column_stack(
            [np.random.uniform(-179, 179, 20), np.random.uniform(-89, 89, 20)]
        )
        X = np.random.randn(20, 2) * 1000  # Very large values
        y = X @ np.array([0.001, -0.002]) + 0.0001 * np.random.randn(
            20
        )  # Very small coefficients

        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(matrix=X, names=["x1", "x2"])

        bayesian_spm = BayesianSPM()

        result = bayesian_spm.fit_bayesian_glm(spm_data, design_matrix.matrix)

        # Should still produce reasonable results
        assert result.beta_coefficients.shape == (2,)
        assert np.all(np.isfinite(result.beta_coefficients))

    def test_posterior_probability_edge_cases(self):
        """Test posterior probability computation edge cases."""
        bayesian_spm = BayesianSPM()

        # Test with no posterior samples
        stat_map = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Model must be fitted"):
            bayesian_spm.posterior_probability_map(stat_map)

    def test_model_comparison_single_model(self):
        """Test model comparison with single model."""
        bayesian_spm = BayesianSPM()

        # Create a simple model
        coordinates = np.column_stack(
            [np.random.uniform(-179, 179, 10), np.random.uniform(-89, 89, 10)]
        )
        X = np.random.randn(10, 2)
        y = X @ np.array([1.0, -0.5]) + 0.1 * np.random.randn(10)

        spm_data = SPMData(data=y, coordinates=coordinates, crs="EPSG:4326")
        design_matrix = DesignMatrix(matrix=X, names=["int", "slope"])

        model = bayesian_spm.fit_bayesian_glm(spm_data, design_matrix.matrix)

        # Single model comparison should work
        comparison = bayesian_spm.bayesian_model_comparison([model])

        assert "method" in comparison
        assert comparison["best_model_index"] == 0

    def test_ess_failure_reports_nan_not_fabricated_constant(self):
        """ESS failure path must report NaN (missing-value marker), not a
        plausible fake baseline constant (GS-124)."""
        bayesian_spm = BayesianSPM()

        class BrokenTrace:
            posterior = None

        ess = bayesian_spm._compute_ess(BrokenTrace())
        assert isinstance(ess, np.ndarray)
        assert len(ess) == 1
        assert np.isnan(ess[0])
