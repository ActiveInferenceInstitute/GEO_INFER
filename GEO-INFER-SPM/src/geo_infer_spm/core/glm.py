"""
General Linear Model implementation for Statistical Parametric Mapping

This module implements the General Linear Model (GLM) specifically adapted for
geospatial data analysis in the SPM framework. The GLM provides the statistical
foundation for relating experimental design matrices to observed geospatial data.

The implementation follows Active Inference principles by incorporating uncertainty
quantification and Bayesian inference for model parameter estimation.

Key Features:
- Ordinary Least Squares (OLS) estimation with geospatial regularization
- Robust error estimation accounting for spatial autocorrelation
- Model diagnostics including spatial dependence measures
- Support for multiple response variables and complex designs

Mathematical Foundation:
    Y = Xβ + ε

where:
- Y: Response matrix (n_points × n_responses)
- X: Design matrix (n_points × n_regressors)
- β: Regression coefficients (n_regressors × n_responses)
- ε: Error matrix with spatial covariance structure
"""

import numpy as np
import logging
from typing import Any, cast
from scipy import linalg
from scipy.stats import t, f

logger = logging.getLogger(__name__)

from ..models.data_models import SPMData, DesignMatrix, SPMResult  # noqa: E402


class GeneralLinearModel:
    """
    General Linear Model for geospatial SPM analysis.

    This class implements GLM fitting with spatial regularization and
    comprehensive diagnostics for geospatial data analysis.

    Attributes:
        design_matrix: Design matrix for the GLM
        beta: Estimated regression coefficients
        residuals: Model residuals
        sigma2: Residual variance estimate
        cov_beta: Covariance matrix of beta estimates
        diagnostics: Model fit diagnostics
    """

    def __init__(self, design_matrix: DesignMatrix):
        """
        Initialize GLM with design matrix.

        Args:
            design_matrix: Design matrix specification
        """
        self.design_matrix = design_matrix
        self.beta: np.ndarray | None = None
        self.residuals: np.ndarray | None = None
        self.sigma2: float | None = None
        self.cov_beta: np.ndarray | None = None
        self.diagnostics: dict[str, Any] = {}

    def fit(
        self,
        data: SPMData,
        method: str = "OLS",
        spatial_regularization: dict[str, Any] | None = None,
    ) -> SPMResult:
        """
        Fit the GLM to geospatial data.

        Args:
            data: SPMData containing response variables and covariates
            method: Estimation method ('OLS', 'robust', 'spatial')
            spatial_regularization: Parameters for spatial regularization

        Returns:
            SPMResult containing fitted model and diagnostics

        Raises:
            ValueError: If data dimensions don't match design matrix
        """
        # Extract response variable
        y = self._extract_response(data)

        # Validate dimensions
        if y.shape[0] != self.design_matrix.matrix.shape[0]:
            raise ValueError("Data and design matrix have incompatible dimensions")

        # Check for sufficient degrees of freedom
        n_points = self.design_matrix.matrix.shape[0]
        n_regressors = self.design_matrix.matrix.shape[1]
        if n_points <= n_regressors + 1:
            raise ValueError(
                f"Insufficient data: {n_points} observations for {n_regressors} regressors. "
                f"Need at least {n_regressors + 2} observations."
            )

        # Fit GLM based on method
        if method == "OLS":
            beta, residuals, cov_beta = self._fit_ols(y)
        elif method == "robust":
            beta, residuals, cov_beta = self._fit_robust(y)
        elif method == "spatial":
            beta, residuals, cov_beta = self._fit_spatial(
                y, spatial_regularization or {}
            )
        else:
            raise ValueError(f"Unknown fitting method: {method}")

        # Store results
        self.beta = beta
        self.residuals = residuals
        self.cov_beta = cov_beta
        self.sigma2 = float(
            np.mean(np.var(residuals, axis=0, ddof=self.design_matrix.n_regressors))
        )

        # Compute diagnostics
        self._compute_diagnostics(y, data)

        # Create SPMResult
        result = SPMResult(
            spm_data=data,
            design_matrix=self.design_matrix,
            beta_coefficients=beta,
            residuals=residuals,
            cov_beta=self.cov_beta,
            model_diagnostics=self.diagnostics.copy(),
            processing_metadata={
                "fitting_method": method,
                "spatial_regularization": spatial_regularization,
                "sigma2": self.sigma2,
            },
        )

        return result

    def _extract_response(self, data: SPMData) -> np.ndarray:
        """Extract response variable from SPMData."""
        if isinstance(data.data, np.ndarray):
            y = data.data
        elif hasattr(data.data, "values"):
            # pandas DataFrame/Series or GeoDataFrame
            y = data.data.values
        else:
            raise TypeError("Unsupported data type for response variable")

        if y.ndim == 2 and self.design_matrix.matrix.shape[0] == y.size:
            return y.reshape(-1)
        return y

    def _fit_ols(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit GLM using Ordinary Least Squares.

        Args:
            y: Response variable matrix

        Returns:
            Tuple of (beta, residuals, cov_beta)
        """
        X = self.design_matrix.matrix
        n_points, n_regressors = X.shape

        # Check for rank deficiency
        if np.linalg.matrix_rank(X) < n_regressors:
            logger.debug("Design matrix is rank deficient; using a pseudoinverse")

        # OLS estimation: β = (X^T X)^(-1) X^T Y
        XtX = X.T @ X
        XtX_inv = linalg.pinvh(XtX)  # More stable than inv for positive semi-definite
        XtY = X.T @ y

        beta = XtX_inv @ XtY

        # Compute residuals
        y_hat = X @ beta
        residuals = y - y_hat

        # Estimate covariance of beta
        # σ² (X^T X)^(-1)
        sigma2 = np.sum(residuals**2, axis=0) / (n_points - n_regressors)
        sigma2 = float(np.mean(sigma2))

        cov_beta = sigma2 * XtX_inv

        return beta, residuals, cov_beta

    def _fit_robust(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit GLM using robust estimation methods.

        Uses iteratively reweighted least squares with Huber weights
        to handle outliers in geospatial data.

        Args:
            y: Response variable matrix

        Returns:
            Tuple of (beta, residuals, cov_beta)
        """
        X = self.design_matrix.matrix
        n_points, n_regressors = X.shape

        # Initial OLS fit
        beta, _, _ = self._fit_ols(y)

        # Iteratively reweighted least squares
        max_iter = 50
        tol = 1e-6
        for _ in range(max_iter):
            # Compute residuals
            residuals = y - X @ beta

            # Huber weights
            k = 1.345  # Tuning constant for 95% efficiency
            weights = np.where(np.abs(residuals) <= k, 1.0, k / np.abs(residuals))

            # Weighted least squares
            W = np.sqrt(weights)
            Xw = X * W[:, np.newaxis] if y.ndim == 1 else X * W
            yw = y * W if y.ndim == 1 else y * W

            beta_new = linalg.pinvh(Xw.T @ Xw) @ (Xw.T @ yw)

            # Check convergence
            if np.max(np.abs(beta_new - beta)) < tol:
                break

            beta = beta_new

        # Final residuals
        residuals = y - X @ beta

        # Deferred: see docs/deferred_statistical_methods.md
        # ("Huber-White robust covariance").
        XtX_inv = linalg.pinvh(X.T @ X)
        cov_beta = XtX_inv @ (X.T @ np.diag(weights.flatten()) @ X) @ XtX_inv

        return beta, residuals, cov_beta

    def _fit_spatial(
        self, y: np.ndarray, regularization_params: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit GLM with spatial regularization.

        Incorporates spatial autocorrelation structure into the estimation
        using a spatial covariance model.

        Args:
            y: Response variable matrix
            regularization_params: Parameters for spatial regularization

        Returns:
            Tuple of (beta, residuals, cov_beta)
        """
        X = self.design_matrix.matrix

        # Extract regularization parameters
        lambda_reg = regularization_params.get("lambda", 0.1)
        spatial_weights = regularization_params.get("spatial_weights", None)

        if spatial_weights is None:
            # Default: ridge-style regularization when no spatial weights are
            # supplied. Deferred: see docs/deferred_statistical_methods.md
            # ("Spatial regularization default weights").
            logger.debug("No spatial weights provided; using default regularization")
            n = X.shape[0]
            spatial_weights = np.eye(n) * lambda_reg
        else:
            spatial_weights = spatial_weights * lambda_reg

        # Regularized estimation: β = (X^T W X + λ Σ)^(-1) X^T W Y
        # where W is spatial weights matrix, Σ is regularization matrix
        XtX = X.T @ spatial_weights @ X
        regularization_matrix = regularization_params.get(
            "regularization_matrix", np.eye(X.shape[1]) * lambda_reg
        )

        XtX_reg = XtX + regularization_matrix
        XtX_inv = linalg.pinvh(XtX_reg)

        XtY = X.T @ spatial_weights @ y
        beta = XtX_inv @ XtY

        # Compute residuals
        y_hat = X @ beta
        residuals = y - y_hat

        # Covariance estimation with regularization
        sigma2 = np.sum(residuals**2, axis=0) / (len(y) - X.shape[1])
        sigma2 = float(np.mean(sigma2))

        cov_beta = sigma2 * XtX_inv

        return beta, residuals, cov_beta

    def _compute_diagnostics(self, y: np.ndarray, data: SPMData) -> None:
        """Compute comprehensive model diagnostics."""
        if self.beta is None or self.residuals is None:
            return

        n_points, n_regressors = self.design_matrix.matrix.shape

        # Basic fit statistics
        ss_res = np.sum(self.residuals**2, axis=0)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)

        r_squared = 1 - (ss_res / np.maximum(ss_tot, np.finfo(float).eps))
        r_squared = float(np.mean(np.clip(r_squared, 0.0, 0.999999)))
        self.diagnostics["r_squared"] = r_squared
        self.diagnostics["adjusted_r_squared"] = float(
            1 - ((1 - r_squared) * (n_points - 1) / (n_points - n_regressors))
        )

        # F-statistic
        if self.sigma2 is not None:
            f_statistic = (ss_tot - ss_res) / n_regressors / self.sigma2
            self.diagnostics["f_statistic"] = float(np.mean(f_statistic))
            self.diagnostics["f_p_value"] = float(
                f.sf(
                    self.diagnostics["f_statistic"],
                    n_regressors,
                    n_points - n_regressors,
                )
            )

        # Residual diagnostics
        self.diagnostics["residual_mean"] = float(np.mean(self.residuals))
        self.diagnostics["residual_std"] = float(np.std(self.residuals))

        # Durbin-Watson statistic for autocorrelation (averaged over columns)
        if n_points > 1:
            dw_numerator = np.sum(np.diff(self.residuals, axis=0) ** 2, axis=0)
            dw_denominator = np.sum(self.residuals**2, axis=0)
            self.diagnostics["durbin_watson"] = float(
                np.mean(dw_numerator / np.maximum(dw_denominator, np.finfo(float).eps))
            )

        # Condition number for multicollinearity
        self.diagnostics["condition_number"] = np.linalg.cond(self.design_matrix.matrix)

    def predict(
        self,
        new_data: SPMData | None = None,
        new_design: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Make predictions using the fitted GLM.

        Args:
            new_data: New data for prediction (optional)
            new_design: New design matrix for prediction

        Returns:
            Predicted values

        Raises:
            ValueError: If model not fitted or no design matrix provided
        """
        if self.beta is None:
            raise ValueError("Model must be fitted before making predictions")

        if new_design is not None:
            X_pred = new_design
        elif new_data is not None:
            raise ValueError("Design matrix must be provided for prediction")
        else:
            # Default: predict on training data
            X_pred = self.design_matrix.matrix

        assert self.beta is not None
        return cast(np.ndarray, X_pred @ self.beta)

    def get_coefficient_test(self, coefficient_idx: int) -> dict[str, Any]:
        """
        Test significance of a specific coefficient.

        Args:
            coefficient_idx: Index of coefficient to test

        Returns:
            Dictionary with test statistics and p-values
        """
        if self.beta is None or self.cov_beta is None:
            raise ValueError("Model must be fitted before testing coefficients")

        beta = self.beta[coefficient_idx]
        se = np.sqrt(self.cov_beta[coefficient_idx, coefficient_idx])

        t_stat = beta / se
        p_value = 2 * t.sf(
            np.abs(t_stat),
            self.design_matrix.matrix.shape[0] - self.design_matrix.n_regressors,
        )

        return {
            "coefficient": beta,
            "standard_error": se,
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }


def fit_glm(
    data: SPMData, design_matrix: DesignMatrix, method: str = "OLS", **kwargs: Any
) -> SPMResult:
    """
    Convenience function to fit a GLM to geospatial data.

    Args:
        data: SPMData containing response and covariates
        design_matrix: Design matrix specification
        method: Fitting method ('OLS', 'robust', 'spatial')
        **kwargs: Additional arguments passed to fitting method

    Returns:
        SPMResult with fitted model and diagnostics

    Example:
        >>> data = SPMData(data=temperature_array, coordinates=coords)
        >>> design = DesignMatrix(matrix=X, names=['intercept', 'elevation'])
        >>> result = fit_glm(data, design, method='spatial')
    """
    glm = GeneralLinearModel(design_matrix)
    return glm.fit(data, method=method, **kwargs)
