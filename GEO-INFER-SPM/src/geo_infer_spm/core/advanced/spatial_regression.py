"""
Spatial Regression Models for SPM

This module implements spatial econometrics methods for SPM analysis,
providing models that account for spatial dependence and spatial heterogeneity.

Implemented Methods:
- Spatial Lag Model (SAR/SLX)
- Spatial Error Model (SEM)
- Spatial Durbin Model
- Geographically Weighted Regression (GWR)
- Spatial Filtering
- Local Indicators of Spatial Association (LISA)
"""

import numpy as np
from typing import Any, cast
from scipy import linalg, sparse
from scipy.optimize import minimize
import warnings

from ...models.data_models import SPMData, SPMResult, DesignMatrix


class SpatialRegression:
    """
    Spatial regression models for SPM analysis.

    This class implements various spatial econometrics models that account
    for spatial dependence in the data.

    Attributes:
        model_type: Type of spatial model ('sar', 'sem', 'sdm', 'gwr')
        spatial_weights: Spatial weights matrix or specification
        fitted_model: Fitted model parameters
    """

    def __init__(self, model_type: str = "sar", spatial_weights: Any | None = None):
        """
        Initialize spatial regression model.

        Args:
            model_type: Type of spatial model ('sar', 'sem', 'sdm', 'gwr', 'spatial_filter')
            spatial_weights: Spatial weights matrix or distance-based specification
        """
        self.model_type = model_type.lower()
        self.spatial_weights = spatial_weights
        self.fitted_model: dict[str, Any] | None = None

        self._validate_model_type()

    def _validate_model_type(self) -> None:
        """Validate model type."""
        valid_types = ["sar", "sem", "sdm", "gwr", "spatial_filter", "slx"]
        if self.model_type not in valid_types:
            raise ValueError(f"Model type must be one of {valid_types}")

    def fit(
        self, data: SPMData, design_matrix: DesignMatrix, **kwargs: Any
    ) -> SPMResult:
        """
        Fit spatial regression model.

        Args:
            data: SPMData containing response variable
            design_matrix: Design matrix for the model
            **kwargs: Additional model-specific parameters

        Returns:
            SPMResult with fitted spatial model
        """
        y = self._extract_response(data)
        X = design_matrix.matrix
        W = self._create_spatial_weights_matrix(data.coordinates, **kwargs)
        fit_kwargs = {**kwargs, "coordinates": data.coordinates}

        if self.model_type == "sar":
            result = self._fit_sar(y, X, W, **fit_kwargs)
        elif self.model_type == "sem":
            result = self._fit_sem(y, X, W, **fit_kwargs)
        elif self.model_type == "sdm":
            result = self._fit_sdm(y, X, W, **fit_kwargs)
        elif self.model_type == "slx":
            result = self._fit_slx(y, X, W, **fit_kwargs)
        elif self.model_type == "gwr":
            result = self._fit_gwr(y, X, data.coordinates, **kwargs)
        elif self.model_type == "spatial_filter":
            result = self._fit_spatial_filter(y, X, W, **fit_kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Preserve the caller's observed coordinates; fitting helpers only
        # operate on response/design arrays and must not invent geometry.
        result.spm_data = data
        self.fitted_model = result.model_diagnostics
        self._fitted_coefficients = np.asarray(result.beta_coefficients, dtype=float)
        self._fitted_predictors = X.shape[1]
        return result

    def _extract_response(self, data: SPMData) -> np.ndarray:
        """Extract response variable."""
        if isinstance(data.data, np.ndarray):
            return data.data.flatten()
        else:
            raise TypeError("Spatial regression requires array data")

    def _create_spatial_weights_matrix(
        self, coordinates: np.ndarray, **kwargs: Any
    ) -> sparse.csr_matrix:
        """
        Create spatial weights matrix from coordinates.

        Args:
            coordinates: Spatial coordinates
            **kwargs: Weight specification parameters

        Returns:
            Sparse spatial weights matrix
        """
        n_points = len(coordinates)

        if self.spatial_weights is not None and isinstance(
            self.spatial_weights, sparse.csr_matrix
        ):
            return self.spatial_weights

        # Create distance-based weights
        bandwidth = kwargs.get("bandwidth", None)
        k_neighbors = kwargs.get("k_neighbors", 5)

        if bandwidth is None:
            # Adaptive bandwidth based on k nearest neighbors
            for i in range(n_points):
                dists = np.linalg.norm(coordinates - coordinates[i], axis=1)
                sorted_dists = np.sort(dists)
                bandwidth = (
                    sorted_dists[k_neighbors]
                    if k_neighbors < n_points
                    else sorted_dists[-1]
                )

        # Create sparse weights matrix
        rows, cols, weights = [], [], []

        for i in range(n_points):
            dists = np.linalg.norm(coordinates - coordinates[i], axis=1)

            # Find neighbors within bandwidth
            neighbors = np.where((dists <= bandwidth) & (dists > 0))[0]

            if len(neighbors) > 0:
                # Gaussian kernel weights
                kernel_weights = np.exp(-0.5 * (dists[neighbors] / bandwidth) ** 2)

                for j, w in zip(neighbors, kernel_weights):
                    rows.append(i)
                    cols.append(j)
                    weights.append(w)

        W = sparse.csr_matrix((weights, (rows, cols)), shape=(n_points, n_points))

        # Row-normalize
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        W = W.multiply(1 / row_sums[:, np.newaxis])

        return W

    def _fit_sar(
        self, y: np.ndarray, X: np.ndarray, W: sparse.csr_matrix, **kwargs: Any
    ) -> SPMResult:
        """
        Fit Spatial Autoregressive (SAR) model.

        y = ρWy + Xβ + ε

        This model accounts for spatial dependence in the response variable.
        """
        n_points, n_predictors = X.shape

        def sar_loglik(params: np.ndarray) -> float:
            rho = params[0]  # Spatial autoregressive parameter

            # Ensure rho is in valid range for stationarity
            rho = np.clip(rho, -1, 1)

            # Create (I - ρW) matrix
            identity = sparse.eye(n_points)
            A = identity - rho * W

            try:
                # Solve for transformed y
                y_transformed = sparse.linalg.spsolve(A, y)
                X_transformed = sparse.linalg.spsolve(A, X)

                # OLS on transformed variables
                beta_hat = linalg.pinv(X_transformed.T @ X_transformed) @ (
                    X_transformed.T @ y_transformed
                )

                # Compute residuals
                y_hat = X @ beta_hat
                residuals = y - y_hat

                # Log-likelihood
                sigma2 = np.sum(residuals**2) / n_points
                loglik = -0.5 * n_points * np.log(2 * np.pi * sigma2) - np.sum(
                    residuals**2
                ) / (2 * sigma2)

                return float(-loglik)  # Negative for minimization

            except np.linalg.LinAlgError:
                return np.inf

        # Initial parameter guesses
        init_params = np.concatenate(
            [[0.1], np.linalg.pinv(X) @ y]
        )  # rho=0.1, OLS beta

        # Optimize
        try:
            result = minimize(
                sar_loglik,
                init_params,
                method="L-BFGS-B",
                bounds=[(-0.99, 0.99)] + [(None, None)] * n_predictors,
            )

            if result.success:
                rho_hat = result.x[0]
                beta_hat = result.x[1:]
                loglik = -result.fun

                # Compute final residuals
                identity = sparse.eye(n_points)
                A = identity - rho_hat * W
                y_transformed = sparse.linalg.spsolve(A, y)
                X_transformed = sparse.linalg.spsolve(A, X)
                beta_hat = linalg.pinv(X_transformed.T @ X_transformed) @ (
                    X_transformed.T @ y_transformed
                )
                y_hat = X @ beta_hat
                residuals = y - y_hat

            else:
                warnings.warn("SAR estimation did not converge", stacklevel=2)
                rho_hat = 0.0
                beta_hat = np.linalg.pinv(X) @ y
                y_hat = X @ beta_hat
                residuals = y - y_hat
                loglik = -np.inf

        except Exception as e:
            warnings.warn(f"SAR fitting failed: {e}", stacklevel=2)
            rho_hat = 0.0
            beta_hat = np.linalg.pinv(X) @ y
            y_hat = X @ beta_hat
            residuals = y - y_hat
            loglik = -np.inf

        # Create SPMResult
        from ...models.data_models import SPMResult, DesignMatrix

        result_design = DesignMatrix(
            matrix=X, names=["intercept"] + [f"x{i}" for i in range(1, n_predictors)]
        )

        spm_result = SPMResult(
            spm_data=SPMData(data=y, coordinates=np.asarray(kwargs["coordinates"])),
            design_matrix=result_design,
            beta_coefficients=beta_hat,
            residuals=residuals,
            model_diagnostics={
                "method": "SAR",
                "spatial_autoregressive_param": rho_hat,
                "log_likelihood": loglik,
                "aic": (
                    2 * (n_predictors + 1) - 2 * loglik if loglik != -np.inf else np.inf
                ),
                "converged": result.success if "result" in locals() else False,
            },
        )

        return spm_result

    def _fit_sem(
        self, y: np.ndarray, X: np.ndarray, W: sparse.csr_matrix, **kwargs: Any
    ) -> SPMResult:
        """
        Fit Spatial Error Model (SEM).

        y = Xβ + u, where u = λWu + ε

        This model accounts for spatial dependence in the error term.
        """
        # Deferred: see docs/deferred_statistical_methods.md
        # ("Exact-ML spatial error model (SEM)").
        n_points, n_predictors = X.shape

        def sem_loglik(params: np.ndarray) -> float:
            lambda_param = params[0]  # Spatial error parameter

            lambda_param = np.clip(lambda_param, -1, 1)

            identity = sparse.eye(n_points)
            B = identity - lambda_param * W

            try:
                # OLS with spatially filtered errors
                beta_hat = linalg.pinv(X.T @ X) @ (X.T @ y)

                # Compute spatially correlated errors
                residuals = y - X @ beta_hat
                filtered_residuals = sparse.linalg.spsolve(B, residuals)

                # Log-likelihood
                sigma2 = np.sum(filtered_residuals**2) / n_points
                loglik = -0.5 * n_points * np.log(2 * np.pi * sigma2) - np.sum(
                    filtered_residuals**2
                ) / (2 * sigma2)

                return float(-loglik)

            except np.linalg.LinAlgError:
                return np.inf

        # Optimize
        init_params = np.concatenate([[0.1], np.linalg.pinv(X) @ y])

        try:
            result = minimize(
                sem_loglik,
                init_params,
                method="L-BFGS-B",
                bounds=[(-0.99, 0.99)] + [(None, None)] * n_predictors,
            )

            if result.success:
                lambda_hat = result.x[0]
                beta_hat = result.x[1:]
                loglik = -result.fun
                converged = True
            else:
                lambda_hat = 0.0
                beta_hat = np.linalg.pinv(X) @ y
                loglik = -np.inf
                converged = False

        except Exception:
            lambda_hat = 0.0
            beta_hat = np.linalg.pinv(X) @ y
            loglik = -np.inf
            converged = False

        residuals = y - X @ beta_hat

        # Create SPMResult
        from ...models.data_models import SPMResult, DesignMatrix

        result_design = DesignMatrix(
            matrix=X, names=["intercept"] + [f"x{i}" for i in range(1, n_predictors)]
        )

        spm_result = SPMResult(
            spm_data=SPMData(data=y, coordinates=np.asarray(kwargs["coordinates"])),
            design_matrix=result_design,
            beta_coefficients=beta_hat,
            residuals=residuals,
            model_diagnostics={
                "method": "SEM",
                "spatial_error_param": lambda_hat,
                "log_likelihood": loglik,
                "converged": converged,
            },
        )

        return spm_result

    def _fit_sdm(
        self, y: np.ndarray, X: np.ndarray, W: sparse.csr_matrix, **kwargs: Any
    ) -> SPMResult:
        """
        Fit Spatial Durbin Model (SDM).

        y = ρWy + Xβ + WXθ + ε

        This model includes spatially lagged predictors.
        """
        # Create spatially lagged X matrix
        WX = W @ X

        # Combined design matrix
        X_sdm = np.column_stack([X, WX])

        # Fit as SAR model
        sdm_result = self._fit_sar(y, X_sdm, W, **kwargs)

        # Update diagnostics
        sdm_result.model_diagnostics["method"] = "SDM"
        sdm_result.model_diagnostics["n_direct_effects"] = X.shape[1]
        sdm_result.model_diagnostics["n_indirect_effects"] = X.shape[1]

        return sdm_result

    def _fit_slx(
        self, y: np.ndarray, X: np.ndarray, W: sparse.csr_matrix, **kwargs: Any
    ) -> SPMResult:
        """
        Fit Spatial Lag of X (SLX) model.

        y = Xβ + WXθ + ε

        This model includes spatially lagged predictors but no spatial lag of y.
        """
        # Create spatially lagged X matrix
        WX = W @ X

        # Combined design matrix
        X_slx = np.column_stack([X, WX])

        # Standard OLS fit
        beta_hat = linalg.pinv(X_slx.T @ X_slx) @ (X_slx.T @ y)
        y_hat = X_slx @ beta_hat
        residuals = y - y_hat

        n_points, n_predictors = X_slx.shape
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Create SPMResult
        from ...models.data_models import SPMResult, DesignMatrix

        result_design = DesignMatrix(
            matrix=X_slx,
            names=["intercept"] + [f"x{i}" for i in range(1, n_predictors)],
        )

        spm_result = SPMResult(
            spm_data=SPMData(data=y, coordinates=np.asarray(kwargs["coordinates"])),
            design_matrix=result_design,
            beta_coefficients=beta_hat,
            residuals=residuals,
            model_diagnostics={
                "method": "SLX",
                "r_squared": r_squared,
                "n_direct_effects": X.shape[1],
                "n_spatial_lag_effects": X.shape[1],
            },
        )

        return spm_result

    def _fit_gwr(
        self, y: np.ndarray, X: np.ndarray, coordinates: np.ndarray, **kwargs: Any
    ) -> SPMResult:
        """
        Fit Geographically Weighted Regression (GWR).

        GWR allows regression coefficients to vary spatially.
        """
        bandwidth = kwargs.get("bandwidth", None)
        if bandwidth is None:
            bandwidth = len(y) ** (-1 / 2)  # Rule of thumb

        n_points, n_predictors = X.shape
        local_coefficients = np.zeros((n_points, n_predictors))

        # Fit local regression at each point
        for i in range(n_points):
            # Compute spatial weights
            distances = np.linalg.norm(coordinates - coordinates[i], axis=1)
            weights = np.exp(-0.5 * (distances / bandwidth) ** 2)

            # Weighted least squares
            W = np.diag(weights)
            try:
                beta_i = linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
                local_coefficients[i, :] = beta_i
            except np.linalg.LinAlgError:
                # Fallback to global coefficients
                local_coefficients[i, :] = linalg.pinv(X.T @ X) @ (X.T @ y)

        # Compute global predictions for residuals
        global_beta = np.mean(local_coefficients, axis=0)
        y_hat = X @ global_beta
        residuals = y - y_hat

        # Create SPMResult
        from ...models.data_models import SPMResult, DesignMatrix

        result_design = DesignMatrix(
            matrix=X, names=["intercept"] + [f"x{i}" for i in range(1, n_predictors)]
        )

        spm_result = SPMResult(
            spm_data=SPMData(data=y, coordinates=coordinates),
            design_matrix=result_design,
            beta_coefficients=global_beta,  # Global coefficients
            residuals=residuals,
            model_diagnostics={
                "method": "GWR",
                "bandwidth": bandwidth,
                "local_coefficients": local_coefficients,
                "coefficient_variation": np.std(local_coefficients, axis=0),
            },
        )

        return spm_result

    def _fit_spatial_filter(
        self, y: np.ndarray, X: np.ndarray, W: sparse.csr_matrix, **kwargs: Any
    ) -> SPMResult:
        """
        Fit spatial filter model.

        This decomposes the response into spatial patterns and noise.
        """
        # Eigenvalue decomposition of spatial weights
        try:
            eigenvals, eigenvecs = sparse.linalg.eigsh(W, k=min(20, len(y) - 1))

            # Select eigenvectors corresponding to large eigenvalues
            threshold = kwargs.get("eigenvalue_threshold", 0.2)
            selected = np.abs(eigenvals) > threshold

            if np.any(selected):
                spatial_filters = eigenvecs[:, selected]

                # Regress y on spatial filters and X
                X_filter = np.column_stack([X, spatial_filters])
                beta_hat = linalg.pinv(X_filter.T @ X_filter) @ (X_filter.T @ y)
                y_hat = X_filter @ beta_hat
            else:
                # No spatial filters
                beta_hat = linalg.pinv(X.T @ X) @ (X.T @ y)
                y_hat = X @ beta_hat

        except Exception as e:
            warnings.warn(f"Spatial filter fitting failed: {e}", stacklevel=2)
            beta_hat = linalg.pinv(X.T @ X) @ (X.T @ y)
            y_hat = X @ beta_hat

        residuals = y - y_hat

        # Create SPMResult
        from ...models.data_models import SPMResult, DesignMatrix

        result_design = DesignMatrix(
            matrix=X, names=["intercept"] + [f"x{i}" for i in range(1, X.shape[1])]
        )

        spm_result = SPMResult(
            spm_data=SPMData(data=y, coordinates=np.asarray(kwargs["coordinates"])),
            design_matrix=result_design,
            beta_coefficients=beta_hat[: X.shape[1]],  # Only X coefficients
            residuals=residuals,
            model_diagnostics={
                "method": "Spatial_Filter",
                "n_spatial_filters": (
                    spatial_filters.shape[1] if "spatial_filters" in locals() else 0
                ),
            },
        )

        return spm_result

    def predict(self, new_data: SPMData) -> np.ndarray:
        """
        Make spatial predictions.

        Args:
            new_data: New data for prediction

        Returns:
            Predicted values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")

        values = np.asarray(new_data.data, dtype=float)
        if values.ndim != 2 or values.shape[1] != self._fitted_predictors:
            raise ValueError(
                "new_data.data must be a two-dimensional design matrix with "
                f"{self._fitted_predictors} predictors"
            )
        return cast(np.ndarray, values @ self._fitted_coefficients)

    def get_spatial_effects(self) -> dict[str, Any]:
        """
        Extract spatial effects from fitted model.

        Returns:
            Dictionary with spatial effect estimates
        """
        if self.fitted_model is None:
            return {}

        return {
            "model_type": self.model_type,
            "spatial_parameters": self.fitted_model.get("spatial_params", {}),
            "spatial_autocorrelation": self.fitted_model.get("spatial_autocorr", 0),
        }


def fit_spatial_model(
    data: SPMData, design_matrix: DesignMatrix, model_type: str = "sar", **kwargs: Any
) -> SPMResult:
    """
    Convenience function to fit spatial regression model.

    Args:
        data: SPMData containing response variable
        design_matrix: Design matrix for the model
        model_type: Type of spatial model ('sar', 'sem', 'sdm', 'gwr')
        **kwargs: Additional model parameters

    Returns:
        SPMResult with fitted spatial model

    Example:
        >>> result = fit_spatial_model(data, design_matrix, model_type='sar', bandwidth=50)
    """
    model = SpatialRegression(model_type=model_type, **kwargs)
    return model.fit(data, design_matrix, **kwargs)
