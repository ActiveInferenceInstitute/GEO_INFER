"""
Model Validation and Cross-Validation for SPM

This module provides comprehensive model validation techniques for SPM analysis,
including cross-validation, bootstrap validation, model comparison metrics,
and diagnostic tests for model adequacy.

Validation Methods:
- K-fold cross-validation
- Leave-one-out cross-validation
- Bootstrap validation
- Spatial cross-validation
- Model comparison (AIC, BIC, DIC)
- Goodness-of-fit tests
- Residual diagnostics
"""

import numpy as np
from typing import Any, cast
from collections.abc import Callable
from scipy import stats
from sklearn.model_selection import KFold, LeaveOneOut
import warnings

from ...models.data_models import SPMData, SPMResult, DesignMatrix


class ModelValidator:
    """
    Comprehensive model validation for SPM analysis.

    This class provides various validation techniques to assess model
    performance, goodness-of-fit, and generalizability.

    Attributes:
        validation_method: Primary validation method to use
        n_folds: Number of folds for cross-validation
        n_bootstraps: Number of bootstrap samples
        random_state: Random seed for reproducibility
    """

    def __init__(
        self,
        validation_method: str = "kfold",
        n_folds: int = 5,
        n_bootstraps: int = 100,
        random_state: int = 42,
    ):
        """
        Initialize model validator.

        Args:
            validation_method: Validation method ('kfold', 'loo', 'bootstrap', 'spatial')
            n_folds: Number of folds for cross-validation
            n_bootstraps: Number of bootstrap samples
            random_state: Random seed
        """
        self.validation_method = validation_method.lower()
        self.n_folds = n_folds
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state

        # Per-instance RNG; never reads or mutates process-wide np.random state.
        self._rng = np.random.default_rng(random_state)

    def cross_validate(
        self,
        model_func: Callable[..., Any],
        data: SPMData,
        design_matrix: DesignMatrix,
        **model_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform cross-validation of SPM model.

        Args:
            model_func: Function that fits the model (e.g., fit_glm)
            data: SPMData for validation
            design_matrix: Design matrix for the model
            **model_kwargs: Additional arguments for model fitting

        Returns:
            Dictionary with cross-validation results
        """
        if self.validation_method == "kfold":
            cv_results = self._kfold_cv(model_func, data, design_matrix, **model_kwargs)
        elif self.validation_method == "loo":
            cv_results = self._loo_cv(model_func, data, design_matrix, **model_kwargs)
        elif self.validation_method == "bootstrap":
            cv_results = self._bootstrap_cv(
                model_func, data, design_matrix, **model_kwargs
            )
        elif self.validation_method == "spatial":
            cv_results = self._spatial_cv(
                model_func, data, design_matrix, **model_kwargs
            )
        else:
            raise ValueError(f"Unknown validation method: {self.validation_method}")

        return cv_results

    def _kfold_cv(
        self,
        model_func: Callable[..., Any],
        data: SPMData,
        design_matrix: DesignMatrix,
        **model_kwargs: Any,
    ) -> dict[str, Any]:
        """Perform k-fold cross-validation."""
        y = self._extract_response(data)
        n_points = len(y)

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        cv_scores = []
        predictions = np.zeros(n_points)
        prediction_counts = np.zeros(n_points)

        for train_idx, test_idx in kf.split(np.arange(n_points)):
            # Fit model on training data
            train_data = self._subset_data(data, train_idx)
            train_design = self._subset_design_matrix(design_matrix, train_idx)

            try:
                model_result = model_func(train_data, train_design, **model_kwargs)

                # Predict on test data
                test_predictions = self._predict_on_subset(
                    model_result, test_idx, data, design_matrix
                )
                predictions[test_idx] += test_predictions
                prediction_counts[test_idx] += 1

                # Compute score on test set
                y_test = y[test_idx]
                mse = np.mean((y_test - test_predictions) ** 2)
                rmse = np.sqrt(mse)
                r2 = 1 - mse / np.var(y_test) if np.var(y_test) > 0 else 0

                cv_scores.append(
                    {"mse": mse, "rmse": rmse, "r2": r2, "n_test": len(test_idx)}
                )

            except Exception as e:
                warnings.warn(f"Cross-validation fold failed: {e}", stacklevel=2)
                continue

        # Average predictions
        valid_predictions = prediction_counts > 0
        predictions[valid_predictions] /= prediction_counts[valid_predictions]

        # Compute overall statistics
        overall_mse = np.mean([score["mse"] for score in cv_scores])
        overall_rmse = np.mean([score["rmse"] for score in cv_scores])
        overall_r2 = np.mean([score["r2"] for score in cv_scores])

        return {
            "method": "kfold",
            "n_folds": self.n_folds,
            "cv_scores": cv_scores,
            "overall_mse": overall_mse,
            "overall_rmse": overall_rmse,
            "overall_r2": overall_r2,
            "predictions": predictions,
            "n_successful_folds": len(cv_scores),
        }

    def _loo_cv(
        self,
        model_func: Callable[..., Any],
        data: SPMData,
        design_matrix: DesignMatrix,
        **model_kwargs: Any,
    ) -> dict[str, Any]:
        """Perform leave-one-out cross-validation."""
        y = self._extract_response(data)
        n_points = len(y)

        loo = LeaveOneOut()
        predictions = np.zeros(n_points)
        errors = []

        for train_idx, test_idx in loo.split(np.arange(n_points)):
            # Fit model leaving out one point
            train_data = self._subset_data(data, train_idx)
            train_design = self._subset_design_matrix(design_matrix, train_idx)

            try:
                model_result = model_func(train_data, train_design, **model_kwargs)

                # Predict the left-out point
                test_prediction = self._predict_on_subset(
                    model_result, test_idx, data, design_matrix
                )
                predictions[test_idx[0]] = test_prediction[0]

                # Compute error
                y_true = y[test_idx[0]]
                error = y_true - test_prediction[0]
                errors.append(error**2)

            except Exception as e:
                warnings.warn(
                    f"LOO-CV prediction failed for point {test_idx[0]}: {e}",
                    stacklevel=2,
                )
                continue

        # Compute statistics
        mse = np.mean(errors)
        rmse = np.sqrt(mse)
        r2 = 1 - mse / np.var(y) if np.var(y) > 0 else 0

        return {
            "method": "loo",
            "n_predictions": len(errors),
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "predictions": predictions,
            "errors": np.array(errors),
        }

    def _bootstrap_cv(
        self,
        model_func: Callable[..., Any],
        data: SPMData,
        design_matrix: DesignMatrix,
        **model_kwargs: Any,
    ) -> dict[str, Any]:
        """Perform bootstrap cross-validation."""
        y = self._extract_response(data)
        n_points = len(y)

        bootstrap_scores = []
        all_predictions = []

        for i in range(self.n_bootstraps):
            # Bootstrap sample
            boot_idx = self._rng.choice(n_points, size=n_points, replace=True)

            # Fit model on bootstrap sample
            boot_data = self._subset_data(data, boot_idx)
            boot_design = self._subset_design_matrix(design_matrix, boot_idx)

            try:
                model_result = model_func(boot_data, boot_design, **model_kwargs)

                # Predict on full dataset
                predictions = self._predict_on_full_dataset(
                    model_result, data, design_matrix
                )
                all_predictions.append(predictions)

                # Compute optimism-corrected score
                boot_score = self._compute_bootstrap_score(
                    model_result, boot_data, predictions[boot_idx]
                )
                bootstrap_scores.append(boot_score)

            except Exception as e:
                warnings.warn(f"Bootstrap iteration {i} failed: {e}", stacklevel=2)
                continue

        # Average predictions across bootstraps
        if all_predictions:
            avg_predictions = np.mean(all_predictions, axis=0)
        else:
            avg_predictions = np.zeros(n_points)

        # Compute overall statistics
        optimism = np.mean(bootstrap_scores) if bootstrap_scores else 0

        return {
            "method": "bootstrap",
            "n_bootstraps": len(bootstrap_scores),
            "optimism": optimism,
            "avg_predictions": avg_predictions,
            "bootstrap_scores": bootstrap_scores,
        }

    def _spatial_cv(
        self,
        model_func: Callable[..., Any],
        data: SPMData,
        design_matrix: DesignMatrix,
        **model_kwargs: Any,
    ) -> dict[str, Any]:
        """Perform spatial block cross-validation.

        Coordinates are clustered into ``n_folds`` spatial blocks with
        K-means; each block is held out in turn while the model is refit on
        the remaining blocks, so test folds are geographically contiguous
        rather than randomly interleaved.
        """
        coordinates = data.coordinates

        # Create spatial folds based on coordinate clusters
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=self.n_folds, random_state=self.random_state, n_init=10
        )
        spatial_clusters = kmeans.fit_predict(coordinates)

        cv_scores = []
        predictions = np.zeros(len(data.data))
        prediction_counts = np.zeros(len(data.data))

        for fold in range(self.n_folds):
            # Test set: one spatial cluster
            test_mask = spatial_clusters == fold
            train_mask = ~test_mask

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(test_idx) == 0:
                continue

            # Fit on training clusters
            train_data = self._subset_data(data, train_idx)
            train_design = self._subset_design_matrix(design_matrix, train_idx)

            try:
                model_result = model_func(train_data, train_design, **model_kwargs)

                # Predict on test cluster
                test_predictions = self._predict_on_subset(
                    model_result, test_idx, data, design_matrix
                )
                predictions[test_idx] += test_predictions
                prediction_counts[test_idx] += 1

                # Score
                y_test = self._extract_response(data)[test_idx]
                mse = np.mean((y_test - test_predictions) ** 2)
                r2 = 1 - mse / np.var(y_test) if np.var(y_test) > 0 else 0

                cv_scores.append(
                    {"fold": fold, "mse": mse, "r2": r2, "n_test": len(test_idx)}
                )

            except Exception as e:
                warnings.warn(f"Spatial CV fold {fold} failed: {e}", stacklevel=2)
                continue

        # Average predictions
        valid_predictions = prediction_counts > 0
        predictions[valid_predictions] /= prediction_counts[valid_predictions]

        # Overall statistics
        overall_mse = np.mean([s["mse"] for s in cv_scores]) if cv_scores else 0
        overall_r2 = np.mean([s["r2"] for s in cv_scores]) if cv_scores else 0

        return {
            "method": "spatial",
            "n_folds": self.n_folds,
            "cv_scores": cv_scores,
            "overall_mse": overall_mse,
            "overall_r2": overall_r2,
            "predictions": predictions,
            "spatial_clusters": spatial_clusters,
        }

    def _extract_response(self, data: SPMData) -> np.ndarray:
        """Extract response variable."""
        if isinstance(data.data, np.ndarray):
            return data.data.flatten()
        else:
            raise TypeError("Validation requires array data")

    def _subset_data(self, data: SPMData, indices: np.ndarray) -> SPMData:
        """Create subset of SPMData."""
        subset_data = (
            data.data[indices] if hasattr(data.data, "__getitem__") else data.data
        )

        subset_covariates = {}
        if data.covariates:
            for key, values in data.covariates.items():
                subset_covariates[key] = values[indices]

        return SPMData(
            data=subset_data,
            coordinates=data.coordinates[indices],
            time=data.time[indices] if data.time is not None else None,
            covariates=subset_covariates,
            metadata=data.metadata,
            crs=data.crs,
        )

    def _subset_design_matrix(
        self, design_matrix: DesignMatrix, indices: np.ndarray
    ) -> DesignMatrix:
        """Create subset of design matrix."""
        from ...models.data_models import DesignMatrix

        subset_matrix = design_matrix.matrix[indices]

        return DesignMatrix(
            matrix=subset_matrix,
            names=design_matrix.names,
            factors=design_matrix.factors,
            covariates=design_matrix.covariates,
        )

    def _predict_on_subset(
        self,
        model_result: SPMResult,
        test_indices: np.ndarray,
        full_data: SPMData,
        full_design: DesignMatrix,
    ) -> np.ndarray:
        """Make predictions on a subset of data."""
        # Deferred: see docs/deferred_statistical_methods.md
        # ("CV prediction layer").
        if (
            hasattr(model_result, "beta_coefficients")
            and len(model_result.beta_coefficients) > 0
        ):
            X_test = full_design.matrix[test_indices]
            return cast(np.ndarray, X_test @ model_result.beta_coefficients)
        else:
            # Fallback to mean prediction
            return np.full(
                len(test_indices), np.mean(self._extract_response(full_data))
            )

    def _predict_on_full_dataset(
        self, model_result: SPMResult, data: SPMData, design_matrix: DesignMatrix
    ) -> np.ndarray:
        """Make predictions on full dataset."""
        return self._predict_on_subset(
            model_result, np.arange(len(data.data)), data, design_matrix
        )

    def _compute_bootstrap_score(
        self, model_result: SPMResult, data: SPMData, predictions: np.ndarray
    ) -> float:
        """Compute bootstrap optimism score."""
        y = self._extract_response(data)
        residuals = y - predictions

        # Deferred: see docs/deferred_statistical_methods.md
        # ("Bootstrap optimism").
        return float(np.var(residuals))

    def compare_models(
        self, model_results: list[SPMResult], method: str = "aic"
    ) -> dict[str, Any]:
        """
        Compare multiple fitted models.

        Args:
            model_results: List of fitted SPMResult objects
            method: Comparison method ('aic', 'bic', 'dic', 'lr_test')

        Returns:
            Dictionary with model comparison results
        """
        if len(model_results) < 2:
            raise ValueError("Need at least 2 models to compare")

        if method.lower() == "aic":
            scores = [self._compute_aic(result) for result in model_results]
            best_idx = np.argmin(scores)
        elif method.lower() == "bic":
            scores = [self._compute_bic(result) for result in model_results]
            best_idx = np.argmin(scores)
        elif method.lower() == "dic":
            scores = [self._compute_dic(result) for result in model_results]
            best_idx = np.argmin(scores)
        else:
            raise ValueError(f"Unknown comparison method: {method}")

        # Compute relative likelihoods
        min_score = min(scores)
        relative_likelihoods = [np.exp(-(score - min_score) / 2) for score in scores]

        # Akaike weights
        total_likelihood = sum(relative_likelihoods)
        akaike_weights = [lik / total_likelihood for lik in relative_likelihoods]

        return {
            "method": method.upper(),
            "scores": scores,
            "best_model_index": best_idx,
            "relative_likelihoods": relative_likelihoods,
            "akaike_weights": akaike_weights,
            "n_models": len(model_results),
        }

    def _compute_aic(self, result: SPMResult) -> float:
        """Compute Akaike Information Criterion."""
        n_params = (
            len(result.beta_coefficients) if hasattr(result, "beta_coefficients") else 1
        )
        ll = result.model_diagnostics.get("log_likelihood", 0)

        return float(2 * n_params - 2 * ll)

    def _compute_bic(self, result: SPMResult) -> float:
        """Compute Bayesian Information Criterion."""
        n_params = (
            len(result.beta_coefficients) if hasattr(result, "beta_coefficients") else 1
        )
        n_points = len(result.residuals)
        ll = result.model_diagnostics.get("log_likelihood", 0)

        return float(np.log(n_points) * n_params - 2 * ll)

    def _compute_dic(self, result: SPMResult) -> float:
        """Compute Deviance Information Criterion."""
        # Deferred: see docs/deferred_statistical_methods.md
        # ("DIC effective-parameters").
        deviance = -2 * result.model_diagnostics.get("log_likelihood", 0)
        n_params = (
            len(result.beta_coefficients) if hasattr(result, "beta_coefficients") else 1
        )

        # Effective number of parameters (see register: "DIC effective-parameters").
        p_d = n_params

        return float(deviance + 2 * p_d)

    def diagnostic_tests(self, model_result: SPMResult) -> dict[str, Any]:
        """
        Perform comprehensive diagnostic tests on fitted model.

        Args:
            model_result: Fitted SPMResult

        Returns:
            Dictionary with diagnostic test results
        """
        y = self._extract_response(model_result.spm_data)
        y_hat = y - model_result.residuals
        residuals = model_result.residuals

        diagnostics: dict[str, Any] = {}

        # Normality tests
        diagnostics["shapiro_wilk"] = self._shapiro_wilk_test(residuals)
        diagnostics["jarque_bera"] = self._jarque_bera_test(residuals)

        # Homoscedasticity tests
        diagnostics["breusch_pagan"] = self._breusch_pagan_test(residuals, y_hat)

        # Autocorrelation tests
        diagnostics["durbin_watson"] = self._durbin_watson_test(residuals)

        # Goodness of fit
        diagnostics["r_squared"] = model_result.model_diagnostics.get("r_squared", 0)
        diagnostics["adjusted_r_squared"] = self._adjusted_r_squared(model_result)

        return diagnostics

    def _shapiro_wilk_test(self, residuals: np.ndarray) -> dict[str, float]:
        """Test for normality using Shapiro-Wilk test."""
        try:
            stat, p_value = stats.shapiro(residuals)
            return {"statistic": stat, "p_value": p_value, "normal": p_value > 0.05}
        except Exception:
            return {"statistic": np.nan, "p_value": np.nan, "normal": False}

    def _jarque_bera_test(self, residuals: np.ndarray) -> dict[str, float]:
        """Test for normality using Jarque-Bera test."""
        try:
            stat, p_value = stats.jarque_bera(residuals)
            return {"statistic": stat, "p_value": p_value, "normal": p_value > 0.05}
        except Exception:
            return {"statistic": np.nan, "p_value": np.nan, "normal": False}

    def _breusch_pagan_test(
        self, residuals: np.ndarray, fitted: np.ndarray
    ) -> dict[str, float]:
        """Test for heteroscedasticity using Breusch-Pagan test."""
        try:
            # Deferred: see docs/deferred_statistical_methods.md
            # ("Studentized Breusch-Pagan test").
            residuals_sq = residuals**2
            X = np.column_stack([np.ones(len(fitted)), fitted])

            beta = np.linalg.pinv(X.T @ X) @ (X.T @ residuals_sq)
            fitted_sq = X @ beta

            ss_res = np.sum((residuals_sq - fitted_sq) ** 2)
            ss_tot = np.sum((residuals_sq - np.mean(residuals_sq)) ** 2)

            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            n, p = X.shape
            f_stat = (r_squared / p) / ((1 - r_squared) / (n - p - 1))
            p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)

            return {
                "statistic": f_stat,
                "p_value": p_value,
                "homoscedastic": p_value > 0.05,
            }
        except Exception:
            return {"statistic": np.nan, "p_value": np.nan, "homoscedastic": True}

    def _durbin_watson_test(self, residuals: np.ndarray) -> dict[str, Any]:
        """Compute Durbin-Watson statistic for autocorrelation."""
        if len(residuals) < 2:
            return {"statistic": np.nan, "autocorrelation": "unknown"}

        diff = np.diff(residuals)
        dw_stat = np.sum(diff**2) / np.sum(residuals**2)

        # Classify autocorrelation
        if dw_stat < 1.5:
            autocorr = "positive"
        elif dw_stat > 2.5:
            autocorr = "negative"
        else:
            autocorr = "none"

        return {"statistic": dw_stat, "autocorrelation": autocorr}

    def _adjusted_r_squared(self, model_result: SPMResult) -> float:
        """Compute adjusted R-squared."""
        r2 = model_result.model_diagnostics.get("r_squared", 0)
        n = len(model_result.residuals)
        p = (
            len(model_result.beta_coefficients)
            if hasattr(model_result, "beta_coefficients")
            else 1
        )

        if n > p + 1:
            return float(1 - (1 - r2) * (n - 1) / (n - p - 1))
        else:
            return float(r2)


def validate_spm_model(
    model_result: SPMResult,
    validation_data: SPMData | None = None,
    method: str = "diagnostics",
) -> dict[str, Any]:
    """
    Convenience function for SPM model validation.

    Args:
        model_result: Fitted SPMResult to validate
        validation_data: Optional separate validation dataset
        method: Validation method ('diagnostics', 'cross_validate')

    Returns:
        Dictionary with validation results. For ``'cross_validate'``, the
        result of spatial block cross-validation: coordinates are clustered
        into contiguous K-means blocks, the model is refit on each training
        block set with the module GLM fitter, and each held-out block is
        scored.

    Example:
        >>> diagnostics = validate_spm_model(model_result, method='diagnostics')
        >>> cv_results = validate_spm_model(model_result, validation_data=test_data, method='cross_validate')
    """
    validator = ModelValidator()

    if method == "diagnostics":
        return validator.diagnostic_tests(model_result)
    elif method == "cross_validate":
        if validation_data is None:
            raise ValueError("validation_data required for cross-validation")

        # Spatial block CV: refit the model on each training block using the
        # module's own GLM fitter and the existing fold/predict primitives.
        from ..glm import fit_glm

        validator = ModelValidator(validation_method="spatial")
        return validator.cross_validate(
            fit_glm, validation_data, model_result.design_matrix
        )
    else:
        raise ValueError(f"Unknown validation method: {method}")
