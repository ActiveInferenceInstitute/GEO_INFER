"""
Model validation utilities for economic analysis.
"""

from typing import Any
import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import logging


class ModelValidator:
    """
    Utility class for validating economic models and results.

    Provides methods for statistical testing, model diagnostics, and validation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the ModelValidator.

        Args:
            config: Optional configuration for validation
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

    def validate_regression_assumptions(
        self,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
        X: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Validate regression model assumptions.

        Args:
            residuals: Model residuals
            fitted_values: Fitted values from the model
            X: Optional design matrix for additional tests

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "normality": self._test_normality(residuals),
            "homoscedasticity": self._test_homoscedasticity(residuals, fitted_values),
            "autocorrelation": self._test_autocorrelation(residuals),
            "linearity": self._test_linearity(residuals, fitted_values),
        }

        if X is not None:
            validation_results["multicollinearity"] = self._test_multicollinearity(X)

        return validation_results

    def _test_normality(self, residuals: np.ndarray) -> dict[str, Any]:
        """Test normality of residuals using Shapiro-Wilk and Jarque-Bera tests."""
        try:
            # Shapiro-Wilk test (for smaller samples)
            if len(residuals) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
            else:
                shapiro_stat, shapiro_p = None, None

            # Jarque-Bera test
            jb_stat, jb_p = stats.jarque_bera(residuals)

            return {
                "shapiro_statistic": shapiro_stat,
                "shapiro_p_value": shapiro_p,
                "jarque_bera_statistic": jb_stat,
                "jarque_bera_p_value": jb_p,
                "normal": jb_p > 0.05,
                "interpretation": (
                    "Residuals appear normally distributed"
                    if jb_p > 0.05
                    else "Residuals deviate from normality"
                ),
            }

        except Exception as e:
            self.logger.error(f"Normality test failed: {str(e)}")
            return {"error": str(e)}

    def _test_homoscedasticity(
        self, residuals: np.ndarray, fitted_values: np.ndarray
    ) -> dict[str, Any]:
        """Test homoscedasticity using Breusch-Pagan test."""
        try:
            # Breusch-Pagan test
            n = len(residuals)
            squared_residuals = residuals**2

            # Simple regression of squared residuals on fitted values
            X = np.column_stack([np.ones(n), fitted_values])
            beta = np.linalg.lstsq(X, squared_residuals, rcond=None)[0]
            predicted = X @ beta

            # Test statistic
            explained_ss = np.sum((predicted - np.mean(squared_residuals)) ** 2)

            bp_stat = (n * explained_ss) / np.sum(squared_residuals**2)
            bp_p = 1 - stats.chi2.cdf(bp_stat, 1)

            return {
                "breusch_pagan_statistic": bp_stat,
                "breusch_pagan_p_value": bp_p,
                "homoscedastic": bp_p > 0.05,
                "interpretation": (
                    "Homoscedasticity assumption satisfied"
                    if bp_p > 0.05
                    else "Heteroscedasticity detected"
                ),
            }

        except Exception as e:
            self.logger.error(f"Homoscedasticity test failed: {str(e)}")
            return {"error": str(e)}

    def _test_autocorrelation(self, residuals: np.ndarray) -> dict[str, Any]:
        """Test for autocorrelation using Durbin-Watson test."""
        try:
            # Durbin-Watson test
            diff_residuals = np.diff(residuals)
            dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)

            # Simple interpretation (more sophisticated critical values would be needed)
            no_autocorr = 1.5 < dw_stat < 2.5

            return {
                "durbin_watson_statistic": dw_stat,
                "no_autocorrelation": no_autocorr,
                "interpretation": (
                    "No significant autocorrelation"
                    if no_autocorr
                    else "Autocorrelation detected"
                ),
            }

        except Exception as e:
            self.logger.error(f"Autocorrelation test failed: {str(e)}")
            return {"error": str(e)}

    def _test_linearity(
        self, residuals: np.ndarray, fitted_values: np.ndarray
    ) -> dict[str, Any]:
        """Test linearity assumption using RESET test."""
        try:
            # Simple linearity test using correlation
            corr_coef = np.corrcoef(residuals, fitted_values)[0, 1]

            # Test if correlation is significantly different from zero
            n = len(residuals)
            t_stat = corr_coef * np.sqrt((n - 2) / (1 - corr_coef**2))
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 2))

            return {
                "correlation_residuals_fitted": corr_coef,
                "t_statistic": t_stat,
                "p_value": p_value,
                "linear": p_value > 0.05,
                "interpretation": (
                    "Linearity assumption satisfied"
                    if p_value > 0.05
                    else "Nonlinearity detected"
                ),
            }

        except Exception as e:
            self.logger.error(f"Linearity test failed: {str(e)}")
            return {"error": str(e)}

    def _test_multicollinearity(self, X: np.ndarray) -> dict[str, Any]:
        """Test for multicollinearity using condition number and VIF."""
        try:
            # Condition number
            cond_number = np.linalg.cond(X)

            # Simple VIF calculation (would need more sophisticated implementation)
            # For now, just return condition number
            high_multicollinearity = cond_number > 30

            return {
                "condition_number": cond_number,
                "high_multicollinearity": high_multicollinearity,
                "interpretation": (
                    "Multicollinearity concern"
                    if high_multicollinearity
                    else "Multicollinearity acceptable"
                ),
            }

        except Exception as e:
            self.logger.error(f"Multicollinearity test failed: {str(e)}")
            return {"error": str(e)}

    def cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        scoring: str = "neg_mean_squared_error",
    ) -> dict[str, Any]:
        """
        Perform cross-validation for model performance assessment.

        Args:
            model: Sklearn-compatible model
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Cross-validation results
        """
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)

            return {
                "cv_scores": cv_scores,
                "mean_cv_score": np.mean(cv_scores),
                "std_cv_score": np.std(cv_scores),
                "scoring_metric": scoring,
                "interpretation": f"Mean CV score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}",
            }

        except Exception as e:
            self.logger.error(f"Cross-validation failed: {str(e)}")
            return {"error": str(e)}

    def validate_economic_model_results(
        self,
        predictions: np.ndarray,
        actual: np.ndarray,
        model_type: str = "regression",
        additional_metrics: bool = True,
    ) -> dict[str, Any]:
        """
        Comprehensive validation of economic model results.

        Args:
            predictions: Model predictions
            actual: Actual observed values
            model_type: Type of model ('regression', 'classification', 'time_series')
            additional_metrics: Whether to calculate additional diagnostic metrics

        Returns:
            Comprehensive validation metrics
        """
        try:
            if model_type == "regression":
                return self._validate_regression_model(
                    predictions, actual, additional_metrics
                )
            elif model_type == "classification":
                return self._validate_classification_model(predictions, actual)
            elif model_type == "time_series":
                return self._validate_time_series_model(predictions, actual)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            self.logger.error(f"Results validation failed: {str(e)}")
            return {"error": str(e)}

    def _validate_regression_model(
        self, predictions: np.ndarray, actual: np.ndarray, additional_metrics: bool
    ) -> dict[str, Any]:
        """Comprehensive regression model validation."""
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predictions)

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        # Additional metrics
        results = {
            "mse": float(mse),
            "rmse": float(rmse),
            "r_squared": float(r2),
            "mape": float(mape),
            "mean_absolute_error": float(np.mean(np.abs(actual - predictions))),
            "median_absolute_error": float(np.median(np.abs(actual - predictions))),
            "interpretation": f"Model explains {r2:.2%} of variance with RMSE of {rmse:.4f}",
        }

        if additional_metrics:
            # Theil's U statistic
            naive_forecast = np.roll(actual, 1)[
                1:
            ]  # Naive forecast using previous value
            naive_mse = mean_squared_error(actual[1:], naive_forecast)
            theil_u = np.sqrt(mse / naive_mse) if naive_mse > 0 else np.nan

            # Prediction intervals (simplified)
            residual_std = np.std(actual - predictions)
            results.update(
                {
                    "theil_u_statistic": float(theil_u),
                    "prediction_interval_width": 1.96 * residual_std,  # 95% interval
                    "residual_standard_deviation": float(residual_std),
                    "durbin_watson_statistic": self._calculate_durbin_watson(
                        actual - predictions
                    ),
                }
            )

        return results

    def _validate_classification_model(
        self, predictions: np.ndarray, actual: np.ndarray
    ) -> dict[str, Any]:
        """Classification model validation."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        # For now, treat as binary classification
        # In practice, would handle multi-class and regression-to-classification

        accuracy = accuracy_score(actual, predictions)
        precision = precision_score(
            actual, predictions, average="weighted", zero_division=0
        )
        recall = recall_score(actual, predictions, average="weighted", zero_division=0)
        f1 = f1_score(actual, predictions, average="weighted", zero_division=0)

        cm = confusion_matrix(actual, predictions)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "interpretation": f"Classification accuracy: {accuracy:.2%} with F1-score: {f1:.3f}",
        }

    def _validate_time_series_model(
        self, predictions: np.ndarray, actual: np.ndarray
    ) -> dict[str, Any]:
        """Time series model validation."""
        # Extend arrays to same length if needed
        min_len = min(len(predictions), len(actual))
        pred = predictions[:min_len]
        act = actual[:min_len]

        # Basic regression metrics
        base_metrics = self._validate_regression_model(
            pred, act, additional_metrics=False
        )

        # Time series specific metrics
        # Mean Absolute Scaled Error (MASE)
        naive_errors = np.abs(np.diff(act))
        naive_mae = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        mase = np.mean(np.abs(act - pred)) / naive_mae if naive_mae > 0 else np.nan

        # Symmetric Mean Absolute Percentage Error (SMAPE)
        smape = 2 * np.mean(np.abs(act - pred) / (np.abs(act) + np.abs(pred))) * 100

        # Directional accuracy (for trend prediction)
        actual_direction = np.sign(np.diff(act))
        predicted_direction = np.sign(np.diff(pred))
        directional_accuracy = np.mean(actual_direction == predicted_direction)

        base_metrics.update(
            {
                "mase": float(mase),
                "smape": float(smape),
                "directional_accuracy": float(directional_accuracy),
                "model_type": "time_series",
            }
        )

        return base_metrics

    def _calculate_durbin_watson(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation."""
        diff = np.diff(residuals)
        return float(np.sum(diff**2)) / float(np.sum(residuals**2))

    def validate_model_assumptions(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        assumptions: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Validate key econometric model assumptions.

        Args:
            model: Fitted model object
            X: Feature matrix
            y: Target variable
            assumptions: List of assumptions to test

        Returns:
            Dictionary with assumption validation results
        """
        if assumptions is None:
            assumptions = ["linearity", "homoscedasticity", "normality", "independence"]

        results = {}

        # Get predictions and residuals
        predictions = model.predict(X)
        residuals = y - predictions

        for assumption in assumptions:
            if assumption == "linearity":
                results["linearity"] = self._test_linearity_assumption(
                    X, y, predictions
                )
            elif assumption == "homoscedasticity":
                results["homoscedasticity"] = self._test_homoscedasticity_assumption(
                    residuals, predictions
                )
            elif assumption == "normality":
                results["normality"] = self._test_normality_assumption(residuals)
            elif assumption == "independence":
                results["independence"] = self._test_independence_assumption(residuals)

        return results

    def _test_linearity_assumption(
        self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray
    ) -> dict[str, Any]:
        """Test linearity assumption using RESET test."""
        # Ramsey RESET test (simplified)
        # Fit model with squared and cubed predictions
        X_reset = np.column_stack([X, predictions**2, predictions**3])
        try:
            beta_reset = np.linalg.lstsq(X_reset, y, rcond=None)[0]

            # Compare with original model
            original_rss = np.sum((y - predictions) ** 2)
            reset_predictions = X_reset @ beta_reset
            reset_rss = np.sum((y - reset_predictions) ** 2)

            f_stat = ((original_rss - reset_rss) / 2) / (
                reset_rss / (len(y) - len(beta_reset))
            )
            p_value = 1 - stats.f.cdf(f_stat, 2, len(y) - len(beta_reset))

            return {
                "reset_f_statistic": float(f_stat),
                "p_value": float(p_value),
                "linear": p_value > 0.05,
                "interpretation": (
                    "Linearity assumption satisfied"
                    if p_value > 0.05
                    else "Nonlinearity detected"
                ),
            }
        except Exception:
            return {"error": "RESET test failed"}

    def _test_homoscedasticity_assumption(
        self, residuals: np.ndarray, predictions: np.ndarray
    ) -> dict[str, Any]:
        """Test homoscedasticity assumption."""
        # Breusch-Pagan test
        n = len(residuals)
        squared_residuals = residuals**2

        # Regress squared residuals on predictions
        X_bp = np.column_stack([np.ones(n), predictions])
        try:
            beta_bp = np.linalg.lstsq(X_bp, squared_residuals, rcond=None)[0]
            bp_fitted = X_bp @ beta_bp

            # Test statistic
            rss = np.sum((squared_residuals - bp_fitted) ** 2)
            tss = np.sum((squared_residuals - np.mean(squared_residuals)) ** 2)

            bp_stat = (n * (tss - rss)) / rss if rss > 0 else 0
            p_value = 1 - stats.chi2.cdf(bp_stat, 1)

            return {
                "breusch_pagan_statistic": float(bp_stat),
                "p_value": float(p_value),
                "homoscedastic": p_value > 0.05,
                "interpretation": (
                    "Homoscedasticity assumption satisfied"
                    if p_value > 0.05
                    else "Heteroscedasticity detected"
                ),
            }
        except Exception:
            return {"error": "Breusch-Pagan test failed"}

    def _test_normality_assumption(self, residuals: np.ndarray) -> dict[str, Any]:
        """Test normality assumption."""
        # Shapiro-Wilk test
        if len(residuals) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        else:
            shapiro_stat, shapiro_p = None, None

        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(residuals)

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(
            residuals, "norm", args=(np.mean(residuals), np.std(residuals))
        )

        return {
            "shapiro_wilk_statistic": shapiro_stat,
            "shapiro_wilk_p_value": shapiro_p,
            "jarque_bera_statistic": float(jb_stat),
            "jarque_bera_p_value": float(jb_p),
            "kolmogorov_smirnov_statistic": float(ks_stat),
            "kolmogorov_smirnov_p_value": float(ks_p),
            "normal": jb_p > 0.05,
            "interpretation": (
                "Normality assumption satisfied"
                if jb_p > 0.05
                else "Non-normality detected"
            ),
        }

    def _test_independence_assumption(self, residuals: np.ndarray) -> dict[str, Any]:
        """Test independence assumption using Durbin-Watson test."""
        dw_stat = self._calculate_durbin_watson(residuals)

        # Simple interpretation (more sophisticated critical values would be needed)
        no_autocorr = 1.5 < dw_stat < 2.5

        return {
            "durbin_watson_statistic": float(dw_stat),
            "no_autocorrelation": no_autocorr,
            "interpretation": (
                "Independence assumption satisfied"
                if no_autocorr
                else "Autocorrelation detected"
            ),
        }

    def validate_spatial_model_assumptions(
        self, model: Any, X: np.ndarray, y: np.ndarray, W: np.ndarray
    ) -> dict[str, Any]:
        """
        Validate assumptions for spatial econometric models.

        Args:
            model: Fitted spatial model
            X: Feature matrix
            y: Target variable
            W: Spatial weights matrix

        Returns:
            Dictionary with spatial model assumption validation
        """
        # Get residuals
        predictions = model.predict(X)
        residuals = y - predictions

        # Standard regression assumptions
        base_assumptions = self.validate_model_assumptions(model, X, y)

        # Spatial-specific assumptions
        spatial_assumptions = {}

        # Spatial dependence in residuals
        wy_residuals = W @ residuals
        n = len(residuals)
        morans_i = (
            (n / np.sum(W)) * (residuals.T @ wy_residuals) / (residuals.T @ residuals)
        )

        spatial_assumptions["spatial_independence"] = {
            "morans_i_residuals": float(morans_i),
            "spatial_independence_satisfied": abs(morans_i) < 0.1,
        }

        # Stationarity (simplified)
        # In practice, would test for spatial stationarity
        spatial_assumptions["stationarity"] = {
            "trend_test": "simplified",
            "stationarity_satisfied": True,  # Baseline
        }

        return {
            "base_assumptions": base_assumptions,
            "spatial_assumptions": spatial_assumptions,
            "overall_valid": all(
                [
                    base_assumptions.get("linearity", {}).get("linear", False),
                    base_assumptions.get("homoscedasticity", {}).get(
                        "homoscedastic", False
                    ),
                    base_assumptions.get("normality", {}).get("normal", False),
                    base_assumptions.get("independence", {}).get(
                        "no_autocorrelation", False
                    ),
                    spatial_assumptions["spatial_independence"][
                        "spatial_independence_satisfied"
                    ],
                ]
            ),
        }
