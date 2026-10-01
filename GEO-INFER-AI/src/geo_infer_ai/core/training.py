"""
Core training and evaluation functionality for geospatial AI models.

This module provides training loops, evaluation metrics, and model management
for geospatial machine learning workflows. Includes data splitting,
cross-validation, and hyperparameter search capabilities.
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geo_infer_ai.core.secure_serialization import (
    CONTEXT_MODEL_SAVE,
    sign_payload,
    verify_payload,
)

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


def _instantiate_with_random_state(
    model_class: Any, random_state: int, params: dict[str, Any]
) -> Any:
    """Instantiate ``model_class`` with ``random_state`` when supported.

    Not every scikit-learn estimator accepts ``random_state`` (e.g.
    KNeighborsClassifier, SVR, LinearRegression). Try the full constructor
    first; on TypeError, retry without ``random_state`` so estimators that
    genuinely reject a caller-supplied parameter still surface a clear
    error rather than a silent misconfiguration.
    """
    import inspect

    try:
        return model_class(random_state=random_state, **params)
    except TypeError:
        signature = inspect.signature(model_class)
        if "random_state" in signature.parameters:
            # The ctor does accept random_state; the TypeError came from
            # params themselves — do not mask the caller's mistake.
            raise
        return model_class(**params)


@dataclass
class TrainingConfig:
    """
    Configuration for model training.

    Attributes:
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        validation_split: Fraction of data to use for validation
        early_stopping_patience: Patience for early stopping
        save_best_model: Whether to save the best model during training
        model_save_path: Path to save trained models
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
    """

    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    save_best_model: bool = True
    model_save_path: str | Path | None = None
    verbose: int = 1

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")


class ModelTrainer:
    """
    Trainer for geospatial AI models with comprehensive evaluation.

    This class provides training loops, evaluation metrics, and model management
    for both classification and regression tasks in geospatial contexts.
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        """
        Initialize the model trainer.

        Args:
            config: Training configuration. If None, uses default configuration.
        """
        self.config = config or TrainingConfig()
        self.training_history: list[dict[str, float]] = []
        self.best_model: Any | None = None
        self.best_score: float = float("-inf")

    def _reset_best(self) -> None:
        """Reset best-model tracking at the start of each train_* call.

        Accuracy (classification) and R² (regression) live on
        incomparable scales, so best_score is only meaningful within a
        single training call's task type.
        """
        self.best_model = None
        self.best_score = float("-inf")

    def train_classifier(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Train a classification model.

        Args:
            model: Scikit-learn compatible classifier
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary containing training history and evaluation metrics
        """
        logger.info(f"Training classifier with {len(X_train)} samples")

        self._reset_best()

        # Split validation data if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.config.validation_split,
                random_state=42,
                stratify=y_train,
            )

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)

        # Store best model
        if accuracy > self.best_score:
            self.best_score = accuracy
            self.best_model = model

        # Save model if configured
        if self.config.save_best_model and self.config.model_save_path:
            self._save_model(model, self.config.model_save_path)

        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "model": model,
        }

        logger.info(f"Training completed. Validation accuracy: {accuracy:.4f}")
        return results

    def train_regressor(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Train a regression model.

        Args:
            model: Scikit-learn compatible regressor
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Dictionary containing training history and evaluation metrics
        """
        logger.info(f"Training regressor with {len(X_train)} samples")

        self._reset_best()
        # Split validation data if not provided
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.config.validation_split,
                random_state=42,
            )

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate on validation set
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mse)

        # Store best model (using R² score)
        if r2 > self.best_score:
            self.best_score = r2
            self.best_model = model

        # Save model if configured
        if self.config.save_best_model and self.config.model_save_path:
            self._save_model(model, self.config.model_save_path)

        results = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "model": model,
        }

        logger.info(f"Training completed. Validation R²: {r2:.4f}, RMSE: {rmse:.4f}")
        return results

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        task_type: str = "classification",
    ) -> dict[str, Any]:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels/targets
            task_type: Type of task ("classification" or "regression")

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {task_type} model on {len(X_test)} test samples")

        y_pred = model.predict(X_test)

        if task_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Handle multi-class vs binary
            if len(np.unique(y_test)) == 2:
                average = "binary"
            else:
                average = "weighted"

            precision = precision_score(
                y_test, y_pred, average=average, zero_division=0
            )
            recall = recall_score(y_test, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            return {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": cm.tolist(),
                "classification_report": report,
                "predictions": (
                    y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred
                ),
            }
        elif task_type == "regression":
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            # sklearn cannot define R² for a single test observation and emits
            # an UndefinedMetricWarning. Keep the evaluation contract finite
            # and explicit for small deterministic integration fixtures.
            if len(y_test) < 2:
                r2 = 1.0 if np.allclose(y_test, y_pred) else 0.0
            else:
                r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)

            # Mean Absolute Percentage Error: standard definition divides by
            # |y_true|. Observations where y_true == 0 are excluded from the
            # mean (the percentage error is undefined there); an all-zero
            # target yields mape == 0.0.
            nonzero = y_test != 0
            if np.any(nonzero):
                mape = (
                    np.mean(
                        np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])
                    )
                    * 100
                )
            else:
                mape = 0.0
            median_ae = np.median(np.abs(y_test - y_pred))  # Median Absolute Error

            # Calculate residuals
            residuals = y_test - y_pred
            residual_std = np.std(residuals)

            return {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
                "mape": float(mape),
                "median_ae": float(median_ae),
                "residual_std": float(residual_std),
                "predictions": (
                    y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred
                ),
            }
        else:
            raise ValueError(
                f"Unknown task_type: {task_type}. Must be 'classification' or 'regression'"
            )

    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        task_type: str = "classification",
        stratified: bool = True,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Perform k-fold cross-validation on a model.

        Args:
            model: Scikit-learn compatible estimator (will be cloned per fold)
            X: Feature matrix
            y: Target values
            n_splits: Number of cross-validation folds
            task_type: 'classification' or 'regression'
            stratified: Whether to use stratified splits (classification only)
            random_state: Random state for reproducibility

        Returns:
            Dictionary containing per-fold and aggregate metrics
        """
        from sklearn.base import clone

        logger.info(
            f"Running {n_splits}-fold cross-validation for {task_type} "
            f"on {len(X)} samples"
        )

        if task_type == "classification" and stratified:
            kf = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            split_iterator = kf.split(X, y)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            split_iterator = kf.split(X)

        fold_results: list[dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(split_iterator):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            fold_model = clone(model)
            fold_model.fit(X_train_fold, y_train_fold)
            y_pred = fold_model.predict(X_val_fold)

            if task_type == "classification":
                fold_metrics = {
                    "fold": fold_idx,
                    "accuracy": float(accuracy_score(y_val_fold, y_pred)),
                }
            else:
                mse = float(mean_squared_error(y_val_fold, y_pred))
                fold_metrics = {
                    "fold": fold_idx,
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "mae": float(mean_absolute_error(y_val_fold, y_pred)),
                    "r2": float(r2_score(y_val_fold, y_pred)),
                }

            fold_results.append(fold_metrics)

        # Aggregate metrics
        metric_keys = [k for k in fold_results[0].keys() if k != "fold"]
        aggregate = {}
        for key in metric_keys:
            values = [f[key] for f in fold_results]
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values))

        result = {
            "n_splits": n_splits,
            "task_type": task_type,
            "fold_results": fold_results,
            "aggregate": aggregate,
        }

        primary_metric = "accuracy_mean" if task_type == "classification" else "r2_mean"
        logger.info(
            f"Cross-validation complete. {primary_metric}: "
            f"{aggregate.get(primary_metric, 0.0):.4f}"
        )
        return result

    def hyperparameter_search(
        self,
        model_class: Any,
        param_grid: dict[str, list[Any]],
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "classification",
        n_splits: int = 3,
        scoring: str | None = None,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Grid search over hyperparameters using cross-validation.

        Args:
            model_class: Scikit-learn model class (not instance)
            param_grid: Dictionary mapping parameter names to lists of values
            X: Feature matrix
            y: Target values
            task_type: 'classification' or 'regression'
            n_splits: Number of CV folds per candidate
            scoring: Scoring metric name (default: accuracy/r2)
            random_state: Random state for reproducibility

        Returns:
            Dictionary containing best parameters, best score, and all results
        """
        import itertools

        supported_metrics = {"accuracy", "r2"}
        if scoring is None:
            scoring = "accuracy" if task_type == "classification" else "r2"
        elif scoring not in supported_metrics:
            raise ValueError(
                f"geo_infer_ai.core.training.hyperparameter_search: unsupported "
                f"scoring {scoring!r}. Supported metrics: "
                f"{sorted(supported_metrics)}"
            )

        logger.info(f"Starting hyperparameter search over {len(param_grid)} parameters")

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        logger.info(f"Evaluating {len(all_combinations)} parameter combinations")

        search_results: list[dict[str, Any]] = []
        best_score = float("-inf")
        best_params: dict[str, Any] = {}

        for combo in all_combinations:
            params = dict(zip(param_names, combo))
            model = _instantiate_with_random_state(model_class, random_state, params)

            cv_result = self.cross_validate(
                model=model,
                X=X,
                y=y,
                n_splits=n_splits,
                task_type=task_type,
                random_state=random_state,
            )

            aggregate = cv_result["aggregate"]
            score_key = f"{scoring}_mean"
            if score_key not in aggregate:
                raise ValueError(
                    f"geo_infer_ai.core.training.hyperparameter_search: cross-"
                    f"validation aggregate missing {score_key!r} for scoring "
                    f"{scoring!r}; aggregate keys: {sorted(aggregate)}"
                )
            score = aggregate[score_key]

            search_results.append(
                {
                    "params": params,
                    "score": score,
                    "score_std": cv_result["aggregate"].get(f"{scoring}_std", 0.0),
                }
            )

            if score > best_score:
                best_score = score
                best_params = params

        # Sort results by score descending
        search_results.sort(key=lambda x: x["score"], reverse=True)

        result = {
            "best_params": best_params,
            "best_score": best_score,
            "scoring": scoring,
            "all_results": search_results,
            "n_combinations": len(all_combinations),
        }

        logger.info(f"Hyperparameter search complete. Best {scoring}: {best_score:.4f}")
        return result

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = False,
        random_state: int = 42,
    ) -> dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Feature matrix
            y: Target values
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set (from remaining)
            stratify: Whether to stratify splits
            random_state: Random state for reproducibility

        Returns:
            Dictionary with 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'
        """
        stratify_y = y if stratify else None

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )

        # Calculate validation fraction from remaining data
        val_fraction = val_size / (1.0 - test_size) if val_size > 0 else 0.0

        if val_fraction > 0:
            stratify_trainval = y_trainval if stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval,
                y_trainval,
                test_size=val_fraction,
                random_state=random_state,
                stratify=stratify_trainval,
            )
        else:
            X_train, X_val = X_trainval, np.array([])
            y_train, y_val = y_trainval, np.array([])

        logger.info(
            f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

    def _save_model(self, model: Any, path: str | Path) -> None:
        """
        Save a trained model to disk using joblib (preferred) or pickle.

        The serialized bytes are wrapped in an authenticated GISP1 envelope;
        ``load_model`` verifies the envelope before deserializing.

        Args:
            model: Model to save
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize to bytes (joblib preferred for sklearn models)
        try:
            import joblib

            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            payload = buffer.getvalue()
            logger.info(f"Model saved to {path} using joblib")
        except ImportError:
            # Fallback to pickle
            import pickle

            payload = pickle.dumps(model)
            logger.info(f"Model saved to {path} using pickle (joblib not available)")

        envelope = sign_payload(payload, context=CONTEXT_MODEL_SAVE)
        with open(path, "wb") as f:
            f.write(envelope)

    def load_model(self, path: str | Path) -> Any:
        """
        Load a trained model from disk (supports both joblib and pickle formats).

        Args:
            path: Path to the saved model

        Returns:
            Loaded model

        Raises:
            geo_infer_ai.core.secure_serialization.PayloadSecurityError:
                If the file is not a valid GISP1 envelope (unsigned,
                tampered, or signed under a different key/context).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Trust boundary: verify the GISP1 envelope before any deserializer
        # sees the bytes; the joblib/pickle fallback below operates only on
        # already-verified bytes.
        verified = verify_payload(path.read_bytes(), context=CONTEXT_MODEL_SAVE)

        # Try joblib first
        try:
            import joblib

            model = joblib.load(io.BytesIO(verified))
            logger.info(f"Model loaded from {path} using joblib")
            return model
        except (ImportError, ValueError):
            # Fallback to pickle
            import pickle

            model = pickle.loads(verified)
            logger.info(f"Model loaded from {path} using pickle")
            return model
