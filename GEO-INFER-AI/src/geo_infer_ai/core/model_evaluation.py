"""
Model evaluation module for geospatial AI models.

Provides geospatial-specific evaluation metrics and validation methods.
"""

import logging
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


class GeospatialModelEvaluator:
    """
    Evaluate geospatial AI models with spatial-specific metrics.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize model evaluator."""
        self.config = config or {}

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[Any] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate classification model.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional label names

        Returns:
            Evaluation metrics
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
        }

        return metrics

    def evaluate_regression(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Evaluate regression model.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Evaluation metrics
        """
        metrics = {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }

        return metrics

    def evaluate_value_tolerance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tolerance: float = 100.0,
    ) -> dict[str, float]:
        """
        Evaluate absolute prediction errors in target-value units.

        Args:
            y_true: True values
            y_pred: Predicted values
            tolerance: Finite nonnegative tolerance in target-value units.

        Returns:
            Error statistics in target-value units plus the
            ``within_tolerance_percentage`` tolerance score
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.ndim != 1 or y_true.size == 0 or y_true.shape != y_pred.shape:
            raise ValueError("Expected nonempty, aligned one-dimensional values")
        if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
            raise ValueError("Prediction values must be finite")
        if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        errors = np.abs(y_true - y_pred)

        metrics = {
            "mean_absolute_error": float(np.mean(errors)),
            "median_absolute_error": float(np.median(errors)),
            "max_absolute_error": float(np.max(errors)),
            "absolute_error_std": float(np.std(errors)),
        }

        metrics["within_tolerance_percentage"] = float(
            np.mean(errors <= tolerance) * 100
        )

        return metrics

    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[Any] | None = None,
        normalize: str | None = None,
    ) -> dict[str, Any]:
        """
        Compute confusion matrix and derived statistics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Optional ordered label list
            normalize: Normalization mode ('true', 'pred', 'all', or None)

        Returns:
            Dictionary with raw matrix, per-class precision/recall, and summary
        """
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix

        cm = sk_confusion_matrix(y_true, y_pred, labels=labels)

        if labels is None:
            labels = sorted(list(set(np.concatenate([y_true, y_pred]))))

        # Normalized version if requested
        cm_normalized = None
        if normalize == "true":
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_normalized = np.divide(
                cm.astype(float),
                row_sums,
                out=np.zeros_like(cm, dtype=float),
                where=row_sums != 0,
            )
        elif normalize == "pred":
            col_sums = cm.sum(axis=0, keepdims=True)
            cm_normalized = np.divide(
                cm.astype(float),
                col_sums,
                out=np.zeros_like(cm, dtype=float),
                where=col_sums != 0,
            )
        elif normalize == "all":
            total = cm.sum()
            cm_normalized = cm.astype(float) / total if total > 0 else cm.astype(float)

        # Per-class statistics
        per_class = {}
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_val = (
                2 * precision_val * recall_val / (precision_val + recall_val)
                if (precision_val + recall_val) > 0
                else 0.0
            )
            per_class[str(label)] = {
                "true_positive": int(tp),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_negative": int(tn),
                "precision": float(precision_val),
                "recall": float(recall_val),
                "f1_score": float(f1_val),
            }

        result: dict[str, Any] = {
            "confusion_matrix": cm.tolist(),
            "labels": [str(label) for label in labels],
            "per_class": per_class,
        }
        if cm_normalized is not None:
            result["confusion_matrix_normalized"] = cm_normalized.tolist()

        return result

    def compute_roc_auc(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        multi_class: str = "ovr",
    ) -> dict[str, Any]:
        """
        Compute ROC-AUC score.

        For binary classification y_score should be probabilities for the positive
        class. For multi-class, y_score should be shape (n_samples, n_classes).

        Args:
            y_true: True labels
            y_score: Predicted probabilities or decision values
            multi_class: Strategy for multi-class ('ovr' or 'ovo')

        Returns:
            Dictionary with AUC score and per-class AUC when applicable
        """
        from sklearn.metrics import roc_auc_score

        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)

        result: dict[str, Any] = {"n_classes": n_classes}

        if n_classes == 2:
            # Binary case
            scores = y_score
            if scores.ndim == 2:
                scores = scores[:, 1]
            auc = float(roc_auc_score(y_true, scores))
            result["roc_auc"] = auc
        elif n_classes > 2:
            # Multi-class case
            if y_score.ndim == 1:
                raise ValueError(
                    "For multi-class ROC-AUC, y_score must be 2D (n_samples, n_classes)"
                )
            auc = float(
                roc_auc_score(
                    y_true, y_score, multi_class=multi_class, average="weighted"
                )
            )
            result["roc_auc"] = auc
            result["multi_class_strategy"] = multi_class

            # Per-class AUC (one-vs-rest)
            from sklearn.preprocessing import label_binarize

            y_bin = label_binarize(y_true, classes=unique_classes)
            per_class_auc = {}
            for i, cls in enumerate(unique_classes):
                try:
                    cls_auc = float(roc_auc_score(y_bin[:, i], y_score[:, i]))
                except ValueError:
                    cls_auc = float("nan")
                per_class_auc[str(cls)] = cls_auc
            result["per_class_auc"] = per_class_auc
        else:
            result["roc_auc"] = float("nan")
            result["error"] = "Only one class present in y_true"

        return result

    def cross_validate_spatial(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        coordinates: np.ndarray,
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """
        Perform spatial block cross-validation.

        The sample locations are partitioned into ``n_splits`` contiguous
        spatial blocks with K-means on ``coordinates``. Each fold trains on
        all blocks except one and evaluates on the held-out block, so that
        spatially autocorrelated neighbors never appear on both sides of a
        split. This is the standard defense against spatial leakage that
        shuffled K-fold cannot provide.

        Args:
            model: Model to evaluate (scikit-learn compatible)
            X: Feature matrix
            y: Target values
            coordinates: Spatial coordinates (n_samples, 2)
            n_splits: Number of spatial blocks (and folds)

        Returns:
            Dictionary with per-fold scores and their mean/std

        Raises:
            ValueError: If ``n_splits`` is below 2 or exceeds the sample count
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if n_splits > len(y):
            raise ValueError("n_splits cannot exceed the number of samples")

        coordinates = np.asarray(coordinates, dtype=float)
        blocks = KMeans(n_clusters=n_splits, n_init=10, random_state=42).fit_predict(
            coordinates
        )

        scores: list[float] = []
        for block in np.unique(blocks):
            test_idx = np.where(blocks == block)[0]
            train_idx = np.where(blocks != block)[0]

            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])

            if len(np.unique(y)) > 10:  # Regression
                score = r2_score(y[test_idx], y_pred)
            else:  # Classification
                score = accuracy_score(y[test_idx], y_pred)

            scores.append(float(score))

        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": scores,
        }
