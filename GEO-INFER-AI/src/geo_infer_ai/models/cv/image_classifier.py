"""
Computer vision models for geospatial image classification.

This module provides scikit-learn-based classifiers (Random Forest and
multi-layer perceptron) over feature vectors extracted from satellite and
aerial imagery, such as land cover classification. CNN architectures, object
detection, and semantic segmentation are not implemented; no pre-trained
weights are shipped.
"""

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

logger = logging.getLogger(__name__)


class ImageClassifier(BaseEstimator, ClassifierMixin):
    """
    Image classifier for geospatial imagery using machine learning.

    Supports both traditional ML (Random Forest) and neural network approaches
    for satellite and aerial image classification tasks.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        n_classes: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the image classifier.

        Args:
            model_type: Type of model to use ("random_forest" or "neural_network")
            n_classes: Number of classes for classification
            **kwargs: Additional arguments passed to the underlying model
        """
        self.model_type = model_type
        self.n_classes = n_classes
        self.model: Any | None = None
        self.classes_: np.ndarray | None = None
        self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs: Any) -> None:
        """Initialize the underlying model based on model_type."""
        if self.model_type == "random_forest":
            rf_params: dict[str, Any] = {
                "n_estimators": 100,
                "max_depth": 20,
                "random_state": 42,
                "n_jobs": -1,
            }
            rf_params.update(kwargs)
            self.model = RandomForestClassifier(**rf_params)
        elif self.model_type == "neural_network":
            mlp_params: dict[str, Any] = {
                "hidden_layer_sizes": (100, 50),
                "max_iter": 500,
                "random_state": 42,
                "early_stopping": True,
                "validation_fraction": 0.1,
            }
            mlp_params.update(kwargs)
            self.model = MLPClassifier(**mlp_params)
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                "Must be 'random_forest' or 'neural_network'"
            )

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "ImageClassifier":
        """
        Train the image classifier.

        Args:
            X: Training images as flattened arrays (n_samples, n_features)
            y: Training labels (n_samples,)
            sample_weight: Optional sample weights

        Returns:
            Self for method chaining
        """
        logger.info(
            f"Training {self.model_type} classifier on {len(X)} samples "
            f"with {X.shape[1]} features"
        )

        # Flatten images if needed (handle 2D/3D arrays)
        X_flat = self._flatten_images(X)

        # Determine number of classes if not specified
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))

        # Train the model
        if self.model is None:
            raise ValueError("Model is not initialized")
        if sample_weight is not None:
            self.model.fit(X_flat, y, sample_weight=sample_weight)
        else:
            self.model.fit(X_flat, y)

        # Store class labels
        if hasattr(self.model, "classes_"):
            self.classes_ = np.asarray(self.model.classes_)
        else:
            self.classes_ = np.unique(y)

        logger.info(f"Training completed. Classes: {self.n_classes}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for images.

        Args:
            X: Images to classify (n_samples, n_features) or (n_samples, height, width, channels)

        Returns:
            Predicted class labels (n_samples,)
        """
        if self.model is None or not hasattr(self.model, "predict"):
            raise ValueError("Model must be trained before prediction")

        X_flat = self._flatten_images(X)
        predictions = self.model.predict(X_flat)
        return np.asarray(predictions)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for images.

        Args:
            X: Images to classify (n_samples, n_features) or (n_samples, height, width, channels)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                f"Model type {self.model_type} does not support probability predictions"
            )

        X_flat = self._flatten_images(X)
        probabilities = self.model.predict_proba(X_flat)
        return np.asarray(probabilities)

    def _flatten_images(self, X: np.ndarray) -> np.ndarray:
        """
        Flatten image arrays to 2D format for ML models.

        Args:
            X: Images in various formats

        Returns:
            Flattened 2D array (n_samples, n_features)
        """
        if len(X.shape) == 2:
            # Already flattened
            return X
        elif len(X.shape) == 3:
            # (n_samples, height, width) - grayscale
            n_samples = X.shape[0]
            return X.reshape(n_samples, -1)
        elif len(X.shape) == 4:
            # (n_samples, height, width, channels) - RGB/multispectral
            n_samples = X.shape[0]
            return X.reshape(n_samples, -1)
        else:
            raise ValueError(
                f"Unsupported image shape: {X.shape}. Expected 2D, 3D, or 4D arrays"
            )

    def get_feature_importance(self) -> np.ndarray | None:
        """
        Get feature importance scores (for Random Forest models).

        Returns:
            Feature importance array or None if not available
        """
        if (
            self.model_type == "random_forest"
            and self.model is not None
            and hasattr(self.model, "feature_importances_")
        ):
            return np.asarray(self.model.feature_importances_)
        return None
