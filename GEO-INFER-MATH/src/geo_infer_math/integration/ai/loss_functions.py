"""Spatial loss functions for neural networks.

Provides geographically-aware loss functions that incorporate spatial
relationships into training objectives for machine learning models.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


class SpatialLossFunctions:
    """Spatial loss functions for neural network training.

    Provides loss functions that incorporate spatial proximity,
    geographic weighting, and distance penalties.
    """

    def __init__(self, epsilon: float = 1e-16) -> None:
        """Initialize spatial loss functions.

        Args:
            epsilon: Numerical stability constant.
        """
        self._epsilon = epsilon
        logger.debug("SpatialLossFunctions initialized")

    def calculate_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        coordinates: np.ndarray | None = None,
        loss_type: str = "spatial_mse",
        **kwargs: Any,
    ) -> dict[str, float]:
        """Calculate spatial loss.

        Args:
            predictions: Model predictions, shape (n,).
            targets: Ground truth values, shape (n,).
            coordinates: Spatial coordinates, shape (n, 2). Required
                for geographically-weighted losses.
            loss_type: 'spatial_mse', 'geo_weighted', or 'distance_penalized'.
            **kwargs: Additional parameters (bandwidth, lambda_dist).

        Returns:
            Dictionary with 'loss', 'base_loss', and 'spatial_penalty'.
        """
        predictions = np.asarray(predictions, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)

        if loss_type == "spatial_mse":
            return self.spatial_mse(predictions, targets, coordinates)
        elif loss_type == "geo_weighted":
            bandwidth = kwargs.get("bandwidth", 1.0)
            return self.geographically_weighted_loss(
                predictions, targets, coordinates, bandwidth
            )
        elif loss_type == "distance_penalized":
            lambda_dist = kwargs.get("lambda_dist", 0.1)
            return self.distance_penalized_loss(
                predictions, targets, coordinates, lambda_dist
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def spatial_mse(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        coordinates: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Spatial mean squared error.

        If coordinates are provided, weights errors by inverse distance
        to encourage spatially smooth predictions.

        Args:
            predictions: Predictions, shape (n,).
            targets: Targets, shape (n,).
            coordinates: Optional spatial coordinates, shape (n, 2).

        Returns:
            Dictionary with 'loss', 'base_loss', 'spatial_penalty'.
        """
        residuals = predictions - targets
        base_mse = float(np.mean(residuals**2))

        if coordinates is not None:
            coordinates = np.asarray(coordinates, dtype=np.float64)
            # Spatial smoothness penalty: variance of residuals among neighbours
            n = len(predictions)
            distances = np.sqrt(
                np.sum(
                    (coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :])
                    ** 2,
                    axis=-1,
                )
            )
            # Gaussian spatial weights
            bandwidth = np.median(distances[distances > 0]) if n > 1 else 1.0
            weights = np.exp(-(distances**2) / (2 * bandwidth**2))
            np.fill_diagonal(weights, 0.0)
            weights = weights / (weights.sum(axis=1, keepdims=True) + self._epsilon)

            # Weighted residual difference
            spatial_penalty = 0.0
            for i in range(n):
                neighbour_residuals = weights[i] @ residuals
                spatial_penalty += (residuals[i] - neighbour_residuals) ** 2
            spatial_penalty = float(spatial_penalty / n)
        else:
            spatial_penalty = 0.0

        loss = base_mse + spatial_penalty
        logger.debug(
            "Spatial MSE: loss=%.6f (base=%.6f, penalty=%.6f)",
            loss,
            base_mse,
            spatial_penalty,
        )
        return {"loss": loss, "base_loss": base_mse, "spatial_penalty": spatial_penalty}

    def geographically_weighted_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        coordinates: np.ndarray | None,
        bandwidth: float = 1.0,
    ) -> dict[str, float]:
        """Geographically weighted regression loss.

        Each observation is weighted by a kernel function based on
        its spatial location, allowing local model fitting.

        Args:
            predictions: Predictions, shape (n,).
            targets: Targets, shape (n,).
            coordinates: Spatial coordinates, shape (n, 2).
            bandwidth: Kernel bandwidth parameter.

        Returns:
            Dictionary with 'loss', 'base_loss', 'spatial_penalty'.
        """
        if coordinates is None:
            raise ValueError(
                "Coordinates are required for geographically weighted loss."
            )

        coordinates = np.asarray(coordinates, dtype=np.float64)
        residuals = predictions - targets
        n = len(predictions)

        # Compute geographic weights for each observation (bisquare kernel)
        losses = np.zeros(n)
        for i in range(n):
            dists = np.sqrt(np.sum((coordinates - coordinates[i]) ** 2, axis=1))
            # Bisquare kernel
            u = dists / bandwidth
            weights = np.where(u < 1.0, (1.0 - u**2) ** 2, 0.0)
            weights[i] = 0.0  # Exclude self
            total_weight = weights.sum() + self._epsilon
            weights = weights / total_weight

            losses[i] = float(np.sum(weights * residuals**2))

        geo_loss = float(np.mean(losses))
        base_loss = float(np.mean(residuals**2))

        logger.debug("GW loss: %.6f (base MSE: %.6f)", geo_loss, base_loss)
        return {
            "loss": geo_loss,
            "base_loss": base_loss,
            "spatial_penalty": geo_loss - base_loss,
        }

    def distance_penalized_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        coordinates: np.ndarray | None,
        lambda_dist: float = 0.1,
    ) -> dict[str, float]:
        """MSE with distance-based regularisation penalty.

        L = MSE + λ * penalty, where penalty encourages nearby points
        to have similar predictions.

        Args:
            predictions: Predictions, shape (n,).
            targets: Targets, shape (n,).
            coordinates: Spatial coordinates, shape (n, 2).
            lambda_dist: Regularisation strength.

        Returns:
            Dictionary with 'loss', 'base_loss', 'spatial_penalty'.
        """
        residuals = predictions - targets
        base_mse = float(np.mean(residuals**2))

        if coordinates is not None:
            coordinates = np.asarray(coordinates, dtype=np.float64)
            n = len(predictions)
            penalty = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((coordinates[i] - coordinates[j]) ** 2))
                    weight = np.exp(-dist)
                    penalty += weight * (predictions[i] - predictions[j]) ** 2
            penalty = float(penalty / max(1, n * (n - 1) / 2))
        else:
            penalty = 0.0

        loss = base_mse + lambda_dist * penalty
        logger.debug(
            "Distance-penalized loss: %.6f (base=%.6f, penalty=%.6f)",
            loss,
            base_mse,
            penalty,
        )
        return {
            "loss": loss,
            "base_loss": base_mse,
            "spatial_penalty": lambda_dist * penalty,
        }
