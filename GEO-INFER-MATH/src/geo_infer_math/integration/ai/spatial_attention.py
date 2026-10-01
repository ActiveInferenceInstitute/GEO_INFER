"""Mathematical foundations for spatial attention mechanisms.

Provides numerically stable attention computations including
scaled dot-product attention, multi-head attention, and
distance-weighted spatial attention.
"""

import numpy as np
from typing import Any, cast
import logging

logger = logging.getLogger(__name__)


class SpatialAttention:
    """Mathematical foundations for spatial attention mechanisms.

    Implements attention mechanisms tailored for geospatial data,
    incorporating spatial relationships into query-key-value
    computations.
    """

    def __init__(self, epsilon: float = 1e-16) -> None:
        """Initialize spatial attention.

        Args:
            epsilon: Numerical stability constant.
        """
        self._epsilon = epsilon
        logger.debug("SpatialAttention initialized")

    def compute_attention_weights(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        coordinates: np.ndarray | None = None,
        distance_weight: float = 0.0,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute scaled dot-product attention with optional spatial weighting.

        attention(Q, K, V) = softmax(Q·K^T / √d_k + λ·S) · V

        where S is a spatial proximity bias matrix.

        Args:
            queries: Query matrix, shape (..., seq_len, d_k).
            keys: Key matrix, shape (..., seq_len, d_k).
            values: Value matrix, shape (..., seq_len, d_v).
            coordinates: Optional spatial coordinates, shape (seq_len, 2).
                Used for distance-weighted attention bias.
            distance_weight: Weight λ for spatial proximity bias (0 = disabled).
            **kwargs: mask, temperature.

        Returns:
            Tuple of (output, attention_weights).
        """
        queries = np.asarray(queries, dtype=np.float64)
        keys = np.asarray(keys, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)

        d_k = queries.shape[-1]
        temperature = kwargs.get("temperature", np.sqrt(d_k))

        # Scaled dot-product scores
        scores = np.dot(queries, keys.T) / temperature

        # Spatial proximity bias
        if coordinates is not None and distance_weight > 0:
            coordinates = np.asarray(coordinates, dtype=np.float64)
            spatial_bias = self._compute_spatial_bias(coordinates)
            scores = scores + distance_weight * spatial_bias

        # Optional masking
        mask = kwargs.get("mask", None)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        # Numerically stable softmax
        weights = self._softmax(scores, axis=-1)

        # Weighted sum of values
        output = np.dot(weights, values)

        logger.debug(
            "Attention: seq_len=%d, d_k=%d, spatial_weight=%.2f",
            queries.shape[-2] if queries.ndim > 1 else 1,
            d_k,
            distance_weight,
        )
        return output, weights

    def multi_head_attention(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        n_heads: int = 4,
        coordinates: np.ndarray | None = None,
        distance_weight: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multi-head attention with optional spatial weighting.

        Splits Q, K, V across heads, computes attention independently,
        and concatenates results.

        Args:
            queries: Shape (seq_len, d_model).
            keys: Shape (seq_len, d_model).
            values: Shape (seq_len, d_model).
            n_heads: Number of attention heads.
            coordinates: Optional spatial coordinates, shape (seq_len, 2).
            distance_weight: Weight for spatial bias.

        Returns:
            Tuple of (output, attention_weights) where output shape is
            (seq_len, d_model) and weights shape is (n_heads, seq_len, seq_len).
        """
        queries = np.asarray(queries, dtype=np.float64)
        keys = np.asarray(keys, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)

        seq_len, d_model = queries.shape
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        d_head = d_model // n_heads

        # Split into heads
        Q_heads = queries.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
        K_heads = keys.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
        V_heads = values.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

        outputs = []
        all_weights = []

        for h in range(n_heads):
            out_h, w_h = self.compute_attention_weights(
                Q_heads[h],
                K_heads[h],
                V_heads[h],
                coordinates=coordinates,
                distance_weight=distance_weight,
            )
            outputs.append(out_h)
            all_weights.append(w_h)

        # Concatenate heads
        output = np.concatenate(outputs, axis=-1)
        weights = np.stack(all_weights)

        logger.debug("Multi-head attention: n_heads=%d, d_head=%d", n_heads, d_head)
        return output, weights

    def _compute_spatial_bias(self, coordinates: np.ndarray) -> np.ndarray:
        """Compute spatial proximity bias matrix.

        Uses inverse distance weighting so nearby points get higher
        attention scores.
        """
        n = coordinates.shape[0]
        diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))

        # Inverse distance (with self-loops set to 0)
        median_dist = np.median(distances[distances > 0]) if n > 1 else 1.0
        bias = np.exp(-distances / (median_dist + self._epsilon))
        np.fill_diagonal(bias, 0.0)

        return cast(np.ndarray, bias)

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax along given axis."""
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return cast(
            np.ndarray,
            exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + self._epsilon),
        )
