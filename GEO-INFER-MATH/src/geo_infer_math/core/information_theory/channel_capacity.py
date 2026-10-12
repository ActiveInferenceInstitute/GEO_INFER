"""
Channel Capacity for Spatial Communication

This module provides channel capacity calculations for spatial
communication systems and data transmission.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


def channel_capacity(
    channel_matrix: np.ndarray,
    noise_power: float | None = None,
    power_constraint: float | None = None,
    base: float = 2.0,
) -> float:
    """
    Calculate channel capacity for a discrete memoryless channel.

    Channel capacity: C = max_p(x) I(X;Y)

    Args:
        channel_matrix: Channel transition matrix p(y|x)
        noise_power: Optional noise power for continuous channels
        power_constraint: Optional power constraint
        base: Logarithm base

    Returns:
        Channel capacity in bits (if base=2) or nats
    """
    channel_matrix = np.asarray(channel_matrix)

    if channel_matrix.ndim != 2:
        raise ValueError("Channel matrix must be 2D")

    n_inputs, n_outputs = channel_matrix.shape

    # Normalize channel matrix (each row should sum to 1)
    row_sums = np.sum(channel_matrix, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    channel_matrix = channel_matrix / row_sums

    # Blahut-Arimoto algorithm: alternate the p(y|x)-induced output
    # marginal with the capacity-achieving input update
    # p_x[x] proportional to exp(D_KL(p(y|x) || p_y)) until the marginal
    # KL between successive input distributions is below tolerance.
    max_iterations = 500
    tolerance = 1e-6
    ln_base = np.log(base)

    p_x = np.ones(n_inputs) / n_inputs

    for _iteration in range(max_iterations):
        # Output marginal p(y) = sum_x p(x) p(y|x)
        p_y = np.maximum(p_x @ channel_matrix, 1e-300)

        # D_x = KL(p(y|x) || p_y) in nats, with 0·log 0 = 0 convention
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(
                channel_matrix > 0,
                np.log(channel_matrix) - np.log(p_y)[np.newaxis, :],
                0.0,
            )
        d_x = np.sum(channel_matrix * log_ratio, axis=1)

        # Capacity-achieving input update: p_new[x] ∝ p_x[x] * exp(d_x)
        shifted = d_x - np.max(d_x)
        p_x_new = p_x * np.exp(shifted)
        total = np.sum(p_x_new)
        p_x_new = p_x_new / total

        # Convergence: KL(p_x || p_x_new) over the input marginal
        with np.errstate(divide="ignore", invalid="ignore"):
            marginal_kl = np.sum(
                np.where(
                    p_x > 0,
                    p_x * (np.log(p_x) - np.log(np.maximum(p_x_new, 1e-300))),
                    0.0,
                )
            )
        p_x = p_x_new
        if marginal_kl < tolerance:
            break

    # Final mutual information I(X;Y) at the converged input distribution
    p_y = np.maximum(p_x @ channel_matrix, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(
            channel_matrix > 0,
            np.log(channel_matrix) - np.log(p_y)[np.newaxis, :],
            0.0,
        )
    mi_nats = float(np.sum(p_x[:, np.newaxis] * channel_matrix * log_ratio))

    return float(mi_nats / ln_base)


def spatial_channel_capacity(
    coordinates: np.ndarray,
    signal_power: np.ndarray,
    noise_power: float,
    path_loss_exponent: float = 2.0,
    base: float = 2.0,
) -> float:
    """
    Calculate channel capacity for spatial communication system.

    Models capacity considering spatial path loss and interference.

    Args:
        coordinates: Spatial coordinates of transmitters/receivers (n x 2)
        signal_power: Signal power at each location (n)
        noise_power: Noise power
        path_loss_exponent: Path loss exponent (typically 2-4)
        base: Logarithm base

    Returns:
        Spatial channel capacity
    """
    coordinates = np.asarray(coordinates)
    signal_power = np.asarray(signal_power).flatten()

    if len(signal_power) != len(coordinates):
        raise ValueError("Signal power must have same length as coordinates")

    # Single-receiver geometry: transmitter-to-receiver distance is measured
    # from the origin, the reference point of the path-loss model
    distances = np.sqrt(np.sum(coordinates**2, axis=1))
    distances = np.maximum(distances, 1e-6)  # Avoid division by zero

    # Calculate received power with path loss
    received_power = signal_power / (distances**path_loss_exponent)

    # Calculate signal-to-noise ratio
    snr = received_power / noise_power
    snr = np.maximum(snr, 1e-10)  # Avoid log(0)

    # Channel capacity: C = log(1 + SNR)
    capacity = np.sum(np.log(1 + snr) / np.log(base))

    return float(capacity)


def awgn_channel_capacity(
    signal_power: float, noise_power: float, bandwidth: float = 1.0, base: float = 2.0
) -> float:
    """
    Calculate capacity of Additive White Gaussian Noise (AWGN) channel.

    Shannon capacity: C = B * log(1 + P/N)

    Args:
        signal_power: Signal power P
        noise_power: Noise power N
        bandwidth: Channel bandwidth B
        base: Logarithm base

    Returns:
        Channel capacity
    """
    if signal_power <= 0 or noise_power <= 0:
        raise ValueError("Power values must be positive")

    snr = signal_power / noise_power
    capacity = bandwidth * (np.log(1 + snr) / np.log(base))

    return float(capacity)


def mimo_channel_capacity(
    channel_matrix: np.ndarray,
    noise_power: float,
    power_constraint: float | None = None,
    base: float = 2.0,
) -> float:
    """
    Calculate capacity of Multiple-Input Multiple-Output (MIMO) channel.

    MIMO capacity: C = log(det(I + (1/N) * H * H^H))

    Args:
        channel_matrix: MIMO channel matrix H (n_r x n_t)
        noise_power: Noise power N
        power_constraint: Optional power constraint
        base: Logarithm base

    Returns:
        MIMO channel capacity
    """
    channel_matrix = np.asarray(channel_matrix)

    if channel_matrix.ndim != 2:
        raise ValueError("Channel matrix must be 2D")

    n_r, n_t = channel_matrix.shape

    # Calculate H * H^H
    h_hh = np.dot(channel_matrix, channel_matrix.conj().T)

    # Calculate capacity
    identity = np.eye(n_r)
    capacity_matrix = identity + (1.0 / noise_power) * h_hh

    # Calculate determinant
    det = np.linalg.det(capacity_matrix)

    if det <= 0:
        return 0.0

    capacity = np.log(det) / np.log(base)

    return float(capacity)


def waterfilling_power_allocation(
    channel_gains: np.ndarray, noise_power: float, total_power: float, base: float = 2.0
) -> tuple[np.ndarray, float]:
    """
    Calculate optimal power allocation using waterfilling algorithm.

    Waterfilling allocates more power to channels with better gains.

    Args:
        channel_gains: Channel gains for each subchannel
        noise_power: Noise power
        total_power: Total power constraint
        base: Logarithm base

    Returns:
        Tuple of (power_allocation, capacity)
    """
    channel_gains = np.asarray(channel_gains).flatten()
    n_channels = len(channel_gains)

    # Sort channels by gain (descending)
    sorted_indices = np.argsort(channel_gains)[::-1]
    sorted_gains = channel_gains[sorted_indices]

    # Waterfilling algorithm
    power_allocation = np.zeros(n_channels)
    water_level = 0.0

    # Find water level
    remaining_power = total_power

    for k in range(1, n_channels + 1):
        # Calculate water level for k channels
        inv_gains_sum = np.sum(1.0 / sorted_gains[:k])
        water_level_candidate = (remaining_power + inv_gains_sum * noise_power) / k

        # Check if water level is valid
        if k == n_channels or water_level_candidate <= noise_power / sorted_gains[k]:
            water_level = water_level_candidate
            break

    # Allocate power
    for i in range(n_channels):
        power = water_level - noise_power / sorted_gains[i]
        power_allocation[sorted_indices[i]] = max(0.0, power)

    # Calculate capacity
    capacity = 0.0
    for i in range(n_channels):
        if power_allocation[i] > 0:
            snr = (channel_gains[i] * power_allocation[i]) / noise_power
            capacity += np.log(1 + snr) / np.log(base)

    return power_allocation, float(capacity)


class ChannelCapacityCalculator:
    """
    Comprehensive channel capacity calculator for spatial communication.

    Provides methods for calculating channel capacity for various
    communication channel models.
    """

    def __init__(self, base: float = 2.0):
        """
        Initialize channel capacity calculator.

        Args:
            base: Logarithm base
        """
        self.base = base

    def calculate(
        self,
        channel_matrix: np.ndarray | None = None,
        signal_power: float | None = None,
        noise_power: float | None = None,
        method: str = "discrete",
    ) -> float:
        """
        Calculate channel capacity.

        Args:
            channel_matrix: Channel transition matrix
            signal_power: Signal power
            noise_power: Noise power
            method: Channel type ('discrete', 'awgn', 'mimo')

        Returns:
            Channel capacity
        """
        if method == "discrete":
            if channel_matrix is None:
                raise ValueError("Channel matrix required for discrete channel")
            return channel_capacity(channel_matrix, base=self.base)

        elif method == "awgn":
            if signal_power is None or noise_power is None:
                raise ValueError("Signal and noise power required for AWGN channel")
            return awgn_channel_capacity(signal_power, noise_power, base=self.base)

        elif method == "mimo":
            if channel_matrix is None or noise_power is None:
                raise ValueError(
                    "Channel matrix and noise power required for MIMO channel"
                )
            return mimo_channel_capacity(channel_matrix, noise_power, base=self.base)

        else:
            raise ValueError(f"Unknown method: {method}")

    def spatial_capacity(
        self,
        coordinates: np.ndarray,
        signal_power: np.ndarray,
        noise_power: float,
        **kwargs: Any,
    ) -> float:
        """
        Calculate spatial channel capacity.

        Args:
            coordinates: Spatial coordinates
            signal_power: Signal power at each location
            noise_power: Noise power
            **kwargs: Additional parameters

        Returns:
            Spatial channel capacity
        """
        return spatial_channel_capacity(
            coordinates, signal_power, noise_power, base=self.base, **kwargs
        )

    def waterfilling(
        self, channel_gains: np.ndarray, noise_power: float, total_power: float
    ) -> tuple[np.ndarray, float]:
        """
        Calculate optimal power allocation using waterfilling.

        Args:
            channel_gains: Channel gains
            noise_power: Noise power
            total_power: Total power constraint

        Returns:
            Tuple of (power_allocation, capacity)
        """
        return waterfilling_power_allocation(
            channel_gains, noise_power, total_power, base=self.base
        )
