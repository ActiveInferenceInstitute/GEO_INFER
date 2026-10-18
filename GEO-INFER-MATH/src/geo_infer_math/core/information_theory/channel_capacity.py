"""
Channel Capacity for Spatial Communication

This module provides channel capacity calculations for spatial
communication systems and data transmission.
"""

import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)


def _real_scalar(value: float, name: str, *, positive: bool = False) -> float:
    """Validate physical scalar parameters without accepting complex powers."""
    if np.ndim(value) != 0 or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite real scalar") from None
    if not np.isfinite(result) or result < 0 or (positive and result == 0):
        constraint = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {constraint}")
    return result


def _log_base(base: float) -> float:
    value = _real_scalar(base, "base", positive=True)
    if value <= 1:
        raise ValueError("base must be greater than one")
    return float(np.log(value))


def _real_array(values: np.ndarray, name: str) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be finite and real")
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite and real") from None
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and real")
    return result


def _log_one_plus_snr(log_snr: np.ndarray | float) -> np.ndarray:
    # A vanishing contribution may underflow to zero; neither SNR itself nor
    # its exponential needs to be representable to obtain a finite log-rate.
    with np.errstate(under="ignore"):
        return np.logaddexp(0.0, log_snr)


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
    ln_base = _log_base(base)

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
        coordinates: Finite transmitter coordinates relative to the origin (n x d)
        signal_power: Non-negative signal power at each location (n)
        noise_power: Finite positive noise power
        path_loss_exponent: Finite positive exponent (typically 2-4)
        base: Finite logarithm base greater than one

    Returns:
        Spatial channel capacity
    """
    coordinates = _real_array(coordinates, "coordinates")
    signal_power = _real_array(signal_power, "signal_power").flatten()
    noise_power = _real_scalar(noise_power, "noise_power", positive=True)
    path_loss_exponent = _real_scalar(
        path_loss_exponent, "path_loss_exponent", positive=True
    )
    ln_base = _log_base(base)

    if coordinates.ndim != 2 or coordinates.shape[1] == 0:
        raise ValueError("Coordinates must have shape (n, d) with d positive")
    if len(signal_power) != len(coordinates):
        raise ValueError("Signal power must have same length as coordinates")
    if np.any(signal_power < 0):
        raise ValueError("Signal power must be non-negative")
    if len(coordinates) == 0:
        return 0.0

    # Single-receiver geometry: transmitter-to-receiver distance is measured
    # from the origin. Preserve the existing minimum distance of 1e-6 in
    # coordinate units, computing norms in scaled/log form to avoid squaring
    # large finite coordinates or overflowing path-loss powers.
    scales = np.max(np.abs(coordinates), axis=1)
    log_distances = np.full(len(scales), np.log(1e-6))
    nonzero = scales > 0
    scaled = coordinates[nonzero] / scales[nonzero, np.newaxis]
    log_distances[nonzero] = np.maximum(
        np.log(scales[nonzero]) + 0.5 * np.log(np.sum(scaled**2, axis=1)),
        np.log(1e-6),
    )
    positive = signal_power > 0
    log_snr = (
        np.log(signal_power[positive])
        - np.log(noise_power)
        - path_loss_exponent * log_distances[positive]
    )
    return float(np.sum(_log_one_plus_snr(log_snr)) / ln_base)


def awgn_channel_capacity(
    signal_power: float, noise_power: float, bandwidth: float = 1.0, base: float = 2.0
) -> float:
    """
    Calculate capacity of Additive White Gaussian Noise (AWGN) channel.

    Shannon capacity: C = B * log(1 + P/N)

    Args:
        signal_power: Finite non-negative signal power P
        noise_power: Finite positive noise power N
        bandwidth: Finite non-negative channel bandwidth B
        base: Finite logarithm base greater than one

    Returns:
        Channel capacity
    """
    signal_power = _real_scalar(signal_power, "signal_power")
    noise_power = _real_scalar(noise_power, "noise_power", positive=True)
    bandwidth = _real_scalar(bandwidth, "bandwidth")
    ln_base = _log_base(base)
    if signal_power == 0 or bandwidth == 0:
        return 0.0
    log_snr = np.log(signal_power) - np.log(noise_power)
    return float(bandwidth * _log_one_plus_snr(log_snr) / ln_base)


def mimo_channel_capacity(
    channel_matrix: np.ndarray,
    noise_power: float,
    power_constraint: float | None = None,
    base: float = 2.0,
) -> float:
    """
    Calculate the MIMO rate for independent equal-power transmit streams.

    With total power P, C = log(det(I + P/(n_t*N) * H * H^H)).
    This is an isotropic transmit covariance, rather than an optimization
    over channel-dependent transmit covariances. Singular values evaluate
    the log-determinant without forming H * H^H or a determinant.

    Args:
        channel_matrix: Finite real or complex channel matrix H (n_r x n_t)
        noise_power: Finite positive noise power N
        power_constraint: Finite non-negative total transmit power P, divided
            equally among n_t transmitters. None preserves the original unit
            power per transmitter (total power n_t).
        base: Finite logarithm base greater than one

    Returns:
        MIMO channel capacity
    """
    dtype = complex if np.iscomplexobj(channel_matrix) else float
    channel_matrix = np.asarray(channel_matrix, dtype=dtype)
    if channel_matrix.ndim != 2 or 0 in channel_matrix.shape:
        raise ValueError("Channel matrix must be non-empty and 2D")
    if not np.all(np.isfinite(channel_matrix)):
        raise ValueError("Channel matrix must be finite")
    noise_power = _real_scalar(noise_power, "noise_power", positive=True)
    ln_base = _log_base(base)
    if power_constraint is None:
        log_power_per_transmitter = 0.0
    else:
        power_constraint = _real_scalar(power_constraint, "power_constraint")
        if power_constraint == 0:
            return 0.0
        log_power_per_transmitter = np.log(power_constraint) - np.log(
            channel_matrix.shape[1]
        )

    # Preserve finite native singular values. Unconditional normalization can
    # erase a small but material mode when gains span a large dynamic range.
    singular_values = np.linalg.svd(channel_matrix, compute_uv=False)
    positive = singular_values > 0
    log_gains = np.empty(len(singular_values))
    finite = positive & np.isfinite(singular_values)
    log_gains[finite] = np.log(singular_values[finite])
    overflowing = np.isposinf(singular_values)
    if np.any(overflowing):
        # A finite matrix can have a singular value above float's maximum.
        # Re-evaluate only those modes in log scale; retain every finite mode
        # from the native computation. Component scaling also avoids abs(H)
        # overflowing when both complex components are large finite numbers.
        scale = max(
            float(np.max(np.abs(channel_matrix.real))),
            float(np.max(np.abs(channel_matrix.imag))),
        )
        with np.errstate(under="ignore"):
            normalized = channel_matrix.real / scale
            if np.iscomplexobj(channel_matrix):
                normalized = normalized + 1j * (channel_matrix.imag / scale)
        scaled_values = np.linalg.svd(normalized, compute_uv=False)
        log_gains[overflowing] = np.log(scaled_values[overflowing]) + np.log(scale)
    log_snr = 2 * log_gains[positive] + log_power_per_transmitter - np.log(noise_power)
    return float(np.sum(_log_one_plus_snr(log_snr)) / ln_base)


def waterfilling_power_allocation(
    channel_gains: np.ndarray, noise_power: float, total_power: float, base: float = 2.0
) -> tuple[np.ndarray, float]:
    """
    Calculate optimal power allocation using waterfilling algorithm.

    Waterfilling allocates more power to channels with better gains.

    Args:
        channel_gains: Finite non-negative power gains for each subchannel
        noise_power: Finite positive noise power
        total_power: Finite non-negative total power budget
        base: Finite logarithm base greater than one

    Returns:
        Tuple of (power_allocation in original channel order, capacity).
        Zero-gain channels receive zero power; if every gain is zero the
        budget is left unused. A zero budget produces zero capacity.
    """
    channel_gains = _real_array(channel_gains, "channel_gains").flatten()
    if np.any(channel_gains < 0):
        raise ValueError("Channel gains must be non-negative")
    noise_power = _real_scalar(noise_power, "noise_power", positive=True)
    total_power = _real_scalar(total_power, "total_power")
    ln_base = _log_base(base)
    power_allocation = np.zeros(len(channel_gains))
    positive_indices = np.flatnonzero(channel_gains > 0)
    if total_power == 0 or len(positive_indices) == 0:
        return power_allocation, 0.0

    sorted_indices = positive_indices[np.argsort(channel_gains[positive_indices])[::-1]]
    gains = channel_gains[sorted_indices]
    # Subtract the best channel's floor N/g_max before waterfilling. Floors
    # themselves can overflow, even when their differences and allocation
    # are representable. Express the differences in units of total_power:
    # (N/g_i - N/g_max) / P = N*(g_max-g_i)/(g_i*g_max*P).
    log_offsets = np.full(len(gains), -np.inf)
    weaker = gains < gains[0]
    log_offsets[weaker] = (
        np.log(noise_power)
        + np.log(gains[0] - gains[weaker])
        - np.log(gains[weaker])
        - np.log(gains[0])
        - np.log(total_power)
    )
    eligible = log_offsets <= 0  # A floor difference above P cannot be active.
    sorted_indices = sorted_indices[eligible]
    with np.errstate(under="ignore"):
        offsets = np.exp(log_offsets[eligible])
    levels = (1 + np.cumsum(offsets)) / np.arange(1, len(offsets) + 1)
    stops = np.flatnonzero(levels[:-1] <= offsets[1:])
    active = int(stops[0] + 1) if len(stops) else len(offsets)
    power_allocation[sorted_indices[:active]] = total_power * np.maximum(
        levels[active - 1] - offsets[:active], 0
    )
    used = power_allocation > 0
    log_snr = (
        np.log(channel_gains[used])
        + np.log(power_allocation[used])
        - np.log(noise_power)
    )
    return power_allocation, float(np.sum(_log_one_plus_snr(log_snr)) / ln_base)


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
