"""
Spatial Coding and Compression

This module provides spatial data compression and coding theory
tools for efficient spatial data representation.
"""

import numpy as np
from typing import Any
import logging
import zlib
import gzip

logger = logging.getLogger(__name__)


def spatial_encoding_efficiency(
    original_data: np.ndarray, encoded_data: Any, method: str = "compression_ratio"
) -> float:
    """
    Calculate encoding efficiency for spatial data.

    Args:
        original_data: Original spatial data
        encoded_data: Encoded/compressed data
        method: Efficiency measure ('compression_ratio', 'bits_per_sample')

    Returns:
        Encoding efficiency metric
    """
    original_data = np.asarray(original_data)

    # Calculate original size
    if original_data.dtype == np.float64:
        original_bits = original_data.size * 64
    elif original_data.dtype == np.float32:
        original_bits = original_data.size * 32
    elif original_data.dtype == np.int32:
        original_bits = original_data.size * 32
    elif original_data.dtype == np.int16:
        original_bits = original_data.size * 16
    else:
        original_bits = original_data.size * 8

    # Calculate encoded size
    if isinstance(encoded_data, (bytes, bytearray)):
        encoded_bits = len(encoded_data) * 8
    elif isinstance(encoded_data, np.ndarray):
        if encoded_data.dtype == np.float64:
            encoded_bits = encoded_data.size * 64
        elif encoded_data.dtype == np.float32:
            encoded_bits = encoded_data.size * 32
        else:
            encoded_bits = encoded_data.size * 8
    else:
        encoded_bits = len(str(encoded_data)) * 8

    if method == "compression_ratio":
        if encoded_bits == 0:
            return 0.0
        return float(original_bits / encoded_bits)

    elif method == "bits_per_sample":
        if original_data.size == 0:
            return 0.0
        return float(encoded_bits / original_data.size)

    else:
        raise ValueError(f"Unknown method: {method}")


def compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    Calculate compression ratio.

    Compression ratio: CR = original_size / compressed_size

    Args:
        original_size: Original data size in bytes
        compressed_size: Compressed data size in bytes

    Returns:
        Compression ratio (higher is better)
    """
    if compressed_size == 0:
        return 0.0

    return float(original_size / compressed_size)


def coding_gain(
    original_snr: float, encoded_snr: float, method: str = "linear"
) -> float:
    """
    Calculate coding gain.

    Coding gain measures improvement in signal-to-noise ratio
    achieved through coding.

    Args:
        original_snr: Original signal-to-noise ratio
        encoded_snr: Encoded signal-to-noise ratio
        method: Gain calculation method ('linear', 'db')

    Returns:
        Coding gain
    """
    if original_snr <= 0:
        raise ValueError("Original SNR must be positive")

    if method == "linear":
        gain = encoded_snr / original_snr
        return float(gain)

    elif method == "db":
        gain_db = 10 * np.log10(encoded_snr / original_snr)
        return float(gain_db)

    else:
        raise ValueError(f"Unknown method: {method}")


def spatial_compression(
    coordinates: np.ndarray,
    values: np.ndarray,
    method: str = "quantization",
    compression_level: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Compress spatial data using various methods.

    Args:
        coordinates: Spatial coordinates (n x 2)
        values: Values at each location (n)
        method: Compression method ('quantization', 'dct', 'wavelet')
        compression_level: Compression level (0 to 1, higher = more compression)

    Returns:
        Tuple of (compressed_data, metadata)
    """
    coordinates = np.asarray(coordinates)
    values = np.asarray(values).flatten()

    if len(values) != len(coordinates):
        raise ValueError("Values must have same length as coordinates")

    metadata = {
        "original_size": values.size,
        "method": method,
        "compression_level": compression_level,
    }

    if method == "quantization":
        # Quantization-based compression
        n_levels = max(2, int(values.size * (1 - compression_level)))

        # Quantize values
        value_min = np.min(values)
        value_max = np.max(values)
        value_range = value_max - value_min

        if value_range > 0:
            quantized = np.round((values - value_min) / value_range * (n_levels - 1))
            quantized = quantized.astype(np.int32)

            metadata["n_levels"] = n_levels
            metadata["value_range"] = (value_min, value_max)
            metadata["compression_ratio"] = compression_ratio(
                values.size * 8, quantized.size * np.int32().itemsize * 8
            )

            return quantized, metadata
        else:
            # Constant values
            return values, metadata

    elif method == "dct":
        # Discrete Cosine Transform compression
        # Reshape to 2D grid if possible
        try:
            # Try to create grid
            x_coords = np.unique(coordinates[:, 0])
            y_coords = np.unique(coordinates[:, 1])

            if len(x_coords) * len(y_coords) == len(values):
                # Create grid
                grid = np.zeros((len(y_coords), len(x_coords)))
                coord_to_idx = {}
                for i, x in enumerate(x_coords):
                    for j, y in enumerate(y_coords):
                        coord_to_idx[(x, y)] = (j, i)

                for k, coord in enumerate(coordinates):
                    idx = coord_to_idx.get((coord[0], coord[1]))
                    if idx:
                        grid[idx] = values[k]

                # Apply DCT
                from scipy.fft import dctn

                dct_coeffs = dctn(grid, norm="ortho")

                # Keep only top coefficients
                n_keep = int(dct_coeffs.size * (1 - compression_level))
                flat_coeffs = dct_coeffs.flatten()
                sorted_indices = np.argsort(np.abs(flat_coeffs))[::-1]

                compressed_coeffs = np.zeros_like(flat_coeffs)
                compressed_coeffs[sorted_indices[:n_keep]] = flat_coeffs[
                    sorted_indices[:n_keep]
                ]

                compressed_coeffs = compressed_coeffs.reshape(dct_coeffs.shape)

                metadata["n_coefficients"] = n_keep
                metadata["compression_ratio"] = compression_ratio(
                    values.size * 8, n_keep * 8
                )

                return compressed_coeffs, metadata
            else:
                return spatial_compression(
                    coordinates,
                    values,
                    method="quantization",
                    compression_level=compression_level,
                )
        except Exception as e:
            logger.warning(f"DCT compression failed: {e}, using quantization")
            return spatial_compression(
                coordinates,
                values,
                method="quantization",
                compression_level=compression_level,
            )

    elif method == "wavelet":
        # Discrete wavelet compression: multi-level Daubechies-4 wavelet
        # decomposition of the value series followed by hard thresholding
        # that keeps the largest-magnitude coefficients.
        try:
            import pywt
        except ImportError as exc:
            raise ValueError(
                "method='wavelet' is not supported without the PyWavelets "
                "package; install it with 'pip install PyWavelets' or use "
                "method='dct' or method='quantization' instead"
            ) from exc

        values_arr = np.asarray(values, dtype=np.float64).ravel()
        coeffs = pywt.wavedec(values_arr, "db4", mode="symmetric")

        # Keep only significant coefficients (largest magnitudes)
        threshold = np.percentile(
            np.abs(np.concatenate(coeffs)), 100 * compression_level
        )

        compressed_coeffs = [c * (np.abs(c) >= threshold) for c in coeffs]

        n_coeffs = sum(len(c) for c in compressed_coeffs)
        metadata["n_coefficients"] = n_coeffs
        metadata["threshold"] = float(threshold)
        metadata["compression_ratio"] = compression_ratio(
            values_arr.size * 8, n_coeffs * 8
        )

        return np.concatenate(compressed_coeffs), metadata

    else:
        raise ValueError(f"Unknown method: {method}")


def entropy_coding(
    data: np.ndarray, method: str = "huffman"
) -> tuple[bytes, dict[str, Any]]:
    """
    Apply entropy coding to spatial data.

    Args:
        data: Input data
        method: Coding method ('huffman', 'arithmetic', 'rle')

    Returns:
        Tuple of (encoded_data, metadata)
    """
    data = np.asarray(data).flatten()

    metadata = {"original_size": data.size, "method": method}

    if method == "rle":
        # Run-length encoding
        encoded_list = []
        current_value = data[0]
        count = 1

        for value in data[1:]:
            if value == current_value:
                count += 1
            else:
                encoded_list.extend([current_value, count])
                current_value = value
                count = 1

        encoded_list.extend([current_value, count])
        encoded_arr = np.array(encoded_list, dtype=data.dtype)

        metadata["compressed_size"] = encoded_arr.size
        metadata["compression_ratio"] = compression_ratio(data.size, encoded_arr.size)

        return encoded_arr.tobytes(), metadata

    elif method == "gzip":
        # GZIP compression
        compressed = gzip.compress(data.tobytes())

        metadata["compressed_size"] = len(compressed)
        metadata["compression_ratio"] = compression_ratio(
            len(data.tobytes()), len(compressed)
        )

        return compressed, metadata

    elif method == "zlib":
        # ZLIB compression
        compressed = zlib.compress(data.tobytes())

        metadata["compressed_size"] = len(compressed)
        metadata["compression_ratio"] = compression_ratio(
            len(data.tobytes()), len(compressed)
        )

        return compressed, metadata

    else:
        raise ValueError(f"Unknown method: {method}")


class SpatialCodingCalculator:
    """
    Comprehensive spatial coding and compression calculator.

    Provides methods for compressing and encoding spatial data
    efficiently.
    """

    def __init__(self) -> None:
        """Initialize spatial coding calculator."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._codec_registry: dict[str, Any] = {}
        self.logger.debug("SpatialCodingCalculator initialized")

    def compress(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        method: str = "quantization",
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Compress spatial data.

        Args:
            coordinates: Spatial coordinates
            values: Values at locations
            method: Compression method
            **kwargs: Additional parameters

        Returns:
            Tuple of (compressed_data, metadata)
        """
        return spatial_compression(coordinates, values, method=method, **kwargs)

    def encode(
        self, data: np.ndarray, method: str = "gzip", **kwargs: Any
    ) -> tuple[bytes, dict[str, Any]]:
        """
        Apply entropy coding.

        Args:
            data: Input data
            method: Coding method
            **kwargs: Additional parameters

        Returns:
            Tuple of (encoded_data, metadata)
        """
        return entropy_coding(data, method=method, **kwargs)

    def efficiency(
        self,
        original_data: np.ndarray,
        encoded_data: np.ndarray | bytes,
        **kwargs: Any,
    ) -> float:
        """
        Calculate encoding efficiency.

        Args:
            original_data: Original data
            encoded_data: Encoded data
            **kwargs: Additional parameters

        Returns:
            Encoding efficiency metric
        """
        return spatial_encoding_efficiency(original_data, encoded_data, **kwargs)
