"""
Data compression utilities for GEO-INFER-DATA.

This module provides data compression capabilities for efficient storage
and transmission of geospatial data.
"""

import logging
from typing import Any
import gzip
import lzma
import bz2
import pickle
from io import BytesIO

import geopandas as gpd
import pandas as pd
import numpy as np

from ..models.schemas import DataFormat
from .secure_serialization import (
    CONTEXT_COMPRESSION,
    UnsignedPayloadError,
    is_signed_envelope,
    verify_payload,
)


logger = logging.getLogger(__name__)


class DataCompressor:
    """
    Data compression for efficient storage.

    This class provides compression utilities for geospatial data
    including various compression algorithms and format-specific
    optimizations.

    Examples:
        >>> compressor = DataCompressor()
        >>>
        >>> # Compress geospatial data
        >>> compressed_data = compressor.compress_data(geodataframe)
        >>>
        >>> # Decompress data
        >>> decompressed_data = compressor.decompress_data(compressed_data)
        >>>
        >>> # Get compression statistics
        >>> stats = compressor.get_compression_stats()
        >>> print(f"Compression ratio: {stats['ratio']:.2f}x")
    """

    def __init__(self, algorithm: str = "gzip", level: int = 6):
        self.algorithm = algorithm
        self.level = level
        self.compression_stats = {
            "total_compressed": 0,
            "total_original": 0,
            "compression_count": 0,
        }

        logger.info(f"Initialized DataCompressor with {algorithm} algorithm")

    def is_enabled(self) -> bool:
        """Check if compression is enabled."""
        return self.algorithm != "none"

    def compress_data(self, data: Any, format: DataFormat | None = None) -> bytes:
        """
        Compress geospatial data.

        Args:
            data: Data to compress
            format: Data format hint

        Returns:
            Compressed data as bytes
        """
        logger.debug(f"Compressing data with {self.algorithm} algorithm")

        # Serialize data first
        serialized_data = self._serialize_data(data, format)

        # Compress based on algorithm
        if self.algorithm == "gzip":
            compressed_data = gzip.compress(serialized_data, compresslevel=self.level)
        elif self.algorithm == "lzma":
            compressed_data = lzma.compress(serialized_data, preset=self.level)
        elif self.algorithm == "bz2":
            compressed_data = bz2.compress(serialized_data, compresslevel=self.level)
        elif self.algorithm == "none":
            compressed_data = serialized_data
        else:
            raise ValueError(f"Unknown compression algorithm: {self.algorithm}")

        # Update statistics
        self.compression_stats["total_compressed"] += len(compressed_data)
        self.compression_stats["total_original"] += len(serialized_data)
        self.compression_stats["compression_count"] += 1

        logger.debug(
            f"Compressed data: {len(serialized_data)} -> {len(compressed_data)} bytes"
        )
        return compressed_data

    def decompress_data(
        self,
        compressed_data: bytes,
        format: DataFormat | None = None,
        verified: bool = False,
    ) -> Any:
        """
        Decompress geospatial data.

        Args:
            compressed_data: Compressed data bytes
            format: Data format hint

        Returns:
            Decompressed data
        """
        logger.debug(f"Decompressing data with {self.algorithm} algorithm")

        # Decompress based on algorithm
        if self.algorithm == "gzip":
            decompressed_data = gzip.decompress(compressed_data)
        elif self.algorithm == "lzma":
            decompressed_data = lzma.decompress(compressed_data)
        elif self.algorithm == "bz2":
            decompressed_data = bz2.decompress(compressed_data)
        elif self.algorithm == "none":
            decompressed_data = compressed_data
        else:
            raise ValueError(f"Unknown compression algorithm: {self.algorithm}")

        # Deserialize data (authenticated-serialization enforced)
        data = self._deserialize_data(decompressed_data, format, verified=verified)

        logger.debug(
            f"Decompressed data: {len(compressed_data)} -> {len(decompressed_data)} bytes"
        )
        return data

    def _serialize_data(self, data: Any, format: DataFormat | None = None) -> bytes:
        """Serialize data to bytes."""
        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            if format == DataFormat.PARQUET or (format is None and len(data) > 1000):
                # Use Parquet for large datasets
                res = data.to_parquet()
                return bytes(res) if isinstance(res, (bytes, bytearray)) else b""
            else:
                # Use pickle for smaller datasets or complex objects
                return pickle.dumps(data)
        elif isinstance(data, np.ndarray):
            return pickle.dumps(data)
        elif isinstance(data, dict):
            return pickle.dumps(data)
        else:
            return pickle.dumps(data)

    def _deserialize_data(
        self,
        data: bytes,
        format: DataFormat | None = None,
        verified: bool = False,
    ) -> Any:
        """Deserialize data from bytes.

        Pickle payloads are only deserialised behind an authenticated
        GISP1 envelope: either the payload carries its own envelope
        (verified here with :func:`verify_payload`) or the caller passes
        ``verified=True`` to vouch that the bytes were already verified
        at the transport trust boundary (e.g. after HMAC verification in
        a storage backend). Bare pickles are rejected.

        Parquet payloads are parsed without pickle and need no envelope.

        Raises:
            UnsignedPayloadError: If an unverified pickle payload carries
                no GISP1 envelope.
            PayloadSecurityError: If envelope verification fails.
        """
        if format == DataFormat.PARQUET or data[:4] == b"PAR1":
            # pandas engines require a file-like object for in-memory parquet.
            return pd.read_parquet(BytesIO(data))
        if verified:
            return pickle.loads(data)
        if not is_signed_envelope(data):
            raise UnsignedPayloadError(
                "decompress_data() refuses bare pickle payloads: pass a "
                "GISP1-signed envelope or verified=True for bytes already "
                "verified by the caller"
            )
        return pickle.loads(verify_payload(data, context=CONTEXT_COMPRESSION, key=None))

    def get_compression_stats(self) -> dict[str, Any]:
        """Get compression statistics."""
        total_compressed = self.compression_stats["total_compressed"]
        total_original = self.compression_stats["total_original"]

        if total_original > 0:
            compression_ratio = total_original / total_compressed
        else:
            compression_ratio = 1.0

        return {
            "algorithm": self.algorithm,
            "level": self.level,
            "total_compressed_bytes": total_compressed,
            "total_original_bytes": total_original,
            "compression_ratio": compression_ratio,
            "compression_count": self.compression_stats["compression_count"],
        }

    def optimize_for_storage(self, data: Any) -> dict[str, Any]:
        """
        Optimize data for storage with compression recommendations.

        Args:
            data: Data to analyze

        Returns:
            Optimization recommendations
        """
        recommendations = {
            "recommended_compression": self.algorithm,
            "estimated_savings": 0.0,
            "format_recommendation": DataFormat.CSV,
        }

        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
            # Analyze data characteristics
            if len(data) > 10000:
                recommendations["format_recommendation"] = DataFormat.PARQUET
                recommendations["estimated_savings"] = 0.7  # 70% compression
            elif isinstance(data, gpd.GeoDataFrame):
                recommendations["format_recommendation"] = DataFormat.GEOPACKAGE
                recommendations["estimated_savings"] = 0.5  # 50% compression
            else:
                recommendations["format_recommendation"] = DataFormat.CSV
                recommendations["estimated_savings"] = 0.3  # 30% compression

        return recommendations
