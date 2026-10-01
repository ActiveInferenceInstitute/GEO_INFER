"""
Custom Exceptions for GEO-INFER-MATH

This module provides custom exception classes for better error handling.
"""


class MathError(Exception):
    """Base exception for mathematical operations."""


class ValidationError(MathError):
    """Exception for invalid mathematical or geospatial inputs."""


class MemoryError(MathError):
    """Exception for memory-constrained mathematical operations."""


class NumericalError(MathError):
    """Exception for numerical computation errors."""


class ConvergenceError(NumericalError):
    """Exception for convergence failures in iterative methods."""


class SingularMatrixError(NumericalError):
    """Exception for singular matrix operations."""


class TheoremProvingError(MathError):
    """Exception for theorem proving errors."""


class ProofVerificationError(TheoremProvingError):
    """Exception for proof verification errors."""


class InformationTheoryError(MathError):
    """Exception for information theory errors."""


class InvalidDistributionError(InformationTheoryError):
    """Exception for invalid probability distributions."""


class SpatialError(MathError):
    """Exception for spatial operation errors."""


class CoordinateError(SpatialError):
    """Exception for coordinate transformation errors."""


class GeometryError(SpatialError):
    """Exception for geometric operation errors."""
