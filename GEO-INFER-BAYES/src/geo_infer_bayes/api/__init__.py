"""API interfaces for Bayesian inference engines."""

from typing import Any

# External library interfaces
try:
    from .pymc_interface import PyMCInterface
except ImportError:
    PyMCInterface: type[Any] | None = None  # type: ignore[no-redef]

try:
    from .stan_interface import StanInterface
except ImportError:
    StanInterface: type[Any] | None = None  # type: ignore[no-redef]

try:
    from .tfp_interface import TFPInterface
except ImportError:
    TFPInterface: type[Any] | None = None  # type: ignore[no-redef]

__all__ = ["PyMCInterface", "StanInterface", "TFPInterface"]
