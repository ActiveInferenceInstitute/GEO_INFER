"""
Core mathematical components for geospatial analysis.

This package provides fundamental mathematical operations and algorithms
that serve as building blocks for more complex geospatial analysis. Its
public interface is the set of submodules listed in ``__all__``; import names
from the owning submodule, e.g.
``from geo_infer_math.core.spatial_statistics import MoranI``. The curated
top-level API lives in :mod:`geo_infer_math`.
"""

from geo_infer_math.core import (
    circulation,
    geometry,
    gpu_acceleration,
    graph_theory,
    information_theory,
    interpolation,
    linalg_tensor,
    numerical_methods,
    optimization,
    spatial_statistics,
    symbolic_math,
    theorem_proving,
    transforms,
)

__all__ = [
    "circulation",
    "spatial_statistics",
    "interpolation",
    "optimization",
    "geometry",
    "numerical_methods",
    "linalg_tensor",
    "transforms",
    "graph_theory",
    "gpu_acceleration",
    "symbolic_math",
    "information_theory",
    "theorem_proving",
]
