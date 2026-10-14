"""
Core analysis modules for GEO-INFER-PLACE

Contains the core analytical components for place-based geospatial analysis:
- BaseAnalysisModule: Abstract base for domain-specific analyzers
- CascadianAgriculturalH3Backend: H3-indexed Cascadia backend
- InteractiveVisualizationEngine: Dashboard and map generation
- PlaceInterface: Unified entry point for all analyses
- PlaceDataManager: GEO-INFER-DATA quality and provenance bridge
- PlaceTemporalAnalyzer: GEO-INFER-TIME trend and anomaly bridge
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_module import BaseAnalysisModule
    from .unified_backend import CascadianAgriculturalH3Backend
    from .visualization_engine import InteractiveVisualizationEngine
    from .place_interface import PlaceInterface
    from .module_bridge import PlaceDataManager, PlaceTemporalAnalyzer

_EXPORTS = {
    "BaseAnalysisModule": "base_module",
    "CascadianAgriculturalH3Backend": "unified_backend",
    "InteractiveVisualizationEngine": "visualization_engine",
    "PlaceInterface": "place_interface",
    "PlaceDataManager": "module_bridge",
    "PlaceTemporalAnalyzer": "module_bridge",
}


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value: object = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "BaseAnalysisModule",
    "CascadianAgriculturalH3Backend",
    "InteractiveVisualizationEngine",
    "PlaceInterface",
    "PlaceDataManager",
    "PlaceTemporalAnalyzer",
]
