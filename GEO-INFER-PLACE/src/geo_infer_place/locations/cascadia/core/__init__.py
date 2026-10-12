"""
Core Processing Package

Contains data management, H3 fusion, analysis engines, and visualization utilities.
"""

from .enhanced_logging import (
    DataSourceLogger,
    ProcessingLogger,
    VisualizationLogger,
)

# These imports depend on geo_infer_place which may not be installed
try:
    from .enhanced_data_manager import EnhancedDataManager
except ImportError:
    EnhancedDataManager = None

try:
    from .enhanced_h3_fusion import EnhancedH3Fusion
except ImportError:
    EnhancedH3Fusion = None

__all__ = [
    "EnhancedDataManager",
    "EnhancedH3Fusion",
    "DataSourceLogger",
    "ProcessingLogger",
    "VisualizationLogger",
]
