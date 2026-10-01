"""
Nested H3 Hexagonal Grid Systems for GEO-INFER-SPACE.

This module provides advanced capabilities for managing nested H3 hexagonal grids,
including hierarchical structures, boundary operations, message passing, and
comprehensive analytics for complex geospatial modeling.

The nested module is organized into several key components:

- **Core**: Fundamental data structures and hierarchy management
- **Boundaries**: Boundary detection, analysis, and management
- **Messaging**: Message passing and routing across boundaries
- **Operations**: Lumping, splitting, and aggregation operations
- **Analytics**: Flow analysis, hierarchy metrics, pattern detection, and performance analysis

Key Features:
- Hierarchical H3 grid management with parent-child relationships
- Advanced boundary detection and flow management
- Comprehensive message passing with multiple protocols
- Dynamic grid operations (lumping, splitting, aggregation)
- Real-time analytics and pattern detection
- Performance monitoring and optimization
"""

import logging
from typing import Any

# Core components
from .core.nested_grid import NestedH3Grid, NestedCell
from .core.hierarchy import HierarchyManager

# Boundary operations
from .boundaries.boundary_manager import H3BoundaryManager
from .boundaries.detector import BoundaryDetector, BoundarySegment, BoundaryType

logger = logging.getLogger(__name__)

# Message passing
try:
    from .messaging.message_broker import (
        H3MessageBroker as H3MessageBroker,
        Message as Message,
        MessageType as MessageType,
    )
    from .messaging.routing import (
        MessageRouter as MessageRouter,
        RoutingStrategy as RoutingStrategy,
    )
    from .messaging.protocols import (
        MessageProtocol as MessageProtocol,
        ProtocolType as ProtocolType,
    )

    MESSAGING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Messaging components not fully available: {e}")
    MESSAGING_AVAILABLE = False

# Operations
try:
    from .operations.lumping import (
        H3LumpingEngine as H3LumpingEngine,
        LumpingStrategy as LumpingStrategy,
    )
    from .operations.splitting import (
        H3SplittingEngine as H3SplittingEngine,
        SplittingStrategy as SplittingStrategy,
    )
    from .operations.aggregation import (
        H3AggregationEngine as H3AggregationEngine,
        AggregationFunction as AggregationFunction,
    )

    OPERATIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Operations components not fully available: {e}")
    OPERATIONS_AVAILABLE = False

# Analytics
try:
    from .analytics.flow_analysis import (
        H3FlowAnalyzer as H3FlowAnalyzer,
        FlowType as FlowType,
        FlowPattern as FlowPattern,
    )
    from .analytics.hierarchy_metrics import (
        H3HierarchyAnalyzer as H3HierarchyAnalyzer,
        HierarchyMetric as HierarchyMetric,
    )
    from .analytics.pattern_detection import (
        H3PatternDetector as H3PatternDetector,
        PatternType as PatternType,
    )
    from .analytics.performance_metrics import (
        H3PerformanceAnalyzer as H3PerformanceAnalyzer,
        PerformanceMetric as PerformanceMetric,
    )

    ANALYTICS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Analytics components not fully available: {e}")
    ANALYTICS_AVAILABLE = False

# Define public API
__all__ = [
    # Core
    "NestedH3Grid",
    "NestedCell",
    "HierarchyManager",
    # Boundaries
    "H3BoundaryManager",
    "BoundaryDetector",
    "BoundarySegment",
    "BoundaryType",
]

# Add messaging components if available
if MESSAGING_AVAILABLE:
    __all__.extend(
        [
            "H3MessageBroker",
            "Message",
            "MessageType",
            "MessageRouter",
            "RoutingStrategy",
            "MessageProtocol",
            "ProtocolType",
        ]
    )

# Add operations components if available
if OPERATIONS_AVAILABLE:
    __all__.extend(
        [
            "H3LumpingEngine",
            "LumpingStrategy",
            "H3SplittingEngine",
            "SplittingStrategy",
            "H3AggregationEngine",
            "AggregationFunction",
        ]
    )

# Add analytics components if available
if ANALYTICS_AVAILABLE:
    __all__.extend(
        [
            "H3FlowAnalyzer",
            "FlowType",
            "FlowPattern",
            "H3HierarchyAnalyzer",
            "HierarchyMetric",
            "H3PatternDetector",
            "PatternType",
            "H3PerformanceAnalyzer",
            "PerformanceMetric",
        ]
    )

# Module metadata
__version__ = "1.0.0"
__author__ = "GEO-INFER Development Team"
__description__ = "Nested H3 Hexagonal Grid Systems for Advanced Geospatial Modeling"

# Availability flags for external checking
NESTED_COMPONENTS = {
    "core": True,
    "boundaries": True,
    "messaging": MESSAGING_AVAILABLE,
    "operations": OPERATIONS_AVAILABLE,
    "analytics": ANALYTICS_AVAILABLE,
}


def get_component_status() -> dict:
    """
    Get the availability status of nested module components.

    Returns:
        Dictionary with component availability status
    """
    return {
        "nested_module_version": __version__,
        "components_available": NESTED_COMPONENTS,
        "total_components": len(NESTED_COMPONENTS),
        "available_components": sum(NESTED_COMPONENTS.values()),
        "component_details": {
            "core": "Nested grid structures and hierarchy management",
            "boundaries": "Boundary detection and management",
            "messaging": "Message passing and routing systems",
            "operations": "Lumping, splitting, and aggregation operations",
            "analytics": "Flow analysis, pattern detection, and performance metrics",
        },
    }


def create_nested_system(system_id: str, **kwargs: Any) -> NestedH3Grid:
    """
    Create a new nested H3 system with default configuration.

    Args:
        system_id: Unique identifier for the system
        **kwargs: Additional configuration parameters

    Returns:
        Configured NestedH3Grid instance
    """
    return NestedH3Grid(name=system_id, **kwargs)


# Log module initialization
logger.info(
    "Nested H3 module initialized - "
    f"{sum(NESTED_COMPONENTS.values())}/{len(NESTED_COMPONENTS)} "
    "components available"
)
