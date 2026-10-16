"""
GEO-INFER-EXAMPLES: Comprehensive demonstration framework for the GEO-INFER ecosystem.
"""

from . import core, models
from .core import (
    LocalAPIConnector,
    ExecutionStrategy,
    ModuleOrchestrator,
    ModuleStatus,
)
from .models import WorkflowDefinition

__version__ = "0.4.0"

__all__ = [
    "LocalAPIConnector",
    "__version__",
    "core",
    "models",
    "ExecutionStrategy",
    "ModuleOrchestrator",
    "ModuleStatus",
    "WorkflowDefinition",
]
