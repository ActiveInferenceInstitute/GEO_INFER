"""
GEO-INFER-EXAMPLES: Comprehensive demonstration framework for the GEO-INFER ecosystem.
"""

from . import core, models
from .core import (
    ExecutionStrategy,
    ModuleOrchestrator,
    ModuleStatus,
)
from .models import WorkflowDefinition

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "core",
    "models",
    "ExecutionStrategy",
    "ModuleOrchestrator",
    "ModuleStatus",
    "WorkflowDefinition",
]
