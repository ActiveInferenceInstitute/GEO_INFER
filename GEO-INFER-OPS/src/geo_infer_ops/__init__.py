"""
GEO-INFER-OPS: Operations and infrastructure management for GEO-INFER framework.

This module provides the core infrastructure for the GEO-INFER framework, ensuring
reliable, scalable, and maintainable operations across all components.
"""

__version__ = "0.4.0"

from geo_infer_ops.core.monitoring import setup_monitoring
from geo_infer_ops.core.config import load_config, get_config
from geo_infer_ops.core.orchestrator import Orchestrator, Task, TaskStatus
from geo_infer_ops.health.checks import HealthChecker, HealthStatus, HealthCheck

# kubernetes is a declared runtime dependency, so this import is unconditional;
# DeploymentManager is always available.
from geo_infer_ops.core.deployment import DeploymentManager

__all__ = [
    "setup_monitoring",
    "load_config",
    "get_config",
    "setup_testing",
    "Orchestrator",
    "Task",
    "TaskStatus",
    "DeploymentManager",
    "HealthChecker",
    "HealthStatus",
    "HealthCheck",
]


def __getattr__(name):
    """Lazily export the test-support helper so `import geo_infer_ops`
    stays clean-install safe: core.testing pulls fastapi.testclient,
    which requires httpx — a dev-only dependency."""
    if name == "setup_testing":
        from geo_infer_ops.core import testing

        return testing.setup_testing
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
