"""
GEO-INFER-APP

Human-computer interaction layer providing accessible geospatial applications,
dashboards, and UI components built on Active Inference principles.

This package bridges agent implementations in GEO-INFER-AGENT with user-facing
interfaces: configuration forms, visualisation widgets, and an async agent API.
"""

from geo_infer_app.models.agent_interface import AgentInterface, AgentState, AgentType
from geo_infer_app.models.agent_factory import AgentFactory
from geo_infer_app.models.agent_visualization import AgentVisualization
from geo_infer_app.models.agent_configuration import AgentConfiguration
from geo_infer_app.api.agent_api import AgentAPIClient, AgentManager

__version__ = "0.4.0"

__all__ = [
    "AgentInterface",
    "AgentState",
    "AgentType",
    "AgentFactory",
    "AgentVisualization",
    "AgentConfiguration",
    "AgentAPIClient",
    "AgentManager",
    "__version__",
]
