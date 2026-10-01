"""
Agent Visualization Module

Provides components and utilities for visualizing agents and their states
in geospatial contexts within the GEO-INFER-APP.
"""

from typing import Any
import json
import logging
import math
from enum import Enum
from dataclasses import dataclass
from .agent_interface import AgentState, AgentType

# Configure logging
logger = logging.getLogger(__name__)


class VisualizationType(Enum):
    """Enumeration of supported visualization types for agents."""

    MAP_MARKER = "map_marker"
    AGENT_DASHBOARD = "agent_dashboard"
    NETWORK_NODE = "network_node"
    TIMELINE_EVENT = "timeline_event"
    STATE_DIAGRAM = "state_diagram"


@dataclass
class VisualizationConfig:
    """Configuration for agent visualization."""

    vis_type: VisualizationType
    title: str
    description: str | None = None
    icon: str | None = None  # Icon identifier or URL
    color: str | None = None  # CSS color value
    scale: float = 1.0
    show_label: bool = True
    custom_props: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate frontend-facing visual configuration."""
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")


def _json_safe(value: Any) -> Any:
    """Convert metadata into values that can be sent as GeoJSON/JSON."""
    return json.loads(json.dumps(value, default=str))


class AgentVisualization:
    """
    Class for converting agent states into visualization-friendly formats.

    This class handles:
    1. Converting agent states to visualization data
    2. Generating visualization configurations for different agent types
    3. Customizing visualizations based on agent properties
    """

    @staticmethod
    def get_default_config(agent_type: AgentType) -> dict[str, VisualizationConfig]:
        """
        Get default visualization configurations for the specified agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Dictionary mapping visualization context to configuration
        """
        configs = {
            "map": VisualizationConfig(
                vis_type=VisualizationType.MAP_MARKER,
                title=f"{agent_type.name} Agent",
                icon="agent",
                color="#3498db",
                show_label=True,
            ),
            "dashboard": VisualizationConfig(
                vis_type=VisualizationType.AGENT_DASHBOARD,
                title=f"{agent_type.name} Dashboard",
                description=f"Dashboard for {agent_type.name} agent type",
                icon="dashboard",
                custom_props={"widgets": ["status", "tasks", "beliefs", "goals"]},
            ),
            "network": VisualizationConfig(
                vis_type=VisualizationType.NETWORK_NODE,
                title=f"{agent_type.name}",
                icon="node",
                color="#2ecc71",
            ),
        }

        # Customize based on agent type
        if agent_type == AgentType.BDI:
            configs["map"].color = "#e74c3c"
            dash_props = configs["dashboard"].custom_props
            if dash_props is not None and "widgets" in dash_props:
                dash_props["widgets"].append("intentions")
        elif agent_type == AgentType.ACTIVE_INFERENCE:
            configs["map"].color = "#f39c12"
            dash_props = configs["dashboard"].custom_props
            if dash_props is not None and "widgets" in dash_props:
                dash_props["widgets"].append("predictions")
        elif agent_type == AgentType.RL:
            configs["map"].color = "#9b59b6"
            dash_props = configs["dashboard"].custom_props
            if dash_props is not None and "widgets" in dash_props:
                dash_props["widgets"].append("rewards")

        return configs

    @staticmethod
    def state_to_map_feature(agent_state: AgentState) -> dict[str, Any]:
        """
        Convert an agent state to a map feature representation.

        Args:
            agent_state: Current state of the agent

        Returns:
            Map feature representation for the agent

        Raises:
            ValueError: If the agent state doesn't include location
        """
        if not agent_state.location:
            raise ValueError(f"Agent {agent_state.agent_id} has no location data")
        try:
            longitude = float(agent_state.location["lng"])
            latitude = float(agent_state.location["lat"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Agent {agent_state.agent_id} location must contain numeric lat/lng values"
            ) from exc
        if not math.isfinite(longitude) or not math.isfinite(latitude):
            raise ValueError(f"Agent {agent_state.agent_id} location must be finite")
        if not -180 <= longitude <= 180:
            raise ValueError(
                f"Agent {agent_state.agent_id} longitude must be between -180 and 180"
            )
        if not -90 <= latitude <= 90:
            raise ValueError(
                f"Agent {agent_state.agent_id} latitude must be between -90 and 90"
            )

        configs = AgentVisualization.get_default_config(agent_state.agent_type)
        map_config = configs["map"]

        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            "properties": {
                "id": agent_state.agent_id,
                "type": agent_state.agent_type.value,
                "status": agent_state.status,
                "title": map_config.title,
                "icon": map_config.icon,
                "color": map_config.color,
                "scale": map_config.scale,
                "showLabel": map_config.show_label,
                "lastUpdated": agent_state.last_updated,
                "metadata": _json_safe(agent_state.metadata or {}),
            },
        }

        return feature

    @staticmethod
    def state_to_dashboard_data(agent_state: AgentState) -> dict[str, Any]:
        """
        Convert an agent state to dashboard data.

        Args:
            agent_state: Current state of the agent

        Returns:
            Dashboard data representation for the agent
        """
        configs = AgentVisualization.get_default_config(agent_state.agent_type)
        dash_config = configs["dashboard"]
        custom_props = dash_config.custom_props or {}
        widgets_list = custom_props.get("widgets", [])

        widgets_data: dict[str, Any] = {}
        dashboard_data: dict[str, Any] = {
            "id": agent_state.agent_id,
            "type": agent_state.agent_type.value,
            "title": dash_config.title,
            "status": agent_state.status,
            "lastUpdated": agent_state.last_updated,
            "widgets": widgets_data,
        }

        # Add data for each configured widget
        if "status" in widgets_list:
            widgets_data["status"] = {
                "title": "Status",
                "value": agent_state.status,
            }

        if "tasks" in widgets_list and agent_state.tasks:
            widgets_data["tasks"] = {
                "title": "Tasks",
                "value": agent_state.tasks,
            }

        if "beliefs" in widgets_list and agent_state.beliefs:
            widgets_data["beliefs"] = {
                "title": "Beliefs",
                "value": agent_state.beliefs,
            }

        if "goals" in widgets_list and agent_state.goals:
            widgets_data["goals"] = {
                "title": "Goals",
                "value": agent_state.goals,
            }

        # Agent type specific widgets
        if agent_state.agent_type == AgentType.BDI and "intentions" in widgets_list:
            if agent_state.metadata and "intentions" in agent_state.metadata:
                widgets_data["intentions"] = {
                    "title": "Intentions",
                    "value": agent_state.metadata["intentions"],
                }

        if (
            agent_state.agent_type == AgentType.ACTIVE_INFERENCE
            and "predictions" in widgets_list
        ):
            if agent_state.metadata and "predictions" in agent_state.metadata:
                widgets_data["predictions"] = {
                    "title": "Predictions",
                    "value": _json_safe(agent_state.metadata["predictions"]),
                }

        if agent_state.agent_type == AgentType.RL and "rewards" in widgets_list:
            if agent_state.metadata and "rewards" in agent_state.metadata:
                widgets_data["rewards"] = {
                    "title": "Rewards",
                    "value": _json_safe(agent_state.metadata["rewards"]),
                }

        return dashboard_data
