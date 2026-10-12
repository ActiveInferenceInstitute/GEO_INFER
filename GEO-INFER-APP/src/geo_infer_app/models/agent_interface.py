"""
Agent Interface Module

Provides the core interfaces for integrating GEO-INFER-AGENT models with the GEO-INFER-APP UI.
This module serves as the bridge between agent implementations and UI components.
"""

from typing import Any
from collections.abc import Callable
from enum import Enum
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enumeration of supported agent types matching GEO-INFER-AGENT implementations."""

    BDI = "bdi"
    ACTIVE_INFERENCE = "active_inference"
    RL = "reinforcement_learning"
    RULE_BASED = "rule_based"
    HYBRID = "hybrid"


@dataclass
class AgentState:
    """Represents the current state of an agent for UI representation."""

    agent_id: str
    agent_type: AgentType
    status: str
    location: dict[str, float] | None = None  # e.g., {"lat": 40.7, "lng": -74.0}
    tasks: list[dict[str, Any]] | None = None
    beliefs: dict[str, Any] | None = None
    goals: list[str] | None = None
    last_updated: str | None = None
    metadata: dict[str, Any] | None = None


class AgentInterface(ABC):
    """
    Abstract base class for all agent interfaces in the application.

    This class defines the contract that all agent interfaces must implement
    to be compatible with the GEO-INFER-APP UI components.
    """

    @abstractmethod
    def get_agent_state(self, agent_id: str) -> AgentState:
        """
        Retrieve the current state of the specified agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            AgentState object containing the current state
        """
        ...

    @abstractmethod
    def list_agents(
        self, filter_params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        List all available agents, with optional filtering.

        Args:
            filter_params: Optional dictionary of filter parameters

        Returns:
            List of agent summary dictionaries
        """
        ...

    @abstractmethod
    def send_command(
        self, agent_id: str, command: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Send a command to an agent.

        Args:
            agent_id: Unique identifier for the agent
            command: Command to send
            params: Parameters for the command

        Returns:
            Response from the agent
        """
        ...

    @abstractmethod
    def register_event_handler(
        self, event_type: str, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """
        Register a callback function to handle agent events.

        Args:
            event_type: Type of event to handle
            callback: Function to call when the event occurs
        """
        ...

    @abstractmethod
    def create_agent(self, agent_type: AgentType, config: dict[str, Any]) -> str:
        """
        Create a new agent instance.

        Args:
            agent_type: Type of agent to create
            config: Configuration parameters for the agent

        Returns:
            ID of the created agent
        """
        ...
