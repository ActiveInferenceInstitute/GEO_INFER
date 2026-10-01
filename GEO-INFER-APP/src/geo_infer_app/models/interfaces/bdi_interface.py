"""
BDI Agent Interface Implementation

Implements the AgentInterface for Belief-Desire-Intention (BDI) agents.
This interface connects the GEO-INFER-APP UI to the BDI agent implementation
in GEO-INFER-AGENT.
"""

from typing import Any
from collections.abc import Callable
import logging
import datetime
import uuid

try:
    # Import BDI agent implementation from GEO-INFER-AGENT
    from geo_infer_agent.models.bdi import (
        BDIAgent,
        BeliefBase,
        DesireSet,
        IntentionStructure,
    )
except ImportError:
    # Deterministic local implementation if GEO-INFER-AGENT is not available
    class BDIAgent:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self.id = str(uuid.uuid4())
            self.name = kwargs.get("name", f"BDI-Agent-{self.id[:8]}")
            self.beliefs: dict[str, Any] = kwargs.get("beliefs", {})
            self.desires: list[Any] = kwargs.get("desires", [])
            self.intentions: list[Any] = []
            self.location = kwargs.get("initial_location", {"lat": 0.0, "lng": 0.0})
            self.status = "idle"

        def update_beliefs(self, beliefs: dict[str, Any]) -> None:
            self.beliefs.update(beliefs)

        def add_desire(self, desire: Any) -> None:
            self.desires.append(desire)

        def deliberate(self) -> None:
            # Simple deterministic local implementation
            for desire in self.desires:
                if desire not in self.intentions:
                    self.intentions.append(desire)

        def execute(self) -> None:
            # Simple deterministic local implementation
            if self.intentions:
                self.status = "executing"
                # Simulate execution by removing the first intention
                if self.intentions:
                    self.intentions.pop(0)
            else:
                self.status = "idle"

    class BeliefBase(dict):  # type: ignore[no-redef]
        """Minimal belief container used when GEO-INFER-AGENT is unavailable."""

    class DesireSet(list):  # type: ignore[no-redef]
        """Minimal desire container used when GEO-INFER-AGENT is unavailable."""

    class IntentionStructure(list):  # type: ignore[no-redef]
        """Minimal intention container used when GEO-INFER-AGENT is unavailable."""


from geo_infer_app.models.agent_interface import AgentInterface, AgentState, AgentType

# Configure logging
logger = logging.getLogger(__name__)


class BDIAgentInterface(AgentInterface):
    """
    Implementation of AgentInterface for BDI agents.

    This class provides the bridge between the UI components in GEO-INFER-APP
    and the BDI agent implementation in GEO-INFER-AGENT.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the BDI agent interface.

        Args:
            **kwargs: Configuration parameters for the agent interface
        """
        self._agents: dict[str, BDIAgent] = {}
        self._event_handlers: dict[str, list[Callable]] = {}
        self._config = kwargs

        logger.info("Initialized BDI agent interface")

    def get_agent_state(self, agent_id: str) -> AgentState:
        """
        Retrieve the current state of the specified agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            AgentState object containing the current state

        Raises:
            ValueError: If the agent with the specified ID does not exist
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent with ID {agent_id} not found")

        agent = self._agents[agent_id]

        # Convert BDI agent state to AgentState
        return AgentState(
            agent_id=agent_id,
            agent_type=AgentType.BDI,
            status=agent.status,
            location=agent.location,
            tasks=[],  # BDI agents don't have tasks in the same way
            beliefs=agent.beliefs,
            goals=agent.desires,  # In BDI, desires are similar to goals
            last_updated=datetime.datetime.now().isoformat(),
            metadata={"intentions": agent.intentions},
        )

    def list_agents(
        self, filter_params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        List all available BDI agents, with optional filtering.

        Args:
            filter_params: Optional dictionary of filter parameters

        Returns:
            List of agent summary dictionaries
        """
        agents = []

        for agent_id, agent in self._agents.items():
            # Skip agents that don't match the filter
            if filter_params:
                if (
                    "status" in filter_params
                    and agent.status != filter_params["status"]
                ):
                    continue

                if "location" in filter_params:
                    # Filter by location within a radius
                    agent_loc = agent.location
                    filter_loc = filter_params["location"]

                    if not self._is_location_in_radius(
                        agent_loc, filter_loc.get("center"), filter_loc.get("radius", 0)
                    ):
                        continue

            # Add agent summary
            agents.append(
                {
                    "id": agent_id,
                    "name": agent.name,
                    "type": "bdi",
                    "status": agent.status,
                    "location": agent.location,
                    "num_beliefs": len(agent.beliefs),
                    "num_desires": len(agent.desires),
                    "num_intentions": len(agent.intentions),
                }
            )

        return agents

    def send_command(
        self, agent_id: str, command: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Send a command to a BDI agent.

        Args:
            agent_id: Unique identifier for the agent
            command: Command to send
            params: Parameters for the command

        Returns:
            Response from the agent

        Raises:
            ValueError: If the agent with the specified ID does not exist
            ValueError: If the command is not supported
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent with ID {agent_id} not found")

        agent = self._agents[agent_id]
        response = {"success": False, "message": "Unknown command"}

        if command == "add_belief":
            if "belief" in params:
                agent.beliefs.update(params["belief"])
                response = {"success": True, "message": "Belief added"}
        elif command == "add_desire":
            if "desire" in params:
                agent.desires.append(params["desire"])
                response = {"success": True, "message": "Desire added"}
        elif command == "deliberate":
            for desire in agent.desires:
                if desire not in agent.intentions:
                    agent.intentions.append(desire)
            response = {"success": True, "message": "Agent deliberated"}
        elif command == "execute":
            agent.status = "executing" if agent.intentions else "idle"
            if agent.intentions:
                agent.intentions.pop(0)
            response = {"success": True, "message": "Agent executed intentions"}
        elif command == "move":
            if "location" in params:
                agent.location = params["location"]
                response = {"success": True, "message": "Agent moved"}
        else:
            raise ValueError(f"Unsupported command: {command}")

        # Trigger event handlers for agent update
        self._trigger_event(
            "agent_updated",
            {"agent_id": agent_id, "state": self.get_agent_state(agent_id)},
        )

        return response

    def register_event_handler(
        self, event_type: str, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        """
        Register a callback function to handle agent events.

        Args:
            event_type: Type of event to handle
            callback: Function to call when the event occurs
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []

        self._event_handlers[event_type].append(callback)
        logger.info(f"Registered event handler for {event_type}")

    def create_agent(self, agent_type: AgentType, config: dict[str, Any]) -> str:
        """
        Create a new BDI agent instance.

        Args:
            agent_type: Type of agent to create (must be AgentType.BDI)
            config: Configuration parameters for the agent

        Returns:
            ID of the created agent

        Raises:
            ValueError: If agent_type is not AgentType.BDI
        """
        if agent_type != AgentType.BDI:
            raise ValueError(
                f"This interface only supports BDI agents, got {agent_type.value}"
            )

        # Create a new BDI agent. GEO-INFER-AGENT's canonical BDIAgent takes
        # (agent_id, config), while older lightweight test doubles accepted
        # flat keyword arguments. Normalize both surfaces to the UI contract.
        agent_id = config.get("agent_id") or str(uuid.uuid4())
        bdi_config = self._to_bdi_config(config)
        try:
            agent = BDIAgent(agent_id=agent_id, config=bdi_config)
        except TypeError:
            agent = BDIAgent(**config)
            agent_id = agent.id
        else:
            self._hydrate_interface_state(agent, agent_id, config)

        # Store the agent
        self._agents[agent_id] = agent

        # Trigger event handlers for agent creation
        self._trigger_event(
            "agent_created",
            {"agent_id": agent_id, "state": self.get_agent_state(agent_id)},
        )

        logger.info(f"Created BDI agent with ID {agent_id}")
        return agent_id

    @staticmethod
    def _to_bdi_config(config: dict[str, Any]) -> dict[str, Any]:
        """Map UI-oriented BDI config keys to GEO-INFER-AGENT BDI config."""
        beliefs = config.get("beliefs", {})
        desires = config.get("desires", [])
        return {
            **config,
            "initial_beliefs": beliefs,
            "initial_desires": [
                (
                    desire
                    if isinstance(desire, dict)
                    else {
                        "name": str(desire),
                        "description": str(desire),
                        "priority": 0.5,
                    }
                )
                for desire in desires
            ],
        }

    @staticmethod
    def _hydrate_interface_state(
        agent: BDIAgent, agent_id: str, config: dict[str, Any]
    ) -> None:
        """Attach UI-facing state fields expected by the APP interface."""
        agent.id = agent_id
        agent.name = config.get("name", f"BDI-Agent-{agent_id[:8]}")
        agent.beliefs = dict(config.get("beliefs", {}))
        agent.desires = list(config.get("desires", []))
        agent.intentions = []
        agent.location = config.get("initial_location", {"lat": 0.0, "lng": 0.0})
        agent.status = "idle"

    def _trigger_event(self, event_type: str, event_data: dict[str, Any]) -> None:
        """
        Trigger all registered handlers for an event.

        Args:
            event_type: Type of event
            event_data: Data for the event
        """
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")

    @staticmethod
    def _is_location_in_radius(
        location: dict[str, float] | None,
        center: dict[str, float] | None,
        radius: float,
    ) -> bool:
        """
        Check if a location is within a radius of a center point using haversine.

        Args:
            location: Location to check {"lat": ..., "lng": ...}
            center: Center location {"lat": ..., "lng": ...}
            radius: Radius in kilometers

        Returns:
            True if the location is within the radius, False otherwise
        """
        import math

        if not location or not center or radius <= 0:
            return False

        # Haversine formula for great-circle distance between two lat/lng points
        r_earth = 6371.0  # Earth radius in kilometres
        lat1 = math.radians(location["lat"])
        lat2 = math.radians(center["lat"])
        dlat = math.radians(center["lat"] - location["lat"])
        dlng = math.radians(center["lng"] - location["lng"])

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
        )
        distance = 2 * r_earth * math.asin(math.sqrt(a))

        return distance <= radius


# Register this interface with the agent factory
from geo_infer_app.models.agent_factory import AgentFactory  # noqa: E402

AgentFactory.register_interface(AgentType.BDI, BDIAgentInterface)
