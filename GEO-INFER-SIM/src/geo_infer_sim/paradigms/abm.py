"""
Agent-Based Modeling (ABM) framework for GEO-INFER-SIM.

This module provides agent-based modeling capabilities for simulating
systems as collections of autonomous, interacting agents.
"""

import logging
from typing import Any
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Agent:
    """
    Base agent class for agent-based models.

    Represents an autonomous entity in the simulation with properties,
    behaviors, and spatial location.
    """

    agent_id: str
    position: np.ndarray  # Spatial position [x, y] or [lon, lat]
    properties: dict[str, Any] = field(default_factory=dict)
    state: str = "active"
    neighbors: list[str] = field(default_factory=list)

    def step(self, time: float, environment: dict[str, Any]) -> None:
        """
        Execute one step of agent behavior.

        Args:
            time: Current simulation time
            environment: Environment state
        """
        self.properties["last_step_time"] = time
        self.properties["last_environment_keys"] = sorted(environment.keys())

    def interact(self, other_agent: "Agent", time: float) -> None:
        """
        Interact with another agent.

        Args:
            other_agent: Other agent to interact with
            time: Current simulation time
        """
        if other_agent.agent_id not in self.neighbors:
            self.neighbors.append(other_agent.agent_id)
        self.properties["last_interaction_time"] = time
        self.properties["interaction_count"] = (
            self.properties.get("interaction_count", 0) + 1
        )


class AgentBasedModel:
    """
    Agent-Based Model for geospatial simulations.

    Manages a population of agents and their interactions within a
    spatial environment.
    """

    def __init__(
        self,
        environment: dict[str, Any] | None = None,
        spatial_bounds: np.ndarray | None = None,
        neighbor_radius: float = 10.0,
    ) -> None:
        """
        Initialize the agent-based model.

        Args:
            environment: Initial environment state
            spatial_bounds: Spatial bounds [[min_x, min_y], [max_x, max_y]];
                advisory metadata only — positions are not clamped
            neighbor_radius: Default search radius used by step() when
                updating agent neighbor lists
        """
        self.agents: dict[str, Agent] = {}
        self.environment = environment or {}
        self.spatial_bounds = spatial_bounds
        self.neighbor_radius = neighbor_radius
        self.time = 0.0

    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the model.

        Args:
            agent: Agent to add
        """
        self.agents[agent.agent_id] = agent
        logger.debug(f"Added agent: {agent.agent_id}")

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the model.

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent was removed, False if not found
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.debug(f"Removed agent: {agent_id}")
            return True
        return False

    def get_agent(self, agent_id: str) -> Agent | None:
        """
        Get an agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent object or None if not found
        """
        return self.agents.get(agent_id)

    def find_neighbors(
        self, agent: Agent, radius: float, max_neighbors: int | None = None
    ) -> list[Agent]:
        """
        Find neighboring agents within a radius.

        Args:
            agent: Agent to find neighbors for
            radius: Search radius
            max_neighbors: Maximum number of neighbors to return

        Returns:
            List of neighboring agents
        """
        neighbors = []

        for other_agent in self.agents.values():
            if other_agent.agent_id == agent.agent_id:
                continue

            distance = np.linalg.norm(agent.position - other_agent.position)

            if distance <= radius:
                neighbors.append(other_agent)

        # Sort by distance and limit
        neighbors.sort(key=lambda a: np.linalg.norm(agent.position - a.position))

        if max_neighbors:
            neighbors = neighbors[:max_neighbors]

        return neighbors

    def step(self, time_step: float) -> None:
        """
        Execute one simulation step for all agents.

        Args:
            time_step: Time step duration
        """
        # Update agent neighbors
        for agent in self.agents.values():
            agent.neighbors = [
                n.agent_id
                for n in self.find_neighbors(agent, radius=self.neighbor_radius)
            ]

        # Execute agent steps
        for agent in list(self.agents.values()):
            if agent.state == "active":
                try:
                    agent.step(self.time, self.environment)
                except Exception as e:
                    logger.error(f"Agent {agent.agent_id} step failed: {e}")

        # Execute agent interactions
        agent_list = list(self.agents.values())
        for i, agent1 in enumerate(agent_list):
            for agent2 in agent_list[i + 1 :]:
                if agent1.state == "active" and agent2.state == "active":
                    try:
                        agent1.interact(agent2, self.time)
                    except Exception as e:
                        logger.error(
                            f"Interaction between {agent1.agent_id} and "
                            f"{agent2.agent_id} failed: {e}"
                        )

        # Update time
        self.time += time_step

    def get_state(self) -> dict[str, Any]:
        """
        Get current model state.

        Returns:
            Model state dictionary
        """
        return {
            "time": self.time,
            "num_agents": len(self.agents),
            "agents": {
                agent_id: {
                    "position": agent.position.tolist(),
                    "state": agent.state,
                    "properties": agent.properties,
                }
                for agent_id, agent in self.agents.items()
            },
            "environment": self.environment,
        }
