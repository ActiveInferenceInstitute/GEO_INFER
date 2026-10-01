#!/usr/bin/env python3

"""
Agent Registry for GEO-INFER-AGENT.

This module manages the creation, retrieval, and lifecycle of agent instances.
It acts as a central registry for all agents in the system.
"""

import uuid
import logging
import asyncio
import importlib
from typing import Any, Optional, cast

from geo_infer_agent.core.agent_base import BaseAgent

logger = logging.getLogger("geo_infer_agent.core.agent_registry")

# Single source of truth for the agent-type -> class mapping and its
# human-readable descriptions.  AgentRegistry and the CLI both derive from
# these constants; adding an agent type means editing only this block.
AGENT_TYPES: dict[str, str] = {
    "default": "geo_infer_agent.core.agent_base.ExampleAgent",
    "data_collector": "geo_infer_agent.agents.data_collector.DataCollectorAgent",
    "bdi": "geo_infer_agent.models.bdi.BDIAgent",
    "active_inference": "geo_infer_agent.models.active_inference.ActiveInferenceAgent",
    "reinforcement_learning": "geo_infer_agent.models.rl.RLAgent",
    "rule_based": "geo_infer_agent.models.rule_based.RuleBasedAgent",
    "hybrid": "geo_infer_agent.models.hybrid.HybridAgent",
}

AGENT_DESCRIPTIONS: dict[str, str] = {
    "default": "Basic example agent for testing",
    "data_collector": "Collects data from configured sources",
    "bdi": "Belief-Desire-Intention cognitive architecture",
    "active_inference": "Free-energy-minimising agent (matrix model)",
    "reinforcement_learning": "Q-learning agent with replay buffer",
    "rule_based": "Decision-tree agents for simple spatial tasks",
    "hybrid": "Combines rule-based and learning architectures",
}


class AgentRegistry:
    """
    Registry for managing agent instances.

    This class handles:
    - Agent creation and registration
    - Agent lookup and retrieval
    - Agent lifecycle management (start/stop)
    """

    _instance: Optional["AgentRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "AgentRegistry":
        """Singleton pattern to ensure a single registry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the agent registry."""
        if self._initialized:
            return

        # Dictionary of agent instances keyed by ID
        self.agents: dict[str, BaseAgent] = {}

        # Set of running agent IDs
        self.running_agents: set[str] = set()

        # Map of agent types to class paths (from the module-level constants)
        self.agent_types = dict(AGENT_TYPES)

        # Running tasks for agents
        self.agent_tasks: dict[str, asyncio.Task] = {}

        self._initialized = True
        logger.info("Agent registry initialized")

    def _load_agent_class(self, agent_type: str) -> type:
        """
        Dynamically load agent class based on type.

        Args:
            agent_type: Type of agent to load

        Returns:
            Agent class

        Raises:
            ValueError: If the agent type is unknown
            ImportError: If the agent module cannot be loaded
        """
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")

        try:
            # Split module path and class name
            module_path, class_name = self.agent_types[agent_type].rsplit(".", 1)

            # Import module
            module = importlib.import_module(module_path)

            # Get class
            agent_class = getattr(module, class_name)

            return cast(type, agent_class)
        except (ImportError, AttributeError) as e:
            logger.error(
                f"Failed to load agent class for type '{agent_type}': {str(e)}"
            )
            raise ImportError(f"Failed to load agent type {agent_type}: {str(e)}")

    async def create_agent(
        self,
        agent_type: str,
        config: dict[str, Any],
        agent_id: str | None = None,
        region: str | None = None,
    ) -> str:
        """
        Create a new agent instance.

        Args:
            agent_type: Type of agent to create
            config: Agent configuration
            agent_id: Custom ID for the agent (auto-generated if None)
            region: Geospatial region for agent operation (GeoJSON)

        Returns:
            ID of the created agent

        Raises:
            ValueError: If the agent type is unknown
            ImportError: If the agent module cannot be loaded
        """
        # Generate agent ID if not provided
        if agent_id is None:
            agent_id = str(uuid.uuid4())

        # Check if ID already exists
        if agent_id in self.agents:
            raise ValueError(f"Agent with ID {agent_id} already exists")

        # Load agent class
        agent_class = self._load_agent_class(agent_type)

        # Update config with region if provided
        if region:
            config = config.copy()
            config["region"] = region

        # Create agent instance
        agent = agent_class(agent_id=agent_id, config=config)

        # Register agent
        self.agents[agent_id] = agent

        logger.info(f"Created agent {agent_id} of type {agent_type}")
        return agent_id

    async def start_agent(self, agent_id: str) -> None:
        """
        Start an agent.

        Args:
            agent_id: ID of the agent to start

        Raises:
            KeyError: If the agent does not exist
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        if agent_id in self.running_agents:
            logger.warning(f"Agent {agent_id} is already running")
            return

        agent = self.agents[agent_id]

        # Create task for agent
        task = asyncio.create_task(agent.run())

        # Store task
        self.agent_tasks[agent_id] = task

        # Mark as running
        self.running_agents.add(agent_id)

        logger.info(f"Started agent {agent_id}")

    async def stop_agent(self, agent_id: str) -> None:
        """
        Stop an agent.

        Args:
            agent_id: ID of the agent to stop

        Raises:
            KeyError: If the agent does not exist
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        if agent_id not in self.running_agents:
            logger.warning(f"Agent {agent_id} is not running")
            return

        agent = self.agents[agent_id]

        # Stop the agent
        agent.stop()

        # Cancel and wait for the task
        if agent_id in self.agent_tasks:
            task = self.agent_tasks[agent_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

            # Clean up
            del self.agent_tasks[agent_id]

        # Mark as not running
        self.running_agents.remove(agent_id)

        logger.info(f"Stopped agent {agent_id}")

    def remove_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the registry.

        Args:
            agent_id: ID of the agent to remove

        Raises:
            KeyError: If the agent does not exist
            RuntimeError: If the agent is still running
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        if agent_id in self.running_agents:
            raise RuntimeError(
                f"Cannot remove running agent {agent_id}. Stop the agent first."
            )

        # Remove agent
        del self.agents[agent_id]

        logger.info(f"Removed agent {agent_id} from registry")

    def get_agent(self, agent_id: str) -> BaseAgent:
        """
        Get an agent instance.

        Args:
            agent_id: ID of the agent

        Returns:
            Agent instance

        Raises:
            KeyError: If the agent does not exist
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        return self.agents[agent_id]

    def get_agent_info(self, agent_id: str) -> dict[str, Any]:
        """
        Get information about an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Dictionary with agent information

        Raises:
            KeyError: If the agent does not exist
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]

        return {
            "agent_id": agent.agent_id,
            "agent_type": agent.__class__.__name__,
            "is_running": agent_id in self.running_agents,
            "created_at": agent.state.creation_time.isoformat(),
            "config": agent.config,
        }

    async def get_agent_state(self, agent_id: str) -> dict[str, Any]:
        """
        Get the current state of an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Dictionary with agent state

        Raises:
            KeyError: If the agent does not exist
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]

        state_dict = {
            "agent_id": agent.agent_id,
            "is_running": agent_id in self.running_agents,
            "state": agent.state.to_dict() if hasattr(agent, "state") else {},
            "last_perception": agent.last_perception
            if hasattr(agent, "last_perception")
            else None,
            "last_action": agent.last_action if hasattr(agent, "last_action") else None,
            "messages": agent.messages if hasattr(agent, "messages") else [],
        }

        return state_dict

    def is_agent_running(self, agent_id: str) -> bool:
        """
        Check if an agent is running.

        Args:
            agent_id: ID of the agent

        Returns:
            True if the agent is running

        Raises:
            KeyError: If the agent does not exist
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        return agent_id in self.running_agents

    def list_agents(self) -> list[dict[str, Any]]:
        """
        List all registered agents.

        Returns:
            List of agent information dictionaries
        """
        return [self.get_agent_info(agent_id) for agent_id in self.agents]

    def list_agent_types(self) -> dict[str, str]:
        """
        List all available agent types.

        Returns:
            Dictionary mapping agent type names to class paths
        """
        return self.agent_types.copy()

    async def agent_action(
        self, agent_id: str, action: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Perform an action on an agent.

        Args:
            agent_id: ID of the agent
            action: Action to perform
            parameters: Action parameters

        Returns:
            Result of the action

        Raises:
            KeyError: If the agent does not exist
            ValueError: If the action is invalid
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found")

        agent = self.agents[agent_id]

        # Check if agent has the action method
        action_method = f"action_{action}"
        if not hasattr(agent, action_method):
            raise ValueError(f"Agent {agent_id} does not support action '{action}'")

        # Call the action method
        method = getattr(agent, action_method)
        result = await method(**parameters)

        logger.debug(f"Performed action '{action}' on agent {agent_id}")
        return cast(dict[str, Any], result)

    async def send_message(
        self, from_agent_id: str, to_agent_id: str, content: dict[str, Any]
    ) -> bool:
        """
        Send a message from one agent to another.

        Args:
            from_agent_id: ID of the sending agent
            to_agent_id: ID of the receiving agent
            content: Message content

        Returns:
            True if the message was sent successfully

        Raises:
            KeyError: If either agent does not exist
        """
        if from_agent_id not in self.agents:
            raise KeyError(f"Sending agent {from_agent_id} not found")

        if to_agent_id not in self.agents:
            raise KeyError(f"Receiving agent {to_agent_id} not found")

        from_agent = self.agents[from_agent_id]

        # Send the message
        success = await from_agent.send_message(to_agent_id, content)

        logger.debug(f"Message sent from agent {from_agent_id} to agent {to_agent_id}")
        return success


# Global instance
agent_registry = AgentRegistry()
