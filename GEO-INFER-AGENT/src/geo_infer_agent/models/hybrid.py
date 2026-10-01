#!/usr/bin/env python3

"""
Hybrid Agent Architecture.

This module implements a hybrid agent architecture that combines multiple
agent models (BDI, Active Inference, RL, Rule-based) to leverage
the strengths of each approach within a unified framework.
"""

import os
import logging
import asyncio
from typing import Any, cast
from datetime import datetime
import json

from geo_infer_agent.core.agent_base import BaseAgent, AgentState

# BDIState and BDIAgent are provided by the bdi package.
# Import from models package which handles the import correctly
from geo_infer_agent.models import BDIAgent
from geo_infer_agent.models.active_inference import ActiveInferenceAgent
from geo_infer_agent.models.rl import RLAgent
from geo_infer_agent.models.rule_based import RuleBasedAgent

logger = logging.getLogger("geo_infer_agent.models.hybrid")


class SubAgentWrapper:
    """
    Wrapper for a sub-agent within a hybrid architecture.

    Tracks the sub-agent and its configuration, activation conditions,
    and priority within the hybrid system.
    """

    def __init__(
        self,
        agent_type: str,
        agent: BaseAgent,
        priority: int = 0,
        activation_conditions: dict[str, Any] | None = None,
        description: str = "",
    ):
        """
        Initialize a sub-agent wrapper.

        Args:
            agent_type: Type of agent (e.g., 'bdi', 'rl', 'rule_based')
            agent: The agent instance
            priority: Priority level for conflict resolution
            activation_conditions: Conditions that determine when this agent is active
            description: Human-readable description
        """
        self.agent_type = agent_type
        self.agent = agent
        self.priority = priority
        self.activation_conditions = activation_conditions or {}
        self.description = description

        # Activation state
        self.is_active = True

        # Performance statistics
        self.decision_count = 0
        self.successful_decision_count = 0
        self.last_activated: datetime | None = None
        self.last_reward = 0.0
        self.total_reward = 0.0

    def check_activation(self, context: dict[str, Any]) -> bool:
        """
        Check if this sub-agent should be activated.

        Args:
            context: Current context (facts, observations, etc.)

        Returns:
            True if agent should be activated, False otherwise
        """
        # If no activation conditions, always active
        if not self.activation_conditions:
            return True

        # Check conditions
        for key, expected_value in self.activation_conditions.items():
            if key not in context:
                return False

            actual_value = context[key]

            # Check value match
            if isinstance(expected_value, dict) and isinstance(actual_value, dict):
                # Recursive check for nested dictionaries
                if not self._nested_dict_matches(expected_value, actual_value):
                    return False
            elif expected_value != actual_value:
                return False

        return True

    def _nested_dict_matches(self, expected: dict, actual: dict) -> bool:
        """
        Check if nested dictionary matches.

        Args:
            expected: Expected dictionary values
            actual: Actual dictionary values

        Returns:
            True if all expected key-value pairs match actual dict
        """
        for key, expected_value in expected.items():
            if key not in actual:
                return False

            actual_value = actual[key]

            if isinstance(expected_value, dict) and isinstance(actual_value, dict):
                # Recursive check
                if not self._nested_dict_matches(expected_value, actual_value):
                    return False
            elif expected_value != actual_value:
                return False

        return True

    def record_decision(self, successful: bool, reward: float = 0.0) -> None:
        """
        Record a decision made by this agent.

        Args:
            successful: Whether the decision was successful
            reward: Reward received for the decision
        """
        self.decision_count += 1
        if successful:
            self.successful_decision_count += 1

        self.last_activated = datetime.now()
        self.last_reward = reward
        self.total_reward += reward

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "agent_type": self.agent_type,
            "agent_id": self.agent.agent_id,
            "priority": self.priority,
            "activation_conditions": self.activation_conditions,
            "description": self.description,
            "is_active": self.is_active,
            "stats": {
                "decision_count": self.decision_count,
                "successful_decision_count": self.successful_decision_count,
                "last_activated": (
                    self.last_activated.isoformat() if self.last_activated else None
                ),
                "last_reward": self.last_reward,
                "total_reward": self.total_reward,
            },
        }


class HybridState(AgentState):
    """
    State for a hybrid agent.

    Maintains the context shared between sub-agents and tracks
    the performance and state of each sub-agent.
    """

    def __init__(self) -> None:
        """Initialize hybrid agent state."""
        super().__init__()

        # Shared context (observations, facts, beliefs, etc.)
        self.context: dict[str, Any] = {}

        # Sub-agent wrappers
        self.sub_agents: dict[str, SubAgentWrapper] = {}

        # Most recent decisions and actions
        self.last_perception: dict[str, Any] = {}
        self.last_decision: dict[str, Any] = {}
        self.last_action: dict[str, Any] = {}
        self.last_result: dict[str, Any] = {}

        # Decision history
        self.decision_history: list[dict[str, Any]] = []
        self.max_history_size = 100

        # Performance metrics
        self.total_decisions = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_reward = 0.0

    def add_sub_agent(self, wrapper: SubAgentWrapper) -> None:
        """
        Add a sub-agent.

        Args:
            wrapper: Sub-agent wrapper
        """
        self.sub_agents[wrapper.agent.agent_id] = wrapper

    def remove_sub_agent(self, agent_id: str) -> bool:
        """
        Remove a sub-agent.

        Args:
            agent_id: ID of agent to remove

        Returns:
            True if agent was removed, False if not found
        """
        if agent_id in self.sub_agents:
            del self.sub_agents[agent_id]
            return True
        return False

    def get_active_agents(self) -> list[SubAgentWrapper]:
        """
        Get all active sub-agents based on current context.

        Returns:
            List of active sub-agents sorted by priority
        """
        active_agents = []

        for wrapper in self.sub_agents.values():
            if wrapper.is_active and wrapper.check_activation(self.context):
                active_agents.append(wrapper)

        # Sort by priority (highest first)
        return sorted(active_agents, key=lambda w: w.priority, reverse=True)

    def update_context(self, key: str, value: Any) -> None:
        """
        Update a value in the shared context.

        Args:
            key: Context key
            value: Context value
        """
        self.context[key] = value

    def get_context_value(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the shared context.

        Args:
            key: Context key
            default: Default value if key not found

        Returns:
            Context value or default
        """
        return self.context.get(key, default)

    def record_decision(self, agent_id: str, decision: dict[str, Any]) -> None:
        """
        Record a decision made by a sub-agent.

        Args:
            agent_id: ID of agent that made the decision
            decision: Decision made
        """
        self.last_decision = {
            "agent_id": agent_id,
            "decision": decision,
            "timestamp": datetime.now().isoformat(),
        }

        self.decision_history.append(self.last_decision)
        self.total_decisions += 1

        # Trim history if needed
        while len(self.decision_history) > self.max_history_size:
            self.decision_history.pop(0)

    def record_result(
        self, result: dict[str, Any], success: bool, reward: float = 0.0
    ) -> None:
        """
        Record the result of an action.

        Args:
            result: Result of the action
            success: Whether the action was successful
            reward: Reward received for the action
        """
        self.last_result = {
            "result": result,
            "success": success,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
        }

        # Update performance metrics
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        self.total_reward += reward

        # Update sub-agent stats if decision was made by a sub-agent
        if "agent_id" in self.last_decision:
            agent_id = self.last_decision["agent_id"]
            if agent_id in self.sub_agents:
                self.sub_agents[agent_id].record_decision(success, reward)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "context": self.context,
            "sub_agents": {
                agent_id: wrapper.to_dict()
                for agent_id, wrapper in self.sub_agents.items()
            },
            "last_perception": self.last_perception,
            "last_decision": self.last_decision,
            "last_action": self.last_action,
            "last_result": self.last_result,
            "decision_history": self.decision_history,
            "max_history_size": self.max_history_size,
            "total_decisions": self.total_decisions,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_reward": self.total_reward,
        }


class HybridAgent(BaseAgent):
    """
    Implementation of a hybrid agent architecture.

    This agent:
    1. Combines multiple agent types into a unified system
    2. Selects the appropriate sub-agent for each task
    3. Coordinates information sharing between sub-agents
    4. Provides conflict resolution mechanisms
    """

    state: HybridState

    def __init__(self, agent_id: str | None = None, config: dict | None = None):
        """
        Initialize hybrid agent.

        Args:
            agent_id: Unique identifier for the agent
            config: Configuration parameters
        """
        super().__init__(agent_id=agent_id, config=config or {})

        # Configuration
        self.config = config or {}

        # Initialize state
        self.state = HybridState()

        # Configure based on config
        self._configure_state()

        # Register action handlers
        self._register_default_action_handlers()

        # Register perception handlers
        self._register_default_perception_handlers()

        # Sub-agent registry (agent_type -> agent_class)
        self._agent_registry: dict[str, Any] = {
            "bdi": BDIAgent,
            "active_inference": ActiveInferenceAgent,
            "rl": RLAgent,
            "rule_based": RuleBasedAgent,
        }

    def _configure_state(self) -> None:
        """Configure state from config."""
        # Set maximum history size
        if "max_history_size" in self.config:
            self.state.max_history_size = self.config["max_history_size"]

    async def initialize(self) -> None:
        """Initialize the agent."""
        logger.info(f"Initializing hybrid agent: {self.agent_id}")

        # Create and initialize sub-agents
        await self._initialize_sub_agents()

        # Load initial context
        self._load_initial_context()

        # Load saved state if available
        state_path = self.config.get("state_path")
        if state_path and os.path.exists(state_path):
            await self._load_state(state_path)

        logger.info("Hybrid agent %s initialization complete", self.agent_id)

    async def _initialize_sub_agents(self) -> None:
        """Create and initialize sub-agents."""
        sub_agents_config = self.config.get("sub_agents", [])

        for agent_config in sub_agents_config:
            try:
                # Extract required fields
                agent_type = agent_config["type"]
                agent_id = agent_config.get(
                    "id", f"{agent_type}_{len(self.state.sub_agents)}"
                )

                # Create and initialize sub-agent
                if agent_type in self._agent_registry:
                    # Create agent
                    agent_class = self._agent_registry[agent_type]
                    agent = agent_class(
                        agent_id=agent_id, config=agent_config.get("config", {})
                    )

                    # Initialize agent
                    await agent.initialize()

                    # Create wrapper
                    wrapper = SubAgentWrapper(
                        agent_type=agent_type,
                        agent=agent,
                        priority=agent_config.get("priority", 0),
                        activation_conditions=agent_config.get("activation_conditions"),
                        description=agent_config.get("description", ""),
                    )

                    # Add to state
                    self.state.add_sub_agent(wrapper)
                    logger.info(f"Initialized sub-agent: {agent_id} ({agent_type})")
                else:
                    logger.error(f"Unknown agent type: {agent_type}")
            except KeyError as e:
                logger.error(f"Missing required field in sub-agent config: {e}")
            except Exception as e:
                logger.error(f"Error initializing sub-agent: {e}")

    def _load_initial_context(self) -> None:
        """Load initial context from configuration."""
        initial_context = self.config.get("initial_context", {})

        for key, value in initial_context.items():
            self.state.update_context(key, value)
            logger.debug(f"Set initial context: {key} = {value}")

    async def perceive(self) -> dict[str, Any]:
        """
        Perceive the environment.

        This implementation:
        1. Calls the base perception method
        2. Forwards perceptions to all sub-agents
        3. Updates the shared context

        Returns:
            Dictionary of perceptions
        """
        perceptions = self._configured_perceptions()

        # Store perceptions
        self.state.last_perception = perceptions.copy() if perceptions else {}

        # Update context with perceptions
        if perceptions:
            self._update_context_from_perceptions(perceptions)

            # Forward perceptions to sub-agents
            await self._forward_perceptions_to_sub_agents(perceptions)

        return perceptions

    def update_beliefs(self, perception: dict[str, Any]) -> None:
        """Update the shared context from a perception dict (mirrors perceive())."""
        if perception:
            self._update_context_from_perceptions(perception)

    def _update_context_from_perceptions(self, perceptions: dict[str, Any]) -> None:
        """
        Update context based on perceptions.

        Args:
            perceptions: Dictionary of perceptions
        """
        # Update context with perception data
        for key, value in perceptions.items():
            # Skip internal keys
            if key.startswith("_"):
                continue

            # Update context
            self.state.update_context(key, value)

        # Add timestamp
        self.state.update_context("_last_perception_time", datetime.now().isoformat())

    async def _forward_perceptions_to_sub_agents(
        self, perceptions: dict[str, Any]
    ) -> None:
        """
        Forward perceptions to all sub-agents.

        Args:
            perceptions: Dictionary of perceptions
        """
        # Get active agents
        active_agents = self.state.get_active_agents()

        # Forward perceptions
        for wrapper in active_agents:
            try:
                # Create agent-specific perception (may be customized by agent type)
                agent_perceptions = self._customize_perceptions_for_agent(
                    wrapper.agent_type, perceptions
                )

                # Set perceptions directly in agent to avoid duplicate I/O
                wrapper.agent.last_perception = agent_perceptions
            except Exception as e:
                logger.error(
                    f"Error forwarding perceptions to {wrapper.agent.agent_id}: {e}"
                )

    def _customize_perceptions_for_agent(
        self, agent_type: str, perceptions: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Customize perceptions for specific agent types.

        Args:
            agent_type: Type of agent
            perceptions: Dictionary of perceptions

        Returns:
            Customized perceptions
        """
        # By default, just return a copy of perceptions
        agent_perceptions = perceptions.copy()

        # Customize based on agent type
        # Future implementations may add agent-type-specific formatting:
        # - BDI agents: belief-specific formatting
        # - Active inference agents: observation vector formatting
        # - RL agents: state representation formatting
        # - Rule-based agents: fact-specific formatting

        return agent_perceptions

    async def decide(self) -> dict[str, Any] | None:
        """
        Decide on the next action.

        This implementation:
        1. Gets a list of active sub-agents
        2. Asks each to make a decision
        3. Selects the best decision based on policy

        Returns:
            Action dictionary or None if no decision
        """
        # Get active agents
        active_agents = self.state.get_active_agents()

        if not active_agents:
            logger.warning("No active sub-agents available for decision making")

            # Get default action if configured
            default_action = self.config.get("default_action")
            if default_action:
                logger.debug("Using default action")
                return cast(dict[str, Any], default_action.copy())

            return None

        # Collect decisions from all active agents
        agent_decisions = []

        for wrapper in active_agents:
            try:
                # Get decision from sub-agent
                decision = await wrapper.agent.decide()

                if decision:
                    agent_decisions.append(
                        {
                            "agent_id": wrapper.agent.agent_id,
                            "agent_type": wrapper.agent_type,
                            "priority": wrapper.priority,
                            "decision": decision,
                        }
                    )
            except Exception as e:
                logger.error(
                    f"Error getting decision from {wrapper.agent.agent_id}: {e}"
                )

        if not agent_decisions:
            logger.warning("No decisions made by sub-agents")
            return None

        # Select decision based on policy
        selected_decision = await self._select_decision(agent_decisions)

        if selected_decision:
            # Record decision
            agent_id = selected_decision["agent_id"]
            decision = selected_decision["decision"]

            self.state.record_decision(agent_id, decision)

            # Add metadata to decision
            decision["_hybrid_source"] = {
                "agent_id": agent_id,
                "agent_type": selected_decision["agent_type"],
            }

            return cast(dict[str, Any], decision)

        return None

    async def _select_decision(
        self, agent_decisions: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """
        Select the best decision from multiple agent decisions.

        Args:
            agent_decisions: List of agent decisions

        Returns:
            Selected decision or None if no selection
        """
        # Get decision policy
        policy = self.config.get("decision_policy", "priority")

        if policy == "priority":
            # Select based on agent priority
            # (agents are already sorted by priority in get_active_agents)
            if agent_decisions:
                return agent_decisions[0]

        elif policy == "voting":
            # Implement a simple voting mechanism
            action_votes: dict[str, dict[str, Any]] = {}

            for decision in agent_decisions:
                # Extract action signature
                action_type = decision["decision"].get("action_type", "unknown")
                action_id = decision["decision"].get("action_id", "unknown")
                signature = f"{action_type}:{action_id}"

                # Count vote for this action
                if signature not in action_votes:
                    action_votes[signature] = {"count": 0, "decisions": []}

                action_votes[signature]["count"] += 1
                action_votes[signature]["decisions"].append(decision)

            # Find action with most votes
            max_votes = 0
            selected_signature = None

            for signature, vote_info in action_votes.items():
                if vote_info["count"] > max_votes:
                    max_votes = vote_info["count"]
                    selected_signature = signature

            # Return decision with highest priority among those with the most votes
            if selected_signature:
                selected_decisions = action_votes[selected_signature]["decisions"]
                selected_decisions.sort(key=lambda d: d["priority"], reverse=True)
                return cast(dict[str, Any], selected_decisions[0])

        elif policy == "negotiation":
            # More complex negotiation logic could be implemented here
            # For now, use priority as fallback
            if agent_decisions:
                return agent_decisions[0]

        return None

    async def act(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Execute an action.

        Args:
            action: Action to execute

        Returns:
            Result of the action
        """
        # Store action for reference
        self.state.last_action = action.copy()

        # Extract source agent info if available
        source_info = action.pop(
            "_hybrid_source", {"agent_id": None, "agent_type": None}
        )
        source_agent_id = source_info["agent_id"]

        result = await self._dispatch_action(action)

        # Determine success
        success = result.get("status", "") == "success"
        reward = result.get("reward", 0.0)

        # Record result
        self.state.record_result(result, success, reward)

        # Update context with result info
        self.state.update_context("last_action_success", success)
        self.state.update_context("last_action_reward", reward)

        # Share result with source sub-agent if identified
        if source_agent_id and source_agent_id in self.state.sub_agents:
            wrapper = self.state.sub_agents[source_agent_id]
            try:
                # Call source agent's act method directly
                # This allows the agent to update its internal state
                await wrapper.agent.act(action)
            except Exception as e:
                logger.error(f"Error updating source agent {source_agent_id}: {e}")

        return result

    async def shutdown(self) -> None:
        """Clean up resources when shutting down the agent."""
        # Save state if configured
        if "state_save_path" in self.config:
            await self._save_state(self.config["state_save_path"])

        # Shutdown all sub-agents
        for wrapper in self.state.sub_agents.values():
            try:
                await wrapper.agent.shutdown()
            except Exception as e:
                logger.error(
                    f"Error shutting down sub-agent {wrapper.agent.agent_id}: {e}"
                )

        logger.info("Hybrid agent %s shutdown complete", self.agent_id)

    def _register_default_action_handlers(self) -> None:
        """Register default action handlers."""
        self.register_action_handler("wait", self._handle_wait_action)
        self.register_action_handler("update_context", self._handle_update_context)
        self.register_action_handler("query_agents", self._handle_query_agents)
        self.register_action_handler("enable_agent", self._handle_enable_agent)
        self.register_action_handler("disable_agent", self._handle_disable_agent)

    def _register_default_perception_handlers(self) -> None:
        """Register default perception handlers."""
        self.register_perception_handler("sensor_data", self._handle_sensor_perceptions)

    async def _handle_wait_action(
        self, agent: "HybridAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle a wait action.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        # Extract duration
        duration = action.get("parameters", {}).get("duration", 1.0)

        # Wait for the specified duration
        await asyncio.sleep(duration)

        return {
            "status": "success",
            "action_id": action.get("action_id", ""),
            "message": f"Waited for {duration} seconds",
            "reward": 0.0,
        }

    async def _handle_update_context(
        self, agent: "HybridAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle updating context.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        params = action.get("parameters", {})

        if "key" not in params or "value" not in params:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": "Missing required parameters: key and value",
                "reward": -0.1,
            }

        key = params["key"]
        value = params["value"]

        # Update context
        agent.state.update_context(key, value)

        return {
            "status": "success",
            "action_id": action.get("action_id", ""),
            "message": f"Updated context: {key} = {value}",
            "reward": 0.0,
        }

    async def _handle_query_agents(
        self, agent: "HybridAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle querying agent information.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary with agent information
        """
        params = action.get("parameters", {})
        query_type = params.get("query_type", "active")

        if query_type == "active":
            # Get active agents
            active_agents = agent.state.get_active_agents()

            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "active_agents": [
                    {
                        "id": wrapper.agent.agent_id,
                        "type": wrapper.agent_type,
                        "priority": wrapper.priority,
                    }
                    for wrapper in active_agents
                ],
                "message": f"Found {len(active_agents)} active agents",
                "reward": 0.0,
            }

        elif query_type == "all":
            # Get all agents
            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "agents": [
                    {
                        "id": wrapper.agent.agent_id,
                        "type": wrapper.agent_type,
                        "priority": wrapper.priority,
                        "is_active": wrapper.is_active,
                    }
                    for wrapper in agent.state.sub_agents.values()
                ],
                "message": f"Found {len(agent.state.sub_agents)} agents",
                "reward": 0.0,
            }

        elif query_type == "performance":
            # Get performance metrics
            agent_metrics = {}

            for agent_id, wrapper in agent.state.sub_agents.items():
                agent_metrics[agent_id] = {
                    "type": wrapper.agent_type,
                    "decision_count": wrapper.decision_count,
                    "success_rate": (
                        wrapper.successful_decision_count / wrapper.decision_count
                        if wrapper.decision_count > 0
                        else 0.0
                    ),
                    "total_reward": wrapper.total_reward,
                }

            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "performance": {
                    "overall": {
                        "success_rate": (
                            agent.state.success_count / agent.state.total_decisions
                            if agent.state.total_decisions > 0
                            else 0.0
                        ),
                        "total_decisions": agent.state.total_decisions,
                        "total_reward": agent.state.total_reward,
                    },
                    "agents": agent_metrics,
                },
                "message": "Retrieved performance metrics",
                "reward": 0.0,
            }

        else:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": f"Unknown query type: {query_type}",
                "reward": -0.1,
            }

    async def _handle_enable_agent(
        self, agent: "HybridAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle enabling a sub-agent.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        params = action.get("parameters", {})

        if "agent_id" not in params:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": "Missing required parameter: agent_id",
                "reward": -0.1,
            }

        agent_id = params["agent_id"]

        if agent_id in agent.state.sub_agents:
            # Enable agent
            agent.state.sub_agents[agent_id].is_active = True

            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "message": f"Enabled agent: {agent_id}",
                "reward": 0.0,
            }
        else:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": f"Agent not found: {agent_id}",
                "reward": -0.1,
            }

    async def _handle_disable_agent(
        self, agent: "HybridAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle disabling a sub-agent.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        params = action.get("parameters", {})

        if "agent_id" not in params:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": "Missing required parameter: agent_id",
                "reward": -0.1,
            }

        agent_id = params["agent_id"]

        if agent_id in agent.state.sub_agents:
            # Disable agent
            agent.state.sub_agents[agent_id].is_active = False

            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "message": f"Disabled agent: {agent_id}",
                "reward": 0.0,
            }
        else:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": f"Agent not found: {agent_id}",
                "reward": -0.1,
            }

    def _handle_sensor_perceptions(
        self, agent: "HybridAgent", perception: dict[str, Any]
    ) -> None:
        """
        Process sensor perceptions.

        Args:
            agent: Agent receiving the perception
            perception: Perception data
        """
        # Extract sensor readings
        sensor_readings = perception.get("readings", {})

        # Update context with sensor readings
        for key, value in sensor_readings.items():
            agent.state.update_context(f"sensor_{key}", value)

    async def _save_state(self, path: str) -> None:
        """
        Save agent state to file.

        Args:
            path: Path to save state
        """
        state_data = self.state.to_dict()

        try:
            with open(path, "w") as f:
                json.dump(state_data, f, indent=2)
            logger.info(f"State saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def _load_state(self, path: str) -> None:
        """
        Load agent state from file.

        Args:
            path: Path to load state from
        """
        try:
            with open(path) as f:
                state_data = json.load(f)

            # Only load certain parts of the state
            # (sub-agents were already created in initialize)

            # Load context
            if "context" in state_data:
                self.state.context = state_data["context"]

            # Load history
            if "decision_history" in state_data:
                self.state.decision_history = state_data["decision_history"]

            # Load performance metrics
            if "total_decisions" in state_data:
                self.state.total_decisions = state_data["total_decisions"]
            if "success_count" in state_data:
                self.state.success_count = state_data["success_count"]
            if "failure_count" in state_data:
                self.state.failure_count = state_data["failure_count"]
            if "total_reward" in state_data:
                self.state.total_reward = state_data["total_reward"]

            logger.info(f"State loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
