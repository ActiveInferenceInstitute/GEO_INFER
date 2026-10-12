#!/usr/bin/env python3

"""
Base Agent for GEO-INFER-AGENT

This module provides the base agent class that all specialized agents
will inherit from, implementing core functionality and interfaces.
"""

import logging
import uuid
import json
import asyncio
import inspect
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Callable

# Configure logger
logger = logging.getLogger(__name__)


class AgentState:
    """
    Represents the internal state of an agent.

    This class tracks the agent's:
    - Beliefs (what it thinks about the world)
    - Desires (goals it wants to achieve)
    - Intentions (current plan of actions)
    - Memory (past observations and actions)
    """

    def __init__(self, capacity: int = 1000):
        """
        Initialize agent state.

        Args:
            capacity: Maximum number of memory items to store
        """
        self.beliefs: dict[str, Any] = {}  # Current world model
        self.desires: list[dict[str, Any]] = []  # Goals
        self.intentions: list[dict[str, Any]] = []  # Planned actions
        self.memory: list[dict[str, Any]] = []  # Past observations and actions
        self.memory_capacity = capacity
        self.creation_time = datetime.now()
        self.last_update = self.creation_time

    def update_belief(self, key: str, value: Any) -> None:
        """
        Update a belief with new information.

        Args:
            key: Belief identifier
            value: New belief value
        """
        old_value = self.beliefs.get(key)
        self.beliefs[key] = value
        self.last_update = datetime.now()

        # Add to memory
        if not self._values_equal(old_value, value):
            self.add_to_memory(
                {
                    "type": "belief_update",
                    "key": key,
                    "old_value": old_value,
                    "new_value": value,
                    "timestamp": self.last_update.isoformat(),
                }
            )

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        """Compare scalar, container, and array-like beliefs safely."""
        if left is right:
            return True
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(
                AgentState._values_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                AgentState._values_equal(a, b) for a, b in zip(left, right)
            )
        try:
            result = left == right
        except (TypeError, ValueError):
            return False
        if isinstance(result, bool):
            return result
        try:
            return bool(result.all())
        except (AttributeError, TypeError, ValueError):
            return False

    def add_desire(self, desire: dict[str, Any]) -> None:
        """
        Add a new goal/desire for the agent.

        Args:
            desire: Dictionary containing goal information
                   Must include 'priority' and 'description' keys
        """
        if "priority" not in desire or "description" not in desire:
            raise ValueError("Desire must include 'priority' and 'description'")

        desire["timestamp"] = datetime.now().isoformat()
        self.desires.append(desire)
        self.desires.sort(key=lambda x: x["priority"], reverse=True)
        self.last_update = datetime.now()

        # Add to memory
        self.add_to_memory(
            {
                "type": "desire_added",
                "desire": desire,
                "timestamp": self.last_update.isoformat(),
            }
        )

    def set_intention(self, intention: dict[str, Any]) -> None:
        """
        Set current intention/plan.

        Args:
            intention: Dictionary containing plan information
                      Must include 'actions' key with list of action steps
        """
        if "actions" not in intention:
            raise ValueError("Intention must include 'actions' list")

        intention["timestamp"] = datetime.now().isoformat()
        self.intentions.append(intention)
        self.last_update = datetime.now()

        # Add to memory
        self.add_to_memory(
            {
                "type": "intention_set",
                "intention": intention,
                "timestamp": self.last_update.isoformat(),
            }
        )

    def add_to_memory(self, item: dict[str, Any]) -> None:
        """
        Add an item to agent's memory.

        Args:
            item: Memory item (observation, action, or belief change)
        """
        if "timestamp" not in item:
            item["timestamp"] = datetime.now().isoformat()

        self.memory.append(item)

        # Enforce memory capacity
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)

    def get_top_desire(self) -> dict[str, Any] | None:
        """
        Get the highest priority desire.

        Returns:
            Highest priority desire or None if no desires
        """
        return self.desires[0] if self.desires else None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert state to dictionary for serialization.

        Returns:
            Dictionary representation of state
        """
        return {
            "beliefs": self.beliefs,
            "desires": self.desires,
            "intentions": self.intentions,
            "memory": self.memory,
            "creation_time": self.creation_time.isoformat(),
            "last_update": self.last_update.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """
        Create state from dictionary.

        Args:
            data: Dictionary representation of state

        Returns:
            AgentState instance
        """
        state = cls()
        state.beliefs = data.get("beliefs", {})
        state.desires = data.get("desires", [])
        state.intentions = data.get("intentions", [])
        state.memory = data.get("memory", [])
        state.creation_time = datetime.fromisoformat(
            data.get("creation_time", datetime.now().isoformat())
        )
        state.last_update = datetime.fromisoformat(
            data.get("last_update", datetime.now().isoformat())
        )
        return state


class BaseAgent(ABC):
    """
    Base agent class that all specialized agents inherit from.

    This class implements:
    - Core agent lifecycle (initialize, run, stop)
    - State management
    - Perception and action interfaces
    - Communication with other agents
    - Persistence mechanisms
    """

    def __init__(self, agent_id: str | None = None, config: dict | None = None):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier for this agent (auto-generated if None)
            config: Configuration dictionary
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.config = config or {}
        self.state = AgentState(capacity=self.config.get("memory_capacity", 1000))
        self.action_handlers: dict[str, Callable[..., Any]] = {}
        self.perception_handlers: dict[str, Callable[..., Any]] = {}
        self.running = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self.start_time: datetime | None = None
        self.stop_time: datetime | None = None
        self.last_perception: dict[str, Any] = {}
        self.last_action: dict[str, Any] = {}
        # Exception raised inside run(), if any.  run() logs and stores crash
        # details here so callers can distinguish a crash from a clean stop
        # without changing run()'s fire-and-forget contract.
        self.last_error: BaseException | None = None
        # Initialize communication channels
        self.message_queue: asyncio.Queue = asyncio.Queue()

        logger.info(f"Agent {self.agent_id} initialized")

    def register_action_handler(
        self, action_type: str, handler: Callable[..., Any]
    ) -> None:
        """Register a callable that executes one action type."""
        if not action_type or not callable(handler):
            raise ValueError("action_type and callable handler are required")
        self.action_handlers[action_type] = handler

    def register_perception_handler(
        self, perception_type: str, handler: Callable[..., Any]
    ) -> None:
        """Register a callable that processes one perception type."""
        if not perception_type or not callable(handler):
            raise ValueError("perception_type and callable handler are required")
        self.perception_handlers[perception_type] = handler

    def _configured_perceptions(self) -> dict[str, Any]:
        """Return explicitly configured sensor readings for this agent."""
        readings = self.config.get("sensor_readings", {})
        if readings is None:
            return {}
        if not isinstance(readings, dict):
            raise TypeError("sensor_readings must be a mapping")
        return dict(readings)

    async def _dispatch_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Dispatch an action through the registered handler for its type."""
        action_type = action.get("action_type")
        if not action_type:
            return {
                "status": "error",
                "message": "action_type is required",
                "reward": 0.0,
            }

        handler = self.action_handlers.get(action_type)
        if handler is None:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": f"No handler registered for action_type '{action_type}'",
                "reward": 0.0,
            }

        result = handler(self, action)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            raise TypeError(f"Action handler '{action_type}' must return a mapping")
        return result

    async def run(self) -> None:
        """
        Main agent execution loop.

        This method:
        1. Initializes the agent
        2. Runs the perception-decision-action loop
        3. Handles shutdown when stopped
        """
        if self.running:
            logger.warning(f"Agent {self.agent_id} is already running")
            return

        self.running = True
        self.start_time = datetime.now()
        logger.info(f"Agent {self.agent_id} starting at {self.start_time}")

        # Store current event loop
        self.loop = asyncio.get_running_loop()

        try:
            # Initialize specific agent implementation
            await self.initialize()

            # Run agent until stopped
            while self.running:
                # Process any incoming messages
                await self.process_messages()

                # Perceive environment
                perception = await self.perceive()

                # Update beliefs based on perception
                self.update_beliefs(perception)

                # Decide what to do
                action = await self.decide()

                # Execute selected action
                if action:
                    result = await self.act(action)
                    # Record action and result
                    self.state.add_to_memory(
                        {
                            "type": "action",
                            "action": action,
                            "result": result,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                # Check if we should stop based on runtime limits
                self._check_runtime_limits()

                # Sleep for specified decision frequency
                await asyncio.sleep(self.config.get("decision_frequency", 1))

        except Exception as e:
            self.last_error = e
            logger.error(
                f"Agent {self.agent_id} encountered an error: {str(e)}", exc_info=True
            )
        finally:
            # Cleanup
            self.running = False
            self.stop_time = datetime.now()
            logger.info(f"Agent {self.agent_id} stopped at {self.stop_time}")

            # Calculate runtime
            assert self.stop_time is not None and self.start_time is not None
            runtime = (self.stop_time - self.start_time).total_seconds()
            logger.info(f"Agent {self.agent_id} ran for {runtime:.2f} seconds")

            # Run shutdown procedure
            await self.shutdown()

    def _check_runtime_limits(self) -> None:
        """Check if the agent should stop based on runtime limits."""
        if not self.start_time:
            return

        max_runtime = self.config.get("max_runtime")
        if max_runtime:
            runtime = (datetime.now() - self.start_time).total_seconds()
            if runtime > max_runtime:
                logger.info(
                    f"Agent {self.agent_id} reached max runtime of {max_runtime} seconds"
                )
                self.stop()

    def stop(self) -> None:
        """Stop the agent execution."""
        logger.info(f"Agent {self.agent_id} stopping...")
        self.running = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize agent before running.

        Subclasses must implement this to set up agent-specific resources.
        """

    @abstractmethod
    async def perceive(self) -> dict[str, Any]:
        """
        Collect information from the environment.

        Returns:
            Dictionary with perception data
        """

    @abstractmethod
    def update_beliefs(self, perception: dict[str, Any]) -> None:
        """
        Update agent's beliefs based on perception.

        Args:
            perception: Data from the perceive method
        """

    @abstractmethod
    async def decide(self) -> dict[str, Any] | None:
        """
        Decide on next action based on beliefs and goals.

        Returns:
            Action to execute or None
        """

    @abstractmethod
    async def act(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a selected action.

        Args:
            action: Action to execute

        Returns:
            Result of the action
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Clean up resources when agent stops.

        Subclasses must implement this to release agent-specific resources.
        """

    async def send_message(self, to_agent_id: str, content: dict[str, Any]) -> bool:
        """
        Send a message to another registered agent.

        The default transport is the in-process :class:`AgentRegistry`: the
        message is placed directly on the recipient's queue via
        :meth:`receive_message`.  There is deliberately no "log-only" success
        path — if the recipient is not registered, this method returns False
        instead of pretending the message was delivered.  Subclasses may
        override this method to integrate an external transport (message
        broker, pub/sub, HTTP, ...).

        Args:
            to_agent_id: Recipient agent ID (must be registered)
            content: Message content

        Returns:
            True if the message was delivered to the recipient's queue,
            False if no agent with ``to_agent_id`` is registered
        """
        # Imported lazily: agent_registry imports this module at module level,
        # so a top-level import here would be circular.
        #
        # Registry identity note: the registry is a singleton, but tests (and
        # hot-reload scenarios) may have re-created it, leaving the module's
        # ``agent_registry`` global and ``AgentRegistry._instance`` pointing
        # at different objects.  The sender lives in whichever registry
        # registered it, so both are consulted before giving up.
        from geo_infer_agent.core.agent_registry import AgentRegistry, agent_registry

        candidate_registries = [agent_registry]
        current = AgentRegistry()
        if current is not agent_registry:
            candidate_registries.append(current)

        recipient = None
        for registry in candidate_registries:
            try:
                recipient = registry.get_agent(to_agent_id)
                break
            except KeyError:
                continue

        if recipient is None:
            logger.warning(
                "Agent %s cannot send message: recipient %s is not registered",
                self.agent_id,
                to_agent_id,
            )
            return False

        message = {
            "from": self.agent_id,
            "to": to_agent_id,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "message_id": str(uuid.uuid4()),
        }
        logger.debug(f"Agent {self.agent_id} sending message to {to_agent_id}")
        await recipient.receive_message(message)
        return True

    async def receive_message(self, message: dict[str, Any]) -> None:
        """
        Receive a message from another agent.

        Args:
            message: Message content
        """
        logger.debug(
            f"Agent {self.agent_id} received message from {message.get('from')}"
        )
        await self.message_queue.put(message)

    async def process_messages(self) -> None:
        """Process all messages in the queue."""
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                await self._handle_message(message)
                self.message_queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """
        Handle a single message.

        Args:
            message: Message to handle
        """
        # Record in memory
        self.state.add_to_memory(
            {
                "type": "message_received",
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Default implementation just logs the message
        # Subclasses should override for specific behavior
        logger.info(f"Agent {self.agent_id} handling message: {message}")

    def save_state(self, filepath: str | None = None) -> str:
        """
        Save agent state to file.

        Args:
            filepath: Path to save state (default: agent_<id>_state.json)

        Returns:
            Filepath where state was saved
        """
        if filepath is None:
            filepath = f"agent_{self.agent_id}_state.json"

        try:
            with open(filepath, "w") as f:
                state_dict = self.state.to_dict()
                state_dict["agent_id"] = self.agent_id
                state_dict["agent_type"] = self.__class__.__name__
                state_dict["save_time"] = datetime.now().isoformat()

                json.dump(state_dict, f, indent=2)

            logger.info(f"Agent {self.agent_id} state saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save agent state: {str(e)}")
            raise

    @classmethod
    def load_state(cls, filepath: str, config: dict | None = None) -> "BaseAgent":
        """
        Create agent from saved state.

        Args:
            filepath: Path to state file
            config: Configuration to use (overrides saved config)

        Returns:
            Instantiated agent with loaded state
        """
        try:
            with open(filepath) as f:
                state_dict = json.load(f)

            agent_id = state_dict.get("agent_id")
            agent = cls(agent_id=agent_id, config=config)
            agent.state = AgentState.from_dict(state_dict)

            logger.info(f"Loaded agent {agent_id} state from {filepath}")
            return agent
        except Exception as e:
            logger.error(f"Failed to load agent state: {str(e)}")
            raise


# Example subclass (minimal implementation)
class ExampleAgent(BaseAgent):
    """Example agent implementation for demonstration."""

    async def initialize(self) -> None:
        """Initialize the example agent."""
        logger.info(f"Example agent {self.agent_id} initializing")

        # Set initial beliefs
        self.state.update_belief("environment_known", False)

        # Set initial desires
        self.state.add_desire(
            {"description": "Explore environment", "priority": 10, "completed": False}
        )

    async def perceive(self) -> dict[str, Any]:
        """Simple perception that just returns current time."""
        return {
            "current_time": datetime.now().isoformat(),
            "random_observation": uuid.uuid4().hex[:8],
        }

    def update_beliefs(self, perception: dict[str, Any]) -> None:
        """Update beliefs based on perception."""
        # Just store the entire perception as beliefs
        for key, value in perception.items():
            self.state.update_belief(key, value)

        # After 5 observations, consider environment known
        if len(self.state.memory) > 5:
            self.state.update_belief("environment_known", True)

    async def decide(self) -> dict[str, Any] | None:
        """Simple decision making."""
        # Get top desire
        desire = self.state.get_top_desire()
        if not desire:
            return None

        # If environment not known, explore
        if not self.state.beliefs.get("environment_known", False):
            return {"type": "explore", "target": "environment", "params": {}}
        else:
            # Mark exploration desire as completed
            for d in self.state.desires:
                if d["description"] == "Explore environment":
                    d["completed"] = True

            # No action needed
            return None

    async def act(self, action: dict[str, Any]) -> dict[str, Any]:
        """Execute action."""
        if action["type"] == "explore":
            logger.info(f"Agent {self.agent_id} exploring {action['target']}")
            # Simulate exploration
            await asyncio.sleep(0.5)
            return {"status": "success", "info": "Exploration completed"}

        return {"status": "error", "info": "Unknown action type"}

    async def shutdown(self) -> None:
        """Clean up resources."""
        logger.info(f"Example agent {self.agent_id} shutting down")
