#!/usr/bin/env python3

"""
Reinforcement Learning Agent.

This module implements RL-based agent architectures including:
- Q-Learning
- Deep Q-Networks (DQN)
- Policy Gradient methods
"""

import hashlib
import os
import logging
import asyncio
import numpy as np
import random
from typing import Any, cast
import json
from collections import deque

from geo_infer_agent.core.agent_base import BaseAgent, AgentState

logger = logging.getLogger("geo_infer_agent.models.rl")


class Experience:
    """
    Represents a single experience tuple (s, a, r, s', done).
    """

    def __init__(
        self, state: Any, action: int, reward: float, next_state: Any, done: bool
    ):
        """
        Initialize experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "state": (
                self.state.tolist() if hasattr(self.state, "tolist") else self.state
            ),
            "action": self.action,
            "reward": self.reward,
            "next_state": (
                self.next_state.tolist()
                if hasattr(self.next_state, "tolist")
                else self.next_state
            ),
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        """Create from dictionary representation."""
        state = (
            np.array(data["state"])
            if isinstance(data["state"], list)
            else data["state"]
        )
        next_state = (
            np.array(data["next_state"])
            if isinstance(data["next_state"], list)
            else data["next_state"]
        )

        return cls(
            state=state,
            action=data["action"],
            reward=data["reward"],
            next_state=next_state,
            done=data["done"],
        )


class QTable:
    """
    Simple Q-table for tabular Q-learning.
    """

    def __init__(self, state_size: int, action_size: int, default_value: float = 0.0):
        """
        Initialize Q-table.

        Args:
            state_size: Number of possible states
            action_size: Number of possible actions
            default_value: Initial value for Q-values
        """
        self.q_table: Any = np.ones((state_size, action_size)) * default_value

        self.state_size = state_size
        self.action_size = action_size

    def get_value(self, state: int, action: int) -> float:
        """
        Get Q-value for state-action pair.

        Args:
            state: State index
            action: Action index

        Returns:
            Q-value
        """
        if state >= self.state_size or action >= self.action_size:
            return 0.0

        return cast(float, self.q_table[state, action])

    def update_value(self, state: int, action: int, value: float) -> None:
        """
        Update Q-value for state-action pair.

        Args:
            state: State index
            action: Action index
            value: New Q-value
        """
        if state >= self.state_size or action >= self.action_size:
            return

        self.q_table[state, action] = value

    def get_best_action(self, state: int) -> int:
        """
        Get action with highest Q-value for state.

        Args:
            state: State index

        Returns:
            Best action index
        """
        if state >= self.state_size:
            return 0

        return cast(int, np.argmax(self.q_table[state]))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        q_table_list = self.q_table.tolist()

        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "q_table": q_table_list,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QTable":
        """Create from dictionary representation."""
        q_table = cls(state_size=data["state_size"], action_size=data["action_size"])

        q_table.q_table = np.array(data["q_table"])

        return q_table


class ReplayBuffer:
    """
    Experience replay buffer for DQN and similar algorithms.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer: deque[Experience] = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, experience: Experience) -> None:
        """
        Add experience to buffer.

        Args:
            experience: Experience to add
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        """
        Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(list(self.buffer), batch_size)

    def size(self) -> int:
        """Get current size of buffer."""
        return len(self.buffer)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "capacity": self.capacity,
            "buffer": [exp.to_dict() for exp in self.buffer],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayBuffer":
        """Create from dictionary representation."""
        buffer = cls(capacity=data["capacity"])

        for exp_data in data["buffer"]:
            exp = Experience.from_dict(exp_data)
            buffer.buffer.append(exp)

        return buffer


class RLState(AgentState):
    """
    State for a Reinforcement Learning agent.
    """

    def __init__(
        self, state_size: int = 10, action_size: int = 5, buffer_capacity: int = 10000
    ):
        """
        Initialize agent state.

        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            buffer_capacity: Capacity of replay buffer
        """
        super().__init__()

        # Q-learning parameters
        self.q_table = QTable(state_size, action_size)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Agent parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32

        # Current state and episode info
        self.current_state: Any | None = None
        self.current_episode = 0
        self.total_reward = 0.0
        self.episode_rewards: list[float] = []

        # Performance tracking
        self.last_100_rewards: deque[float] = deque(maxlen=100)
        self.training_iterations = 0

    def select_action(self, state: Any) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Selected action
        """
        # Convert state to index if needed
        state_idx = self._get_state_index(state)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.q_table.action_size - 1)
        else:
            # Exploitation: best known action
            return self.q_table.get_best_action(state_idx)

    def update_q_values(self, experience: Experience) -> None:
        """
        Update Q-values based on experience.

        Args:
            experience: Experience tuple
        """
        # Add to replay buffer
        self.replay_buffer.add(experience)

        # Convert states to indices if needed
        state_idx = self._get_state_index(experience.state)
        next_state_idx = self._get_state_index(experience.next_state)

        # Get current Q-value
        current_q = self.q_table.get_value(state_idx, experience.action)

        # Get next Q-value (max Q-value for next state)
        next_q = 0.0
        if not experience.done:
            next_q = max(
                [
                    self.q_table.get_value(next_state_idx, a)
                    for a in range(self.q_table.action_size)
                ]
            )

        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            experience.reward + self.discount_factor * next_q - current_q
        )

        # Update Q-table
        self.q_table.update_value(state_idx, experience.action, new_q)

        # Update epsilon with decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _get_state_index(self, state: Any) -> int:
        """
        Convert state to index if needed.

        Uses a stable hash (SHA-256 over the state's bytes/repr) so that the
        same state always maps to the same index, both across sign changes of
        individual entries and across interpreter runs with different
        ``PYTHONHASHSEED`` values (required for reproducible saved Q-tables).

        Args:
            state: State representation (int index, numpy array, or other)

        Returns:
            State index
        """
        # If state is already an index, return it
        if isinstance(state, int):
            return state

        # If state is a numpy array, hash its raw bytes.  ascontiguousarray
        # normalizes layout/striding so equal arrays hash equal.
        if isinstance(state, np.ndarray):
            digest = hashlib.sha256(np.ascontiguousarray(state).tobytes()).digest()
            return int.from_bytes(digest, "big") % self.q_table.state_size

        # Any other state type: stable hash of its repr
        digest = hashlib.sha256(repr(state).encode("utf-8")).digest()
        return int.from_bytes(digest, "big") % self.q_table.state_size

    def record_episode_reward(self, reward: float, episode_done: bool) -> None:
        """
        Record reward for current episode.

        Args:
            reward: Reward received
            episode_done: Whether episode is complete
        """
        self.total_reward += reward

        if episode_done:
            self.episode_rewards.append(self.total_reward)
            self.last_100_rewards.append(self.total_reward)
            self.current_episode += 1
            self.total_reward = 0.0

    def train_from_buffer(self, batch_size: int | None = None) -> None:
        """
        Train from replay buffer.

        Args:
            batch_size: Size of batch to sample
        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.replay_buffer.size() < batch_size:
            return

        # Sample batch of experiences
        experiences = self.replay_buffer.sample(batch_size)

        # Update Q-values for each experience
        for exp in experiences:
            self.update_q_values(exp)

        self.training_iterations += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "q_table": self.q_table.to_dict(),
            "replay_buffer": self.replay_buffer.to_dict(),
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "batch_size": self.batch_size,
            "current_episode": self.current_episode,
            "total_reward": self.total_reward,
            "episode_rewards": list(self.episode_rewards),
            "last_100_rewards": list(self.last_100_rewards),
            "training_iterations": self.training_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RLState":
        """Create from dictionary representation."""
        q_table_data = data["q_table"]
        buffer_data = data["replay_buffer"]

        state = cls(
            state_size=q_table_data["state_size"],
            action_size=q_table_data["action_size"],
            buffer_capacity=buffer_data["capacity"],
        )

        state.q_table = QTable.from_dict(q_table_data)
        state.replay_buffer = ReplayBuffer.from_dict(buffer_data)
        state.learning_rate = data["learning_rate"]
        state.discount_factor = data["discount_factor"]
        state.epsilon = data["epsilon"]
        state.epsilon_decay = data["epsilon_decay"]
        state.epsilon_min = data["epsilon_min"]
        state.batch_size = data["batch_size"]
        state.current_episode = data["current_episode"]
        state.total_reward = data["total_reward"]
        state.episode_rewards = data["episode_rewards"]
        state.last_100_rewards = deque(data["last_100_rewards"], maxlen=100)
        state.training_iterations = data["training_iterations"]

        return state


class RLAgent(BaseAgent):
    """
    Implementation of a Reinforcement Learning agent.

    Supports tabular Q-learning with optional extensions for
    Deep Q-Networks (DQN) and Policy Gradient methods.
    """

    state: RLState

    def __init__(self, agent_id: str | None = None, config: dict | None = None):
        """
        Initialize RL agent.

        Args:
            agent_id: Unique identifier for the agent
            config: Configuration parameters
        """
        super().__init__(agent_id=agent_id, config=config or {})

        # Extract configuration
        self.config = config or {}
        state_size = self.config.get("state_size", 100)
        action_size = self.config.get("action_size", 5)
        buffer_capacity = self.config.get("buffer_capacity", 10000)

        # Initialize state
        self.state = RLState(
            state_size=state_size,
            action_size=action_size,
            buffer_capacity=buffer_capacity,
        )

        # Configure RL parameters from config
        self._configure_rl_params()

        # Register action handlers
        self._register_default_action_handlers()

        # Training settings
        self.train_frequency = self.config.get("train_frequency", 4)
        self.train_batch_size = self.config.get("train_batch_size", 32)
        self.action_count = 0

    def _configure_rl_params(self) -> None:
        """Configure RL parameters from config."""
        if "learning_rate" in self.config:
            self.state.learning_rate = self.config["learning_rate"]

        if "discount_factor" in self.config:
            self.state.discount_factor = self.config["discount_factor"]

        if "epsilon" in self.config:
            self.state.epsilon = self.config["epsilon"]

        if "epsilon_decay" in self.config:
            self.state.epsilon_decay = self.config["epsilon_decay"]

        if "epsilon_min" in self.config:
            self.state.epsilon_min = self.config["epsilon_min"]

        if "batch_size" in self.config:
            self.state.batch_size = self.config["batch_size"]

    async def initialize(self) -> None:
        """Initialize the agent."""
        logger.info(f"Initializing RL agent: {self.agent_id}")

        # Load model if available
        model_path = self.config.get("model_path")
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

        # Initialize environment if needed
        if "initial_state" in self.config:
            self.state.current_state = self.config["initial_state"]

        logger.info("RL agent %s initialization complete", self.agent_id)

    async def perceive(self) -> dict[str, Any]:
        """
        Perceive the environment.

        Returns:
            Dictionary of observations
        """
        observations = self._configured_perceptions()

        # Extract state from observations
        if observations:
            if "state" in observations:
                self.state.current_state = observations["state"]
            elif "vector_state" in observations:
                self.state.current_state = np.array(observations["vector_state"])

        return observations

    def update_beliefs(self, perception: dict[str, Any]) -> None:
        """Update the RL state from a perception dict (mirrors perceive())."""
        if "state" in perception:
            self.state.current_state = perception["state"]
        elif "vector_state" in perception:
            self.state.current_state = np.array(perception["vector_state"])

    async def decide(self) -> dict[str, Any] | None:
        """
        Decide on the next action.

        Returns:
            Action dictionary
        """
        if self.state.current_state is None:
            logger.warning("No current state available for decision making")
            return None

        # Select action using epsilon-greedy policy
        action_idx = self.state.select_action(self.state.current_state)

        # Convert action index to action dictionary
        action = self._convert_action_index_to_action(action_idx)

        # Store action for later learning
        action["selected_idx"] = action_idx

        return action

    def _convert_action_index_to_action(self, action_idx: int) -> dict[str, Any]:
        """
        Convert action index to action dictionary.

        Args:
            action_idx: Index of selected action

        Returns:
            Action dictionary
        """
        # Check for action mapping in config
        action_mapping = self.config.get("action_mapping", {})

        if str(action_idx) in action_mapping:
            # Use predefined mapping
            return cast(dict[str, Any], action_mapping[str(action_idx)])
        else:
            # Default action format
            return {
                "action_type": "execute",
                "action_id": f"action_{action_idx}",
                "parameters": {"index": action_idx},
            }

    async def act(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Execute an action.

        Args:
            action: Action to execute

        Returns:
            Result of the action
        """
        # Store current state before action
        prev_state = self.state.current_state

        result = await self._dispatch_action(action)

        # Extract reward and episode termination flag
        reward = result.get("reward", 0.0)
        done = result.get("episode_done", False)

        # Record reward for performance tracking
        self.state.record_episode_reward(reward, done)

        # Get new state after action
        new_state = self.state.current_state

        # If we have valid states, create experience and learn
        if (
            prev_state is not None
            and new_state is not None
            and "selected_idx" in action
        ):
            action_idx = action["selected_idx"]

            # Create experience tuple
            experience = Experience(
                state=prev_state,
                action=action_idx,
                reward=reward,
                next_state=new_state,
                done=done,
            )

            # Update Q-values
            self.state.update_q_values(experience)

            # Periodic training from replay buffer
            self.action_count += 1
            if self.action_count % self.train_frequency == 0:
                self.state.train_from_buffer(self.train_batch_size)

        return result

    async def shutdown(self) -> None:
        """Clean up resources when shutting down the agent."""
        # Save model if configured
        if "model_save_path" in self.config:
            self._save_model(self.config["model_save_path"])

        logger.info("RL agent %s shutdown complete", self.agent_id)

    def _register_default_action_handlers(self) -> None:
        """Register default action handlers."""
        self.register_action_handler("wait", self._handle_wait_action)
        self.register_action_handler("query_state", self._handle_query_state)
        self.register_action_handler("set_learning_params", self._handle_set_params)

    async def _handle_wait_action(
        self, agent: "RLAgent", action: dict[str, Any]
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
            "episode_done": False,
        }

    async def _handle_query_state(
        self, agent: "RLAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle state query action.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            State information
        """
        query_type = action.get("parameters", {}).get("query_type", "performance")

        if query_type == "performance":
            # Return performance metrics
            avg_reward = 0.0
            if agent.state.last_100_rewards:
                avg_reward = sum(agent.state.last_100_rewards) / len(
                    agent.state.last_100_rewards
                )

            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "avg_reward_100": avg_reward,
                "total_episodes": agent.state.current_episode,
                "current_epsilon": agent.state.epsilon,
                "reward": 0.0,
                "episode_done": False,
            }
        elif query_type == "q_values":
            # Return Q-values for current state
            state_idx = agent.state._get_state_index(agent.state.current_state)
            q_values = [
                agent.state.q_table.get_value(state_idx, a)
                for a in range(agent.state.q_table.action_size)
            ]

            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "q_values": q_values,
                "current_state_idx": state_idx,
                "reward": 0.0,
                "episode_done": False,
            }
        else:
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": f"Unknown query type: {query_type}",
                "reward": -0.1,
                "episode_done": False,
            }

    async def _handle_set_params(
        self, agent: "RLAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle setting learning parameters.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        params = action.get("parameters", {})
        updated = []

        if "learning_rate" in params:
            agent.state.learning_rate = params["learning_rate"]
            updated.append("learning_rate")

        if "epsilon" in params:
            agent.state.epsilon = params["epsilon"]
            updated.append("epsilon")

        if "epsilon_decay" in params:
            agent.state.epsilon_decay = params["epsilon_decay"]
            updated.append("epsilon_decay")

        if "discount_factor" in params:
            agent.state.discount_factor = params["discount_factor"]
            updated.append("discount_factor")

        if updated:
            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "message": f"Updated parameters: {', '.join(updated)}",
                "reward": 0.0,
                "episode_done": False,
            }
        else:
            return {
                "status": "warning",
                "action_id": action.get("action_id", ""),
                "message": "No parameters were updated",
                "reward": 0.0,
                "episode_done": False,
            }

    def _save_model(self, path: str) -> None:
        """
        Save model to file.

        Args:
            path: Path to save model
        """
        model_data = self.state.to_dict()

        try:
            with open(path, "w") as f:
                json.dump(model_data, f, indent=2)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_model(self, path: str) -> None:
        """
        Load model from file.

        Args:
            path: Path to load model from
        """
        try:
            with open(path) as f:
                model_data = json.load(f)

            self.state = RLState.from_dict(model_data)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
