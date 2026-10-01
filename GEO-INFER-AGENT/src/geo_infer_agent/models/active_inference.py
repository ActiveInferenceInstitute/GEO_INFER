#!/usr/bin/env python3

"""
Active Inference Agent.

This module implements the Active Inference framework for cognitive agents,
based on the Free Energy Principle. Active Inference agents perceive and act
to minimize surprise, continually refining their internal generative models
to better predict their environment.
"""

import logging
import os
import asyncio
import numpy as np
from typing import Any, cast
from collections.abc import Callable
import json

from geo_infer_agent.core.agent_base import BaseAgent, AgentState

logger = logging.getLogger("geo_infer_agent.models.active_inference")


class GenerativeModel:
    """
    Represents an agent's generative model of the world.

    The generative model encodes the agent's beliefs about:
    - How hidden states generate observations (likelihood)
    - How hidden states evolve over time (transition)
    - The agent's preferences over states (prior preferences)
    """

    def __init__(
        self,
        state_dimensions: int,
        observation_dimensions: int,
        control_dimensions: int,
        learning_rate: float = 0.01,
    ):
        """
        Initialize a generative model.

        Args:
            state_dimensions: Number of dimensions in the state space
            observation_dimensions: Number of dimensions in the observation space
            control_dimensions: Number of possible control actions
            learning_rate: Rate at which the model updates based on new evidence
        """
        # Model parameters
        self.state_dimensions = state_dimensions
        self.observation_dimensions = observation_dimensions
        self.control_dimensions = control_dimensions
        self.learning_rate = learning_rate

        # Initialize matrices
        # A: Likelihood mapping (observation given state)
        # Use slight diagonal-like bias to break symmetry so inference is non-trivial
        # Seeded RNG keeps initialization and stochastic action selection deterministic
        self.rng: np.random.Generator = np.random.default_rng(42)
        self.A = np.ones((observation_dimensions, state_dimensions)) / state_dimensions
        self.A += self.rng.uniform(0, 0.01, (observation_dimensions, state_dimensions))

        # B: Transition probabilities (next state given current state and action)
        self.B = np.zeros((state_dimensions, state_dimensions, control_dimensions))
        for i in range(control_dimensions):
            self.B[:, :, i] = np.eye(state_dimensions)

        # C: Prior preferences over observations (log probabilities)
        self.C = np.zeros(observation_dimensions)

        # D: Prior beliefs about initial states
        self.D = np.ones(state_dimensions) / state_dimensions

        # Current beliefs
        self.current_state_beliefs = self.D.copy()

        # History of beliefs and updates
        self.history: list[Any] = []

    def update_likelihood(self, observation: np.ndarray, state: np.ndarray) -> None:
        """
        Update the likelihood mapping (A) based on observed data.

        Args:
            observation: Observed data vector
            state: State vector
        """
        # Simple Hebbian-like learning for the likelihood mapping
        delta_A = np.outer(observation, state) - self.A
        self.A += self.learning_rate * delta_A

        # Normalize to ensure valid probability distributions
        self.A = self.A / np.sum(self.A, axis=0, keepdims=True)

    def update_transition(
        self, prev_state: np.ndarray, current_state: np.ndarray, action: int
    ) -> None:
        """
        Update the transition mapping (B) based on observed state transitions.

        Args:
            prev_state: Previous state vector
            current_state: Current state vector
            action: Action that was taken
        """
        # Update transition beliefs based on observed transitions
        delta_B = np.outer(current_state, prev_state) - self.B[:, :, action]
        self.B[:, :, action] += self.learning_rate * delta_B

        # Normalize to ensure valid probability distributions
        for a in range(self.control_dimensions):
            self.B[:, :, a] = self.B[:, :, a] / np.sum(
                self.B[:, :, a], axis=0, keepdims=True
            )

    def update_preferences(self, preferred_observations: np.ndarray) -> None:
        """
        Update preferences over observations.

        Args:
            preferred_observations: Vector of preferred observations
        """
        # Update the prior preferences based on new information
        self.C = (
            1 - self.learning_rate
        ) * self.C + self.learning_rate * preferred_observations

    def infer_state(self, observation: np.ndarray) -> np.ndarray:
        """
        Perform state inference given an observation.

        Args:
            observation: The observed data

        Returns:
            Posterior belief about the current state
        """
        # Bayesian inference using log-space to avoid numerical underflow
        # log-likelihood for each hidden state
        log_likelihood = np.zeros(self.state_dimensions)
        # Clamp A to avoid log(0)
        A_safe = np.clip(self.A, 1e-16, None)
        for i in range(self.state_dimensions):
            log_likelihood[i] = np.sum(observation * np.log(A_safe[:, i]))

        # log-posterior = log-likelihood + log-prior
        prior_safe = np.clip(self.current_state_beliefs, 1e-16, None)
        log_posterior = log_likelihood + np.log(prior_safe)

        # Shift for numerical stability, then exponentiate
        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)

        # Normalize
        total = np.sum(posterior)
        if total > 0:
            posterior = posterior / total
        else:
            # Fallback: uniform distribution (should not happen with log-space)
            posterior = np.ones(self.state_dimensions) / self.state_dimensions

        # Update current beliefs
        self.current_state_beliefs = posterior

        return np.asarray(posterior)

    def predict_next_state(self, current_state: np.ndarray, action: int) -> np.ndarray:
        """
        Predict the next state given current state and action.

        Args:
            current_state: Current state belief
            action: Action to take

        Returns:
            Predicted next state
        """
        # Apply transition matrix B
        return np.asarray(self.B[:, :, action] @ current_state)

    def expected_free_energy(
        self, state_belief: np.ndarray, action: int, planning_horizon: int = 1
    ) -> float:
        """
        Calculate the expected free energy for a policy.

        Args:
            state_belief: Current belief about the state
            action: Proposed action
            planning_horizon: How many steps to look ahead

        Returns:
            Expected free energy (lower is better)
        """
        # Initialize
        free_energy = 0
        current_belief = state_belief.copy()

        # Look ahead for planning_horizon steps
        for _ in range(planning_horizon):
            # Predict next state
            next_state = self.predict_next_state(current_belief, action)

            # Expected observations
            expected_obs = self.A @ next_state

            # Calculate information gain (epistemic value)
            information_gain = 0
            for i in range(self.state_dimensions):
                if next_state[i] > 0:
                    post_entropy = -next_state[i] * np.log(next_state[i])
                    information_gain += post_entropy

            # Calculate expected utility (pragmatic value)
            expected_utility = np.sum(expected_obs * self.C)

            # Combine epistemic and pragmatic value
            step_free_energy = information_gain - expected_utility
            free_energy += step_free_energy

            # Update for next iteration
            current_belief = next_state

        return free_energy

    def select_action(
        self, current_state_belief: np.ndarray, planning_horizon: int = 1
    ) -> int:
        """
        Select the best action to minimize expected free energy.

        Args:
            current_state_belief: Current belief about the state
            planning_horizon: How many steps to look ahead

        Returns:
            Selected action index
        """
        # Calculate expected free energy for each action
        ef_values = np.zeros(self.control_dimensions)

        for action in range(self.control_dimensions):
            ef_values[action] = self.expected_free_energy(
                current_state_belief, action, planning_horizon
            )

        # Select action with lowest expected free energy
        # Using softmax for stochastic action selection
        temperature = 1.0  # Adjust for exploration/exploitation tradeoff
        action_probabilities = np.exp(-ef_values / temperature)
        action_probabilities = action_probabilities / np.sum(action_probabilities)

        # Select action (either deterministically or stochastically)
        # Deterministic: return np.argmin(ef_values)
        # Stochastic:
        return self.rng.choice(self.control_dimensions, p=action_probabilities)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "state_dimensions": self.state_dimensions,
            "observation_dimensions": self.observation_dimensions,
            "control_dimensions": self.control_dimensions,
            "learning_rate": self.learning_rate,
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "C": self.C.tolist(),
            "D": self.D.tolist(),
            "current_state_beliefs": self.current_state_beliefs.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerativeModel":
        """Create a generative model from a dictionary."""
        model = cls(
            state_dimensions=data["state_dimensions"],
            observation_dimensions=data["observation_dimensions"],
            control_dimensions=data["control_dimensions"],
            learning_rate=data["learning_rate"],
        )

        model.A = np.array(data["A"])
        model.B = np.array(data["B"])
        model.C = np.array(data["C"])
        model.D = np.array(data["D"])
        model.current_state_beliefs = np.array(data["current_state_beliefs"])

        return model


class ActiveInferenceState(AgentState):
    """
    State for an Active Inference agent.

    Tracks the agent's generative model, observation history, and action history.
    """

    def __init__(
        self,
        state_dimensions: int = 10,
        observation_dimensions: int = 10,
        control_dimensions: int = 5,
    ):
        """
        Initialize the Active Inference agent state.

        Args:
            state_dimensions: Number of dimensions in the state space
            observation_dimensions: Number of dimensions in the observation space
            control_dimensions: Number of possible control actions
        """
        super().__init__()

        # Create the generative model
        self.model = GenerativeModel(
            state_dimensions=state_dimensions,
            observation_dimensions=observation_dimensions,
            control_dimensions=control_dimensions,
        )

        # History of observations, states, and actions
        self.observation_history: list[Any] = []
        self.state_history: list[np.ndarray] = []
        self.action_history: list[dict[str, Any]] = []

        # Current beliefs and states
        self.current_observation = np.zeros(observation_dimensions)
        self.current_state_belief = self.model.D.copy()

        # Performance metrics
        self.total_reward = 0.0
        self.prediction_errors: list[Any] = []

    @property
    def state_dimensions(self) -> int:
        return self.model.state_dimensions

    @property
    def observation_dimensions(self) -> int:
        return self.model.observation_dimensions

    @property
    def control_dimensions(self) -> int:
        return self.model.control_dimensions

    @property
    def generative_model(self) -> "GenerativeModel":
        return self.model

    def update_with_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Update agent state with a new observation.

        Args:
            observation: New observation vector

        Returns:
            Updated state belief
        """
        # Store observation as dict
        self.observation_history.append({"observation": observation.copy()})
        self.current_observation = observation.copy()

        # Infer state
        new_state_belief = self.model.infer_state(observation)

        # Store state
        self.state_history.append(new_state_belief.copy())
        self.current_state_belief = new_state_belief.copy()

        # Calculate prediction error
        if len(self.observation_history) > 1:
            predicted_obs = self.model.A @ self.current_state_belief
            error = np.mean((observation - predicted_obs) ** 2)
            self.prediction_errors.append(error)

        return new_state_belief

    def record_action(self, action: int, reward: float = 0.0) -> None:
        """
        Record an action taken by the agent.

        Args:
            action: Action index
            reward: Reward received
        """
        self.action_history.append({"action": action, "reward": reward})
        self.total_reward += reward

        # If we have enough history, update the model
        if len(self.state_history) >= 2 and len(self.action_history) >= 1:
            prev_state = self.state_history[-2]
            current_state = self.state_history[-1]

            # Update transition model
            self.model.update_transition(
                prev_state=prev_state, current_state=current_state, action=action
            )

    def update_preferences(self, preferred_obs: np.ndarray) -> None:
        """
        Update the agent's preferences.

        Args:
            preferred_obs: Vector of preferred observations
        """
        self.model.update_preferences(preferred_obs)

    def select_action(self, planning_horizon: int = 1) -> int:
        """
        Select the next action using active inference.

        Args:
            planning_horizon: How many steps to look ahead

        Returns:
            Selected action index
        """
        return self.model.select_action(self.current_state_belief, planning_horizon)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary representation."""
        model_dict = self.model.to_dict()
        obs_history = []
        for entry in self.observation_history:
            if isinstance(entry, dict):
                obs_history.append(
                    {
                        k: v.tolist() if hasattr(v, "tolist") else v
                        for k, v in entry.items()
                    }
                )
            else:
                obs_history.append(
                    entry.tolist() if hasattr(entry, "tolist") else entry
                )
        return {
            "state_dimensions": self.model.state_dimensions,
            "observation_dimensions": self.model.observation_dimensions,
            "control_dimensions": self.model.control_dimensions,
            "generative_model": model_dict,
            "observation_history": obs_history,
            "state_history": [state.tolist() for state in self.state_history],
            "action_history": self.action_history,
            "current_observation": self.current_observation.tolist(),
            "current_state_belief": self.current_state_belief.tolist(),
            "total_reward": self.total_reward,
            "prediction_errors": self.prediction_errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActiveInferenceState":
        """Create agent state from dictionary."""
        model_data = data.get("generative_model") or data.get("model", {})

        state = cls(
            state_dimensions=model_data.get(
                "state_dimensions", data.get("state_dimensions", 10)
            ),
            observation_dimensions=model_data.get(
                "observation_dimensions", data.get("observation_dimensions", 10)
            ),
            control_dimensions=model_data.get(
                "control_dimensions", data.get("control_dimensions", 5)
            ),
        )

        state.model = GenerativeModel.from_dict(model_data)
        raw_obs = data.get("observation_history", [])
        state.observation_history = []
        for entry in raw_obs:
            if isinstance(entry, dict):
                state.observation_history.append(
                    {
                        k: np.array(v) if isinstance(v, list) else v
                        for k, v in entry.items()
                    }
                )
            else:
                state.observation_history.append({"observation": np.array(entry)})
        state.state_history = [np.array(s) for s in data.get("state_history", [])]
        state.action_history = data.get("action_history", [])
        state.current_observation = np.array(data.get("current_observation", []))
        state.current_state_belief = np.array(data.get("current_state_belief", []))
        state.total_reward = data.get("total_reward", 0.0)
        state.prediction_errors = data.get("prediction_errors", [])

        return state


class ActiveInferenceAgent(BaseAgent):
    """
    Implementation of an Active Inference agent.

    This agent:
    1. Maintains a generative model of its environment
    2. Updates beliefs through perception
    3. Selects actions to minimize expected free energy
    4. Continuously learns and adapts its model
    """

    state: Any = None

    def __init__(self, agent_id: str | None = None, config: dict | None = None):
        """
        Initialize the Active Inference agent.

        Args:
            agent_id: Unique identifier for the agent
            config: Configuration parameters
        """
        super().__init__(agent_id=agent_id, config=config or {})

        # Convenience alias so internal code can use self.id consistently
        self.id = self.agent_id

        # Action and perception handler registries (underscore prefix for convention)
        self._action_handlers: dict[str, Any] = {}
        self._perception_handlers: dict[str, Any] = {}
        # Public aliases
        self.action_handlers = self._action_handlers
        self.perception_handlers = self._perception_handlers

        # Extract configuration
        self.config = config or {}

        # Planning parameters
        self.planning_horizon = self.config.get("planning_horizon", 3)

        # State is None until initialize() is called
        self.state = None

    async def initialize(self) -> None:
        """Initialize the agent, creating state and registering handlers."""
        logger.info(f"Initializing Active Inference agent: {self.id}")

        state_dims = self.config.get("state_dimensions", 10)
        obs_dims = self.config.get("observation_dimensions", 10)
        control_dims = self.config.get("control_dimensions", 5)

        # Create state now
        self.state = ActiveInferenceState(
            state_dimensions=state_dims,
            observation_dimensions=obs_dims,
            control_dimensions=control_dims,
        )

        # Register default handlers
        self._register_default_action_handlers()
        self._register_default_perception_handlers()

        # Load model if available
        model_path = self.config.get("model_path")
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

        # Initialize default preferences if provided
        if "default_preferences" in self.config:
            prefs = np.array(self.config["default_preferences"])
            self.state.update_preferences(prefs)

        # Register custom handlers if defined
        if "custom_handlers" in self.config:
            logger.debug("custom_handlers config provided; registered by subclass")

        logger.info("Active inference agent %s initialization complete", self.id)

    def update_beliefs(self, perception: dict[str, Any]) -> None:
        """
        Update agent beliefs based on perception data.

        Uses active inference principles to update the generative model's
        beliefs about hidden states given new observations.  Handles empty
        or None perception gracefully and is safe to call before
        ``initialize()`` (no-op when state is not yet created).

        Args:
            perception: Dictionary of observation data from the perceive step
        """
        if not perception or self.state is None:
            return

        # Convert numeric perception values to an observation vector.
        obs_values: list[float] = []
        for key in sorted(perception.keys()):
            val = perception[key]
            if isinstance(val, (int, float)):
                obs_values.append(float(val))
            elif isinstance(val, dict):
                for sub_key in sorted(val.keys()):
                    sub_val = val[sub_key]
                    if isinstance(sub_val, (int, float)):
                        obs_values.append(float(sub_val))

        if obs_values:
            obs_array = np.array(obs_values[: self.state.observation_dimensions])
            self.state.update_with_observation(obs_array)

        # Store all perception keys as named beliefs.
        for key, value in perception.items():
            self.state.update_belief(key, value)

    async def perceive(self) -> dict[str, Any]:
        """
        Perceive the environment.

        Returns:
            Dictionary of observations
        """
        # Collect observations from all registered sensors
        observations = {}

        # Read only explicitly configured sensor data.
        sensor_data = self._configured_perceptions()

        if sensor_data:
            observations.update(sensor_data)

        # Convert to the format needed by the model
        if observations:
            self._process_observations(observations)

        return observations

    def _process_observations(self, observations: dict[str, Any]) -> None:
        """
        Process raw observations into the format needed by the model.

        Args:
            observations: Raw observations from sensors
        """
        # Convert the dictionary observations to a vector for the model
        obs_vector = np.zeros(self.state.model.observation_dimensions)

        # Process different observation types
        if "vector_obs" in observations:
            # Direct vector observations
            raw_vector = observations["vector_obs"]
            length = min(len(raw_vector), len(obs_vector))
            obs_vector[:length] = raw_vector[:length]

        elif "categorical_obs" in observations:
            # One-hot encoding for categorical observations
            categories = observations["categorical_obs"]
            for i, cat in enumerate(categories):
                if i < self.state.model.observation_dimensions:
                    obs_vector[i] = cat

        else:
            # Try to extract numerical values from various sensors
            for key, value in observations.items():
                if isinstance(value, (int, float)) and key.startswith("sensor_"):
                    # Extract index from sensor name (e.g., sensor_0 -> 0)
                    try:
                        idx = int(key.split("_")[1])
                        if idx < self.state.model.observation_dimensions:
                            obs_vector[idx] = value
                    except (IndexError, ValueError):
                        logger.debug(
                            "Ignoring observation key %r: not a valid sensor index",
                            key,
                        )

        # Update state with the processed observation
        self.state.update_with_observation(obs_vector)

    async def decide(self) -> dict[str, Any] | None:
        """
        Decide on the next action.

        Returns:
            Action dictionary
        """
        # Select action using active inference
        action_idx = self.state.select_action(planning_horizon=self.planning_horizon)

        # Convert action index to action dictionary
        action = self._convert_action_index_to_action(action_idx)

        # Record the selected action
        self.state.record_action(action_idx)

        return action

    def _convert_action_index_to_action(self, action_idx: int) -> dict[str, Any]:
        """
        Convert an action index to an action dictionary.

        Args:
            action_idx: Index of the selected action

        Returns:
            Action dictionary
        """
        # Define action mapping
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

        Dispatches to the registered handler for the action's ``action_type``.
        If no handler is registered, returns a default success result so that
        the perception-decision-action loop can continue uninterrupted.

        Args:
            action: Action to execute

        Returns:
            Result of the action
        """
        action_type = action.get("action_type", "")
        action_id = action.get("action_id", "")

        logger.info(f"Agent {self.id} executing action: {action_type} - {action_id}")

        handler = self._action_handlers.get(action_type)
        if handler:
            result = await handler(self, action)
        else:
            logger.debug(
                f"Agent {self.id}: no handler for action_type '{action_type}', using default"
            )
            result = {
                "status": "success",
                "action_id": action_id,
                "message": f"No handler registered for action_type '{action_type}'",
                "reward": 0.0,
            }

        # Extract reward if available
        reward = result.get("reward", 0.0)

        # Update model with action results if action contained index
        if "parameters" in action and "index" in action["parameters"]:
            self.state.record_action(action["parameters"]["index"], reward=reward)

        return cast(dict[str, Any], result)

    async def shutdown(self) -> None:
        """Clean up resources when shutting down the agent."""
        # Save model if configured
        if "model_save_path" in self.config:
            self._save_model(self.config["model_save_path"])

        logger.info("Active inference agent %s shutdown complete", self.id)

    def register_action_handler(
        self, action_type: str, handler: Callable[..., Any]
    ) -> None:
        """Register a handler for a specific action type."""
        self._action_handlers[action_type] = handler

    def register_perception_handler(
        self, perception_type: str, handler: Callable[..., Any]
    ) -> None:
        """Register a handler for a specific perception type."""
        self._perception_handlers[perception_type] = handler

    def _register_default_action_handlers(self) -> None:
        """Register default action handlers."""
        self.register_action_handler("wait", self._handle_wait_action)
        self.register_action_handler(
            "update_preferences", self._handle_update_preferences
        )
        self.register_action_handler("query_model", self._handle_query_model)

    def _register_default_perception_handlers(self) -> None:
        """Register default perception handlers."""
        self.register_perception_handler("sensor_data", self._handle_sensor_perceptions)

    async def _handle_wait_action(
        self, agent: "ActiveInferenceAgent", action: dict[str, Any]
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

    async def _handle_update_preferences(
        self, agent: "ActiveInferenceAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle updating agent preferences.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary
        """
        # Extract preferences
        preferences = action.get("parameters", {}).get("preferences", [])

        if not preferences or not isinstance(preferences, list):
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": "Invalid preferences format",
                "reward": -0.1,
            }

        # Convert to numpy array of right dimensions
        pref_vector = np.zeros(agent.state.model.observation_dimensions)
        for i, p in enumerate(preferences):
            if i < len(pref_vector):
                pref_vector[i] = p

        # Update preferences
        agent.state.update_preferences(pref_vector)

        return {
            "status": "success",
            "action_id": action.get("action_id", ""),
            "message": "Preferences updated",
            "reward": 0.1,
        }

    async def _handle_query_model(
        self, agent: "ActiveInferenceAgent", action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle querying the agent's generative model.

        Args:
            agent: Agent executing the action
            action: Action parameters

        Returns:
            Result dictionary with model information
        """
        query_type = action.get("parameters", {}).get("query_type", "state")

        if query_type == "state":
            # Return current state belief
            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "state_belief": agent.state.current_state_belief.tolist(),
                "reward": 0.0,
            }
        elif query_type == "model":
            # Return model parameters
            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "model_parameters": {
                    "A": agent.state.model.A.tolist(),
                    "B": agent.state.model.B.tolist(),
                    "C": agent.state.model.C.tolist(),
                    "D": agent.state.model.D.tolist(),
                },
                "reward": 0.0,
            }
        elif query_type == "prediction":
            # Make predictions for the next state
            action_idx = action.get("parameters", {}).get("action_index", 0)
            next_state = agent.state.model.predict_next_state(
                agent.state.current_state_belief, action_idx
            )
            expected_obs = agent.state.model.A @ next_state

            return {
                "status": "success",
                "action_id": action.get("action_id", ""),
                "predicted_state": next_state.tolist(),
                "predicted_observation": expected_obs.tolist(),
                "reward": 0.0,
            }
        else:
            # Invalid query type
            return {
                "status": "error",
                "action_id": action.get("action_id", ""),
                "message": f"Invalid query type: {query_type}",
                "reward": -0.1,
            }

    def _handle_sensor_perceptions(
        self, agent: "ActiveInferenceAgent", perception: dict[str, Any]
    ) -> None:
        """
        Process sensor perceptions.

        Args:
            agent: Agent receiving the perception
            perception: Perception data
        """
        # Extract sensor readings
        sensor_readings = perception.get("readings", {})

        # Convert to observation vector
        if sensor_readings:
            obs = np.zeros(agent.state.model.observation_dimensions)

            for key, value in sensor_readings.items():
                if isinstance(value, (int, float)) and key.startswith("sensor_"):
                    try:
                        idx = int(key.split("_")[1])
                        if idx < len(obs):
                            obs[idx] = value
                    except (IndexError, ValueError):
                        logger.debug(
                            "Ignoring sensor key %r: not a valid sensor index",
                            key,
                        )

            # Update the state
            agent.state.update_with_observation(obs)

    def _save_model(self, path: str) -> None:
        """
        Save the generative model to a file.

        Args:
            path: File path to save to
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
        Load a generative model from a file.

        Args:
            path: File path to load from
        """
        try:
            with open(path) as f:
                model_data = json.load(f)

            self.state = ActiveInferenceState.from_dict(model_data)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
