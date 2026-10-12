#!/usr/bin/env python3

"""
Active Inference for GEO-INFER-AGENT

This module implements active inference principles for agent decision-making.
Active inference is a framework based on the free energy principle that unifies
perception, learning and decision-making as minimizing variational free energy.
"""

import numpy as np
import logging
from typing import Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical, kl_divergence

# Configure logger
logger = logging.getLogger("geo_infer_agent.active_inference")


@dataclass
class ActiveInferenceConfig:
    """Configuration for active inference models."""

    # General settings
    planning_horizon: int = 5  # Number of time steps to plan ahead
    precision: float = 1.0  # Precision parameter for information gain
    learning_rate: float = 0.01  # Learning rate for model updates
    use_gpu: bool = False  # Whether to use GPU if available
    random_seed: int | None = None  # Seed for stochastic batch sampling

    # Optimization settings
    optimization_steps: int = 100  # Steps for action optimization
    optimization_method: str = "gradient"  # "gradient" or "sampling"

    # Model architecture
    hidden_size: int = 64  # Size of hidden layers
    n_hidden_layers: int = 2  # Number of hidden layers

    # Inference settings
    n_samples: int = 10  # Number of samples for sampling-based inference
    beta: float = 1.0  # Temperature parameter for softmax


class GenerativeModel(nn.Module):
    """
    Neural network-based generative model for active inference.

    This model predicts:
    1. Observations (o) given states (s)
    2. Next states (s') given current states (s) and actions (a)
    3. The prior over states P(s)

    It also computes the expected free energy for action selection.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        action_dim: int,
        config: ActiveInferenceConfig | None = None,
    ):
        """
        Initialize the generative model.

        Args:
            state_dim: Dimensionality of the state space
            obs_dim: Dimensionality of the observation space
            action_dim: Dimensionality of the action space
            config: Active inference configuration
        """
        super().__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or ActiveInferenceConfig()

        # Device configuration
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        )

        # Define the likelihood model p(o|s) - observation given state
        self.likelihood_model = self._build_network(
            self.state_dim,
            self.obs_dim * 2,  # Mean and log variance for each obs dimension
        )

        # Define the transition model p(s'|s,a) - next state given current state and action
        self.transition_model = self._build_network(
            self.state_dim + self.action_dim,
            self.state_dim * 2,  # Mean and log variance for each state dimension
        )

        # Define the prior model p(s) - prior belief about states
        self.prior_model = self._build_network(
            1,  # Input is a constant prior token.
            self.state_dim * 2,  # Mean and log variance for each state dimension
        )

        # Define the policy network p(a|s) - action selection
        self.policy_model = self._build_network(self.state_dim, self.action_dim)

        # Define the preference model p(o) - preferred observations
        self.preference_model = nn.Parameter(
            torch.zeros(self.obs_dim, device=self.device), requires_grad=True
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)

        # Move model to appropriate device
        self.to(self.device)

        logger.info(
            f"Initialized generative model with state_dim={state_dim}, obs_dim={obs_dim}, action_dim={action_dim}"
        )

    def _build_network(self, input_dim: int, output_dim: int) -> nn.Sequential:
        """
        Build a neural network for the various components.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension

        Returns:
            Neural network model
        """
        layers: list[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(input_dim, self.config.hidden_size))
        layers.append(nn.ELU())

        # Hidden layers
        for _ in range(self.config.n_hidden_layers):
            layers.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
            layers.append(nn.ELU())

        # Output layer
        layers.append(nn.Linear(self.config.hidden_size, output_dim))

        return nn.Sequential(*layers)

    def likelihood(self, state: torch.Tensor) -> Normal:
        """
        Compute the likelihood distribution p(o|s).

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Normal distribution over observations
        """
        output = self.likelihood_model(state)
        mean, log_var = torch.split(output, self.obs_dim, dim=-1)
        var = torch.exp(log_var.clamp(min=-10, max=10))
        return Normal(mean, torch.sqrt(var))

    def transition(self, state: torch.Tensor, action: torch.Tensor) -> Normal:
        """
        Compute the transition distribution p(s'|s,a).

        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]

        Returns:
            Normal distribution over next states
        """
        x = torch.cat([state, action], dim=-1)
        output = self.transition_model(x)
        mean, log_var = torch.split(output, self.state_dim, dim=-1)
        var = torch.exp(log_var.clamp(min=-10, max=10))
        return Normal(mean, torch.sqrt(var))

    def prior(self, batch_size: int = 1) -> Normal:
        """
        Compute the prior distribution p(s).

        Args:
            batch_size: Batch size

        Returns:
            Normal distribution over states
        """
        # The prior network still expects a tensor-shaped input even though
        # its value is not conditioned on observations.
        prior_input = torch.ones(batch_size, 1, device=self.device)
        output = self.prior_model(prior_input)
        mean, log_var = torch.split(output, self.state_dim, dim=-1)
        var = torch.exp(log_var.clamp(min=-10, max=10))
        return Normal(mean, torch.sqrt(var))

    def policy(self, state: torch.Tensor) -> Categorical:
        """
        Compute the policy distribution p(a|s).

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Categorical distribution over actions
        """
        logits = self.policy_model(state)
        return Categorical(logits=logits)

    def encode(self, observation: torch.Tensor, steps: int = 100) -> Normal:
        """
        Infer the state given an observation (recognition/perception).

        Args:
            observation: Observation tensor [batch_size, obs_dim]
            steps: Number of optimization steps

        Returns:
            Inferred state distribution q(s)
        """
        batch_size = observation.shape[0]

        # Initialize state distribution parameters
        mean = torch.zeros(
            batch_size, self.state_dim, device=self.device, requires_grad=True
        )
        log_var = torch.zeros(
            batch_size, self.state_dim, device=self.device, requires_grad=True
        )

        # Optimizer for variational parameters
        var_optimizer = optim.Adam([mean, log_var], lr=0.1)

        # Iteratively update the variational distribution
        for step in range(steps):
            var_optimizer.zero_grad()

            # Current variational distribution
            var = torch.exp(log_var.clamp(min=-10, max=10))
            q_s = Normal(mean, torch.sqrt(var))

            # Prior distribution
            p_s = self.prior(batch_size)

            # Sample from variational distribution
            s_sample = q_s.rsample()

            # Likelihood
            p_o_given_s = self.likelihood(s_sample)

            # Compute free energy (negative ELBO)
            kl_term = kl_divergence(q_s, p_s).sum(dim=-1).mean()
            log_likelihood = p_o_given_s.log_prob(observation).sum(dim=-1).mean()
            free_energy = kl_term - log_likelihood

            # Backprop and update
            free_energy.backward()
            var_optimizer.step()

            if step % 20 == 0:
                logger.debug(
                    f"Perception step {step}: FE={free_energy.item():.4f}, KL={kl_term.item():.4f}, LL={log_likelihood.item():.4f}"
                )

        # Return the final variational distribution
        var = torch.exp(log_var.clamp(min=-10, max=10))
        return Normal(mean.detach(), torch.sqrt(var.detach()))

    def expected_free_energy(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the expected free energy for a state-action pair.

        Args:
            state: Current state [batch_size, state_dim]
            action: Action [batch_size, action_dim]

        Returns:
            Expected free energy value
        """
        batch_size = state.shape[0]

        # 1. Compute the predicted next state distribution
        next_state_dist = self.transition(state, action)

        # 2. Sample several possible next states
        next_states = next_state_dist.rsample(
            [self.config.n_samples]
        )  # [n_samples, batch_size, state_dim]
        next_states = next_states.view(
            -1, self.state_dim
        )  # [n_samples*batch_size, state_dim]

        # 3. For each next state, predict the observation
        obs_dist = self.likelihood(next_states)
        pred_obs = obs_dist.mean  # [n_samples*batch_size, obs_dim]
        pred_obs = pred_obs.view(
            self.config.n_samples, batch_size, self.obs_dim
        )  # [n_samples, batch_size, obs_dim]

        # 4. Compute the expected information gain (epistemic value)
        # This is approximated as the negative entropy of predicted observations
        epistemic_value = obs_dist.entropy().sum(dim=-1)  # [n_samples*batch_size]
        epistemic_value = epistemic_value.view(self.config.n_samples, batch_size).mean(
            dim=0
        )  # [batch_size]

        # 5. Compute pragmatic value (how close to preferred observations)
        # Expand preferences to match batch dimensions
        preferences = self.preference_model.expand(
            self.config.n_samples, batch_size, self.obs_dim
        )

        # Calculate average distance to preferences
        pragmatic_value = -torch.norm(pred_obs - preferences, dim=-1).mean(
            dim=0
        )  # [batch_size]

        # 6. Combine epistemic and pragmatic value
        # Note: In active inference, we want to minimize expected free energy
        # which is the sum of expected surprisal and expected divergence
        G: torch.Tensor = -self.config.precision * epistemic_value - pragmatic_value

        return G

    def plan_actions(self, current_state: torch.Tensor) -> list[torch.Tensor]:
        """
        Plan a sequence of actions to minimize expected free energy.

        Args:
            current_state: Current state tensor [batch_size, state_dim]

        Returns:
            List of planned actions
        """
        batch_size = current_state.shape[0]
        horizon = self.config.planning_horizon

        # Initialize action sequence with zeros
        action_sequence = [
            torch.zeros(
                batch_size, self.action_dim, device=self.device, requires_grad=True
            )
            for _ in range(horizon)
        ]

        # Create optimizer for action sequence
        action_optimizer = optim.Adam(action_sequence, lr=0.1)

        # Iteratively optimize the action sequence
        for step in range(self.config.optimization_steps):
            action_optimizer.zero_grad()

            # Initialize total EFE
            total_efe = torch.zeros(batch_size, device=self.device)

            # Initialize state for planning
            state = current_state

            # Roll out for each step in horizon
            for t in range(horizon):
                # Get action for this step
                action = action_sequence[t]

                # Calculate EFE for this step
                efe = self.expected_free_energy(state, action)
                total_efe = total_efe + efe

                # Predict next state
                next_state_dist = self.transition(state, action)
                state = next_state_dist.mean  # Use mean for planning

            # We want to minimize EFE
            loss = total_efe.mean()

            # Backprop and update
            loss.backward()
            action_optimizer.step()

            if step % 20 == 0:
                logger.debug(f"Planning step {step}: EFE={loss.item():.4f}")

        # Return the optimized action sequence
        return [a.detach() for a in action_sequence]

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select next action based on current state by minimizing expected free energy.

        Args:
            state: Current state tensor [batch_size, state_dim]

        Returns:
            Selected action tensor [batch_size, action_dim]
        """
        # Plan a sequence of actions
        action_sequence = self.plan_actions(state)

        # Return the first action in the sequence
        return action_sequence[0]

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        observations: torch.Tensor,
    ) -> dict[str, float]:
        """
        Update the generative model based on experience.

        Args:
            states: State tensor [batch_size, state_dim]
            actions: Action tensor [batch_size, action_dim]
            next_states: Next state tensor [batch_size, state_dim]
            observations: Observation tensor [batch_size, obs_dim]

        Returns:
            Dictionary with loss values
        """
        batch_size = states.shape[0]
        self.optimizer.zero_grad()

        # 1. Likelihood loss: p(o|s)
        obs_dist = self.likelihood(states)
        likelihood_loss = -obs_dist.log_prob(observations).sum(dim=-1).mean()

        # 2. Transition loss: p(s'|s,a)
        next_state_dist = self.transition(states, actions)
        transition_loss = -next_state_dist.log_prob(next_states).sum(dim=-1).mean()

        # 3. Prior loss: regularize toward standard normal
        prior_dist = self.prior(batch_size)
        standard_normal = Normal(
            torch.zeros_like(prior_dist.mean), torch.ones_like(prior_dist.stddev)
        )
        prior_loss = kl_divergence(prior_dist, standard_normal).sum(dim=-1).mean()

        # 4. Policy loss: encourage policies that minimize EFE
        efe = self.expected_free_energy(states, actions)
        policy_loss = efe.mean()

        # Total loss
        total_loss = likelihood_loss + transition_loss + 0.1 * prior_loss + policy_loss

        # Backprop and update
        total_loss.backward()
        self.optimizer.step()

        return {
            "likelihood_loss": likelihood_loss.item(),
            "transition_loss": transition_loss.item(),
            "prior_loss": prior_loss.item(),
            "policy_loss": policy_loss.item(),
            "total_loss": total_loss.item(),
        }


class ActiveInferenceAgent:
    """
    Agent that uses active inference for perception and action selection.

    This class provides a high-level interface for using active inference
    within GEO-INFER-AGENT, focusing on:
    - Perception (state inference from observations)
    - Planning (action selection to minimize expected free energy)
    - Learning (updating the generative model)
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        action_dim: int,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the active inference agent.

        Args:
            state_dim: Dimensionality of the state space
            obs_dim: Dimensionality of the observation space
            action_dim: Dimensionality of the action space
            config: Configuration dictionary
        """
        # Convert dictionary config to dataclass
        ai_config = None
        if config:
            ai_config = ActiveInferenceConfig(
                planning_horizon=config.get("planning_horizon", 5),
                precision=config.get("precision", 1.0),
                learning_rate=config.get("learning_rate", 0.01),
                use_gpu=config.get("use_gpu", False),
                random_seed=config.get("random_seed"),
                optimization_steps=config.get("iterations", 100),
                optimization_method=config.get("method", "gradient"),
                hidden_size=config.get("hidden_size", 64),
                n_hidden_layers=config.get("n_hidden_layers", 2),
                n_samples=config.get("n_samples", 10),
                beta=config.get("beta", 1.0),
            )

        # Initialize the generative model
        self.model = GenerativeModel(state_dim, obs_dim, action_dim, config=ai_config)

        # Seeded RNG for stochastic experience sampling
        seed = ai_config.random_seed if ai_config else None
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # Buffer for experience
        self.experience_buffer: dict[str, list[np.ndarray]] = {
            "states": [],
            "actions": [],
            "next_states": [],
            "observations": [],
        }

        logger.info(
            f"Initialized active inference agent with state_dim={state_dim}, obs_dim={obs_dim}, action_dim={action_dim}"
        )

    def perceive(self, observation: np.ndarray) -> np.ndarray:
        """
        Infer the current state given an observation.

        Args:
            observation: Observation vector

        Returns:
            Inferred state vector
        """
        # Convert numpy to torch
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.model.device)

        # Infer state
        state_dist = self.model.encode(obs_tensor)

        # Return mean of inferred state
        return state_dist.mean.squeeze(0).cpu().numpy()

    def plan(self, state: np.ndarray) -> list[np.ndarray]:
        """
        Plan a sequence of actions given the current state.

        Args:
            state: Current state vector

        Returns:
            List of planned action vectors
        """
        # Convert numpy to torch
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)

        # Plan actions
        action_sequence = self.model.plan_actions(state_tensor)

        # Convert back to numpy
        return [a.squeeze(0).cpu().numpy() for a in action_sequence]

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Select an action given the current state.

        Args:
            state: Current state vector

        Returns:
            Selected action vector
        """
        # Convert numpy to torch
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)

        # Select action
        action = self.model.select_action(state_tensor)

        # Convert back to numpy
        return action.squeeze(0).cpu().numpy()

    def add_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        observation: np.ndarray,
    ) -> None:
        """
        Add an experience to the buffer.

        Args:
            state: Current state vector
            action: Action vector
            next_state: Next state vector
            observation: Observation vector
        """
        self.experience_buffer["states"].append(state)
        self.experience_buffer["actions"].append(action)
        self.experience_buffer["next_states"].append(next_state)
        self.experience_buffer["observations"].append(observation)

    def learn(self, batch_size: int | None = None) -> dict[str, float]:
        """
        Update the generative model based on collected experience.

        Args:
            batch_size: Number of experiences to use (all if None)

        Returns:
            Dictionary with loss values
        """
        # Check if we have enough experience
        if len(self.experience_buffer["states"]) == 0:
            logger.warning("No experience to learn from")
            return {}

        # Determine batch size
        buffer_size = len(self.experience_buffer["states"])
        if batch_size is None or batch_size > buffer_size:
            batch_size = buffer_size

        # Sample random batch
        indices = self.rng.choice(buffer_size, size=batch_size, replace=False)

        # Prepare batch (stack first: converting a list of numpy arrays to a
        # tensor directly is extremely slow and warns under -W error)
        states = torch.FloatTensor(
            np.stack([self.experience_buffer["states"][i] for i in indices])
        ).to(self.model.device)
        actions = torch.FloatTensor(
            np.stack([self.experience_buffer["actions"][i] for i in indices])
        ).to(self.model.device)
        next_states = torch.FloatTensor(
            np.stack([self.experience_buffer["next_states"][i] for i in indices])
        ).to(self.model.device)
        observations = torch.FloatTensor(
            np.stack([self.experience_buffer["observations"][i] for i in indices])
        ).to(self.model.device)

        # Update model
        return self.model.update(states, actions, next_states, observations)

    def save(self, filepath: str) -> None:
        """
        Save the agent's model.

        Args:
            filepath: Path to save to
        """
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load the agent's model.

        Args:
            filepath: Path to load from

        Raises:
            RuntimeError: If the checkpoint is not a plain
                ``state_dict`` (``weights_only=True`` refuses pickled
                arbitrary objects to prevent code execution on load).
        """
        self.model.load_state_dict(torch.load(filepath, weights_only=True))
        logger.info(f"Model loaded from {filepath}")

    def clear_experience(self) -> None:
        """Clear the experience buffer."""
        self.experience_buffer = {
            "states": [],
            "actions": [],
            "next_states": [],
            "observations": [],
        }
        logger.info("Experience buffer cleared")
