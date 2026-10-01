#!/usr/bin/env python3

"""
Active Inference Agent for Geospatial Exploration

This example demonstrates how to use the Active Inference agent framework
for geospatial exploration and decision-making.

The agent explores a simulated geospatial environment, updating its
generative model of the world while optimizing its information gathering
and goal-seeking behavior.
"""

import asyncio
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Any

from geo_infer_agent.models.active_inference import ActiveInferenceAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeospatialEnvironment:
    """
    A simplified geospatial environment for agent testing.

    This environment simulates a 2D spatial grid with:
    - Various land cover types
    - Points of interest
    - Spatial constraints
    """

    def __init__(self, width: int = 20, height: int = 20, seed: int = 42):
        """
        Initialize the environment.

        Args:
            width: Width of the environment grid
            height: Height of the environment grid
            seed: Random seed for reproducibility
        """
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

        # Create land cover types (0: water, 1: forest, 2: urban, 3: agriculture)
        self.land_cover = self._generate_land_cover()

        # Create points of interest (locations with high reward)
        self.points_of_interest = self._generate_points_of_interest(num_points=5)

        # Current agent position
        self.agent_position = [width // 2, height // 2]

        # Observation history
        self.observation_history = []

    def _generate_land_cover(self) -> np.ndarray:
        """Generate a random land cover map."""
        # Start with all water
        land_cover = np.zeros((self.height, self.width))

        # Create some landmasses using cellular automata
        for _ in range(10):
            # Random seed points
            for _ in range(10):
                x = int(self.rng.integers(0, self.width))
                y = int(self.rng.integers(0, self.height))
                land_cover[y, x] = self.rng.integers(1, 4)  # Random land type

            # Expand regions
            for _ in range(3):
                new_land = land_cover.copy()
                for y in range(1, self.height - 1):
                    for x in range(1, self.width - 1):
                        if land_cover[y, x] == 0:
                            neighbors = land_cover[y - 1 : y + 2, x - 1 : x + 2]
                            if np.any(neighbors > 0):
                                # Take most common neighbor type
                                neighbor_types = neighbors[neighbors > 0]
                                if len(neighbor_types) > 0:
                                    new_land[y, x] = int(
                                        self.rng.choice(neighbor_types)
                                    )
                land_cover = new_land

        return land_cover

    def _generate_points_of_interest(
        self, num_points: int
    ) -> list[tuple[int, int, float]]:
        """
        Generate points of interest with associated rewards.

        Args:
            num_points: Number of points to generate

        Returns:
            List of (x, y, reward) tuples
        """
        points = []
        for _ in range(num_points):
            x = int(self.rng.integers(0, self.width))
            y = int(self.rng.integers(0, self.height))
            reward = self.rng.uniform(0.5, 1.0)
            points.append((x, y, reward))
        return points

    def get_observation(self, position: list[int], obs_radius: int = 2) -> np.ndarray:
        """
        Get an observation centered on the given position.

        Args:
            position: [x, y] position
            obs_radius: Radius of observation window

        Returns:
            Observation vector
        """
        x, y = position

        # Extract local land cover within observation radius
        x_min, x_max = max(0, x - obs_radius), min(self.width, x + obs_radius + 1)
        y_min, y_max = max(0, y - obs_radius), min(self.height, y + obs_radius + 1)
        local_land = self.land_cover[y_min:y_max, x_min:x_max]

        # Count land cover types
        land_type_counts = [
            np.sum(local_land == 0),  # water
            np.sum(local_land == 1),  # forest
            np.sum(local_land == 2),  # urban
            np.sum(local_land == 3),  # agriculture
        ]

        # Calculate distance to nearest point of interest
        min_distance = float("inf")
        for poi_x, poi_y, _ in self.points_of_interest:
            dist = np.sqrt((x - poi_x) ** 2 + (y - poi_y) ** 2)
            min_distance = min(min_distance, dist)

        # Normalize and combine into observation vector
        obs = np.zeros(5)
        obs[:4] = np.array(land_type_counts) / sum(land_type_counts)
        obs[4] = min(1.0, 1.0 / (min_distance + 1))  # Higher value when closer

        return obs

    def take_action(self, action_idx: int) -> tuple[np.ndarray, float]:
        """
        Execute an action and return observation and reward.

        Actions:
        0: Move North
        1: Move South
        2: Move East
        3: Move West
        4: Sample current location

        Args:
            action_idx: Index of the action to take

        Returns:
            Tuple of (observation, reward)
        """
        x, y = self.agent_position

        # Move agent based on action
        if action_idx == 0 and y > 0:  # North
            self.agent_position[1] -= 1
        elif action_idx == 1 and y < self.height - 1:  # South
            self.agent_position[1] += 1
        elif action_idx == 2 and x < self.width - 1:  # East
            self.agent_position[0] += 1
        elif action_idx == 3 and x > 0:  # West
            self.agent_position[0] -= 1
        elif action_idx == 4:  # Sample
            pass  # No movement

        # Get new position
        x, y = self.agent_position

        # Calculate reward
        reward = 0.0

        # Small negative reward for all actions (cost of taking action)
        reward -= 0.01

        # Reward for being close to points of interest
        for poi_x, poi_y, poi_reward in self.points_of_interest:
            dist = np.sqrt((x - poi_x) ** 2 + (y - poi_y) ** 2)
            if dist < 1.5:  # Close enough to "discover" the point
                reward += poi_reward

        # Land cover specific rewards
        land_type = int(self.land_cover[y, x])
        if land_type == 0:  # water
            reward -= 0.05  # penalty for water
        elif land_type == 2:  # urban
            reward += 0.02  # small bonus for urban

        # Get observation at new position
        observation = self.get_observation(self.agent_position)

        # Record observation
        self.observation_history.append(
            (self.agent_position.copy(), observation, reward)
        )

        return observation, reward

    def visualize(self, save_path: str = None):
        """
        Visualize the environment and agent's path.

        Args:
            save_path: Path to save the visualization image, None for display
        """
        plt.figure(figsize=(10, 8))

        # Plot land cover
        cmap = plt.cm.get_cmap("viridis", 4)
        plt.imshow(self.land_cover, cmap=cmap, origin="lower")
        plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5], label="Land Cover Type")
        plt.clim(0, 4)

        # Plot points of interest
        for x, y, reward in self.points_of_interest:
            size = reward * 100
            plt.scatter(x, y, c="red", s=size, marker="*", edgecolor="black")

        # Plot agent path
        if self.observation_history:
            path_x = [pos[0] for pos, _, _ in self.observation_history]
            path_y = [pos[1] for pos, _, _ in self.observation_history]
            plt.plot(path_x, path_y, "b-", linewidth=1.5, alpha=0.7)
            plt.scatter(path_x, path_y, c="blue", s=10)
            plt.scatter(
                path_x[-1], path_y[-1], c="blue", s=50, marker="o", edgecolor="black"
            )

        plt.title("Geospatial Environment and Agent Path")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()


class GeospatialActiveInferenceAgent(ActiveInferenceAgent):
    """Extended Active Inference Agent for geospatial tasks."""

    def __init__(self, agent_id=None, config=None):
        """Initialize the geospatial agent."""
        super().__init__(agent_id=agent_id, config=config)

        # Add geospatial-specific configuration
        self.config.setdefault(
            "observation_dimensions", 5
        )  # Land cover (4) + POI distance (1)
        self.config.setdefault("control_dimensions", 5)  # N, S, E, W, Sample

        # Environment reference (set later)
        self.environment = None

    async def _handle_move_action(
        self, agent, action: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle movement actions in the environment.

        Args:
            agent: The agent instance
            action: Action dictionary

        Returns:
            Result dictionary
        """
        if not self.environment:
            return {"status": "failed", "reason": "No environment connected"}

        action_idx = action.get("action_idx", 0)

        # Take action in environment
        observation, reward = self.environment.take_action(action_idx)

        # Update agent's model with new observation
        self.state.update_with_observation(observation)

        # Record the action and reward
        self.state.record_action(action_idx, reward)

        # Update preferences based on reward
        preferred_obs = np.zeros(self.config["observation_dimensions"])
        if reward > 0:
            # If positive reward, set preference toward current observation
            preferred_obs = observation
        self.state.update_preferences(preferred_obs)

        return {
            "status": "success",
            "observation": observation.tolist(),
            "reward": reward,
            "position": self.environment.agent_position.copy(),
        }

    def _convert_action_index_to_action(self, action_idx: int) -> dict[str, Any]:
        """Convert action index to action dictionary."""
        action_names = ["move_north", "move_south", "move_east", "move_west", "sample"]

        return {
            "type": "move",
            "action_idx": action_idx,
            "action_name": action_names[action_idx]
            if action_idx < len(action_names)
            else "unknown",
        }

    async def explore(self, steps: int) -> list[dict[str, Any]]:
        """
        Run an exploration sequence for a given number of steps.

        Args:
            steps: Number of exploration steps

        Returns:
            List of action results
        """
        results = []

        for i in range(steps):
            # Perceive (updates the agent's beliefs in place)
            await self.perceive()
            logger.info(f"Step {i + 1}/{steps}: Perceived environment")

            # Decide
            action = await self.decide()
            if not action:
                logger.warning("No action selected, stopping exploration")
                break

            logger.info(f"Selected action: {action['action_name']}")

            # Act
            result = await self.act(action)
            logger.info(f"Action result: reward = {result.get('reward', 0)}")

            results.append(result)

            # Optional small delay for visualization
            await asyncio.sleep(0.1)

        return results


async def main():
    """Run the geospatial agent example."""
    # Create environment
    env = GeospatialEnvironment(width=30, height=30, seed=42)

    # Configure agent
    config = {
        "state_dimensions": 10,
        "observation_dimensions": 5,
        "control_dimensions": 5,
        "learning_rate": 0.1,
        "planning_horizon": 2,
    }

    # Create and initialize agent
    agent = GeospatialActiveInferenceAgent(agent_id="geo_explorer", config=config)
    await agent.initialize()

    # Connect agent to environment
    agent.environment = env

    # Register custom action handler
    agent._action_handlers["move"] = agent._handle_move_action

    # Initial observation
    initial_obs = env.get_observation(env.agent_position)
    agent.state.update_with_observation(initial_obs)

    logger.info("Starting exploration sequence...")

    # Run exploration
    results = await agent.explore(steps=50)

    # Visualize environment and agent path
    env.visualize(save_path="geospatial_exploration.png")

    # Report statistics
    total_reward = sum(r.get("reward", 0) for r in results)
    logger.info(f"Exploration complete. Total reward: {total_reward:.2f}")

    # Save agent model
    agent._save_model("geospatial_agent_model.json")
    logger.info("Saved agent model to geospatial_agent_model.json")


if __name__ == "__main__":
    asyncio.run(main())
