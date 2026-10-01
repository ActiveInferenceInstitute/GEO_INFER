"""
Multi-agent model for active inference.
"""

from typing import Any
from collections.abc import Callable
import numpy as np
import logging

from geo_infer_act.models.base import BaseActiveInferenceModel, CategoricalModel
from geo_infer_act.utils.h3_adapter import (
    get_h3_adapter,
    get_nested_h3_grid_class,
    normalize_belief_vector,
)

logger = logging.getLogger(__name__)


class MultiAgentModel(BaseActiveInferenceModel):
    """Multi-agent coordination using active inference and stigmergy."""

    def __init__(
        self,
        n_agents: int = 3,
        n_resources: int = 4,
        n_locations: int = 5,
        planning_horizon: int = 10,
        config: dict[str, Any] | None = None,
        environmental_engine: Any | None = None,
        random_seed: int | None = None,
    ):
        super().__init__(config)
        if random_seed is None and config:
            random_seed = config.get("random_seed")
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        for name, value in (
            ("n_agents", n_agents),
            ("n_resources", n_resources),
            ("n_locations", n_locations),
            ("planning_horizon", planning_horizon),
        ):
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if n_resources == 0 or n_locations == 0:
            raise ValueError("n_resources and n_locations must be positive")
        self.n_agents = int(n_agents)
        self.n_resources = int(n_resources)
        self.n_locations = int(n_locations)
        self.planning_horizon = int(planning_horizon)
        self.environmental_engine = environmental_engine

        # Initialize agent models with enhanced active inference capabilities
        self.agent_models = []
        for i in range(self.n_agents):
            agent = CategoricalModel(
                state_dim=4, obs_dim=4
            )  # 4-state environmental model
            # Set up proper observation and transition models
            agent.set_likelihood_matrix(self._create_environmental_observation_model())
            agent.set_transition_matrix(self._create_environmental_transition_model())
            self.agent_models.append(agent)

        self.resource_distribution = self.rng.random(
            (self.n_resources, self.n_locations)
        )
        self.location_connectivity = np.eye(
            self.n_locations
        )  # Example, adjust as needed
        self.agent_preferences = self.rng.random((self.n_agents, self.n_resources))
        self._initial_resource_distribution = self.resource_distribution.copy()
        self._initial_agent_preferences = self.agent_preferences.copy()
        self.agent_locations = np.zeros(self.n_agents, dtype=int)
        self.step_count = 0
        self.history: list[dict[str, Any]] = []

        # H3 spatial properties
        self.spatial_mode = False
        self.h3_cells: list[str] = []
        self.h3_resolution = 8
        self.spatial_graph: dict[int, list[int]] = {}
        self.agent_location_map: dict[str, int] = {}

    def reset(self) -> None:
        """Restore deterministic initial resource and preference state."""
        self.resource_distribution = self._initial_resource_distribution.copy()
        self.agent_preferences = self._initial_agent_preferences.copy()
        for agent in self.agent_models:
            reset = getattr(agent, "reset", None)
            if callable(reset):
                reset()
        self.agent_locations = np.zeros(len(self.agent_models), dtype=int)
        self.step_count = 0
        self.history = []
        for agent in self.agent_models:
            agent.location = 0

    def _create_environmental_observation_model(self) -> np.ndarray:
        """Create realistic environmental observation model for agents."""
        # Observation model: P(observation | environmental_state)
        # States: [poor, fair, good, excellent] environmental quality
        # Observations: [low_temp, med_temp, high_temp, vegetation] indicators

        obs_model = np.array(
            [
                [0.7, 0.5, 0.3, 0.1],  # Low temperature observation
                [0.2, 0.3, 0.4, 0.3],  # Medium temperature observation
                [0.1, 0.2, 0.3, 0.6],  # High temperature observation
                [0.1, 0.3, 0.6, 0.8],  # Vegetation density observation
            ]
        )

        # Normalize columns (each state sums to 1)
        normed: np.ndarray = obs_model / obs_model.sum(axis=0, keepdims=True)
        return normed

    def _create_environmental_transition_model(self) -> np.ndarray:
        """Create environmental state transition model."""
        # Transition model: P(next_state | current_state)
        # Environmental states tend to persist with some probability of change

        transition_model = np.array(
            [
                [0.7, 0.2, 0.05, 0.05],  # From poor state
                [0.2, 0.6, 0.15, 0.05],  # From fair state
                [0.05, 0.15, 0.6, 0.2],  # From good state
                [0.05, 0.05, 0.2, 0.7],  # From excellent state
            ]
        )

        return transition_model

    def step(
        self, actions: list[dict[str, Any]] | None = None
    ) -> tuple[dict[str, Any], bool]:
        """Run one multi-agent perception, action, and resource step.

        Action dictionaries may contain ``agent_id``, ``location`` (or
        ``move_to``), ``resource``, ``amount``/``harvest``, and an optional
        four-element ``observation``.  Missing observations are derived from
        the resource profile at the agent's current location. When no explicit
        actions are supplied, agents advance using those local observations
        without changing resources or locations.
        """
        if actions is not None and not isinstance(actions, (list, tuple)):
            raise ValueError("actions must be a sequence of action mappings")
        action_by_agent: dict[int, dict[str, Any]] = {}
        for position, action in enumerate(actions or []):
            if not isinstance(action, dict):
                raise ValueError("each multi-agent action must be a mapping")
            agent_id = int(action.get("agent_id", position))
            if agent_id < 0 or agent_id >= len(self.agent_models):
                raise ValueError(
                    f"agent_id {agent_id} is outside the active agent range"
                )
            action_by_agent[agent_id] = dict(action)

        harvest_yield = np.zeros_like(self.resource_distribution, dtype=float)
        beliefs = []
        free_energies = []
        chosen_actions = []

        for agent_id, agent in enumerate(self.agent_models):
            current_location = int(self.agent_locations[agent_id])
            action = action_by_agent.get(agent_id, {})
            observation = action.get("observation")
            if observation is None:
                observation = self._observation_for_location(current_location)
            observation = np.asarray(observation, dtype=float).reshape(-1)
            if observation.shape != (agent.obs_dim,):
                raise ValueError(
                    f"agent {agent_id} observation must have shape ({agent.obs_dim},)"
                )
            updated = agent.update_beliefs(observation)
            beliefs.append(updated.copy())
            free_energies.append(float(agent.compute_free_energy()))

            requested_location = action.get("location", action.get("move_to"))
            if requested_location is not None:
                requested_location = int(requested_location)
                if not 0 <= requested_location < self.n_locations:
                    raise ValueError(
                        f"location {requested_location} is outside [0, {self.n_locations})"
                    )
                self.agent_locations[agent_id] = requested_location
                current_location = requested_location
            agent.location = current_location

            resource = action.get("resource")
            amount = action.get("amount", action.get("harvest", 0.0))
            if resource is not None:
                resource = int(resource)
                amount = float(amount)
                if not 0 <= resource < self.n_resources:
                    raise ValueError(
                        f"resource {resource} is outside [0, {self.n_resources})"
                    )
                if not np.isfinite(amount) or amount < 0:
                    raise ValueError("harvest amount must be finite and non-negative")
                yield_value = min(
                    amount, self.resource_distribution[resource, current_location]
                )
                self.resource_distribution[resource, current_location] -= yield_value
                harvest_yield[resource, current_location] = yield_value
            chosen_actions.append(action.copy())

        self.step_count += 1
        state = {
            "resource_distribution": self.resource_distribution.copy(),
            "agent_locations": self.agent_locations.tolist(),
            "beliefs": [belief.tolist() for belief in beliefs],
            "free_energy": free_energies,
            "actions": chosen_actions,
            "harvest_yield": harvest_yield,
            "total_resources": float(np.sum(self.resource_distribution)),
            "step": self.step_count,
        }
        self.history.append(state)
        done = self.planning_horizon > 0 and self.step_count >= self.planning_horizon
        return state, done

    def _observation_for_location(self, location: int) -> np.ndarray:
        """Build a normalized four-state observation from local resources."""
        if not 0 <= location < self.resource_distribution.shape[1]:
            return np.ones(4, dtype=float) / 4.0
        resource_level = float(np.mean(self.resource_distribution[:, location]))
        resource_level = float(np.clip(resource_level, 0.0, 1.0))
        observation = np.array(
            [
                1.0 - resource_level,
                0.5 + 0.5 * resource_level,
                resource_level,
                resource_level**2,
            ],
            dtype=float,
        )
        return np.asarray(normalize_belief_vector(observation), dtype=float)

    def enable_h3_spatial(self, resolution: int, boundary: dict[str, Any]) -> None:
        """
        Enable H3 spatial modeling for multi-agent active inference.

        A real H3 v4 cell set is created from the supplied boundary, each cell
        receives a categorical active-inference agent, and a neighbor graph is
        built from H3 ring adjacency for distributed belief coordination.
        """
        from geo_infer_act.utils.integration import create_h3_spatial_model

        try:
            adapter = get_h3_adapter()
            result = create_h3_spatial_model({}, resolution, boundary)
            if result["status"] == "success":
                self.spatial_mode = True
                self.h3_resolution = resolution
                self.h3_cells = result["model_config"]["boundary_cells"]
                if not self.h3_cells:
                    raise ValueError("H3 spatial model did not produce any cells")
                self.h3_cells = adapter.validate_cells(self.h3_cells)
                self.n_locations = len(self.h3_cells)

                # Create one agent per H3 cell for distributed spatial inference
                self.agent_models = []
                for i, cell in enumerate(self.h3_cells):
                    agent = CategoricalModel(state_dim=4, obs_dim=4)
                    agent.cell_id = cell
                    agent.spatial_index = i

                    # Set up environmental models
                    agent.set_likelihood_matrix(
                        self._create_environmental_observation_model()
                    )
                    agent.set_transition_matrix(
                        self._create_environmental_transition_model()
                    )

                    # Initialize with spatial-dependent priors
                    try:
                        lat, lng = adapter.cell_to_latlng(cell)
                        if self.environmental_engine is not None and hasattr(
                            self.environmental_engine, "compute_spatial_priors"
                        ):
                            # Use true Moran's I weighted spatial priors from the environmental engine
                            try:
                                priors_map = (
                                    self.environmental_engine.compute_spatial_priors(
                                        "vegetation_density", 4
                                    )
                                )
                                initial_beliefs = priors_map.get(cell, np.ones(4) / 4)
                            except Exception as e:
                                logger.warning(f"Failed to fetch Moran prior: {e}")
                                initial_beliefs = np.ones(4) / 4
                        else:
                            # Create location-dependent initial beliefs from coordinates.
                            spatial_bias = np.array(
                                [0.1, 0.2, 0.4, 0.3]
                            )  # Slight bias toward good states
                            # Add some spatial variation based on coordinates
                            spatial_variation = (
                                0.1 * np.sin(lat * 10) * np.cos(lng * 10)
                            )
                            initial_beliefs = spatial_bias + spatial_variation
                            initial_beliefs = normalize_belief_vector(initial_beliefs)

                        agent.beliefs = initial_beliefs
                    except Exception as e:
                        logger.warning(
                            "Spatial belief initialization used uniform beliefs: %s",
                            e,
                        )
                        agent.beliefs = np.ones(4) / 4

                    self.agent_models.append(agent)

                # Create spatial coordination graph
                self._create_spatial_coordination_graph()
                self.agent_locations = np.zeros(len(self.agent_models), dtype=int)
                for agent in self.agent_models:
                    agent.location = 0

                logger.info(
                    f"Enabled H3 spatial mode with {self.n_locations} cells and {len(self.agent_models)} agents"
                )
            else:
                raise RuntimeError(
                    f"H3 spatial initialization failed: {result['message']}"
                )
        except Exception as e:
            logger.error(f"Failed to enable H3 spatial mode: {e}")
            raise

    def enable_nested_h3_spatial(
        self,
        resolutions: list[int],
        boundary: dict[str, Any] | None = None,
        cells: list[str] | None = None,
        top_down_weight: float = 0.15,
    ) -> dict[str, Any]:
        """
        Enable nested H3 multi-agent modeling over finest-resolution leaf cells.

        Parent H3 cells summarize child-agent beliefs; agents remain attached to
        leaf cells so existing flat H3 coordination behavior stays intact.
        """
        if not boundary and not cells:
            raise ValueError("Provide either a boundary or H3 cells")
        if not 0.0 <= float(top_down_weight) <= 1.0:
            raise ValueError("top_down_weight must be between 0.0 and 1.0")
        NestedH3Grid = get_nested_h3_grid_class()
        grid = NestedH3Grid(name="multi_agent_nested_h3")
        if boundary is not None:
            hierarchy = grid.build_h3_hierarchy_from_boundary(boundary, resolutions)
        else:
            hierarchy = grid.build_h3_hierarchy_from_cells(cells or [], resolutions)
        validation = hierarchy.get("validation", {})
        if not validation.get("is_valid", False):
            raise ValueError(f"Invalid nested H3 hierarchy: {validation}")

        adapter = get_h3_adapter()
        self.nested_h3_mode = True
        self.nested_h3_grid = grid
        self.nested_h3_hierarchy = hierarchy
        self.nested_h3_resolutions = [int(value) for value in hierarchy["resolutions"]]
        self.nested_h3_top_down_weight = float(top_down_weight)
        self.spatial_mode = True
        self.h3_resolution = self.nested_h3_resolutions[-1]
        self.h3_cells = adapter.validate_cells(hierarchy["leaf_cells"])
        self.n_locations = len(self.h3_cells)
        self.agent_models = []
        for i, cell in enumerate(self.h3_cells):
            agent = CategoricalModel(state_dim=4, obs_dim=4)
            agent.spatial_index = i
            agent.set_likelihood_matrix(self._create_environmental_observation_model())
            agent.set_transition_matrix(self._create_environmental_transition_model())
            try:
                lat, lng = adapter.cell_to_latlng(cell)
                spatial_bias = np.array([0.1, 0.2, 0.4, 0.3])
                spatial_variation = 0.1 * np.sin(lat * 10) * np.cos(lng * 10)
                agent.beliefs = normalize_belief_vector(
                    spatial_bias + spatial_variation
                )
            except Exception as e:
                logger.warning(
                    "Spatial belief initialization used uniform beliefs: %s", e
                )
                agent.beliefs = np.ones(4) / 4
            self.agent_models.append(agent)
        self._create_spatial_coordination_graph()
        self.agent_locations = np.zeros(len(self.agent_models), dtype=int)
        for agent in self.agent_models:
            agent.location = 0
        return dict(hierarchy)

    def _create_spatial_coordination_graph(self) -> None:
        """Create coordination graph between spatially neighboring agents."""
        if not self.spatial_mode or not self.h3_cells:
            return

        adapter = get_h3_adapter()
        self.spatial_graph = {}
        cell_indices = {cell: index for index, cell in enumerate(self.h3_cells)}

        for i, cell in enumerate(self.h3_cells):
            neighbors = []
            try:
                h3_neighbors = adapter.grid_ring(cell, 1)
                valid_neighbors = set(h3_neighbors) & set(cell_indices)

                for neighbor_cell in valid_neighbors:
                    if neighbor_cell in cell_indices:
                        neighbor_idx = cell_indices[neighbor_cell]
                        neighbors.append(neighbor_idx)

            except Exception as e:
                logger.debug("Failed to get H3 neighbors for cell %s: %s", cell, e)

            self.spatial_graph[i] = neighbors

    def simulate_h3_lattice(
        self, timesteps: int, obs_gen: Callable[[str], np.ndarray]
    ) -> list[dict[str, dict]]:
        """
        Simulate active inference on H3 lattice with proper perception-action loops.

        Each timestep generates one normalized observation vector per H3 cell,
        updates the corresponding agent beliefs, records finite free-energy
        diagnostics, and then performs neighbor-based belief coordination or
        environmental stigmergy when an environmental engine is attached.

        Args:
            timesteps: Number of simulation timesteps
            obs_gen: Function that generates observations for each H3 cell

        Returns:
            History of simulation states
        """
        if not self.spatial_mode:
            raise ValueError("Enable H3 spatial mode first")
        if isinstance(timesteps, bool) or int(timesteps) != timesteps or timesteps < 0:
            raise ValueError("timesteps must be a non-negative integer")
        if not callable(obs_gen):
            raise ValueError("obs_gen must be callable")

        history = []

        for t in range(timesteps):
            step_data = {}

            # Perception phase: each agent updates beliefs based on observations
            for i, (cell, agent) in enumerate(zip(self.h3_cells, self.agent_models)):
                # Generate environmental observation for this cell
                obs = np.asarray(obs_gen(cell), dtype=float).reshape(-1)
                if obs.size == 0 or not np.all(np.isfinite(obs)):
                    raise ValueError(
                        f"Observation generator returned invalid data for {cell}"
                    )
                if np.any(obs < 0):
                    raise ValueError(
                        f"Observation generator returned negative values for {cell}"
                    )

                # Ensure observation is properly formatted (4-dimensional for our model)
                if obs.size != 4:
                    # Convert to 4D observation vector
                    obs_4d = np.zeros(4)
                    for j in range(min(len(obs), 4)):
                        obs_4d[j] = obs[j]
                    obs = obs_4d

                # Normalize observation to probability distribution
                obs = normalize_belief_vector(obs)

                # Update agent beliefs using Bayesian inference
                try:
                    updated_beliefs = agent.update_beliefs(obs)

                    # Compute free energy for this agent
                    free_energy = agent.compute_free_energy()

                    # Store agent state
                    step_data[cell] = {
                        "beliefs": updated_beliefs.tolist(),
                        "observations": obs.tolist(),
                        "free_energy": free_energy,
                        "agent_index": i,
                    }

                except Exception as e:
                    raise RuntimeError(
                        f"Failed to update beliefs for agent {i} at cell {cell}: {e}"
                    ) from e

            # Action/Coordination phase: agents coordinate and influence each other
            self._spatial_belief_coordination(step_data)

            history.append(step_data)

            if t % 5 == 0:
                logger.debug(f"H3 simulation timestep {t}/{timesteps} completed")

        logger.info(
            f"Completed H3 lattice simulation: {timesteps} timesteps, {len(self.h3_cells)} cells"
        )
        return history

    def simulate_nested_h3_lattice(
        self,
        timesteps: int,
        obs_gen: Callable[[str], np.ndarray],
        top_down_weight: float | None = None,
    ) -> dict[str, Any]:
        """
        Simulate H3 leaf agents and return nested parent summaries per timestep.
        """
        if not getattr(self, "nested_h3_mode", False):
            raise ValueError("Enable nested H3 spatial mode first")
        history = self.simulate_h3_lattice(timesteps, obs_gen)
        nested_history = []
        for timestep, step_data in enumerate(history):
            leaf_beliefs = {
                cell: np.asarray(payload["beliefs"], dtype=float)
                for cell, payload in step_data.items()
            }
            parent_beliefs = self._aggregate_nested_agent_beliefs(
                leaf_beliefs,
                top_down_weight=top_down_weight,
            )
            nested_history.append(
                {
                    "timestep": timestep,
                    "leaf_cell_count": len(leaf_beliefs),
                    "parent_beliefs": {
                        cell: belief.tolist() for cell, belief in parent_beliefs.items()
                    },
                    "level_summaries": self._nested_agent_level_summaries(
                        leaf_beliefs,
                        parent_beliefs,
                    ),
                }
            )
        return {
            "history": history,
            "nested_history": nested_history,
            "resolutions": list(self.nested_h3_resolutions),
            "leaf_cell_count": len(self.h3_cells),
            "parent_count": len(self.nested_h3_hierarchy.get("parent_child_map", {})),
        }

    def _aggregate_nested_agent_beliefs(
        self,
        leaf_beliefs: dict[str, np.ndarray],
        top_down_weight: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Aggregate leaf-agent beliefs to every configured parent resolution."""
        adapter = get_h3_adapter()
        parent_beliefs: dict[str, np.ndarray] = {}
        for target_resolution in reversed(self.nested_h3_resolutions[:-1]):
            grouped: dict[str, list[np.ndarray]] = {}
            for child, belief in leaf_beliefs.items():
                parent = adapter.cell_to_parent(child, target_resolution)
                grouped.setdefault(parent, []).append(normalize_belief_vector(belief))
            for parent, beliefs in grouped.items():
                parent_beliefs[parent] = normalize_belief_vector(
                    np.mean(beliefs, axis=0)
                )

        weight = (
            float(top_down_weight)
            if top_down_weight is not None
            else float(self.nested_h3_top_down_weight)
        )
        if weight > 0.0:
            child_parent_map = self.nested_h3_hierarchy.get("child_parent_map", {})
            for i, cell in enumerate(self.h3_cells):
                parent = child_parent_map.get(cell)
                if parent in parent_beliefs and i < len(self.agent_models):
                    current = normalize_belief_vector(self.agent_models[i].beliefs)
                    self.agent_models[i].beliefs = normalize_belief_vector(
                        ((1.0 - weight) * current) + (weight * parent_beliefs[parent])
                    )
        return dict(sorted(parent_beliefs.items()))

    def _nested_agent_level_summaries(
        self,
        leaf_beliefs: dict[str, np.ndarray],
        parent_beliefs: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        """Return per-resolution nested multi-agent belief summaries."""
        adapter = get_h3_adapter()
        by_level: dict[int, dict[str, np.ndarray]] = {}
        for cell, belief in parent_beliefs.items():
            by_level.setdefault(adapter.get_resolution(cell), {})[cell] = belief
        for cell, belief in leaf_beliefs.items():
            by_level.setdefault(adapter.get_resolution(cell), {})[cell] = belief
        summaries = []
        for resolution in self.nested_h3_resolutions:
            beliefs = by_level.get(resolution, {})
            if beliefs:
                entropy_values = [
                    float(
                        -np.sum(
                            normalize_belief_vector(b)
                            * np.log(normalize_belief_vector(b) + 1e-12)
                        )
                    )
                    for b in beliefs.values()
                ]
            else:
                entropy_values = [0.0]
            summaries.append(
                {
                    "resolution": int(resolution),
                    "cell_count": len(beliefs),
                    "mean_entropy": float(np.mean(entropy_values)),
                }
            )
        return summaries

    def _spatial_belief_coordination(self, step_data: dict[str, dict]) -> None:
        """Coordinate agents through environmental updates or spatial belief sharing."""
        if not self.spatial_graph:
            return

        # True Stigmergic Communication:
        # Agents modify the shared environmental manifold directly (stigmergy).
        if self.environmental_engine is not None:
            self._apply_stigmergy(step_data)
            # Beliefs will be naturally updated in the next perception cycle via `obs_gen`
            return

        coordination_strength = 0.1  # How much neighbors influence each other

        # Create a copy of current beliefs for simultaneous update
        updated_beliefs = {}

        for i, cell in enumerate(self.h3_cells):
            if cell not in step_data:
                continue

            current_beliefs = np.array(step_data[cell]["beliefs"])
            neighbor_indices = self.spatial_graph.get(i, [])

            if neighbor_indices:
                # Aggregate neighbor beliefs
                neighbor_beliefs = []
                for neighbor_idx in neighbor_indices:
                    if neighbor_idx < len(self.h3_cells):
                        neighbor_cell = self.h3_cells[neighbor_idx]
                        if neighbor_cell in step_data:
                            neighbor_beliefs.append(
                                np.array(step_data[neighbor_cell]["beliefs"])
                            )

                if neighbor_beliefs:
                    avg_neighbor_belief = np.mean(neighbor_beliefs, axis=0)

                    # Coordinate beliefs with neighbors
                    coordinated_beliefs = (
                        1 - coordination_strength
                    ) * current_beliefs + coordination_strength * avg_neighbor_belief
                    coordinated_beliefs = coordinated_beliefs / (
                        np.sum(coordinated_beliefs) + 1e-8
                    )

                    updated_beliefs[cell] = coordinated_beliefs
                else:
                    updated_beliefs[cell] = current_beliefs
            else:
                updated_beliefs[cell] = current_beliefs

        # Update step data with coordinated beliefs
        for cell, new_beliefs in updated_beliefs.items():
            if cell in step_data:
                step_data[cell]["beliefs"] = new_beliefs.tolist()

                # Update agent model beliefs for next timestep
                agent_idx = step_data[cell]["agent_index"]
                if agent_idx < len(self.agent_models):
                    self.agent_models[agent_idx].beliefs = new_beliefs

    def _apply_stigmergy(self, step_data: dict[str, dict]) -> None:
        """Apply stigmergic pheromone modification to the environmental engine."""
        if self.environmental_engine is None:
            return
        engine = self.environmental_engine
        stigmergy_strength = 0.05
        current_time = 0.0  # Could be synced with global time if needed

        observations_to_push = {}
        for cell, data in step_data.items():
            if cell in engine.environmental_states:
                # Based on the agent's beliefs (e.g. they believe it's a good state), they modify the environment
                # E.g. high belief in state index 3 (excellent)
                belief_tensor = np.array(data["beliefs"])
                # Stigmergic traces accumulate: an agent with weak positive
                # evidence should not erase activity deposited by earlier
                # agents merely because its current belief favors state 0.
                positive_pull = max(float(belief_tensor[3] - belief_tensor[0]), 0.0)

                # We modify standard fields to reflect the presence/activity of agents
                # Example: human_activity increases, which can subsequently influence observations
                current_activity = getattr(
                    engine.environmental_states[cell],
                    "human_activity",
                    0.0,
                )
                new_activity = np.clip(
                    current_activity + (positive_pull * stigmergy_strength), 0.0, 1.0
                )

                observations_to_push[cell] = {"human_activity": new_activity}

        # Push all modifications back to the environmental engine as new observations
        if observations_to_push:
            engine.observe_environment(observations_to_push, timestamp=current_time)

    def coordinate_agents(self) -> dict[str, Any]:
        """
        Coordinate agents through message passing and shared information.

        Returns:
            Coordination results including coherence metrics
        """
        if not hasattr(self, "spatial_mode") or not self.spatial_mode:
            # Simple coordination for non-spatial case
            coordination_matrix = self.rng.random((self.n_agents, self.n_agents))
            coordination_matrix = (
                coordination_matrix + coordination_matrix.T
            ) / 2  # Make symmetric
            np.fill_diagonal(coordination_matrix, 1.0)  # Perfect self-coordination

            return {
                "coordination_matrix": coordination_matrix,
                "average_coordination": np.mean(
                    coordination_matrix[np.triu_indices_from(coordination_matrix, k=1)]
                ),
                "coordination_variance": np.var(
                    coordination_matrix[np.triu_indices_from(coordination_matrix, k=1)]
                ),
            }

        # Spatial coordination using H3 cells
        n_cells = len(self.h3_cells)
        coordination_matrix = np.eye(n_cells)  # Start with identity

        # Add spatial coordination based on H3 neighbor relationships.
        adapter = get_h3_adapter()
        h3_cells = adapter.validate_cells(self.h3_cells)
        for i, cell_i in enumerate(h3_cells):
            neighbors = set(adapter.grid_ring(cell_i, 1))
            for j, cell_j in enumerate(h3_cells):
                if cell_j in neighbors and i != j:
                    agent_i = self.agent_models[i]
                    agent_j = self.agent_models[j]

                    if hasattr(agent_i, "beliefs") and hasattr(agent_j, "beliefs"):
                        belief_i = normalize_belief_vector(agent_i.beliefs)
                        belief_j = normalize_belief_vector(agent_j.beliefs)
                        belief_similarity = 1.0 / (
                            1.0 + np.linalg.norm(belief_i - belief_j)
                        )
                        coordination_matrix[i, j] = belief_similarity
                    else:
                        coordination_matrix[i, j] = 0.5

        return {
            "coordination_matrix": coordination_matrix,
            "average_coordination": np.mean(
                coordination_matrix[np.triu_indices_from(coordination_matrix, k=1)]
            ),
            "coordination_variance": np.var(
                coordination_matrix[np.triu_indices_from(coordination_matrix, k=1)]
            ),
            "n_coordinated_agents": n_cells,
        }

    def score_spatial_information_gain(
        self, target_resolution: int | None = None
    ) -> dict[str, Any]:
        """
        Score the H3 spatial grid by expected information gain for active
        sensing, aggregating agent belief uncertainty across H3 resolutions.

        Each leaf agent's score is its belief entropy normalized by the
        maximum entropy of a uniform distribution over its state space.  When
        ``target_resolution`` is given, leaf scores are averaged up to their
        coarser H3 parent cells so the swarm can decide where sensing or
        coordination effort should concentrate.

        Args:
            target_resolution: Optional coarser H3 resolution to aggregate to.

        Returns:
            Dict with ``scores``, ``best_cells``, ``best_score``,
            ``count_cells``, ``mean_score``, ``resolution`` and
            ``uncertain_cell_fraction``.
        """
        if not self.spatial_mode or not self.h3_cells:
            return {
                "scores": {},
                "best_cells": [],
                "count_cells": 0,
                "mean_score": 0.0,
                "resolution": int(self.h3_resolution),
                "uncertain_cell_fraction": 0.0,
            }
        adapter = get_h3_adapter()
        state_dim = 4
        max_entropy = float(np.log(state_dim))
        per_cell: dict[str, float] = {}
        for cell, agent in zip(self.h3_cells, self.agent_models):
            vec = normalize_belief_vector(np.asarray(agent.beliefs, dtype=float))
            if vec.size != state_dim:
                vec = np.resize(vec, state_dim)
            entropy = float(-np.sum(vec * np.log(vec + 1e-12)))
            per_cell[str(cell)] = float(np.clip(entropy / max_entropy, 0.0, 1.0))

        resolution = int(self.h3_resolution)
        scores = per_cell
        if target_resolution is not None:
            try:
                grouped: dict[str, list[float]] = {}
                for cell, score in per_cell.items():
                    cell_res = adapter.get_resolution(cell)
                    parent = (
                        adapter.cell_to_parent(cell, target_resolution)
                        if cell_res > target_resolution
                        else cell
                    )
                    grouped.setdefault(parent, []).append(score)
                scores = {
                    parent: float(np.mean(values)) for parent, values in grouped.items()
                }
                resolution = int(target_resolution)
            except Exception as exc:
                logger.debug("Multi-agent info-gain aggregation failed: %s", exc)
                resolution = int(self.h3_resolution)
        else:
            resolution = int(self.h3_resolution)

        best = (
            sorted(scores.keys(), key=lambda cell: float(scores[cell]), reverse=True)
            if scores
            else []
        )
        values = list(scores.values()) if scores else [0.0]
        return {
            "scores": scores,
            "best_cells": best,
            "best_score": float(scores[best[0]]) if best else 0.0,
            "count_cells": len(scores),
            "resolution": resolution,
            "mean_score": float(np.mean(values)),
            "uncertain_cell_fraction": float(
                np.mean([1.0 if value > 0.5 else 0.0 for value in values])
            ),
        }

    def get_agent_messages(self, agent_id: int) -> dict[str, Any]:
        """Get messages for inter-agent communication from the specified agent.

        Returns a snapshot of the agent's current beliefs, position, and
        recent interactions for use in multi-agent coordination protocols.

        Args:
            agent_id: Integer index of the agent

        Returns:
            Dict with 'beliefs', 'position', 'last_action', 'free_energy',
            and 'timestamp' keys. Empty dict if agent_id is invalid.
        """
        if agent_id < 0 or agent_id >= len(self.agent_models):
            logger.warning(
                f"Invalid agent_id {agent_id}, valid range [0, {len(self.agent_models)})"
            )
            return {}

        agent = self.agent_models[agent_id]
        message: dict[str, Any] = {
            "agent_id": agent_id,
            "beliefs": (
                agent.beliefs.tolist()
                if hasattr(agent, "beliefs") and agent.beliefs is not None
                else []
            ),
        }

        # Include spatial position if H3 mode is active
        if (
            hasattr(self, "spatial_mode")
            and self.spatial_mode
            and agent_id < len(self.h3_cells)
        ):
            message["position"] = self.h3_cells[agent_id]

        # Include recent action info from history
        if hasattr(self, "history") and self.history:
            last_step = self.history[-1]
            if isinstance(last_step, dict):
                cell_key = (
                    self.h3_cells[agent_id]
                    if hasattr(self, "h3_cells") and agent_id < len(self.h3_cells)
                    else str(agent_id)
                )
                if cell_key in last_step:
                    cell_data = last_step[cell_key]
                    message["last_action"] = cell_data.get("action")
                    message["free_energy"] = cell_data.get("free_energy")

        return message
