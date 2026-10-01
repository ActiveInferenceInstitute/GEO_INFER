"""
Agent Population Dynamics for GEO-INFER-ANT

This module provides comprehensive population management for swarm intelligence systems,
including agent creation, lifecycle management, behavioral coordination, and
emergent behavior analysis.

Key Features:
- Dynamic agent population management
- Spatial distribution and clustering
- Behavioral rule configuration
- Environmental interaction simulation
- Data collection and analysis
- Integration with spatial indexing systems
"""

import numpy as np
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Optional
from collections.abc import Callable
from datetime import datetime
from dataclasses import dataclass, field
import json

from geo_infer_ant.utils.spatial import validate_bounds

if TYPE_CHECKING:
    from .agent_base import SwarmAgent

# Integration imports
try:
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface
except ImportError as e:
    logging.getLogger(__name__).debug("Optional spatial integration unavailable: %s", e)
    SpatialIndexingInterface = None
    SpatialAnalyticsInterface = None

# GEO-INFER-TIME currently exposes analysis and forecasting engines rather than
# the historical TemporalManager class. Keep this integration point explicit so
# population construction never performs a misleading import-time fallback.
TemporalManager: Any = None

logger = logging.getLogger(__name__)


@dataclass
class PopulationConfig:
    """Configuration for agent population dynamics."""

    population_size: int = 1000
    agent_types: list[str] = field(
        default_factory=lambda: ["worker", "scout", "soldier"]
    )
    spatial_distribution: str = "random"  # 'random', 'clustered', 'uniform', 'custom'
    behavioral_heterogeneity: str = (
        "stochastic"  # 'stochastic', 'deterministic', 'adaptive'
    )

    # Spatial configuration
    spatial_bounds: dict[str, float] | None = None
    clustering_centers: list[np.ndarray] | None = None
    clustering_radius: float = 50.0

    # Behavioral configuration
    foraging_rules: dict[str, Any] | None = None
    communication_rules: dict[str, Any] | None = None
    adaptation_rules: dict[str, Any] | None = None

    # Simulation configuration
    time_step: float = 1.0
    max_simulation_time: float | None = None
    parallel_processing: bool = True
    max_workers: int = 4
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")

        if not self.agent_types:
            raise ValueError("At least one agent type must be specified")

        if self.spatial_distribution not in [
            "random",
            "clustered",
            "uniform",
            "custom",
        ]:
            raise ValueError("Invalid spatial distribution type")
        if self.spatial_bounds is not None:
            self.spatial_bounds = validate_bounds(self.spatial_bounds)
        if self.clustering_radius < 0 or not np.isfinite(self.clustering_radius):
            raise ValueError("clustering_radius must be finite and non-negative")
        if self.time_step <= 0 or not np.isfinite(self.time_step):
            raise ValueError("time_step must be finite and positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")


@dataclass
class EnvironmentalState:
    """Current state of the simulation environment."""

    spatial_bounds: dict[str, float]
    resource_distribution: dict[str, Any] = field(default_factory=dict)
    obstacle_map: dict[str, Any] = field(default_factory=dict)
    pheromone_diffusion: dict[str, Any] = field(default_factory=dict)
    environmental_factors: dict[str, Any] = field(default_factory=dict)

    # Dynamic environmental state
    current_time: datetime = field(default_factory=datetime.now)
    weather_conditions: dict[str, Any] | None = None
    seasonal_effects: dict[str, Any] | None = None

    def update_environmental_factors(self, factors: dict[str, Any]) -> None:
        """Update environmental factors."""
        self.environmental_factors.update(factors)
        self.current_time = datetime.now()

    def get_resource_at_location(self, location: np.ndarray) -> dict[str, Any]:
        """Get resources available at a specific location."""
        # Evaluate the configured resource fields locally so simulations remain
        # deterministic even when optional spatial analytics are unavailable.
        resources = {}

        for resource_type, distribution in self.resource_distribution.items():
            if distribution.get("type") == "spatial_field":
                # Calculate resource density at location
                resources[resource_type] = self._calculate_resource_density(
                    location, distribution
                )

        return resources

    def _calculate_resource_density(
        self, location: np.ndarray, distribution: dict[str, Any]
    ) -> float:
        """Calculate resource density at given location."""
        # Interpolate the configured radial resource field at the query point.
        centers = distribution.get("centers", [])
        if not centers:
            return 0.0

        # Find nearest resource center
        distances = [np.linalg.norm(location - np.array(center)) for center in centers]
        min_distance = min(distances)

        # Exponential decay with distance
        max_density = distribution.get("max_density", 1.0)
        decay_rate = distribution.get("decay_rate", 0.1)

        return float(max_density * np.exp(-decay_rate * min_distance))


@dataclass
class SimulationResults:
    """Results from population dynamics simulation."""

    trajectories: list[np.ndarray] = field(default_factory=list)
    interactions: list[dict[str, Any]] = field(default_factory=list)
    emergent_patterns: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, Any] = field(default_factory=dict)

    # Simulation metadata
    simulation_time: float = 0.0
    time_steps: int = 0
    population_size: int = 0

    def add_trajectory(self, step: int, positions: np.ndarray) -> None:
        """Add trajectory data for a simulation step."""
        self.trajectories.append(positions.copy())
        self.time_steps = step + 1

    def add_interaction(self, interaction: dict[str, Any]) -> None:
        """Add agent interaction data."""
        self.interactions.append(interaction.copy())

    def update_emergent_patterns(self, patterns: dict[str, Any]) -> None:
        """Update emergent pattern analysis."""
        self.emergent_patterns.update(patterns)

    def update_performance_metrics(self, metrics: dict[str, Any]) -> None:
        """Update performance metrics."""
        self.performance_metrics.update(metrics)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "trajectories": [
                traj.tolist() if hasattr(traj, "tolist") else traj
                for traj in self.trajectories
            ],
            "interactions": self.interactions,
            "emergent_patterns": self.emergent_patterns,
            "performance_metrics": self.performance_metrics,
            "simulation_time": self.simulation_time,
            "time_steps": self.time_steps,
            "population_size": self.population_size,
        }


class AgentPopulation:
    """
    Management system for collections of interacting swarm agents.

    Provides comprehensive population dynamics simulation including:
    - Agent creation and lifecycle management
    - Spatial distribution and movement
    - Behavioral coordination and interaction
    - Environmental interaction simulation
    - Data collection and analysis

    Integration Points:
    - GEO-INFER-SPACE: Spatial indexing and analytics for agent operations
    - GEO-INFER-TIME: Temporal dynamics and scheduling for simulation timing
    - GEO-INFER-AGENT: Agent lifecycle management and coordination
    """

    def __init__(
        self,
        population_size: int = 1000,
        agent_types: list[str] | None = None,
        spatial_distribution: str = "random",
        behavioral_heterogeneity: str = "stochastic",
        spatial_bounds: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize agent population.

        Args:
            population_size: Number of agents in population
            agent_types: List of agent type names
            spatial_distribution: How agents are initially distributed ('random', 'clustered', etc.)
            behavioral_heterogeneity: Level of behavioral variation ('stochastic', 'deterministic')
            spatial_bounds: Geographic bounds for agent movement
            **kwargs: Additional configuration parameters
        """
        config_kwargs = {
            key: kwargs[key]
            for key in (
                "clustering_centers",
                "clustering_radius",
                "foraging_rules",
                "communication_rules",
                "adaptation_rules",
                "time_step",
                "max_simulation_time",
                "parallel_processing",
                "max_workers",
                "random_seed",
            )
            if key in kwargs
        }
        self.config = PopulationConfig(
            population_size=population_size,
            agent_types=(
                agent_types
                if agent_types is not None
                else ["worker", "scout", "soldier"]
            ),
            spatial_distribution=spatial_distribution,
            behavioral_heterogeneity=behavioral_heterogeneity,
            spatial_bounds=spatial_bounds,
            **config_kwargs,
        )

        # Expose population size as direct attribute
        self.population_size = population_size

        # Population state
        self.agents: list[SwarmAgent] = []
        self.environment: EnvironmentalState | None = None
        self.simulation_results = SimulationResults()

        # Integration components
        self.spatial_indexer = None
        self.spatial_analytics = None
        self.temporal_manager = None
        self.rng = np.random.default_rng(self.config.random_seed)
        self.pheromone_system = kwargs.get("pheromone_system")

        # Behavioral rules
        self.foraging_rules: dict[str, Any] = {}
        self.communication_rules: dict[str, Any] = {}
        self.adaptation_rules: dict[str, Any] = {}

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []

        # Initialize integrations
        self._initialize_integrations()

        # Create agents automatically
        self.create_agents()

        logger.info(f"AgentPopulation initialized with {self.population_size} agents")

    def _initialize_integrations(self) -> None:
        """Initialize integration with other GEO-INFER modules."""
        # Initialize spatial indexing
        if SpatialIndexingInterface:
            try:
                self.spatial_indexer = SpatialIndexingInterface(backend="h3")
                logger.info("Spatial indexer initialized for population")
            except Exception as e:
                logger.warning(f"Failed to initialize spatial indexer: {e}")

        # Initialize spatial analytics
        if SpatialAnalyticsInterface:
            try:
                self.spatial_analytics = SpatialAnalyticsInterface(backend="h3")
                logger.info("Spatial analytics initialized for population")
            except Exception as e:
                logger.warning(f"Failed to initialize spatial analytics: {e}")

        # Initialize temporal manager
        if TemporalManager:
            try:
                self.temporal_manager = TemporalManager()
                logger.info("Temporal manager initialized for population")
            except Exception as e:
                logger.warning(f"Failed to initialize temporal manager: {e}")

    def set_behavioral_rules(
        self,
        foraging_rules: dict[str, Any] | None = None,
        communication_rules: dict[str, Any] | None = None,
        adaptation_rules: dict[str, Any] | None = None,
    ) -> None:
        """
        Configure behavioral rules for the population.

        Args:
            foraging_rules: Rules governing resource acquisition behavior
            communication_rules: Rules for inter-agent communication
            adaptation_rules: Rules for behavioral adaptation and learning
        """
        if foraging_rules:
            self.foraging_rules.update(foraging_rules)
            logger.info("Foraging rules updated")

        if communication_rules:
            self.communication_rules.update(communication_rules)
            logger.info("Communication rules updated")

        if adaptation_rules:
            self.adaptation_rules.update(adaptation_rules)
            logger.info("Adaptation rules updated")

    def initialize_environment(
        self,
        spatial_bounds: dict[str, float] | None = None,
        resource_distribution: dict[str, Any] | None = None,
        obstacle_map: dict[str, Any] | None = None,
        pheromone_diffusion: dict[str, Any] | None = None,
        environmental_factors: dict[str, Any] | None = None,
    ) -> EnvironmentalState:
        """
        Initialize the spatial environment for agent simulation.

        Args:
            spatial_bounds: Geographic bounds for the simulation area
            resource_distribution: Distribution of resources in the environment
            obstacle_map: Physical and environmental obstacles
            pheromone_diffusion: Pheromone diffusion parameters
            environmental_factors: Environmental conditions and factors

        Returns:
            Initialized environmental state
        """
        # Set spatial bounds
        if spatial_bounds:
            self.config.spatial_bounds = spatial_bounds

        bounds = self.config.spatial_bounds or {
            "min_lat": -10,
            "max_lat": 10,
            "min_lng": -10,
            "max_lng": 10,
        }
        bounds = validate_bounds(bounds)

        # Initialize environmental state
        self.environment = EnvironmentalState(
            spatial_bounds=bounds,
            resource_distribution=resource_distribution
            or self._default_resource_distribution(),
            obstacle_map=obstacle_map or {},
            pheromone_diffusion=pheromone_diffusion
            or self._default_pheromone_diffusion(),
            environmental_factors=environmental_factors or {},
        )

        if self.pheromone_system is None:
            from .stigmergy import PheromoneSystem

            self.pheromone_system = PheromoneSystem(bounds=bounds)
        for agent in self.agents:
            agent.spatial_bounds = bounds
            agent.pheromone_system = self.pheromone_system

        logger.info(f"Environment initialized with bounds: {bounds}")
        return self.environment

    def _default_resource_distribution(self) -> dict[str, Any]:
        """Generate default resource distribution."""
        bounds = self.config.spatial_bounds or {
            "min_lat": -10,
            "max_lat": 10,
            "min_lng": -10,
            "max_lng": 10,
        }

        def random_point() -> np.ndarray:
            return np.array(
                [
                    self.rng.uniform(bounds["min_lat"], bounds["max_lat"]),
                    self.rng.uniform(bounds["min_lng"], bounds["max_lng"]),
                ]
            )

        return {
            "food": {
                "type": "spatial_field",
                "centers": [random_point() for _ in range(10)],
                "max_density": 1.0,
                "decay_rate": 0.1,
                "regeneration_rate": 0.05,
            },
            "water": {
                "type": "spatial_field",
                "centers": [random_point() for _ in range(5)],
                "max_density": 0.8,
                "decay_rate": 0.15,
                "regeneration_rate": 0.1,
            },
        }

    def _default_pheromone_diffusion(self) -> dict[str, Any]:
        """Generate default pheromone diffusion parameters."""
        return {
            "trail": {
                "evaporation_rate": 0.1,
                "diffusion_rate": 0.05,
                "max_intensity": 2.0,
            },
            "food": {
                "evaporation_rate": 0.05,
                "diffusion_rate": 0.1,
                "max_intensity": 1.5,
            },
            "alarm": {
                "evaporation_rate": 0.2,
                "diffusion_rate": 0.2,
                "max_intensity": 3.0,
            },
        }

    def create_agents(self) -> list["SwarmAgent"]:
        """
        Create and initialize all agents in the population.

        Returns:
            List of initialized swarm agents
        """
        from .agent_base import SwarmAgent

        agents = []
        agent_counts = self._distribute_agent_types()

        for agent_type, count in agent_counts.items():
            for i in range(count):
                agent_id = f"{agent_type}_{i + 1:03d}"

                # Generate initial position based on distribution strategy
                position = self._generate_initial_position(agent_type, i)

                # Create agent with type-specific parameters
                agent_config = self._get_agent_config(agent_type)

                agent_config["spatial_bounds"] = self.config.spatial_bounds or {
                    "min_lat": -10,
                    "max_lat": 10,
                    "min_lng": -10,
                    "max_lng": 10,
                }
                agent_config["random_seed"] = int(self.rng.integers(0, 2**32 - 1))
                if self.pheromone_system is not None:
                    agent_config["pheromone_system"] = self.pheromone_system
                agent = SwarmAgent(agent_id=agent_id, position=position, **agent_config)

                # Set agent type and initial state
                agent.agent_type = agent_type
                agent.population = self

                agents.append(agent)

        self.agents = agents
        self.simulation_results.population_size = self.population_size

        logger.info(f"Created {len(agents)} agents")
        return agents

    def _distribute_agent_types(self) -> dict[str, int]:
        """Distribute population across agent types."""
        agent_counts = {}

        if len(self.config.agent_types) == 1:
            # All agents of same type
            agent_counts[self.config.agent_types[0]] = self.config.population_size
        else:
            # Distribute proportionally
            base_count = self.config.population_size // len(self.config.agent_types)
            remainder = self.config.population_size % len(self.config.agent_types)

            for i, agent_type in enumerate(self.config.agent_types):
                count = base_count + (1 if i < remainder else 0)
                agent_counts[agent_type] = count

        return agent_counts

    def _generate_initial_position(self, agent_type: str, index: int) -> np.ndarray:
        """Generate initial position for agent based on distribution strategy."""
        if self.config.spatial_distribution == "random":
            bounds = self.config.spatial_bounds or {
                "min_lat": -10,
                "max_lat": 10,
                "min_lng": -10,
                "max_lng": 10,
            }
            return np.array(
                [
                    self.rng.uniform(bounds["min_lat"], bounds["max_lat"]),
                    self.rng.uniform(bounds["min_lng"], bounds["max_lng"]),
                ]
            )

        elif self.config.spatial_distribution == "clustered":
            # Create clusters around predefined centers
            centers = self.config.clustering_centers or [
                np.array([0, 0]),
                np.array([5, 5]),
                np.array([-5, -5]),
            ]

            center_idx = index % len(centers)
            center = centers[center_idx]

            # Generate position around cluster center
            angle = self.rng.uniform(0, 2 * np.pi)
            distance = self.rng.uniform(0, self.config.clustering_radius)

            position = center + np.array(
                [distance * np.cos(angle), distance * np.sin(angle)]
            )
            config_bounds = self.config.spatial_bounds
            if config_bounds:
                position = np.clip(
                    position,
                    [config_bounds["min_lat"], config_bounds["min_lng"]],
                    [config_bounds["max_lat"], config_bounds["max_lng"]],
                )
            return np.asarray(position)

        elif self.config.spatial_distribution == "uniform":
            # Grid-like distribution
            grid_size = int(np.ceil(np.sqrt(self.config.population_size)))
            bounds = self.config.spatial_bounds or {
                "min_lat": -10,
                "max_lat": 10,
                "min_lng": -10,
                "max_lng": 10,
            }

            row = index // grid_size
            col = index % grid_size

            lat_step = (bounds["max_lat"] - bounds["min_lat"]) / grid_size
            lng_step = (bounds["max_lng"] - bounds["min_lng"]) / grid_size

            return np.array(
                [
                    bounds["min_lat"] + (row + 0.5) * lat_step,
                    bounds["min_lng"] + (col + 0.5) * lng_step,
                ]
            )

        else:  # custom or fallback to random
            return self._generate_initial_position("random", index)

    def _get_agent_config(self, agent_type: str) -> dict[str, Any]:
        """Get configuration parameters for specific agent type."""
        base_config: dict[str, Any] = {
            "sensory_range": 100.0,
            "movement_speed": 1.5,
            "active_inference_enabled": True,
            "spatial_backend": "h3",
        }

        # Type-specific modifications
        if agent_type == "scout":
            base_config.update(
                {"sensory_range": 150.0, "movement_speed": 2.0, "initial_energy": 1.2}
            )
        elif agent_type == "soldier":
            base_config.update(
                {"sensory_range": 80.0, "movement_speed": 1.2, "initial_energy": 1.5}
            )
        elif agent_type == "queen":
            base_config.update(
                {"sensory_range": 200.0, "movement_speed": 0.8, "initial_energy": 2.0}
            )

        # Add behavioral heterogeneity if configured
        if self.config.behavioral_heterogeneity == "stochastic":
            # Add random variation to parameters
            for key in ["sensory_range", "movement_speed"]:
                if key in base_config:
                    variation = self.rng.normal(1.0, 0.1)  # 10% variation
                    base_config[key] *= variation

        return base_config

    async def run_simulation(
        self,
        time_steps: int = 1000,
        environmental_changes: list[dict[str, Any]] | None = None,
        data_collection: list[str] | None = None,
        progress_callback: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> SimulationResults:
        """
        Run population dynamics simulation.

        Args:
            time_steps: Number of simulation time steps
            environmental_changes: Scheduled environmental changes
            data_collection: Types of data to collect during simulation
            progress_callback: Optional callback for progress updates

        Returns:
            Comprehensive simulation results
        """
        if not isinstance(time_steps, int) or time_steps < 0:
            raise ValueError("time_steps must be a non-negative integer")
        logger.info(f"Starting simulation with {time_steps} time steps")

        # Initialize if needed
        if not self.agents:
            self.create_agents()

        if not self.environment:
            self.initialize_environment()

        # Set up data collection
        data_types = data_collection or [
            "trajectories",
            "interactions",
            "emergent_patterns",
        ]
        self.simulation_results = SimulationResults()
        self.simulation_results.population_size = len(self.agents)

        # Initialize environmental changes schedule
        environmental_schedule = self._create_environmental_schedule(
            environmental_changes or []
        )

        # Simulation loop
        for step in range(time_steps):
            _step_start_time = datetime.now()

            # Update environment
            await self._update_environment(step, environmental_schedule)

            # Update all agents
            await self._update_agents(step)

            # Collect data
            await self._collect_simulation_data(step, data_types)
            self.simulation_results.time_steps = step + 1

            # Progress reporting
            if progress_callback and step % 10 == 0:
                progress_info = {
                    "step": step,
                    "time_steps": time_steps,
                    "agents_alive": sum(
                        1 for agent in self.agents if agent.energy_level > 0
                    ),
                    "simulation_time": self.simulation_results.simulation_time,
                }
                progress_callback(step, progress_info)

            # Update simulation time
            self.simulation_results.simulation_time += self.config.time_step

            # Check for early termination
            if self._should_terminate_simulation():
                logger.info(f"Simulation terminated early at step {step}")
                break

        # Final data collection and analysis
        await self._finalize_simulation(data_types)

        logger.info(
            f"Simulation completed: {self.simulation_results.time_steps} steps, "
            f"{self.simulation_results.simulation_time:.2f} simulation time"
        )

        return self.simulation_results

    async def _update_environment(
        self, step: int, environmental_schedule: list[dict[str, Any]]
    ) -> None:
        """Update environmental state for current time step."""
        if not self.environment:
            return

        current_time = self.simulation_results.simulation_time

        # Apply scheduled environmental changes
        for change in environmental_schedule:
            if change["start_time"] <= current_time <= change["end_time"]:
                self.environment.update_environmental_factors(change["factors"])

        # Update resource distribution (regeneration/depletion)
        if self.environment.resource_distribution:
            for distribution in self.environment.resource_distribution.values():
                if "regeneration_rate" in distribution:
                    # Simple resource regeneration
                    max_density = distribution.get("max_density", 1.0)
                    current_density = distribution.get("current_density", max_density)
                    regeneration = (
                        distribution["regeneration_rate"] * self.config.time_step
                    )

                    distribution["current_density"] = min(
                        max_density, current_density + regeneration
                    )

    async def _update_agents(self, step: int) -> None:
        """Update all agents for current time step."""
        if self.config.parallel_processing and len(self.agents) > 10:
            # Parallel update for large populations
            await self._parallel_agent_update(step)
        else:
            # Sequential update for smaller populations
            await self._sequential_agent_update(step)

    async def _parallel_agent_update(self, step: int) -> None:
        """Update agents concurrently on the active event loop.

        Agent updates are async APIs. Creating a thread and a new event loop for
        every agent on every simulation step added substantial overhead and made
        integrations that own event-loop state fragile. A semaphore preserves
        the configured concurrency limit while keeping all updates on the
        simulation loop.
        """
        semaphore = asyncio.Semaphore(max(1, self.config.max_workers))

        async def update_single_agent(agent: "SwarmAgent") -> bool:
            """Update one agent while respecting the concurrency limit."""
            async with semaphore:
                try:
                    await self._update_single_agent(agent, step)
                    return True
                except Exception as e:
                    logger.error(f"Error updating agent {agent.agent_id}: {e}")
                    return False

        results = await asyncio.gather(
            *(update_single_agent(agent) for agent in self.agents)
        )

        successful_updates = sum(results)
        logger.debug(
            f"Parallel update completed: {successful_updates}/{len(self.agents)} agents updated"
        )

    async def _sequential_agent_update(self, step: int) -> None:
        """Update agents sequentially."""
        for agent in self.agents:
            await self._update_single_agent(agent, step)

    async def _update_single_agent(self, agent: "SwarmAgent", step: int) -> None:
        """Update a single agent for current time step."""
        try:
            # Agent updates require an initialized environment.
            if self.environment is None:
                return
            # Check if agent is still active
            if agent.energy_level <= 0:
                return  # Agent is inactive

            # Get environmental context at agent location
            environmental_context = self._get_environmental_context(agent.position)

            # Agent perception
            sensory_input = await agent.perceive_environment(
                spatial_context={
                    "position": agent.position,
                    "bounds": self.environment.spatial_bounds,
                },
                environmental_signals=environmental_context,
                social_signals=self._get_social_context(agent),
                temporal_context={
                    "simulation_time": self.simulation_results.simulation_time
                },
            )

            # Agent decision making
            motivations = self._get_agent_motivations(agent, sensory_input)
            decision = agent.make_decision(
                sensory_input, motivations, self._get_behavioral_rules(agent)
            )

            # Agent action execution
            if decision:
                result = await agent.execute_action(decision)

                # Record interaction if significant
                if result.get("success", False):
                    self.simulation_results.add_interaction(
                        {
                            "step": step,
                            "agent_id": agent.agent_id,
                            "action_type": decision.action_type,
                            "result": result,
                            "position": agent.position.tolist(),
                        }
                    )

        except Exception as e:
            logger.error(f"Error updating agent {agent.agent_id}: {e}")

    def _get_environmental_context(self, position: np.ndarray) -> dict[str, Any]:
        """Get environmental context at given position."""
        if not self.environment:
            return {}

        context = {}

        # Get resources at location
        resources = self.environment.get_resource_at_location(position)
        context.update(resources)

        # Add environmental factors
        context.update(self.environment.environmental_factors)

        # Add weather conditions if available
        if self.environment.weather_conditions:
            context.update(self.environment.weather_conditions)

        return context

    def _get_social_context(self, agent: "SwarmAgent") -> dict[str, Any]:
        """Get social context for agent (nearby agents)."""
        context: dict[str, Any] = {
            "nearby_agents": 0,
            "nearby_agent_types": {},
            "social_signals": [],
        }

        # Population positions are the authoritative agent state. The generic
        # SPACE cell-neighbor API returns cells, not agents, so using it here
        # both miscounted neighbors and passed coordinates to a cell API. Keep
        # the exact in-memory distance calculation until a population-aware
        # spatial index is maintained.
        active_agents = [
            other_agent
            for other_agent in self.agents
            if other_agent is not agent and other_agent.energy_level > 0
        ]
        if not active_agents:
            return context

        positions = np.asarray([other_agent.position for other_agent in active_agents])
        distances = np.linalg.norm(positions - agent.position, axis=1)
        for other_agent, distance in zip(active_agents, distances):
            if distance <= agent.sensory_range:
                context["nearby_agents"] += 1
                agent_type = getattr(other_agent, "agent_type", "unknown")
                context["nearby_agent_types"][agent_type] = (
                    context["nearby_agent_types"].get(agent_type, 0) + 1
                )

        return context

    def _get_agent_motivations(
        self, agent: "SwarmAgent", sensory_input: Any
    ) -> dict[str, float]:
        """Get internal motivations for agent based on current state."""
        motivations = {
            "energy_conservation": max(0, 1.0 - agent.energy_level),
            "task_completion": 0.5,
            "social_coordination": 0.3,
            "exploration": 0.2,
        }

        # Adjust based on agent type
        agent_type = getattr(agent, "agent_type", "worker")
        if agent_type == "scout":
            motivations["exploration"] = 0.8
        elif agent_type == "soldier":
            motivations["social_coordination"] = 0.8

        # Adjust based on environmental context
        processed = (
            sensory_input.processed_data
            if hasattr(sensory_input, "processed_data")
            else {}
        )

        if processed.get("env_food_nearby", False):
            motivations["task_completion"] = 0.9

        if processed.get("social_nearby_agents", 0) > 5:
            motivations["social_coordination"] = 0.7

        return motivations

    def _get_behavioral_rules(self, agent: "SwarmAgent") -> dict[str, Any]:
        """Get behavioral rules for agent."""
        rules: dict[str, Any] = {}

        # Apply foraging rules
        if self.foraging_rules:
            rules["foraging"] = self.foraging_rules

        # Apply communication rules
        if self.communication_rules:
            rules["communication"] = self.communication_rules

        # Apply adaptation rules
        if self.adaptation_rules:
            rules["adaptation"] = self.adaptation_rules

        # Add agent type specific rules
        agent_type = getattr(agent, "agent_type", "worker")
        rules["agent_type"] = agent_type

        return rules

    async def _collect_simulation_data(self, step: int, data_types: list[str]) -> None:
        """Collect simulation data for current step."""
        # Collect agent positions for trajectory analysis
        if "trajectories" in data_types:
            active_agents = [agent for agent in self.agents if agent.energy_level > 0]
            tracked = active_agents if active_agents else self.agents
            positions = np.array([agent.position for agent in tracked])
            self.simulation_results.add_trajectory(step, positions)

        # Collect performance metrics
        if "performance" in data_types or "emergent_patterns" in data_types:
            metrics = self._calculate_population_metrics()
            self.simulation_results.update_performance_metrics(metrics)

            if "emergent_patterns" in data_types:
                patterns = self._analyze_emergent_patterns()
                self.simulation_results.update_emergent_patterns(patterns)

    def _calculate_population_metrics(self) -> dict[str, Any]:
        """Calculate population-level performance metrics."""
        if not self.agents:
            return {}

        active_agents = [agent for agent in self.agents if agent.energy_level > 0]

        if not active_agents:
            return {"population_status": "extinct"}

        # Basic population metrics
        metrics = {
            "active_agents": len(active_agents),
            "total_agents": len(self.agents),
            "average_energy": np.mean([agent.energy_level for agent in active_agents]),
            "energy_std": np.std([agent.energy_level for agent in active_agents]),
        }

        # Spatial distribution metrics
        positions = np.array([agent.position for agent in active_agents])
        if len(positions) > 1:
            metrics["spatial_center"] = np.mean(positions, axis=0).tolist()
            metrics["spatial_spread"] = np.std(positions, axis=0).tolist()

            # Pairwise distances provide a stable, dependency-light clustering
            # summary for the current active population.
            if len(positions) > 2:
                from scipy.spatial.distance import pdist

                distances = pdist(positions)
                metrics["average_inter_agent_distance"] = np.mean(distances)
                metrics["max_inter_agent_distance"] = np.max(distances)

        # Agent type distribution
        agent_types: dict[str, int] = {}
        for agent in active_agents:
            agent_type = getattr(agent, "agent_type", "unknown")
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1

        metrics["agent_type_distribution"] = agent_types

        return metrics

    def _analyze_emergent_patterns(self) -> dict[str, Any]:
        """Analyze emergent patterns in population behavior."""
        patterns: dict[str, Any] = {}

        if not self.agents:
            return patterns

        # Get recent performance history
        recent_metrics = (
            self.performance_history[-10:] if self.performance_history else []
        )

        if len(recent_metrics) < 3:
            return {"status": "insufficient_data"}

        # Analyze spatial clustering
        try:
            active_agents = [agent for agent in self.agents if agent.energy_level > 0]
            positions = np.array([agent.position for agent in active_agents])

            if len(positions) > 2:
                # Simple clustering analysis using k-means
                from sklearn.cluster import KMeans

                n_clusters = min(5, len(positions) // 10 + 1)
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
                    cluster_labels = kmeans.fit_predict(positions)

                    patterns["spatial_clusters"] = {
                        "n_clusters": n_clusters,
                        "cluster_centers": kmeans.cluster_centers_.tolist(),
                        "cluster_sizes": np.bincount(cluster_labels).tolist(),
                    }

        except Exception as e:
            logger.warning(f"Clustering analysis failed: {e}")

        # Analyze temporal patterns
        if recent_metrics:
            energy_trend = [m.get("average_energy", 0) for m in recent_metrics]
            if len(energy_trend) > 2:
                patterns["energy_trend"] = (
                    "increasing" if energy_trend[-1] > energy_trend[0] else "decreasing"
                )

        return patterns

    def _create_environmental_schedule(
        self, changes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create schedule for environmental changes."""
        schedule = []

        for change in changes:
            schedule_item = {
                "start_time": change.get("start_time", 0),
                "end_time": change.get("end_time", float("inf")),
                "factors": change.get("factors", {}),
            }
            schedule.append(schedule_item)

        # Sort by start time
        schedule.sort(key=lambda x: x["start_time"])

        return schedule

    def _should_terminate_simulation(self) -> bool:
        """Check if simulation should terminate early."""
        # Terminate if maximum time reached (only when explicitly configured)
        if self.config.max_simulation_time:
            active_count = sum(1 for agent in self.agents if agent.energy_level > 0)
            if active_count == 0:
                return True
            if (
                self.simulation_results.simulation_time
                >= self.config.max_simulation_time
            ):
                return True

        return False

    async def _finalize_simulation(self, data_types: list[str]) -> None:
        """Finalize simulation and perform final analysis."""
        # Calculate final performance metrics
        final_metrics = self._calculate_population_metrics()
        self.simulation_results.update_performance_metrics(final_metrics)

        # Analyze final emergent patterns
        if "emergent_patterns" in data_types:
            final_patterns = self._analyze_emergent_patterns()
            self.simulation_results.update_emergent_patterns(final_patterns)
            # Also include in performance_metrics for test compatibility
            self.simulation_results.performance_metrics["emergent_patterns"] = (
                final_patterns
            )

        # Generate summary statistics
        self.simulation_results.performance_metrics["summary"] = {
            "total_simulation_time": self.simulation_results.simulation_time,
            "total_time_steps": self.simulation_results.time_steps,
            "final_population_size": self.simulation_results.population_size,
            "data_collection_complete": True,
        }

    def get_agent_by_id(self, agent_id: str) -> Optional["SwarmAgent"]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def get_agents_by_type(self, agent_type: str) -> list["SwarmAgent"]:
        """Get all agents of specified type."""
        return [
            agent
            for agent in self.agents
            if getattr(agent, "agent_type", None) == agent_type
        ]

    def get_agents_in_region(
        self, center: np.ndarray, radius: float
    ) -> list["SwarmAgent"]:
        """Get all agents within specified radius of center."""
        agents_in_region = []

        for agent in self.agents:
            if agent.energy_level > 0:  # Only active agents
                distance = np.linalg.norm(agent.position - center)
                if distance <= radius:
                    agents_in_region.append(agent)

        return agents_in_region

    def save_simulation_results(self, filepath: str) -> None:
        """Save simulation results to file."""
        try:
            with open(filepath, "w") as f:
                json.dump(self.simulation_results.to_dict(), f, indent=2)
            logger.info(f"Simulation results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save simulation results: {e}")
            raise

    def load_simulation_results(self, filepath: str) -> SimulationResults:
        """Load simulation results from file."""
        try:
            with open(filepath) as f:
                data = json.load(f)

            results = SimulationResults()
            results.trajectories = [
                np.array(traj) for traj in data.get("trajectories", [])
            ]
            results.interactions = data.get("interactions", [])
            results.emergent_patterns = data.get("emergent_patterns", {})
            results.performance_metrics = data.get("performance_metrics", {})
            results.simulation_time = data.get("simulation_time", 0.0)
            results.time_steps = data.get("time_steps", 0)
            results.population_size = data.get("population_size", 0)

            self.simulation_results = results
            logger.info(f"Simulation results loaded from {filepath}")
            return results

        except Exception as e:
            logger.error(f"Failed to load simulation results: {e}")
            raise
