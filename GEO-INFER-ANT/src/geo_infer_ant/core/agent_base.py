"""
Swarm Agent Base Classes for GEO-INFER-ANT

This module provides the foundational classes for swarm intelligence agents,
integrating with Active Inference (ACT), spatial reasoning (SPACE), and agent
management (AGENT) modules to create sophisticated collective intelligence systems.
"""

import numpy as np
import logging
from typing import Any, cast
from datetime import datetime
from dataclasses import dataclass, field

# Integration modules are optional; ANT degrades gracefully when they are
# absent (the same pattern as aco.py, pso.py, population.py, stigmergy.py).
try:
    from geo_infer_act.core.active_inference import ActiveInferenceModel
except ImportError as e:
    logging.getLogger(__name__).debug(
        "Optional active-inference integration unavailable: %s", e
    )
    ActiveInferenceModel = None

try:
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface
except ImportError as e:
    logging.getLogger(__name__).debug("Optional spatial integration unavailable: %s", e)
    SpatialIndexingInterface = None
    SpatialAnalyticsInterface = None

try:
    from geo_infer_agent.core.agent_base import BaseAgent
except ImportError as e:
    logging.getLogger(__name__).debug(
        "Optional agent-framework integration unavailable: %s", e
    )
    BaseAgent = None


logger = logging.getLogger(__name__)


@dataclass
class SensoryInput:
    """
    Structured sensory input for swarm agents.

    Integrates multiple types of environmental and social signals
    for comprehensive agent perception.
    """

    spatial_context: dict[str, Any] = field(default_factory=dict)
    environmental_signals: dict[str, Any] = field(default_factory=dict)
    social_signals: dict[str, Any] = field(default_factory=dict)
    stigmergic_signals: dict[str, Any] = field(default_factory=dict)
    temporal_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and process sensory input after initialization."""
        self.timestamp = datetime.now()
        self.processed = False

    def process(self) -> dict[str, Any]:
        """Process and integrate all sensory inputs."""
        if self.processed:
            return self.to_dict()

        # Integrate spatial context
        processed = {
            "spatial_position": self.spatial_context.get("position", np.array([0, 0])),
            "spatial_bounds": self.spatial_context.get("bounds"),
            "spatial_resolution": self.spatial_context.get("resolution", "h3_r8"),
        }

        # Integrate environmental signals
        for key, value in self.environmental_signals.items():
            processed[f"env_{key}"] = value

        # Integrate social signals
        for key, value in self.social_signals.items():
            processed[f"social_{key}"] = value

        # Integrate stigmergic signals
        for key, value in self.stigmergic_signals.items():
            processed[f"stigmergic_{key}"] = value

        # Add temporal context
        processed.update(self.temporal_context)

        self.processed = True
        self.processed_data = processed
        return processed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "spatial_context": self.spatial_context,
            "environmental_signals": self.environmental_signals,
            "social_signals": self.social_signals,
            "stigmergic_signals": self.stigmergic_signals,
            "temporal_context": self.temporal_context,
            "timestamp": self.timestamp.isoformat(),
            "processed": self.processed,
            "processed_data": getattr(self, "processed_data", {}),
        }


@dataclass
class ActionDecision:
    """
    Structured action decision for swarm agents.

    Represents the output of agent decision-making processes,
    integrating multiple action types and confidence measures.
    """

    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    expected_outcome: dict[str, Any] = field(default_factory=dict)
    alternative_actions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate action decision after initialization."""
        self.timestamp = datetime.now()
        self.execution_priority = self.calculate_priority()

    def calculate_priority(self) -> float:
        """Calculate execution priority based on confidence and context."""
        base_priority = self.confidence

        # Adjust based on action type urgency
        urgency_multipliers = {
            "emergency_response": 2.0,
            "resource_acquisition": 1.5,
            "communication": 1.2,
            "movement": 1.0,
            "monitoring": 0.8,
        }

        multiplier = urgency_multipliers.get(self.action_type, 1.0)
        return base_priority * multiplier

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "expected_outcome": self.expected_outcome,
            "alternative_actions": self.alternative_actions,
            "timestamp": self.timestamp.isoformat(),
        }


_SwarmAgentBase: type = BaseAgent if BaseAgent is not None else object


class SwarmAgent(_SwarmAgentBase):
    """
    Base class for swarm intelligence agents.

    Integrates with Active Inference (ACT), spatial reasoning (SPACE),
    and agent management (AGENT) when those optional workspace packages
    are installed; without them the agent runs standalone with all
    ANT-specific behavior intact (active-inference and spatial hooks
    stay inert, and the AGENT belief state is simply absent).

    Key Features:
    - Active Inference decision making
    - Spatial awareness and navigation
    - Stigmergic communication capabilities
    - Multi-modal sensory processing
    - Adaptive learning and behavior modification
    """

    def __init__(
        self,
        agent_id: str,
        position: np.ndarray,
        sensory_range: float = 100.0,
        movement_speed: float = 1.5,
        active_inference_enabled: bool = True,
        spatial_backend: str = "h3",
        **kwargs: Any,
    ) -> None:
        """
        Initialize swarm agent.

        Args:
            agent_id: Unique identifier for this agent
            position: Initial spatial position as numpy array [lat, lng]
            sensory_range: Maximum distance for environmental perception (meters)
            movement_speed: Maximum movement speed (m/s)
            active_inference_enabled: Whether to use Active Inference for decision making
            spatial_backend: Backend for spatial operations ('h3', 'srai', 'geopandas')
            **kwargs: Additional configuration parameters
        """
        # Validate inputs
        position_arr = np.array(position, dtype=np.float64)
        if position_arr.shape != (2,) or not np.all(np.isfinite(position_arr)):
            raise ValueError(
                "Agent position must be a finite [latitude, longitude] pair"
            )
        if sensory_range < 0 or not np.isfinite(sensory_range):
            raise ValueError("sensory_range must be non-negative")
        if movement_speed < 0 or not np.isfinite(movement_speed):
            raise ValueError("movement_speed must be finite and non-negative")

        if BaseAgent is not None:
            # Full AGENT-framework base: lifecycle, belief state, handlers.
            super().__init__(agent_id, kwargs)
        else:
            # Standalone mode without geo-infer-agent: supply the two
            # attributes SwarmAgent itself relies on. AGENT framework
            # features (run loop, save_state, belief state) are simply absent.
            self.agent_id = agent_id
            self.config: dict[str, Any] = dict(kwargs)

        # Swarm-specific attributes
        self.position = position_arr
        self.sensory_range = sensory_range
        self.movement_speed = movement_speed
        self.active_inference_enabled = active_inference_enabled

        # Integration components
        self.active_inference_model: Any = None
        self.spatial_indexer: Any = None
        self.spatial_analytics: Any = None

        # Agent state
        self.energy_level = max(0.0, float(kwargs.get("initial_energy", 1.0)))
        self.rng = np.random.default_rng(kwargs.get("random_seed", kwargs.get("seed")))
        self.spatial_bounds = kwargs.get(
            "spatial_bounds",
            {"min_lat": -90.0, "max_lat": 90.0, "min_lng": -180.0, "max_lng": 180.0},
        )
        self.pheromone_system = kwargs.get("pheromone_system")
        self.task_memory: list[dict[str, Any]] = []
        self.social_signals: dict[str, Any] = {}

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []
        self.interaction_history: list[dict[str, Any]] = []

        # Initialize integrations
        self._initialize_integrations(spatial_backend)

        logger.info(f"SwarmAgent {agent_id} initialized at position {position}")

    def _initialize_integrations(self, spatial_backend: str) -> None:
        """Initialize integration with other GEO-INFER modules."""
        # Initialize Active Inference model
        if self.active_inference_enabled and ActiveInferenceModel:
            try:
                self.active_inference_model = ActiveInferenceModel(
                    model_type="spatial_temporal",
                    preferences={
                        "energy_conservation": 0.8,
                        "task_completion": 0.9,
                        "social_coordination": 0.7,
                    },
                    precision_parameters={
                        "observation_precision": 1.0,
                        "action_precision": 0.8,
                        "state_precision": 0.6,
                    },
                )
                logger.info(
                    f"Active Inference model initialized for agent {self.agent_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Active Inference: {e}")
                self.active_inference_enabled = False

        # Initialize spatial indexing
        if SpatialIndexingInterface:
            try:
                self.spatial_indexer = SpatialIndexingInterface(backend=spatial_backend)
                logger.info(
                    f"Spatial indexer initialized with {spatial_backend} backend"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize spatial indexer: {e}")

        # Initialize spatial analytics
        if SpatialAnalyticsInterface:
            try:
                self.spatial_analytics = SpatialAnalyticsInterface(
                    backend=spatial_backend
                )
                logger.info(
                    f"Spatial analytics initialized with {spatial_backend} backend"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize spatial analytics: {e}")

    async def perceive_environment(
        self,
        spatial_context: dict[str, Any] | None = None,
        environmental_signals: dict[str, Any] | None = None,
        social_signals: dict[str, Any] | None = None,
        stigmergic_signals: dict[str, Any] | None = None,
        temporal_context: dict[str, Any] | None = None,
    ) -> SensoryInput:
        """
        Perceive and integrate environmental information.

        This method creates a comprehensive sensory input by gathering
        information from multiple sources and integrating them into
        a unified representation.

        Args:
            spatial_context: Current spatial position and bounds
            environmental_signals: Environmental sensor readings
            social_signals: Communications from other agents
            stigmergic_signals: Pheromone or marker information
            temporal_context: Current time and temporal patterns

        Returns:
            Processed sensory input structure
        """
        # Create sensory input structure
        sensory_input = SensoryInput(
            spatial_context=spatial_context or {"position": self.position},
            environmental_signals=environmental_signals or {},
            social_signals=social_signals or {},
            stigmergic_signals=stigmergic_signals or {},
            temporal_context=temporal_context or {"current_time": datetime.now()},
        )

        # Process through Active Inference if enabled
        if self.active_inference_enabled and self.active_inference_model:
            try:
                processed_data = sensory_input.process()

                # Update Active Inference observations
                observations = self._extract_observations(processed_data)
                beliefs = self._update_active_inference(observations)
                sensory_input.processed_data["active_inference_observations"] = (
                    observations
                )
                if beliefs is not None:
                    # Store processed beliefs in sensory input.
                    sensory_input.processed_data.update(
                        {
                            "active_inference_beliefs": beliefs,
                            "free_energy": getattr(
                                self.active_inference_model,
                                "current_free_energy",
                                0.0,
                            ),
                        }
                    )

            except Exception as e:
                logger.warning(f"Active Inference processing failed: {e}")

        # Use spatial analytics if available
        if self.spatial_analytics and sensory_input.spatial_context:
            try:
                # Analyze spatial context
                spatial_analysis = self.spatial_analytics.analyze_context(
                    sensory_input.spatial_context
                )
                sensory_input.processed_data.update(
                    {"spatial_analysis": spatial_analysis}
                )
            except Exception as e:
                logger.warning(f"Spatial analysis failed: {e}")

        logger.debug(f"Agent {self.agent_id} processed sensory input")
        return sensory_input

    def _update_active_inference(self, observations: dict[str, Any]) -> Any | None:
        """Store observations and update beliefs when a model is configured.

        GEO-INFER-ACT exposes ``update_observations`` for structured context and
        ``perceive`` for numeric belief updates.  A newly-created ANT agent has
        no generative model yet, so it must retain observations without calling
        a belief API that cannot run.  The guarded path also keeps compatibility
        with older injected models exposing a zero-argument ``update_beliefs``.
        """
        model = self.active_inference_model
        if model is None:
            return None

        update_observations = getattr(model, "update_observations", None)
        if callable(update_observations):
            update_observations(observations)

        if getattr(model, "generative_model", None) is not None:
            perceive = getattr(model, "perceive", None)
            if callable(perceive):
                numeric_observations = self._numeric_observation_vector(observations)
                if numeric_observations.size:
                    return perceive(numeric_observations)

        update_beliefs = getattr(model, "update_beliefs", None)
        if callable(update_beliefs):
            return update_beliefs()
        return None

    @staticmethod
    def _numeric_observation_vector(observations: dict[str, Any]) -> np.ndarray:
        """Flatten finite numeric observations for configured ACT models."""
        values: list[float] = []
        for value in observations.values():
            try:
                array = np.asarray(value, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            values.extend(array[np.isfinite(array)].tolist())
        return np.asarray(values, dtype=float)

    def _extract_observations(self, processed_data: dict[str, Any]) -> dict[str, Any]:
        """Extract observations for Active Inference model."""
        observations = {}

        # Extract spatial observations
        if "spatial_position" in processed_data:
            observations["spatial_position"] = processed_data["spatial_position"]

        # Extract environmental observations
        env_keys = [k for k in processed_data.keys() if k.startswith("env_")]
        for key in env_keys:
            observations[key[4:]] = processed_data[key]  # Remove 'env_' prefix

        # Extract social observations
        social_keys = [k for k in processed_data.keys() if k.startswith("social_")]
        for key in social_keys:
            observations[key[7:]] = processed_data[key]  # Remove 'social_' prefix

        return observations

    def make_decision(
        self,
        sensory_input: SensoryInput,
        internal_motivations: dict[str, float] | None = None,
        behavioral_rules: dict[str, Any] | None = None,
    ) -> ActionDecision:
        """
        Make behavioral decision based on sensory input and internal state.

        Integrates Active Inference, spatial reasoning, and behavioral rules
        to generate optimal action decisions.

        Args:
            sensory_input: Processed sensory information
            internal_motivations: Internal drives and preferences
            behavioral_rules: Species-specific behavioral constraints

        Returns:
            Structured action decision
        """
        processed_data = sensory_input.process()

        # Default motivations if none provided
        if internal_motivations is None:
            internal_motivations = {
                "energy_conservation": 0.8,
                "task_completion": 0.9,
                "social_coordination": 0.7,
                "exploration": 0.5,
            }

        # Use Active Inference for decision making if enabled
        if self.active_inference_enabled and self.active_inference_model:
            try:
                # Update model with current context
                self.active_inference_model.update_preferences(internal_motivations)

                # Generate policy options
                available_actions = self._generate_action_space(processed_data)
                policies = self.active_inference_model.generate_policies(
                    available_actions
                )

                # Select optimal action
                optimal_policy = self.active_inference_model.select_policy(policies)
                expected_fe = self.active_inference_model.compute_expected_free_energy(
                    optimal_policy
                )

                # Create action decision
                decision = ActionDecision(
                    action_type=optimal_policy["action_type"],
                    parameters=optimal_policy.get("parameters", {}),
                    confidence=1.0 - min(expected_fe, 1.0),  # Convert FE to confidence
                    expected_outcome=optimal_policy.get("expected_outcome", {}),
                    alternative_actions=[p for p in policies if p != optimal_policy],
                )

            except Exception as e:
                logger.warning(f"Active Inference decision making failed: {e}")
                decision = self._rule_based_decision_making(
                    processed_data, internal_motivations, behavioral_rules
                )

        else:
            # Use the explicit rule-based policy when Active Inference is disabled.
            decision = self._rule_based_decision_making(
                processed_data, internal_motivations, behavioral_rules
            )

        logger.debug(f"Agent {self.agent_id} made decision: {decision.action_type}")
        return decision

    def _generate_action_space(
        self, processed_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate possible actions based on current context."""
        actions = []

        # Movement actions
        if "spatial_position" in processed_data:
            actions.extend(
                [
                    {
                        "action_type": "move_toward_resource",
                        "parameters": {"target": "nearest_resource"},
                        "expected_outcome": {"energy_gain": 0.3},
                    },
                    {
                        "action_type": "move_away_from_threat",
                        "parameters": {"target": "safe_area"},
                        "expected_outcome": {"safety_increase": 0.5},
                    },
                    {
                        "action_type": "explore_unknown",
                        "parameters": {"target": "high_uncertainty_area"},
                        "expected_outcome": {"information_gain": 0.4},
                    },
                ]
            )

        # Communication actions
        if (
            "social_nearby_agents" in processed_data
            or processed_data.get("social_nearby_agents", 0) > 0
        ):
            actions.extend(
                [
                    {
                        "action_type": "communicate_status",
                        "parameters": {"message_type": "status_update"},
                        "expected_outcome": {"coordination_improvement": 0.2},
                    },
                    {
                        "action_type": "request_assistance",
                        "parameters": {"request_type": "task_help"},
                        "expected_outcome": {"task_completion_rate": 0.3},
                    },
                ]
            )

        # Stigmergic actions
        actions.extend(
            [
                {
                    "action_type": "deposit_pheromone",
                    "parameters": {"pheromone_type": "trail", "intensity": 1.0},
                    "expected_outcome": {"trail_strength": 0.8},
                },
                {
                    "action_type": "follow_pheromone",
                    "parameters": {"pheromone_type": "food"},
                    "expected_outcome": {"resource_discovery": 0.6},
                },
            ]
        )

        # Task-specific actions
        actions.extend(
            [
                {
                    "action_type": "forage",
                    "parameters": {"target_type": "food"},
                    "expected_outcome": {"energy_gain": 0.5},
                },
                {
                    "action_type": "rest",
                    "parameters": {"duration": 10},
                    "expected_outcome": {"energy_recovery": 0.2},
                },
                {
                    "action_type": "monitor_environment",
                    "parameters": {"sensor_types": ["temperature", "humidity"]},
                    "expected_outcome": {"information_gain": 0.3},
                },
            ]
        )

        return actions

    def _rule_based_decision_making(
        self,
        processed_data: dict[str, Any],
        internal_motivations: dict[str, float],
        behavioral_rules: dict[str, Any] | None = None,
    ) -> ActionDecision:
        """Select an action with the configured rule-based policy."""
        # Simple priority-based decision making
        current_energy = processed_data.get("energy_level", self.energy_level)

        # Energy-based decisions
        if current_energy < 0.3:
            # Low energy - prioritize foraging or resting
            if processed_data.get("env_food_nearby", False):
                return ActionDecision(
                    action_type="forage",
                    parameters={"target": "nearest_food"},
                    confidence=0.9,
                    expected_outcome={"energy_gain": 0.4},
                )
            else:
                return ActionDecision(
                    action_type="rest",
                    parameters={"duration": 30},
                    confidence=0.8,
                    expected_outcome={"energy_recovery": 0.3},
                )

        # Social coordination decisions
        nearby_agents = processed_data.get("social_nearby_agents", 0)
        if nearby_agents > 3:
            return ActionDecision(
                action_type="coordinate_with_swarm",
                parameters={"coordination_type": "task_allocation"},
                confidence=0.7,
                expected_outcome={"efficiency_gain": 0.2},
            )

        # Exploration decisions
        if internal_motivations.get("exploration", 0) > 0.6:
            return ActionDecision(
                action_type="explore",
                parameters={"target": "unknown_area"},
                confidence=0.6,
                expected_outcome={"information_gain": 0.3},
            )

        # Default monitoring behavior
        return ActionDecision(
            action_type="monitor_environment",
            parameters={"sensor_types": ["general"]},
            confidence=0.5,
            expected_outcome={"information_gain": 0.2},
        )

    async def execute_action(self, decision: ActionDecision) -> dict[str, Any]:
        """
        Execute the chosen action and return results.

        Args:
            decision: Action decision to execute

        Returns:
            Execution results and outcomes
        """
        logger.info(f"Agent {self.agent_id} executing action: {decision.action_type}")

        execution_result: dict[str, Any] = {
            "action_type": decision.action_type,
            "start_time": datetime.now(),
            "success": False,
            "actual_outcome": {},
            "energy_cost": 0.0,
        }

        try:
            # Route to appropriate action handler
            if decision.action_type == "move_toward_resource":
                result = await self._execute_movement_action(decision)
            elif decision.action_type in {
                "move_away_from_threat",
                "explore_unknown",
                "explore",
            }:
                result = await self._execute_movement_action(decision)
            elif decision.action_type == "deposit_pheromone":
                result = await self._execute_stigmergic_action(decision)
            elif decision.action_type == "follow_pheromone":
                result = await self._execute_follow_pheromone_action(decision)
            elif decision.action_type == "communicate_status":
                result = await self._execute_communication_action(decision)
            elif decision.action_type in {
                "communicate",
                "request_assistance",
                "coordinate_with_swarm",
            }:
                result = await self._execute_communication_action(decision)
            elif decision.action_type == "forage":
                result = await self._execute_foraging_action(decision)
            elif decision.action_type == "rest":
                result = await self._execute_rest_action(decision)
            elif decision.action_type == "monitor_environment":
                result = await self._execute_monitoring_action(decision)
            else:
                result = await self._execute_generic_action(decision)

            # Update execution result
            execution_result.update(result)
            execution_result["success"] = bool(result.get("success", True))

            # Update agent state based on execution
            self._update_agent_state(decision, execution_result)

            # Record in performance history
            self.performance_history.append(
                {
                    "timestamp": datetime.now(),
                    "action": decision.to_dict(),
                    "result": execution_result,
                }
            )

            # Update Active Inference model if enabled
            if self.active_inference_enabled and self.active_inference_model:
                self.active_inference_model.update_with_outcome(
                    decision.to_dict(), execution_result
                )

        except Exception as e:
            logger.error(f"Action execution failed for agent {self.agent_id}: {e}")
            execution_result["error"] = str(e)
            execution_result["success"] = False

        execution_result["end_time"] = datetime.now()
        execution_result["duration"] = (
            execution_result["end_time"] - execution_result["start_time"]
        ).total_seconds()

        logger.debug(f"Agent {self.agent_id} completed action execution")
        return execution_result

    async def _execute_movement_action(
        self, decision: ActionDecision
    ) -> dict[str, Any]:
        """Execute movement-related actions."""
        params = decision.parameters
        target = params.get("target", "default")

        if isinstance(target, (list, tuple, np.ndarray)):
            target_position = np.asarray(target, dtype=float)
            if target_position.shape != self.position.shape or not np.all(
                np.isfinite(target_position)
            ):
                raise ValueError("movement target must be a finite coordinate pair")
            direction = target_position - self.position
            norm = np.linalg.norm(direction)
            displacement = (
                direction / norm * min(norm, self.movement_speed)
                if norm > 1e-12
                else np.zeros(2)
            )
        else:
            # A named target without a supplied map target is an exploration
            # request.  Use the private RNG and a bounded step so simulations
            # remain reproducible and physically scaled by movement_speed.
            direction = self.rng.normal(size=2)
            direction /= cast(float, max(np.linalg.norm(direction), 1e-12))
            if target == "safe_area":
                direction = np.ones(2) / np.sqrt(2)
            elif target == "unknown_area":
                direction = self.rng.normal(size=2)
                direction /= cast(float, max(np.linalg.norm(direction), 1e-12))
            displacement = direction * self.movement_speed
        new_position = self.position + displacement

        # Update position
        old_position = self.position.copy()
        self.position = np.clip(
            new_position,
            [self.spatial_bounds["min_lat"], self.spatial_bounds["min_lng"]],
            [self.spatial_bounds["max_lat"], self.spatial_bounds["max_lng"]],
        )

        return {
            "old_position": old_position,
            "new_position": self.position,
            "distance_moved": np.linalg.norm(self.position - old_position),
            "energy_cost": 0.1,
            "actual_outcome": {"position_updated": True},
        }

    async def _execute_stigmergic_action(
        self, decision: ActionDecision
    ) -> dict[str, Any]:
        """Execute stigmergic (pheromone) actions."""
        params = decision.parameters
        pheromone_type = params.get("pheromone_type", "trail")
        intensity = params.get("intensity", 1.0)

        if self.pheromone_system is None:
            return {
                "success": False,
                "error": "No pheromone_system is attached to this agent",
                "energy_cost": 0.0,
                "actual_outcome": {"pheromone_deposited": False},
            }
        deposited = await self.pheromone_system.deposit_pheromone(
            agent_id=self.agent_id,
            pheromone_type=pheromone_type,
            location=self.position.copy(),
            intensity=float(intensity),
        )
        stigmergic_event = {
            "pheromone_type": pheromone_type,
            "intensity": intensity,
            "location": self.position,
            "timestamp": datetime.now(),
        }

        return {
            "stigmergic_event": stigmergic_event,
            "energy_cost": 0.05,
            "success": bool(deposited),
            "actual_outcome": {"pheromone_deposited": bool(deposited)},
        }

    async def _execute_follow_pheromone_action(
        self, decision: ActionDecision
    ) -> dict[str, Any]:
        """Move along the strongest local pheromone gradient."""
        if self.pheromone_system is None:
            return {
                "success": False,
                "error": "No pheromone_system is attached to this agent",
            }
        pheromone_type = decision.parameters.get("pheromone_type", "trail")
        magnitude, direction = self.pheromone_system.get_pheromone_gradient(
            self.position,
            pheromone_type,
            radius=float(decision.parameters.get("radius", 1.0)),
        )
        if magnitude <= 0:
            return {
                "success": False,
                "pheromone_type": pheromone_type,
                "actual_outcome": {"gradient_found": False},
            }
        old_position = self.position.copy()
        self.position = np.clip(
            self.position + np.asarray(direction) * self.movement_speed,
            [self.spatial_bounds["min_lat"], self.spatial_bounds["min_lng"]],
            [self.spatial_bounds["max_lat"], self.spatial_bounds["max_lng"]],
        )
        return {
            "success": True,
            "old_position": old_position,
            "new_position": self.position.copy(),
            "distance_moved": float(np.linalg.norm(self.position - old_position)),
            "pheromone_type": pheromone_type,
            "energy_cost": 0.05,
            "actual_outcome": {"gradient_found": True},
        }

    async def _execute_communication_action(
        self, decision: ActionDecision
    ) -> dict[str, Any]:
        """Execute communication actions."""
        params = decision.parameters
        message_type = params.get("message_type", "status")

        # Create communication message
        message = {
            "from": self.agent_id,
            "type": message_type,
            "content": {
                "position": self.position.tolist(),
                "energy_level": self.energy_level,
                "current_task": getattr(self, "current_task", "none"),
            },
            "timestamp": datetime.now(),
        }

        return {
            "message": message,
            "recipients": params.get("recipients", "all_nearby"),
            "energy_cost": 0.02,
            "actual_outcome": {"message_sent": True},
        }

    async def _execute_foraging_action(
        self, decision: ActionDecision
    ) -> dict[str, Any]:
        """Execute foraging actions."""
        params = decision.parameters
        target_type = params.get("target_type", "food")

        success_probability = float(params.get("success_probability", 0.7))
        if not 0 <= success_probability <= 1:
            raise ValueError("success_probability must be between 0 and 1")
        success = bool(self.rng.random() < success_probability)

        energy_gain = 0.3 if success else 0.0
        self.energy_level = min(1.0, self.energy_level + energy_gain)

        return {
            "target_type": target_type,
            "success": success,
            "energy_gain": energy_gain,
            "energy_cost": 0.2,
            "actual_outcome": {"foraging_completed": success},
        }

    async def _execute_rest_action(self, decision: ActionDecision) -> dict[str, Any]:
        """Execute rest/recovery actions."""
        params = decision.parameters
        duration = params.get("duration", 10)

        # Simulate rest and energy recovery
        recovery_rate = 0.1
        energy_recovered = min(duration * recovery_rate / 10, 1.0 - self.energy_level)
        self.energy_level += energy_recovered

        return {
            "duration": duration,
            "energy_recovered": energy_recovered,
            "final_energy": self.energy_level,
            "energy_cost": 0.0,
            "actual_outcome": {"rest_completed": True},
        }

    async def _execute_monitoring_action(
        self, decision: ActionDecision
    ) -> dict[str, Any]:
        """Execute environmental monitoring actions."""
        params = decision.parameters
        sensor_types = params.get("sensor_types", ["general"])

        # Simulate sensor readings
        readings = {}
        for sensor in sensor_types:
            if sensor == "temperature":
                readings["temperature"] = self.rng.normal(20, 5)
            elif sensor == "humidity":
                readings["humidity"] = self.rng.uniform(30, 80)
            elif sensor == "general":
                readings["environmental_quality"] = self.rng.uniform(0.5, 1.0)

        return {
            "sensor_types": sensor_types,
            "readings": readings,
            "location": self.position,
            "energy_cost": 0.1,
            "actual_outcome": {"monitoring_completed": True},
        }

    async def _execute_generic_action(self, decision: ActionDecision) -> dict[str, Any]:
        """Reject actions without a registered handler."""
        return {
            "action_type": decision.action_type,
            "parameters": decision.parameters,
            "success": False,
            "energy_cost": 0.0,
            "error": f"Unsupported action type: {decision.action_type}",
            "actual_outcome": {"generic_action_completed": False},
        }

    def _update_agent_state(
        self, decision: ActionDecision, result: dict[str, Any]
    ) -> None:
        """Update agent internal state based on action execution."""
        # Update energy level
        energy_cost = result.get("energy_cost", 0.0)
        self.energy_level = max(0.0, self.energy_level - energy_cost)

        # Update task memory
        self.task_memory.append(
            {
                "action": decision.action_type,
                "result": result,
                "timestamp": datetime.now(),
            }
        )

        # Keep memory within capacity
        max_memory = self.config.get("memory_capacity", 50)
        if len(self.task_memory) > max_memory:
            self.task_memory.pop(0)

        # Update state beliefs
        if hasattr(self, "state"):
            self.state.update_belief("last_action", decision.action_type)
            self.state.update_belief("energy_level", self.energy_level)
            self.state.update_belief("position", self.position.tolist())

    # Abstract methods for subclasses (if not using BaseAgent)
    async def initialize(self) -> None:
        """Initialize agent before running."""
        logger.info(f"SwarmAgent {self.agent_id} initializing")
        if hasattr(self, "state"):
            self.state.update_belief("status", "initialized")

    async def perceive(self) -> dict[str, Any]:
        """Default perception method."""
        return {
            "position": self.position,
            "energy_level": self.energy_level,
            "timestamp": datetime.now(),
        }

    def update_beliefs(self, perception: dict[str, Any]) -> None:
        """Update beliefs based on perception."""
        for key, value in perception.items():
            if hasattr(self, "state"):
                self.state.update_belief(key, value)

    async def decide(self) -> dict[str, Any] | None:
        """Default decision method."""
        return {"type": "monitor", "parameters": {}}

    async def act(self, action: dict[str, Any]) -> dict[str, Any]:
        """Default action method."""
        return {"status": "completed"}

    async def shutdown(self) -> None:
        """Clean up resources."""
        logger.info(f"SwarmAgent {self.agent_id} shutting down")

    def to_dict(self) -> dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "agent_id": self.agent_id,
            "position": self.position.tolist(),
            "sensory_range": self.sensory_range,
            "movement_speed": self.movement_speed,
            "energy_level": self.energy_level,
            "active_inference_enabled": self.active_inference_enabled,
            "config": self.config,
            "state": self.state.to_dict()
            if getattr(self, "state", None) is not None
            and hasattr(self.state, "to_dict")
            else {},
        }
