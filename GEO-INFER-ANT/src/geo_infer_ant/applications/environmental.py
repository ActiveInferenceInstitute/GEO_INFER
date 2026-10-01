"""
Environmental Monitoring Swarm Applications for GEO-INFER-ANT

This module implements specialized swarm applications for environmental monitoring,
including air quality, water quality, biodiversity, and ecological system monitoring.
The applications integrate multiple swarm intelligence algorithms with real-time
data collection and adaptive sampling strategies.

Key Features:
- Multi-objective environmental monitoring
- Adaptive sampling based on environmental variability
- Real-time sensor data integration
- Biodiversity and ecosystem health assessment
- Pollution tracking and source identification
- Integration with IoT sensor networks
"""

import numpy as np
import logging
from typing import Any, cast
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

# Integration imports: geo_infer_space is optional (guarded with a None
# sentinel consumed at call sites); geo_infer_ant submodules ship with this
# module and must import unconditionally.
try:
    from geo_infer_space.core.spatial_indexing import (
        SpatialIndexingInterface,  # noqa: F401
    )  # noqa: F401
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface  # noqa: F401
except ImportError as e:
    logging.getLogger(__name__).debug("Optional spatial integration unavailable: %s", e)
    SpatialIndexingInterface = None
    SpatialAnalyticsInterface = None

from geo_infer_ant.core.agent_base import SwarmAgent
from geo_infer_ant.core.stigmergy import PheromoneSystem
from geo_infer_ant.core.digital_stigmergy import DigitalStigmergy
from geo_infer_ant.algorithms.aco import AntColonyOptimization
from geo_infer_ant.algorithms.pso import ParticleSwarmOptimization
from geo_infer_ant.utils.spatial import validate_bounds

logger = logging.getLogger(__name__)


@dataclass
class MonitoringObjective:
    """Configuration for environmental monitoring objectives."""

    name: str
    sensor_types: list[str]
    priority: float = 1.0
    target_accuracy: float = 0.9
    temporal_resolution: str = "continuous"  # 'continuous', 'hourly', 'daily'
    spatial_resolution: str = "high"  # 'low', 'medium', 'high'
    alert_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass
class SensorReading:
    """Individual sensor reading from monitoring agent."""

    agent_id: str
    sensor_type: str
    value: float
    location: np.ndarray
    timestamp: datetime
    quality_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class EnvironmentalMonitoringSwarm:
    """
    Comprehensive environmental monitoring system using swarm intelligence.

    This application coordinates multiple types of monitoring agents to collect
    environmental data, detect anomalies, optimize sampling strategies, and
    provide real-time environmental intelligence.

    Key Features:
    - Multi-sensor environmental monitoring
    - Adaptive sampling strategies
    - Anomaly detection and alerting
    - Spatial interpolation and mapping
    - Real-time data integration
    - Multi-objective optimization
    """

    def __init__(
        self,
        swarm_size: int = 200,
        monitoring_objectives: list[str] | None = None,
        spatial_coverage: dict[str, float] | None = None,
        temporal_coverage: str = "continuous",
        adaptive_sampling: bool = True,
        real_time_processing: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize environmental monitoring swarm.

        Args:
            swarm_size: Number of monitoring agents
            monitoring_objectives: Types of environmental monitoring to perform
            spatial_coverage: Geographic area to monitor
            temporal_coverage: Temporal resolution of monitoring
            adaptive_sampling: Whether to use adaptive sampling strategies
            real_time_processing: Whether to process data in real-time
            **kwargs: Additional configuration parameters
        """
        if not isinstance(swarm_size, int) or swarm_size <= 0:
            raise ValueError("swarm_size must be a positive integer")
        self.swarm_size = swarm_size
        self.monitoring_objectives = monitoring_objectives or [
            "air_quality",
            "water_quality",
            "biodiversity",
        ]
        self.spatial_coverage = spatial_coverage or {
            "min_lat": 35,
            "max_lat": 40,
            "min_lng": -120,
            "max_lng": -115,
        }
        if validate_bounds is not None:
            self.spatial_coverage = validate_bounds(self.spatial_coverage)
        self.temporal_coverage = temporal_coverage
        self.adaptive_sampling = adaptive_sampling
        self.real_time_processing = real_time_processing
        self.random_seed = kwargs.pop("random_seed", kwargs.pop("seed", None))
        self.rng = np.random.default_rng(self.random_seed)
        self.sensor_range = float(kwargs.pop("sensor_range", 0.005))
        if not np.isfinite(self.sensor_range) or self.sensor_range <= 0:
            raise ValueError("sensor_range must be finite and positive")

        # Monitoring system state
        self.monitoring_agents: list[SwarmAgent] = []
        self.pheromone_system: PheromoneSystem | None = None
        self.digital_stigmergy: DigitalStigmergy | None = None
        self.spatial_analytics: Any | None = None

        # Data collection and analysis
        self.sensor_data: list[SensorReading] = []
        self.anomaly_history: list[dict[str, Any]] = []
        self.coverage_maps: dict[str, np.ndarray] = {}
        self.alerts: list[dict[str, Any]] = []

        # Optimization components
        self.sampling_optimizer: AntColonyOptimization | None = None
        self.coverage_optimizer: ParticleSwarmOptimization | None = None

        # Performance tracking
        self.monitoring_efficiency: float = 0.0
        self.coverage_quality: float = 0.0
        self.anomaly_detection_rate: float = 0.0

        # Initialize system components
        self._initialize_monitoring_system()

        logger.info(
            f"EnvironmentalMonitoringSwarm initialized with {swarm_size} agents"
        )

    def _initialize_monitoring_system(self) -> None:
        """Initialize all monitoring system components."""
        # Initialize pheromone system for coordination
        if PheromoneSystem is not None:
            try:
                self.pheromone_system = PheromoneSystem(
                    spatial_resolution="h3_r8",
                    pheromone_types=["monitoring", "anomaly", "coverage", "priority"],
                    bounds=self.spatial_coverage,
                )
                logger.info("Pheromone system initialized for environmental monitoring")
            except Exception as e:
                logger.warning(f"Failed to initialize pheromone system: {e}")

        # Initialize digital stigmergy for information sharing
        if DigitalStigmergy is not None:
            try:
                self.digital_stigmergy = DigitalStigmergy(
                    communication_medium="iot_network",
                    information_types=[
                        "sensor_data",
                        "anomaly_detection",
                        "coverage_info",
                        "alerts",
                    ],
                    persistence_model="temporal_decay",
                )
                logger.info(
                    "Digital stigmergy initialized for environmental monitoring"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize digital stigmergy: {e}")

        # Initialize optimization algorithms
        if AntColonyOptimization is not None:
            try:
                self.sampling_optimizer = AntColonyOptimization(
                    number_of_ants=30, max_iterations=50, variant="ACS"
                )
                logger.info("Sampling optimizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize sampling optimizer: {e}")

        if ParticleSwarmOptimization is not None:
            try:
                self.coverage_optimizer = ParticleSwarmOptimization(
                    swarm_size=50,
                    dimensions=2,
                    bounds=[
                        (
                            self.spatial_coverage["min_lat"],
                            self.spatial_coverage["max_lat"],
                        ),
                        (
                            self.spatial_coverage["min_lng"],
                            self.spatial_coverage["max_lng"],
                        ),
                    ],
                    max_iterations=100,
                )
                logger.info("Coverage optimizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize coverage optimizer: {e}")

    async def deploy_agents(
        self,
        initial_positions: list[np.ndarray] | None = None,
        environmental_priorities: dict[str, float] | None = None,
        logistical_constraints: dict[str, Any] | None = None,
        communication_requirements: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Deploy monitoring agents across the target area.

        Args:
            initial_positions: Initial positions for agents (optimized if None)
            environmental_priorities: Priority areas for monitoring
            logistical_constraints: Constraints on agent deployment
            communication_requirements: Communication network requirements

        Returns:
            Deployment plan and agent configurations
        """
        logger.info(f"Deploying {self.swarm_size} monitoring agents")

        deployment_plan: dict[str, Any] = {
            "agents": [],
            "deployment_strategy": "optimized",
            "coverage_achieved": 0.0,
            "deployment_time": datetime.now(),
            "agent_configurations": {},
        }

        # Optimize initial positions if not provided
        if initial_positions is None:
            initial_positions = await self._optimize_initial_positions(
                environmental_priorities or {}, logistical_constraints or {}
            )

        # Create monitoring agents
        for i in range(self.swarm_size):
            agent_id = f"env_monitor_{i + 1:03d}"

            # Determine agent position
            position = (
                initial_positions[i]
                if i < len(initial_positions)
                else self._generate_random_position()
            )

            # Configure agent based on monitoring objectives
            agent_config = self._configure_monitoring_agent(
                agent_id, position, self.monitoring_objectives
            )

            # Keep the deployment record compatible with SwarmAgent creation;
            # the application also remains usable when that optional adapter is absent.
            agent_info = {
                "agent_id": agent_id,
                "position": position,
                "config": agent_config,
                "monitoring_objectives": self.monitoring_objectives,
                "deployment_time": datetime.now(),
            }

            deployment_plan["agents"].append(agent_info)
            deployment_plan["agent_configurations"][agent_id] = agent_config

        # Calculate expected coverage
        deployment_plan["coverage_achieved"] = self._calculate_deployment_coverage(
            deployment_plan["agents"]
        )

        logger.info(
            f"Agent deployment completed: {len(deployment_plan['agents'])} agents deployed"
        )
        return deployment_plan

    async def _optimize_initial_positions(
        self,
        environmental_priorities: dict[str, float],
        logistical_constraints: dict[str, Any],
    ) -> list[np.ndarray]:
        """Optimize initial positions for maximum coverage and priority alignment."""
        candidates = self._generate_grid_positions()
        forbidden_regions = logistical_constraints.get("forbidden_regions", [])
        if forbidden_regions:
            candidates = [
                candidate
                for candidate in candidates
                if not any(
                    self._point_in_region(candidate, region)
                    for region in forbidden_regions
                )
            ]
        if not candidates:
            raise ValueError("logistical constraints exclude every candidate position")

        # Greedy max-dispersion selection solves the multi-agent placement
        # objective directly: PSO optimizes one point at a time, whereas this
        # application must optimize a set of positions jointly.
        selected: list[np.ndarray] = []
        diagonal = np.hypot(
            self.spatial_coverage["max_lat"] - self.spatial_coverage["min_lat"],
            self.spatial_coverage["max_lng"] - self.spatial_coverage["min_lng"],
        )
        remaining = list(candidates)
        while len(selected) < self.swarm_size:

            def candidate_score(candidate: np.ndarray) -> float:
                priority = self._calculate_position_priority(
                    candidate, environmental_priorities
                )
                separation = (
                    min(np.linalg.norm(candidate - other) for other in selected)
                    if selected
                    else diagonal
                )
                return float(priority + min(1.0, separation / max(diagonal, 1e-12)))

            best_index = max(
                range(len(remaining)),
                key=lambda index: candidate_score(remaining[index]),
            )
            selected.append(remaining.pop(best_index))
            if not remaining:
                remaining = list(candidates)
        return [position.copy() for position in selected]

    @staticmethod
    def _point_in_region(point: np.ndarray, region: dict[str, Any]) -> bool:
        """Check a point against a rectangular or circular exclusion region."""
        if "center" in region and "radius" in region:
            center = np.asarray(region["center"], dtype=float)
            return bool(np.linalg.norm(point - center) <= float(region["radius"]))
        bounds = region.get("bounds", region)
        return bool(
            bounds.get("min_lat", -90) <= point[0] <= bounds.get("max_lat", 90)
            and bounds.get("min_lng", -180) <= point[1] <= bounds.get("max_lng", 180)
        )

    def _generate_grid_positions(self) -> list[np.ndarray]:
        """Generate grid-based initial positions."""
        positions = []

        # Calculate grid dimensions
        area_width = self.spatial_coverage["max_lng"] - self.spatial_coverage["min_lng"]
        area_height = (
            self.spatial_coverage["max_lat"] - self.spatial_coverage["min_lat"]
        )

        grid_cols = int(np.sqrt(self.swarm_size))
        grid_rows = (self.swarm_size + grid_cols - 1) // grid_cols

        lat_step = area_height / grid_rows if grid_rows > 1 else 0
        lng_step = area_width / grid_cols if grid_cols > 1 else 0

        for i in range(self.swarm_size):
            row = i // grid_cols
            col = i % grid_cols

            lat = self.spatial_coverage["min_lat"] + (row + 0.5) * lat_step
            lng = self.spatial_coverage["min_lng"] + (col + 0.5) * lng_step

            positions.append(np.array([lat, lng]))

        return positions

    def _generate_random_position(self) -> np.ndarray:
        """Generate random position within coverage area."""
        return np.array(
            [
                self.rng.uniform(
                    self.spatial_coverage["min_lat"], self.spatial_coverage["max_lat"]
                ),
                self.rng.uniform(
                    self.spatial_coverage["min_lng"], self.spatial_coverage["max_lng"]
                ),
            ]
        )

    def _configure_monitoring_agent(
        self, agent_id: str, position: np.ndarray, objectives: list[str]
    ) -> dict[str, Any]:
        """Configure individual monitoring agent."""
        config = {
            "agent_id": agent_id,
            "position": position,
            "monitoring_objectives": objectives,
            "sensory_capabilities": self._get_sensory_capabilities(objectives),
            "communication_range": 1000.0,  # meters
            "energy_capacity": 100.0,
            "sampling_frequency": self._get_sampling_frequency(),
            "data_quality_threshold": 0.8,
            "adaptive_behavior": self.adaptive_sampling,
        }

        return config

    def _get_sensory_capabilities(self, objectives: list[str]) -> list[str]:
        """Get sensory capabilities based on monitoring objectives."""
        capabilities = []

        for objective in objectives:
            if objective == "air_quality":
                capabilities.extend(
                    [
                        "pm25_sensor",
                        "no2_sensor",
                        "o3_sensor",
                        "temperature",
                        "humidity",
                    ]
                )
            elif objective == "water_quality":
                capabilities.extend(
                    [
                        "ph_sensor",
                        "turbidity_sensor",
                        "conductivity_sensor",
                        "temperature",
                    ]
                )
            elif objective == "biodiversity":
                capabilities.extend(
                    ["camera", "microphone", "species_detector", "habitat_sensor"]
                )
            elif objective == "soil_quality":
                capabilities.extend(
                    [
                        "moisture_sensor",
                        "ph_sensor",
                        "nutrient_sensor",
                        "compaction_sensor",
                    ]
                )

        return list(set(capabilities))  # Remove duplicates

    def _get_sampling_frequency(self) -> str:
        """Get sampling frequency based on temporal coverage."""
        frequency_map = {"continuous": "1_minute", "hourly": "1_hour", "daily": "1_day"}
        return frequency_map.get(self.temporal_coverage, "1_hour")

    def _calculate_position_priority(
        self, position: np.ndarray, priorities: dict[str, float]
    ) -> float:
        """Calculate priority score for a position."""
        if not priorities:
            return 0.0

        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite [lat, lng] coordinate")

        diagonal = np.hypot(
            self.spatial_coverage["max_lat"] - self.spatial_coverage["min_lat"],
            self.spatial_coverage["max_lng"] - self.spatial_coverage["min_lng"],
        )
        score = 0.0
        total_weight = 0.0
        for priority_type, raw_weight in priorities.items():
            if priority_type.endswith("_locations") or not isinstance(
                raw_weight, (int, float)
            ):
                continue
            weight = max(0.0, min(1.0, float(raw_weight)))
            total_weight += weight
            features: list[Any] = cast(
                list[Any], priorities.get(f"{priority_type}_locations", [])
            )
            if not features:
                score += weight
                continue
            feature_scores = []
            for feature in features:
                if isinstance(feature, dict):
                    location = feature.get("location", feature.get("center"))
                    radius = float(feature.get("radius", self.sensor_range))
                else:
                    location = feature
                    radius = self.sensor_range
                if location is None or radius <= 0:
                    continue
                feature_location = np.asarray(location, dtype=float)
                if feature_location.shape != (2,) or not np.all(
                    np.isfinite(feature_location)
                ):
                    continue
                distance = np.linalg.norm(position - feature_location)
                feature_scores.append(np.exp(-distance / max(radius, diagonal * 1e-6)))
            score += weight * (max(feature_scores) if feature_scores else 0.0)

        return float(score / total_weight) if total_weight else 0.0

    def _calculate_deployment_coverage(
        self, deployed_agents: list[dict[str, Any]]
    ) -> float:
        """Calculate expected coverage quality of deployment."""
        if not deployed_agents:
            return 0.0

        positions = np.array([agent["position"] for agent in deployed_agents])
        return self._calculate_monitoring_coverage(positions.tolist())

    async def coordinate_monitoring(
        self,
        agent_positions: list[np.ndarray],
        environmental_conditions: dict[str, Any] | None = None,
        data_priorities: dict[str, float] | None = None,
        energy_constraints: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Coordinate monitoring activities across all agents.

        Args:
            agent_positions: Current positions of all agents
            environmental_conditions: Current environmental conditions
            data_priorities: Priority levels for different data types
            energy_constraints: Energy limitations for agents

        Returns:
            Coordinated monitoring plan and instructions
        """
        logger.info(f"Coordinating monitoring for {len(agent_positions)} agents")

        coordination_plan: dict[str, Any] = {
            "monitoring_instructions": {},
            "sampling_strategy": "adaptive" if self.adaptive_sampling else "uniform",
            "communication_protocol": "pheromone_digital_hybrid",
            "coordination_time": datetime.now(),
            "estimated_coverage": 0.0,
            "priority_areas": [],
        }

        # Update environmental conditions in pheromone system
        if self.pheromone_system and environmental_conditions:
            await self.pheromone_system.diffuse_pheromones(
                time_step=60.0,  # 1 minute
                environmental_conditions=environmental_conditions,
            )

        # Generate adaptive sampling strategy
        if self.adaptive_sampling:
            sampling_strategy = await self._generate_adaptive_sampling_strategy(
                agent_positions, environmental_conditions, data_priorities
            )
            coordination_plan["sampling_strategy"] = sampling_strategy

        # Generate communication instructions
        communication_plan = self._generate_communication_plan(agent_positions)
        coordination_plan.update(communication_plan)

        # Calculate estimated coverage
        coordination_plan["estimated_coverage"] = self._calculate_monitoring_coverage(
            agent_positions
        )

        # Identify priority monitoring areas
        if data_priorities:
            priority_areas = self._identify_priority_areas(data_priorities)
            coordination_plan["priority_areas"] = priority_areas

        logger.info(
            f"Monitoring coordination completed: {len(coordination_plan['monitoring_instructions'])} instructions generated"
        )
        return coordination_plan

    async def _generate_adaptive_sampling_strategy(
        self,
        agent_positions: list[np.ndarray],
        environmental_conditions: dict[str, Any] | None = None,
        data_priorities: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Generate adaptive sampling strategy based on current conditions."""
        strategy = {
            "strategy_type": "adaptive",
            "sampling_zones": {},
            "agent_assignments": {},
            "sampling_frequencies": {},
        }

        try:
            zones = self._create_sampling_zones(
                agent_positions, environmental_conditions or {}
            )
            strategy["sampling_zones"] = zones

            # Assign agents to zones
            assignments = self._assign_agents_to_zones(agent_positions, zones)
            strategy["agent_assignments"] = assignments

            # Set sampling frequencies based on priorities and conditions
            strategy["sampling_frequencies"] = self._calculate_sampling_frequencies(
                data_priorities or {}, environmental_conditions or {}
            )

        except Exception as e:
            logger.warning(f"Adaptive sampling strategy generation failed: {e}")
            strategy["strategy_type"] = "uniform"
            strategy["error"] = str(e)

        return strategy

    def _create_sampling_zones(
        self, agent_positions: list[np.ndarray], conditions: dict[str, Any]
    ) -> dict[str, Any]:
        """Create sampling zones based on environmental variability."""
        custom_zones = conditions.get("sampling_zones", []) if conditions else []
        if custom_zones:
            zones = {}
            for index, zone in enumerate(custom_zones):
                bounds = validate_bounds(zone["bounds"])
                zones[zone.get("zone_id", f"zone_{index + 1}")] = {
                    "bounds": bounds,
                    "required_agents": max(0, int(zone.get("required_agents", 0))),
                    "sampling_frequency": zone.get("sampling_frequency", "15_minutes"),
                }
            return zones

        # Build three non-overlapping latitude bands when no external priority
        # map is supplied.  Required counts are proportional to band area and
        # always sum to the available agents.
        min_lat = self.spatial_coverage["min_lat"]
        max_lat = self.spatial_coverage["max_lat"]
        band_edges = np.linspace(min_lat, max_lat, 4)
        counts = [
            int(np.ceil(len(agent_positions) / 2)),
            int(np.floor(len(agent_positions) / 3)),
        ]
        counts.append(max(0, len(agent_positions) - sum(counts)))
        priorities = ["high_priority", "medium_priority", "low_priority"]
        frequencies = ["5_minutes", "15_minutes", "1_hour"]
        return {
            name: {
                "bounds": {
                    "min_lat": float(band_edges[index]),
                    "max_lat": float(band_edges[index + 1]),
                    "min_lng": self.spatial_coverage["min_lng"],
                    "max_lng": self.spatial_coverage["max_lng"],
                },
                "required_agents": count,
                "sampling_frequency": frequencies[index],
            }
            for index, (name, count) in enumerate(zip(priorities, counts))
        }

    def _assign_agents_to_zones(
        self, agent_positions: list[np.ndarray], zones: dict[str, Any]
    ) -> dict[str, list[str]]:
        """Assign agents to sampling zones."""
        assignments: dict[str, list[Any]] = {
            zone_name: [] for zone_name in zones.keys()
        }

        # Simple assignment based on position
        for i, position in enumerate(agent_positions):
            agent_id = f"agent_{i}"

            # Assign to zone with highest priority that needs more agents
            for zone_name, zone_info in zones.items():
                if len(assignments[zone_name]) < zone_info["required_agents"]:
                    assignments[zone_name].append(agent_id)
                    break

        return assignments

    def _calculate_sampling_frequencies(
        self, data_priorities: dict[str, float], conditions: dict[str, Any]
    ) -> dict[str, str]:
        """Calculate optimal sampling frequencies."""
        frequencies = {}

        for objective, priority in data_priorities.items():
            if priority > 0.8:
                frequencies[objective] = "1_minute"
            elif priority > 0.5:
                frequencies[objective] = "5_minutes"
            elif priority > 0.4:
                frequencies[objective] = "15_minutes"
            else:
                frequencies[objective] = "1_hour"

        return frequencies

    def _generate_communication_plan(
        self, agent_positions: list[np.ndarray]
    ) -> dict[str, Any]:
        """Generate communication plan for agents."""
        plan: dict[str, Any] = {
            "communication_instructions": {},
            "information_sharing_rules": {},
            "coordination_signals": {},
        }

        # Define communication rules
        plan["information_sharing_rules"] = {
            "share_anomalies": True,
            "share_sensor_data": True,
            "share_position_updates": True,
            "coordinate_sampling": self.adaptive_sampling,
        }

        # Generate coordination signals via pheromones
        if self.pheromone_system:
            plan["coordination_signals"]["pheromone_types"] = [
                "monitoring",
                "anomaly",
                "coverage",
            ]

        # Generate digital stigmergy coordination
        if self.digital_stigmergy:
            plan["coordination_signals"]["digital_types"] = [
                "sensor_data",
                "anomaly_detection",
                "coverage_info",
            ]

        return plan

    def _calculate_monitoring_coverage(
        self, agent_positions: list[np.ndarray]
    ) -> float:
        """Calculate current monitoring coverage quality."""
        if not agent_positions:
            return 0.0

        positions = np.asarray(agent_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("agent_positions must be an array of [lat, lng] points")
        covered_area = self._estimate_covered_area(positions)
        total_area = self._calculate_total_area()
        return float(np.clip(covered_area / total_area, 0.0, 1.0))

    def _estimate_covered_area(self, positions: np.ndarray) -> float:
        """Estimate area covered by agent sensors."""
        if positions.size == 0:
            return 0.0
        inside = positions[
            (positions[:, 0] >= self.spatial_coverage["min_lat"])
            & (positions[:, 0] <= self.spatial_coverage["max_lat"])
            & (positions[:, 1] >= self.spatial_coverage["min_lng"])
            & (positions[:, 1] <= self.spatial_coverage["max_lng"])
        ]
        if inside.size == 0:
            return 0.0
        radius = self.sensor_range
        covered_area = len(inside) * np.pi * radius**2
        # Subtract pairwise overlap for equal-radius sensor footprints. This
        # remains positive for sparse deployments where a raster would miss
        # every small footprint, while accounting for co-located sensors.
        pair_dists = np.linalg.norm(inside[:, None, :] - inside[None, :, :], axis=-1)
        first_idx, second_idx = np.triu_indices(inside.shape[0], k=1)
        distances = pair_dists[first_idx, second_idx]
        close = distances < 2.0 * radius
        if close.any():
            close_dists = distances[close]
            ratio = np.clip(close_dists / (2.0 * radius), -1.0, 1.0)
            overlaps = 2.0 * radius**2 * np.arccos(ratio) - 0.5 * close_dists * np.sqrt(
                np.maximum(0.0, 4.0 * radius**2 - close_dists**2)
            )
            overlaps[close_dists == 0.0] = np.pi * radius**2
            for overlap in overlaps:
                covered_area -= overlap
        return float(np.clip(covered_area, 0.0, self._calculate_total_area()))

    def _calculate_total_area(self) -> float:
        """Calculate total area of monitoring region."""
        width = self.spatial_coverage["max_lng"] - self.spatial_coverage["min_lng"]
        height = self.spatial_coverage["max_lat"] - self.spatial_coverage["min_lat"]
        return width * height

    def _identify_priority_areas(
        self, data_priorities: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Identify priority areas for focused monitoring."""
        priority_areas = []

        for priority_type, priority_level in data_priorities.items():
            if priority_level > 0.6:  # High priority threshold
                # Define priority area based on type
                if priority_type == "air_quality":
                    priority_area = {
                        "type": "air_quality_hotspot",
                        "bounds": self.spatial_coverage,  # Would be more specific in practice
                        "priority_level": priority_level,
                        "monitoring_frequency": "5_minutes",
                    }
                elif priority_type == "water_quality":
                    priority_area = {
                        "type": "water_body",
                        "bounds": self.spatial_coverage,
                        "priority_level": priority_level,
                        "monitoring_frequency": "15_minutes",
                    }
                else:
                    priority_area = {
                        "type": priority_type,
                        "bounds": self.spatial_coverage,
                        "priority_level": priority_level,
                        "monitoring_frequency": "15_minutes",
                    }

                priority_areas.append(priority_area)

        return priority_areas

    async def process_collective_intelligence(
        self,
        individual_measurements: list[SensorReading],
        spatial_interpolation: str = "kriging",
        uncertainty_quantification: str = "bayesian",
        anomaly_detection: str = "statistical",
    ) -> dict[str, Any]:
        """
        Process collective environmental intelligence from all agents.

        Args:
            individual_measurements: Sensor readings from all agents
            spatial_interpolation: Method for spatial interpolation
            uncertainty_quantification: Method for uncertainty analysis
            anomaly_detection: Method for anomaly detection

        Returns:
            Comprehensive environmental assessment
        """
        logger.info(
            f"Processing collective intelligence from {len(individual_measurements)} measurements"
        )

        # Convert dict readings to SensorReading objects if needed
        converted = []
        for m in individual_measurements:
            if isinstance(m, dict):
                converted.append(
                    SensorReading(
                        agent_id=m.get("agent_id", "unknown"),
                        sensor_type=m.get("sensor_type", "unknown"),
                        value=float(m.get("value", 0.0)),
                        location=np.array(m.get("location", [0, 0])),
                        timestamp=m.get("timestamp", datetime.now()),
                        quality_score=float(m.get("quality_score", 1.0)),
                        metadata=m.get("metadata", {}),
                    )
                )
            else:
                converted.append(m)
        individual_measurements = converted

        assessment: dict[str, Any] = {
            "assessment_time": datetime.now(),
            "data_summary": self._summarize_measurements(individual_measurements),
            "spatial_analysis": {},
            "anomaly_detection": {},
            "uncertainty_analysis": {},
            "recommendations": [],
        }

        try:
            # Spatial analysis and interpolation
            if spatial_interpolation != "none" and self.spatial_analytics:
                spatial_results = await self._perform_spatial_analysis(
                    individual_measurements, spatial_interpolation
                )
                assessment["spatial_analysis"] = spatial_results

            # Anomaly detection
            if anomaly_detection != "none":
                anomalies = await self._detect_anomalies(
                    individual_measurements, anomaly_detection
                )
                assessment["anomaly_detection"] = anomalies

                # Update anomaly history
                self.anomaly_history.extend(anomalies)

            # Uncertainty quantification
            if uncertainty_quantification != "none":
                uncertainty = self._quantify_uncertainty(
                    individual_measurements, uncertainty_quantification
                )
                assessment["uncertainty_analysis"] = uncertainty

            # Generate recommendations
            recommendations = self._generate_monitoring_recommendations(assessment)
            assessment["recommendations"] = recommendations

            # Update performance metrics
            self._update_performance_metrics(individual_measurements, assessment)

        except Exception as e:
            logger.error(f"Collective intelligence processing failed: {e}")
            assessment["error"] = str(e)

        logger.info(
            f"Collective intelligence assessment completed: {len(assessment['recommendations'])} recommendations"
        )
        return assessment

    def _summarize_measurements(
        self, measurements: list[SensorReading]
    ) -> dict[str, Any]:
        """Summarize measurement data."""
        if not measurements:
            return {"total_measurements": 0}

        summary: dict[str, Any] = {
            "total_measurements": len(measurements),
            "measurement_types": list(set([m.sensor_type for m in measurements])),
            "time_range": {
                "start": min([m.timestamp for m in measurements]),
                "end": max([m.timestamp for m in measurements]),
            },
            "spatial_range": {
                "min_lat": min([m.location[0] for m in measurements]),
                "max_lat": max([m.location[0] for m in measurements]),
                "min_lng": min([m.location[1] for m in measurements]),
                "max_lng": max([m.location[1] for m in measurements]),
            },
        }

        # Statistical summaries by sensor type
        sensor_stats = defaultdict(list)
        for measurement in measurements:
            sensor_stats[measurement.sensor_type].append(measurement.value)

        summary["sensor_statistics"] = {}
        for sensor_type, values in sensor_stats.items():
            finite_values = np.asarray(values, dtype=float)
            finite_values = finite_values[np.isfinite(finite_values)]
            if finite_values.size == 0:
                finite_values = np.array([0.0])
            summary["sensor_statistics"][sensor_type] = {
                "count": len(values),
                "mean": float(np.mean(finite_values)),
                "std": float(np.std(finite_values)),
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
            }

        return summary

    async def _perform_spatial_analysis(
        self, measurements: list[SensorReading], interpolation_method: str
    ) -> dict[str, Any]:
        """Perform spatial analysis and interpolation."""
        try:
            # Group measurements by type
            measurements_by_type = defaultdict(list)
            for measurement in measurements:
                measurements_by_type[measurement.sensor_type].append(measurement)

            spatial_results = {}

            for sensor_type, type_measurements in measurements_by_type.items():
                if len(type_measurements) < 2:
                    continue

                # Extract locations and values
                locations = np.array([m.location for m in type_measurements])
                values = np.array([m.value for m in type_measurements])

                if interpolation_method == "kriging":
                    interpolated_field = self._simple_kriging_interpolation(
                        locations, values
                    )
                elif interpolation_method == "idw":
                    # Inverse distance weighting
                    interpolated_field = self._inverse_distance_weighting(
                        locations, values
                    )
                else:
                    interpolated_field = {"method": "none", "values": values}

                spatial_results[sensor_type] = {
                    "interpolation_method": interpolation_method,
                    "interpolated_field": interpolated_field,
                    "sample_points": len(type_measurements),
                    "spatial_coverage": self._calculate_spatial_coverage(locations),
                }

            return spatial_results

        except Exception as e:
            logger.warning(f"Spatial analysis failed: {e}")
            return {"error": str(e)}

    def _simple_kriging_interpolation(
        self, locations: np.ndarray, values: np.ndarray
    ) -> dict[str, Any]:
        """
        Estimate the value at the sample centroid with an ordinary kriging system.

        The returned field is the estimate at the representative target point;
        callers can use the weights and variogram parameters to reproduce the
        estimate at other target points.
        """
        locations = np.asarray(locations, dtype=float)
        values = np.asarray(values, dtype=float)
        finite = np.all(np.isfinite(locations), axis=1) & np.isfinite(values)
        locations = locations[finite]
        values = values[finite]
        if len(locations) < 2:
            return {
                "method": "simple_kriging",
                "estimated_field": np.mean(values) if len(values) > 0 else 0.0,
                "variance": np.var(values) if len(values) > 0 else 0.0,
            }

        # Calculate distances between all points
        n_points = len(locations)
        distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                distances[i, j] = np.sqrt(np.sum((locations[i] - locations[j]) ** 2))

        # Estimate variogram parameters
        # Use empirical variogram to estimate sill, range, and nugget
        nonzero_distances = distances[distances > 0]
        max_distance: float = (
            cast(float, np.max(nonzero_distances)) if nonzero_distances.size else 0.0
        )
        sill: float = cast(float, max(cast(float, np.var(values)), np.finfo(float).eps))
        range_param: float = cast(float, max(max_distance * 0.3, np.finfo(float).eps))
        nugget = sill * 0.1

        # Spherical variogram function
        def spherical_variogram(h: Any) -> np.ndarray:
            """Spherical variogram model."""
            h = np.asarray(h, dtype=float)
            result = np.zeros_like(h)
            mask = h <= range_param
            result[mask] = nugget + (sill - nugget) * (
                1.5 * h[mask] / range_param - 0.5 * (h[mask] / range_param) ** 3
            )
            result[~mask] = sill
            return np.asarray(result)

        # Calculate variogram matrix for known points
        variogram_matrix = spherical_variogram(distances)
        np.fill_diagonal(variogram_matrix, 0.0)
        kriging_matrix = np.empty((n_points + 1, n_points + 1), dtype=float)
        kriging_matrix[:-1, :-1] = variogram_matrix
        kriging_matrix[:-1, -1] = 1.0
        kriging_matrix[-1, :-1] = 1.0
        kriging_matrix[-1, -1] = 0.0
        target = np.mean(locations, axis=0)
        target_distances = np.linalg.norm(locations - target, axis=1)
        rhs = np.concatenate((spherical_variogram(target_distances), [1.0]))

        try:
            solution = np.linalg.solve(kriging_matrix, rhs)
            weights = solution[:-1]
            lagrange_multiplier = solution[-1]
        except np.linalg.LinAlgError:
            weights = np.ones(n_points, dtype=float) / n_points
            lagrange_multiplier = 0.0

        estimated_field = float(np.dot(weights, values))
        kriging_variance = max(
            0.0, float(np.dot(weights, rhs[:-1]) + lagrange_multiplier)
        )

        return {
            "method": "ordinary_kriging",
            "target": target.tolist(),
            "estimated_field": estimated_field,
            "variance": float(kriging_variance),
            "variogram_params": {
                "sill": float(sill),
                "range": float(range_param),
                "nugget": float(nugget),
            },
            "weights": weights.tolist(),
        }

    def _inverse_distance_weighting(
        self, locations: np.ndarray, values: np.ndarray, power: float = 2.0
    ) -> dict[str, Any]:
        """
        Inverse distance weighting interpolation.

        Implements IDW with configurable power parameter for distance decay.
        """
        if len(locations) < 2:
            return {
                "method": "inverse_distance_weighting",
                "estimated_field": np.mean(values) if len(values) > 0 else 0.0,
                "weights": [1.0] if len(locations) > 0 else [],
            }

        # Calculate centroid (for estimation point)
        centroid = np.mean(locations, axis=0)

        # Calculate distances from centroid to all points
        distances = np.array(
            [np.sqrt(np.sum((loc - centroid) ** 2)) for loc in locations]
        )

        # Avoid division by zero for points at exact same location
        min_distance = 1e-10
        distances = np.maximum(distances, min_distance)

        # Calculate inverse distance weights
        inv_distances = 1.0 / (distances**power)
        weights = inv_distances / np.sum(inv_distances)

        # Calculate weighted average
        estimated_field = np.sum(weights * values)

        # Calculate variance estimate
        variance = np.sum(weights * (values - estimated_field) ** 2)

        return {
            "method": "inverse_distance_weighting",
            "estimated_field": float(estimated_field),
            "variance": float(variance),
            "weights": weights.tolist(),
            "power": power,
            "distances": distances.tolist(),
        }

    def _calculate_spatial_coverage(self, locations: np.ndarray) -> float:
        """Calculate spatial coverage of measurement locations."""
        if len(locations) < 2:
            return 1.0

        # Simple coverage calculation based on area spanned
        lat_range = np.max(locations[:, 0]) - np.min(locations[:, 0])
        lng_range = np.max(locations[:, 1]) - np.min(locations[:, 1])

        total_range = lat_range + lng_range
        max_range = (
            self.spatial_coverage["max_lat"]
            - self.spatial_coverage["min_lat"]
            + self.spatial_coverage["max_lng"]
            - self.spatial_coverage["min_lng"]
        )

        return float(min(1.0, total_range / max_range))

    async def _detect_anomalies(
        self, measurements: list[SensorReading], detection_method: str
    ) -> list[dict[str, Any]]:
        """Detect anomalies in environmental measurements."""
        anomalies = []

        try:
            # Group measurements by type for anomaly detection
            measurements_by_type = defaultdict(list)
            for measurement in measurements:
                measurements_by_type[measurement.sensor_type].append(measurement)

            for sensor_type, type_measurements in measurements_by_type.items():
                if len(type_measurements) < 5:
                    continue  # Need sufficient data for anomaly detection

                values = np.array([m.value for m in type_measurements])
                _timestamps = [m.timestamp for m in type_measurements]
                _locations = np.array([m.location for m in type_measurements])

                # Statistical anomaly detection
                if detection_method == "statistical":
                    anomaly_indices = self._statistical_anomaly_detection(values)

                elif detection_method == "isolation_forest":
                    anomaly_indices = self._isolation_forest_anomaly_detection(values)

                elif detection_method == "zscore":
                    anomaly_indices = self._zscore_anomaly_detection(values)

                else:
                    anomaly_indices = []

                # Create anomaly records
                for idx in anomaly_indices:
                    measurement = type_measurements[idx]
                    anomaly = {
                        "anomaly_id": f"{sensor_type}_{measurement.agent_id}_{measurement.timestamp.isoformat()}",
                        "sensor_type": sensor_type,
                        "agent_id": measurement.agent_id,
                        "value": measurement.value,
                        "location": measurement.location,
                        "timestamp": measurement.timestamp,
                        "severity": self._calculate_anomaly_severity(
                            measurement.value, sensor_type
                        ),
                        "detection_method": detection_method,
                    }
                    anomalies.append(anomaly)

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")

        return anomalies

    def _statistical_anomaly_detection(self, values: np.ndarray) -> list[int]:
        """Statistical outlier detection using IQR method."""
        if len(values) < 4:
            return []

        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        anomaly_indices = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                anomaly_indices.append(i)

        return anomaly_indices

    def _isolation_forest_anomaly_detection(self, values: np.ndarray) -> list[int]:
        """Isolation forest anomaly detection using scikit-learn."""
        try:
            from sklearn.ensemble import IsolationForest

            if len(values) < 5:
                return []

            # Reshape for sklearn (needs 2D array)
            values_2d = values.reshape(-1, 1)

            # Fit Isolation Forest
            isolation_forest = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_estimators=100,
            )
            predictions = isolation_forest.fit_predict(values_2d)

            # Return indices where prediction is -1 (anomaly)
            anomaly_indices = [i for i, pred in enumerate(predictions) if pred == -1]

            return anomaly_indices

        except ImportError:
            # Fallback to statistical method if sklearn not available
            logger.warning(
                "scikit-learn not available, using statistical anomaly detection"
            )
            return self._statistical_anomaly_detection(values)
        except Exception as e:
            logger.warning(
                f"Isolation forest anomaly detection failed: {e}, using statistical method"
            )
            return self._statistical_anomaly_detection(values)

    def _zscore_anomaly_detection(self, values: np.ndarray) -> list[int]:
        """Z-score based anomaly detection."""
        if len(values) < 3:
            return []

        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return []

        anomaly_indices = []
        for i, value in enumerate(values):
            zscore = abs((value - mean_val) / std_val)
            if zscore > 3.0:  # 3-sigma threshold
                anomaly_indices.append(i)

        return anomaly_indices

    def _calculate_anomaly_severity(self, value: float, sensor_type: str) -> str:
        """Calculate severity level of anomaly."""
        # Define severity thresholds by sensor type
        severity_thresholds: dict[str, dict[str, float]] = {
            "pm25_sensor": {"low": 25, "medium": 50, "high": 100},
            "no2_sensor": {"low": 0.05, "medium": 0.1, "high": 0.2},
            "ph_sensor": {"low": 1.0, "medium": 2.0, "high": 3.0},
            "temperature": {
                "low": 5,
                "medium": 10,
                "high": 15,
            },  # deviation from normal
        }

        thresholds = severity_thresholds.get(
            sensor_type, {"low": 1, "medium": 2, "high": 3}
        )

        if abs(value) > thresholds["high"]:
            return "high"
        elif abs(value) > thresholds["medium"]:
            return "medium"
        elif abs(value) > thresholds["low"]:
            return "low"
        else:
            return "minimal"

    def _quantify_uncertainty(
        self, measurements: list[SensorReading], method: str
    ) -> dict[str, Any]:
        """Quantify uncertainty in measurements."""
        uncertainty: dict[str, Any] = {
            "method": method,
            "overall_uncertainty": 0.0,
            "sensor_uncertainties": {},
            "spatial_uncertainty": 0.0,
            "temporal_uncertainty": 0.0,
        }

        if not measurements:
            return uncertainty

        try:
            # Calculate sensor-specific uncertainties
            sensor_uncertainties = defaultdict(list)
            for measurement in measurements:
                sensor_uncertainties[measurement.sensor_type].append(
                    measurement.quality_score
                )

            for sensor_type, quality_scores in sensor_uncertainties.items():
                uncertainty["sensor_uncertainties"][sensor_type] = {
                    "mean_quality": np.mean(quality_scores),
                    "quality_std": np.std(quality_scores),
                    "uncertainty_score": 1.0
                    - np.mean(quality_scores),  # Convert quality to uncertainty
                }

            # Calculate spatial uncertainty (gaps in coverage)
            if len(measurements) > 1:
                locations = np.array([m.location for m in measurements])
                spatial_uncertainty = self._calculate_spatial_uncertainty(locations)
                uncertainty["spatial_uncertainty"] = spatial_uncertainty

            # Calculate temporal uncertainty (time gaps)
            if len(measurements) > 1:
                timestamps = [m.timestamp for m in measurements]
                temporal_uncertainty = self._calculate_temporal_uncertainty(timestamps)
                uncertainty["temporal_uncertainty"] = temporal_uncertainty

            # Overall uncertainty (weighted combination)
            weights = {"sensor": 0.5, "spatial": 0.3, "temporal": 0.2}
            sensor_scores = [
                v["uncertainty_score"]
                for v in uncertainty["sensor_uncertainties"].values()
                if isinstance(v, dict) and "uncertainty_score" in v
            ]
            overall = (
                (np.mean(sensor_scores) if sensor_scores else 0.0) * weights["sensor"]
                + uncertainty["spatial_uncertainty"] * weights["spatial"]
                + uncertainty["temporal_uncertainty"] * weights["temporal"]
            )
            uncertainty["overall_uncertainty"] = overall

        except Exception as e:
            logger.warning(f"Uncertainty quantification failed: {e}")
            uncertainty["error"] = str(e)

        return uncertainty

    def _calculate_spatial_uncertainty(self, locations: np.ndarray) -> float:
        """Calculate spatial uncertainty based on location distribution."""
        if len(locations) < 2:
            return 1.0

        # Calculate average nearest neighbor distance
        pair_dists = np.linalg.norm(
            locations[:, None, :] - locations[None, :, :], axis=-1
        )
        np.fill_diagonal(pair_dists, np.inf)
        avg_distance = np.mean(pair_dists.min(axis=1))

        # Uncertainty increases with distance (sparser coverage)
        # Normalize to 0-1 scale (assuming 0.01 degrees ≈ 1km)
        max_expected_distance = 0.01  # Expected distance for good coverage
        uncertainty = min(1.0, avg_distance / max_expected_distance)

        return float(uncertainty)

    def _calculate_temporal_uncertainty(self, timestamps: list[datetime]) -> float:
        """Calculate temporal uncertainty based on time gaps."""
        if len(timestamps) < 2:
            return 1.0

        # Calculate time gaps
        sorted_timestamps = sorted(timestamps)
        time_gaps = []

        for i in range(1, len(sorted_timestamps)):
            gap = (sorted_timestamps[i] - sorted_timestamps[i - 1]).total_seconds()
            time_gaps.append(gap)

        avg_gap = np.mean(time_gaps)

        # Expected gap based on sampling frequency
        expected_gaps = {
            "1_minute": 60,
            "5_minutes": 300,
            "15_minutes": 900,
            "1_hour": 3600,
        }

        expected_gap = expected_gaps.get(self._get_sampling_frequency(), 900)
        uncertainty = min(1.0, avg_gap / expected_gap)
        return cast(float, uncertainty)

    def _generate_monitoring_recommendations(
        self, assessment: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Generate recommendations for improving monitoring."""
        recommendations = []

        try:
            # Coverage-based recommendations
            coverage = assessment.get("spatial_analysis", {}).get("coverage", 0.5)
            if coverage < 0.7:
                recommendations.append(
                    {
                        "type": "coverage_improvement",
                        "priority": "high",
                        "description": f"Increase monitoring coverage from {coverage:.2f} to >0.8",
                        "actions": [
                            "deploy_additional_agents",
                            "optimize_agent_positions",
                        ],
                    }
                )

            # Anomaly-based recommendations
            anomalies = assessment.get("anomaly_detection", {})
            if anomalies and len(anomalies) > 0:
                high_severity_anomalies = [
                    a for a in anomalies if a.get("severity") == "high"
                ]
                if high_severity_anomalies:
                    recommendations.append(
                        {
                            "type": "anomaly_investigation",
                            "priority": "urgent",
                            "description": f"Investigate {len(high_severity_anomalies)} high-severity anomalies",
                            "actions": [
                                "increase_sampling_frequency",
                                "deploy_specialized_sensors",
                            ],
                        }
                    )

            # Uncertainty-based recommendations
            uncertainty = assessment.get("uncertainty_analysis", {}).get(
                "overall_uncertainty", 0.5
            )
            if uncertainty > 0.6:
                recommendations.append(
                    {
                        "type": "uncertainty_reduction",
                        "priority": "medium",
                        "description": f"Reduce measurement uncertainty from {uncertainty:.2f} to <0.4",
                        "actions": [
                            "calibrate_sensors",
                            "increase_redundancy",
                            "improve_spatial_coverage",
                        ],
                    }
                )

        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")

        return recommendations

    def _update_performance_metrics(
        self, measurements: list[SensorReading], assessment: dict[str, Any]
    ) -> None:
        """Update system performance metrics."""
        if not measurements:
            return

        # Calculate monitoring efficiency
        expected_measurements = len(self.monitoring_agents) * 10  # Expected per hour
        actual_measurements = len(measurements)
        self.monitoring_efficiency = min(
            1.0, actual_measurements / max(1, expected_measurements)
        )

        # Calculate coverage quality
        spatial_analysis = assessment.get("spatial_analysis", {})
        self.coverage_quality = spatial_analysis.get("coverage", 0.5)

        # Calculate anomaly detection rate
        anomalies = assessment.get("anomaly_detection", [])
        self.anomaly_detection_rate = (
            len(anomalies) / max(1, len(measurements)) * 1000
        )  # Per 1000 measurements

    def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring system status."""
        status = {
            "system_status": "operational",
            "monitoring_objectives": self.monitoring_objectives,
            "spatial_coverage": self.spatial_coverage,
            "temporal_coverage": self.temporal_coverage,
            "adaptive_sampling": self.adaptive_sampling,
            "performance_metrics": {
                "monitoring_efficiency": self.monitoring_efficiency,
                "coverage_quality": self.coverage_quality,
                "anomaly_detection_rate": self.anomaly_detection_rate,
            },
            "component_status": {
                "pheromone_system": self.pheromone_system is not None,
                "digital_stigmergy": self.digital_stigmergy is not None,
                "spatial_analytics": self.spatial_analytics is not None,
                "sampling_optimizer": self.sampling_optimizer is not None,
                "coverage_optimizer": self.coverage_optimizer is not None,
            },
            "data_summary": {
                "total_sensor_readings": len(self.sensor_data),
                "total_anomalies": len(self.anomaly_history),
                "active_alerts": len(self.alerts),
            },
        }

        return status
