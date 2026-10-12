"""
Pheromone-Based Stigmergic Communication for GEO-INFER-ANT

This module implements traditional pheromone-based stigmergic communication systems,
where agents indirectly coordinate through modifications of their shared environment.
These systems are inspired by ant colonies and other social insects.

Key Features:
- Multi-type pheromone systems (trail, food, alarm, nest)
- Spatially-aware pheromone diffusion and evaporation
- Environmental factor integration (wind, temperature, humidity)
- H3 spatial indexing integration for efficient spatial operations
- Real-time pheromone field updates
- Pheromone trail optimization and analysis
"""

import numpy as np
import logging
from typing import Any, cast
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import math

from geo_infer_ant.utils.spatial import parse_h3_resolution, validate_bounds

# Integration imports
try:
    from geo_infer_space.core.spatial_indexing import SpatialIndexingInterface
    from geo_infer_space.core.analytics import SpatialAnalyticsInterface
except ImportError as e:
    logging.getLogger(__name__).debug("Optional spatial integration unavailable: %s", e)
    SpatialIndexingInterface = None
    SpatialAnalyticsInterface = None

logger = logging.getLogger(__name__)


@dataclass
class PheromoneType:
    """Configuration for a specific pheromone type."""

    name: str
    evaporation_rate: float = 0.1  # Rate of pheromone decay per time unit
    diffusion_rate: float = 0.05  # Rate of spatial diffusion
    deposition_amount: float = 1.0  # Amount deposited by agents
    persistence_time: float = 300.0  # Maximum persistence time (seconds)
    max_intensity: float = 2.0  # Maximum allowed intensity
    min_intensity: float = 0.01  # Minimum detectable intensity

    # Environmental sensitivity
    wind_sensitivity: float = 0.5  # How much wind affects diffusion
    temperature_sensitivity: float = 0.3  # Temperature effect on evaporation
    humidity_sensitivity: float = 0.2  # Humidity effect on persistence

    def __post_init__(self) -> None:
        """Validate pheromone type configuration."""
        if not 0 < self.evaporation_rate <= 1:
            raise ValueError("Evaporation rate must be between 0 and 1")
        if not 0 <= self.diffusion_rate <= 1:
            raise ValueError("Diffusion rate must be between 0 and 1")


@dataclass
class PheromoneDeposit:
    """Record of a pheromone deposit by an agent."""

    agent_id: str
    pheromone_type: str
    intensity: float
    location: np.ndarray  # [lat, lng] coordinates
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate deposit after initialization."""
        if self.intensity <= 0:
            raise ValueError("Pheromone intensity must be positive")


@dataclass
class PheromoneField:
    """Spatial field representing pheromone concentrations."""

    pheromone_type: str
    spatial_resolution: str  # H3 resolution (e.g., 'h3_r8')
    bounds: dict[str, float]  # Spatial bounds

    # Pheromone concentration data
    concentrations: dict[str, float] = field(
        default_factory=dict
    )  # h3_cell_id -> concentration
    deposits: list[PheromoneDeposit] = field(default_factory=list)

    # Field metadata
    last_update: datetime = field(default_factory=datetime.now)
    update_count: int = 0

    # Integration references set by PheromoneSystem after field creation
    spatial_indexer: Any = field(default=None)
    pheromone_types: list[Any] = field(default_factory=list)

    def get_concentration(self, location: np.ndarray) -> float:
        """Get pheromone concentration at specific location."""
        if not self.spatial_indexer:
            return self._calculate_concentration_from_deposits(location)

        try:
            # Use spatial indexing to find relevant cells
            cell_id = self.spatial_indexer.latlng_to_cell(
                location[0], location[1], parse_h3_resolution(self.spatial_resolution)
            )
            return self.concentrations.get(cell_id, 0.0)
        except Exception as e:
            logger.warning(f"Failed to get concentration via spatial indexing: {e}")
            return self._calculate_concentration_from_deposits(location)

    def _calculate_concentration_from_deposits(self, location: np.ndarray) -> float:
        """Calculate concentration directly from recorded deposits."""
        if not self.deposits:
            return 0.0

        # Find deposits within range and calculate weighted concentration
        max_range = 1000.0  # meters
        total_concentration = 0.0
        total_weight = 0.0

        for deposit in self.deposits:
            distance = np.linalg.norm(location - deposit.location)

            if distance <= max_range:
                # Exponential decay with distance
                weight = math.exp(-distance / 100.0)  # 100m decay constant
                time_decay = self._calculate_time_decay(deposit.timestamp)

                concentration = deposit.intensity * weight * time_decay
                total_concentration += concentration
                total_weight += weight

        return total_concentration / max(total_weight, 1.0)

    def _calculate_time_decay(self, deposit_time: datetime) -> float:
        """Calculate time-based decay factor for a deposit."""
        time_elapsed = (datetime.now() - deposit_time).total_seconds()
        pheromone_type = next(
            (pt for pt in self.pheromone_types if pt.name == self.pheromone_type), None
        )

        if not pheromone_type:
            return 1.0

        # Exponential decay based on evaporation rate
        decay_factor = math.exp(
            -pheromone_type.evaporation_rate * time_elapsed / 60.0
        )  # per minute
        return max(decay_factor, 0.01)  # Minimum 1% of original intensity


class PheromoneSystem:
    """
    Comprehensive pheromone-based stigmergic communication system.

    Manages multiple pheromone types across spatial environments, handling
    deposition, diffusion, evaporation, and spatial querying of pheromone
    concentrations. Integrates with H3 spatial indexing for efficient
    geospatial operations.

    Key Features:
    - Multi-type pheromone management
    - Spatial diffusion and evaporation modeling
    - Environmental factor integration
    - H3 spatial indexing for performance
    - Real-time pheromone field updates
    - Pheromone trail optimization
    """

    def __init__(
        self,
        spatial_resolution: str = "h3_r8",
        pheromone_types: list[str] | None = None,
        bounds: dict[str, float] | None = None,
        environmental_factors: dict[str, Any] | None = None,
        spatial_backend: str = "h3",
        evaporation_rate: float | None = None,
    ):
        """
        Initialize pheromone communication system.

        Args:
            spatial_resolution: H3 resolution for spatial indexing
            pheromone_types: Types of pheromones to support
            bounds: Spatial bounds for the pheromone field
            environmental_factors: Environmental conditions affecting pheromones
            spatial_backend: Backend for spatial operations ('h3', 'srai', 'geopandas')
        """
        self.spatial_resolution = spatial_resolution
        self.h3_resolution = parse_h3_resolution(spatial_resolution)
        self.bounds = validate_bounds(
            bounds or {"min_lat": -90, "max_lat": 90, "min_lng": -180, "max_lng": 180}
        )
        self.environmental_factors = environmental_factors or {}
        self._evaporation_rate = evaporation_rate

        # Configure pheromone types
        self.pheromone_types = self._initialize_pheromone_types(
            pheromone_types or ["trail", "food", "alarm", "nest"]
        )

        # Pheromone fields for each type
        self.pheromone_fields: dict[str, PheromoneField] = {}

        # Integration components
        self.spatial_indexer: Any = None
        self.spatial_analytics: Any = None

        # Performance tracking
        self.performance_stats = {
            "deposits_total": 0,
            "updates_total": 0,
            "queries_total": 0,
            "avg_response_time": 0.0,
        }

        # Initialize spatial integration
        self._initialize_spatial_integration(spatial_backend)

        # Initialize pheromone fields
        self._initialize_pheromone_fields()

        logger.info(
            f"PheromoneSystem initialized with {len(self.pheromone_types)} pheromone types"
        )

    def _initialize_pheromone_types(
        self, pheromone_type_names: list[str]
    ) -> list[PheromoneType]:
        """Initialize pheromone type configurations."""
        types = []

        # Default configurations for common pheromone types
        default_configs = {
            "trail": {
                "evaporation_rate": 0.1,
                "diffusion_rate": 0.05,
                "deposition_amount": 1.0,
                "persistence_time": 300.0,
                "max_intensity": 2.0,
                "wind_sensitivity": 0.5,
                "temperature_sensitivity": 0.3,
                "humidity_sensitivity": 0.2,
            },
            "food": {
                "evaporation_rate": 0.05,
                "diffusion_rate": 0.1,
                "deposition_amount": 2.0,
                "persistence_time": 600.0,
                "max_intensity": 1.5,
                "wind_sensitivity": 0.3,
                "temperature_sensitivity": 0.2,
                "humidity_sensitivity": 0.4,
            },
            "alarm": {
                "evaporation_rate": 0.2,
                "diffusion_rate": 0.2,
                "deposition_amount": 3.0,
                "persistence_time": 120.0,
                "max_intensity": 3.0,
                "wind_sensitivity": 0.7,
                "temperature_sensitivity": 0.5,
                "humidity_sensitivity": 0.1,
            },
            "nest": {
                "evaporation_rate": 0.02,
                "diffusion_rate": 0.02,
                "deposition_amount": 1.5,
                "persistence_time": 3600.0,
                "max_intensity": 2.5,
                "wind_sensitivity": 0.1,
                "temperature_sensitivity": 0.1,
                "humidity_sensitivity": 0.3,
            },
        }

        for name in pheromone_type_names:
            config = default_configs.get(name, default_configs["trail"]).copy()
            if self._evaporation_rate is not None:
                config["evaporation_rate"] = self._evaporation_rate
            types.append(PheromoneType(name=name, **config))

        return types

    def _initialize_spatial_integration(self, backend: str) -> None:
        """Initialize spatial integration components."""
        # Initialize spatial indexer
        if SpatialIndexingInterface:
            try:
                self.spatial_indexer = SpatialIndexingInterface(backend=backend)
                logger.info(f"Spatial indexer initialized with {backend} backend")
            except Exception as e:
                logger.warning(f"Failed to initialize spatial indexer: {e}")

        # Initialize spatial analytics
        if SpatialAnalyticsInterface:
            try:
                self.spatial_analytics = SpatialAnalyticsInterface(backend=backend)
                logger.info(f"Spatial analytics initialized with {backend} backend")
            except Exception as e:
                logger.warning(f"Failed to initialize spatial analytics: {e}")

    def _initialize_pheromone_fields(self) -> None:
        """Initialize pheromone fields for all types."""
        for pheromone_type in self.pheromone_types:
            f = PheromoneField(
                pheromone_type=pheromone_type.name,
                spatial_resolution=self.spatial_resolution,
                bounds=self.bounds,
            )
            # Provide references needed by PheromoneField methods
            f.spatial_indexer = self.spatial_indexer
            f.pheromone_types = self.pheromone_types
            self.pheromone_fields[pheromone_type.name] = f

    async def deposit_pheromone(
        self,
        agent_id: str,
        pheromone_type: str,
        location: np.ndarray,
        intensity: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Deposit pheromone at specified location.

        Args:
            agent_id: ID of the depositing agent
            pheromone_type: Type of pheromone to deposit
            location: Spatial location [lat, lng]
            intensity: Pheromone intensity (uses default if None)
            metadata: Additional metadata about the deposit

        Returns:
            True if deposit was successful
        """
        start_time = datetime.now()

        try:
            # Validate pheromone type
            if pheromone_type not in self.pheromone_fields:
                logger.error(f"Unknown pheromone type: {pheromone_type}")
                return False

            # Get pheromone type configuration
            phero_type = next(
                (pt for pt in self.pheromone_types if pt.name == pheromone_type), None
            )
            if not phero_type:
                return False

            # Use default intensity if not specified
            if intensity is None:
                intensity = phero_type.deposition_amount

            # Validate intensity bounds
            if not np.isfinite(intensity) or intensity <= 0:
                logger.warning(
                    f"Pheromone intensity {intensity} out of bounds for type {pheromone_type}"
                )
                return False
            if intensity > phero_type.max_intensity:
                logger.warning(
                    f"Pheromone intensity {intensity} capped for type {pheromone_type}"
                )
                intensity = phero_type.max_intensity

            # Create deposit record
            deposit = PheromoneDeposit(
                agent_id=agent_id,
                pheromone_type=pheromone_type,
                intensity=intensity,
                location=location.copy(),
                timestamp=datetime.now(),
                metadata=metadata or {},
            )

            # Update pheromone field
            field = self.pheromone_fields[pheromone_type]

            # Use spatial indexing if available
            if self.spatial_indexer:
                try:
                    cell_id = self.spatial_indexer.latlng_to_cell(
                        location[0], location[1], self.h3_resolution
                    )

                    # Add to field concentration
                    current_concentration = field.concentrations.get(cell_id, 0.0)
                    new_concentration = min(
                        current_concentration + intensity, phero_type.max_intensity
                    )
                    field.concentrations[cell_id] = new_concentration

                except Exception as e:
                    logger.warning(f"Spatial indexing failed for deposit: {e}")
                    logger.warning(
                        "Spatial indexing failed; using deposit records for queries"
                    )
            # Deposit records are the canonical audit trail and fallback query
            # source even when indexed concentrations are available.
            field.deposits.append(deposit)

            # Update field metadata
            field.last_update = datetime.now()
            field.update_count += 1

            # Update performance stats
            self.performance_stats["deposits_total"] += 1

            response_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats["avg_response_time"] = (
                self.performance_stats["avg_response_time"] + response_time
            ) / 2

            logger.debug(
                f"Pheromone {pheromone_type} deposited by {agent_id} at {location}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to deposit pheromone: {e}")
            return False

    async def sense_pheromones(
        self,
        location: np.ndarray,
        sensory_range: float,
        pheromone_types: list[str] | None = None,
        sensitivity_threshold: float = 0.01,
    ) -> dict[str, float]:
        """
        Sense pheromone concentrations around a location.

        Args:
            location: Center location for sensing [lat, lng]
            sensory_range: Maximum sensing distance (meters)
            pheromone_types: Types of pheromones to sense (all if None)
            sensitivity_threshold: Minimum detectable concentration

        Returns:
            Dictionary mapping pheromone types to concentrations
        """
        start_time = datetime.now()

        try:
            # Determine which pheromone types to sense
            if pheromone_types is None:
                types_to_sense = list(self.pheromone_fields.keys())
            else:
                types_to_sense = [
                    pt for pt in pheromone_types if pt in self.pheromone_fields
                ]

            if not types_to_sense:
                return {}

            sensed_pheromones = {}

            # Sense each requested pheromone type
            for phero_type in types_to_sense:
                field = self.pheromone_fields[phero_type]

                # Get concentration at location
                concentration = field.get_concentration(location)

                # Apply environmental modifications
                concentration = self._apply_environmental_modifications(
                    concentration, phero_type, location
                )

                # Check sensitivity threshold
                if concentration >= sensitivity_threshold:
                    sensed_pheromones[phero_type] = concentration
                else:
                    sensed_pheromones[phero_type] = 0.0

            # Update performance stats
            response_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats["queries_total"] += 1
            self.performance_stats["avg_response_time"] = (
                self.performance_stats["avg_response_time"] + response_time
            ) / 2

            return sensed_pheromones

        except Exception as e:
            logger.error(f"Failed to sense pheromones: {e}")
            return {}

    def _apply_environmental_modifications(
        self, concentration: float, pheromone_type: str, location: np.ndarray
    ) -> float:
        """Apply environmental factors to pheromone concentration."""
        modified_concentration = concentration

        # Get pheromone type configuration
        phero_type = next(
            (pt for pt in self.pheromone_types if pt.name == pheromone_type), None
        )
        if not phero_type:
            return modified_concentration

        # Apply wind effects (directional diffusion)
        wind_effect = self.environmental_factors.get("wind_speed", 0.0)
        if wind_effect > 0:
            wind_factor = 1.0 + (wind_effect / 10.0) * phero_type.wind_sensitivity
            modified_concentration *= wind_factor

        # Apply temperature effects (evaporation rate)
        temperature = self.environmental_factors.get("temperature", 20.0)
        temp_factor = (
            1.0 + ((temperature - 20.0) / 20.0) * phero_type.temperature_sensitivity
        )
        modified_concentration *= temp_factor

        # Apply humidity effects (persistence)
        humidity = self.environmental_factors.get("humidity", 50.0)
        humidity_factor = (
            1.0 + ((humidity - 50.0) / 50.0) * phero_type.humidity_sensitivity
        )
        modified_concentration *= humidity_factor

        return float(max(modified_concentration, 0.0))

    async def diffuse_pheromones(
        self,
        time_step: float,
        environmental_conditions: dict[str, Any] | None = None,
        spatial_barriers: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Update pheromone diffusion and evaporation across all fields.

        Args:
            time_step: Time step for diffusion calculation (seconds)
            environmental_conditions: Current environmental conditions
            spatial_barriers: Spatial barriers affecting diffusion

        Returns:
            Summary of diffusion updates for each pheromone type
        """
        start_time = datetime.now()

        if not np.isfinite(time_step) or time_step < 0:
            raise ValueError("time_step must be a finite non-negative number")
        try:
            # Update environmental factors
            if environmental_conditions:
                self.environmental_factors.update(environmental_conditions)

            diffusion_summary = {}

            # Update each pheromone field
            for phero_type in self.pheromone_types:
                field = self.pheromone_fields[phero_type.name]

                # Apply evaporation
                await self._apply_evaporation(field, phero_type, time_step)

                # Apply diffusion if rate > 0
                if phero_type.diffusion_rate > 0:
                    await self._apply_diffusion(
                        field, phero_type, time_step, spatial_barriers
                    )

                # Update field metadata
                field.last_update = datetime.now()
                field.update_count += 1

                # Record summary
                diffusion_summary[phero_type.name] = {
                    "cells_updated": len(field.concentrations),
                    "total_deposits": len(field.deposits),
                    "max_concentration": (
                        max(field.concentrations.values())
                        if field.concentrations
                        else 0.0
                    ),
                    "avg_concentration": (
                        np.mean(list(field.concentrations.values()))
                        if field.concentrations
                        else 0.0
                    ),
                }

            # Update performance stats
            self.performance_stats["updates_total"] += 1

            response_time = (datetime.now() - start_time).total_seconds()
            self.performance_stats["avg_response_time"] = (
                self.performance_stats["avg_response_time"] + response_time
            ) / 2

            logger.debug(
                f"Pheromone diffusion completed for {len(self.pheromone_types)} types"
            )
            return diffusion_summary

        except Exception as e:
            logger.error(f"Failed to diffuse pheromones: {e}")
            return {}

    async def _apply_evaporation(
        self, field: PheromoneField, phero_type: PheromoneType, time_step: float
    ) -> None:
        """Apply evaporation to pheromone field."""
        # Update concentrations using spatial indexing
        if self.spatial_indexer and field.concentrations:
            try:
                cells_to_remove = []

                for cell_id, concentration in field.concentrations.items():
                    # Calculate evaporation
                    evaporation = (
                        concentration * phero_type.evaporation_rate * (time_step / 60.0)
                    )  # per minute

                    # Apply environmental modifications
                    env_factor = self._calculate_environmental_evaporation_factor(
                        phero_type
                    )
                    evaporation *= env_factor

                    # Update concentration
                    new_concentration = max(concentration - evaporation, 0.0)

                    if new_concentration <= phero_type.min_intensity:
                        cells_to_remove.append(cell_id)
                    else:
                        field.concentrations[cell_id] = new_concentration

                # Keep the deposit audit trail in sync with indexed fields.
                self._evaporate_deposits(field, phero_type, time_step)

                # Remove depleted cells
                for cell_id in cells_to_remove:
                    del field.concentrations[cell_id]

            except Exception as e:
                logger.warning(f"Spatial evaporation failed: {e}")
                # Apply the deposit-based evaporation path.
                self._evaporate_deposits(field, phero_type, time_step)
        else:
            # Apply the deposit-based evaporation path.
            self._evaporate_deposits(field, phero_type, time_step)

    def _evaporate_deposits(
        self, field: PheromoneField, phero_type: PheromoneType, time_step: float
    ) -> None:
        """Evaporate pheromone deposits."""
        if not field.deposits:
            return

        env_factor = self._calculate_environmental_evaporation_factor(phero_type)
        decay = math.exp(-phero_type.evaporation_rate * env_factor * (time_step / 60.0))

        # Remove old deposits beyond persistence time
        current_time = datetime.now()
        max_age = timedelta(seconds=phero_type.persistence_time)

        retained = []
        for deposit in field.deposits:
            deposit.intensity *= decay
            if (
                current_time - deposit.timestamp
            ) <= max_age and deposit.intensity >= phero_type.min_intensity:
                retained.append(deposit)
        field.deposits = retained

    def _calculate_environmental_evaporation_factor(
        self, phero_type: PheromoneType
    ) -> float:
        """Calculate environmental factor for evaporation."""
        factor = 1.0

        # Temperature effect
        temperature = self.environmental_factors.get("temperature", 20.0)
        if temperature > 25:
            factor *= (
                1.0 + ((temperature - 25) / 25.0) * phero_type.temperature_sensitivity
            )

        # Humidity effect
        humidity = self.environmental_factors.get("humidity", 50.0)
        if humidity < 40:
            factor *= 1.0 + ((40 - humidity) / 40.0) * phero_type.humidity_sensitivity

        return factor

    async def _apply_diffusion(
        self,
        field: PheromoneField,
        phero_type: PheromoneType,
        time_step: float,
        barriers: dict[str, Any] | None = None,
    ) -> None:
        """Apply spatial diffusion to pheromone field."""
        if not self.spatial_indexer or not field.concentrations:
            return

        try:
            # Get all cells with pheromone
            active_cells = list(field.concentrations.keys())
            if not active_cells:
                return

            # Transfer mass out of each source and distribute it to neighbors.
            next_concentrations = dict(field.concentrations)
            blocked_cells = set((barriers or {}).get("blocked_cells", []))

            for cell_id in active_cells:
                if cell_id in blocked_cells:
                    continue
                concentration = field.concentrations[cell_id]

                # Get neighboring cells
                neighbors = self.spatial_indexer.get_cell_neighbors(cell_id)

                if not neighbors:
                    continue

                neighbors = [
                    neighbor for neighbor in neighbors if neighbor not in blocked_cells
                ]
                if not neighbors:
                    continue
                diffusion_amount = min(
                    concentration,
                    concentration * phero_type.diffusion_rate * (time_step / 60.0),
                )
                next_concentrations[cell_id] = max(
                    0.0, next_concentrations[cell_id] - diffusion_amount
                )

                for neighbor_id in neighbors:
                    current = next_concentrations.get(neighbor_id, 0.0)
                    next_concentrations[neighbor_id] = min(
                        current + diffusion_amount / len(neighbors),
                        phero_type.max_intensity,
                    )

            field.concentrations = {
                cell_id: concentration
                for cell_id, concentration in next_concentrations.items()
                if concentration >= phero_type.min_intensity
            }

        except Exception as e:
            logger.warning(f"Spatial diffusion failed: {e}")

    def get_pheromone_intensity(
        self, location: np.ndarray, pheromone_type: str
    ) -> float:
        """
        Get pheromone intensity at specific location.

        Args:
            location: Location [lat, lng]
            pheromone_type: Type of pheromone to query

        Returns:
            Pheromone intensity at location
        """
        if pheromone_type not in self.pheromone_fields:
            return 0.0

        field = self.pheromone_fields[pheromone_type]
        return field.get_concentration(location)

    def get_pheromone_gradient(
        self, location: np.ndarray, pheromone_type: str, radius: float = 100.0
    ) -> tuple[float, np.ndarray]:
        """
        Get pheromone gradient (intensity and direction) at location.

        Args:
            location: Center location [lat, lng]
            pheromone_type: Type of pheromone to analyze
            radius: Radius for gradient calculation (meters)

        Returns:
            Tuple of (gradient_magnitude, gradient_direction_vector)
        """
        if pheromone_type not in self.pheromone_fields:
            return 0.0, np.array([0.0, 0.0])

        try:
            # Sample multiple points around location
            n_samples = 8
            angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

            gradients = []
            for angle in angles:
                # Sample point at radius distance
                sample_location = location + radius * np.array(
                    [np.cos(angle), np.sin(angle)]
                )

                # Get intensity at sample point
                intensity = self.get_pheromone_intensity(
                    sample_location, pheromone_type
                )
                gradients.append(intensity)

            # Calculate gradient magnitude
            gradient_array = np.array(gradients)
            gradient_magnitude = np.max(gradient_array) - np.min(gradient_array)

            # Calculate gradient direction (toward highest concentration)
            max_idx = np.argmax(gradient_array)
            max_angle = angles[max_idx]
            gradient_direction = np.array([np.cos(max_angle), np.sin(max_angle)])

            return gradient_magnitude, gradient_direction

        except Exception as e:
            logger.warning(f"Failed to calculate pheromone gradient: {e}")
            return 0.0, np.array([0.0, 0.0])

    def find_strongest_trail(
        self,
        start_location: np.ndarray,
        pheromone_type: str = "trail",
        search_radius: float = 1000.0,
    ) -> dict[str, Any] | None:
        """
        Find the strongest pheromone trail within search radius.

        Args:
            start_location: Starting location for search
            pheromone_type: Type of pheromone trail to find
            search_radius: Maximum search radius (meters)

        Returns:
            Trail information or None if no trail found
        """
        if pheromone_type not in self.pheromone_fields:
            return None

        try:
            field = self.pheromone_fields[pheromone_type]

            # Use spatial analytics if available for efficient search
            if self.spatial_analytics:
                try:
                    # Find high-concentration areas
                    hotspots = self.spatial_analytics.find_hotspots(
                        concentration_field=field.concentrations,
                        min_intensity=0.1,
                        max_results=5,
                    )

                    if hotspots:
                        # Return closest hotspot
                        closest_hotspot = min(
                            hotspots,
                            key=lambda h: np.linalg.norm(
                                np.array(h["center"]) - start_location
                            ),
                        )
                        return cast(dict[str, Any], closest_hotspot)

                except Exception as e:
                    logger.warning(f"Spatial analytics search failed: {e}")

            # Search recorded deposits when no indexed cell is available.
            if field.deposits:
                # Find deposits within range
                nearby_deposits = []
                for deposit in field.deposits:
                    distance = np.linalg.norm(start_location - deposit.location)
                    if distance <= search_radius:
                        nearby_deposits.append((deposit, distance))

                if nearby_deposits:
                    # Return strongest nearby deposit
                    strongest_deposit, distance = max(
                        nearby_deposits, key=lambda x: x[0].intensity
                    )
                    return {
                        "location": strongest_deposit.location,
                        "intensity": strongest_deposit.intensity,
                        "distance": distance,
                        "timestamp": strongest_deposit.timestamp,
                        "agent_id": strongest_deposit.agent_id,
                    }

            return None

        except Exception as e:
            logger.error(f"Failed to find pheromone trail: {e}")
            return None

    def get_field_statistics(self, pheromone_type: str) -> dict[str, Any]:
        """
        Get statistical summary of pheromone field.

        Args:
            pheromone_type: Type of pheromone field to analyze

        Returns:
            Statistical summary of the field
        """
        if pheromone_type not in self.pheromone_fields:
            return {}

        field = self.pheromone_fields[pheromone_type]

        stats = {
            "pheromone_type": pheromone_type,
            "total_deposits": len(field.deposits),
            "active_cells": len(field.concentrations) or len(field.deposits),
            "last_update": field.last_update.isoformat(),
            "update_count": field.update_count,
        }

        if field.concentrations:
            concentrations = list(field.concentrations.values())
            stats.update(
                {
                    "max_concentration": np.max(concentrations),
                    "min_concentration": np.min(concentrations),
                    "avg_concentration": np.mean(concentrations),
                    "std_concentration": np.std(concentrations),
                }
            )
        elif field.deposits:
            # No spatial indexing available — derive concentration stats from raw deposits
            intensities = [d.intensity for d in field.deposits]
            stats.update(
                {
                    "max_concentration": np.max(intensities),
                    "min_concentration": np.min(intensities),
                    "avg_concentration": np.mean(intensities),
                    "std_concentration": np.std(intensities),
                }
            )

        if field.deposits:
            intensities = [d.intensity for d in field.deposits]
            stats.update(
                {
                    "max_deposit": np.max(intensities),
                    "min_deposit": np.min(intensities),
                    "avg_deposit": np.mean(intensities),
                }
            )

        return stats

    def clear_pheromone_field(self, pheromone_type: str) -> bool:
        """
        Clear all pheromones of specified type.

        Args:
            pheromone_type: Type of pheromone field to clear

        Returns:
            True if clearing was successful
        """
        if pheromone_type not in self.pheromone_fields:
            return False

        field = self.pheromone_fields[pheromone_type]
        field.concentrations.clear()
        field.deposits.clear()
        field.last_update = datetime.now()
        field.update_count = 0

        logger.info(f"Cleared pheromone field: {pheromone_type}")
        return True

    def get_performance_statistics(self) -> dict[str, Any]:
        """Get performance statistics for the pheromone system."""
        return self.performance_stats.copy()

    def save_pheromone_fields(self, filepath: str) -> bool:
        """Save pheromone fields to file."""
        try:
            import json

            data: dict[str, Any] = {
                "pheromone_types": [pt.name for pt in self.pheromone_types],
                "spatial_resolution": self.spatial_resolution,
                "bounds": self.bounds,
                "environmental_factors": self.environmental_factors,
                "fields": {},
            }

            for phero_type, field in self.pheromone_fields.items():
                data["fields"][phero_type] = {
                    "concentrations": field.concentrations,
                    "deposits": [
                        {
                            "agent_id": d.agent_id,
                            "intensity": d.intensity,
                            "location": d.location.tolist(),
                            "timestamp": d.timestamp.isoformat(),
                            "metadata": d.metadata,
                        }
                        for d in field.deposits
                    ],
                    "last_update": field.last_update.isoformat(),
                    "update_count": field.update_count,
                }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Pheromone fields saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save pheromone fields: {e}")
            return False

    def load_pheromone_fields(self, filepath: str) -> bool:
        """Load pheromone fields from file."""
        try:
            import json

            with open(filepath) as f:
                data = json.load(f)

            # Restore pheromone fields
            for phero_type, field_data in data["fields"].items():
                if phero_type in self.pheromone_fields:
                    field = self.pheromone_fields[phero_type]

                    # Restore concentrations
                    field.concentrations = field_data["concentrations"]

                    # Restore deposits
                    field.deposits = [
                        PheromoneDeposit(
                            agent_id=d["agent_id"],
                            pheromone_type=phero_type,
                            intensity=d["intensity"],
                            location=np.array(d["location"]),
                            timestamp=datetime.fromisoformat(d["timestamp"]),
                            metadata=d["metadata"],
                        )
                        for d in field_data["deposits"]
                    ]

                    # Restore metadata
                    field.last_update = datetime.fromisoformat(
                        field_data["last_update"]
                    )
                    field.update_count = field_data["update_count"]

            logger.info(f"Pheromone fields loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load pheromone fields: {e}")
            return False
