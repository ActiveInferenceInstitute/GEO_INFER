"""
Advanced Geospatial Active Inference Methods

This module is part of GEO-INFER-AI and implements sophisticated geospatial active inference algorithms for
environmental modeling, resource optimization, predictive spatial dynamics, and
multi-scale hierarchical analysis. Integrates Active Inference principles with
cutting-edge geospatial analysis techniques.

Mathematical Foundation:
The geospatial active inference framework extends the free energy principle to
spatial domains by modeling:
1. Spatial generative models with environmental dynamics
2. Multi-scale hierarchical belief propagation
3. Spatial-temporal predictive coding
4. Resource allocation under environmental uncertainty

Key Components:
- Environmental Active Inference Engine
- Multi-scale Hierarchical Spatial Models
- Resource Optimization under Uncertainty
- Predictive Environmental Dynamics
- Spatial Attention Mechanisms
- H3-based Hexagonal Grid Analysis
"""

import numpy as np
import h3
from typing import Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from contextlib import contextmanager
import warnings

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_estimator_warnings() -> Any:
    """Silence scikit-learn convergence chatter for one estimator call.

    Suppression is scoped to the wrapped call; importing this module never
    mutates process-wide warning filters.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class H3SpatialGraph:
    """H3 neighbor graph keyed by cell and grid distance."""

    def __init__(self, cells: list[str], max_distance: int = 3):
        """Build distance-indexed H3 neighbor sets for known cells."""
        self.cells = cells
        self.neighbors: dict[str, dict[int, set]] = {}
        cell_set = set(cells)
        for cell in cells:
            self.neighbors[cell] = {}
            for distance in range(1, max_distance + 1):
                try:
                    ring_neighbors = h3.grid_ring(cell, distance)
                    self.neighbors[cell][distance] = set(ring_neighbors) & cell_set
                except Exception:
                    self.neighbors[cell][distance] = set()


class LevelSpatialGraph:
    """First-order H3 neighbor graph for one hierarchy level."""

    def __init__(self, cells: list[str]):
        """Build direct H3 neighbor sets for known cells."""
        self.cells = cells
        self.neighbors: dict[str, set] = {}
        cell_set = set(cells)
        for cell in cells:
            try:
                self.neighbors[cell] = set(h3.grid_ring(cell, 1)) & cell_set
            except Exception:
                self.neighbors[cell] = set()


@dataclass
class EnvironmentalState:
    """Represents environmental state at a spatial location."""

    location: str  # H3 cell ID
    temperature: float = 20.0
    humidity: float = 0.5
    vegetation_density: float = 0.5
    water_availability: float = 0.5
    soil_quality: float = 0.5
    biodiversity_index: float = 0.5
    human_activity: float = 0.0
    carbon_flux: float = 0.0
    timestamp: float | None = None
    uncertainty: dict[str, float] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Represents resource allocation decision."""

    location: str
    resource_type: str
    allocation_amount: float
    priority_score: float
    expected_benefit: float
    uncertainty: float
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialPrediction:
    """Spatial prediction with uncertainty quantification."""

    location: str
    predicted_value: float
    uncertainty: float
    confidence_interval: tuple[float, float]
    prediction_horizon: float
    contributing_factors: dict[str, float] = field(default_factory=dict)


class EnvironmentalActiveInferenceEngine:
    """
    Advanced Environmental Active Inference Engine for geospatial modeling.

    This engine implements sophisticated environmental modeling using Active Inference
    principles, integrating spatial-temporal dynamics, uncertainty quantification,
    and resource optimization for environmental management.
    """

    def __init__(
        self,
        h3_resolution: int = 8,
        environmental_variables: list[str] | None = None,
        prediction_horizon: int = 10,
        uncertainty_threshold: float = 0.1,
    ) -> None:
        """
        Initialize Environmental Active Inference Engine.

        Args:
            h3_resolution: H3 hexagonal grid resolution for spatial modeling
            environmental_variables: List of environmental variables to model
            prediction_horizon: Number of timesteps for predictions
            uncertainty_threshold: Threshold for high uncertainty detection
        """
        self.h3_resolution = h3_resolution
        self.environmental_variables = environmental_variables or [
            "temperature",
            "humidity",
            "vegetation_density",
            "water_availability",
            "soil_quality",
            "biodiversity_index",
            "carbon_flux",
        ]
        self.prediction_horizon = prediction_horizon
        self.uncertainty_threshold = uncertainty_threshold

        # Initialize internal models
        self.spatial_graph: H3SpatialGraph | None = None
        self.environmental_states: dict[str, EnvironmentalState] = {}
        self.observation_history: list[Any] = []
        self.prediction_models: dict[str, Any] = {}
        self.resource_optimizer = None

        # Gaussian Process models for each environmental variable
        self._initialize_gp_models()

        logger.info(
            f"Initialized EnvironmentalActiveInferenceEngine with {len(self.environmental_variables)} variables"
        )

    def _initialize_gp_models(self) -> None:
        """Initialize Gaussian Process models for environmental prediction."""
        self.gp_models = {}

        for var in self.environmental_variables:
            # Use Matern kernel for environmental modeling (captures spatial correlations)
            kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.01)
            self.gp_models[var] = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5
            )

    def initialize_spatial_domain(self, boundary: dict[str, Any]) -> None:
        """
        Initialize spatial domain using H3 hexagonal grid.

        Args:
            boundary: GeoJSON boundary specification
        """
        try:
            # Generate H3 cells within boundary
            h3_cells = self._generate_h3_cells_from_boundary(boundary)

            # Initialize environmental states
            for cell in h3_cells:
                self.environmental_states[cell] = EnvironmentalState(
                    location=cell, timestamp=0.0
                )

            # Create spatial graph for neighbor relationships
            self.spatial_graph = self._create_h3_spatial_graph(
                h3_cells, max_neighbor_distance=3
            )

            logger.info(f"Initialized spatial domain with {len(h3_cells)} H3 cells")

        except Exception as e:
            logger.error(f"Failed to initialize spatial domain: {e}")
            raise

    def _generate_h3_cells_from_boundary(self, boundary: dict[str, Any]) -> list[str]:
        """Generate H3 cells from boundary specification.

        Handles GeoJSON Polygon, MultiPolygon, and bare coordinate lists.
        """
        cells = set()

        if "coordinates" in boundary:
            raw = boundary["coordinates"]

            def _is_coord_pair(item: Any) -> bool:
                return (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[0], (int, float))
                    and isinstance(item[1], (int, float))
                )

            def _extract_rings(data: Any) -> list[Any]:
                if not isinstance(data, (list, tuple)) or len(data) == 0:
                    return []
                if _is_coord_pair(data[0]):
                    return [data]
                rings = []
                for sub in data:
                    rings.extend(_extract_rings(sub))
                return rings

            coordinate_rings = _extract_rings(raw)

            for coordinate_ring in coordinate_rings:
                for coord in coordinate_ring:
                    try:
                        if not _is_coord_pair(coord):
                            continue
                        lng, lat = float(coord[0]), float(coord[1])
                        cell = h3.latlng_to_cell(lat, lng, self.h3_resolution)
                        cells.add(cell)

                        # Add nearby cells for coverage
                        for ring_distance in range(1, 3):
                            try:
                                neighbors = h3.grid_ring(cell, ring_distance)
                                if isinstance(neighbors, list):
                                    cells.update(neighbors)
                                else:
                                    cells.update(list(neighbors))
                            except Exception:
                                continue

                    except (ValueError, TypeError):
                        continue

        return list(cells)

    def _create_h3_spatial_graph(
        self, h3_cells: list[str], max_neighbor_distance: int = 3
    ) -> H3SpatialGraph:
        """Create a simple H3 spatial graph for neighbor relationships."""
        return H3SpatialGraph(h3_cells, max_neighbor_distance)

    def observe_environment(
        self, observations: dict[str, dict[str, float]], timestamp: float
    ) -> None:
        """
        Update environmental state beliefs based on new observations.

        Args:
            observations: Dictionary mapping H3 cells to environmental observations
            timestamp: Observation timestamp
        """
        observation_data: dict[str, Any] = {
            "timestamp": timestamp,
            "observations": observations,
            "updated_cells": [],
        }

        for cell, obs_data in observations.items():
            if cell in self.environmental_states:
                # Update environmental state with new observations
                env_state = self.environmental_states[cell]

                for var, value in obs_data.items():
                    if var in self.environmental_variables:
                        setattr(env_state, var, value)

                env_state.timestamp = timestamp
                observation_data["updated_cells"].append(cell)

        self.observation_history.append(observation_data)

        # Update prediction models with new data
        self._update_prediction_models()

        logger.debug(
            f"Updated {len(observation_data['updated_cells'])} cells with new observations"
        )

    def _update_prediction_models(self) -> None:
        """Update Gaussian Process models with latest observations."""
        if len(self.observation_history) < 2:
            return

        # Prepare training data for each environmental variable
        for var in self.environmental_variables:
            X_spatial: Any = []  # [lat, lng, time]
            y_values: Any = []  # environmental variable values

            for obs_record in self.observation_history:
                timestamp = obs_record["timestamp"]

                for cell, obs_data in obs_record["observations"].items():
                    if var in obs_data and cell in self.environmental_states:
                        lat, lng = h3.cell_to_latlng(cell)
                        X_spatial.append([lat, lng, timestamp])
                        y_values.append(obs_data[var])

            if len(X_spatial) > 5:  # Minimum samples for GP training
                X_spatial = np.array(X_spatial)
                y_values = np.array(y_values)

                try:
                    with _suppress_estimator_warnings():
                        self.gp_models[var].fit(X_spatial, y_values)
                    logger.debug(
                        f"Updated GP model for {var} with {len(X_spatial)} samples"
                    )
                except Exception as e:
                    logger.warning(f"Failed to update GP model for {var}: {e}")

    def predict_environmental_dynamics(
        self, forecast_timesteps: int | None = None
    ) -> dict[str, list[SpatialPrediction]]:
        """
        Predict future environmental states using learned dynamics.

        Args:
            forecast_timesteps: Number of timesteps to forecast

        Returns:
            Dictionary mapping variables to spatial predictions
        """
        if forecast_timesteps is None:
            forecast_timesteps = self.prediction_horizon

        predictions = defaultdict(list)
        current_time = (
            max([obs["timestamp"] for obs in self.observation_history])
            if self.observation_history
            else 0.0
        )

        for timestep in range(1, forecast_timesteps + 1):
            future_time = current_time + timestep

            for cell in self.environmental_states:
                lat, lng = h3.cell_to_latlng(cell)

                for var in self.environmental_variables:
                    if var in self.gp_models:
                        try:
                            # Predict at future timestep
                            X_pred = np.array([[lat, lng, future_time]])
                            with _suppress_estimator_warnings():
                                mean_pred, std_pred = self.gp_models[var].predict(
                                    X_pred, return_std=True
                                )
                            mean_pred = np.atleast_1d(mean_pred)
                            std_pred = np.atleast_1d(std_pred)

                            # Create prediction object
                            prediction = SpatialPrediction(
                                location=cell,
                                predicted_value=float(mean_pred[0]),
                                uncertainty=float(std_pred[0]),
                                confidence_interval=(
                                    float(mean_pred[0] - 1.96 * std_pred[0]),
                                    float(mean_pred[0] + 1.96 * std_pred[0]),
                                ),
                                prediction_horizon=timestep,
                                contributing_factors=self._analyze_prediction_factors(
                                    cell, var
                                ),
                            )

                            predictions[var].append(prediction)

                        except Exception as e:
                            logger.warning(
                                f"Prediction failed for {var} at {cell}: {e}"
                            )

        logger.info(
            f"Generated {sum(len(v) for v in predictions.values())} spatial predictions"
        )
        return dict(predictions)

    def _analyze_prediction_factors(self, cell: str, variable: str) -> dict[str, float]:
        """Analyze factors contributing to predictions."""
        factors = {}

        # Spatial factors (neighboring cells influence)
        if self.spatial_graph:
            neighbors = self.spatial_graph.neighbors.get(cell, {}).get(1, set())
            neighbor_values = []

            for neighbor in neighbors:
                if neighbor in self.environmental_states:
                    env_state = self.environmental_states[neighbor]
                    if hasattr(env_state, variable):
                        neighbor_values.append(getattr(env_state, variable))

            if neighbor_values:
                try:
                    data = [
                        getattr(self.environmental_states[cell], variable, 0.5)
                    ] + neighbor_values
                    corr_mat = np.corrcoef(data)
                    # Handle scalar or 0-d output if it happens, though unlikely for list input
                    if np.ndim(corr_mat) == 2:
                        factors["spatial_correlation"] = corr_mat[0, 1:].mean()
                    else:
                        factors["spatial_correlation"] = 0.0
                except (
                    ValueError,
                    TypeError,
                    FloatingPointError,
                    RuntimeWarning,
                ) as exc:
                    logger.debug(
                        "spatial_correlation fell back to 0.0: %s", exc, exc_info=True
                    )
                    factors["spatial_correlation"] = 0.0
            else:
                factors["spatial_correlation"] = 0.0

        # Temporal autocorrelation
        historical_values = []
        for obs_record in self.observation_history[-10:]:  # Last 10 observations
            if (
                cell in obs_record["observations"]
                and variable in obs_record["observations"][cell]
            ):
                historical_values.append(obs_record["observations"][cell][variable])

        if len(historical_values) > 1:
            try:
                corr_mat = np.corrcoef(historical_values[:-1], historical_values[1:])
                if np.ndim(corr_mat) == 2:
                    autocorr = corr_mat[0, 1]
                    factors["temporal_autocorrelation"] = (
                        autocorr if not np.isnan(autocorr) else 0.0
                    )
                else:
                    factors["temporal_autocorrelation"] = 0.0
            except (ValueError, TypeError, FloatingPointError, RuntimeWarning) as exc:
                logger.debug(
                    "temporal_autocorrelation fell back to 0.0: %s",
                    exc,
                    exc_info=True,
                )
                factors["temporal_autocorrelation"] = 0.0
        else:
            factors["temporal_autocorrelation"] = 0.0

        # Environmental interactions
        current_state = self.environmental_states[cell]
        factors["temperature_vegetation_interaction"] = (
            current_state.temperature * current_state.vegetation_density
        )
        factors["water_soil_interaction"] = (
            current_state.water_availability * current_state.soil_quality
        )

        return factors

    def optimize_resource_allocation(
        self,
        resource_budget: float,
        resource_types: list[str],
        optimization_objective: str = "biodiversity",
    ) -> list[ResourceAllocation]:
        """
        Optimize resource allocation using active inference principles.

        This method formulates resource allocation as an active inference problem
        where the agent (resource manager) must select actions (resource allocations)
        that minimize expected free energy, balancing:
        1. Pragmatic value: achieving environmental objectives
        2. Epistemic value: reducing uncertainty about environmental dynamics

        Mathematical Framework:
        The expected free energy for resource allocation is:
        G(π) = E_q[log q(s|π)] - E_q[log p(o,s|π)]
             = Risk + Ambiguity

        Where Risk is the cost of not achieving objectives and Ambiguity
        is the uncertainty about environmental outcomes.

        Args:
            resource_budget: Total budget for resource allocation
            resource_types: Types of resources to allocate
            optimization_objective: Primary objective ('biodiversity', 'carbon', 'stability')

        Returns:
            List of optimal resource allocations
        """
        logger.info(f"Optimizing resource allocation with budget {resource_budget}")

        allocations = []
        available_budget = resource_budget

        # Get current environmental predictions
        predictions = self.predict_environmental_dynamics(forecast_timesteps=5)

        # Score each location for resource allocation priority
        location_scores = self._compute_allocation_priorities(
            predictions, optimization_objective
        )

        # Sort locations by priority
        sorted_locations = sorted(
            location_scores.items(), key=lambda x: x[1]["priority"], reverse=True
        )

        for cell, score_data in sorted_locations:
            if available_budget <= 0:
                break

            # Determine optimal allocation for this location
            optimal_allocation = self._optimize_location_allocation(
                cell,
                score_data,
                resource_types,
                available_budget,
                optimization_objective,
            )

            if (
                optimal_allocation
                and optimal_allocation.allocation_amount <= available_budget
            ):
                allocations.append(optimal_allocation)
                available_budget -= optimal_allocation.allocation_amount

        logger.info(
            f"Generated {len(allocations)} resource allocations, used {resource_budget - available_budget:.2f}/{resource_budget:.2f} budget"
        )
        return allocations

    def _compute_allocation_priorities(
        self, predictions: dict[str, list[SpatialPrediction]], objective: str
    ) -> dict[str, dict[str, float]]:
        """Compute priority scores for resource allocation."""
        priorities = {}

        for cell in self.environmental_states:
            # Initialize priority metrics
            priority_data = {
                "priority": 0.0,
                "urgency": 0.0,
                "impact_potential": 0.0,
                "uncertainty": 0.0,
                "feasibility": 1.0,
            }

            # Analyze current state
            current_state = self.environmental_states[cell]

            # Compute urgency based on environmental degradation
            if objective == "biodiversity":
                urgency = max(
                    0, 0.5 - current_state.biodiversity_index
                )  # Higher urgency for low biodiversity
                impact_potential = (
                    1.0 - current_state.biodiversity_index
                )  # Higher potential if currently low
            elif objective == "carbon":
                urgency = max(0, -current_state.carbon_flux)  # Urgency for carbon loss
                impact_potential = abs(
                    current_state.carbon_flux
                )  # Potential for carbon improvement
            elif objective == "stability":
                # Stability based on prediction uncertainty
                uncertainties = []
                for var_predictions in predictions.values():
                    for pred in var_predictions:
                        if pred.location == cell:
                            uncertainties.append(pred.uncertainty)
                avg_uncertainty = np.mean(uncertainties) if uncertainties else 0.5
                urgency = float(avg_uncertainty)  # High uncertainty = high urgency
                impact_potential = float(
                    avg_uncertainty  # High uncertainty = high impact potential
                )
            else:
                urgency = 0.5
                impact_potential = 0.5

            # Compute overall uncertainty
            prediction_uncertainties = []
            for var_predictions in predictions.values():
                for pred in var_predictions:
                    if pred.location == cell and pred.prediction_horizon <= 3:
                        prediction_uncertainties.append(pred.uncertainty)

            avg_uncertainty = (
                np.mean(prediction_uncertainties) if prediction_uncertainties else 0.5
            )

            # Compute feasibility based on environmental conditions
            feasibility = (
                current_state.water_availability + current_state.soil_quality
            ) / 2.0

            # Combined priority score using active inference principles
            # Higher priority for high urgency, high impact potential, and high uncertainty (epistemic value)
            priority_score = (
                0.4 * urgency  # Pragmatic value (current need)
                + 0.3 * impact_potential  # Pragmatic value (potential benefit)
                + 0.2 * avg_uncertainty  # Epistemic value (learning opportunity)
                + 0.1 * feasibility  # Feasibility constraint
            )

            priority_data.update(
                {
                    "priority": float(priority_score),
                    "urgency": float(urgency),
                    "impact_potential": float(impact_potential),
                    "uncertainty": float(avg_uncertainty),
                    "feasibility": float(feasibility),
                }
            )

            priorities[cell] = priority_data

        return priorities

    def _optimize_location_allocation(
        self,
        cell: str,
        score_data: dict[str, float],
        resource_types: list[str],
        available_budget: float,
        objective: str,
    ) -> ResourceAllocation | None:
        """Optimize resource allocation for a specific location."""

        # Simple allocation strategy based on priority score
        if score_data["priority"] < 0.3:  # Low priority threshold
            return None

        # Allocate based on urgency and available budget
        base_allocation = min(
            available_budget * 0.1,  # Maximum 10% of budget per location
            score_data["priority"] * available_budget * 0.05,  # Scale by priority
        )

        # Select most appropriate resource type
        current_state = self.environmental_states[cell]

        if objective == "biodiversity":
            if current_state.vegetation_density < 0.3:
                resource_type = "vegetation_restoration"
            elif current_state.water_availability < 0.3:
                resource_type = "water_conservation"
            else:
                resource_type = "habitat_protection"
        elif objective == "carbon":
            resource_type = "carbon_sequestration"
        else:
            resource_type = "environmental_monitoring"

        # Select from available resource types
        selected_resource = resource_types[0] if resource_types else resource_type

        # Compute expected benefit using simple heuristic
        expected_benefit = score_data["priority"] * score_data["feasibility"]

        return ResourceAllocation(
            location=cell,
            resource_type=selected_resource,
            allocation_amount=base_allocation,
            priority_score=score_data["priority"],
            expected_benefit=expected_benefit,
            uncertainty=score_data["uncertainty"],
            constraints={
                "urgency": score_data["urgency"],
                "feasibility": score_data["feasibility"],
            },
        )

    def analyze_environmental_uncertainty(self) -> dict[str, Any]:
        """
        Analyze environmental uncertainty across the spatial domain.

        Returns:
            Comprehensive uncertainty analysis
        """
        uncertainty_analysis: dict[str, Any] = {
            "global_uncertainty": {},
            "spatial_uncertainty_patterns": {},
            "temporal_uncertainty_trends": {},
            "high_uncertainty_regions": [],
            "uncertainty_sources": {},
        }

        # Global uncertainty metrics
        all_uncertainties = defaultdict(list)

        for cell, env_state in self.environmental_states.items():
            for var in self.environmental_variables:
                if var in env_state.uncertainty:
                    all_uncertainties[var].append(env_state.uncertainty[var])

        for var, uncertainties in all_uncertainties.items():
            if uncertainties:
                uncertainty_analysis["global_uncertainty"][var] = {
                    "mean": np.mean(uncertainties),
                    "std": np.std(uncertainties),
                    "max": np.max(uncertainties),
                    "min": np.min(uncertainties),
                }

        # Identify high uncertainty regions
        high_uncertainty_cells = []
        for cell, env_state in self.environmental_states.items():
            total_uncertainty = (
                sum(env_state.uncertainty.values()) / len(env_state.uncertainty)
                if env_state.uncertainty
                else 0.5
            )
            if total_uncertainty > self.uncertainty_threshold:
                high_uncertainty_cells.append(
                    {
                        "cell": cell,
                        "uncertainty": total_uncertainty,
                        "coordinates": h3.cell_to_latlng(cell),
                    }
                )

        uncertainty_analysis["high_uncertainty_regions"] = high_uncertainty_cells

        # Spatial clustering of uncertainty
        if high_uncertainty_cells:
            coords = np.array(
                [h3.cell_to_latlng(item["cell"]) for item in high_uncertainty_cells]
            )
            try:
                with _suppress_estimator_warnings():
                    clustering = DBSCAN(eps=0.01, min_samples=2).fit(coords)
                n_clusters = len(set(clustering.labels_)) - (
                    1 if -1 in clustering.labels_ else 0
                )
                uncertainty_analysis["spatial_uncertainty_patterns"]["n_clusters"] = (
                    n_clusters
                )
                uncertainty_analysis["spatial_uncertainty_patterns"][
                    "clustered_fraction"
                ] = np.sum(clustering.labels_ != -1) / len(clustering.labels_)
            except (ValueError, TypeError) as exc:
                logger.debug(
                    "Uncertainty clustering fell back to error report: %s",
                    exc,
                    exc_info=True,
                )
                uncertainty_analysis["spatial_uncertainty_patterns"] = {
                    "error": "Clustering failed"
                }

        return uncertainty_analysis

    def compute_environmental_free_energy(self) -> dict[str, float]:
        """
        Compute environmental free energy across the spatial domain.

        The environmental free energy represents the total "surprise" or
        mismatch between expected and observed environmental states.

        Mathematical Foundation:
        F = E_q[log q(s)] - E_q[log p(o,s)]
          = KL[q(s)||p(s)] - E_q[log p(o|s)]
          = Complexity - Accuracy

        Returns:
            Dictionary containing free energy metrics
        """
        free_energy_metrics: dict[str, Any] = {
            "total_free_energy": 0.0,
            "spatial_free_energy": {},
            "variable_free_energy": {},
            "complexity_term": 0.0,
            "accuracy_term": 0.0,
        }

        total_fe = 0.0
        complexity_sum = 0.0
        accuracy_sum = 0.0

        for cell, env_state in self.environmental_states.items():
            cell_fe = 0.0

            # For each environmental variable, compute its contribution to free energy
            for var in self.environmental_variables:
                if hasattr(env_state, var):
                    observed_value = getattr(env_state, var)

                    # Prior belief (uniform distribution in [0, 1] range)
                    prior_mean = 0.5
                    prior_variance = 1.0 / 12.0  # Uniform distribution variance

                    # Prediction from GP model if available
                    if var in self.gp_models and len(self.observation_history) > 5:
                        try:
                            lat, lng = h3.cell_to_latlng(cell)
                            current_time = env_state.timestamp or 0.0
                            X_pred = np.array([[lat, lng, current_time]])
                            with _suppress_estimator_warnings():
                                predicted_mean, predicted_std = self.gp_models[
                                    var
                                ].predict(X_pred, return_std=True)
                            predicted_mean = np.atleast_1d(predicted_mean)
                            predicted_std = np.atleast_1d(predicted_std)

                            # Complexity: KL divergence between posterior and prior
                            complexity = 0.5 * (
                                np.log(prior_variance / (predicted_std[0] ** 2 + 1e-8))
                                + (
                                    predicted_std[0] ** 2
                                    + (predicted_mean[0] - prior_mean) ** 2
                                )
                                / prior_variance
                                - 1
                            )

                            # Accuracy: log likelihood of observation under posterior
                            accuracy = -0.5 * (
                                (observed_value - predicted_mean[0]) ** 2
                                / (predicted_std[0] ** 2 + 1e-8)
                                + np.log(2 * np.pi * (predicted_std[0] ** 2 + 1e-8))
                            )

                        except (
                            ValueError,
                            TypeError,
                            FloatingPointError,
                            RuntimeWarning,
                        ) as exc:
                            logger.debug(
                                "GP free-energy prediction failed; using prior-based "
                                "fallback: %s",
                                exc,
                                exc_info=True,
                            )
                            # Fallback to simple calculation
                            complexity = (
                                0.5
                                * (observed_value - prior_mean) ** 2
                                / prior_variance
                            )
                            accuracy = -0.5 * np.log(2 * np.pi * prior_variance)
                    else:
                        # Simple free energy based on deviation from prior
                        complexity = (
                            0.5 * (observed_value - prior_mean) ** 2 / prior_variance
                        )
                        accuracy = -0.5 * np.log(2 * np.pi * prior_variance)

                    var_fe = complexity - accuracy
                    cell_fe += var_fe
                    complexity_sum += complexity
                    accuracy_sum += accuracy

                    # Store variable-specific free energy
                    if var not in free_energy_metrics["variable_free_energy"]:
                        free_energy_metrics["variable_free_energy"][var] = 0.0
                    free_energy_metrics["variable_free_energy"][var] += var_fe

            free_energy_metrics["spatial_free_energy"][cell] = cell_fe
            total_fe += cell_fe

        free_energy_metrics["total_free_energy"] = total_fe
        free_energy_metrics["complexity_term"] = complexity_sum
        free_energy_metrics["accuracy_term"] = accuracy_sum
        free_energy_metrics["components"] = {
            "complexity": complexity_sum,
            "accuracy": accuracy_sum,
        }

        return free_energy_metrics

    def compute_spatial_priors(
        self, variable: str = "vegetation_density", n_states: int = 4
    ) -> dict[str, np.ndarray]:
        """
        Compute Moran's I weighted spatial priors for Active Inference models.

        This generates Dirichlet-style prior parameters (or probability distributions)
        for each cell, skewed by the local spatial autocorrelation (Local Moran's I)
        of the specified environmental variable.

        Args:
            variable: The environmental variable to compute priors from
            n_states: Number of discrete states in the generative model

        Returns:
            Dictionary mapping H3 cell to prior probability distribution
        """
        priors: dict[str, np.ndarray] = {}

        # Extract values
        values = []
        cells = list(self.environmental_states.keys())
        for cell in cells:
            state = self.environmental_states[cell]
            values.append(getattr(state, variable, 0.5))

        if not values or not self.spatial_graph:
            # Fallback uniform priors
            return {c: np.ones(n_states) / n_states for c in cells}

        values_arr = np.array(values)
        mean_val = np.mean(values_arr)
        std_val = np.std(values_arr) + 1e-8
        z_scores = (values_arr - mean_val) / std_val

        # Compute Local Moran's I
        local_morans = {}
        for i, cell in enumerate(cells):
            neighbors = self.spatial_graph.neighbors.get(cell, {}).get(1, set())
            if not neighbors:
                local_morans[cell] = 0.0
                continue

            neighbor_z = []
            for n in neighbors:
                try:
                    idx = cells.index(n)
                    neighbor_z.append(z_scores[idx])
                except ValueError:
                    continue

            if neighbor_z:
                # Local Moran's I: z_i * sum(w_ij * z_j) where w_ij is row-normalized weight
                w_ij = 1.0 / len(neighbor_z)
                lag_z = np.sum(np.array(neighbor_z) * w_ij)
                local_morans[cell] = z_scores[i] * lag_z
            else:
                local_morans[cell] = 0.0

        # Map Moran's I into categorical priors
        for cell in cells:
            I_i = local_morans[cell]
            val = getattr(self.environmental_states[cell], variable, 0.5)

            # Base prior centered around value
            # e.g., if val=0.8, skew towards higher states
            idx = int(np.clip(val * n_states, 0, n_states - 1))
            prior = np.ones(n_states) * 0.1
            prior[idx] += 0.5

            # Use Local Moran's I as a precision weighting (confidence in spatial cluster)
            # High I_i (positive clustering) -> sharper prior
            # Low/Negative I_i (outlier/noise) -> flatter prior
            precision_factor = np.clip(1.0 + I_i, 0.1, 5.0)
            prior = prior**precision_factor

            priors[cell] = prior / np.sum(prior)

        return priors

    def get_environmental_summary(self) -> dict[str, Any]:
        """
        Get comprehensive summary of environmental state and analysis.

        Returns:
            Dictionary containing environmental summary statistics
        """
        summary = {
            "spatial_domain": {
                "n_cells": len(self.environmental_states),
                "h3_resolution": self.h3_resolution,
                "boundary_extent": self._compute_spatial_extent(),
                "coverage_area_km2": sum(
                    h3.cell_area(c, "km^2") for c in self.environmental_states
                ),
            },
            "temporal_domain": {
                "n_observations": len(self.observation_history),
                "time_span": self._compute_time_span(),
                "last_update": (
                    max([obs["timestamp"] for obs in self.observation_history])
                    if self.observation_history
                    else None
                ),
            },
            "environmental_variables": self.environmental_variables,
            "observation_history": {
                "n_observations": len(self.observation_history),
                "time_span": self._compute_time_span(),
                "last_update": (
                    max([obs["timestamp"] for obs in self.observation_history])
                    if self.observation_history
                    else None
                ),
            },
            "prediction_models": {
                "trained_variables": list(self.gp_models.keys()),
                "model_status": self._assess_model_status(),
            },
            "model_status": self._assess_model_status(),
        }

        # Variable statistics
        for var in self.environmental_variables:
            values = []
            for env_state in self.environmental_states.values():
                if hasattr(env_state, var):
                    values.append(getattr(env_state, var))

            if values:
                summary[f"{var}_stats"] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }

        return summary

    def _compute_spatial_extent(self) -> dict[str, float]:
        """Compute spatial extent of the domain."""
        if not self.environmental_states:
            return {}

        lats, lngs = [], []
        for cell in self.environmental_states.keys():
            lat, lng = h3.cell_to_latlng(cell)
            lats.append(lat)
            lngs.append(lng)

        return {
            "lat_min": min(lats),
            "lat_max": max(lats),
            "lng_min": min(lngs),
            "lng_max": max(lngs),
            "lat_span": max(lats) - min(lats),
            "lng_span": max(lngs) - min(lngs),
        }

    def _compute_time_span(self) -> dict[str, float]:
        """Compute time span of observations."""
        if not self.observation_history:
            return {}

        timestamps = [obs["timestamp"] for obs in self.observation_history]
        return {
            "start_time": min(timestamps),
            "end_time": max(timestamps),
            "duration": max(timestamps) - min(timestamps),
        }

    def _assess_model_status(self) -> dict[str, str]:
        """Assess status of prediction models."""
        status = {}
        for var, model in self.gp_models.items():
            if hasattr(model, "X_train_") and model.X_train_ is not None:
                n_training_points = len(model.X_train_)
                if n_training_points >= 10:
                    status[var] = "well_trained"
                elif n_training_points >= 5:
                    status[var] = "partially_trained"
                else:
                    status[var] = "minimal_training"
            else:
                status[var] = "untrained"
        return status


class MultiScaleHierarchicalAnalyzer:
    """
    Multi-scale hierarchical analyzer for geospatial active inference.

    This analyzer implements hierarchical active inference across multiple spatial
    scales using H3 hexagonal grids, enabling analysis of emergent patterns,
    cross-scale interactions, and hierarchical belief propagation.
    """

    def __init__(
        self, base_resolution: int = 8, hierarchy_levels: int = 3, scale_factor: int = 3
    ):
        """
        Initialize Multi-Scale Hierarchical Analyzer.

        Args:
            base_resolution: Base H3 resolution (finest scale)
            hierarchy_levels: Number of hierarchical levels
            scale_factor: Factor for scaling between levels
        """
        self.base_resolution = base_resolution
        self.hierarchy_levels = hierarchy_levels
        self.scale_factor = scale_factor

        # Hierarchical structures
        self.hierarchical_graphs: dict[str, LevelSpatialGraph] = {}
        self.hierarchical_beliefs: dict[str, dict[str, np.ndarray]] = {}
        self.scale_relationships: dict[str, Any] = {}

        logger.info(
            f"Initialized MultiScaleHierarchicalAnalyzer with {hierarchy_levels} levels"
        )

    def _create_hierarchical_h3_model(
        self, base_resolution: int, boundary: dict[str, Any], levels: int
    ) -> dict[str, LevelSpatialGraph]:
        """Create hierarchical H3 models at different resolutions."""
        hierarchical_graphs: dict[str, LevelSpatialGraph] = {}

        for level in range(levels):
            resolution = max(0, base_resolution - level)
            level_name = f"level_{level}_res_{resolution}"

            # Generate H3 cells for this resolution
            level_cells = self._generate_h3_cells_from_boundary(boundary, resolution)

            # Create spatial graph for this level
            level_graph = self._create_level_spatial_graph(level_cells)
            hierarchical_graphs[level_name] = level_graph

        return hierarchical_graphs

    def _generate_h3_cells_from_boundary(
        self, boundary: dict[str, Any], resolution: int
    ) -> list[str]:
        """Generate H3 cells from boundary at specified resolution."""
        cells = set()

        if "coordinates" in boundary:
            coord_list = boundary["coordinates"][0][0]
            for coord in coord_list:
                try:
                    lng, lat = float(coord[0]), float(coord[1])
                    cell = h3.latlng_to_cell(lat, lng, resolution)
                    cells.add(cell)

                    # Add nearby cells for coverage
                    for ring_distance in range(1, 2):
                        neighbors = h3.grid_ring(cell, ring_distance)
                        if isinstance(neighbors, list):
                            cells.update(neighbors)
                        else:
                            cells.update(list(neighbors))

                except (ValueError, TypeError):
                    continue

        return list(cells)

    def _create_level_spatial_graph(self, cells: list[str]) -> LevelSpatialGraph:
        """Create spatial graph for a hierarchical level."""
        return LevelSpatialGraph(cells)

    def initialize_hierarchy(self, boundary: dict[str, Any]) -> None:
        """Initialize hierarchical structure."""
        self.hierarchical_graphs = self._create_hierarchical_h3_model(
            self.base_resolution, boundary, self.hierarchy_levels
        )

        # Initialize beliefs for each level
        for level_name, graph in self.hierarchical_graphs.items():
            self.hierarchical_beliefs[level_name] = {}
            for cell in graph.cells:
                # Initialize with uniform beliefs (4-state model)
                self.hierarchical_beliefs[level_name][cell] = np.ones(4) / 4

        logger.info(
            f"Initialized hierarchy with {len(self.hierarchical_graphs)} levels"
        )

    def propagate_beliefs_hierarchically(
        self,
        bottom_up_evidence: dict[str, dict[str, np.ndarray]],
        top_down_priors: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Propagate beliefs hierarchically using message passing.

        Implements hierarchical active inference with bottom-up evidence
        propagation and top-down prior propagation.

        Args:
            bottom_up_evidence: Evidence from lower levels
            top_down_priors: Prior beliefs from higher levels

        Returns:
            Updated hierarchical beliefs
        """
        updated_beliefs = {}

        # Bottom-up propagation
        for level_name in sorted(self.hierarchical_graphs.keys()):
            if level_name in bottom_up_evidence:
                # Update beliefs with evidence
                for cell, evidence in bottom_up_evidence[level_name].items():
                    if cell in self.hierarchical_beliefs[level_name]:
                        # Bayesian update
                        prior = self.hierarchical_beliefs[level_name][cell]
                        posterior = prior * evidence
                        posterior = posterior / (np.sum(posterior) + 1e-8)
                        self.hierarchical_beliefs[level_name][cell] = posterior

                # Propagate to higher levels
                self._aggregate_beliefs_upward(level_name)

        # Top-down propagation if priors provided
        if top_down_priors:
            for level_name in sorted(self.hierarchical_graphs.keys(), reverse=True):
                if level_name in top_down_priors:
                    self._propagate_priors_downward(level_name)

        # Collect updated beliefs
        for level_name in self.hierarchical_graphs.keys():
            updated_beliefs[level_name] = self.hierarchical_beliefs[level_name].copy()

        return updated_beliefs

    def _aggregate_beliefs_upward(
        self, lower_level: str, higher_level: str | None = None
    ) -> None:
        """Aggregate beliefs from lower level to higher level."""
        if higher_level is None:
            # Find the next higher level
            level_numbers = []
            for level_name in self.hierarchical_graphs.keys():
                if level_name.startswith("level_"):
                    level_num = int(level_name.split("_")[1])
                    level_numbers.append(level_num)

            current_level_num = int(lower_level.split("_")[1])
            higher_level_nums = [
                num for num in level_numbers if num > current_level_num
            ]

            if not higher_level_nums:
                return  # This is the highest level

            higher_level_num = min(higher_level_nums)
            higher_level = f"level_{higher_level_num}_res_{max(0, self.base_resolution - higher_level_num)}"

        if higher_level not in self.hierarchical_graphs:
            return

        # Aggregate beliefs from child cells to parent cells
        for parent_cell in self.hierarchical_beliefs[higher_level].keys():
            child_cells = self._find_child_cells(parent_cell, lower_level)

            if child_cells:
                # Average beliefs from children
                aggregated_belief = np.zeros(4)
                for child_cell in child_cells:
                    if child_cell in self.hierarchical_beliefs[lower_level]:
                        aggregated_belief += self.hierarchical_beliefs[lower_level][
                            child_cell
                        ]

                aggregated_belief = aggregated_belief / (len(child_cells) + 1e-8)

                # Update parent belief with aggregated evidence
                prior = self.hierarchical_beliefs[higher_level][parent_cell]
                posterior = (
                    0.7 * prior + 0.3 * aggregated_belief
                )  # Weighted combination
                posterior = posterior / (np.sum(posterior) + 1e-8)
                self.hierarchical_beliefs[higher_level][parent_cell] = posterior

    def _propagate_priors_downward(
        self, higher_level: str, lower_level: str | None = None
    ) -> None:
        """Propagate priors from higher level to lower level."""
        if lower_level is None:
            # Find the next lower level
            level_numbers = []
            for level_name in self.hierarchical_graphs.keys():
                if level_name.startswith("level_"):
                    level_num = int(level_name.split("_")[1])
                    level_numbers.append(level_num)

            current_level_num = int(higher_level.split("_")[1])
            lower_level_nums = [num for num in level_numbers if num < current_level_num]

            if not lower_level_nums:
                return  # This is the lowest level

            lower_level_num = max(lower_level_nums)
            lower_level = f"level_{lower_level_num}_res_{max(0, self.base_resolution - lower_level_num)}"

        if lower_level not in self.hierarchical_graphs:
            return

        # Propagate priors from parent to children
        for parent_cell, parent_belief in self.hierarchical_beliefs[
            higher_level
        ].items():
            child_cells = self._find_child_cells(parent_cell, lower_level)

            for child_cell in child_cells:
                if child_cell in self.hierarchical_beliefs[lower_level]:
                    # Modulate child belief with parent prior
                    child_belief = self.hierarchical_beliefs[lower_level][child_cell]
                    modulated_belief = 0.8 * child_belief + 0.2 * parent_belief
                    modulated_belief = modulated_belief / (
                        np.sum(modulated_belief) + 1e-8
                    )
                    self.hierarchical_beliefs[lower_level][child_cell] = (
                        modulated_belief
                    )

    def _find_child_cells(self, parent_cell: str, child_level: str) -> list[str]:
        """Find child cells that map to a parent cell."""
        child_cells = []

        # Get parent cell coordinates
        parent_lat, parent_lng = h3.cell_to_latlng(parent_cell)

        # Check which child cells fall within parent area
        if child_level in self.hierarchical_graphs:
            for child_cell in self.hierarchical_graphs[child_level].cells:
                try:
                    # Check if child cell is descendant of parent
                    child_resolution = h3.get_resolution(child_cell)
                    parent_resolution = h3.get_resolution(parent_cell)

                    if child_resolution > parent_resolution:
                        # Get parent of child at parent resolution
                        child_parent = h3.cell_to_parent(child_cell, parent_resolution)
                        if child_parent == parent_cell:
                            child_cells.append(child_cell)
                except (ValueError, TypeError):
                    logger.debug(
                        "Child-cell lookup failed for %s under %s",
                        child_cell,
                        parent_cell,
                        exc_info=True,
                    )
                    continue

        return child_cells

    def _find_parent_cell(self, child_cell: str, parent_level: str) -> str | None:
        """Find parent cell for a given child cell."""
        try:
            # Extract resolution from parent level name
            parent_resolution = int(parent_level.split("_res_")[1])
            parent_cell = h3.cell_to_parent(child_cell, parent_resolution)

            # Check if parent exists in the level
            if parent_level in self.hierarchical_graphs:
                if parent_cell in self.hierarchical_graphs[parent_level].cells:
                    return str(parent_cell)
        except Exception as e:
            logging.getLogger(__name__).debug("Parent cell lookup failed: %s", e)

        return None

    def analyze_cross_scale_interactions(self) -> dict[str, Any]:
        """
        Analyze interactions across different spatial scales.

        Returns:
            Analysis of cross-scale patterns and interactions
        """
        interactions: dict[str, dict[str, Any]] = {
            "scale_coherence": {},
            "information_flow": {},
            "emergence_indicators": {},
            "scale_dependencies": {},
        }

        level_names = sorted(self.hierarchical_graphs.keys())

        # Analyze coherence between adjacent scales
        for i in range(len(level_names) - 1):
            lower_level = level_names[i]
            higher_level = level_names[i + 1]

            coherence = self._compute_scale_coherence(lower_level, higher_level)
            interactions["scale_coherence"][f"{lower_level}_to_{higher_level}"] = (
                coherence
            )

        # Analyze information flow efficiency
        for level_name in level_names:
            beliefs = self.hierarchical_beliefs.get(level_name, {})
            if beliefs:
                entropies = []
                for belief in beliefs.values():
                    entropy = -np.sum(belief * np.log(belief + 1e-8))
                    entropies.append(entropy)

            interactions["information_flow"][level_name] = {
                "entropy": np.mean(entropies),
                "n_cells": len(beliefs),
            }

        # Detect emergence indicators
        for level_name in level_names:
            beliefs = self.hierarchical_beliefs.get(level_name, {})
            if beliefs and len(beliefs) > 1:
                belief_matrix = np.array(list(beliefs.values()))

                # Measure spatial correlation
                correlations = []
                for i in range(belief_matrix.shape[1]):  # For each belief dimension
                    for j in range(i + 1, belief_matrix.shape[1]):
                        left = belief_matrix[:, i]
                        right = belief_matrix[:, j]
                        if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
                            continue
                        corr = np.corrcoef(belief_matrix[:, i], belief_matrix[:, j])[
                            0, 1
                        ]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

                emergence_score = np.mean(correlations) if correlations else 0.0
                interactions["emergence_indicators"][level_name] = emergence_score

        return interactions

    def _compute_scale_coherence(self, lower_level: str, higher_level: str) -> float:
        """Compute coherence between two hierarchical levels."""
        if (
            lower_level not in self.hierarchical_beliefs
            or higher_level not in self.hierarchical_beliefs
        ):
            return 0.0

        coherence_scores = []

        # For each parent-child relationship
        for parent_cell in self.hierarchical_beliefs[higher_level].keys():
            child_cells = self._find_child_cells(parent_cell, lower_level)

            if len(child_cells) > 1:
                parent_belief = self.hierarchical_beliefs[higher_level][parent_cell]

                # Aggregate child beliefs
                child_beliefs = []
                for child_cell in child_cells:
                    if child_cell in self.hierarchical_beliefs[lower_level]:
                        child_beliefs.append(
                            self.hierarchical_beliefs[lower_level][child_cell]
                        )

                if child_beliefs:
                    avg_child_belief = np.mean(child_beliefs, axis=0)

                    # Compute similarity between parent and aggregated child beliefs
                    similarity = np.dot(parent_belief, avg_child_belief)
                    coherence_scores.append(similarity)

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def detect_emergent_patterns(self) -> list[dict[str, Any]]:
        """
        Detect emergent spatial patterns across hierarchical levels.

        Returns:
            List of detected emergent patterns
        """
        patterns = []

        for level_name, beliefs in self.hierarchical_beliefs.items():
            if len(beliefs) < 3:
                continue

            # Convert beliefs to matrix
            cells = list(beliefs.keys())
            belief_matrix = np.array([beliefs[cell] for cell in cells])

            # Detect spatial clusters of similar beliefs
            try:
                # Cluster based on belief similarity
                from sklearn.cluster import KMeans

                n_clusters = min(5, len(cells) // 2)

                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    belief_clusters = kmeans.fit_predict(belief_matrix)

                    # Analyze each cluster
                    for cluster_id in range(n_clusters):
                        cluster_mask = belief_clusters == cluster_id
                        cluster_cells = [
                            cells[i] for i in range(len(cells)) if cluster_mask[i]
                        ]

                        if len(cluster_cells) >= 2:
                            # Compute spatial extent
                            spatial_extent = self._compute_spatial_extent(cluster_cells)

                            # Compute belief coherence within cluster
                            cluster_beliefs = belief_matrix[cluster_mask]
                            coherence = 1.0 - np.mean(np.std(cluster_beliefs, axis=0))

                            pattern = {
                                "level": level_name,
                                "type": "belief_cluster",
                                "pattern_type": "belief_cluster",
                                "size": len(cluster_cells),
                                "coherence": coherence,
                                "spatial_extent": spatial_extent,
                                "cells": cluster_cells[:10],  # Limit for storage
                                "representative_belief": np.mean(
                                    cluster_beliefs, axis=0
                                ).tolist(),
                            }

                            patterns.append(pattern)

            except Exception as e:
                logger.warning(f"Pattern detection failed for {level_name}: {e}")

        # Sort patterns by size and coherence
        patterns.sort(key=lambda p: p["size"] * p["coherence"], reverse=True)

        return patterns

    def _compute_spatial_extent(self, cells: list[str]) -> dict[str, float]:
        """Compute spatial extent of a set of cells."""
        if not cells:
            return {}

        coordinates = [h3.cell_to_latlng(cell) for cell in cells]
        lats = [coord[0] for coord in coordinates]
        lngs = [coord[1] for coord in coordinates]

        return {
            "lat_min": min(lats),
            "lat_max": max(lats),
            "lng_min": min(lngs),
            "lng_max": max(lngs),
            "lat_span": max(lats) - min(lats),
            "lng_span": max(lngs) - min(lngs),
            "centroid_lat": np.mean(lats),
            "centroid_lng": np.mean(lngs),
            "area_km2": sum(h3.cell_area(c, "km^2") for c in cells),
        }


def analyze_multi_scale_patterns(
    hierarchical_graphs: dict[str, Any],
    hierarchical_beliefs: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """
    Analyze multi-scale patterns in hierarchical belief structures.

    Args:
        hierarchical_graphs: Hierarchical spatial graphs
        hierarchical_beliefs: Hierarchical belief distributions

    Returns:
        Multi-scale pattern analysis results
    """
    analysis: dict[str, Any] = {
        "scale_statistics": {},
        "pattern_diversity": {},
        "information_integration": {},
        "scale_relationships": {},
        "cross_scale_interactions": {},
        "emergent_patterns": [],
    }

    # Analyze each scale
    for level_name in hierarchical_graphs.keys():
        if level_name in hierarchical_beliefs:
            beliefs = hierarchical_beliefs[level_name]

            # Basic statistics
            n_cells = len(beliefs)

            # Entropy and information content
            entropies = []
            for belief in beliefs.values():
                entropy = -np.sum(belief * np.log(belief + 1e-8))
                entropies.append(entropy)

            mean_entropy = np.mean(entropies)
            entropy_variance = np.var(entropies)

            # Pattern diversity
            diversity = 0.0
            if len(beliefs) > 1:
                belief_matrix = np.stack(list(beliefs.values()))
                pair_dists = np.linalg.norm(
                    belief_matrix[:, None, :] - belief_matrix[None, :, :], axis=-1
                )
                first_idx, second_idx = np.triu_indices(belief_matrix.shape[0], k=1)
                diversity = np.mean(pair_dists[first_idx, second_idx])

            analysis["scale_statistics"][level_name] = {
                "n_cells": n_cells,
                "mean_entropy": mean_entropy,
                "entropy_variance": entropy_variance,
                "pattern_diversity": diversity,
                "mean_confidence": np.mean(
                    [np.max(belief) for belief in beliefs.values()]
                ),
            }

    # Cross-scale relationships
    level_names = sorted(hierarchical_graphs.keys())
    for i in range(len(level_names) - 1):
        lower_level = level_names[i]
        higher_level = level_names[i + 1]

        if lower_level in hierarchical_beliefs and higher_level in hierarchical_beliefs:
            # Information integration measure
            lower_entropies = []
            higher_entropies = []

            for belief in hierarchical_beliefs[lower_level].values():
                entropy = -np.sum(belief * np.log(belief + 1e-8))
                lower_entropies.append(entropy)

            for belief in hierarchical_beliefs[higher_level].values():
                entropy = -np.sum(belief * np.log(belief + 1e-8))
                higher_entropies.append(entropy)

            integration_ratio = (np.mean(higher_entropies) + 1e-8) / (
                np.mean(lower_entropies) + 1e-8
            )

            relationship_key = f"{lower_level}_to_{higher_level}"
            analysis["scale_relationships"][relationship_key] = {
                "integration_ratio": integration_ratio,
                "lower_complexity": np.mean(lower_entropies),
                "higher_complexity": np.mean(higher_entropies),
            }

    return analysis
