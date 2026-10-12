"""
Resource management model for active inference.

This model implements resource allocation and management using Active Inference
principles. Agents allocate resources across locations, balancing exploitation
of known high-value sites with exploration of uncertain areas.
"""

from typing import Any
import copy
import numpy as np
import logging

from geo_infer_act.models.base import BaseActiveInferenceModel, _dimension

logger = logging.getLogger(__name__)


class ResourceModel(BaseActiveInferenceModel):
    """Resource allocation modeling using active inference.

    Simulates resource dynamics across spatial locations with:
    - Resource depletion through harvesting actions
    - Natural replenishment over time
    - Location connectivity for resource flow
    - Free-energy-based allocation scoring

    States:
        resource_distribution: (n_resources x n_locations) matrix of resource levels
        location_demand: (n_locations,) vector of demand at each location

    Actions:
        Allocation vectors specifying how much of each resource to direct to each location.
    """

    def __init__(
        self,
        n_resources: int = 4,
        n_locations: int = 5,
        planning_horizon: int = 10,
        replenishment_rate: float = 0.05,
        depletion_rate: float = 0.1,
        config: dict[str, Any] | None = None,
        random_seed: int | None = None,
    ):
        """Initialize the Resource Model.

        Args:
            n_resources: Number of distinct resource types
            n_locations: Number of spatial locations
            planning_horizon: Steps to look ahead for planning
            replenishment_rate: Natural resource replenishment per step
            depletion_rate: Base resource depletion per harvesting action
            config: Optional configuration dictionary
        """
        super().__init__(config)
        if random_seed is None and config:
            random_seed = config.get("random_seed")
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.n_resources = _dimension(n_resources, "n_resources")
        self.n_locations = _dimension(n_locations, "n_locations")
        self.planning_horizon = _dimension(planning_horizon, "planning_horizon")
        for name, value in (
            ("replenishment_rate", replenishment_rate),
            ("depletion_rate", depletion_rate),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between zero and one")
        self.replenishment_rate = replenishment_rate
        self.depletion_rate = depletion_rate
        self.step_count = 0

        # Resource distribution: each entry in [0, 1] representing abundance
        self.resource_distribution = self.rng.uniform(
            0.3, 0.9, (self.n_resources, self.n_locations)
        )

        # Location connectivity: adjacency-like matrix for resource flow
        self.location_connectivity = self._build_connectivity()

        # Demand profile per location (how much each location consumes)
        self.location_demand = self.rng.uniform(0.02, 0.08, self.n_locations)
        self._initial_resource_distribution = self.resource_distribution.copy()
        self._initial_location_demand = self.location_demand.copy()

        # History tracking
        self.history: list[dict[str, Any]] = []

        logger.info(
            f"ResourceModel initialized: {n_resources} resources × {n_locations} locations, "
            f"replenishment={replenishment_rate}, depletion={depletion_rate}"
        )

    def _build_connectivity(self) -> np.ndarray:
        """Build location connectivity matrix with nearest-neighbor coupling."""
        conn = np.eye(self.n_locations) * 0.5
        for i in range(self.n_locations - 1):
            conn[i, i + 1] = 0.25
            conn[i + 1, i] = 0.25
        # Normalize rows
        row_sums = conn.sum(axis=1, keepdims=True)
        conn = conn / np.where(row_sums > 0, row_sums, 1.0)
        return np.asarray(conn)

    def step(self, actions: Any | None = None) -> tuple[dict[str, Any], bool]:
        """Advance the resource model by one step.

        Args:
            actions: Optional allocation actions. Can be:
                - None: no harvesting, only natural dynamics
                - np.ndarray of shape (n_resources, n_locations): harvest amounts
                - Dict with 'allocations' key containing the above

        Returns:
            Tuple of (state_dict, done_flag):
                - state_dict: current resource state and metrics
                - done_flag: True if any resource is fully depleted everywhere
        """
        alloc = None
        if actions is not None:
            if isinstance(actions, dict):
                if "allocations" not in actions:
                    raise ValueError("Allocation action must contain allocations")
                actions = actions["allocations"]
            alloc = np.asarray(actions, dtype=float)
            if (
                alloc.shape != self.resource_distribution.shape
                or not np.all(np.isfinite(alloc))
                or np.any(alloc < 0)
            ):
                raise ValueError(
                    "Allocation actions must be finite nonnegative arrays matching the resource grid"
                )
        self.step_count += 1
        prev_distribution = self.resource_distribution.copy()

        # 1. Natural replenishment (logistic growth toward carrying capacity of 1.0)
        growth = (
            self.replenishment_rate
            * self.resource_distribution
            * (1.0 - self.resource_distribution)
        )
        self.resource_distribution += growth

        # 2. Natural demand depletion at each location
        for loc in range(self.n_locations):
            self.resource_distribution[:, loc] -= self.location_demand[loc]
        self.resource_distribution = np.maximum(self.resource_distribution, 0)

        # 3. Apply harvesting actions if provided
        harvest_yield = np.zeros((self.n_resources, self.n_locations))
        if alloc is not None:
            harvest = np.minimum(
                alloc * self.depletion_rate, self.resource_distribution
            )
            self.resource_distribution -= harvest
            harvest_yield = harvest

        # 4. Resource flow between connected locations
        # Transport conserves resource mass under row-stochastic connectivity.
        flow = 0.1 * (
            self.resource_distribution @ self.location_connectivity
            - self.resource_distribution
        )
        self.resource_distribution += flow

        # 5. Clamp to [0, 1]
        self.resource_distribution = np.clip(self.resource_distribution, 0.0, 1.0)

        # Compute metrics
        total_resources = self.resource_distribution.sum()
        change = self.resource_distribution - prev_distribution
        sustainability_score = float(np.mean(self.resource_distribution > 0.2))

        # Free-energy-inspired score: surprise from deviation from preferred (high) resource levels
        preferred = np.ones_like(self.resource_distribution) * 0.7
        free_energy = float(np.sum((self.resource_distribution - preferred) ** 2))

        state = {
            "resource_distribution": self.resource_distribution.copy(),
            "total_resources": float(total_resources),
            "harvest_yield": harvest_yield,
            "resource_change": change,
            "sustainability_score": sustainability_score,
            "free_energy": free_energy,
            "step": self.step_count,
        }

        self.history.append(copy.deepcopy(state))

        # Done if any resource type is depleted across all locations
        done = bool(np.any(self.resource_distribution.sum(axis=1) < 0.01))

        return state, done

    def reset(self) -> dict[str, Any]:
        """Reset the resource model to initial random state.

        Returns:
            Initial state dictionary
        """
        self.resource_distribution = self._initial_resource_distribution.copy()
        self.location_demand = self._initial_location_demand.copy()
        self.step_count = 0
        self.history = []
        logger.info("ResourceModel reset")
        return {
            "resource_distribution": self.resource_distribution.copy(),
            "total_resources": float(self.resource_distribution.sum()),
            "step": 0,
        }

    def get_allocation_scores(self) -> np.ndarray:
        """Compute free-energy-based allocation priority scores.

        Locations with lower resources get higher allocation priority,
        weighted by demand. This implements the epistemic value of
        reducing uncertainty about resource sufficiency.

        Returns:
            Priority scores of shape (n_resources, n_locations)
        """
        # Priority increases where resources are scarce relative to demand
        scarcity = 1.0 - self.resource_distribution
        demand_weight = self.location_demand / (self.location_demand.sum() + 1e-10)
        scores = scarcity * demand_weight[np.newaxis, :]
        # Normalize per resource type
        row_sums = scores.sum(axis=1, keepdims=True)
        scores = scores / np.where(row_sums > 0, row_sums, 1.0)
        return np.asarray(scores)
