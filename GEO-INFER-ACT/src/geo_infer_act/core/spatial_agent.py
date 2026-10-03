#!/usr/bin/env python
"""
Spatial Active Inference Agent for GEO-INFER-ACT.

This module provides a comprehensive spatial active inference agent that operates
on H3 hexagonal grids, implementing real spatial belief propagation, precision
dynamics, and policy selection using pymdp-compatible methods.

Features:
- H3-based spatial state representations with neighbor propagation
- Spatial precision dynamics weighted by local coherence
- Geospatial observation likelihoods with H3 cells
- Real pymdp integration for spatial policy selection
- Comprehensive logging of spatial beliefs and free energy

References:
    - Parr, T., Pezzulo, G., & Friston, K. (2022). Active Inference
    - Uber's H3 hexagonal hierarchical geospatial indexing system
"""

import logging
import numpy as np
from typing import Any, cast
from collections.abc import Iterable
from datetime import datetime
import json

from geo_infer_act.core.types import (
    ActiveInferenceStepResult,
    H3GridInferenceResult,
    H3SpatialConsistency,
    NestedH3BeliefUpdateResult,
    NestedH3GridInferenceResult,
    SpatialInferenceTrace,
)
from geo_infer_act.utils.h3_adapter import (
    edge_count_from_graph,
    get_h3_adapter,
    get_nested_h3_grid_class,
    normalize_belief_vector,
)
from geo_infer_act.utils.pymdp_adapter import run_pymdp_step
from geo_infer_act.utils.spatial_diagnostics import SpatialDiagnostics

logger = logging.getLogger(__name__)


class SpatialActiveInferenceAgent:
    """
    Active Inference agent operating on H3 hexagonal spatial grids.

    This agent implements spatial active inference with:
    - Belief propagation across H3 neighbors
    - Precision-weighted spatial diffusion
    - Expected free energy minimization for spatial policies
    - Comprehensive logging and diagnostics

    Attributes:
        h3_resolution: H3 resolution level (0-15)
        cells: List of H3 cell indices
        beliefs: Current belief distribution over states per cell
        precision: Spatial precision matrix
        free_energy_history: Record of free energy over time
    """

    def __init__(
        self,
        h3_resolution: int = 9,
        boundary: dict[str, Any] | None = None,
        initial_cells: list[str] | None = None,
        state_dim: int = 4,
        obs_dim: int = 4,
        diffusion_rate: float = 0.1,
        precision_scale: float = 1.0,
        enable_logging: bool = True,
        max_entries: int = 1_000_000,
    ):
        """
        Initialize the Spatial Active Inference Agent.

        Args:
            h3_resolution: H3 resolution level (0-15), higher = smaller cells
            boundary: GeoJSON-like boundary dict with 'coordinates' key
            initial_cells: Pre-specified list of H3 cell indices
            state_dim: Dimension of hidden states per cell
            obs_dim: Dimension of observations per cell
            diffusion_rate: Rate of belief diffusion to neighbors (0-1)
            precision_scale: Base precision for observations
            enable_logging: Whether to log inference steps
        """
        for name, value, minimum, maximum in (
            ("max_entries", max_entries, 1, None),
            ("h3_resolution", h3_resolution, 0, 15),
            ("state_dim", state_dim, 1, max_entries),
            ("obs_dim", obs_dim, 1, max_entries),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < minimum
                or (maximum is not None and value > maximum)
            ):
                raise ValueError(f"{name} must be an integer in its supported range")
        if not np.isfinite(diffusion_rate) or not 0 <= diffusion_rate <= 1:
            raise ValueError("diffusion_rate must be finite and between zero and one")
        if not np.isfinite(precision_scale) or precision_scale <= 0:
            raise ValueError("precision_scale must be finite and positive")
        self.max_entries = max_entries
        self.h3_resolution = h3_resolution
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.diffusion_rate = float(diffusion_rate)
        self.precision_scale = precision_scale
        self.enable_logging = enable_logging

        # Initialize cells
        self.cells: list[str] = []
        self.cell_to_idx: dict[str, int] = {}
        self.neighbor_map: dict[str, list[str]] = {}

        if initial_cells is not None:
            self._initialize_from_cells(initial_cells)
        elif boundary is not None:
            self._initialize_from_boundary(boundary)
        else:
            # Default: single cell at null island
            self._initialize_from_cells(self._get_default_cells())

        # Initialize beliefs uniformly
        n_cells = len(self.cells)
        self.beliefs = np.ones((n_cells, state_dim)) / state_dim

        # Precision matrix: n_cells x n_cells, higher = more confidence
        self.precision = np.eye(n_cells) * precision_scale

        # Observation model (A matrix): P(o|s) per cell
        # Shape: (n_cells, obs_dim, state_dim)
        self.observation_model = self._initialize_observation_model()

        # Transition model (B matrix): P(s'|s,a) per cell
        # Shape: (n_cells, state_dim, state_dim, n_actions)
        self.n_actions = 5  # stay, north, south, east, west
        self.transition_model = self._initialize_transition_model()

        # Preferences (C vector): preferred observations
        self.preferences = np.zeros((n_cells, obs_dim))
        self.preferences[:, 0] = 1.0  # Prefer first observation type

        # History tracking
        self.step_count = 0
        self.free_energy_history: list[float] = []
        self.belief_history: list[np.ndarray] = []
        self.action_history: list[dict] = []
        self.observation_history: list[dict] = []
        self.latest_pymdp_cell_metadata: dict[str, dict[str, Any]] = {}

        # Logging
        self.log_entries: list[dict] = []

        if self.enable_logging:
            logger.info(
                f"SpatialActiveInferenceAgent initialized: "
                f"{len(self.cells)} cells at resolution {h3_resolution}"
            )

    def enable_nested_h3_spatial(
        self,
        resolutions: list[int],
        boundary: dict[str, Any] | None = None,
        cells: list[str] | None = None,
        top_down_weight: float = 0.15,
    ) -> dict[str, Any]:
        """
        Reconfigure the agent to operate on the leaf cells of a nested H3 grid.

        The hierarchy is constructed by GEO-INFER-SPACE and stored for
        ``step_nested`` diagnostics. The regular ``step`` method remains a flat
        leaf-level H3 update.
        """
        if not boundary and not cells:
            raise ValueError("Provide either a boundary or H3 cells")
        if not 0.0 <= float(top_down_weight) <= 1.0:
            raise ValueError("top_down_weight must be between 0.0 and 1.0")
        NestedH3Grid = get_nested_h3_grid_class()
        grid = NestedH3Grid(name="spatial_agent_nested_h3")
        if boundary is not None:
            hierarchy = grid.build_h3_hierarchy_from_boundary(boundary, resolutions)
        else:
            hierarchy = grid.build_h3_hierarchy_from_cells(cells or [], resolutions)
        validation = hierarchy.get("validation", {})
        if not validation.get("is_valid", False):
            raise ValueError(f"Invalid nested H3 hierarchy: {validation}")

        self.nested_h3_mode = True
        self.nested_h3_grid = grid
        self.nested_h3_hierarchy = hierarchy
        self.nested_h3_resolutions = [int(value) for value in hierarchy["resolutions"]]
        self.nested_h3_top_down_weight = float(top_down_weight)
        self.h3_resolution = self.nested_h3_resolutions[-1]
        self._initialize_from_cells(hierarchy["leaf_cells"])
        n_cells = len(self.cells)
        self.beliefs = np.ones((n_cells, self.state_dim)) / self.state_dim
        self.precision = np.eye(n_cells) * self.precision_scale
        self.observation_model = self._initialize_observation_model()
        self.transition_model = self._initialize_transition_model()
        self.preferences = np.zeros((n_cells, self.obs_dim))
        self.preferences[:, 0] = 1.0
        return cast(dict[str, Any], hierarchy)

    def _get_default_cells(self) -> list[str]:
        """Generate default H3 cells for testing."""
        try:
            # San Francisco Bay area center
            adapter = get_h3_adapter()
            center_lat, center_lng = 37.7749, -122.4194
            center_cell = adapter.latlng_to_cell(
                center_lat, center_lng, self.h3_resolution
            )
            cells = [center_cell] + list(adapter.grid_ring(center_cell, 1))
            return cells
        except RuntimeError as exc:
            raise RuntimeError(
                "SpatialActiveInferenceAgent requires GEO-INFER-SPACE or h3-py"
            ) from exc

    def _initialize_from_cells(self, cells: list[str]) -> None:
        """Initialize agent from list of H3 cells."""
        adapter = get_h3_adapter()
        candidate_cells = adapter.validate_cells(cells)
        if not candidate_cells or any(
            adapter.get_resolution(cell) != self.h3_resolution
            for cell in candidate_cells
        ):
            raise ValueError("Cells must be nonempty and match h3_resolution")
        n_cells = len(candidate_cells)
        if (
            n_cells * n_cells
            + n_cells
            * (
                self.state_dim
                + self.obs_dim * self.state_dim
                + 5 * self.state_dim * self.state_dim
            )
            > self.max_entries
        ):
            raise ValueError("Spatial agent exceeds max_entries allocation budget")
        self.cells = candidate_cells
        self.cell_to_idx = {cell: idx for idx, cell in enumerate(self.cells)}
        self._build_neighbor_map()

    def _initialize_from_boundary(self, boundary: dict) -> None:
        """Initialize agent from GeoJSON boundary."""
        adapter = get_h3_adapter()
        cells = adapter.polygon_to_cells(boundary, self.h3_resolution)
        if not cells:
            raise ValueError(
                "Boundary contains no H3 cell centers at the requested resolution"
            )
        self._initialize_from_cells(cells)

    def _build_neighbor_map(self) -> None:
        """Build mapping of each cell to its neighbors."""
        if not self.cells:
            self.neighbor_map = {}
            return

        adapter = get_h3_adapter()
        adapter.validate_cells(self.cells)
        for cell in self.cells:
            neighbors = list(adapter.grid_ring(cell, 1))
            self.neighbor_map[cell] = [
                neighbor for neighbor in neighbors if neighbor in self.cell_to_idx
            ]

    def _initialize_observation_model(self) -> np.ndarray:
        """Initialize observation likelihood model P(o|s)."""
        n_cells = len(self.cells)
        # Each cell has its own observation model
        # Shape: (n_cells, obs_dim, state_dim) - A[o,s] = P(o|s)
        A = np.zeros((n_cells, self.obs_dim, self.state_dim))

        for c in range(n_cells):
            # Build observation model mapping states to observations
            # Start with a reasonable default that works for any dimension combo
            # Create base mapping
            for o in range(self.obs_dim):
                for s in range(self.state_dim):
                    if o == s:
                        A[c, o, s] = 0.8  # High probability on diagonal
                    elif abs(o - s) == 1:
                        A[c, o, s] = 0.1  # Some probability for adjacent
                    else:
                        A[c, o, s] = 0.02  # Small probability elsewhere

            # Normalize each column (sum over observations for each state = 1)
            for s in range(self.state_dim):
                col_sum = A[c, :, s].sum()
                if col_sum > 0:
                    A[c, :, s] = A[c, :, s] / col_sum
                else:
                    A[c, :, s] = 1.0 / self.obs_dim

        return A

    def _initialize_transition_model(self) -> np.ndarray:
        """Initialize transition model P(s'|s,a)."""
        n_cells = len(self.cells)
        B = np.zeros((n_cells, self.state_dim, self.state_dim, self.n_actions))
        for c in range(n_cells):
            for a in range(self.n_actions):
                if a == 0:  # Stay action: identity
                    B[c, :, :, a] = np.eye(self.state_dim)
                else:
                    # Movement actions: cycle states
                    shift = a % self.state_dim
                    B[c, :, :, a] = np.roll(np.eye(self.state_dim), shift, axis=0)
            # Normalize
            B[c] = B[c] / B[c].sum(axis=0, keepdims=True)
        return B

    def spatial_perception(
        self, observations: dict[str, np.ndarray], propagate_beliefs: bool = True
    ) -> dict[str, np.ndarray]:
        """
        Update beliefs based on spatial observations with neighbor propagation.

        This implements spatial active inference perception:
        1. Validates that observed cells belong to this agent's H3 lattice.
        2. Performs a local Bayesian update at each observed cell.
        3. Normalizes every posterior belief vector.
        4. Optionally diffuses beliefs to H3 neighbors using the precision matrix.
        5. Records a finite spatial free-energy diagnostic.

        Args:
            observations: Dict mapping H3 cell IDs to observation vectors
            propagate_beliefs: Whether to propagate beliefs to neighbors

        Returns:
            Dict mapping cell IDs to updated belief vectors

        Raises:
            ValueError: If an observation references a cell outside the current
                H3 lattice or a non-H3 cell is supplied on a real H3 path.
        """
        observations = self._validate_observations(observations)
        self.step_count += 1
        start_time = datetime.now()

        # Store observations
        self.observation_history.append(
            {
                "step": self.step_count,
                "observations": {k: v.tolist() for k, v in observations.items()},
                "timestamp": start_time.isoformat(),
            }
        )

        pre_beliefs = self.beliefs.copy()
        self.latest_pymdp_cell_metadata = {}

        # Step 1: pymdp 1.0.3 update at observed cells
        for cell_id, obs in observations.items():
            idx = self.cell_to_idx[cell_id]
            obs = np.asarray(obs).flatten()[: self.obs_dim]

            # Pad observation if needed
            if len(obs) < self.obs_dim:
                obs = np.concatenate([obs, np.zeros(self.obs_dim - len(obs))])

            pymdp_result = run_pymdp_step(
                observation=obs,
                observation_model=self.observation_model[idx],
                transition_model=self.transition_model[idx],
                preferences=self.preferences[idx],
                prior=self.beliefs[idx],
                action_count=self.n_actions,
                random_seed=(self.step_count * 1000) + idx,
            )
            self.beliefs[idx] = pymdp_result.beliefs
            self.latest_pymdp_cell_metadata[cell_id] = pymdp_result.to_metadata()

        # Step 2: Precision-weighted belief propagation
        if propagate_beliefs:
            self._propagate_beliefs_to_neighbors()

        # Step 3: Compute spatial free energy
        current_fe = self._compute_spatial_free_energy(observations)
        self.free_energy_history.append(current_fe)

        # Store belief history
        self.belief_history.append(self.beliefs.copy())

        # Logging
        if self.enable_logging:
            log_entry = {
                "step": self.step_count,
                "type": "perception",
                "n_observations": len(observations),
                "free_energy": current_fe,
                "belief_change": float(np.mean(np.abs(self.beliefs - pre_beliefs))),
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000,
            }
            self.log_entries.append(log_entry)
            logger.debug(f"Perception step {self.step_count}: FE={current_fe:.4f}")

        # Return updated beliefs as dict
        return {cell: self.beliefs[idx] for cell, idx in self.cell_to_idx.items()}

    def _propagate_beliefs_to_neighbors(self) -> None:
        """Propagate beliefs to neighboring cells with precision weighting."""
        if self.diffusion_rate <= 0:
            return

        new_beliefs = self.beliefs.copy()

        for cell_id in self.cells:
            idx = self.cell_to_idx[cell_id]
            neighbors = self.neighbor_map.get(cell_id, [])

            if not neighbors:
                continue

            # Collect neighbor beliefs
            neighbor_beliefs = []
            for n_cell in neighbors:
                n_idx = self.cell_to_idx[n_cell]
                neighbor_beliefs.append(self.beliefs[n_idx])

            if neighbor_beliefs:
                # Compute mean neighbor belief
                mean_neighbor = np.mean(neighbor_beliefs, axis=0)

                # Spatial coherence: precision-weighted average
                local_precision = self.precision[idx, idx]
                diffusion = self.diffusion_rate / (1 + local_precision)

                # Update: blend with neighbors
                new_beliefs[idx] = (1 - diffusion) * self.beliefs[
                    idx
                ] + diffusion * mean_neighbor
                new_beliefs[idx] = new_beliefs[idx] / (new_beliefs[idx].sum() + 1e-8)

        self.beliefs = new_beliefs

    def _compute_spatial_free_energy(
        self, observations: dict[str, np.ndarray]
    ) -> float:
        """
        Compute variational free energy across spatial domain.

        F = Sum over cells [D_KL(q(s)||p(s)) - E_q[log p(o|s)]]
        """
        total_fe = 0.0

        for cell_id, obs in observations.items():
            if cell_id not in self.cell_to_idx:
                continue

            idx = self.cell_to_idx[cell_id]
            q = self.beliefs[idx]  # Posterior

            # Prior: uniform
            p = np.ones(self.state_dim) / self.state_dim

            # KL divergence (complexity)
            kl = np.sum(q * np.log((q + 1e-8) / (p + 1e-8)))

            # Expected log likelihood (accuracy)
            obs = np.asarray(obs).flatten()[: self.obs_dim]
            if len(obs) < self.obs_dim:
                obs = np.concatenate([obs, np.zeros(self.obs_dim - len(obs))])

            A = self.observation_model[idx]
            expected_log_lik = 0.0
            for o_idx, o_val in enumerate(obs):
                expected_log_lik += np.sum(q * np.log(A[o_idx, :] + 1e-8)) * o_val

            # Free energy = complexity - accuracy
            cell_fe = kl - expected_log_lik
            total_fe += cell_fe

        return total_fe

    def spatial_action(self) -> dict[str, Any]:
        """
        Select action based on expected free energy minimization.

        This implements spatial policy selection by:
        1. Computing expected free energy for each action across cells
        2. Selecting action that minimizes global EFE
        3. Returning action with spatial context

        Returns:
            Dict with 'action', 'efe', 'cell_actions', 'confidence'
        """
        start_time = datetime.now()

        if self.latest_pymdp_cell_metadata:
            posterior_stack = np.asarray(
                [
                    meta["action_posterior"]
                    for meta in self.latest_pymdp_cell_metadata.values()
                ],
                dtype=float,
            )
            neg_efe_stack = np.asarray(
                [
                    meta["negative_expected_free_energy"]
                    for meta in self.latest_pymdp_cell_metadata.values()
                ],
                dtype=float,
            )
            pi = posterior_stack.mean(axis=0)
            pi = pi / (pi.sum() + 1e-12)
            neg_efe = neg_efe_stack.mean(axis=0)
            selected_action = int(np.argmax(pi))
            expected_free_energy = -float(neg_efe[selected_action])
            confidence = float(pi[selected_action])
            action_names = ["stay", "north", "south", "east", "west"]
            result = {
                "action": selected_action,
                "action_name": action_names[selected_action],
                "efe": expected_free_energy,
                "efe_all": (-neg_efe).astype(float).tolist(),
                "negative_expected_free_energy": neg_efe.astype(float).tolist(),
                "policy_distribution": pi.astype(float).tolist(),
                "confidence": confidence,
                "step": self.step_count,
                "backend": "inferactively-pymdp",
                "pymdp_cell_count": len(self.latest_pymdp_cell_metadata),
            }
            self.action_history.append(result)
            if self.enable_logging:
                self.log_entries.append(
                    {
                        "step": self.step_count,
                        "type": "action",
                        "selected": action_names[selected_action],
                        "confidence": confidence,
                        "efe": result["efe"],
                        "backend": "inferactively-pymdp",
                        "duration_ms": (datetime.now() - start_time).total_seconds()
                        * 1000,
                    }
                )
            return result

        # Compute expected free energy for each action
        efe_per_action = np.zeros(self.n_actions)
        cell_efe = np.zeros((len(self.cells), self.n_actions))

        for c_idx in range(len(self.cells)):
            for a in range(self.n_actions):
                # Predict next state: P(s'|s,a)
                B_a = self.transition_model[c_idx, :, :, a]
                predicted_state = B_a @ self.beliefs[c_idx]

                # Predict observation: P(o|s')
                A = self.observation_model[c_idx]
                predicted_obs = A @ predicted_state

                predictive_entropy = -np.sum(
                    predicted_state * np.log(predicted_state + 1e-8)
                )

                preferences = np.clip(self.preferences[c_idx], 1e-8, None)
                preferences = preferences / preferences.sum()
                pragmatic_surprise = -np.sum(predicted_obs * np.log(preferences + 1e-8))

                cell_efe[c_idx, a] = pragmatic_surprise - predictive_entropy

        # Aggregate across cells
        efe_per_action = np.sum(cell_efe, axis=0)

        # Softmax policy selection
        efe_normalized = efe_per_action - np.min(efe_per_action)
        exp_neg_efe = np.exp(-efe_normalized)
        pi = exp_neg_efe / (np.sum(exp_neg_efe) + 1e-8)

        # Select action
        selected_action = int(np.argmin(efe_per_action))
        confidence = float(pi[selected_action])

        action_names = ["stay", "north", "south", "east", "west"]

        result = {
            "action": selected_action,
            "action_name": action_names[selected_action],
            "efe": float(efe_per_action[selected_action]),
            "efe_all": efe_per_action.tolist(),
            "policy_distribution": pi.tolist(),
            "confidence": confidence,
            "step": self.step_count,
        }

        # Store action
        self.action_history.append(result)

        # Logging
        if self.enable_logging:
            log_entry = {
                "step": self.step_count,
                "type": "action",
                "selected": action_names[selected_action],
                "confidence": confidence,
                "efe": result["efe"],
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000,
            }
            self.log_entries.append(log_entry)
            logger.debug(
                f"Action step {self.step_count}: {action_names[selected_action]} (conf={confidence:.3f})"
            )

        return result

    def step(
        self,
        observations: dict[str, np.ndarray],
        propagate_beliefs: bool = True,
        return_result: bool = False,
    ) -> dict[str, Any] | H3GridInferenceResult:
        """
        Execute one full perception-action cycle.

        Args:
            observations: Dict mapping H3 cell IDs to observation vectors
            propagate_beliefs: Whether to propagate beliefs to neighbors
            return_result: Return ``H3GridInferenceResult`` when true

        Returns:
            Dict with ``beliefs``, ``action``, ``free_energy``, and ``step`` by
            default. With ``return_result=True``, returns
            ``H3GridInferenceResult`` containing per-cell
            ``ActiveInferenceStepResult`` values, aggregate free energy,
            spatial consistency, selected action metadata, and H3 resolution.
        """
        # Perception: update beliefs
        updated_beliefs = self.spatial_perception(observations, propagate_beliefs)

        # Action: select based on EFE
        action_result = self.spatial_action()

        result = {
            "beliefs": updated_beliefs,
            "action": action_result,
            "free_energy": self.free_energy_history[-1],
            "step": self.step_count,
        }
        if not return_result:
            return result

        consistency = self._compute_h3_result_consistency(updated_beliefs)
        cell_results = {
            cell: ActiveInferenceStepResult(
                beliefs=belief,
                action=action_result,
                free_energy=float(self.free_energy_history[-1]),
                expected_free_energy=action_result["efe"],
                observation=observations.get(cell),
                metadata={
                    "h3_cell": cell,
                    "step": self.step_count,
                    "pymdp": self.latest_pymdp_cell_metadata.get(cell),
                },
            )
            for cell, belief in updated_beliefs.items()
        }
        return H3GridInferenceResult(
            cell_results=cell_results,
            aggregate_free_energy=float(self.free_energy_history[-1]),
            spatial_consistency=consistency,
            metadata={
                "h3_resolution": self.h3_resolution,
                "selected_action": action_result,
                "step": self.step_count,
                "pymdp_backend": "inferactively-pymdp",
            },
        )

    def trace_step(
        self,
        observations: dict[str, np.ndarray],
        *,
        propagate_beliefs: bool = True,
        grid_result: H3GridInferenceResult | None = None,
        timestep: int | None = None,
        previous_beliefs: dict[str, Any] | None = None,
    ) -> SpatialInferenceTrace:
        """
        Return typed research diagnostics for one spatial H3 agent step.

        When a ``grid_result`` is supplied, it is reused so trace construction
        does not advance the agent a second time.
        """
        if grid_result is None:
            step_result = self.step(
                observations,
                propagate_beliefs=propagate_beliefs,
                return_result=True,
            )
            if not isinstance(step_result, H3GridInferenceResult):
                raise TypeError("step() must return a typed H3GridInferenceResult")
            grid_result = step_result
        return SpatialDiagnostics.build_h3_trace(
            scenario="spatial",
            timestep=self.step_count if timestep is None else timestep,
            cell_results=grid_result.cell_results,
            neighbor_map=self.neighbor_map,
            previous_beliefs=previous_beliefs,
            backend_metadata=grid_result.metadata,
            metadata={
                "aggregate_free_energy": grid_result.aggregate_free_energy,
                "spatial_consistency": grid_result.spatial_consistency,
            },
        )

    def step_nested(
        self,
        observations: dict[str, np.ndarray],
        propagate_beliefs: bool = True,
        return_result: bool = False,
        top_down_weight: float | None = None,
    ) -> dict[str, Any] | NestedH3GridInferenceResult:
        """
        Execute one nested H3 perception-action cycle on hierarchy leaf cells.

        Returns the regular leaf-level step results plus nested parent
        summaries, cross-level coherence, and aggregate free-energy diagnostics.
        """
        if not getattr(self, "nested_h3_mode", False):
            raise ValueError("Enable nested H3 spatial mode first")
        flat_result = self.step(
            observations,
            propagate_beliefs=propagate_beliefs,
            return_result=True,
        )
        if not isinstance(flat_result, H3GridInferenceResult):
            raise TypeError("step() must return a typed H3GridInferenceResult")
        from geo_infer_act.core.generative_model import GenerativeModel  # noqa: PLC0415

        gen = GenerativeModel(
            "categorical",
            {"state_dim": self.state_dim, "obs_dim": self.obs_dim},
        )
        gen.spatial_mode = True
        gen.nested_h3_mode = True
        gen.nested_h3_hierarchy = self.nested_h3_hierarchy
        gen.nested_h3_resolutions = self.nested_h3_resolutions
        gen.nested_h3_top_down_weight = float(
            top_down_weight
            if top_down_weight is not None
            else self.nested_h3_top_down_weight
        )
        gen.h3_cells = list(self.cells)
        finest = self.nested_h3_resolutions[-1]
        gen.spatial_graph = {
            cell: set(neighbors)
            for cell, neighbors in self.nested_h3_hierarchy["same_level_neighbors"]
            .get(str(finest), {})
            .items()
        }
        nested_update = gen.update_nested_h3_beliefs(
            observations,
            return_result=True,
            top_down_weight=top_down_weight,
        )
        if not isinstance(nested_update, NestedH3BeliefUpdateResult):
            raise TypeError(
                "update_nested_h3_beliefs() must return a typed "
                "NestedH3BeliefUpdateResult"
            )
        result = NestedH3GridInferenceResult(
            cell_results=flat_result.cell_results,
            nested_belief_update=nested_update,
            aggregate_free_energy=nested_update.aggregate_free_energy,
            spatial_consistency=nested_update.spatial_consistency,
            metadata={
                **flat_result.metadata,
                "nested_h3": True,
                "resolutions": list(self.nested_h3_resolutions),
            },
        )
        return result if return_result else result.to_dict()

    def trace_nested_step(
        self,
        observations: dict[str, np.ndarray],
        *,
        propagate_beliefs: bool = True,
        grid_result: NestedH3GridInferenceResult | None = None,
        timestep: int | None = None,
        previous_beliefs: dict[str, Any] | None = None,
        top_down_weight: float | None = None,
    ) -> SpatialInferenceTrace:
        """
        Return typed research diagnostics for one nested spatial H3 agent step.
        """
        if not getattr(self, "nested_h3_mode", False):
            raise ValueError("Enable nested H3 spatial mode first")
        if grid_result is None:
            step_result = self.step_nested(
                observations,
                propagate_beliefs=propagate_beliefs,
                return_result=True,
                top_down_weight=top_down_weight,
            )
            if not isinstance(step_result, NestedH3GridInferenceResult):
                raise TypeError(
                    "step_nested() must return a typed NestedH3GridInferenceResult"
                )
            grid_result = step_result
        nested_update = grid_result.nested_belief_update
        return SpatialDiagnostics.build_h3_trace(
            scenario="spatial",
            timestep=self.step_count if timestep is None else timestep,
            cell_results=grid_result.cell_results,
            neighbor_map=self.neighbor_map,
            previous_beliefs=previous_beliefs,
            hierarchy=self.nested_h3_hierarchy,
            parent_beliefs=nested_update.parent_beliefs,
            backend_metadata=grid_result.metadata,
            metadata={
                "aggregate_free_energy": grid_result.aggregate_free_energy,
                "spatial_consistency": grid_result.spatial_consistency,
                "nested_h3": True,
            },
        )

    def _validate_observations(
        self, observations: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Validate observation cell IDs and normalize keys to strings."""
        normalized = {str(cell): value for cell, value in observations.items()}
        unknown = sorted(set(normalized) - set(self.cell_to_idx))
        if unknown:
            raise ValueError(
                f"Observed cells are outside this spatial agent: {unknown[:5]}"
            )
        if normalized and not all(cell.startswith("cell_") for cell in self.cells):
            adapter = get_h3_adapter()
            adapter.validate_cells(normalized.keys())
        return normalized

    def set_preferences(self, preferences: dict[str, np.ndarray]) -> None:
        """Set preferred observations per cell."""
        for cell_id, pref in preferences.items():
            if cell_id in self.cell_to_idx:
                idx = self.cell_to_idx[cell_id]
                pref = np.asarray(pref).flatten()[: self.obs_dim]
                if len(pref) < self.obs_dim:
                    pref = np.concatenate([pref, np.zeros(self.obs_dim - len(pref))])
                self.preferences[idx] = pref

    def set_observation_model(self, cell_id: str, A: np.ndarray) -> None:
        """Set observation model for specific cell."""
        if cell_id in self.cell_to_idx:
            idx = self.cell_to_idx[cell_id]
            A = np.asarray(A)
            if A.shape == (self.obs_dim, self.state_dim):
                self.observation_model[idx] = A

    def set_transition_model(self, cell_id: str, B: np.ndarray) -> None:
        """Set transition model for specific cell."""
        if cell_id in self.cell_to_idx:
            idx = self.cell_to_idx[cell_id]
            B = np.asarray(B)
            if B.shape == (self.state_dim, self.state_dim, self.n_actions):
                self.transition_model[idx] = B

    def update_precision(self, cell_id: str, precision: float) -> None:
        """Update precision for a specific cell."""
        if cell_id in self.cell_to_idx:
            idx = self.cell_to_idx[cell_id]
            self.precision[idx, idx] = precision

    def get_diagnostics(self) -> dict[str, Any]:
        """
        Get comprehensive diagnostics for the agent.

        Returns:
            Dict with spatial metrics, belief stats, and history summaries
        """
        diagnostics = {
            "agent_info": {
                "n_cells": len(self.cells),
                "h3_resolution": self.h3_resolution,
                "state_dim": self.state_dim,
                "obs_dim": self.obs_dim,
                "step_count": self.step_count,
            },
            "belief_stats": {
                "mean_entropy": float(
                    np.mean([-np.sum(b * np.log(b + 1e-8)) for b in self.beliefs])
                ),
                "belief_variance": float(np.var(self.beliefs)),
                "max_belief": float(np.max(self.beliefs)),
                "min_belief": float(np.min(self.beliefs)),
            },
            "free_energy": {
                "current": (
                    self.free_energy_history[-1] if self.free_energy_history else None
                ),
                "mean": (
                    float(np.mean(self.free_energy_history))
                    if self.free_energy_history
                    else None
                ),
                "trend": self._compute_fe_trend(),
            },
            "spatial_coherence": self._compute_spatial_coherence(),
            "action_distribution": self._compute_action_distribution(),
        }
        return diagnostics

    def _compute_fe_trend(self) -> str:
        """Compute free energy trend."""
        if len(self.free_energy_history) < 3:
            return "insufficient_data"

        recent = self.free_energy_history[-5:]
        if len(recent) < 2:
            return "insufficient_data"

        slope = (recent[-1] - recent[0]) / len(recent)
        if slope < -0.01:
            return "decreasing"
        elif slope > 0.01:
            return "increasing"
        else:
            return "stable"

    def _compute_spatial_coherence(self) -> dict[str, float]:
        """Compute spatial coherence metrics."""
        coherences = []
        for cell_id in self.cells:
            idx = self.cell_to_idx[cell_id]
            neighbors = self.neighbor_map.get(cell_id, [])

            if neighbors:
                neighbor_beliefs = [
                    self.beliefs[self.cell_to_idx[n]] for n in neighbors
                ]
                mean_neighbor = np.mean(neighbor_beliefs, axis=0)
                coherence = 1 - np.mean(np.abs(self.beliefs[idx] - mean_neighbor))
                coherences.append(coherence)

        return {
            "mean": float(np.mean(coherences)) if coherences else 0.0,
            "std": float(np.std(coherences)) if coherences else 0.0,
        }

    def score_spatial_information_gain(
        self,
        target_resolution: int | None = None,
        beliefs: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """
        Score H3 cells by information gain (expected uncertainty reduction)
        for active-sensing / where-to-look-next decisions.

        Each cell's score is its belief entropy normalized by the maximum
        entropy of a uniform distribution over the state space, so a score of
        1.0 means maximally uncertain (highest information-gain opportunity)
        and 0.0 means fully resolved.  When ``target_resolution`` is supplied
        the per-cell scores are aggregated up to that coarser H3 resolution
        (average child score per parent cell), giving a multi-resolution
        spatial grid score.

        Args:
            target_resolution: Optional coarser H3 resolution to aggregate to.
            beliefs: Optional explicit per-cell belief map.  Defaults to the
                agent's current ``self.beliefs``.

        Returns:
            Dict with ``scores`` (per cell or per parent), ``best_cells``,
            ``resolution``, ``mean_score`` and ``uncertain_cell_fraction``.
        """
        n_cells = len(self.cells)
        working = (
            beliefs
            if beliefs is not None
            else {self.cells[i]: self.beliefs[i] for i in range(n_cells)}
        )
        scores: dict[str, float] = {}
        max_entropy = float(np.log(self.state_dim)) if self.state_dim > 1 else 1.0
        for cell in self.cells:
            value = working.get(cell)
            if value is None:
                continue
            vec = normalize_belief_vector(np.asarray(value, dtype=float).reshape(-1))
            if vec.size != self.state_dim:
                vec = np.resize(vec, self.state_dim)
            entropy = float(-np.sum(vec * np.log(vec + 1e-12)))
            if max_entropy > 1e-12:
                scores[cell] = float(np.clip(entropy / max_entropy, 0.0, 1.0))
            else:
                scores[cell] = 0.0

        if target_resolution is not None and scores:
            parent_scores: dict[str, list[float]] = {}
            try:
                adapter = get_h3_adapter()
                for cell, score in scores.items():
                    if adapter.get_resolution(cell) > target_resolution:
                        parent = adapter.cell_to_parent(cell, target_resolution)
                    else:
                        parent = cell
                    parent_scores.setdefault(parent, []).append(score)
                scores = {
                    parent: float(np.mean(child_scores))
                    for parent, child_scores in parent_scores.items()
                }
            except Exception as exc:
                logger.debug("Information-gain aggregation failed: %s", exc)
                aggregated_resolution = self.h3_resolution
            else:
                aggregated_resolution = target_resolution
        else:
            aggregated_resolution = self.h3_resolution

        best_cells = (
            sorted(scores.keys(), key=lambda cell: float(scores[cell]), reverse=True)
            if scores
            else []
        )
        score_values = list(scores.values()) if scores else [0.0]
        uncertain_fraction = float(
            np.mean([1.0 if value > 0.5 else 0.0 for value in score_values])
        )
        return {
            "scores": scores,
            "best_cells": best_cells,
            "best_score": float(scores[best_cells[0]]) if best_cells else 0.0,
            "resolution": int(aggregated_resolution),
            "mean_score": float(np.mean(score_values)),
            "uncertain_cell_fraction": uncertain_fraction,
            "max_entropy": max_entropy,
        }

    def _compute_h3_result_consistency(
        self, beliefs: dict[str, np.ndarray]
    ) -> H3SpatialConsistency:
        """Compute typed H3 spatial consistency for current cell beliefs."""
        if not beliefs:
            return H3SpatialConsistency(
                global_coherence=0.0,
                neighbor_correlations=0.0,
                cell_count=0,
                edge_count=0,
            )

        normalized = {
            cell: normalize_belief_vector(belief) for cell, belief in beliefs.items()
        }
        belief_matrix = np.vstack(list(normalized.values()))
        global_coherence = float(
            np.clip(1.0 - np.mean(np.std(belief_matrix, axis=0)), 0.0, 1.0)
        )
        correlations = []
        for cell, neighbors in self.neighbor_map.items():
            if cell not in normalized:
                continue
            source = normalized[cell]
            for neighbor in neighbors:
                if neighbor not in normalized:
                    continue
                target = normalized[neighbor]
                if np.std(source) <= 1e-12 or np.std(target) <= 1e-12:
                    correlations.append(1.0 if np.allclose(source, target) else 0.0)
                else:
                    correlations.append(float(np.corrcoef(source, target)[0, 1]))
        neighbor_correlations = (
            float(np.nanmean(correlations)) if correlations else global_coherence
        )
        return H3SpatialConsistency(
            global_coherence=global_coherence,
            neighbor_correlations=neighbor_correlations,
            cell_count=len(normalized),
            edge_count=edge_count_from_graph(
                cast(dict[str, Iterable[str]], self.neighbor_map)
            ),
        )

    def _compute_action_distribution(self) -> dict[str, int]:
        """Compute distribution of selected actions."""
        if not self.action_history:
            return {}

        action_counts: dict[str, int] = {}
        for entry in self.action_history:
            name = entry.get("action_name", "unknown")
            action_counts[name] = action_counts.get(name, 0) + 1
        return action_counts

    def export_results(self, filepath: str) -> None:
        """Export agent results to JSON file."""
        results = {
            "diagnostics": self.get_diagnostics(),
            "free_energy_history": self.free_energy_history,
            "action_history": self.action_history,
            "log_entries": self.log_entries,
            "cells": self.cells,
            "final_beliefs": {
                cell: self.beliefs[idx].tolist()
                for cell, idx in self.cell_to_idx.items()
            },
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)

        if self.enable_logging:
            logger.info(f"Results exported to {filepath}")

    def reset(self) -> None:
        """Reset agent to initial state."""
        n_cells = len(self.cells)
        self.beliefs = np.ones((n_cells, self.state_dim)) / self.state_dim
        self.step_count = 0
        self.free_energy_history = []
        self.belief_history = []
        self.action_history = []
        self.observation_history = []
        self.log_entries = []

        if self.enable_logging:
            logger.info("Agent reset to initial state")
