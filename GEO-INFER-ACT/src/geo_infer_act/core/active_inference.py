"""
Active Inference model implementation.

This module contains the main ActiveInferenceModel class that orchestrates
all components of active inference including belief updating, policy selection,
and free energy minimization.
"""

import copy
import numpy as np
from typing import Any, cast
from collections.abc import Iterable
import logging

from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.core.free_energy import FreeEnergyCalculator
from geo_infer_act.core.policy_selection import PolicySelector
from geo_infer_act.core.belief_updating import BayesianBeliefUpdate
from geo_infer_act.core.types import (
    ActiveInferenceStepResult,
    H3GridInferenceResult,
    H3SpatialConsistency,
    NestedH3BeliefUpdateResult,
    NestedH3GridInferenceResult,
    PolicyEvaluation,
    SpatialInferenceTrace,
)
from geo_infer_act.utils.h3_adapter import (
    edge_count_from_graph,
    get_h3_adapter,
    normalize_belief_vector,
)
from geo_infer_act.utils.math import normalize_distribution
from geo_infer_act.utils.analysis import ActiveInferenceAnalyzer
from geo_infer_act.utils.pymdp_adapter import PymdpStepResult, run_model_step

logger = logging.getLogger(__name__)


def _coerce_action_count(value: Any, default: int = 3) -> int:
    """Return a positive action count from scalar or pymdp-style values."""
    if value is None:
        value = default
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("num_controls must be a scalar or a single-item sequence")
    count = int(array.reshape(-1)[0])
    if count < 1:
        raise ValueError("num_controls must be positive")
    return count


def _local_perception_free_energy(
    prior: np.ndarray,
    posterior: np.ndarray,
    observation: np.ndarray,
    likelihood: np.ndarray,
) -> float:
    """Evaluate local count-likelihood VFE against its actual perception prior."""
    matrix = np.asarray(likelihood, dtype=float)
    matrix = matrix / matrix.sum(axis=0, keepdims=True)
    positive = posterior > 0
    observed = observation > 0
    log_likelihood = observation[observed] @ np.log(matrix[observed][:, positive])
    return float(
        np.sum(
            posterior[positive]
            * (np.log(posterior[positive]) - np.log(prior[positive]) - log_likelihood)
        )
    )


class ActiveInferenceModel:
    """
    Main class for active inference agents with support for nested models.
    """

    def __init__(self, model_type: str = "categorical", **kwargs: Any) -> None:
        """
        Initialize an Active Inference model.

        Args:
            model_type: Type of underlying generative model
            **kwargs: Additional parameters
        """
        self.model_type = model_type
        self.parameters = dict(kwargs)
        self.preferences = self.parameters.pop("preferences", None)
        policy_temperature = self.parameters.pop(
            "policy_temperature", self.parameters.pop("temperature", 1.0)
        )
        policy_selection_mode = self.parameters.pop(
            "policy_selection_mode", self.parameters.pop("selection_mode", "sample")
        )
        random_seed = self.parameters.pop("random_seed", None)
        self.random_seed = random_seed

        # Initialize core components
        self.generative_model: GenerativeModel | None = None
        self.free_energy_calculator = FreeEnergyCalculator()
        self.policy_selector = PolicySelector(
            temperature=policy_temperature,
            selection_mode=policy_selection_mode,
            random_seed=random_seed,
        )
        self.belief_updater = BayesianBeliefUpdate()

        # Analyzer Integration
        self.output_dir = kwargs.get("output_dir", None)
        self.analyzer = None
        if self.output_dir:
            self.analyzer = ActiveInferenceAnalyzer(self.output_dir)
            logger.info(
                f"Active Inference Analyzer enabled. Logging to {self.output_dir}"
            )

        # State variables
        self.current_beliefs: Any | None = None
        self.current_observations: Any | None = None
        self.current_actions: Any | None = None
        self.latest_policy_evaluation: PolicyEvaluation | None = None
        self.latest_policy_selection: dict[str, Any] | None = None
        self.latest_pymdp_result: PymdpStepResult | None = None
        self._perception_free_energy: float | None = None
        self.history: list[dict[str, Any]] = []

        logger.info(f"Initialized ActiveInferenceModel with type: {model_type}")

    def set_generative_model(self, model: GenerativeModel) -> None:
        """Set the generative model for this active inference agent."""
        self.generative_model = model
        self.current_observations = None
        self.current_actions = None
        self.latest_pymdp_result = None
        self.latest_policy_evaluation = None
        self.latest_policy_selection = None
        self._perception_free_energy = None
        if getattr(model, "model_type", None) and model.model_type != self.model_type:
            logger.info(
                "Aligning active inference model type with generative model: %s",
                model.model_type,
            )
            self.model_type = model.model_type
        self.current_beliefs = self._extract_model_beliefs(model)
        self._initial_beliefs = self._clone_beliefs(self.current_beliefs)
        if self.preferences is None:
            self.preferences = self._extract_model_preferences(model)

    def perceive(self, observation: np.ndarray) -> np.ndarray:
        """
        Update beliefs based on new observation.

        Args:
            observation: New sensory observation

        Returns:
            Updated beliefs (posterior distribution)
        """
        if self.generative_model is None:
            raise ValueError("Generative model must be set before perception")

        observation = np.asarray(observation, dtype=float).reshape(-1)
        self.current_observations = observation
        self.latest_pymdp_result = None
        self._perception_free_energy = None

        perception_prior = (
            self._extract_belief_vector(
                self.current_beliefs
                if self.current_beliefs is not None
                else self._extract_model_beliefs(self.generative_model)
            )
            if self.model_type == "categorical" and self._supports_pymdp_adapter()
            else None
        )
        updated_beliefs = self._update_beliefs_with_model(observation)
        self.current_beliefs = updated_beliefs
        if self.model_type == "categorical" and self._supports_pymdp_adapter():
            self.generative_model.beliefs = self._clone_beliefs(updated_beliefs)
            self._perception_free_energy = (
                self.latest_pymdp_result.free_energy
                if self.latest_pymdp_result is not None
                else _local_perception_free_energy(
                    perception_prior,
                    self._extract_belief_vector(updated_beliefs),
                    observation,
                    self.generative_model.observation_model,
                )
            )

        return cast(np.ndarray, self._clone_beliefs(self.current_beliefs))

    def act(self, available_actions: list[Any] | None = None) -> Any:
        """
        Select action based on expected free energy minimization.

        Args:
            available_actions: List of available actions

        Returns:
            Selected action
        """
        if self.generative_model is None:
            raise ValueError("Generative model must be set before action selection")
        if available_actions is not None and not available_actions:
            raise ValueError("available_actions must contain at least one action")

        if (
            self.model_type == "categorical"
            and self.current_observations is not None
            and self._supports_pymdp_adapter()
        ):
            try:
                action_count = (
                    len(available_actions)
                    if available_actions is not None
                    else _coerce_action_count(
                        self.parameters.get("num_controls"), default=3
                    )
                )
                pymdp_result = run_model_step(
                    self.generative_model,
                    self.current_observations,
                    action_count=action_count,
                    random_seed=int(self.random_seed or 0) + len(self.history),
                    prior=self.current_beliefs,
                    posterior=(
                        self._extract_belief_vector(self.current_beliefs)
                        if self._perception_free_energy is not None
                        else None
                    ),
                    perception_free_energy=self._perception_free_energy,
                )
                self.latest_pymdp_result = pymdp_result
                selected_index = pymdp_result.selected_action_index
                action = (
                    available_actions[selected_index % len(available_actions)]
                    if available_actions
                    else selected_index
                )
                selected_probability = float(
                    pymdp_result.policy_posterior[
                        selected_index % len(pymdp_result.policy_posterior)
                    ]
                )
                selected_neg_efe = float(
                    pymdp_result.negative_expected_free_energy[
                        selected_index % len(pymdp_result.negative_expected_free_energy)
                    ]
                )
                self.latest_policy_evaluation = PolicyEvaluation(
                    policy=action,
                    expected_free_energy=-selected_neg_efe,
                    probability=selected_probability,
                    index=selected_index,
                    metadata=pymdp_result.to_metadata(),
                )
                self.latest_policy_selection = {
                    "backend": "inferactively-pymdp",
                    "policy": action,
                    "evaluation": self.latest_policy_evaluation,
                    "pymdp": pymdp_result.to_metadata(),
                }
                self.current_actions = action
                return action
            except Exception as exc:
                if not self.parameters.get("allow_local_pymdp_fallback", False):
                    raise
                logger.debug(
                    "pymdp action selection failed; using local fallback: %s", exc
                )

        # Local expected-free-energy implementation.

        # A failed optional backend attempt must not leave diagnostics from a
        # previous backend-backed step attached to the local result.
        self.latest_pymdp_result = None

        # Generate default actions if none provided
        if available_actions is None:
            available_actions = list(
                range(
                    _coerce_action_count(
                        getattr(self.generative_model, "num_controls", None),
                        default=3,
                    )
                )
            )

        belief_vector = self._extract_belief_vector(self.current_beliefs)
        if belief_vector is None:
            belief_vector = np.ones(len(available_actions), dtype=float) / len(
                available_actions
            )

        policy_candidates = [
            {
                "action": action,
                "exploration_bonus": self.parameters.get("exploration_bonus", 0.1),
            }
            for action in available_actions
        ]
        policy_info = self.policy_selector.select_policy(
            belief_vector,
            policy_candidates,
            self._get_preferences_vector(len(belief_vector)),
        )
        policy = policy_info.get("policy", policy_info)
        selected_action = policy.get("action", policy)
        self.latest_policy_evaluation = policy_info.get("evaluation")
        self.latest_policy_selection = policy_info

        self.current_actions = selected_action
        return selected_action

    def update_observations(self, observations: dict[str, Any]) -> None:
        """Update observations for the active inference model."""
        self.current_observations = observations
        self._perception_free_energy = None
        self.latest_pymdp_result = None

    def update_preferences(self, preferences: dict[str, float]) -> None:
        """Update preferences for the active inference model."""
        self.set_preferences(preferences)

    def update_with_outcome(
        self, decision: dict[str, Any], outcome: dict[str, Any]
    ) -> None:
        """
        Update model based on decision outcome for learning.

        Stores the decision-outcome pair in history and updates beliefs
        if outcome contains observation data, implementing the
        perception-action loop closure in Active Inference.

        Args:
            decision: The decision/action taken
            outcome: The observed outcome
        """
        self.history.append(
            {
                "decision": decision,
                "outcome": outcome,
                "beliefs_at_decision": self._clone_beliefs(self.current_beliefs),
            }
        )
        if "observation" in outcome:
            obs = np.asarray(outcome["observation"], dtype=float).reshape(-1)
            self.perceive(obs)

    def generate_policies(
        self, available_actions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate policy options from available actions."""
        return available_actions

    def select_policy(self, policies: list[dict[str, Any]]) -> dict[str, Any]:
        """Select optimal policy from candidates."""
        if not policies:
            return {}
        belief_vector = self._extract_belief_vector(self.current_beliefs)
        if belief_vector is None:
            belief_vector = np.ones(len(policies), dtype=float) / len(policies)
        policy_info = self.policy_selector.select_policy(
            belief_vector,
            policies,
            self._get_preferences_vector(len(belief_vector)),
        )
        self.latest_policy_evaluation = policy_info.get("evaluation")
        self.latest_policy_selection = policy_info
        return cast(dict[str, Any], policy_info.get("policy", {}))

    def compute_expected_free_energy(self, policy: dict[str, Any]) -> float:
        """Compute expected free energy for a given policy.

        Delegates to FreeEnergyCalculator.compute_expected_free_energy using
        the model's current beliefs and preferences. EFE balances epistemic
        value (information gain) and pragmatic value (preference satisfaction).

        Args:
            policy: Policy dict, may contain 'exploration_bonus' and 'risk_preference'

        Returns:
            Expected free energy (float). Lower means the policy is more preferred.
        """
        belief_vector = self._extract_belief_vector(self.current_beliefs)
        if belief_vector is None:
            return float("inf")

        preferences = self._get_preferences_vector(len(belief_vector))

        return cast(
            float,
            self.free_energy_calculator.compute_expected_free_energy(
                belief_vector, policy, preferences
            ),
        )

    def step(
        self,
        observation: np.ndarray,
        available_actions: list[Any] | None = None,
        return_result: bool = False,
    ) -> tuple[np.ndarray, Any] | ActiveInferenceStepResult:
        """
        Perform one complete active inference step.

        Args:
            observation: Current observation
            available_actions: Available actions

        Returns:
            Tuple of (updated_beliefs, selected_action), or an
            ActiveInferenceStepResult when ``return_result`` is true.
        """
        observation_array = np.asarray(observation, dtype=float).reshape(-1)

        # Perception: update beliefs
        beliefs = self.perceive(observation_array)

        # Action: select policy
        action = self.act(available_actions)

        # Store step in history
        pymdp_metadata = (
            self.latest_pymdp_result.to_metadata()
            if self.latest_pymdp_result is not None
            else None
        )
        step_free_energy = (
            float(self.latest_pymdp_result.free_energy)
            if self.latest_pymdp_result is not None
            else (self.compute_free_energy() if beliefs is not None else np.inf)
        )
        step_data = {
            "observation": observation_array.copy(),
            "beliefs": self._clone_beliefs(beliefs),
            "action": action,
            "free_energy": step_free_energy,
        }
        self.history.append(step_data)

        # Comprehensive Logging via Analyzer
        if self.analyzer:
            # Gather deep diagnostics
            policies = {
                "selected": (
                    self.latest_policy_evaluation.policy
                    if self.latest_policy_evaluation is not None
                    else action
                ),
                "probability": (
                    self.latest_policy_evaluation.probability
                    if self.latest_policy_evaluation is not None
                    else None
                ),
                "expected_free_energy": (
                    self.latest_policy_evaluation.expected_free_energy
                    if self.latest_policy_evaluation is not None
                    else None
                ),
            }

            metrics = {
                "model_type": self.model_type,
                "free_energy": step_data["free_energy"],
            }

            self.analyzer.record_step(
                beliefs=step_data["beliefs"],
                observations=step_data["observation"],
                actions=action,
                policies=policies,
                free_energy=step_data["free_energy"],
                metrics=metrics,
            )

        if return_result:
            return ActiveInferenceStepResult(
                beliefs=self._clone_beliefs(beliefs),
                action=action,
                free_energy=float(step_data["free_energy"]),
                expected_free_energy=(
                    self.latest_policy_evaluation.expected_free_energy
                    if self.latest_policy_evaluation is not None
                    else None
                ),
                policy_evaluation=self.latest_policy_evaluation,
                observation=observation_array.copy(),
                metadata={
                    "policy_selection": self.latest_policy_selection,
                    "pymdp": pymdp_metadata,
                },
            )

        return beliefs, action

    def compute_free_energy(self) -> float:
        """Compute current variational free energy."""

        # Delegate to GenerativeModel if it supports free energy computation (handles hierarchical etc.)
        if self.generative_model is not None and hasattr(
            self.generative_model, "compute_free_energy"
        ):
            return self.generative_model.compute_free_energy()

        if self.current_beliefs is None:
            return np.inf

        if self.model_type == "categorical":
            belief_vector = self._extract_belief_vector(self.current_beliefs)
            if belief_vector is None:
                return np.inf
            observation_vector = self._get_observation_vector(len(belief_vector))
            preferences = self._get_preferences_vector(len(belief_vector))
            return cast(
                float,
                self.free_energy_calculator.compute_categorical_free_energy(
                    belief_vector, observation_vector, preferences
                ),
            )

        if self.model_type == "gaussian":
            gaussian_beliefs = self._ensure_gaussian_beliefs(self.current_beliefs)
            if gaussian_beliefs is None or self.current_observations is None:
                return np.inf
            gaussian_preferences = self._get_gaussian_preferences()
            return self.free_energy_calculator.compute_gaussian_free_energy(
                gaussian_beliefs["mean"],
                gaussian_beliefs["precision"],
                self.current_observations,
                gaussian_preferences.get("mean"),
                gaussian_preferences.get("precision"),
            )

        return np.inf

    def reset(self) -> None:
        """Reset the model to initial state."""
        if self.generative_model is not None:
            initial_beliefs = getattr(self, "_initial_beliefs", None)
            if initial_beliefs is None:
                initial_beliefs = self._extract_model_beliefs(self.generative_model)
                self._initial_beliefs = self._clone_beliefs(initial_beliefs)
            self.generative_model.beliefs = self._clone_beliefs(initial_beliefs)
            self.current_beliefs = self._clone_beliefs(initial_beliefs)
        else:
            self.current_beliefs = None

        self.current_observations = None
        self.current_actions = None
        self.latest_policy_evaluation = None
        self.latest_policy_selection = None
        self.latest_pymdp_result = None
        self._perception_free_energy = None
        self.history = []

    def get_history(self) -> list[dict[str, Any]]:
        """Get the complete history of interactions."""
        return [copy.deepcopy(entry) for entry in self.history]

    def get_current_state(self) -> dict[str, Any]:
        """Get current model state."""
        return {
            "beliefs": self._clone_beliefs(self.current_beliefs),
            "observations": (
                self.current_observations.copy()
                if isinstance(self.current_observations, np.ndarray)
                else self.current_observations
            ),
            "actions": self.current_actions,
            "free_energy": self.compute_free_energy(),
            "model_type": self.model_type,
        }

    def apply_to_h3(
        self, h3_obs: dict[str, np.ndarray], return_result: bool = False
    ) -> Any:
        """
        Update a spatial generative model from H3-indexed observations.

        This is the canonical ACT entry point for H3 belief updates. It requires
        a spatially enabled ``GenerativeModel``, validates real H3 v4 cell IDs
        through the model, normalizes per-cell beliefs, and returns finite
        aggregate free-energy and spatial-consistency diagnostics.

        Args:
            h3_obs: Mapping of H3 cell IDs to observation vectors.
            return_result: Return ``H3BeliefUpdateResult`` when true.

        Returns:
            Result returned by ``GenerativeModel.update_h3_beliefs`` containing
            per-cell beliefs, aggregate beliefs, and spatial consistency metrics.
        """
        if self.generative_model is None:
            raise ValueError("Set generative model first")
        return self.generative_model.update_h3_beliefs(
            h3_obs, return_result=return_result
        )

    def infer_over_h3_grid(
        self, h3_grid: dict[str, Any], return_result: bool = False
    ) -> Any:
        """
        Run independent one-step inference across an H3 observation grid.

        The method preserves the agent's current beliefs, observations, action,
        and interaction history after evaluating each cell, so callers can use
        it as read-only spatial scoring.

        Cell identifiers are validated with the ACT H3 adapter, policy
        selection is evaluated via expected free energy for each cell, and typed
        mode returns per-cell ``ActiveInferenceStepResult`` objects plus an
        ``H3SpatialConsistency`` summary.

        Args:
            h3_grid: Mapping of H3 cell IDs to observation vectors.
            return_result: Return ``H3GridInferenceResult`` when true.

        Returns:
            Mapping of each cell to updated beliefs, selected action,
            free-energy value, and precision diagnostic, or a typed grid
            inference result when ``return_result`` is true.
        """
        adapter = get_h3_adapter()
        observations_by_cell = {str(cell): obs for cell, obs in h3_grid.items()}
        observed_cells = adapter.validate_cells(observations_by_cell.keys())
        results = {}
        typed_results: dict[str, ActiveInferenceStepResult] = {}
        original_beliefs = self._clone_beliefs(self.current_beliefs)
        original_observations = (
            self.current_observations.copy()
            if isinstance(self.current_observations, np.ndarray)
            else self.current_observations
        )
        original_actions = self.current_actions
        original_policy_evaluation = self.latest_policy_evaluation
        original_policy_selection = self.latest_policy_selection
        original_pymdp_result = self.latest_pymdp_result
        original_perception_free_energy = self._perception_free_energy
        original_policy_rng_state = copy.deepcopy(
            self.policy_selector.rng.bit_generator.state
        )
        original_model_beliefs: Any = (
            copy.deepcopy(getattr(self.generative_model, "beliefs", None))
            if self.generative_model is not None
            else None
        )
        history_len = len(self.history)

        try:
            for cell in observed_cells:
                obs = observations_by_cell[cell]
                step_result = cast(
                    ActiveInferenceStepResult, self.step(obs, return_result=True)
                )
                typed_results[cell] = step_result
                results[cell] = {
                    "beliefs": self._clone_beliefs(step_result.beliefs),
                    "action": step_result.action,
                    "free_energy": step_result.free_energy,
                    "expected_free_energy": step_result.expected_free_energy,
                    "policy_evaluation": step_result.policy_evaluation,
                    "pymdp": step_result.metadata.get("pymdp"),
                    "precision": (
                        self.current_beliefs.get("precision", 1.0)
                        if isinstance(self.current_beliefs, dict)
                        else 1.0
                    ),
                }
        finally:
            self.current_beliefs = original_beliefs
            self.current_observations = original_observations
            self.current_actions = original_actions
            self.latest_policy_evaluation = original_policy_evaluation
            self.latest_policy_selection = original_policy_selection
            self.latest_pymdp_result = original_pymdp_result
            self._perception_free_energy = original_perception_free_energy
            self.policy_selector.rng.bit_generator.state = original_policy_rng_state
            if self.generative_model is not None:
                self.generative_model.beliefs = original_model_beliefs
            if len(self.history) > history_len:
                self.history = self.history[:history_len]

        if return_result:
            cell_beliefs = {
                cell: self._extract_belief_vector(result.beliefs)
                for cell, result in typed_results.items()
            }
            spatial_consistency = self._compute_h3_grid_consistency(cell_beliefs)
            aggregate_free_energy = (
                float(
                    np.mean([result.free_energy for result in typed_results.values()])
                )
                if typed_results
                else 0.0
            )
            return H3GridInferenceResult(
                cell_results=typed_results,
                aggregate_free_energy=aggregate_free_energy,
                spatial_consistency=spatial_consistency,
                metadata={
                    "adapter_source": adapter.source,
                    "h3_resolution": (
                        adapter.get_resolution(observed_cells[0])
                        if observed_cells
                        else None
                    ),
                    "pymdp_backend": "inferactively-pymdp",
                },
            )

        return results

    def trace_over_h3_grid(
        self,
        h3_grid: dict[str, Any],
        *,
        timestep: int = 0,
        previous_beliefs: dict[str, Any] | None = None,
        grid_result: H3GridInferenceResult | None = None,
        scenario: str = "h3",
    ) -> SpatialInferenceTrace:
        """
        Return research diagnostics for one H3 grid inference step.

        The trace includes per-cell posterior, VFE, negative-EFE, policy
        posterior, policy entropy, local coherence, posterior delta, belief
        flux, edge diagnostics, and per-resolution summaries. When
        ``grid_result`` is provided, it is reused to avoid re-running pymdp.
        """
        if self.generative_model is None:
            raise ValueError("Set generative model first")
        if grid_result is None:
            grid_result = self.infer_over_h3_grid(h3_grid, return_result=True)
        return self.generative_model.compute_h3_cell_diagnostics(
            grid_result.cell_results,
            timestep=timestep,
            scenario=scenario,
            previous_beliefs=previous_beliefs,
            backend_metadata=grid_result.metadata,
            metadata={
                "aggregate_free_energy": grid_result.aggregate_free_energy,
                "spatial_consistency": grid_result.spatial_consistency,
            },
        )

    def infer_over_nested_h3_grid(
        self,
        h3_grid: dict[str, Any],
        return_result: bool = False,
        top_down_weight: float | None = None,
    ) -> Any:
        """
        Run one-step inference across an enabled nested H3 hierarchy.

        The per-cell step scoring follows ``infer_over_h3_grid`` and preserves
        active-model state. The nested belief update is computed through the
        configured generative model and its belief state is restored before the
        method returns.
        """
        if self.generative_model is None:
            raise ValueError("Set generative model first")
        if not getattr(self.generative_model, "nested_h3_mode", False):
            raise ValueError("Enable nested H3 spatial mode on the generative model")

        original_model_beliefs: Any = copy.deepcopy(
            getattr(self.generative_model, "beliefs", None)
        )
        grid_result = self.infer_over_h3_grid(h3_grid, return_result=True)
        try:
            nested_update = self.generative_model.update_nested_h3_beliefs(
                h3_grid,
                return_result=True,
                top_down_weight=top_down_weight,
            )
        finally:
            self.generative_model.beliefs = original_model_beliefs

        if not isinstance(nested_update, NestedH3BeliefUpdateResult):
            raise TypeError(
                "update_nested_h3_beliefs() must return a typed "
                "NestedH3BeliefUpdateResult"
            )
        result = NestedH3GridInferenceResult(
            cell_results=grid_result.cell_results,
            nested_belief_update=nested_update,
            aggregate_free_energy=nested_update.aggregate_free_energy,
            spatial_consistency=nested_update.spatial_consistency,
            metadata={
                **grid_result.metadata,
                "nested_h3": True,
                "resolutions": nested_update.metadata.get("resolutions", []),
            },
        )
        if return_result:
            return result
        return result.to_dict()

    def trace_over_nested_h3_grid(
        self,
        h3_grid: dict[str, Any],
        *,
        timestep: int = 0,
        previous_beliefs: dict[str, Any] | None = None,
        grid_result: NestedH3GridInferenceResult | None = None,
        top_down_weight: float | None = None,
        scenario: str = "h3",
    ) -> SpatialInferenceTrace:
        """
        Return research diagnostics for one nested H3 grid inference step.

        Nested traces include leaf-cell pymdp diagnostics plus parent aggregate
        cells, same-resolution edge diagnostics, and parent/child consistency.
        """
        if self.generative_model is None:
            raise ValueError("Set generative model first")
        if grid_result is None:
            grid_result = self.infer_over_nested_h3_grid(
                h3_grid,
                return_result=True,
                top_down_weight=top_down_weight,
            )
        nested_update = grid_result.nested_belief_update
        hierarchy = getattr(self.generative_model, "nested_h3_hierarchy", None)
        return self.generative_model.compute_h3_cell_diagnostics(
            grid_result.cell_results,
            timestep=timestep,
            scenario=scenario,
            previous_beliefs=previous_beliefs,
            hierarchy=hierarchy,
            parent_beliefs=nested_update.parent_beliefs,
            backend_metadata=grid_result.metadata,
            metadata={
                "aggregate_free_energy": grid_result.aggregate_free_energy,
                "spatial_consistency": grid_result.spatial_consistency,
                "nested_h3": True,
            },
        )

    def _compute_h3_grid_consistency(
        self, cell_beliefs: dict[str, np.ndarray | None]
    ) -> H3SpatialConsistency:
        """Compute spatial consistency for an H3 grid inference result."""
        valid_beliefs = {
            cell: normalize_belief_vector(belief)
            for cell, belief in cell_beliefs.items()
            if belief is not None
        }
        if not valid_beliefs:
            return H3SpatialConsistency(
                global_coherence=0.0,
                neighbor_correlations=0.0,
                cell_count=0,
                edge_count=0,
            )

        cells = set(valid_beliefs)
        graph = {}
        spatial_graph = getattr(self.generative_model, "spatial_graph", None)
        if self.generative_model is not None and isinstance(spatial_graph, dict):
            graph = {
                cell: {neighbor for neighbor in neighbors if neighbor in cells}
                for cell, neighbors in spatial_graph.items()
                if cell in cells
            }
        else:
            adapter = get_h3_adapter()
            graph = {
                cell: {
                    neighbor
                    for neighbor in adapter.grid_ring(cell, 1)
                    if neighbor in cells
                }
                for cell in cells
            }

        belief_matrix = np.vstack(list(valid_beliefs.values()))
        global_coherence = float(
            np.clip(1.0 - np.mean(np.std(belief_matrix, axis=0)), 0.0, 1.0)
        )
        correlations = []
        for cell, neighbors in graph.items():
            source = valid_beliefs[cell]
            for neighbor in neighbors:
                target = valid_beliefs[neighbor]
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
            cell_count=len(valid_beliefs),
            edge_count=edge_count_from_graph(cast("dict[str, Iterable[str]]", graph)),
        )

    def set_preferences(self, preferences: np.ndarray | dict[str, Any]) -> None:
        """Override prior preferences used during inference."""
        self.preferences = copy.deepcopy(preferences)
        if self.generative_model is None:
            return
        model_preferences = getattr(self.generative_model, "preferences", None)
        if isinstance(model_preferences, dict) and isinstance(preferences, dict):
            self.generative_model.set_preferences(preferences)
        else:
            self.generative_model.preferences = cast(Any, copy.deepcopy(preferences))

    def _extract_model_beliefs(self, model: GenerativeModel) -> Any:
        beliefs = getattr(model, "beliefs", None)
        if beliefs is None:
            s_dim = getattr(model, "state_dim", None)
            if s_dim is None:
                s_dim = getattr(model, "num_states", 0)

            if self.model_type == "categorical" and (
                isinstance(s_dim, int)
                and s_dim > 0
                or isinstance(s_dim, list)
                and len(s_dim) > 0
            ):
                if isinstance(s_dim, list):
                    return {
                        "states": [normalize_distribution(np.ones(d)) for d in s_dim]
                    }
                return {"states": normalize_distribution(np.ones(s_dim))}
            if self.model_type == "gaussian" and (
                isinstance(s_dim, int)
                and s_dim > 0
                or isinstance(s_dim, list)
                and len(s_dim) > 0
            ):
                if isinstance(s_dim, list):
                    s_dim = s_dim[0]  # Simplify for gaussian
                precision = np.eye(s_dim) * getattr(model, "prior_precision", 1.0)
                return {"mean": np.zeros(s_dim), "precision": precision}
            return None

        if isinstance(beliefs, dict):
            if self.model_type == "categorical":
                if "states" in beliefs:
                    # Return list/array structure as is, just wrapped in dict if not already
                    return {"states": beliefs["states"]}
                for value in beliefs.values():
                    if isinstance(value, dict) and "states" in value:
                        return {"states": value["states"]}
            if (
                self.model_type == "gaussian"
                and "mean" in beliefs
                and "precision" in beliefs
            ):
                return {
                    "mean": np.asarray(beliefs["mean"], dtype=float).copy(),
                    "precision": np.asarray(beliefs["precision"], dtype=float).copy(),
                }

        if isinstance(beliefs, (np.ndarray, list)):
            return {"states": beliefs}

        return None

    def _extract_model_preferences(self, model: GenerativeModel) -> Any:
        prefs = getattr(model, "preferences", None)
        if prefs is None:
            return None
        if isinstance(prefs, dict):
            if self.model_type == "categorical":
                extracted: dict[str, Any] = {}
                if "states" in prefs:
                    states_flat = self._safe_flatten(prefs["states"])
                    if states_flat is not None:
                        extracted["states"] = normalize_distribution(states_flat)
                if "observations" in prefs:
                    observations_flat = self._safe_flatten(prefs["observations"])
                    if observations_flat is not None:
                        extracted["observations"] = normalize_distribution(
                            observations_flat
                        )
                return extracted or None
            if self.model_type == "gaussian":
                result: dict[str, Any] = {}
                if "mean" in prefs:
                    result["mean"] = np.asarray(prefs["mean"], dtype=float).copy()
                if "precision" in prefs:
                    result["precision"] = np.asarray(
                        prefs["precision"], dtype=float
                    ).copy()
                return result or None
        if isinstance(prefs, (np.ndarray, list)):
            # Safe flatten handles list or object array
            flat = self._safe_flatten(prefs)
            if flat is None:
                return None
            return normalize_distribution(flat)
        return None

    def _update_beliefs_with_model(self, observation: np.ndarray) -> Any:
        if self.generative_model is None:
            raise ValueError("Generative model must be set before perception")

        try:
            if self.model_type == "categorical" and self._supports_pymdp_adapter():
                pymdp_result = run_model_step(
                    self.generative_model,
                    observation,
                    random_seed=int(self.random_seed or 0) + len(self.history),
                    prior=self.current_beliefs,
                )
                self.latest_pymdp_result = pymdp_result
                self.generative_model.beliefs["states"] = pymdp_result.beliefs.copy()
                return {"states": pymdp_result.beliefs.copy()}
            elif self.model_type == "categorical":
                updated: Any = self.generative_model.update_beliefs(
                    {"observations": observation}
                )
                if isinstance(updated, dict) and any(
                    k.startswith("level_") for k in updated.keys()
                ):
                    return updated
                if isinstance(updated, dict) and "states" in updated:
                    states_flat = self._safe_flatten(updated["states"])
                    if states_flat is not None:
                        vec = normalize_distribution(states_flat)
                        return {"states": vec}
                if isinstance(updated, np.ndarray):
                    vec = normalize_distribution(updated.astype(float))
                    return {"states": vec}
            elif self.model_type == "gaussian":
                updated = self.generative_model.update_beliefs(
                    {"observations": observation}
                )
                if (
                    isinstance(updated, dict)
                    and "mean" in updated
                    and "precision" in updated
                ):
                    return {
                        "mean": np.asarray(updated["mean"], dtype=float).copy(),
                        "precision": np.asarray(
                            updated["precision"], dtype=float
                        ).copy(),
                    }
        except Exception as exc:  # pragma: no cover - defensive path
            unsupported_factorized = (
                self.model_type == "categorical" and not self._supports_pymdp_adapter()
            )
            if not unsupported_factorized and not self.parameters.get(
                "allow_local_pymdp_fallback", False
            ):
                raise
            logger.debug("Falling back to local belief update: %s", exc)

        return self._update_beliefs_direct(observation)

    def _supports_pymdp_adapter(self) -> bool:
        """Return true for the simple categorical matrix contract used by H3."""
        model = self.generative_model
        if model is None or getattr(model, "model_type", None) != "categorical":
            return False
        observation_model = getattr(model, "observation_model", None)
        transition_model = getattr(model, "transition_model", None)
        try:
            obs_array = np.asarray(observation_model, dtype=float)
            trans_array = np.asarray(transition_model, dtype=float)
        except (TypeError, ValueError):
            return False
        return obs_array.ndim == 2 and trans_array.ndim in {2, 3}

    def _update_beliefs_direct(self, observation: np.ndarray) -> Any:
        if self.generative_model is None:
            raise ValueError("Generative model must be set before perception")
        generative_model = self.generative_model
        if self.current_beliefs is None:
            self.current_beliefs = self._extract_model_beliefs(generative_model)

        # PYMDP Integration Check
        if self.model_type == "categorical":
            A = getattr(self.generative_model, "observation_model", None)

            # Local Bayes update for simple categorical matrices.
            prior_vec = self._extract_belief_vector(
                self.current_beliefs
            )  # Will extract from dict or array
            if isinstance(A, np.ndarray) and A.dtype != object:
                if prior_vec is None:
                    return None
                if A.shape[1] != len(prior_vec):
                    raise ValueError(
                        "Observation model state dimension does not match beliefs: "
                        f"A.shape={A.shape}, beliefs={len(prior_vec)}"
                    )
                updated_beliefs = self.belief_updater.update_categorical(
                    prior_vec, observation, self.generative_model.observation_model
                )
                self.current_beliefs = {"states": updated_beliefs}
                return self.current_beliefs

        if self.model_type == "gaussian":
            gaussian_beliefs = self._ensure_gaussian_beliefs(self.current_beliefs)
            if gaussian_beliefs is None or not isinstance(
                self.generative_model.observation_model, dict
            ):
                return self.current_beliefs
            observation_model = self.generative_model.observation_model
            return self.belief_updater.update_gaussian(
                gaussian_beliefs["mean"],
                gaussian_beliefs["precision"],
                observation,
                observation_model.get("C", np.eye(len(gaussian_beliefs["mean"]))),
                np.linalg.inv(observation_model.get("R", np.eye(len(observation)))),
            )

        return self.current_beliefs

    def _safe_flatten(self, data: Any) -> np.ndarray | None:
        """Safely flatten data that might be a list of arrays or object array."""
        if data is None:
            return None

        if isinstance(data, dict):
            try:
                # Recursively flatten values
                arrays = [
                    arr
                    for arr in (self._safe_flatten(v) for v in data.values())
                    if arr is not None
                ]
                return cast(np.ndarray, np.concatenate(arrays))
            except Exception:
                logger.debug(
                    "Dict flattening failed, falling through to list/array handler"
                )

        if isinstance(data, (list, tuple)):
            try:
                arrays = [np.asarray(x, dtype=float).reshape(-1) for x in data]
                return cast(np.ndarray, np.concatenate(arrays))
            except Exception:
                return np.asarray(data, dtype=float).reshape(-1)

        if isinstance(data, np.ndarray):
            if data.dtype == object:
                try:
                    arrays = [np.asarray(x, dtype=float).reshape(-1) for x in data.flat]
                    return cast(np.ndarray, np.concatenate(arrays))
                except Exception:
                    logger.debug(
                        "Object array element conversion failed, treating as flat float array"
                    )
            return np.asarray(data, dtype=float).reshape(-1)

        return np.array([float(data)])

    def _extract_belief_vector(self, beliefs: Any) -> np.ndarray | None:
        if beliefs is None:
            return None
        if isinstance(beliefs, dict) and "states" in beliefs:
            states_flat = self._safe_flatten(beliefs["states"])
            if states_flat is None:
                return None
            return normalize_distribution(states_flat)
        flat = self._safe_flatten(beliefs)
        if flat is None:
            return None
        return normalize_distribution(flat)

    def _ensure_gaussian_beliefs(self, beliefs: Any) -> dict[str, np.ndarray] | None:
        if isinstance(beliefs, dict) and "mean" in beliefs and "precision" in beliefs:
            return {
                "mean": np.asarray(beliefs["mean"], dtype=float),
                "precision": np.asarray(beliefs["precision"], dtype=float),
            }
        if self.generative_model is not None:
            gm_beliefs = getattr(self.generative_model, "beliefs", None)
            if (
                isinstance(gm_beliefs, dict)
                and "mean" in gm_beliefs
                and "precision" in gm_beliefs
            ):
                return {
                    "mean": np.asarray(gm_beliefs["mean"], dtype=float),
                    "precision": np.asarray(gm_beliefs["precision"], dtype=float),
                }
        return None

    def _get_preferences_vector(self, length: int | None = None) -> np.ndarray:
        vector = None

        # 1. Try to get state preferences directly
        if (
            self.preferences is not None
            and isinstance(self.preferences, dict)
            and "states" in self.preferences
        ):
            vector = self._safe_flatten(self.preferences["states"])

        if vector is None and self.generative_model is not None:
            model_prefs = self._extract_model_preferences(self.generative_model)
            if isinstance(model_prefs, dict) and "states" in model_prefs:
                vector = self._safe_flatten(model_prefs["states"])
            elif isinstance(model_prefs, np.ndarray):
                vector = self._safe_flatten(model_prefs)

        # 2. If no state preferences, try to map observation preferences to states
        if vector is None:
            obs_prefs = None
            if (
                self.preferences
                and isinstance(self.preferences, dict)
                and "observations" in self.preferences
            ):
                obs_prefs = self._safe_flatten(self.preferences["observations"])
            elif self.generative_model is not None:
                model_prefs = self._extract_model_preferences(self.generative_model)
                if isinstance(model_prefs, dict) and "observations" in model_prefs:
                    obs_prefs = self._safe_flatten(model_prefs["observations"])

            if obs_prefs is not None:
                # Use A matrix to map preferences: P(s) propto A.T @ P(o)
                observation_model = getattr(
                    self.generative_model, "observation_model", None
                )
                # Also support 'A' attribute directly if observation_model isn't set
                if observation_model is None:
                    observation_model = getattr(self.generative_model, "A", None)

                if isinstance(observation_model, np.ndarray):
                    # Verify dimensions
                    # A is (obs_dim, state_dim)
                    # obs_prefs is (obs_dim)
                    if observation_model.shape[0] == len(obs_prefs):
                        vector = observation_model.T @ obs_prefs

        # 3. Use a uniform preference vector when no preferences are configured.
        if vector is None:
            belief_vector = self._extract_belief_vector(self.current_beliefs)
            default_length = (
                length
                if length is not None
                else (len(belief_vector) if belief_vector is not None else 1)
            )
            vector = np.ones(default_length) / max(default_length, 1)

        if length is not None:
            vector = self._align_vector(vector, length)

        return normalize_distribution(vector)

    def _get_gaussian_preferences(self) -> dict[str, np.ndarray]:
        prefs: dict[str, np.ndarray] = {}
        if isinstance(self.preferences, dict):
            if "mean" in self.preferences:
                prefs["mean"] = np.asarray(self.preferences["mean"], dtype=float)
            if "precision" in self.preferences:
                prefs["precision"] = np.asarray(
                    self.preferences["precision"], dtype=float
                )
        if not prefs and self.generative_model is not None:
            gm_prefs = getattr(self.generative_model, "preferences", {})
            if isinstance(gm_prefs, dict):
                if "mean" in gm_prefs:
                    prefs["mean"] = np.asarray(gm_prefs["mean"], dtype=float)
                if "precision" in gm_prefs:
                    prefs["precision"] = np.asarray(gm_prefs["precision"], dtype=float)
        return prefs

    def _get_observation_vector(self, length: int) -> np.ndarray:
        if isinstance(self.current_observations, np.ndarray):
            obs = self.current_observations
        elif self.generative_model is not None:
            obs_dim = getattr(self.generative_model, "obs_dim", length)
            obs = np.ones(obs_dim) / max(obs_dim, 1)
        else:
            obs = np.ones(length) / max(length, 1)

        obs = np.asarray(obs, dtype=float).reshape(-1)
        if len(obs) != length:
            obs = self._align_vector(obs, length)
        return normalize_distribution(obs)

    def _align_vector(self, vector: np.ndarray, length: int) -> np.ndarray:
        # Ensure input is at least 1D
        vector = np.atleast_1d(vector)

        if len(vector) == length:
            return vector
        if len(vector) < length:
            padding = np.zeros(length - len(vector))
            return np.concatenate([vector, padding])
        return vector[:length]

    def _clone_beliefs(self, beliefs: Any) -> Any:
        if isinstance(beliefs, np.ndarray):
            return beliefs.copy()
        if isinstance(beliefs, dict):
            return {key: self._clone_beliefs(value) for key, value in beliefs.items()}
        return copy.deepcopy(beliefs)
