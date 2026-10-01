"""
Active Inference API Interface for GEO-INFER-ACT.

This module provides a high-level interface for creating and managing
active inference models, including belief updating and policy selection.
"""

import numpy as np
from typing import Any
import logging

from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.utils.config import load_config

logger = logging.getLogger(__name__)


class ActiveInferenceInterface:
    """
    High-level interface for active inference models.

    This class provides a simplified API for creating, configuring,
    and running active inference models without requiring detailed
    knowledge of the underlying mathematical machinery.
    """

    def __init__(self, config_path: str | None = None):
        """
        Initialize the active inference interface.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path) if config_path else {}
        self.models: dict[str, Any] = {}

        logger.info("ActiveInferenceInterface initialized")

    def create_model(
        self, model_id: str, model_type: str, parameters: dict[str, Any]
    ) -> None:
        """
        Create a new active inference model.

        Args:
            model_id: Unique identifier for the model
            model_type: Type of model ('categorical', 'gaussian', 'hierarchical_gaussian')
            parameters: Model configuration parameters
        """
        from geo_infer_act.core.active_inference import ActiveInferenceModel

        # Enhanced parameters with more dynamic defaults
        enhanced_params = {
            "learning_rate": 0.1,
            "temporal_precision": 1.0,
            "enable_learning": True,
            "enable_adaptation": True,
            **parameters,
        }

        if model_type == "gaussian":
            if "state_dim" not in enhanced_params:
                if "mean" in enhanced_params:
                    enhanced_params["state_dim"] = len(
                        np.asarray(enhanced_params["mean"]).reshape(-1)
                    )
                elif "cov" in enhanced_params:
                    enhanced_params["state_dim"] = int(
                        np.asarray(enhanced_params["cov"]).shape[0]
                    )
            enhanced_params.setdefault("obs_dim", enhanced_params.get("state_dim", 1))
            if "precision" not in enhanced_params and "cov" in enhanced_params:
                enhanced_params["precision"] = np.linalg.inv(
                    np.asarray(enhanced_params["cov"], dtype=float)
                )

        # Create Core Active Inference Model
        agent = ActiveInferenceModel(model_type=model_type, **enhanced_params)

        # Create Generative Model
        gen_model = GenerativeModel(
            model_type=model_type, parameters=enhanced_params, model_id=model_id
        )

        # Link them
        agent.set_generative_model(gen_model)

        self.models[model_id] = agent
        logger.info(f"Created {model_type} model: {model_id}")

    def update_beliefs(
        self, model_id: str, observations: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Update model beliefs based on observations.

        Args:
            model_id: Model identifier
            observations: Observed data

        Returns:
            Updated beliefs
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        agent = self.models[model_id]
        obs_data = observations.get("observations")

        if obs_data is None:
            numeric_values = [
                float(value)
                for key, value in sorted(observations.items())
                if key != "observations" and isinstance(value, (int, float))
            ]
            if not numeric_values:
                raise ValueError(
                    "observations must include an 'observations' vector or numeric values"
                )
            obs_data = np.asarray(numeric_values, dtype=float)

        updated_beliefs = agent.perceive(obs_data)

        # Return in expected format
        if agent.model_type == "categorical":
            if isinstance(updated_beliefs, dict) and any(
                k.startswith("level_") for k in updated_beliefs.keys()
            ):
                return updated_beliefs
            if isinstance(updated_beliefs, dict) and "states" in updated_beliefs:
                return updated_beliefs
            return {"states": updated_beliefs}

        if isinstance(updated_beliefs, dict):
            return updated_beliefs
        return {"states": updated_beliefs}

    def select_policy(self, model_id: str) -> dict[str, Any]:
        """
        Select optimal policy based on current beliefs.

        Args:
            model_id: Model identifier

        Returns:
            Selected policy information
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        agent = self.models[model_id]

        action = agent.act()
        selection = agent.latest_policy_selection or {}
        evaluation = agent.latest_policy_evaluation

        policy = selection.get("policy")
        probability = float(selection.get("probability", 1.0))
        all_probs = selection.get("all_probabilities", np.array([1.0]))
        all_free_energies = selection.get("all_free_energies", np.array([]))
        expected_free_energy = (
            float(evaluation.expected_free_energy) if evaluation is not None else None
        )
        policy_payload = (
            dict(policy) if isinstance(policy, dict) else {"action": action}
        )
        policy_payload.setdefault("expected_free_energy", expected_free_energy)

        return {
            "policy": policy_payload,
            "selected_action": action,
            "probability": probability,
            "expected_free_energy": expected_free_energy,
            "all_probabilities": (
                all_probs.tolist() if hasattr(all_probs, "tolist") else all_probs
            ),
            "all_free_energies": (
                all_free_energies.tolist()
                if hasattr(all_free_energies, "tolist")
                else all_free_energies
            ),
            "evaluation": evaluation.to_dict() if evaluation is not None else None,
        }

    def set_preferences(self, model_id: str, preferences: dict[str, Any]) -> None:
        """
        Set prior preferences for the model.

        Args:
            model_id: Model identifier
            preferences: Preference specifications
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        agent = self.models[model_id]
        agent.update_preferences(preferences)
        logger.info(f"Set preferences for model {model_id}")

    def get_free_energy(self, model_id: str) -> float:
        """Calculate free energy for the current model state."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        return float(self.models[model_id].compute_free_energy())
