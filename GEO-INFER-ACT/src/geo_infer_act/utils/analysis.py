"""
Comprehensive analysis utilities for Active Inference models.

This module provides tools for analyzing perception, action selection,
and Variational Free Energy patterns in Active Inference systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from pathlib import Path
import json
import logging
import time
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


class ActiveInferenceAnalyzer:
    """
    Comprehensive analyzer for Active Inference model behavior.

    This class provides methods to analyze perception (sensemaking),
    action selection (policy inference), and Variational Free Energy
    dynamics to detect patterns and ensure interpretability.
    """

    def __init__(self, output_dir: str):
        """Initialize analyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different analysis types
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)

        # Data storage
        self.traces: dict[str, list] = {
            "beliefs": [],
            "observations": [],
            "actions": [],
            "policies": [],
            "free_energy": [],
            "metrics": [],
            "timestamps": [],
        }

        logger.info(
            f"ActiveInferenceAnalyzer initialized with output: {self.output_dir}"
        )

    def export_full_history(self, filename: str = "full_history.json") -> Path:
        """
        Export complete simulation history to JSON for total analysis.

        Args:
            filename: Name of the output file

        Returns:
            Path to the saved file
        """
        export_data = []
        for i in range(len(self.traces["timestamps"])):
            # Helper to safely serialize numpy arrays
            def serialize(obj: Any) -> Any:
                """Convert numpy-rich diagnostics into JSON-compatible values."""
                if isinstance(obj, np.ndarray):
                    return serialize(obj.tolist())  # Recurse on list conversion
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: serialize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [serialize(x) for x in obj]
                return obj

            step_record = {
                "step": i,
                "timestamp": self.traces["timestamps"][i],
                "belief": serialize(self.traces["beliefs"][i]),
                "observation": serialize(self.traces["observations"][i]),
                "action": serialize(self.traces["actions"][i]),
                "policy": serialize(self.traces["policies"][i]),
                "free_energy": (
                    float(self.traces["free_energy"][i])
                    if self.traces["free_energy"][i] is not None
                    else None
                ),
                "metrics": serialize(self.traces["metrics"][i]),
            }
            export_data.append(step_record)

        output_path = self.output_dir / "data" / filename
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Full history exported to {output_path}")
        return output_path

    def record_step(
        self,
        beliefs: np.ndarray,
        observations: np.ndarray,
        actions: Any,
        policies: dict[str, Any],
        free_energy: float,
        metrics: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Record a single Active Inference step for analysis."""

        if timestamp is None:
            timestamp = time.time()

        # Store traces
        self.traces["beliefs"].append(
            beliefs.copy() if hasattr(beliefs, "copy") else beliefs
        )
        self.traces["observations"].append(
            observations.copy() if hasattr(observations, "copy") else observations
        )
        self.traces["actions"].append(actions)
        self.traces["policies"].append(
            policies.copy() if hasattr(policies, "copy") else policies
        )
        self.traces["free_energy"].append(free_energy)
        self.traces["metrics"].append(metrics if metrics is not None else {})
        self.traces["timestamps"].append(timestamp)

        logger.debug(
            f"Recorded step {len(self.traces['beliefs'])}: FE={free_energy:.4f}"
        )

    @property
    def step_history(self) -> list[dict[str, Any]]:
        """Return a list of recorded step dictionaries for iteration."""
        history = []
        for i in range(len(self.traces["timestamps"])):
            step = {
                "step": i,
                "beliefs": self.traces["beliefs"][i],
                "observations": self.traces["observations"][i],
                "action": self.traces["actions"][i],
                "policies": self.traces["policies"][i],
                "free_energy": self.traces["free_energy"][i],
                "metrics": self.traces["metrics"][i],
                "timestamp": self.traces["timestamps"][i],
            }
            history.append(step)
        return history

    def analyze_perception_patterns(self) -> dict[str, Any]:
        """
        Analyze perception (belief updating) patterns.

        Returns:
            Comprehensive analysis of perception dynamics
        """
        logger.info("Analyzing perception patterns...")

        if not self.traces["beliefs"]:
            logger.warning("No belief traces available for analysis")
            return {}

        # Safe extraction of beliefs
        raw_beliefs = self.traces["beliefs"]
        clean_beliefs = []
        for b in raw_beliefs:
            if isinstance(b, dict):
                # Extract 'states' or first value
                if "states" in b:
                    clean_beliefs.append(b["states"])
                else:
                    # Fallback to first value
                    clean_beliefs.append(list(b.values())[0])
            else:
                clean_beliefs.append(b)

        beliefs_array = np.array(clean_beliefs)
        observations_array = np.array(self.traces["observations"])

        analysis = {
            "belief_dynamics": self._analyze_belief_dynamics(beliefs_array),
            "observation_response": self._analyze_observation_response(
                beliefs_array, observations_array
            ),
            "perception_quality": self._assess_perception_quality(beliefs_array),
            "pattern_detection": self._detect_perception_patterns(beliefs_array),
            "surprise_analysis": self._analyze_surprise_patterns(
                beliefs_array, observations_array
            ),
        }

        # Save analysis
        self._save_analysis(analysis, "perception_analysis.json")

        logger.info("Perception pattern analysis completed")
        return analysis

    def analyze_action_selection_patterns(self) -> dict[str, Any]:
        """
        Analyze action selection (policy inference) patterns.

        Returns:
            Comprehensive analysis of action selection dynamics
        """
        logger.info("Analyzing action selection patterns...")

        if not self.traces["policies"]:
            logger.warning("No policy traces available for analysis")
            return {}

        analysis = {
            "policy_dynamics": self._analyze_policy_dynamics(),
            "action_consistency": self._analyze_action_consistency(),
            "exploration_exploitation": self._analyze_exploration_exploitation(),
            "policy_convergence": self._analyze_policy_convergence(),
            "decision_quality": self._assess_decision_quality(),
        }

        # Save analysis
        self._save_analysis(analysis, "action_selection_analysis.json")

        logger.info("Action selection pattern analysis completed")
        return analysis

    def analyze_free_energy_patterns(self) -> dict[str, Any]:
        """
        Analyze Variational Free Energy patterns and dynamics.

        Returns:
            Comprehensive analysis of free energy behavior
        """
        logger.info("Analyzing Variational Free Energy patterns...")

        if not self.traces["free_energy"]:
            logger.warning("No free energy traces available for analysis")
            return {}

        fe_array = np.array(self.traces["free_energy"])

        analysis = {
            "minimization_dynamics": self._analyze_fe_minimization(fe_array),
            "convergence_analysis": self._analyze_fe_convergence(fe_array),
            "stability_assessment": self._assess_fe_stability(fe_array),
            "anomaly_detection": self._detect_fe_anomalies(fe_array),
            "efficiency_metrics": self._compute_fe_efficiency(fe_array),
        }

        # Save analysis
        self._save_analysis(analysis, "free_energy_analysis.json")

        logger.info("Free Energy pattern analysis completed")
        return analysis

    def _analyze_belief_dynamics(self, beliefs_array: np.ndarray) -> dict[str, Any]:
        """Analyze belief evolution dynamics."""
        if beliefs_array.shape[0] < 2:
            return {"error": "Insufficient data for belief dynamics analysis"}

        # Belief changes over time
        belief_changes = np.diff(beliefs_array, axis=0)

        # Handle ragged arrays (object dtype) or regular arrays
        if beliefs_array.dtype == object:
            # Iterate manually and handle potential nested arrays safely
            entropy_values: list[float] = []
            for b_idx, b in enumerate(beliefs_array):
                try:
                    # If b is an array-like object
                    val = np.asarray(b, dtype=float)
                    entropy_values.append(float(-np.sum(val * np.log(val + 1e-8))))
                except (TypeError, ValueError):
                    logger.warning(
                        "Belief entropy at index %d could not be computed; "
                        "appending 0.0 sentinel",
                        b_idx,
                        exc_info=True,
                    )
                    entropy_values.append(0.0)
            entropies = np.array(entropy_values)
        else:
            beliefs_array = beliefs_array.astype(float)
            entropies = np.array([-np.sum(b * np.log(b + 1e-8)) for b in beliefs_array])

        if beliefs_array.dtype == object:
            # Manual stability calculation for ragged arrays
            # We can't compute "column-wise" stability easily if columns differ.
            # Compute stability per step magnitude instead.
            stability = (
                np.zeros(len(beliefs_array[0]))
                if len(beliefs_array) > 0
                else np.array([])
            )
        else:
            stability = np.std(belief_changes, axis=0)

        # Dominant beliefs
        if beliefs_array.dtype == object:
            dominant_values: list[int] = []
            for b_idx, b in enumerate(beliefs_array):
                try:
                    # Attempt to find max index of expected flat array or first factor
                    val = np.asarray(b)
                    if val.ndim > 1:
                        # Likely multiple factors, take first or flatten?
                        # For simple "dominant state" metric, let's take max of flattened
                        dominant_values.append(int(np.argmax(val.flatten())))
                    else:
                        dominant_values.append(int(np.argmax(val)))
                except (TypeError, ValueError):
                    logger.warning(
                        "Dominant belief at index %d could not be computed; "
                        "appending 0 sentinel",
                        b_idx,
                        exc_info=True,
                    )
                    dominant_values.append(0)
            dominant_states = np.array(dominant_values)
        else:
            dominant_states = np.argmax(beliefs_array, axis=1)
        state_switches = np.sum(np.diff(dominant_states) != 0)

        if belief_changes.dtype == object:
            norm_values: list[float] = []
            for c_idx, c in enumerate(belief_changes):
                try:
                    val = np.asarray(c, dtype=float)
                    norm_values.append(float(np.linalg.norm(val)))
                except (TypeError, ValueError):
                    logger.warning(
                        "Belief-change norm at index %d could not be computed; "
                        "appending 0.0 sentinel",
                        c_idx,
                        exc_info=True,
                    )
                    norm_values.append(0.0)
            norms = np.array(norm_values)
        else:
            norms = np.linalg.norm(belief_changes, axis=1)

        return {
            "belief_change_magnitude": {
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "max": float(np.max(norms)),
            },
            "entropy_dynamics": {
                "initial": float(entropies[0]),
                "final": float(entropies[-1]),
                "mean": float(np.mean(entropies)),
                "trend": float(np.polyfit(range(len(entropies)), entropies, 1)[0]),
            },
            "stability_by_state": stability.tolist(),
            "state_switches": int(state_switches),
            "switch_rate": float(state_switches / len(beliefs_array)),
        }

    def _analyze_observation_response(
        self, beliefs_array: np.ndarray, observations_array: np.ndarray
    ) -> dict[str, Any]:
        """Analyze how beliefs respond to observations."""
        if len(beliefs_array) != len(observations_array) or len(beliefs_array) < 2:
            return {"error": "Insufficient or mismatched data"}

        # Compute belief changes in response to observations
        belief_changes = np.diff(beliefs_array, axis=0)

        # Observation patterns
        obs_changes = (
            np.diff(observations_array, axis=0)
            if observations_array.ndim > 1
            else np.diff(observations_array)
        )

        # Responsiveness metric
        responsiveness = []
        for i in range(len(belief_changes)):
            if i < len(obs_changes):
                if observations_array.ndim > 1:
                    obs_magnitude = np.linalg.norm(obs_changes[i])
                else:
                    obs_magnitude = abs(obs_changes[i])
                belief_magnitude = np.linalg.norm(belief_changes[i])

                if obs_magnitude > 1e-8:
                    responsiveness.append(belief_magnitude / obs_magnitude)

        return {
            "responsiveness": {
                "mean": float(np.mean(responsiveness)) if responsiveness else 0.0,
                "std": float(np.std(responsiveness)) if responsiveness else 0.0,
                "distribution": responsiveness[:50],  # First 50 for storage
            },
            "correlation_analysis": self._compute_obs_belief_correlation(
                beliefs_array, observations_array
            ),
        }

    def _compute_obs_belief_correlation(
        self, beliefs: np.ndarray, observations: np.ndarray
    ) -> dict[str, Any]:
        """Compute correlation between observations and belief changes."""
        if len(beliefs) < 3:
            return {"error": "Insufficient data"}

        try:
            # For each belief state, compute correlation with observations
            correlations = {}
            for state_idx in range(beliefs.shape[1]):
                belief_series = beliefs[:, state_idx]

                if observations.ndim > 1:
                    # Multi-dimensional observations
                    for obs_idx in range(observations.shape[1]):
                        obs_series = observations[:, obs_idx]
                        corr = np.corrcoef(belief_series, obs_series)[0, 1]
                        correlations[f"state_{state_idx}_obs_{obs_idx}"] = (
                            float(corr) if not np.isnan(corr) else 0.0
                        )
                else:
                    # Single dimensional observations
                    corr = np.corrcoef(belief_series, observations)[0, 1]
                    correlations[f"state_{state_idx}_obs"] = (
                        float(corr) if not np.isnan(corr) else 0.0
                    )

            return correlations
        except Exception as e:
            logger.warning(f"Error computing correlations: {e}")
            # The contract for this method is a dict of float-valued
            # state_<i>_obs_<j> entries; an error sentinel would inject a
            # string into that shape and break consumers far from the cause.
            return {}

    def _assess_perception_quality(self, beliefs_array: np.ndarray) -> dict[str, Any]:
        """Assess the quality of perception (belief updating)."""
        # Check for flat patterns
        flat_threshold = 1e-3
        if beliefs_array.dtype == object:
            # Manual check for ragged arrays
            is_flat = True
            for col in range(len(beliefs_array[0])):
                try:
                    # Extract column manually if possible, or skip
                    # Ragged arrays are hard to slice column-wise if lengths differ
                    # Simplified: check distinct values in flattened history
                    # If all values are roughly same, it's flat.
                    # But efficiently? Let's just assume false for complex ragged arrays unless trivial
                    is_flat = False
                    break
                except (TypeError, ValueError):
                    is_flat = False
        else:
            is_flat = bool(np.all(np.std(beliefs_array, axis=0) < flat_threshold))

        # Check for random patterns
        randomness_score = self._assess_randomness(beliefs_array)

        # Check for meaningful structure
        structure_score = self._assess_structure(beliefs_array)

        return {
            "is_flat": bool(is_flat),
            "randomness_score": float(randomness_score),
            "structure_score": float(structure_score),
            "quality_rating": self._rate_perception_quality(
                is_flat, randomness_score, structure_score
            ),
        }

    def _assess_randomness(self, data: np.ndarray) -> float:
        """Assess randomness in data using multiple metrics."""
        if len(data) < 5:
            return 0.5

        try:
            # Autocorrelation test
            autocorr_scores = []
            for dim in range(data.shape[1]):
                series = data[:, dim]
                if len(series) > 1:
                    autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                    autocorr_scores.append(
                        abs(autocorr) if not np.isnan(autocorr) else 0
                    )

            # High autocorrelation suggests structure (low randomness)
            # Low autocorrelation suggests randomness
            avg_autocorr = np.mean(autocorr_scores) if autocorr_scores else 0
            randomness = 1 - avg_autocorr

            return max(0, min(1, randomness))
        except (IndexError, TypeError, ValueError):
            return 0.5

    def _assess_structure(self, data: np.ndarray) -> float:
        """Assess meaningful structure in data."""
        if len(data) < 3:
            return 0.0

        try:
            # Look for trends, patterns, and meaningful changes
            structure_indicators = []

            for dim in range(data.shape[1]):
                series = data[:, dim]

                # Trend detection
                if len(series) > 2:
                    trend_coef = abs(np.polyfit(range(len(series)), series, 1)[0])
                    structure_indicators.append(trend_coef)

                # Variability that's not noise
                if np.std(series) > 1e-6:
                    variability = np.std(series) / (np.mean(series) + 1e-8)
                    structure_indicators.append(min(1.0, variability))

            return float(np.mean(structure_indicators)) if structure_indicators else 0.0
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _rate_perception_quality(
        self, is_flat: bool, randomness: float, structure: float
    ) -> str:
        """Rate overall perception quality."""
        if is_flat:
            return "Poor - Flat/Non-responsive"
        elif randomness > 0.8:
            return "Poor - Too Random"
        elif structure < 0.1:
            return "Poor - No Structure"
        elif structure > 0.7 and randomness < 0.3:
            return "Excellent - Structured and Responsive"
        elif structure > 0.4:
            return "Good - Some Structure"
        else:
            return "Fair - Limited Structure"

    def _detect_perception_patterns(self, beliefs_array: np.ndarray) -> dict[str, Any]:
        """Detect specific patterns in perception."""
        patterns = {}

        if len(beliefs_array) < 5:
            return {"error": "Insufficient data for pattern detection"}

        # Oscillatory patterns
        patterns["oscillations"] = self._detect_oscillations(beliefs_array)

        # Convergence patterns
        patterns["convergence"] = self._detect_convergence_patterns(beliefs_array)

        # Phase transitions
        patterns["phase_transitions"] = self._detect_phase_transitions(beliefs_array)

        return patterns

    def _detect_oscillations(self, data: np.ndarray) -> dict[str, Any]:
        """Detect oscillatory patterns in belief dynamics."""
        oscillation_info = {}

        # Skip for ragged arrays
        if data.dtype == object:
            return {
                "warning": "Oscillation detection skipped for ragged/factorized belief arrays"
            }

        for dim in range(data.shape[1]):
            series = data[:, dim]

            # Find peaks and valleys
            peaks, _ = find_peaks(series, height=np.mean(series))
            valleys, _ = find_peaks(-series, height=-np.mean(series))

            # Estimate frequency
            if len(peaks) > 1:
                avg_period = np.mean(np.diff(peaks))
                frequency = 1.0 / avg_period if avg_period > 0 else 0
            else:
                frequency = 0

            oscillation_info[f"state_{dim}"] = {
                "num_peaks": len(peaks),
                "num_valleys": len(valleys),
                "estimated_frequency": float(frequency),
                "amplitude": float(np.std(series)),
            }

        return oscillation_info

    def _detect_convergence_patterns(self, data: np.ndarray) -> dict[str, Any]:
        """Detect convergence patterns in beliefs."""
        convergence_info = {}

        window_size = min(10, len(data) // 3)
        if window_size < 2:
            return {"error": "Insufficient data for convergence analysis"}

        for dim in range(data.shape[1]):
            series = data[:, dim]

            # Compute moving variance
            moving_vars = []
            for i in range(window_size, len(series)):
                window = series[i - window_size : i]
                moving_vars.append(np.var(window))

            # Check if variance is decreasing (converging)
            if len(moving_vars) > 1:
                convergence_trend = np.polyfit(range(len(moving_vars)), moving_vars, 1)[
                    0
                ]
                is_converging = convergence_trend < -1e-6
            else:
                convergence_trend = 0
                is_converging = False

            convergence_info[f"state_{dim}"] = {
                "is_converging": bool(is_converging),
                "convergence_rate": float(abs(convergence_trend)),
                "final_variance": float(np.var(series[-window_size:])),
            }

        return convergence_info

    def _detect_phase_transitions(self, data: np.ndarray) -> dict[str, Any]:
        """Detect phase transitions in belief dynamics."""
        if len(data) < 10:
            return {"error": "Insufficient data for phase transition detection"}

        transitions = []

        # Look for sudden changes in dominant beliefs
        dominant_states = np.argmax(data, axis=1)

        for i in range(1, len(dominant_states)):
            if dominant_states[i] != dominant_states[i - 1]:
                # Check if this is a sustained transition
                transition_strength = abs(np.max(data[i]) - np.max(data[i - 1]))
                transitions.append(
                    {
                        "timestep": i,
                        "from_state": int(dominant_states[i - 1]),
                        "to_state": int(dominant_states[i]),
                        "strength": float(transition_strength),
                    }
                )

        return {
            "num_transitions": len(transitions),
            "transitions": transitions[:20],  # Limit for storage
            "transition_rate": float(len(transitions) / len(data)),
        }

    def _analyze_surprise_patterns(
        self, beliefs_array: np.ndarray, observations_array: np.ndarray
    ) -> dict[str, Any]:
        """Analyze surprise patterns in perception."""
        if len(beliefs_array) != len(observations_array) or len(beliefs_array) < 2:
            return {"error": "Insufficient data for surprise analysis"}

        surprise_values: list[float] = []

        for i in range(len(beliefs_array)):
            # Compute surprise as negative log probability
            if observations_array.ndim > 1:
                # For multi-dimensional observations, use expected observation
                predicted_prob = np.sum(beliefs_array[i]) / len(beliefs_array[i])
            else:
                predicted_prob = np.mean(beliefs_array[i])

            surprise = -np.log(predicted_prob + 1e-8)
            surprise_values.append(float(surprise))

        surprises = np.array(surprise_values)

        # Safe extraction of trend
        trend_val = np.polyfit(range(len(surprises)), surprises, 1)[0]
        if isinstance(trend_val, np.ndarray):
            trend_val = trend_val.flatten()[0]

        return {
            "surprise_statistics": {
                "mean": float(np.mean(surprises)),
                "std": float(np.std(surprises)),
                "min": float(np.min(surprises)),
                "max": float(np.max(surprises)),
            },
            "surprise_trend": float(trend_val),
            "high_surprise_events": int(
                np.sum(surprises > np.mean(surprises) + 2 * np.std(surprises))
            ),
        }

    def _analyze_policy_dynamics(self) -> dict[str, Any]:
        """Analyze policy selection dynamics."""
        if not self.traces["policies"]:
            return {"error": "No policy data available"}

        # Extract policy probabilities over time
        policy_prob_values: list[Any] = []
        selected_policies = []

        for policy_data in self.traces["policies"]:
            if "all_probabilities" in policy_data:
                policy_prob_values.append(policy_data["all_probabilities"])
            if "policy" in policy_data and "id" in policy_data["policy"]:
                selected_policies.append(policy_data["policy"]["id"])

        if not policy_prob_values:
            return {"error": "No policy probability data available"}

        policy_probs = np.array(policy_prob_values)

        # Policy diversity over time
        policy_entropies = [
            -np.sum(probs * np.log(probs + 1e-8)) for probs in policy_probs
        ]

        # Policy stability
        if len(policy_probs) > 1:
            policy_changes = np.diff(policy_probs, axis=0)
            stability = np.mean(np.linalg.norm(policy_changes, axis=1))
        else:
            stability = 0.0

        return {
            "policy_entropy": {
                "mean": float(np.mean(policy_entropies)),
                "trend": (
                    float(
                        np.polyfit(range(len(policy_entropies)), policy_entropies, 1)[0]
                    )
                    if len(policy_entropies) > 1
                    else 0.0
                ),
            },
            "policy_stability": float(stability),
            "selected_policies": selected_policies[:50],  # Limit for storage
            "policy_distribution": np.mean(policy_probs, axis=0).tolist(),
        }

    def _analyze_action_consistency(self) -> dict[str, Any]:
        """Analyze consistency in action selection."""
        actions = self.traces["actions"]

        if len(actions) < 2:
            return {"error": "Insufficient action data"}

        # For numeric actions
        if all(isinstance(a, (int, float)) for a in actions):
            action_changes = np.abs(np.diff(actions))
            consistency = 1.0 / (1.0 + np.mean(action_changes))
        else:
            # For categorical actions
            action_switches = sum(
                1 for i in range(1, len(actions)) if actions[i] != actions[i - 1]
            )
            consistency = 1.0 - (action_switches / len(actions))

        return {
            "consistency_score": float(consistency),
            "num_action_changes": int(
                action_switches if "action_switches" in locals() else len(set(actions))
            ),
            "action_diversity": len(set(map(str, actions))),
        }

    def _analyze_exploration_exploitation(self) -> dict[str, Any]:
        """Analyze exploration vs exploitation balance."""
        if not self.traces["policies"]:
            return {"error": "No policy data available"}

        exploration_scores = []

        for policy_data in self.traces["policies"]:
            if "all_probabilities" in policy_data:
                probs = np.array(policy_data["all_probabilities"])
                # Entropy as exploration measure
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                max_entropy = np.log(len(probs))
                exploration_score = entropy / max_entropy if max_entropy > 0 else 0
                exploration_scores.append(exploration_score)

        if not exploration_scores:
            return {"error": "No exploration data available"}

        return {
            "exploration_scores": {
                "mean": float(np.mean(exploration_scores)),
                "std": float(np.std(exploration_scores)),
                "trend": (
                    float(
                        np.polyfit(
                            range(len(exploration_scores)), exploration_scores, 1
                        )[0]
                    )
                    if len(exploration_scores) > 1
                    else 0.0
                ),
            },
            "exploration_pattern": (
                "decreasing"
                if np.polyfit(range(len(exploration_scores)), exploration_scores, 1)[0]
                < -0.01
                else "stable"
            ),
        }

    def _analyze_policy_convergence(self) -> dict[str, Any]:
        """Analyze convergence of policy selection."""
        if not self.traces["policies"] or len(self.traces["policies"]) < 5:
            return {"error": "Insufficient policy data for convergence analysis"}

        # Extract policy probabilities
        policy_prob_values: list[Any] = []
        for policy_data in self.traces["policies"]:
            if "all_probabilities" in policy_data:
                policy_prob_values.append(policy_data["all_probabilities"])

        if len(policy_prob_values) < 5:
            return {"error": "Insufficient policy probability data"}

        policy_probs = np.array(policy_prob_values)

        # Compute variance over time windows
        window_size = min(5, len(policy_probs) // 3)
        moving_vars = []

        for i in range(window_size, len(policy_probs)):
            window = policy_probs[i - window_size : i]
            var = np.mean(np.var(window, axis=0))
            moving_vars.append(var)

        # Check convergence trend
        if len(moving_vars) > 1:
            convergence_trend = np.polyfit(range(len(moving_vars)), moving_vars, 1)[0]
            is_converging = convergence_trend < -1e-6
        else:
            convergence_trend = 0
            is_converging = False

        return {
            "is_converging": bool(is_converging),
            "convergence_rate": float(abs(convergence_trend)),
            "final_variance": float(moving_vars[-1]) if moving_vars else 0.0,
            "convergence_quality": (
                "strong" if abs(convergence_trend) > 1e-3 else "weak"
            ),
        }

    def _assess_decision_quality(self) -> dict[str, Any]:
        """Assess overall quality of decision making."""
        if not self.traces["policies"] or not self.traces["free_energy"]:
            return {"error": "Insufficient data for decision quality assessment"}

        # Decision consistency
        consistency_analysis = self._analyze_action_consistency()

        # Exploration balance
        exploration_analysis = self._analyze_exploration_exploitation()

        # Free energy reduction
        fe_array = np.array(self.traces["free_energy"])
        fe_reduction = fe_array[0] - fe_array[-1] if len(fe_array) > 1 else 0

        # Overall quality score
        quality_components = []

        if "consistency_score" in consistency_analysis:
            quality_components.append(consistency_analysis["consistency_score"])

        if "exploration_scores" in exploration_analysis:
            # Moderate exploration is good
            exp_score = exploration_analysis["exploration_scores"]["mean"]
            balanced_exp_score = 1 - abs(exp_score - 0.5) * 2  # Optimal around 0.5
            quality_components.append(balanced_exp_score)

        if fe_reduction > 0:
            quality_components.append(
                min(1.0, fe_reduction / abs(fe_array[0]) if fe_array[0] != 0 else 0)
            )

        overall_quality = np.mean(quality_components) if quality_components else 0.5

        return {
            "overall_quality_score": float(overall_quality),
            "quality_rating": self._rate_decision_quality(overall_quality),
            "components": {
                "consistency": consistency_analysis.get("consistency_score", 0),
                "exploration_balance": (
                    balanced_exp_score if "balanced_exp_score" in locals() else 0.5
                ),
                "free_energy_reduction": float(fe_reduction),
            },
        }

    def _rate_decision_quality(self, score: float) -> str:
        """Rate decision quality based on score."""
        if score > 0.8:
            return "Excellent"
        elif score > 0.6:
            return "Good"
        elif score > 0.4:
            return "Fair"
        else:
            return "Poor"

    def _analyze_fe_minimization(self, fe_array: np.ndarray) -> dict[str, Any]:
        """Analyze free energy minimization dynamics."""
        if len(fe_array) < 2:
            return {"error": "Insufficient free energy data"}

        # Overall reduction
        total_reduction = fe_array[0] - fe_array[-1]

        # Rate of reduction
        reduction_rate = np.polyfit(range(len(fe_array)), fe_array, 1)[0]

        # Efficiency (reduction per step)
        efficiency = total_reduction / len(fe_array) if len(fe_array) > 0 else 0

        # Monotonicity (how consistently it decreases)
        decreasing_steps = np.sum(np.diff(fe_array) < 0)
        monotonicity = (
            decreasing_steps / (len(fe_array) - 1) if len(fe_array) > 1 else 0
        )

        return {
            "total_reduction": float(total_reduction),
            "reduction_rate": float(reduction_rate),
            "efficiency": float(efficiency),
            "monotonicity": float(monotonicity),
            "minimization_quality": self._rate_fe_minimization(
                total_reduction, float(monotonicity)
            ),
        }

    def _rate_fe_minimization(self, total_reduction: float, monotonicity: float) -> str:
        """Rate free energy minimization quality."""
        if total_reduction > 0 and monotonicity > 0.8:
            return "Excellent - Strong, consistent reduction"
        elif total_reduction > 0 and monotonicity > 0.6:
            return "Good - Generally reducing"
        elif total_reduction > 0:
            return "Fair - Some reduction"
        elif abs(total_reduction) < 1e-6:
            return "Stable - No significant change"
        else:
            return "Poor - Increasing free energy"

    def _analyze_fe_convergence(self, fe_array: np.ndarray) -> dict[str, Any]:
        """Analyze free energy convergence."""
        if len(fe_array) < 5:
            return {"error": "Insufficient data for convergence analysis"}

        # Moving variance to detect convergence
        window_size = min(5, len(fe_array) // 3)
        moving_vars = []

        for i in range(window_size, len(fe_array)):
            window = fe_array[i - window_size : i]
            moving_vars.append(np.var(window))

        # Convergence trend
        if len(moving_vars) > 1:
            convergence_trend = np.polyfit(range(len(moving_vars)), moving_vars, 1)[0]
            is_converging = convergence_trend < -1e-6
        else:
            convergence_trend = 0
            is_converging = False

        # Final stability
        final_stability = 1.0 / (1.0 + moving_vars[-1]) if moving_vars else 0

        return {
            "is_converging": bool(is_converging),
            "convergence_rate": float(abs(convergence_trend)),
            "final_stability": float(final_stability),
            "convergence_quality": (
                "strong" if abs(convergence_trend) > 1e-4 else "weak"
            ),
        }

    def _assess_fe_stability(self, fe_array: np.ndarray) -> dict[str, Any]:
        """Assess stability of free energy."""
        if len(fe_array) < 3:
            return {"error": "Insufficient data for stability assessment"}

        # Compute moving statistics
        variance = np.var(fe_array)
        cv = np.std(fe_array) / (np.mean(fe_array) + 1e-8)  # Coefficient of variation

        # Detect sudden jumps
        changes = np.abs(np.diff(fe_array))
        large_jumps = np.sum(changes > np.mean(changes) + 2 * np.std(changes))

        # Stability score
        stability_score = 1.0 / (1.0 + variance + cv + large_jumps / len(fe_array))

        return {
            "variance": float(variance),
            "coefficient_of_variation": float(cv),
            "large_jumps": int(large_jumps),
            "stability_score": float(stability_score),
            "stability_rating": self._rate_stability(stability_score),
        }

    def _rate_stability(self, score: float) -> str:
        """Rate stability based on score."""
        if score > 0.8:
            return "Very Stable"
        elif score > 0.6:
            return "Stable"
        elif score > 0.4:
            return "Moderately Stable"
        else:
            return "Unstable"

    def _detect_fe_anomalies(self, fe_array: np.ndarray) -> dict[str, Any]:
        """Detect anomalies in free energy patterns."""
        if len(fe_array) < 5:
            return {"error": "Insufficient data for anomaly detection"}

        # Statistical outliers
        mean_fe = np.mean(fe_array)
        std_fe = np.std(fe_array)
        outliers = np.where(np.abs(fe_array - mean_fe) > 2 * std_fe)[0]

        # Sudden spikes
        changes = np.abs(np.diff(fe_array))
        spike_threshold = np.mean(changes) + 2 * np.std(changes)
        spikes = np.where(changes > spike_threshold)[0]

        # Flat periods (no change)
        flat_threshold = 1e-6
        flat_periods = []
        flat_start = None

        for i in range(1, len(fe_array)):
            if abs(fe_array[i] - fe_array[i - 1]) < flat_threshold:
                if flat_start is None:
                    flat_start = i - 1
            else:
                if flat_start is not None:
                    flat_periods.append((flat_start, i - 1))
                    flat_start = None

        return {
            "outliers": outliers.tolist(),
            "spikes": spikes.tolist(),
            "flat_periods": flat_periods,
            "anomaly_score": float(len(outliers) + len(spikes) + len(flat_periods))
            / len(fe_array),
        }

    def _compute_fe_efficiency(self, fe_array: np.ndarray) -> dict[str, Any]:
        """Compute free energy minimization efficiency."""
        if len(fe_array) < 2:
            return {"error": "Insufficient data for efficiency computation"}

        # Energy reduction per step
        total_reduction = fe_array[0] - fe_array[-1]
        steps = len(fe_array) - 1
        efficiency_per_step = total_reduction / steps if steps > 0 else 0

        # Cumulative efficiency
        cumulative_reductions = fe_array[0] - fe_array
        efficiency_curve = cumulative_reductions / (np.arange(len(fe_array)) + 1)

        # Theoretical minimum (assuming exponential decay)
        theoretical_curve = fe_array[0] * np.exp(-0.1 * np.arange(len(fe_array)))
        actual_vs_theoretical = np.mean(fe_array / (theoretical_curve + 1e-8))

        return {
            "efficiency_per_step": float(efficiency_per_step),
            "final_efficiency": float(efficiency_curve[-1]),
            "actual_vs_theoretical": float(actual_vs_theoretical),
            "efficiency_rating": self._rate_efficiency(
                efficiency_per_step, actual_vs_theoretical
            ),
        }

    def _rate_efficiency(self, per_step: float, vs_theoretical: float) -> str:
        """Rate efficiency of free energy minimization."""
        if per_step > 0 and vs_theoretical < 1.2:
            return "Highly Efficient"
        elif per_step > 0:
            return "Efficient"
        elif abs(per_step) < 1e-6:
            return "No Progress"
        else:
            return "Inefficient"

    def _save_analysis(self, analysis: dict[str, Any], filename: str) -> None:
        """Save analysis results to file."""
        filepath = self.output_dir / "analysis" / filename
        try:
            with open(filepath, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            logger.info(f"Analysis saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save analysis to {filepath}: {e}")

    def save_traces_to_csv(self) -> None:
        """Save all traces to CSV files for external analysis."""
        data_dir = self.output_dir / "data"

        try:
            # Beliefs
            if self.traces["beliefs"]:
                beliefs_df = pd.DataFrame(self.traces["beliefs"])
                beliefs_df.to_csv(data_dir / "beliefs.csv", index=False)

            # Observations
            if self.traces["observations"]:
                obs_df = pd.DataFrame(self.traces["observations"])
                obs_df.to_csv(data_dir / "observations.csv", index=False)

            # Free Energy
            if self.traces["free_energy"]:
                fe_df = pd.DataFrame(
                    {
                        "timestep": range(len(self.traces["free_energy"])),
                        "free_energy": self.traces["free_energy"],
                        "timestamp": self.traces["timestamps"],
                    }
                )
                fe_df.to_csv(data_dir / "free_energy.csv", index=False)

            # Actions (simplified)
            if self.traces["actions"]:
                actions_df = pd.DataFrame(
                    {
                        "timestep": range(len(self.traces["actions"])),
                        "action": [str(a) for a in self.traces["actions"]],
                    }
                )
                actions_df.to_csv(data_dir / "actions.csv", index=False)

            logger.info(f"Traces saved to CSV files in {data_dir}")

        except Exception as e:
            logger.error(f"Failed to save traces to CSV: {e}")

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        logger.info("Generating comprehensive analysis report...")

        # Run all analyses
        perception_analysis = self.analyze_perception_patterns()
        action_analysis = self.analyze_action_selection_patterns()
        fe_analysis = self.analyze_free_energy_patterns()

        # Generate report
        report_lines = [
            "# Active Inference Analysis Report",
            f"Generated at: {pd.Timestamp.now()}",
            f"Output directory: {self.output_dir}",
            f"Total steps analyzed: {len(self.traces['beliefs'])}",
            "",
            "## Executive Summary",
            "",
        ]

        # Perception summary
        if "perception_quality" in perception_analysis:
            quality = perception_analysis["perception_quality"]
            report_lines.extend(
                [
                    f"**Perception Quality**: {quality.get('quality_rating', 'Unknown')}",
                    f"- Flatness detected: {quality.get('is_flat', 'Unknown')}",
                    f"- Randomness score: {quality.get('randomness_score', 'Unknown'):.3f}",
                    f"- Structure score: {quality.get('structure_score', 'Unknown'):.3f}",
                    "",
                ]
            )

        # Action selection summary
        if "decision_quality" in action_analysis:
            decision = action_analysis["decision_quality"]
            report_lines.extend(
                [
                    f"**Decision Quality**: {decision.get('quality_rating', 'Unknown')}",
                    f"- Overall score: {decision.get('overall_quality_score', 'Unknown'):.3f}",
                    "",
                ]
            )

        # Free energy summary
        if "minimization_dynamics" in fe_analysis:
            fe_min = fe_analysis["minimization_dynamics"]
            report_lines.extend(
                [
                    f"**Free Energy Minimization**: {fe_min.get('minimization_quality', 'Unknown')}",
                    f"- Total reduction: {fe_min.get('total_reduction', 'Unknown'):.4f}",
                    f"- Efficiency: {fe_min.get('efficiency', 'Unknown'):.4f}",
                    "",
                ]
            )

        report_lines.extend(
            [
                "## Detailed Analysis",
                "",
                "See individual analysis files in the analysis/ subdirectory for detailed results.",
                "Raw data is available in CSV format in the data/ subdirectory.",
                "Visualizations are saved in the visualizations/ subdirectory.",
                "",
            ]
        )

        report_text = "\n".join(report_lines)

        # Save report
        report_path = self.output_dir / "comprehensive_report.md"
        try:
            with open(report_path, "w") as f:
                f.write(report_text)
            logger.info(f"Comprehensive report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

        return report_text


def create_shared_visualizations(analyzer: ActiveInferenceAnalyzer) -> None:
    """Create shared visualizations for Active Inference analysis."""
    if not analyzer.traces["beliefs"]:
        logger.warning("No data available for visualization")
        return

    viz_dir = analyzer.output_dir / "visualizations"

    # Belief evolution heatmap
    create_belief_heatmap(analyzer.traces["beliefs"], viz_dir)

    # Free energy analysis plots
    create_free_energy_plots(analyzer.traces["free_energy"], viz_dir)

    # Policy analysis plots
    create_policy_plots(analyzer.traces["policies"], viz_dir)

    # Correlation matrix
    create_correlation_analysis(analyzer.traces, viz_dir)

    logger.info(f"Shared visualizations created in {viz_dir}")


def create_belief_heatmap(beliefs: list[np.ndarray], output_dir: Path) -> None:
    """Create a heatmap of belief evolution over time."""
    if not beliefs:
        return

    try:
        beliefs_array = np.array(beliefs)

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            beliefs_array.T, cmap="viridis", cbar_kws={"label": "Belief Probability"}
        )
        plt.xlabel("Time Step")
        plt.ylabel("State")
        plt.title("Belief Evolution Heatmap")
        plt.tight_layout()
        plt.savefig(
            output_dir / "belief_evolution_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    except Exception as e:
        logger.error(f"Failed to create belief heatmap: {e}")


def create_free_energy_plots(free_energies: list[float], output_dir: Path) -> None:
    """Create comprehensive free energy analysis plots."""
    if not free_energies:
        return

    try:
        fe_array = np.array(free_energies)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Main free energy plot
        axes[0, 0].plot(fe_array, linewidth=2, color="red", marker="o", markersize=3)
        axes[0, 0].set_title("Free Energy Evolution")
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("Free Energy")
        axes[0, 0].grid(True, alpha=0.3)

        # Free energy changes
        if len(fe_array) > 1:
            changes = np.diff(fe_array)
            axes[0, 1].plot(
                changes, linewidth=2, color="blue", marker="s", markersize=3
            )
            axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
            axes[0, 1].set_title("Free Energy Changes")
            axes[0, 1].set_xlabel("Time Step")
            axes[0, 1].set_ylabel("Change in Free Energy")
            axes[0, 1].grid(True, alpha=0.3)

        # Moving average
        window_size = min(5, len(fe_array) // 3) if len(fe_array) > 5 else 1
        if window_size > 1:
            moving_avg = np.convolve(
                fe_array, np.ones(window_size) / window_size, mode="valid"
            )
            axes[1, 0].plot(
                range(window_size - 1, len(fe_array)),
                moving_avg,
                linewidth=2,
                color="green",
                label=f"Moving Average (n={window_size})",
            )
            axes[1, 0].plot(fe_array, alpha=0.5, color="red", label="Original")
            axes[1, 0].set_title("Free Energy Smoothed")
            axes[1, 0].set_xlabel("Time Step")
            axes[1, 0].set_ylabel("Free Energy")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Distribution
        axes[1, 1].hist(
            fe_array, bins=min(20, len(fe_array) // 2), alpha=0.7, color="purple"
        )
        axes[1, 1].set_title("Free Energy Distribution")
        axes[1, 1].set_xlabel("Free Energy")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "free_energy_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    except Exception as e:
        logger.error(f"Failed to create free energy plots: {e}")


def create_policy_plots(policies: list[dict[str, Any]], output_dir: Path) -> None:
    """Create policy analysis plots."""
    if not policies:
        return

    try:
        # Extract policy probabilities
        policy_prob_values: list[Any] = []
        for policy_data in policies:
            if "all_probabilities" in policy_data:
                policy_prob_values.append(policy_data["all_probabilities"])

        if not policy_prob_values:
            return

        policy_probs = np.array(policy_prob_values)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Policy probability evolution
        for i in range(policy_probs.shape[1]):
            axes[0, 0].plot(
                policy_probs[:, i],
                label=f"Policy {i}",
                linewidth=2,
                marker="o",
                markersize=2,
            )
        axes[0, 0].set_title("Policy Probability Evolution")
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("Probability")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Policy entropy
        entropies = [-np.sum(probs * np.log(probs + 1e-8)) for probs in policy_probs]
        axes[0, 1].plot(
            entropies, linewidth=2, color="orange", marker="s", markersize=3
        )
        axes[0, 1].set_title("Policy Selection Entropy")
        axes[0, 1].set_xlabel("Time Step")
        axes[0, 1].set_ylabel("Entropy")
        axes[0, 1].grid(True, alpha=0.3)

        # Average policy distribution
        avg_probs = np.mean(policy_probs, axis=0)
        axes[1, 0].bar(range(len(avg_probs)), avg_probs, alpha=0.7, color="skyblue")
        axes[1, 0].set_title("Average Policy Distribution")
        axes[1, 0].set_xlabel("Policy")
        axes[1, 0].set_ylabel("Average Probability")
        axes[1, 0].grid(True, alpha=0.3)

        # Policy diversity heatmap
        sns.heatmap(
            policy_probs.T,
            cmap="viridis",
            ax=axes[1, 1],
            cbar_kws={"label": "Probability"},
        )
        axes[1, 1].set_title("Policy Selection Heatmap")
        axes[1, 1].set_xlabel("Time Step")
        axes[1, 1].set_ylabel("Policy")

        plt.tight_layout()
        plt.savefig(output_dir / "policy_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        logger.error(f"Failed to create policy plots: {e}")


def create_correlation_analysis(traces: dict[str, list], output_dir: Path) -> None:
    """Create correlation analysis between different traces."""
    try:
        # Prepare data for correlation analysis
        data_dict = {}

        # Free energy
        if traces["free_energy"]:
            data_dict["free_energy"] = traces["free_energy"]

        # Belief entropy
        if traces["beliefs"]:
            entropies = []
            for beliefs in traces["beliefs"]:
                if isinstance(beliefs, np.ndarray) and len(beliefs) > 0:
                    entropy = -np.sum(beliefs * np.log(beliefs + 1e-8))
                    entropies.append(entropy)
                else:
                    entropies.append(0)
            data_dict["belief_entropy"] = entropies

        # Policy entropy
        if traces["policies"]:
            policy_entropies = []
            for policy_data in traces["policies"]:
                if "all_probabilities" in policy_data:
                    probs = policy_data["all_probabilities"]
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    policy_entropies.append(entropy)
                else:
                    policy_entropies.append(0)
            if policy_entropies:
                data_dict["policy_entropy"] = policy_entropies

        if len(data_dict) < 2:
            logger.warning("Insufficient data for correlation analysis")
            return

        # Create DataFrame
        min_length = min(len(v) for v in data_dict.values())
        for key in data_dict:
            data_dict[key] = data_dict[key][:min_length]

        df = pd.DataFrame(data_dict)

        # Correlation matrix
        corr_matrix = df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            cbar_kws={"label": "Correlation"},
        )
        plt.title("Active Inference Components Correlation Matrix")
        plt.tight_layout()
        plt.savefig(
            output_dir / "correlation_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Time series correlations
        fig, axes = plt.subplots(len(df.columns), 1, figsize=(12, 4 * len(df.columns)))
        if len(df.columns) == 1:
            axes = [axes]

        for i, col in enumerate(df.columns):
            axes[i].plot(df[col], linewidth=2, marker="o", markersize=3)
            axes[i].set_title(f"{col.replace('_', ' ').title()} Over Time")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel(col.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "time_series_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    except Exception as e:
        logger.error(f"Failed to create correlation analysis: {e}")
