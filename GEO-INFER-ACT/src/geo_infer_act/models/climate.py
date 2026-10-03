"""
Climate Model for Active Inference.

This module implements a concrete Active Inference model for climate adaptation,
defining the specific state space, observations, and dynamics (A, B, C, D matrices).
"""

from typing import Any
import numpy as np
import logging

from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.core.generative_model import GenerativeModel


class _PymdpCompatUtils:
    """Object-array and distribution helpers used by the climate model."""

    @staticmethod
    def obj_array(n: int) -> np.ndarray:
        """Create an object array of size n."""
        return np.empty(n, dtype=object)

    @staticmethod
    def norm_dist(dist: np.ndarray) -> np.ndarray:
        """Normalize a distribution along columns."""
        dist = dist.astype(float)
        col_sums = dist.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        result: np.ndarray = dist / col_sums
        return result


utils = _PymdpCompatUtils()

logger = logging.getLogger(__name__)


class ClimateModel(ActiveInferenceModel):
    """
    Climate adaptation modeling using Active Inference.

    States:
        - Temperature: [Normal, Elevated, High]
        - CO2 Level: [Safe, Warning, Critical]

    Observations:
        - Thermometer: [Normal, Elevated, High]
        - CO2 Sensor: [Safe, Warning, Critical]

    Actions:
        - Mitigation: [Do Nothing, Reduce Emissions, Geo-engineering]
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        random_seed: int | None = None,
    ):
        """
        Initialize the Climate Model.

        Args:
            config: Configuration dictionary
        """
        # Define dimensions
        self.state_factors = ["Temperature", "CO2"]
        self.obs_factors = ["Thermometer", "CO2Sensor"]

        # Dimensions: Temp(3), CO2(3)
        self.num_states = [3, 3]
        # Dimensions: Therm(3), CO2Sens(3)
        self.num_obs = [3, 3]
        # Actions: Mitigation(3) for both factors (or just one control factor affecting both)
        # Actions: Mitigation for Temp, Mitigation for CO2
        # We explicitly model them as separate controls to avoid pymdp indexing errors
        # (Since generic implementation assumes 1-to-1 mapping between state factors and control factors)
        self.num_controls = [3, 3]

        # Initialize generative model matrices
        A = self._build_likelihood_A()
        B = self._build_transition_B()
        C = self._build_preferences_C()
        D = self._build_prior_D()

        # Define parameters for the generic engine
        params: dict[str, Any] = {
            "state_dim": self.num_states,  # This might need adaptation in core if it expects int
            "obs_dim": self.num_obs,  # This might need adaptation
            "action_dim": self.num_controls,
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "model_type": "categorical",
            "random_seed": random_seed,
        }

        if config:
            params.update(config)

        super().__init__(**params)

        # We need to manually inject the matrices into the generative model
        # because the base class might try to auto-init them with defaults.
        # So we override the generative model.
        self.generative_model = GenerativeModel(
            model_type="categorical", parameters=params
        )
        # Also set the internal active inference components to use these
        self.set_generative_model(self.generative_model)

        logger.info(
            "ClimateModel initialized with %s states and %s observations",
            self.num_states,
            self.num_obs,
        )

    def _build_likelihood_A(self) -> np.ndarray:
        """Build A matrix: P(o|s)."""
        # A[modality][observation, state_factor_combinations...] ??
        # Standard pymdp: A[modality][observation, state_i, state_j, ...]

        # Initialize A
        A = utils.obj_array(len(self.num_obs))

        # 1. Thermometer mapping (Measures Temperature accurately with some noise)
        # State 0 (Temp) -> Obs 0 (Therm)
        # We need to map state indices to sensor indices.

        # For simplicity, let's assume fully factorized mapping implies:
        # matrix for Thermometer depends ONLY on Temperature state?
        # If A is factorized, we usually provide A for each modality.

        # A[0]: Thermometer (3) x Temperature(3) x CO2(3)
        # If we assume independence from CO2, we broadcast.

        A_therm = np.zeros((3, 3, 3))
        # Perfect correlation with Temp (dim 1), independent of CO2 (dim 2)
        for i_temp in range(3):
            for i_co2 in range(3):
                # Peak probability at correct temp
                A_therm[i_temp, i_temp, i_co2] = 0.8
                # Noise
                others = [x for x in range(3) if x != i_temp]
                for o in others:
                    A_therm[o, i_temp, i_co2] = 0.1

        A[0] = A_therm

        # 2. CO2 Sensor mapping
        A_co2 = np.zeros((3, 3, 3))
        for i_temp in range(3):
            for i_co2 in range(3):
                A_co2[i_co2, i_temp, i_co2] = 0.9  # More precise
                others = [x for x in range(3) if x != i_co2]
                for o in others:
                    A_co2[o, i_temp, i_co2] = 0.05

        A[1] = A_co2

        return A

    def _build_transition_B(self) -> np.ndarray:
        """Build B matrix: P(s'|s, u)."""
        # B[factor][next_state, current_state, action]
        B = utils.obj_array(len(self.num_states))

        # Actions: 0=DoNothing, 1=ReduceEmissions, 2=GeoEngineer

        # Factor 0: Temperature
        B_temp = np.zeros((3, 3, 3))

        # Action 0: Do Nothing -> Temp likely increases.
        # Shape is (next, curr). Columns sum to 1.
        # Col 0 (from Normal):
        B_temp[:, 0, 0] = [0.7, 0.3, 0.0]  # 30% chance to heat up
        B_temp[:, 1, 0] = [0.0, 0.7, 0.3]  # 30% chance to heat up
        B_temp[:, 2, 0] = [0.0, 0.0, 1.0]  # Stuck at High

        # Action 1: Reduce Emissions -> Stabilize
        B_temp[:, 0, 1] = [0.9, 0.1, 0.0]
        B_temp[:, 1, 1] = [0.1, 0.8, 0.1]
        B_temp[:, 2, 1] = [0.0, 0.1, 0.9]

        # Action 2: GeoEngineer -> Cool down fast but risky
        B_temp[:, 0, 2] = [1.0, 0.5, 0.2]
        B_temp[:, 1, 2] = [0.0, 0.4, 0.5]
        B_temp[:, 2, 2] = [0.0, 0.1, 0.3]

        # Normalize just in case
        for a in range(3):
            B_temp[:, :, a] = utils.norm_dist(B_temp[:, :, a])

        B[0] = B_temp

        # Factor 1: CO2.
        B_co2 = np.zeros((3, 3, 3))
        # Same column-stochastic (next, curr) layout as the temperature factor.
        # Action 0 Do nothing
        B_co2[:, 0, 0] = [0.6, 0.4, 0.0]
        B_co2[:, 1, 0] = [0.0, 0.6, 0.4]
        B_co2[:, 2, 0] = [0.0, 0.0, 1.0]

        # Action 1 Reduce
        B_co2[:, 0, 1] = [0.9, 0.1, 0.0]
        B_co2[:, 1, 1] = [0.2, 0.7, 0.1]
        B_co2[:, 2, 1] = [0.0, 0.1, 0.9]

        # Action 2 GeoEng: modeled as solar-radiation management, which cools
        # the temperature factor but does not change CO2 directly. The CO2
        # transition under GeoEngineering therefore equals the Reduce
        # Emissions transition.
        B_co2[:, :, 2] = B_co2[:, :, 1]

        B[1] = B_co2

        return B

    def _build_preferences_C(self) -> np.ndarray:
        """Build C matrix: P(o) (Preferences)."""
        # C[modality][observation]
        C = utils.obj_array(len(self.num_obs))

        # Prefer Normal Temp (Index 0)
        C[0] = np.array(
            [3.0, 0.0, -5.0]
        )  # High preference for Normal, Very low for High

        # Prefer Safe CO2 (Index 0)
        C[1] = np.array([2.0, 0.0, -2.0])

        return C

    def _build_prior_D(self) -> np.ndarray:
        """Build D matrix: P(s) (Initial beliefs)."""
        # D[factor][state]
        D = utils.obj_array(len(self.num_states))

        # Start at Normal Temp, Safe CO2
        D[0] = np.array([0.9, 0.1, 0.0])
        D[1] = np.array([0.9, 0.1, 0.0])

        return D

    def step(self, observations: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        """
        Execute one step of active inference.

        Args:
            observations: List of observations indices [obs_modality_1, obs_modality_2]
        """
        if observations is None:
            # Default observation (e.g. from environment)
            obs = [0, 0]
        else:
            obs = observations

        return super().step(obs, **kwargs)
