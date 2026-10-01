"""
Ecological model for active inference.
"""

from typing import Any
import numpy as np

from geo_infer_act.core.active_inference import ActiveInferenceModel
from geo_infer_act.utils.pymdp_adapter import obj_array_zeros


class EcologicalModel(ActiveInferenceModel):
    """
    Ecological niche modeling using Active Inference.

    This model simulates an organism adapting to its ecological niche by inferring
    hidden environmental states (Resources, Predation Risk) from observations
    (Food Availability, Threat Signals) and selecting adaptive policies (Forage, Hide, Migrate).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        random_seed: int | None = None,
    ):
        """
        Initialize the Ecological Model.

        Args:
            config: Configuration dictionary with optional matrix overrides.
        """
        config = config or {}
        if random_seed is not None:
            config["random_seed"] = random_seed

        # 1. Define State Space (Hidden States)
        # Factor 0: Resource Level (0=Low, 1=Medium, 2=High)
        # Factor 1: Predation Risk (0=Safe, 1=Risky)
        self.num_states = [3, 2]
        self.num_factors = len(self.num_states)

        # 2. Define Observation Space
        # Modality 0: Food Signal (0=None, 1=Scant, 2=Abundant)
        # Modality 1: Threat Signal (0=Quiet, 1=Noise/Scent)
        self.num_obs = [3, 2]
        self.num_modalities = len(self.num_obs)

        # 3. Define Action Space (Control Factors)
        # Factor 0: Action (0=Wait, 1=Forage, 2=Hide)
        self.num_controls = [3]

        # 4. Construct Matrices (if not provided in config)
        if "A" not in config:
            config["A"] = self._build_A_matrix()
        if "B" not in config:
            config["B"] = self._build_B_matrix()
        if "C" not in config:
            config["C"] = self._build_C_matrix()
        if "D" not in config:
            config["D"] = self._build_D_matrix()

        # Initialize base Active Inference Agent
        super().__init__(model_type="categorical", **config)

        # Ensure GenerativeModel is set with these parameters
        from geo_infer_act.core.generative_model import GenerativeModel

        gen_model = GenerativeModel(
            model_type="categorical", parameters=config, model_id="ecological_agent"
        )
        self.set_generative_model(gen_model)

    def _build_A_matrix(self) -> list[np.ndarray]:
        """Build Likelihood Matrix A: P(o|s)."""
        A: list[np.ndarray]
        A = obj_array_zeros([self.num_obs, self.num_states])

        # --- Modality 0: Food Signal (mapping from Resource Level) ---
        # State Factor 0 (Resources): Low(0) -> None(0), Med(1) -> Scant(1), High(2) -> Abundant(2)
        # State Factor 1 (Risk): Irrelevant for food signal

        A_food = np.zeros((3, 3, 2))
        # Dimensions: [Obs_Food, State_Res, State_Risk]

        # If Resources=Low(0), Food=None(0) with high prob
        A_food[0, 0, :] = 0.8
        A_food[1, 0, :] = 0.2
        A_food[2, 0, :] = 0.0

        # If Resources=Med(1), Food=Scant(1) mostly
        A_food[0, 1, :] = 0.1
        A_food[1, 1, :] = 0.8
        A_food[2, 1, :] = 0.1

        # If Resources=High(2), Food=Abundant(2) mostly
        A_food[0, 2, :] = 0.0
        A_food[1, 2, :] = 0.2
        A_food[2, 2, :] = 0.8

        # Flatten A_food to (Obs_Food, Total_States=6)
        A[0] = A_food.reshape(3, 6)

        # --- Modality 1: Threat Signal (mapping from Risk) ---
        # State Factor 0 (Resources): Irrelevant
        # State Factor 1 (Risk): Safe(0) -> Quiet(0), Risky(1) -> Noise(1)

        A_threat = np.zeros((2, 3, 2))

        # If Risk=Safe(0), Threat=Quiet(0)
        A_threat[0, :, 0] = 0.9
        A_threat[1, :, 0] = 0.1

        # If Risk=Risky(1), Threat=Noise(1)
        A_threat[0, :, 1] = 0.2
        A_threat[1, :, 1] = 0.8

        A[1] = A_threat.reshape(2, 6)

        return A

    def _build_B_matrix(self) -> list[np.ndarray]:
        """Build Transition Matrix B: P(s'|s,u)."""
        # Initialize B as list of arrays for independent factors
        # Factor 0: Resource (3 states)
        # Factor 1: Risk (2 states)
        # Control: Action (3 actions)
        B = [
            np.zeros((self.num_states[0], self.num_states[0], self.num_controls[0])),
            np.zeros((self.num_states[1], self.num_states[1], self.num_controls[0])),
        ]

        # --- Factor 0: Resource Dynamics ---
        # Action 0 (Wait): Resources stay same (mostly)
        # Action 1 (Forage): Resources deplete (High->Med, Med->Low)
        # Action 2 (Hide): Resources stay same

        # Action 0: Wait
        B[0][:, :, 0] = np.eye(3)

        # Action 1: Forage (Depletion dynamics)
        B_forage = np.zeros((3, 3))
        # Low -> Low
        B_forage[0, 0] = 1.0
        # Med -> Low (consumed)
        B_forage[0, 1] = 0.6
        B_forage[1, 1] = 0.4
        # High -> Med (consumed)
        B_forage[1, 2] = 0.6
        B_forage[2, 2] = 0.4

        B[0][:, :, 1] = B_forage

        # Action 2: Hide (Regeneration chance)
        B_hide = np.eye(3)
        # Small chance of regeneration from Low->Med if hiding (nature heals)
        B_hide[0, 0] = 0.9
        B_hide[1, 0] = 0.1

        B[0][:, :, 2] = B_hide

        # --- Factor 1: Risk Dynamics ---
        # Action 0 (Wait): Risk stays same
        # Action 1 (Forage): Increases risk (Safe -> Risky)
        # Action 2 (Hide): Decreases risk (Risky -> Safe)

        # Action 0: Wait
        B[1][:, :, 0] = np.eye(2)

        # Action 1: Forage (Risk increases)
        B_risk_forage = np.zeros((2, 2))
        B_risk_forage[0, 0] = 0.4  # Safe stays safe? Less likely
        B_risk_forage[1, 0] = 0.6  # Safe becomes Risky
        B_risk_forage[0, 1] = 0.0
        B_risk_forage[1, 1] = 1.0  # Risky stays Risky
        B[1][:, :, 1] = B_risk_forage

        # Action 2: Hide (Risk decreases)
        B_risk_hide = np.zeros((2, 2))
        B_risk_hide[0, 0] = 1.0  # Safe stays Safe
        B_risk_hide[0, 1] = 0.8  # Risky becomes Safe
        B_risk_hide[1, 1] = 0.2
        B[1][:, :, 2] = B_risk_hide

        return B

    def _build_C_matrix(self) -> list[np.ndarray]:
        """Build Preference Matrix C: P(o)."""
        C: list[np.ndarray]
        C = obj_array_zeros(self.num_obs)

        # Prefer Abundant Food (Modality 0, Index 2)
        # C values are log-probabilities (utilities)
        C[0][0] = -2.0  # None (Dislike)
        C[0][1] = 0.0  # Scant (Neutral)
        C[0][2] = 4.0  # Abundant (Like)

        # Prefer Quiet/Safe (Modality 1, Index 0)
        C[1][0] = 2.0  # Quiet (Safe)
        C[1][1] = -4.0  # Noise (Danger)

        return C

    def _build_D_matrix(self) -> list[np.ndarray]:
        """Build Prior Matrix D: P(s)."""
        D: list[np.ndarray]
        D = obj_array_zeros(self.num_states)

        # Start expecting High Resources
        D[0][0] = 0.1
        D[0][1] = 0.3
        D[0][2] = 0.6

        # Start expecting Safe
        D[1][0] = 0.8
        D[1][1] = 0.2

        return D

    def step(self, observation: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        """
        Advance the ecological model by one step.

        Args:
            observation: List of integers [Food_Obs_Idx, Threat_Obs_Idx]
        """
        if observation is None:
            obs = np.array([0, 0])
        else:
            obs = np.asarray(observation)

        # Perceive
        beliefs = self.perceive(obs)

        # Act
        action = self.act()

        return {"beliefs": beliefs, "action": action, "observation": obs.tolist()}
