"""
Urban planning model using active inference.
"""

from typing import Any
import numpy as np

from geo_infer_act.models.base import BaseActiveInferenceModel, _dimension
from geo_infer_act.core.generative_model import GenerativeModel
from geo_infer_act.core.active_inference import ActiveInferenceModel as Agent


class UrbanModel(BaseActiveInferenceModel):
    """
    Urban planning model using active inference.

    This model simulates urban development with multiple agents (Stakeholders/Residents)
    navigating a city graph to optimize their resource access (Housing, Transport, Amenities).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        n_agents: int = 3,
        n_resources: int = 3,  # Reduced to 3 for 'Low', 'Med', 'High' levels of amenity
        n_locations: int = 5,
        planning_horizon: int = 2,
        random_seed: int | None = None,
    ):
        """
        Initialize the urban planning model.

        Args:
            config: Configuration dictionary
            n_agents: Number of agents
            n_resources: Number of resource levels (not types, for simplicity in this POMDP)
            n_locations: Number of spatial locations (nodes)
            planning_horizon: Planning time horizon
        """
        super().__init__(config)
        if random_seed is None and config:
            random_seed = config.get("random_seed")
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.n_agents = _dimension(n_agents, "n_agents")
        self.n_resources = _dimension(n_resources, "n_resources")
        self.n_locations = _dimension(n_locations, "n_locations")
        self.planning_horizon = _dimension(planning_horizon, "planning_horizon")

        # Environment State
        # resource_levels[loc] = level (0=Low, 1=Med, 2=High)
        self.resource_levels = self.rng.integers(0, n_resources, size=n_locations)

        # Connectivity Graph (Adjacency Matrix)
        # Create a ring graph + random connections
        self.connectivity = np.eye(n_locations)
        for i in range(n_locations):
            self.connectivity[i, (i + 1) % n_locations] = 1
            self.connectivity[i, (i - 1) % n_locations] = 1

        # Agents
        self.agents: list[dict[str, Any]] = []
        self._initialize_agents()
        self._initial_resource_levels = self.resource_levels.copy()
        self._initial_agent_locations = [agent["location"] for agent in self.agents]

    def _initialize_agents(self) -> None:
        """Initialize active inference agents with concrete models."""
        for i in range(self.n_agents):
            agent_id = f"resident_{i}"

            # --- Define Generative Model Matrices ---

            # 1. State Space: Location (where am I?)
            num_states = [self.n_locations]

            # 2. Observation Space:
            # Modality 0: GPS/Location Sensor (Perfect)
            # Modality 1: Resource Sensor (Noisy detection of local amenity level)
            num_obs = [self.n_locations, self.n_resources]

            # 3. Control Space: Move to adjacent node (or stay)
            # Action: Go to node 0, 1, ..., N-1. (Valid only if connected)
            num_controls = [self.n_locations]

            # --- A Matrix (Likelihood) ---
            # P(o|s)

            # Modality 0 (Location): Identity mapping (Perfect GPS)
            A_loc = np.eye(self.n_locations)

            # Modality 1 (Resource): Depends on location!
            # But the Agent's A matrix encodes *beliefs* about resources.
            # If agent knows the city map, A encodes the map.
            # If agent doesn't know, A is uniform (exploration).
            # Let's assume agents have a "Mental Map" of resource quality.
            # For simplicity, we initialize with a random belief map that they updates?
            # Actually, standard ActInf doesn't learn A matrix parameters online easily without 'learning' flag.
            # Let's give them a noisy map.

            A_res = np.ones((self.n_resources, self.n_locations)) * (
                0.2 / self.n_resources
            )
            for loc in range(self.n_locations):
                true_level = self.resource_levels[loc]
                # High prob of seeing the true level
                A_res[true_level, loc] += 0.8

            A = [A_loc, A_res]

            # --- B Matrix (Transition) ---
            # P(s'|s, u) - Movement dynamics
            B_mat = np.zeros((self.n_locations, self.n_locations, self.n_locations))
            for u in range(self.n_locations):  # Action: "Try to go to u"
                for s in range(self.n_locations):  # Current state
                    if self.connectivity[s, u]:
                        B_mat[u, s, u] = 1.0  # Successful move
                    else:
                        B_mat[s, s, u] = 1.0  # Stay if move invalid

            # Wrap B for pymdp (factorized)
            B = [B_mat]

            # --- C Matrix (Preferences) ---
            # Agents prefer High Resources (Modality 1, Index 2)
            C_loc = np.zeros(self.n_locations)  # No location preference
            C_res = np.linspace(-2.0, 2.0, self.n_resources)
            C = [C_loc, C_res]

            # --- D Matrix (Prior) ---
            # Start at random location
            D = [np.ones(self.n_locations) / self.n_locations]

            # Initialize Agent
            config = {
                "A": A,
                "B": B,
                "C": C,
                "D": D,
                "num_states": num_states,
                "num_obs": num_obs,
                "num_controls": num_controls,
                "inference_horizon": self.planning_horizon,
                "random_seed": (
                    None if self.random_seed is None else self.random_seed + i
                ),
            }

            agent = Agent(model_type="categorical", **config)

            # Set Generative Model explicitly
            gen_model = GenerativeModel(
                model_type="categorical", parameters=config, model_id=agent_id
            )
            agent.set_generative_model(gen_model)
            from geo_infer_act.core.factored_runtime import build_runtime_artifact

            build_runtime_artifact(gen_model)

            # Initial absolute state (simulation truth)
            start_loc = int(self.rng.integers(0, self.n_locations))

            self.agents.append(
                {"id": agent_id, "model": agent, "location": start_loc, "history": []}
            )

    def step(self, input_actions: Any | None = None) -> tuple[dict[str, Any], bool]:
        """Advance one simulation step."""
        if input_actions is not None:
            raise ValueError(
                "UrbanModel selects its own actions; input_actions are unsupported"
            )
        states = []

        for agent_data in self.agents:
            agent = agent_data["model"]
            loc = agent_data["location"]

            # 1. Generate Observation from Environment
            # Obs 0: Location (Identity)
            obs_loc = loc
            # Obs 1: Resource Level at current location
            obs_res = self.resource_levels[loc]

            observation = [int(obs_loc), int(obs_res)]

            # 2. Agent Perceives and Acts
            # We use the 'step' method of Agent
            # But wait, Agent.step returns (beliefs, action)
            # and 'action' might be an index.

            # Note: Agent 'step' runs perceive() then act()
            # We need to make sure perceive() handles the list observation [int, int]
            # My 'interface' update handled this, but 'Agent._update_beliefs_direct'
            # handles list of ints for pymdp.

            beliefs, action = agent.step(observation)

            # 3. Execute Action (Update Environment/Agent State)
            # Action is "Try to move to node X"
            target_loc = (
                int(action[0])
                if isinstance(action, (list, np.ndarray))
                else int(action)
            )

            if self.connectivity[loc, target_loc]:
                agent_data["location"] = target_loc

            agent_data["history"].append(
                {"loc": loc, "action": target_loc, "obs": observation}
            )

            states.append(
                {
                    "agent_id": agent_data["id"],
                    "location": agent_data["location"],
                    "beliefs": beliefs,
                }
            )

        return {"states": states, "resource_map": self.resource_levels.tolist()}, False

    def run_simulation(self, n_steps: int = 10) -> list[dict[str, Any]]:
        """Run repeated urban planning steps and return the state history."""
        if (
            isinstance(n_steps, bool)
            or not isinstance(n_steps, (int, np.integer))
            or n_steps < 0
        ):
            raise ValueError("n_steps must be a nonnegative integer")
        history = []
        for _ in range(n_steps):
            state, _ = self.step()
            history.append(state)
        return history

    def reset(self) -> dict[str, Any]:
        """Restore the seeded environment and every agent's initial location."""
        self.resource_levels = self._initial_resource_levels.copy()
        for agent_data, initial_location in zip(
            self.agents, self._initial_agent_locations
        ):
            agent_data["location"] = initial_location
            agent_data["history"] = []
            agent_data["model"].reset()
        return {"resource_map": self.resource_levels.tolist(), "states": []}
