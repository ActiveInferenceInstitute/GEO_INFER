"""
Integration utilities for connecting with other GEO-INFER modules and modern tools.

Enhanced with support for RxInfer, Bayeux, pymdp, and other state-of-the-art
Active Inference frameworks based on Active Inference Institute resources.
"""

from typing import Any
from collections.abc import Callable
import copy
import importlib
import logging
import numpy as np

from geo_infer_act.utils.config import get_config_value

try:
    import geo_infer_space as space_h3
except ModuleNotFoundError as exc:
    if exc.name != "geo_infer_space":
        raise
    space_h3 = None


def initialize_logger() -> logging.Logger:
    """Return the module logger without configuring process-wide handlers."""
    return logging.getLogger(__name__)


logger = initialize_logger()


class ModernToolsIntegration:
    """
    Integration hub for modern Active Inference tools and frameworks.

    Supports integration with:
    - RxInfer.jl (Julia-based factor graphs)
    - Bayeux (JAX-based probabilistic programming)
    - pymdp (Python discrete active inference)
    - PyMC (Probabilistic programming)
    - Pyro (Deep probabilistic programming)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the integration hub.

        Args:
            config: Configuration dictionary
        """
        self.config = copy.deepcopy(config or {})
        self.available_tools = self._check_available_tools()
        logger.info(f"Available tools: {list(self.available_tools.keys())}")

    def _execute_dynamic_source(
        self, source: str, namespace: dict[str, Any], description: str
    ) -> dict[str, Any]:
        """Execute optional model source only after explicit caller opt-in."""
        if self.config.get("allow_dynamic_code") is not True:
            raise RuntimeError(
                f"{description} requires config['allow_dynamic_code']=True"
            )
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"{description} must be a non-empty source string")
        namespace = dict(namespace)
        exec(source, namespace, namespace)
        return namespace

    def _check_available_tools(self) -> dict[str, bool]:
        """Check which modern tools are available in the environment."""
        tools = {}

        # Check for RxInfer (Julia package - requires julia and PyJulia)
        try:
            import julia

            j = julia.Julia(compiled_modules=False)
            j.eval("using RxInfer")
            tools["rxinfer"] = True
            logger.debug("RxInfer.jl available")
        except Exception:
            tools["rxinfer"] = False
            logger.debug("RxInfer.jl not available")

        tools["bayeux"] = importlib.util.find_spec("bayeux") is not None
        logger.debug("Bayeux available" if tools["bayeux"] else "Bayeux not available")

        # Check for pymdp
        tools["pymdp"] = importlib.util.find_spec("pymdp") is not None
        logger.debug("pymdp available" if tools["pymdp"] else "pymdp not available")

        # Check for PyMC
        tools["pymc"] = importlib.util.find_spec("pymc") is not None
        logger.debug("PyMC available" if tools["pymc"] else "PyMC not available")

        # Check for Pyro
        tools["pyro"] = importlib.util.find_spec("pyro") is not None
        logger.debug("Pyro available" if tools["pyro"] else "Pyro not available")

        # Check for JAX
        tools["jax"] = importlib.util.find_spec("jax") is not None
        logger.debug("JAX available" if tools["jax"] else "JAX not available")

        return tools

    def create_rxinfer_model(
        self, model_spec: str, data: dict[str, Any], *, backend: str = "rxinfer"
    ) -> dict[str, Any]:
        """Run the declared Julia model or an explicitly chosen local Gaussian.

        The local Gaussian backend conditions a scalar normal prior on the
        supplied observations. It does not execute the Julia specification.
        Backend errors propagate and never select a different model.
        """
        if "allow_local_fallback" in self.config:
            raise ValueError(
                "allow_local_fallback was removed; choose backend='local_gaussian' explicitly"
            )
        from geo_infer_act.core.generative_model import GenerativeModel

        result = GenerativeModel("categorical", {"state_dim": 1}).integrate_rxinfer(
            model_spec, data, backend=backend
        )
        return {**result, "tool": result["backend"]}

    def create_bayeux_model(
        self,
        log_density_fn: str | Callable,
        test_point: dict[str, Any],
        transform_fn: str | Callable | None = None,
        *,
        n_samples: int = 1000,
        warmup: int = 100,
    ) -> dict[str, Any]:
        """Run real Bayeux NumPyro NUTS for a mapping-valued parameter point.

        Density and transform callables receive the parameter mapping. Python
        source strings require ``allow_dynamic_code=True`` and are evaluated
        in a fresh namespace. No optimizer or other sampler is substituted.
        """
        namespace = {"__name__": "geo_infer_act.dynamic.bayeux"}
        if isinstance(log_density_fn, str):
            namespace = self._execute_dynamic_source(
                log_density_fn, namespace, "Bayeux log-density source"
            )
            density = namespace.get("log_density")
        else:
            density = log_density_fn
        if not callable(density):
            raise ValueError("log_density_fn must be callable or define 'log_density'")
        if isinstance(transform_fn, str):
            namespace = self._execute_dynamic_source(
                transform_fn, namespace, "Bayeux transform source"
            )
            transform = namespace.get("transform_fn")
        else:
            transform = transform_fn
        if transform_fn is not None and not callable(transform):
            raise ValueError("transform_fn must be callable or define 'transform_fn'")
        from geo_infer_act.core.generative_model import GenerativeModel

        result = GenerativeModel(
            "categorical",
            {"state_dim": 1, "random_seed": self.config.get("random_seed", 0)},
        ).integrate_bayeux(
            lambda **point: density(point),
            test_point,
            backend="bayeux",
            n_samples=n_samples,
            warmup=warmup,
            transform_fn=transform,
        )
        return {**result, "tool": "bayeux"}

    def create_pymdp_agent(
        self,
        num_obs: list[int],
        num_states: list[int],
        A: np.ndarray | None = None,
        B: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Create pymdp agent for discrete Active Inference.

        Args:
            num_obs: Number of observations for each modality
            num_states: Number of states for each factor
            A: Observation model (optional)
            B: Transition model (optional)

        Returns:
            Agent and initial results
        """
        if not self.available_tools.get("pymdp", False):
            raise RuntimeError(
                "pymdp not available. Please install: uv pip install pymdp"
            )

        try:
            from geo_infer_act.utils.pymdp_adapter import (
                build_agent,
                jax_prng_key,
                onehot_batch,
                random_A_array,
                random_B_array,
            )

            # Create observation model if not provided
            if A is None:
                A = random_A_array(jax_prng_key(0), num_obs, num_states)

            # Create transition model if not provided
            if B is None:
                B = random_B_array(jax_prng_key(1), num_states, num_states)

            # Create agent through the module's only pymdp bridge
            agent = build_agent(
                A=A,
                B=B,
                C=[np.zeros(obs_dim) for obs_dim in num_obs],
                D=[np.ones(state_dim) / state_dim for state_dim in num_states],
                num_controls=list(num_states),
                categorical_obs=True,
                batch_size=1,
            )

            # Test inference with a deterministic observation.  This helper is
            # a contract smoke test, not a source of model randomness.
            rng = np.random.default_rng(0)
            obs = onehot_batch(
                num_obs, [int(rng.integers(0, num_obs[i])) for i in range(len(num_obs))]
            )
            qs = agent.infer_states(obs, empirical_prior=agent.D)

            # Test policy inference
            q_pi, G = agent.infer_policies(qs)

            return {
                "status": "success",
                "agent": agent,
                "initial_beliefs": qs,
                "policy_probabilities": q_pi,
                "expected_free_energies": G,
                "tool": "pymdp",
            }

        except Exception as e:
            logger.error(f"pymdp integration failed: {e}")
            return {"status": "error", "message": str(e), "tool": "pymdp"}

    def create_pymc_model(
        self, model_spec: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create PyMC model for Bayesian inference.

        Args:
            model_spec: PyMC model specification string
            data: Data for inference

        Returns:
            Inference results
        """
        if not self.available_tools.get("pymc", False):
            raise RuntimeError(
                "PyMC not available. Please install: uv pip install pymc"
            )

        try:
            import pymc as pm
            import arviz as az

            # Create model context and execute specification in an isolated
            # namespace after explicit caller opt-in.
            model_context = self._execute_dynamic_source(
                model_spec,
                {"pm": pm, "data": data},
                "PyMC model source",
            )
            model = model_context.get("model")

            if model is None:
                raise ValueError(
                    "Model specification must create a variable named 'model'"
                )

            # Sample from model
            with model:
                trace = pm.sample(
                    draws=1000,
                    tune=1000,
                    chains=4,
                    return_inferencedata=True,
                    progressbar=False,
                    random_seed=self.config.get("random_seed", 42),
                )

            # Compute diagnostics
            summary = az.summary(trace)

            return {
                "status": "success",
                "trace": trace,
                "summary": summary,
                "tool": "pymc",
            }

        except Exception as e:
            logger.error(f"PyMC integration failed: {e}")
            return {"status": "error", "message": str(e), "tool": "pymc"}

    def create_pyro_model(
        self, model_fn: str, guide_fn: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create Pyro model for deep probabilistic programming.

        Args:
            model_fn: Pyro model function string
            guide_fn: Pyro guide function string
            data: Data for inference

        Returns:
            Inference results
        """
        if not self.available_tools.get("pyro", False):
            raise RuntimeError(
                "Pyro not available. Please install: uv pip install pyro-ppl"
            )

        try:
            import pyro
            import pyro.distributions as dist
            from pyro.infer import SVI, Trace_ELBO
            from pyro import optim as pyro_optim
            import torch

            Adam = pyro_optim.Adam  # type: ignore[attr-defined]

            # Clear Pyro parameter store
            pyro.clear_param_store()

            # Create model and guide functions
            exec_env = {
                "pyro": pyro,
                "dist": dist,
                "torch": torch,
                "np": np,
            }
            exec_env = self._execute_dynamic_source(
                model_fn, exec_env, "Pyro model source"
            )
            exec_env = self._execute_dynamic_source(
                guide_fn, exec_env, "Pyro guide source"
            )

            model = exec_env.get("model")
            guide = exec_env.get("guide")

            if model is None or guide is None:
                raise ValueError("Must define 'model' and 'guide' functions")

            # Set up SVI
            optimizer = Adam({"lr": 0.01})
            svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

            # Train
            losses = []
            for step in range(1000):
                loss = svi.step(data)
                losses.append(loss)

                if step % 100 == 0:
                    logger.debug(f"Step {step}, Loss: {loss}")

            # Extract learned parameters
            learned_params = {
                name: param.detach().numpy()
                for name, param in pyro.get_param_store().items()
            }

            return {
                "status": "success",
                "losses": losses,
                "learned_params": learned_params,
                "final_loss": losses[-1],
                "tool": "pyro",
            }

        except Exception as e:
            logger.error(f"Pyro integration failed: {e}")
            return {"status": "error", "message": str(e), "tool": "pyro"}


def integrate_rxinfer(
    config: dict[str, Any], model_params: dict[str, Any]
) -> dict[str, Any]:
    """Run the explicit backend and supplied observations; never invent data."""
    if "data" not in model_params:
        raise ValueError("RxInfer integration requires explicit data")
    backend = model_params.get("backend", "rxinfer")
    if backend == "rxinfer" and "model_specification" not in model_params:
        raise ValueError("RxInfer integration requires explicit model_specification")
    integration_hub = ModernToolsIntegration(config)
    return integration_hub.create_rxinfer_model(
        model_params.get("model_specification", ""),
        model_params["data"],
        backend=backend,
    )


def integrate_bayeux(
    config: dict[str, Any], model_params: dict[str, Any]
) -> dict[str, Any]:
    """Run Bayeux NUTS on the caller's explicit density and parameter point."""
    if not {"log_density", "test_point"} <= model_params.keys():
        raise ValueError(
            "Bayeux integration requires explicit log_density and test_point"
        )
    integration_hub = ModernToolsIntegration(config)
    return integration_hub.create_bayeux_model(
        model_params["log_density"],
        model_params["test_point"],
        model_params.get("transform_fn"),
        n_samples=model_params.get("n_samples", 1000),
        warmup=model_params.get("warmup", 100),
    )


def integrate_pymdp(
    config: dict[str, Any], model_params: dict[str, Any]
) -> dict[str, Any]:
    """Integrate with pymdp for discrete Active Inference."""
    integration_hub = ModernToolsIntegration(config)

    num_obs = model_params.get("num_obs", [4, 3])  # Two modalities
    num_states = model_params.get("num_states", [3, 2])  # Two factors
    A = model_params.get("A", None)
    B = model_params.get("B", None)

    return integration_hub.create_pymdp_agent(num_obs, num_states, A, B)


def integrate_space(
    config: dict[str, Any], data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Integrate with GEO-INFER-SPACE module.

    Args:
        config: Configuration dictionary
        data: Optional data supplied to the space module

    Returns:
        Results from space module integration
    """
    # Check if integration is enabled
    is_enabled = get_config_value(config, "integration.space_module.enabled", False)

    if not is_enabled:
        logger.info("Space module integration is disabled in config")
        return {}

    # Get API endpoint from config
    api_endpoint = get_config_value(
        config, "integration.space_module.api_endpoint", None
    )

    if not api_endpoint:
        logger.warning("Space module API endpoint not configured")
        return {}

    try:
        # Import the space module API
        module_path, api_class = api_endpoint.rsplit(".", 1)
        space_module = importlib.import_module(module_path)
        api_cls = getattr(space_module, api_class)

        # Initialize API
        space_api = api_cls()

        # Call API methods based on the provided data
        if data is None:
            data = {}

        if "action" not in data:
            logger.warning("No action specified for space module integration")
            return {}

        action = data["action"]
        action_params = data.get("params", {})

        if hasattr(space_api, action):
            action_method = getattr(space_api, action)
            result = action_method(**action_params)
            return {"status": "success", "result": result}
        else:
            logger.warning(f"Action {action} not found in space module API")
            return {"status": "error", "message": f"Action {action} not supported"}

    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import space module: {str(e)}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Error in space module integration: {str(e)}")
        return {"status": "error", "message": str(e)}


def integrate_time(
    config: dict[str, Any], data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Integrate with GEO-INFER-TIME module.

    Args:
        config: Configuration dictionary
        data: Optional data supplied to the time module

    Returns:
        Results from time module integration
    """
    # Implementation similar to integrate_space
    is_enabled = get_config_value(config, "integration.time_module.enabled", False)

    if not is_enabled:
        logger.info("Time module integration is disabled in config")
        return {}

    api_endpoint = get_config_value(
        config, "integration.time_module.api_endpoint", None
    )

    if not api_endpoint:
        logger.warning("Time module API endpoint not configured")
        return {}

    try:
        # Import the time module API
        module_path, api_class = api_endpoint.rsplit(".", 1)
        time_module = importlib.import_module(module_path)
        api_cls = getattr(time_module, api_class)

        # Initialize API
        time_api = api_cls()

        # Call API methods based on the provided data
        if data is None:
            data = {}

        if "action" not in data:
            logger.warning("No action specified for time module integration")
            return {}

        action = data["action"]
        action_params = data.get("params", {})

        if hasattr(time_api, action):
            action_method = getattr(time_api, action)
            result = action_method(**action_params)
            return {"status": "success", "result": result}
        else:
            logger.warning(f"Action {action} not found in time module API")
            return {"status": "error", "message": f"Action {action} not supported"}

    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import time module: {str(e)}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Error in time module integration: {str(e)}")
        return {"status": "error", "message": str(e)}


def integrate_sim(
    config: dict[str, Any], data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Integrate with GEO-INFER-SIM module.

    Args:
        config: Configuration dictionary
        data: Optional data supplied to the simulation module

    Returns:
        Results from simulation module integration
    """
    # Implementation similar to integrate_space
    is_enabled = get_config_value(config, "integration.sim_module.enabled", False)

    if not is_enabled:
        logger.info("Simulation module integration is disabled in config")
        return {}

    api_endpoint = get_config_value(config, "integration.sim_module.api_endpoint", None)

    if not api_endpoint:
        logger.warning("Simulation module API endpoint not configured")
        return {}

    try:
        # Import the simulation module API
        module_path, api_class = api_endpoint.rsplit(".", 1)
        sim_module = importlib.import_module(module_path)
        api_cls = getattr(sim_module, api_class)

        # Initialize API
        sim_api = api_cls()

        # Call API methods based on the provided data
        if data is None:
            data = {}

        if "action" not in data:
            logger.warning("No action specified for simulation module integration")
            return {}

        action = data["action"]
        action_params = data.get("params", {})

        if hasattr(sim_api, action):
            action_method = getattr(sim_api, action)
            result = action_method(**action_params)
            return {"status": "success", "result": result}
        else:
            logger.warning(f"Action {action} not found in simulation module API")
            return {"status": "error", "message": f"Action {action} not supported"}

    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import simulation module: {str(e)}")
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error(f"Error in simulation module integration: {str(e)}")
        return {"status": "error", "message": str(e)}


def create_h3_spatial_model(
    config: dict[str, Any], h3_resolution: int, boundary: dict[str, Any]
) -> dict[str, Any]:
    """
    Create H3-based spatial Active Inference model.

    Args:
        config: Configuration dictionary
        h3_resolution: H3 hexagonal grid resolution
        boundary: GeoJSON Polygon or MultiPolygon geometry, or a Feature or
            FeatureCollection containing those geometries. Every mapping must
            carry a GeoJSON ``type`` member.

    Returns:
        H3 spatial model configuration
    """
    try:
        settings = config or {}
        max_cells = settings.get("max_cells", 100_000)
        if (
            isinstance(max_cells, bool)
            or not isinstance(max_cells, (int, np.integer))
            or max_cells < 1
        ):
            raise ValueError("max_cells must be at least 1")
        from geo_infer_act.utils.h3_adapter import get_h3_adapter

        adapter = get_h3_adapter()

        if (
            isinstance(h3_resolution, bool)
            or not isinstance(h3_resolution, (int, np.integer))
            or not 0 <= h3_resolution <= 15
        ):
            raise ValueError("h3_resolution must be between 0 and 15")
        if hasattr(boundary, "__geo_interface__"):
            boundary = boundary.__geo_interface__
        if not isinstance(boundary, dict):
            raise ValueError("boundary must be a GeoJSON-like mapping")

        # Invalid, untyped, empty, point, or malformed inputs fail closed
        # rather than silently generating an unrelated grid.
        boundary_type = boundary.get("type")
        if boundary_type not in {
            "Polygon",
            "MultiPolygon",
            "Feature",
            "FeatureCollection",
        }:
            raise ValueError(
                "boundary must be a Polygon, MultiPolygon, Feature, or "
                "FeatureCollection"
            )

        try:
            boundary_cells = set(adapter.polygon_to_cells(boundary, h3_resolution))
        except Exception as poly_error:
            raise ValueError(
                f"H3 boundary conversion failed: {poly_error}"
            ) from poly_error
        if not boundary_cells:
            raise ValueError(
                "H3 boundary produced no cells at the requested resolution"
            )

        num_cells = len(boundary_cells)
        if num_cells > max_cells:
            return {
                "status": "error",
                "message": (
                    f"H3 boundary produced {num_cells} cells, exceeding the "
                    f"max_cells limit of {max_cells}; use a coarser resolution "
                    "or raise config['max_cells']"
                ),
            }
        return {
            "status": "success",
            "model_config": {
                "boundary_cells": list(boundary_cells),
                "estimated_cells": num_cells,
                "max_cells": max_cells,
            },
        }
    except RuntimeError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as e:
        logger.error(f"H3 spatial model creation failed: {e}")
        return {"status": "error", "message": str(e)}


def coordinate_multi_agent_system(
    config: dict[str, Any], agents: list[dict[str, Any]], environment: dict[str, Any]
) -> dict[str, Any]:
    """
    Coordinate multiple Active Inference agents.

    Args:
        config: Configuration dictionary
        agents: List of agent specifications
        environment: Shared environment specification

    Returns:
        Multi-agent coordination results
    """
    try:
        coordination_protocol = config.get("coordination_protocol", "consensus")
        communication_range = config.get("communication_range", 1.0)

        # Initialize coordination state
        coordination_state = {
            "agents": {},
            "environment": environment,
            "communication_graph": {},
            "collective_beliefs": {},
            "coordination_protocol": coordination_protocol,
        }

        # Set up agents
        for agent_spec in agents:
            agent_id = agent_spec["agent_id"]
            coordination_state["agents"][agent_id] = {
                "model_id": agent_spec["model_id"],
                "position": agent_spec.get("initial_position", [0, 0]),
                "capabilities": agent_spec.get("capabilities", []),
                "communication_range": communication_range,
                "local_beliefs": {},
                "shared_beliefs": {},
            }

        # Create communication graph
        agent_ids = list(coordination_state["agents"].keys())
        for i, agent_a in enumerate(agent_ids):
            coordination_state["communication_graph"][agent_a] = []
            pos_a = coordination_state["agents"][agent_a]["position"]

            for j, agent_b in enumerate(agent_ids):
                if i != j:
                    pos_b = coordination_state["agents"][agent_b]["position"]
                    distance = np.linalg.norm(np.array(pos_a) - np.array(pos_b))

                    if distance <= communication_range:
                        coordination_state["communication_graph"][agent_a].append(
                            agent_b
                        )

        # Initialize collective belief updating
        if coordination_protocol == "consensus":
            coordination_algorithm = _consensus_belief_updating
        elif coordination_protocol == "hierarchical":
            coordination_algorithm = _hierarchical_coordination
        else:
            coordination_algorithm = _pairwise_coordination

        logger.info(f"Initialized multi-agent system with {len(agents)} agents")

        return {
            "status": "success",
            "coordination_state": coordination_state,
            "coordination_algorithm": coordination_algorithm.__name__,
            "communication_graph_size": sum(
                len(neighbors)
                for neighbors in coordination_state["communication_graph"].values()
            ),
        }

    except Exception as e:
        logger.error(f"Multi-agent coordination setup failed: {e}")
        return {"status": "error", "message": str(e)}


def _consensus_belief_updating(coordination_state: dict[str, Any]) -> dict[str, Any]:
    """Implement consensus-based belief updating among agents."""
    # Simplified consensus algorithm
    agents = coordination_state["agents"]
    communication_graph = coordination_state["communication_graph"]

    # Update beliefs through consensus
    for agent_id in agents:
        neighbors = communication_graph[agent_id]
        if neighbors:
            # Average beliefs with neighbors (simplified)
            agents[agent_id]["shared_beliefs"] = {
                "consensus_reached": len(neighbors) > 0,
                "neighbor_count": len(neighbors),
            }

    return coordination_state


def _hierarchical_coordination(coordination_state: dict[str, Any]) -> dict[str, Any]:
    """Implement hierarchical coordination among agents."""
    # Simplified hierarchical coordination
    agents = coordination_state["agents"]

    # Designate first agent as coordinator
    agent_ids = list(agents.keys())
    if agent_ids:
        coordinator_id = agent_ids[0]
        agents[coordinator_id]["role"] = "coordinator"

        for agent_id in agent_ids[1:]:
            agents[agent_id]["role"] = "follower"
            agents[agent_id]["coordinator"] = coordinator_id

    return coordination_state


def _pairwise_coordination(coordination_state: dict[str, Any]) -> dict[str, Any]:
    """Implement pairwise coordination among agents."""
    # Simplified pairwise coordination
    agents = coordination_state["agents"]
    communication_graph = coordination_state["communication_graph"]

    # Create pairwise coordination links
    for agent_id, neighbors in communication_graph.items():
        agents[agent_id]["pairwise_links"] = neighbors

    return coordination_state


class IntegrationUtils:
    """
    Utility class for integrating with other modules and tools.

    Provides convenience methods for common integration tasks.
    """

    @staticmethod
    def get_modern_tools() -> ModernToolsIntegration:
        """Get available modern tools integration."""
        return ModernToolsIntegration()

    @staticmethod
    def integrate_with_space(spatial_data: dict[str, Any]) -> dict[str, Any]:
        """Integrate with GEO-INFER-SPACE module."""
        return integrate_space(spatial_data)

    @staticmethod
    def integrate_with_time(temporal_data: dict[str, Any]) -> dict[str, Any]:
        """Integrate with GEO-INFER-TIME module."""
        return integrate_time(temporal_data)

    @staticmethod
    def create_multi_agent_system(
        agent_configs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create and coordinate a multi-agent system."""
        return coordinate_multi_agent_system({}, agent_configs, {})


# Export integration functions from the canonical integration module.
__all__ = [
    "IntegrationUtils",
    "ModernToolsIntegration",
    "integrate_rxinfer",
    "integrate_bayeux",
    "integrate_pymdp",
    "integrate_space",
    "integrate_time",
    "integrate_sim",
    "create_h3_spatial_model",
    "coordinate_multi_agent_system",
]
