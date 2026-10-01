"""
Integration utilities for connecting with other GEO-INFER modules and modern tools.

Enhanced with support for RxInfer, Bayeux, pymdp, and other state-of-the-art
Active Inference frameworks based on Active Inference Institute resources.
"""

from typing import Any
import importlib
import logging
import numpy as np

from geo_infer_act.utils.config import get_config_value

try:
    import geo_infer_space as space_h3
except ImportError:
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
        self.config = config or {}
        self.available_tools = self._check_available_tools()
        logger.info(f"Available tools: {list(self.available_tools.keys())}")

    def _execute_dynamic_source(
        self, source: str, namespace: dict[str, Any], description: str
    ) -> dict[str, Any]:
        """Execute optional model source only after explicit caller opt-in."""
        if not self.config.get("allow_dynamic_code", False):
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
        self, model_spec: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Create and run RxInfer model for constrained Bayesian inference.

        Args:
            model_spec: Julia model specification string
            data: Data for inference

        Returns:
            Inference results
        """
        if not self.available_tools.get("rxinfer", False):
            if not self.config.get("allow_local_fallback", False):
                raise RuntimeError(
                    "RxInfer not available. Please install Julia, PyJulia, and RxInfer"
                )
            observations = np.asarray(data.get("observations", []), dtype=float)
            if observations.size == 0 or not np.isfinite(observations).all():
                raise ValueError("RxInfer data must contain finite observations")
            return {
                "status": "success",
                "backend": "deterministic-local",
                "posterior_marginals": {
                    "mean": float(np.mean(observations)),
                    "variance": float(np.var(observations)),
                },
                "iterations": int(observations.size),
                "tool": "rxinfer-compatible",
            }

        try:
            import julia

            j = julia.Julia(compiled_modules=False)
            j.eval("using RxInfer, Rocket, GraphPPL")

            # Execute model specification
            j.eval(model_spec)

            # Prepare data
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    j.eval(f"{key} = {value.tolist()}")
                else:
                    j.eval(f"{key} = {value}")

            # Run inference
            inference_code = """
            result = infer(
                model = model,
                data = (y = observations,),
                iterations = 100,
                options = (
                    schedule = :parallel,
                    addons = AddonLogScale()
                )
            )
            """
            j.eval(inference_code)

            # Extract results
            posterior_marginals = j.eval("result.posteriors")
            model_evidence = j.eval("result.free_energy")

            return {
                "status": "success",
                "posterior_marginals": posterior_marginals,
                "model_evidence": float(model_evidence),
                "tool": "rxinfer",
            }

        except Exception as e:
            logger.error(f"RxInfer integration failed: {e}")
            return {"status": "error", "message": str(e), "tool": "rxinfer"}

    def create_bayeux_model(
        self,
        log_density_fn: str,
        test_point: dict[str, Any],
        transform_fn: str | None = None,
    ) -> dict[str, Any]:
        """
        Create and optimize Bayeux model for scalable inference.

        Args:
            log_density_fn: Python function string for log density
            test_point: Test point for model validation
            transform_fn: Optional transformation function

        Returns:
            Optimization results
        """
        if not self.available_tools.get("bayeux", False):
            raise RuntimeError(
                "Bayeux not available. Please install: uv pip install bayeux-ml"
            )

        try:
            import bayeux as bx

            # Execute user-provided sources in a per-call namespace.  Dynamic
            # model code is opt-in through ``allow_dynamic_code``.
            source_namespace = self._execute_dynamic_source(
                log_density_fn,
                {"__name__": "geo_infer_act.dynamic.bayeux"},
                "Bayeux log-density source",
            )
            log_density = source_namespace.get("log_density")

            if transform_fn:
                source_namespace = self._execute_dynamic_source(
                    transform_fn,
                    source_namespace,
                    "Bayeux transform source",
                )
                transform_function = source_namespace.get("transform_fn")
            else:
                transform_function = None

            if not callable(log_density):
                raise ValueError(
                    "Log-density source must define callable 'log_density'"
                )
            if transform_fn and not callable(transform_function):
                raise ValueError("Transform source must define callable 'transform_fn'")

            # Create Bayeux model
            model = bx.Model(
                log_density=log_density,
                test_point=test_point,
                transform_fn=transform_function,
            )

            # Optimize using different methods
            methods = ["optax_adam", "nuts"]
            results = {}

            for method in methods:
                try:
                    if method == "optax_adam":
                        result = model.optimize.optax_adam(seed=42, num_iters=1000)
                    elif method == "nuts":
                        result = model.mcmc.nuts(
                            seed=42, num_samples=1000, num_chains=4
                        )

                    results[method] = {
                        "params": (
                            result.params if hasattr(result, "params") else result
                        ),
                        "success": True,
                    }

                except Exception as e:
                    results[method] = {"error": str(e), "success": False}

            # Report per-method results; the top-level status reflects whether
            # each requested method succeeded, not just that the fan-out ran.
            all_ok = bool(results) and all(
                r.get("success", False) for r in results.values()
            )
            return {
                "status": "success" if all_ok else "partial",
                "results": results,
                "tool": "bayeux",
            }

        except Exception as e:
            logger.error(f"Bayeux integration failed: {e}")
            return {"status": "error", "message": str(e), "tool": "bayeux"}

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
    """Integrate with RxInfer for scalable nested inference."""
    integration_config = dict(config or {})
    integration_config.setdefault("allow_local_fallback", True)
    integration_hub = ModernToolsIntegration(integration_config)

    # Default RxInfer model for spatial inference
    default_model = """
    @model function spatial_active_inference(n_states, n_obs)
        # Define priors
        μ ~ NormalMeanVariance(0.0, 1.0)
        τ ~ Gamma(1.0, 1.0)

        # State transitions with spatial structure
        x = Vector{Random.Variable}(undef, n_states)
        for i in 1:n_states
            if i == 1
                x[i] ~ NormalMeanPrecision(μ, τ)
            else
                x[i] ~ NormalMeanPrecision(x[i-1], τ)  # Spatial continuity
            end
        end

        # Observations
        y = Vector{Random.Variable}(undef, n_obs)
        for i in 1:n_obs
            state_idx = min(i, n_states)
            y[i] ~ NormalMeanPrecision(x[state_idx], 1.0)
        end
    end

    model = spatial_active_inference
    """

    model_spec = model_params.get("model_specification", default_model)
    data = model_params.get(
        "data",
        {
            "observations": np.random.default_rng(config.get("random_seed", 0)).normal(
                size=10
            )
        },
    )

    return integration_hub.create_rxinfer_model(model_spec, data)


def integrate_bayeux(
    config: dict[str, Any], model_params: dict[str, Any]
) -> dict[str, Any]:
    """Integrate with Bayeux for JAX-based scalable inference."""
    integration_hub = ModernToolsIntegration(config)

    # Default log density for spatial model
    default_log_density = """
def log_density(params):
    import jax.numpy as jnp

    # Spatial prior
    spatial_prior = -0.5 * jnp.sum(params['location']**2)

    # Observation likelihood
    observations = jnp.array([1.0, 2.0, 1.5])
    predicted = params['location'][0] + params['scale'] * jnp.array([0, 1, 0.5])
    likelihood = -0.5 * jnp.sum((observations - predicted)**2)

    return spatial_prior + likelihood
"""

    default_transform = """
def transform_fn(params):
    import jax.numpy as jnp
    return {
        'location': params['location'],
        'scale': jnp.exp(params['scale_log'])  # Ensure positive scale
    }
"""

    log_density_fn = model_params.get("log_density", default_log_density)
    test_point = model_params.get(
        "test_point", {"location": np.zeros(2), "scale_log": 0.0}
    )
    transform_fn = model_params.get("transform_fn", default_transform)

    return integration_hub.create_bayeux_model(log_density_fn, test_point, transform_fn)


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
        boundary: GeoJSON boundary specification. Accepts:
            - GeoJSON Polygon or MultiPolygon geometries
            - GeoJSON Feature or FeatureCollection containing those geometries
            - Legacy coordinate-only Polygon dictionaries

    Returns:
        H3 spatial model configuration
    """
    try:
        settings = config or {}
        max_cells = int(settings.get("max_cells", 100_000))
        if max_cells < 1:
            raise ValueError("max_cells must be at least 1")
        from geo_infer_act.utils.h3_adapter import get_h3_adapter

        adapter = get_h3_adapter()

        if not 0 <= h3_resolution <= 15:
            raise ValueError("h3_resolution must be between 0 and 15")
        if hasattr(boundary, "__geo_interface__"):
            boundary = boundary.__geo_interface__
        if not isinstance(boundary, dict):
            raise ValueError("boundary must be a GeoJSON-like mapping")

        # Preserve the historical coordinate-only Polygon form, but normalize
        # every accepted boundary through the same native H3 adapter. Invalid,
        # empty, point, or malformed inputs must fail closed rather than
        # silently generating an unrelated San Francisco grid.
        boundary_type = boundary.get("type")
        if boundary_type is None and "coordinates" in boundary:
            boundary = {"type": "Polygon", "coordinates": boundary["coordinates"]}
            boundary_type = "Polygon"
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
