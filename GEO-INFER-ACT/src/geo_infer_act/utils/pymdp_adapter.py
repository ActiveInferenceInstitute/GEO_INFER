"""pymdp 1.0.3 runtime adapter for GEO-INFER-ACT.

This module is the only production bridge from GEO-INFER active-inference
runtime code into inferactively-pymdp. It targets the JAX-first 1.0.3 API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import Any
import warnings

import numpy as np

EXPECTED_PYMDP_VERSION = "1.0.3"


@dataclass
class PymdpStepResult:
    """Normalized diagnostics from one pymdp Agent perception-action step."""

    beliefs: np.ndarray
    policy_posterior: np.ndarray
    negative_expected_free_energy: np.ndarray
    selected_action_index: int
    free_energy: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-safe pymdp backend metadata."""
        return {
            "backend": "inferactively-pymdp",
            "pymdp_version": self.metadata.get("pymdp_version", EXPECTED_PYMDP_VERSION),
            "h3_version": self.metadata.get("h3_version"),
            "selected_action_index": int(self.selected_action_index),
            "action_posterior": self.policy_posterior.astype(float).tolist(),
            "negative_expected_free_energy": self.negative_expected_free_energy.astype(
                float
            ).tolist(),
            "free_energy": float(self.free_energy),
            **{
                key: value
                for key, value in self.metadata.items()
                if key not in {"pymdp_version", "h3_version"}
            },
        }


def installed_pymdp_version() -> str:
    """Return the installed inferactively-pymdp distribution version."""
    return metadata.version("inferactively-pymdp")


def validate_pymdp_version(expected: str = EXPECTED_PYMDP_VERSION) -> str:
    """Fail unless the exact supported inferactively-pymdp version is installed."""
    version = installed_pymdp_version()
    if version != expected:
        raise RuntimeError(
            f"inferactively-pymdp {expected} is required; installed version is {version}"
        )
    return version


def real_h3_version_metadata() -> dict[str, Any]:
    """Return h3-py runtime version metadata and fail if h3 is unavailable."""
    import h3  # noqa: PLC0415

    versions = h3.versions()
    return {
        "h3_version": versions.get("python"),
        "h3_c_version": versions.get("c"),
    }


def _normalize_distribution(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError("pymdp distributions must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("pymdp distributions must contain finite values")
    array = np.maximum(array, 0.0)
    total = float(np.sum(array))
    if total <= 1e-12:
        return np.ones_like(array) / array.size
    return array / total


def _normalize_likelihood(matrix: Any) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Observation model must be 2D, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Observation model must contain finite values")
    array = np.maximum(array, 1e-12)
    col_sums = np.sum(array, axis=0, keepdims=True)
    res = array / np.maximum(col_sums, 1e-12)
    return np.asarray(res, dtype=float)


def _normalize_transition(tensor: Any, state_dim: int, action_count: int) -> np.ndarray:
    array = np.asarray(tensor, dtype=float)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], action_count, axis=2)
    elif array.ndim != 3:
        raise ValueError(f"Transition model must be 2D or 3D, got shape {array.shape}")
    if array.shape[0] != state_dim or array.shape[1] != state_dim:
        array = np.repeat(np.eye(state_dim)[:, :, None], action_count, axis=2)
    if array.shape[2] != action_count:
        if array.shape[2] > action_count:
            array = array[:, :, :action_count]
        else:
            repeats = [array[:, :, idx % array.shape[2]] for idx in range(action_count)]
            array = np.stack(repeats, axis=2)
    if not np.all(np.isfinite(array)):
        raise ValueError("Transition model must contain finite values")
    array = np.maximum(array, 1e-12)
    sums = np.sum(array, axis=0, keepdims=True)
    res = array / np.maximum(sums, 1e-12)
    return np.asarray(res, dtype=float)


def _preferences_vector(values: Any, obs_dim: int) -> np.ndarray:
    if isinstance(values, dict):
        values = values.get("observations", np.ones(obs_dim) / obs_dim)
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != obs_dim:
        vector = np.resize(vector, obs_dim)
    if not np.all(np.isfinite(vector)):
        vector = np.zeros(obs_dim, dtype=float)
    return vector.astype(float)


def _belief_vector(values: Any, state_dim: int) -> np.ndarray:
    if isinstance(values, dict):
        values = values.get("states", np.ones(state_dim) / state_dim)
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size != state_dim:
        vector = np.ones(state_dim, dtype=float) / state_dim
    return _normalize_distribution(vector)


def _coerce_action_count(value: Any, default: int = 3) -> int:
    """Return a positive action count from scalar or pymdp-style values."""
    if value is None:
        value = default
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("action_count must be a scalar or a single-item sequence")
    count = int(array.reshape(-1)[0])
    if count < 1:
        raise ValueError("action_count must be positive")
    return count


def _model_num_controls(model: Any, default: int = 3) -> int:
    """Read and normalize a model's pymdp-style control-count value."""
    value = getattr(model, "num_controls", None)
    if value is None:
        value = getattr(model, "parameters", {}).get("num_controls", default)
    return _coerce_action_count(value, default=default)


def _extract_agent_belief(qs: list[Any]) -> np.ndarray:
    if not qs:
        raise ValueError("pymdp returned no posterior states")
    array = np.asarray(qs[0], dtype=float)
    if array.ndim >= 1:
        array = array.reshape(-1, array.shape[-1])[-1]
    return _normalize_distribution(array)


def _scalar_free_energy(info: dict[str, Any]) -> float:
    for key in ("vfe", "free_energy", "F"):
        if key in info:
            array = np.asarray(info[key], dtype=float)
            finite = array[np.isfinite(array)]
            if finite.size:
                return float(np.mean(finite))
    components = info.get("vfe_components")
    if isinstance(components, dict):
        values = []
        for value in components.values():
            array = np.asarray(value, dtype=float)
            values.extend(array[np.isfinite(array)].reshape(-1).tolist())
        if values:
            return float(np.mean(values))
    return 0.0


def run_pymdp_step(
    *,
    observation: Any,
    observation_model: Any,
    transition_model: Any,
    preferences: Any,
    prior: Any,
    action_count: int = 3,
    random_seed: int = 0,
    action_selection: str = "deterministic",
    policy_prior: Any = None,
    strict: bool = False,
    posterior: Any = None,
    perception_free_energy: float | None = None,
) -> PymdpStepResult:
    """Run pymdp perception/policy inference, optionally rejecting all repairs.

    ``strict`` preserves valid matrix zeros and rejects mismatched dimensions.
    ``policy_prior`` is E over one-step policies, in action index order.
    ``posterior`` supplies already-conditioned beliefs for policy-only inference.
    In that mode ``perception_free_energy`` must carry the original perception
    diagnostic; the observation is not assimilated again and B is used only to
    evaluate future policies.
    """
    if strict and (
        isinstance(action_count, bool)
        or not isinstance(action_count, (int, np.integer))
    ):
        raise ValueError("Strict action_count must be a positive integer")
    action_count = _coerce_action_count(action_count)
    version = validate_pymdp_version()
    h3_versions = real_h3_version_metadata()

    import jax.numpy as jnp  # noqa: PLC0415
    import jax.random as jr  # noqa: PLC0415
    from pymdp.agent import Agent  # noqa: PLC0415

    if strict:
        obs_model = np.asarray(observation_model, dtype=float)
        if obs_model.ndim != 2:
            raise ValueError("Strict likelihood must be two-dimensional")
        obs_dim, state_dim = obs_model.shape
        trans_model = np.asarray(transition_model, dtype=float)
        preference_vector = np.asarray(preferences, dtype=float)
        prior_vector = np.asarray(prior, dtype=float)
        observation_vector = np.asarray(observation, dtype=float)
        for name, array, shape in (
            ("A", obs_model, (obs_dim, state_dim)),
            ("B", trans_model, (state_dim, state_dim, action_count)),
            ("C", preference_vector, (obs_dim,)),
            ("D", prior_vector, (state_dim,)),
            ("observation", observation_vector, (obs_dim,)),
        ):
            if (
                array.shape != shape
                or array.size == 0
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(
                    f"Strict {name} requires finite values with shape {shape}"
                )
            if name != "C" and (
                np.any(array < 0)
                or not np.allclose(array.sum(axis=0), 1, rtol=0, atol=1e-8)
            ):
                raise ValueError(
                    f"Strict {name} requires normalized nonnegative probabilities"
                )
    else:
        observation_vector = _normalize_distribution(observation)
        obs_model = _normalize_likelihood(observation_model)
        obs_dim, state_dim = obs_model.shape
        if observation_vector.size != obs_dim:
            if observation_vector.size == state_dim or obs_dim == 1:
                state_dim = int(observation_vector.size)
                obs_dim = int(observation_vector.size)
                obs_model = np.eye(state_dim, dtype=float)
            else:
                raise ValueError(
                    "Observation dimension does not match pymdp observation model: "
                    f"{observation_vector.size} != {obs_dim}"
                )
        trans_model = _normalize_transition(transition_model, state_dim, action_count)
        preference_vector = _preferences_vector(preferences, obs_dim)
        prior_vector = _belief_vector(prior, state_dim)

    policy_vector = None
    if policy_prior is not None:
        policy_vector = np.asarray(policy_prior, dtype=float)
        if (
            policy_vector.shape != (action_count,)
            or not np.all(np.isfinite(policy_vector))
            or np.any(policy_vector < 0)
            or not np.isclose(policy_vector.sum(), 1, rtol=0, atol=1e-8)
        ):
            raise ValueError("Policy prior E must be a normalized action-count vector")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="A JAX array is being set as static!*",
            category=UserWarning,
        )
        agent = Agent(
            A=[jnp.asarray(obs_model)],
            B=[jnp.asarray(trans_model)],
            C=[jnp.asarray(preference_vector)],
            D=[jnp.asarray(prior_vector)],
            E=None if policy_vector is None else jnp.asarray(policy_vector),
            num_controls=[action_count],
            categorical_obs=True,
            batch_size=1,
            policy_len=1,
            action_selection=action_selection,
            sampling_mode="marginal",
        )
    if posterior is None:
        if perception_free_energy is not None:
            raise ValueError("perception_free_energy requires an explicit posterior")
        qs, info = agent.infer_states(
            [jnp.asarray(observation_vector.reshape(1, -1))],
            empirical_prior=agent.D,
            return_info=True,
        )
        free_energy = _scalar_free_energy(info)
    else:
        posterior_vector = np.asarray(posterior, dtype=float)
        if (
            posterior_vector.shape != (state_dim,)
            or not np.all(np.isfinite(posterior_vector))
            or np.any(posterior_vector < 0)
            or not np.isclose(posterior_vector.sum(), 1, rtol=0, atol=1e-8)
        ):
            raise ValueError("Policy-only posterior must be a normalized state vector")
        if perception_free_energy is None or not np.isfinite(perception_free_energy):
            raise ValueError(
                "Policy-only inference requires finite perception_free_energy"
            )
        qs = [jnp.asarray(posterior_vector.reshape(1, 1, -1))]
        free_energy = float(perception_free_energy)
    q_pi, neg_efe = agent.infer_policies(qs)
    rng_key = jr.split(jr.PRNGKey(int(random_seed)), agent.batch_size)
    action = agent.sample_action(q_pi, rng_key=rng_key)

    q_pi_np = np.asarray(q_pi, dtype=float).reshape(-1)
    neg_efe_np = np.asarray(neg_efe, dtype=float).reshape(-1)
    selected = int(np.asarray(action).reshape(-1)[0])
    selected %= max(1, len(q_pi_np))

    return PymdpStepResult(
        beliefs=_extract_agent_belief(qs),
        policy_posterior=_normalize_distribution(q_pi_np),
        negative_expected_free_energy=neg_efe_np,
        selected_action_index=selected,
        free_energy=free_energy,
        metadata={
            "pymdp_version": version,
            **h3_versions,
            "action_count": action_count,
            "categorical_obs": True,
            "policy_len": 1,
            "batch_size": 1,
            "inference_mode": "perception_policy"
            if posterior is None
            else "policy_only",
        },
    )


def run_model_step(
    model: Any,
    observation: Any,
    *,
    action_count: int | None = None,
    random_seed: int = 0,
    prior: Any = None,
    posterior: Any = None,
    perception_free_energy: float | None = None,
) -> PymdpStepResult:
    """Run pymdp inference for a GEO-INFER categorical GenerativeModel."""
    if getattr(model, "model_type", None) != "categorical":
        raise ValueError("pymdp adapter currently supports categorical models")
    controls = _coerce_action_count(
        action_count if action_count is not None else _model_num_controls(model)
    )
    return run_pymdp_step(
        observation=observation,
        observation_model=model.observation_model,
        transition_model=model.transition_model,
        preferences=getattr(model, "preferences", None),
        prior=prior if prior is not None else getattr(model, "beliefs", None),
        action_count=controls,
        random_seed=random_seed,
        posterior=posterior,
        perception_free_energy=perception_free_energy,
    )


def obj_array_zeros(shape_list: list[Any]) -> Any:
    """Zero-init pymdp-style matrix list under legacy ``obj_array_zeros`` semantics.

    pymdp 1.0.3 renamed this helper to ``list_array_zeros`` and returns immutable
    JAX arrays. GEO-INFER matrix builders mutate entries after allocation, so
    this facade returns mutable numpy equivalents with identical shapes.
    """

    def _as_shape(dim: Any) -> tuple:
        return (dim,) if isinstance(dim, int) else tuple(dim)

    return [np.zeros(_as_shape(shape)) for shape in shape_list]


def jax_prng_key(seed: int) -> Any:
    """Bridge facade for ``jax.random.PRNGKey``."""
    import jax.random as jr  # noqa: PLC0415

    return jr.PRNGKey(seed)


def random_A_array(rng_key: Any, num_obs: list[int], num_states: list[int]) -> Any:
    """Bridge facade for ``pymdp.utils.random_A_array`` with version validation."""
    validate_pymdp_version()
    from pymdp.utils import random_A_array as _pymdp_random_A_array  # noqa: PLC0415

    return _pymdp_random_A_array(rng_key, num_obs, num_states)


def random_B_array(rng_key: Any, num_states: list[int], num_controls: list[int]) -> Any:
    """Bridge facade for ``pymdp.utils.random_B_array`` with version validation."""
    validate_pymdp_version()
    from pymdp.utils import random_B_array as _pymdp_random_B_array  # noqa: PLC0415

    return _pymdp_random_B_array(rng_key, num_states, num_controls)


def onehot_batch(num_obs: list[int], indices: list[Any]) -> list[Any]:
    """Return one-hot (1, dim) JAX rows for each modality, pymdp batch style."""
    import jax.numpy as jnp  # noqa: PLC0415

    return [
        jnp.eye(num_obs[i])[int(indices[i])].reshape(1, -1) for i in range(len(num_obs))
    ]


def build_agent(
    *,
    A: Any,
    B: Any,
    C: Any,
    D: Any,
    num_controls: list[int],
    categorical_obs: bool = True,
    batch_size: int = 1,
) -> Any:
    """Build a JAX ``pymdp.agent.Agent`` under the adapter's version contract."""
    validate_pymdp_version()
    import jax.numpy as jnp  # noqa: PLC0415
    from pymdp.agent import Agent  # noqa: PLC0415

    return Agent(
        A=A,
        B=B,
        C=[jnp.asarray(c) for c in C],
        D=[jnp.asarray(d) for d in D],
        num_controls=num_controls,
        categorical_obs=categorical_obs,
        batch_size=batch_size,
    )
