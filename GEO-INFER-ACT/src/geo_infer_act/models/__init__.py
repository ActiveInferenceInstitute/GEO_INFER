"""
Data models and structures for active inference.

This module provides data structures and model specifications
for active inference applications in various domains.
"""

from importlib import import_module as _import_module

_EXPORTS = {
    "BaseActiveInferenceModel": (
        "geo_infer_act.models.base",
        "BaseActiveInferenceModel",
    ),
    "CategoricalModel": ("geo_infer_act.models.base", "CategoricalModel"),
    "GaussianModel": ("geo_infer_act.models.base", "GaussianModel"),
    "EcologicalModel": ("geo_infer_act.models.ecological", "EcologicalModel"),
    "UrbanModel": ("geo_infer_act.models.urban", "UrbanModel"),
    "ClimateModel": ("geo_infer_act.models.climate", "ClimateModel"),
    "ResourceModel": ("geo_infer_act.models.resource", "ResourceModel"),
    "MultiAgentModel": ("geo_infer_act.models.multi_agent", "MultiAgentModel"),
    "ContinuousPOMDPActiveInference": (
        "geo_infer_act.models.continuous_pomdp",
        "ContinuousPOMDPActiveInference",
    ),
}
_SUBMODULES = {
    "base": "geo_infer_act.models.base",
    "ecological": "geo_infer_act.models.ecological",
    "urban": "geo_infer_act.models.urban",
    "climate": "geo_infer_act.models.climate",
    "resource": "geo_infer_act.models.resource",
    "multi_agent": "geo_infer_act.models.multi_agent",
    "continuous_pomdp": "geo_infer_act.models.continuous_pomdp",
}

__all__ = [
    "BaseActiveInferenceModel",
    "CategoricalModel",
    "GaussianModel",
    "EcologicalModel",
    "UrbanModel",
    "ClimateModel",
    "ResourceModel",
    "MultiAgentModel",
    "ContinuousPOMDPActiveInference",
]


def __getattr__(name: str):
    """Load existing public exports only when requested."""
    if name in _EXPORTS:
        module, member = _EXPORTS[name]
        value = getattr(_import_module(module, __name__), member)
    elif name in _SUBMODULES:
        value = _import_module(_SUBMODULES[name])
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))
