"""
Processing algorithm registry endpoints for the GEO-INFER-API.

Exposes the GeoLibre-style processing algorithm registry
(:mod:`geo_infer_space.core.algorithm_registry`) as a read-only REST surface:

- ``GET /api/v1/algorithms`` — list registered algorithms with their declared
  parameters.
- ``POST /api/v1/algorithms/{algorithm_id}/run`` — run one algorithm against a
  set of layers and parameters.

The registry is imported gracefully: when the full uv workspace is synced,
``geo_infer_space`` provides the reference registry; otherwise the endpoints
report the algorithms service as unavailable instead of failing at import
time. This keeps GEO-INFER-API importable without the SPACE dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from geo_infer_api.core.exceptions import NotFoundError

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from geo_infer_space.core.algorithm_registry import (
        AlgorithmRegistry,
        ProcessingAlgorithm,
        ProcessingContext,
    )

try:  # pragma: no cover - workspace availability is environment dependent
    from geo_infer_space.core.algorithm_registry import (
        AlgorithmRegistry,
        ProcessingAlgorithm,
        ProcessingContext,
        build_reference_registry,
    )

    _REGISTRY: AlgorithmRegistry | None = build_reference_registry()
    HAS_ALGORITHM_REGISTRY: bool = True
except ImportError:  # pragma: no cover - exercised when geo_infer_space is absent
    _REGISTRY = None
    HAS_ALGORITHM_REGISTRY = False

router = APIRouter()


class AlgorithmParameterOut(BaseModel):
    """Declared parameter for a registered algorithm."""

    id: str
    label: str
    required: bool = False
    default: Any | None = None
    type: str = "auto"


class AlgorithmOut(BaseModel):
    """A registered processing algorithm, as seen by API clients."""

    id: str
    name: str
    description: str
    parameters: list[AlgorithmParameterOut] = Field(default_factory=list)


class AlgorithmRunRequest(BaseModel):
    """Request body for running a registered algorithm."""

    layers: list[dict[str, Any]] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


class AlgorithmRunResponse(BaseModel):
    """Result of running a registered algorithm."""

    algorithm_id: str
    result: Any | None = None
    logs: list[str] = Field(default_factory=list)


def _algorithm_out(algorithm: ProcessingAlgorithm) -> AlgorithmOut:
    """Convert a registry algorithm into its API representation."""
    return AlgorithmOut(
        id=algorithm.id,
        name=algorithm.name,
        description=algorithm.description,
        parameters=[
            AlgorithmParameterOut(
                id=p.id,
                label=p.label,
                required=p.required,
                default=p.default,
                type=p.type,
            )
            for p in algorithm.parameters
        ],
    )


def _require_registry() -> AlgorithmRegistry:
    """Return the registry or raise a service-unavailable HTTP error."""
    if not HAS_ALGORITHM_REGISTRY or _REGISTRY is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="processing algorithm registry unavailable: "
            "geo_infer_space is not importable in this environment",
        )
    return _REGISTRY


@router.get("/algorithms", summary="List registered processing algorithms")
async def list_algorithms() -> dict[str, Any]:
    """List all registered processing algorithms and their parameters."""
    registry = _require_registry()
    algorithms = [_algorithm_out(a) for a in registry.list()]
    return {"count": len(algorithms), "algorithms": algorithms}


@router.get("/algorithms/{algorithm_id}", summary="Describe one algorithm")
async def get_algorithm(algorithm_id: str) -> AlgorithmOut:
    """Return the declaration for a single registered algorithm."""
    registry = _require_registry()
    try:
        algorithm = registry.get(algorithm_id)
    except KeyError:
        raise NotFoundError(f"unknown algorithm: {algorithm_id}")
    return _algorithm_out(algorithm)


@router.post(
    "/algorithms/{algorithm_id}/run",
    summary="Run a registered processing algorithm",
    status_code=status.HTTP_200_OK,
)
async def run_algorithm(
    algorithm_id: str,
    request: AlgorithmRunRequest,
) -> AlgorithmRunResponse:
    """Run one registered algorithm against the supplied layers/parameters."""
    registry = _require_registry()
    try:
        registry.get(algorithm_id)
    except KeyError:
        raise NotFoundError(f"unknown algorithm: {algorithm_id}")

    # ProcessingContext is bound by the import above; _require_registry()
    # raises 503 first when the import failed, so this is safe at runtime.
    context = ProcessingContext(
        layers=request.layers,
        parameters=request.parameters,
    )
    result = registry.run(algorithm_id, context)
    return AlgorithmRunResponse(
        algorithm_id=algorithm_id,
        result=result,
        logs=list(context.logs),
    )
