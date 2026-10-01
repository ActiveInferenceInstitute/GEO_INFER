"""
API endpoints for supply chain functionality in GEO-INFER-LOG.

This module provides FastAPI endpoints for supply chain modeling,
resilience analysis, network optimization, and facility location.
"""

from functools import lru_cache

from fastapi import APIRouter, HTTPException, Depends
from pydantic import ConfigDict, Field
from geo_infer_log.models.base import BaseModel

from geo_infer_log.models.schemas import SupplyChainNetwork
from geo_infer_log.core.supply_chain import (
    SupplyChainModel,
    ResilienceAnalyzer,
    NetworkOptimizer,
    FacilityLocator,
)


router = APIRouter(
    prefix="/supply-chain",
    tags=["supply-chain"],
    responses={404: {"description": "Not found"}},
)


class NetworkRequest(BaseModel):
    """Request model for supply chain network operations."""

    network: SupplyChainNetwork

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "network": {
                    "id": "network-001",
                    "name": "European Distribution Network",
                    "facilities": [
                        {
                            "id": "dc-001",
                            "name": "Berlin Distribution Center",
                            "location": [13.4050, 52.5200],
                            "type": "distribution_center",
                            "capacity": 5000,
                            "operating_cost": 10000,
                        },
                        {
                            "id": "wh-001",
                            "name": "Munich Warehouse",
                            "location": [11.5820, 48.1351],
                            "type": "warehouse",
                            "capacity": 3000,
                            "operating_cost": 8000,
                        },
                    ],
                    "links": [
                        {
                            "from": "dc-001",
                            "to": "wh-001",
                            "distance": 504,
                            "time": 300,
                            "cost": 600,
                            "capacity": 1000,
                        }
                    ],
                }
            }
        }
    )


class FlowOptimizationRequest(BaseModel):
    """Request model for supply chain flow optimization."""

    network_id: str
    demand_points: list[dict]
    supply_points: list[dict]
    objective: str = "cost"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "network_id": "network-001",
                "demand_points": [
                    {
                        "id": "dp-001",
                        "location": [8.6821, 50.1109],
                        "demand": 200,
                        "priority": 1,
                    }
                ],
                "supply_points": [
                    {
                        "id": "sp-001",
                        "location": [18.0686, 59.3293],
                        "supply": 500,
                        "reliability": 0.95,
                    }
                ],
                "objective": "cost",
            }
        }
    )


class DisruptionAnalysisRequest(BaseModel):
    """Request model for supply chain disruption analysis."""

    network_id: str
    disrupted_nodes: list[str] = Field(default_factory=list)
    disrupted_edges: list[tuple[str, str]] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "network_id": "network-001",
                "disrupted_nodes": ["wh-001"],
                "disrupted_edges": [["dc-001", "wh-002"]],
            }
        }
    )


class FacilityLocationRequest(BaseModel):
    """Request model for facility location optimization."""

    candidates: list[dict]
    demand_points: list[dict]
    num_facilities: int
    max_distance: float | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "candidates": [
                    {"id": "c1", "location": [13.4050, 52.5200], "cost": 10000},
                    {"id": "c2", "location": [11.5820, 48.1351], "cost": 8000},
                ],
                "demand_points": [
                    {"id": "d1", "location": [8.6821, 50.1109], "demand": 200},
                    {"id": "d2", "location": [9.9937, 53.5511], "demand": 300},
                ],
                "num_facilities": 1,
                "max_distance": 500,
            }
        }
    )


class NetworkOptimizationRequest(BaseModel):
    """Request model for network design optimization."""

    locations: list[dict]
    demand_points: list[dict]
    constraints: dict

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "locations": [
                    {"id": "loc1", "location": [13.4050, 52.5200], "cost": 10000},
                    {"id": "loc2", "location": [11.5820, 48.1351], "cost": 8000},
                ],
                "demand_points": [
                    {"id": "d1", "location": [8.6821, 50.1109], "demand": 200},
                    {"id": "d2", "location": [9.9937, 53.5511], "demand": 300},
                ],
                "constraints": {
                    "max_facilities": 3,
                    "max_distance": 500,
                    "budget": 30000,
                },
            }
        }
    )


# Cached module-level singletons: built networks and analysis state must
# survive across requests for the API to behave correctly.
@lru_cache(maxsize=1)
def get_supply_chain_model() -> SupplyChainModel:
    """Dependency for supply chain model."""
    return SupplyChainModel()


@lru_cache(maxsize=1)
def get_resilience_analyzer(
    model: SupplyChainModel = Depends(get_supply_chain_model),
) -> ResilienceAnalyzer:
    """Dependency for resilience analyzer."""
    return ResilienceAnalyzer(model)


@lru_cache(maxsize=1)
def get_network_optimizer() -> NetworkOptimizer:
    """Dependency for network optimizer."""
    return NetworkOptimizer()


@lru_cache(maxsize=1)
def get_facility_locator() -> FacilityLocator:
    """Dependency for facility locator."""
    return FacilityLocator()


@router.post("/networks", response_model=dict)
async def create_network(
    request: NetworkRequest, model: SupplyChainModel = Depends(get_supply_chain_model)
) -> dict:
    """Create a supply chain network."""
    try:
        model.load_network(request.network)
        return {"status": "success", "message": f"Network {request.network.id} created"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/flow", response_model=dict)
async def optimize_flow(
    request: FlowOptimizationRequest,
    model: SupplyChainModel = Depends(get_supply_chain_model),
) -> dict:
    """Optimize flow in a supply chain network."""
    try:
        result = model.optimize_flow(
            demand_points=request.demand_points,
            supply_points=request.supply_points,
            objective=request.objective,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/resilience/disruption", response_model=dict)
async def analyze_disruption(
    request: DisruptionAnalysisRequest,
    analyzer: ResilienceAnalyzer = Depends(get_resilience_analyzer),
) -> dict:
    """Analyze the impact of a disruption in the supply chain."""
    try:
        result = analyzer.simulate_disruption(
            disrupted_nodes=request.disrupted_nodes,
            disrupted_edges=request.disrupted_edges,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/resilience/critical-nodes", response_model=list[str])
async def get_critical_nodes(
    analyzer: ResilienceAnalyzer = Depends(get_resilience_analyzer),
) -> list[str]:
    """Identify critical nodes in the supply chain network."""
    try:
        return analyzer.identify_critical_nodes()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/resilience/improvements", response_model=list[dict])
async def get_improvement_suggestions(
    analyzer: ResilienceAnalyzer = Depends(get_resilience_analyzer),
) -> list[dict]:
    """Get improvement suggestions for supply chain resilience."""
    try:
        return analyzer.suggest_improvements()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/facility-location", response_model=list[dict])
async def optimize_facility_locations(
    request: FacilityLocationRequest,
    locator: FacilityLocator = Depends(get_facility_locator),
) -> list[dict]:
    """Optimize facility locations."""
    try:
        result = locator.locate_facilities(
            candidates=request.candidates,
            demand_points=request.demand_points,
            num_facilities=request.num_facilities,
            max_distance=request.max_distance,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/network-optimization", response_model=dict)
async def optimize_network(
    request: NetworkOptimizationRequest,
    optimizer: NetworkOptimizer = Depends(get_network_optimizer),
) -> dict:
    """Optimize supply chain network design."""
    try:
        result = optimizer.optimize_network(
            locations=request.locations,
            demand_points=request.demand_points,
            constraints=request.constraints,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
