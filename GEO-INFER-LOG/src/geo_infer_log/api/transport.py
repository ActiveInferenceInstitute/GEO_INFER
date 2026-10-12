"""
API endpoints for multimodal transportation planning in GEO-INFER-LOG.

This module provides FastAPI endpoints for multimodal transportation planning,
transportation network analysis, and emissions calculation.
"""

import pickle
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Any
from pydantic import ConfigDict, Field
from geo_infer_log.models.base import BaseModel

from geo_infer_log.models.schemas import Vehicle, Route
from geo_infer_log.core.transport import (
    MultiModalPlanner,
    TransportationNetworkAnalyzer,
    TrafficSimulator,
    EmissionsCalculator,
)


router = APIRouter(
    prefix="/transport",
    tags=["transport"],
    responses={404: {"description": "Not found"}},
)


class RouteRequest(BaseModel):
    """Request model for multimodal route planning."""

    origin: tuple[float, float] = Field(..., description="(lon, lat) of origin")
    destination: tuple[float, float] = Field(
        ..., description="(lon, lat) of destination"
    )
    allowed_modes: list[str] = Field(
        ..., description="List of allowed transportation modes"
    )
    preferences: dict | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "origin": [13.404954, 52.520008],  # Berlin
                "destination": [11.576124, 48.137154],  # Munich
                "allowed_modes": ["car", "train", "bus"],
                "preferences": {
                    "cost_weight": 1.0,
                    "time_weight": 1.5,
                    "emissions_weight": 2.0,
                    "transfers_weight": 0.5,
                },
            }
        }
    )


class CompareRoutesRequest(BaseModel):
    """Request model for route comparison."""

    origin: tuple[float, float] = Field(..., description="(lon, lat) of origin")
    destination: tuple[float, float] = Field(
        ..., description="(lon, lat) of destination"
    )
    mode_combinations: list[list[str]] = Field(
        ..., description="List of mode combinations to compare"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "origin": [13.404954, 52.520008],  # Berlin
                "destination": [11.576124, 48.137154],  # Munich
                "mode_combinations": [["car"], ["train"], ["car", "train"]],
            }
        }
    )


class NetworkLoadRequest(BaseModel):
    """Request model for loading a transportation network."""

    network_file: str = Field(..., description="Path to the pickled network file")

    model_config = ConfigDict(
        json_schema_extra={"example": {"network_file": "networks/transport.gpickle"}}
    )


class TimePeriodsRequest(BaseModel):
    """Request model for setting traffic-simulation time periods."""

    periods: list[str] = Field(..., description="Time-period labels")

    model_config = ConfigDict(
        json_schema_extra={"example": {"periods": ["morning_peak", "evening_peak"]}}
    )


class NetworkMetricsRequest(BaseModel):
    """Request model for network metrics calculation.

    The analyzer must have a network loaded first via POST
    ``/transport/network/load``; metrics are computed for that loaded
    network (there is no network-selection parameter).
    """


class TrafficSimulationRequest(BaseModel):
    """Request model for traffic simulation."""

    origin: str
    destination: str
    departure_time: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "origin": "node-001",
                "destination": "node-045",
                "departure_time": "morning_peak",
            }
        }
    )


class EmissionsCalculationRequest(BaseModel):
    """Request model for emissions calculation."""

    vehicle: Vehicle
    distance: float
    load_factor: float = 1.0
    terrain_factor: float = 1.0

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vehicle": {
                    "id": "truck-001",
                    "type": "truck",
                    "capacity": 1000,
                    "max_range": 500,
                    "speed": 80,
                    "cost_per_km": 1.2,
                    "emissions_per_km": 0.8,
                    "location": [13.404954, 52.520008],
                    "fuel_type": "diesel",
                },
                "distance": 150,
                "load_factor": 0.8,
                "terrain_factor": 1.2,
            }
        }
    )


class EmissionsComparisonRequest(BaseModel):
    """Request model for emissions comparison."""

    route: dict
    vehicle_options: list[Vehicle]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "route": {
                    "distance": 150,
                    "origin": [13.404954, 52.520008],
                    "destination": [11.576124, 48.137154],
                },
                "vehicle_options": [
                    {
                        "id": "truck-001",
                        "type": "truck",
                        "capacity": 1000,
                        "max_range": 500,
                        "speed": 80,
                        "cost_per_km": 1.2,
                        "emissions_per_km": 0.8,
                        "location": [13.404954, 52.520008],
                        "fuel_type": "diesel",
                    },
                    {
                        "id": "truck-002",
                        "type": "truck",
                        "capacity": 1000,
                        "max_range": 400,
                        "speed": 70,
                        "cost_per_km": 1.0,
                        "emissions_per_km": 0.5,
                        "location": [13.404954, 52.520008],
                        "fuel_type": "electric",
                    },
                ],
            }
        }
    )


# Cached module-level singletons: loaded networks and simulation state must
# survive across requests for the API to behave correctly.
@lru_cache(maxsize=1)
def get_multimodal_planner() -> MultiModalPlanner:
    """Dependency for multimodal planner."""
    return MultiModalPlanner()


@lru_cache(maxsize=1)
def get_network_analyzer() -> TransportationNetworkAnalyzer:
    """Dependency for transportation network analyzer."""
    return TransportationNetworkAnalyzer()


@lru_cache(maxsize=1)
def get_traffic_simulator() -> TrafficSimulator:
    """Dependency for traffic simulator."""
    return TrafficSimulator()


@lru_cache(maxsize=1)
def get_emissions_calculator() -> EmissionsCalculator:
    """Dependency for emissions calculator."""
    return EmissionsCalculator()


@router.post("/route", response_model=dict)
async def plan_route(
    request: RouteRequest, planner: MultiModalPlanner = Depends(get_multimodal_planner)
) -> dict:
    """Plan a multimodal route between origin and destination."""
    try:
        route = planner.plan_route(
            origin=request.origin,
            destination=request.destination,
            allowed_modes=request.allowed_modes,
            preferences=request.preferences,
        )
        return route
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/compare-routes", response_model=dict)
async def compare_routes(
    request: CompareRoutesRequest,
    planner: MultiModalPlanner = Depends(get_multimodal_planner),
) -> dict:
    """Compare different multimodal routes between origin and destination."""
    try:
        df = planner.compare_routes(
            origin=request.origin,
            destination=request.destination,
            mode_combinations=request.mode_combinations,
        )

        # Convert DataFrame to dict
        return {"comparisons": df.to_dict(orient="records")}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/network/load", response_model=dict)
async def load_network(
    request: NetworkLoadRequest,
    analyzer: TransportationNetworkAnalyzer = Depends(get_network_analyzer),
) -> dict:
    """Load a transportation network for metrics and critical-link analysis.

    Must be called before POST ``/network/metrics`` or
    ``/network/critical-links``; those endpoints operate on the network
    loaded here.
    """
    try:
        analyzer.load_network(request.network_file)
    except (OSError, ValueError, pickle.UnpicklingError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    network = analyzer.network
    assert network is not None
    return {
        "loaded": True,
        "num_nodes": network.number_of_nodes(),
        "num_edges": network.number_of_edges(),
    }


@router.post("/network/metrics", response_model=dict)
async def get_network_metrics(
    request: NetworkMetricsRequest,
    analyzer: TransportationNetworkAnalyzer = Depends(get_network_analyzer),
) -> dict:
    """Calculate metrics for a transportation network."""
    try:
        metrics = analyzer.calculate_network_metrics()
        return metrics
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/network/critical-links", response_model=list[list[str]])
async def identify_critical_links(
    request: NetworkMetricsRequest,
    analyzer: TransportationNetworkAnalyzer = Depends(get_network_analyzer),
    top_n: int = Query(10, description="Number of critical links to identify"),
) -> list[Any]:
    """Identify critical links in a transportation network."""
    try:
        links = analyzer.identify_critical_links(top_n=top_n)
        return links
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/traffic/load", response_model=dict)
async def load_traffic_network(
    request: NetworkLoadRequest,
    simulator: TrafficSimulator = Depends(get_traffic_simulator),
) -> dict:
    """Load a transportation network for traffic simulation.

    Must be called before POST ``/traffic/simulate`` or
    ``/traffic/congestion``, together with POST ``/traffic/time-periods``.
    """
    try:
        simulator.load_network(request.network_file)
    except (OSError, ValueError, pickle.UnpicklingError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    network = simulator.network
    assert network is not None
    return {
        "loaded": True,
        "num_nodes": network.number_of_nodes(),
        "num_edges": network.number_of_edges(),
    }


@router.post("/traffic/time-periods", response_model=dict)
async def set_time_periods(
    request: TimePeriodsRequest,
    simulator: TrafficSimulator = Depends(get_traffic_simulator),
) -> dict:
    """Set the time periods available for traffic simulation.

    POST ``/traffic/simulate`` only accepts departure times among these
    periods.
    """
    try:
        simulator.set_time_periods(request.periods)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"time_periods": list(simulator.time_periods)}


@router.post("/traffic/simulate", response_model=dict)
async def simulate_traffic(
    request: TrafficSimulationRequest,
    simulator: TrafficSimulator = Depends(get_traffic_simulator),
) -> dict:
    """Simulate traffic for a route."""
    try:
        result = simulator.simulate_traffic(
            origin=request.origin,
            destination=request.destination,
            departure_time=request.departure_time,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/traffic/congestion", response_model=dict)
async def analyze_congestion(
    simulator: TrafficSimulator = Depends(get_traffic_simulator),
    time_period: str | None = Query(None, description="Time period to analyze"),
    congestion_threshold: float = Query(0.7, description="Congestion threshold"),
) -> dict:
    """Analyze network congestion."""
    try:
        result = simulator.analyze_congestion(
            time_period=time_period, congestion_threshold=congestion_threshold
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/emissions/calculate", response_model=float)
async def calculate_emissions(
    request: EmissionsCalculationRequest,
    calculator: EmissionsCalculator = Depends(get_emissions_calculator),
) -> float:
    """Calculate emissions for a route with a specific vehicle."""
    try:
        emissions = calculator.calculate_route_emissions(
            vehicle=request.vehicle,
            distance=request.distance,
            load_factor=request.load_factor,
            terrain_factor=request.terrain_factor,
        )
        return emissions
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/emissions/compare", response_model=dict)
async def compare_vehicle_emissions(
    request: EmissionsComparisonRequest,
    calculator: EmissionsCalculator = Depends(get_emissions_calculator),
) -> dict:
    """Compare emissions for different vehicle options on a route."""
    try:
        df = calculator.compare_emissions(
            route=request.route, vehicle_options=request.vehicle_options
        )

        # Convert DataFrame to dict
        return {"comparisons": df.to_dict(orient="records")}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/emissions/fleet", response_model=dict)
async def calculate_fleet_emissions(
    fleet: list[Vehicle],
    routes: list[Route],
    calculator: EmissionsCalculator = Depends(get_emissions_calculator),
) -> dict:
    """Calculate total emissions for a fleet of vehicles."""
    try:
        result = calculator.calculate_fleet_emissions(fleet=fleet, routes=routes)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
