"""
Normative API module for inferring, analyzing, and simulating social norms.

This module provides API endpoints for working with the normative inference system
to analyze, model, and predict the diffusion of social norms across geographies.
"""

from typing import Any
import datetime
import csv
import json
import uuid
from pathlib import Path as FilePath
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field, ConfigDict
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry

from geo_infer_norms.core.normative_inference import (
    NormEvaluationError,
    NormativeInference,
    SocialNormDiffusion,
)


# Pydantic models for API request/response
class GeometryModel(BaseModel):
    """Model for GeoJSON geometry data"""

    type: str
    coordinates: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)


class SocialNormCreate(BaseModel):
    """Request model for creating a social norm"""

    name: str = Field(..., description="Name of the social norm")
    description: str | None = Field(None, description="Description of the norm")
    category: str = Field(
        ..., description="Category of norm (e.g., environmental, health, etc.)"
    )
    strength: float = Field(
        ..., ge=0.0, le=1.0, description="Initial strength of the norm (0.0 to 1.0)"
    )
    jurisdiction_ids: list[str] = Field(
        ..., description="IDs of jurisdictions where this norm applies"
    )
    factors: dict[str, float] | None = Field(
        None, description="Factors that influence norm strength"
    )
    related_policies: list[str] | None = Field(
        None, description="IDs of related policies"
    )
    tags: list[str] | None = Field(None, description="Tags for the norm")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Recycling Behavior",
                "description": "Social norm around household recycling behavior",
                "category": "environmental",
                "strength": 0.65,
                "jurisdiction_ids": ["city-001", "city-002"],
                "factors": {
                    "infrastructure_access": 0.8,
                    "public_education": 0.7,
                    "peer_pressure": 0.6,
                },
                "related_policies": ["policy-001"],
                "tags": ["recycling", "waste", "environmental"],
            }
        }
    )


class NormDiffusionRequest(BaseModel):
    """Request model for social norm diffusion simulation"""

    norm_id: str = Field(..., description="ID of the norm to simulate")
    time_steps: int = Field(..., gt=0, description="Number of time steps to simulate")
    initial_conditions: dict[str, Any] | None = Field(
        None, description="Initial conditions for simulation"
    )
    parameters: dict[str, Any] | None = Field(None, description="Simulation parameters")
    spatial_extent: GeometryModel | None = Field(
        None, description="GeoJSON geometry of simulation area"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "norm_id": "norm-001",
                "time_steps": 20,
                "initial_conditions": {
                    "seed_jurisdictions": ["city-001"],
                    "initial_strength": 0.8,
                },
                "parameters": {
                    "diffusion_rate": 0.1,
                    "decay_rate": 0.02,
                    "influence_distance": 50000,  # meters
                },
                "spatial_extent": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        }
    )


class NormativeInferenceRequest(BaseModel):
    """Request model for normative inference from data"""

    data_source: str = Field(..., description="Source of data for inference")
    inference_type: str = Field(..., description="Type of inference to perform")
    spatial_extent: GeometryModel | None = Field(
        None, description="GeoJSON geometry of inference area"
    )
    temporal_range: dict[str, str] | None = Field(
        None, description="Temporal range for inference"
    )
    parameters: dict[str, Any] | None = Field(None, description="Inference parameters")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data_source": "survey_responses",
                "inference_type": "bayesian",
                "spatial_extent": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
                "temporal_range": {
                    "start_date": "2022-01-01",
                    "end_date": "2022-12-31",
                },
                "parameters": {
                    "confidence_threshold": 0.7,
                    "prior_strength": 0.5,
                    "included_factors": ["income", "education", "age"],
                },
            }
        }
    )


class NormPolicyImpactRequest(BaseModel):
    """Request model for assessing policy impact on norms"""

    norm_id: str = Field(..., description="ID of the norm to assess")
    policy_id: str = Field(..., description="ID of the policy to assess")
    time_horizon: int = Field(
        ..., gt=0, description="Time horizon for assessment (in months)"
    )
    parameters: dict[str, Any] | None = Field(None, description="Assessment parameters")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "norm_id": "norm-001",
                "policy_id": "policy-001",
                "time_horizon": 24,
                "parameters": {
                    "policy_effectiveness": 0.7,
                    "public_awareness": 0.6,
                    "enforcement_level": 0.8,
                },
            }
        }
    )


class PointLocation(BaseModel):
    """Model for a geographic point location"""

    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude")


class NormativeAPI:
    """API for normative inference and social norm analysis"""

    def __init__(
        self,
        normative_inference: NormativeInference | None = None,
        social_norm_diffusion: SocialNormDiffusion | None = None,
    ):
        """
        Initialize the NormativeAPI.

        Args:
            normative_inference: Optional NormativeInference instance to use
            social_norm_diffusion: Optional SocialNormDiffusion instance to use
        """
        self.normative_inference = normative_inference or NormativeInference()
        self.social_norm_diffusion = social_norm_diffusion or SocialNormDiffusion()
        self.router = APIRouter()
        self._setup_routes()

        # Temporary storage for social norms
        self._social_norms: dict[str, dict[str, Any]] = {}

    def _setup_routes(self) -> None:
        """Set up API routes"""
        # Social norm endpoints
        self.router.post("/norms", response_model=dict[str, Any])(
            self.create_social_norm
        )
        self.router.get("/norms", response_model=list[dict[str, Any]])(
            self.list_social_norms
        )
        self.router.get("/norms/{norm_id}", response_model=dict[str, Any])(
            self.get_social_norm
        )

        # Norm diffusion endpoints
        self.router.post("/diffusion/simulate", response_model=dict[str, Any])(
            self.simulate_norm_diffusion
        )
        self.router.get("/diffusion/factors/{norm_id}", response_model=dict[str, Any])(
            self.get_diffusion_factors
        )

        # Normative inference endpoints
        self.router.post("/inference/analyze", response_model=dict[str, Any])(
            self.perform_normative_inference
        )
        self.router.post("/inference/spatial-patterns", response_model=dict[str, Any])(
            self.analyze_spatial_patterns
        )

        # Policy impact endpoints
        self.router.post("/policy-impact", response_model=dict[str, Any])(
            self.assess_policy_impact
        )

        # Spatial query endpoints
        self.router.post(
            "/spatial/norms-at-point", response_model=list[dict[str, Any]]
        )(self.get_norms_at_point)

        # Export endpoints
        self.router.get("/export/geojson", response_model=dict[str, Any])(
            self.export_to_geojson
        )

    # Helper methods

    def _geometry_from_model(
        self, geometry_model: GeometryModel | None
    ) -> BaseGeometry | None:
        """
        Convert a GeometryModel to a Shapely geometry object.

        Args:
            geometry_model: GeometryModel to convert

        Returns:
            Shapely geometry object or None
        """
        if not geometry_model:
            return None

        geojson = {
            "type": geometry_model.type,
            "coordinates": geometry_model.coordinates,
        }

        try:
            return shape(geojson)
        except Exception as e:
            raise ValueError(f"Invalid geometry: {str(e)}")

    def _social_norm_to_dict(self, norm: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a social norm to a dictionary for API response.

        Args:
            norm: Social norm dictionary

        Returns:
            Dictionary representation
        """
        return {
            "id": norm["id"],
            "name": norm["name"],
            "description": norm["description"],
            "category": norm["category"],
            "strength": norm["strength"],
            "jurisdiction_ids": norm["jurisdiction_ids"],
            "factors": norm["factors"],
            "related_policies": norm["related_policies"],
            "tags": norm["tags"],
            "created_at": norm["created_at"],
        }

    # Social norm endpoints

    async def create_social_norm(self, norm_data: SocialNormCreate) -> dict[str, Any]:
        """
        Create a new social norm.

        Args:
            norm_data: The social norm data

        Returns:
            Success message with norm ID
        """
        try:
            # Create norm object
            norm_id = f"norm-{uuid.uuid4().hex}"
            norm = {
                "id": norm_id,
                "name": norm_data.name,
                "description": norm_data.description or "",
                "category": norm_data.category,
                "strength": norm_data.strength,
                "jurisdiction_ids": norm_data.jurisdiction_ids,
                "factors": norm_data.factors or {},
                "related_policies": norm_data.related_policies or [],
                "tags": norm_data.tags or [],
                "created_at": datetime.datetime.now().isoformat(),
            }

            # Store norm
            self._social_norms[norm_id] = norm

            # Register the same entities and norm with the diffusion engine so
            # the simulation endpoint operates on the submitted jurisdiction
            # set rather than constructing an unrelated response.
            for jurisdiction_id in norm_data.jurisdiction_ids:
                self.social_norm_diffusion.add_entity(
                    jurisdiction_id,
                    attributes={"jurisdiction_id": jurisdiction_id},
                )
            self.social_norm_diffusion.add_norm(
                norm_id,
                norm_data.name,
                spatial_factor=(
                    float(norm_data.factors.get("geographic_proximity", 0.0))
                    if norm_data.factors
                    else 0.0
                ),
                network_factor=(
                    float(norm_data.factors.get("social_networks", 0.0))
                    if norm_data.factors
                    else 0.0
                ),
                content_factor=(
                    float(norm_data.factors.get("content", 0.0))
                    if norm_data.factors
                    else 0.0
                ),
                attributes=norm_data.factors or {},
            )

            return {
                "status": "success",
                "message": "Social norm created successfully",
                "norm_id": norm_id,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def list_social_norms(
        self,
        category: str | None = Query(None, description="Filter by category"),
        jurisdiction_id: str | None = Query(
            None, description="Filter by jurisdiction ID"
        ),
        tag: str | None = Query(None, description="Filter by tag"),
        min_strength: float | None = Query(
            None, ge=0.0, le=1.0, description="Minimum strength"
        ),
    ) -> list[dict[str, Any]]:
        """
        List all social norms, optionally filtered.

        Args:
            category: Optional category to filter by
            jurisdiction_id: Optional jurisdiction ID to filter by
            tag: Optional tag to filter by
            min_strength: Optional minimum strength to filter by

        Returns:
            List of social norms
        """
        try:
            norms = list(self._social_norms.values())

            if category:
                norms = [n for n in norms if n["category"] == category]

            if jurisdiction_id:
                norms = [n for n in norms if jurisdiction_id in n["jurisdiction_ids"]]

            if tag:
                norms = [n for n in norms if tag in n["tags"]]

            if min_strength is not None:
                norms = [n for n in norms if n["strength"] >= min_strength]

            return [self._social_norm_to_dict(n) for n in norms]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def get_social_norm(
        self, norm_id: str = Path(..., description="ID of the social norm")
    ) -> dict[str, Any]:
        """
        Get a social norm by ID.

        Args:
            norm_id: The ID of the social norm

        Returns:
            Social norm details
        """
        try:
            norm = self._social_norms.get(norm_id)

            if not norm:
                raise HTTPException(
                    status_code=404, detail=f"Social norm with ID {norm_id} not found"
                )

            return self._social_norm_to_dict(norm)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Norm diffusion endpoints

    async def simulate_norm_diffusion(
        self, diffusion_request: NormDiffusionRequest
    ) -> dict[str, Any]:
        """
        Simulate the diffusion of a social norm over time.

        Args:
            diffusion_request: The norm diffusion request

        Returns:
            Diffusion simulation results
        """
        try:
            # Check if norm exists
            if diffusion_request.norm_id not in self._social_norms:
                raise HTTPException(
                    status_code=404,
                    detail=f"Social norm with ID {diffusion_request.norm_id} not found",
                )

            norm = self._social_norms[diffusion_request.norm_id]
            initial_conditions = diffusion_request.initial_conditions or {}
            seed_jurisdictions = initial_conditions.get("seed_jurisdictions", [])
            if not seed_jurisdictions:
                seed_jurisdictions = norm["jurisdiction_ids"][:1]

            diffusion = self.social_norm_diffusion
            for jurisdiction_id in norm["jurisdiction_ids"]:
                if jurisdiction_id not in diffusion.entities:
                    diffusion.add_entity(
                        jurisdiction_id,
                        attributes={"jurisdiction_id": jurisdiction_id},
                    )
            diffusion.add_norm(
                diffusion_request.norm_id,
                norm["name"],
                initial_adopters=[
                    jurisdiction_id
                    for jurisdiction_id in seed_jurisdictions
                    if jurisdiction_id in norm["jurisdiction_ids"]
                ],
                attributes=norm["factors"],
            )
            steps = diffusion.simulate(diffusion_request.time_steps)
            history = diffusion.get_adoption_history().get(
                diffusion_request.norm_id, []
            )
            summary = diffusion.get_adoption_summary().get(
                diffusion_request.norm_id, {}
            )
            time_series = [
                {"time_step": index, "adoption_rate": rate}
                for index, rate in enumerate(history)
            ]
            jurisdictions = [
                {
                    "id": jurisdiction_id,
                    "adopted": diffusion.adoption_state[diffusion_request.norm_id].get(
                        jurisdiction_id, False
                    ),
                }
                for jurisdiction_id in norm["jurisdiction_ids"]
            ]

            return {
                "status": "success",
                "message": "Norm diffusion simulation completed",
                "norm_id": norm["id"],
                "norm_name": norm["name"],
                "time_steps": diffusion_request.time_steps,
                "time_series": time_series,
                "jurisdictions": jurisdictions,
                "parameters": diffusion_request.parameters or {},
                "steps": steps,
                "final_adoption_rate": summary.get("adoption_rate", 0.0),
                "adoption_summary": summary,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def get_diffusion_factors(
        self, norm_id: str = Path(..., description="ID of the social norm")
    ) -> dict[str, Any]:
        """
        Get the factors that influence the diffusion of a social norm.

        Args:
            norm_id: The ID of the social norm

        Returns:
            Diffusion factors
        """
        try:
            # Check if norm exists
            if norm_id not in self._social_norms:
                raise HTTPException(
                    status_code=404, detail=f"Social norm with ID {norm_id} not found"
                )

            norm = self._social_norms[norm_id]

            norm_factors = norm["factors"] or {}
            factor_analysis = {
                factor: {"importance": value} for factor, value in norm_factors.items()
            }

            return {
                "status": "success",
                "norm_id": norm_id,
                "norm_name": norm["name"],
                "factors": factor_analysis,
                "overall_diffusion_potential": (
                    sum(float(f["importance"]) for f in factor_analysis.values())
                    / len(factor_analysis)
                    if factor_analysis
                    else 0
                ),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Normative inference endpoints

    async def perform_normative_inference(
        self, inference_request: NormativeInferenceRequest
    ) -> dict[str, Any]:
        """
        Perform normative inference from data.

        Args:
            inference_request: The normative inference request

        Returns:
            Inference results
        """
        try:
            source_path = FilePath(inference_request.data_source)
            if not source_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Inference data source not found: {source_path}",
                )
            if source_path.suffix.lower() == ".csv":
                with source_path.open(newline="", encoding="utf-8") as handle:
                    records = list(csv.DictReader(handle))
            elif source_path.suffix.lower() in {".json", ".geojson"}:
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                records = (
                    payload.get("features", [])
                    if payload.get("type") == "FeatureCollection"
                    else payload
                )
                if isinstance(records, dict):
                    records = [records]
            else:
                raise HTTPException(
                    status_code=415,
                    detail="Inference data source must be CSV, JSON, or GeoJSON",
                )

            parameters = inference_request.parameters or {}
            behavior = str(parameters.get("behavior", "value"))
            value_field = str(parameters.get("value_field", behavior))
            expected_value = parameters.get("expected_value")
            norm_id = self.normative_inference.add_norm(
                name=str(parameters.get("norm_name", inference_request.inference_type)),
                condition=lambda observations: (
                    observations.get(behavior) == expected_value
                    if expected_value is not None
                    else bool(observations.get(behavior))
                ),
                probability=float(parameters.get("prior_probability", 0.5)),
                description="Norm inferred from the submitted observation source",
            )
            entity_ids = []
            for index, record in enumerate(records):
                properties = (
                    record.get("properties", record) if isinstance(record, dict) else {}
                )
                entity_id = str(
                    properties.get("entity_id", properties.get("id", index))
                )
                if value_field not in properties:
                    continue
                self.normative_inference.add_observation(
                    entity_id,
                    behavior,
                    properties[value_field],
                )
                entity_ids.append(entity_id)
            if not entity_ids:
                raise HTTPException(
                    status_code=422,
                    detail=f"No records contained the configured value field '{value_field}'",
                )
            inferred_norms = [
                {
                    "entity_id": entity_id,
                    "norm_id": norm_id,
                    "compliance_probability": self.normative_inference.infer_compliance(
                        entity_id, norm_id
                    ),
                }
                for entity_id in entity_ids
            ]

            return {
                "status": "success",
                "message": "Normative inference completed",
                "data_source": inference_request.data_source,
                "inference_type": inference_request.inference_type,
                "temporal_range": inference_request.temporal_range,
                "parameters": inference_request.parameters,
                "inferred_norms": inferred_norms,
                "inference_id": f"infer-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            }
        except NormEvaluationError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Norm condition evaluation failed: {e}",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def analyze_spatial_patterns(
        self, inference_request: NormativeInferenceRequest
    ) -> dict[str, Any]:
        """
        Analyze spatial patterns of social norms.

        Args:
            inference_request: The normative inference request

        Returns:
            Spatial pattern analysis results
        """
        try:
            geometry = self._geometry_from_model(inference_request.spatial_extent)
            locations = []
            for entity_id, behaviors in self.normative_inference.observations.items():
                for observations in behaviors.values():
                    latest = max(observations, key=lambda item: item["timestamp"])
                    location = latest.get("location")
                    if location is not None and (
                        geometry is None or geometry.contains(location)
                    ):
                        locations.append(
                            {
                                "entity_id": entity_id,
                                "longitude": location.x,
                                "latitude": location.y,
                            }
                        )
            return {
                "status": "success",
                "message": "Spatial pattern analysis completed",
                "data_source": inference_request.data_source,
                "spatial_patterns": [
                    {"pattern_type": "observations", "locations": locations}
                ],
                "observation_count": len(locations),
                "analysis_id": f"spatial-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Policy impact endpoints

    async def assess_policy_impact(
        self, impact_request: NormPolicyImpactRequest
    ) -> dict[str, Any]:
        """
        Assess the impact of a policy on a social norm.

        Args:
            impact_request: The policy impact assessment request

        Returns:
            Impact assessment results
        """
        try:
            # Check if norm exists
            if impact_request.norm_id not in self._social_norms:
                raise HTTPException(
                    status_code=404,
                    detail=f"Social norm with ID {impact_request.norm_id} not found",
                )

            norm = self._social_norms[impact_request.norm_id]

            # This would be a real impact assessment in a production implementation
            # For now, generate a deterministic response

            # Extract parameters
            effectiveness = (
                impact_request.parameters.get("policy_effectiveness", 0.5)
                if impact_request.parameters
                else 0.5
            )
            awareness = (
                impact_request.parameters.get("public_awareness", 0.5)
                if impact_request.parameters
                else 0.5
            )
            enforcement = (
                impact_request.parameters.get("enforcement_level", 0.5)
                if impact_request.parameters
                else 0.5
            )

            # Generate impact projections
            time_points = [0, 6, 12, 18, 24]
            if impact_request.time_horizon <= 24:
                time_points = [
                    t for t in time_points if t <= impact_request.time_horizon
                ]
                if impact_request.time_horizon not in time_points:
                    time_points.append(impact_request.time_horizon)
            else:
                step = impact_request.time_horizon // 5
                time_points = [i * step for i in range(6)]

            initial_strength = norm["strength"]

            # Simple model for impact
            combined_effect = (effectiveness + awareness + enforcement) / 3
            impact_projections = []

            for t in time_points:
                # Sigmoid growth curve with max at combined_effect
                relative_t = t / impact_request.time_horizon
                if relative_t == 0:
                    impact = 0
                else:
                    # S-curve: slow start, accelerate, then plateau
                    impact = combined_effect * (
                        1 / (1 + np.exp(-10 * (relative_t - 0.5)))
                    )

                new_strength = min(
                    1.0, initial_strength + impact * (1 - initial_strength)
                )

                impact_projections.append(
                    {
                        "time_point": t,  # months
                        "norm_strength": new_strength,
                        "impact": new_strength - initial_strength,
                    }
                )

            return {
                "status": "success",
                "message": "Policy impact assessment completed",
                "norm_id": norm["id"],
                "norm_name": norm["name"],
                "policy_id": impact_request.policy_id,
                "time_horizon": impact_request.time_horizon,
                "initial_strength": initial_strength,
                "final_strength": impact_projections[-1]["norm_strength"],
                "max_impact": max(p["impact"] for p in impact_projections),
                "impact_projections": impact_projections,
                "key_factors": {
                    "policy_effectiveness": effectiveness,
                    "public_awareness": awareness,
                    "enforcement_level": enforcement,
                    "combined_effect": combined_effect,
                },
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Spatial query endpoints

    async def get_norms_at_point(self, point: PointLocation) -> list[dict[str, Any]]:
        """
        Get all social norms that apply to a specific geographic point.

        Args:
            point: Geographic point location

        Returns:
            List of applicable social norms
        """
        try:
            norms = [
                norm
                for norm in self._social_norms.values()
                if norm.get("spatial_geometry") is not None
                and norm["spatial_geometry"].contains(Point(point.lon, point.lat))
            ]
            return [self._social_norm_to_dict(n) for n in norms]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Export endpoints

    async def export_to_geojson(
        self,
        category: str | None = Query(None, description="Filter by category"),
        min_strength: float | None = Query(
            None, ge=0.0, le=1.0, description="Minimum strength"
        ),
    ) -> dict[str, Any]:
        """
        Export social norm spatial distribution to GeoJSON.

        Args:
            category: Optional category to filter by
            min_strength: Optional minimum strength to filter by

        Returns:
            GeoJSON data
        """
        try:
            # Filter norms
            norms = list(self._social_norms.values())

            if category:
                norms = [n for n in norms if n["category"] == category]

            if min_strength is not None:
                norms = [n for n in norms if n["strength"] >= min_strength]

            features = []

            for norm in norms:
                geometry = norm.get("spatial_geometry")
                if geometry is None:
                    continue
                for jid in norm["jurisdiction_ids"]:
                    features.append(
                        {
                            "type": "Feature",
                            "geometry": geometry.__geo_interface__,
                            "properties": {
                                "jurisdiction_id": jid,
                                "norm_id": norm["id"],
                                "norm_name": norm["name"],
                                "norm_strength": norm["strength"],
                                "norm_category": norm["category"],
                            },
                        }
                    )

            geojson = {"type": "FeatureCollection", "features": features}

            return {
                "status": "success",
                "message": "Social norms exported to GeoJSON successfully",
                "feature_count": len(features),
                "geojson": geojson,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
