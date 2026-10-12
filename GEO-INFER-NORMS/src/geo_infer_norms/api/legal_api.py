"""
Legal API module for geospatial legal analysis and jurisdictions.

This module provides API endpoints for interacting with legal frameworks,
regulations, and jurisdictional data.
"""

from typing import Any
import datetime
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field, ConfigDict
import geopandas as gpd
from shapely.geometry import Point, shape
import json
import uuid
from shapely.geometry.base import BaseGeometry

from geo_infer_norms.core.legal_frameworks import LegalFramework, JurisdictionHandler
from geo_infer_norms.models.legal_entity import Jurisdiction
from geo_infer_norms.models.regulation import Regulation, RegulatoryFramework


# Pydantic models for API request/response
class GeometryModel(BaseModel):
    """Model for GeoJSON geometry data"""

    type: str
    coordinates: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)


class JurisdictionCreate(BaseModel):
    """Request model for creating a jurisdiction"""

    name: str = Field(..., description="Name of the jurisdiction")
    level: str = Field(
        ..., description="Level of jurisdiction (federal, state, local, etc.)"
    )
    description: str | None = Field(None, description="Description of the jurisdiction")
    code: str | None = Field(None, description="Jurisdiction code")
    parent_id: str | None = Field(None, description="Parent jurisdiction ID")
    geometry: GeometryModel | None = Field(None, description="GeoJSON geometry")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Sample City",
                "level": "local",
                "description": "A sample city jurisdiction",
                "code": "CITY-001",
                "parent_id": "state-001",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        }
    )


class RegulationCreate(BaseModel):
    """Request model for creating a regulation"""

    name: str = Field(..., description="Name of the regulation")
    description: str | None = Field(None, description="Description of the regulation")
    code: str | None = Field(None, description="Regulation code or identifier")
    category: str | None = Field(None, description="Category of regulation")
    applicable_jurisdictions: list[str] = Field(
        ..., description="IDs of jurisdictions where this regulation applies"
    )
    effective_date: datetime.datetime | None = Field(
        None, description="When the regulation takes effect"
    )
    expiration_date: datetime.datetime | None = Field(
        None, description="When the regulation expires"
    )
    superseded_regulation_id: str | None = Field(
        None, description="ID of regulation this supersedes"
    )
    source_url: str | None = Field(None, description="URL to the source document")
    tags: list[str] | None = Field(None, description="Tags for the regulation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Water Conservation Ordinance",
                "description": "Restrictions on water usage during drought conditions",
                "code": "WCO-2023",
                "category": "environmental",
                "applicable_jurisdictions": ["city-001", "city-002"],
                "effective_date": "2023-06-01T00:00:00Z",
                "tags": ["water", "conservation", "drought"],
            }
        }
    )


class RegulatoryFrameworkCreate(BaseModel):
    """Request model for creating a regulatory framework"""

    name: str = Field(..., description="Name of the framework")
    description: str | None = Field(None, description="Description of the framework")
    authority: str | None = Field(
        None, description="Authority that established the framework"
    )
    sector: str | None = Field(None, description="Sector the framework applies to")
    regulation_ids: list[str] | None = Field(
        None, description="IDs of regulations in this framework"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Environmental Protection Framework",
                "description": "Framework of regulations for environmental protection",
                "authority": "Environmental Protection Agency",
                "sector": "environment",
                "regulation_ids": ["reg-001", "reg-002", "reg-003"],
            }
        }
    )


class PointLocation(BaseModel):
    """Model for a geographic point location"""

    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude")


class LegalAPI:
    """API for legal frameworks and jurisdictions"""

    def __init__(
        self,
        legal_framework: LegalFramework | None = None,
        jurisdiction_handler: JurisdictionHandler | None = None,
    ):
        """
        Initialize the LegalAPI.

        Args:
            legal_framework: Optional LegalFramework instance to use
            jurisdiction_handler: Optional JurisdictionHandler instance to use
        """
        self.legal_framework = legal_framework or LegalFramework(
            name="Default Legal Framework",
            description="Default framework for legal analysis",
        )
        self.jurisdiction_handler = jurisdiction_handler or JurisdictionHandler()
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up API routes"""
        # Jurisdiction endpoints
        self.router.post("/jurisdictions", response_model=dict[str, Any])(
            self.create_jurisdiction
        )
        self.router.get("/jurisdictions", response_model=list[dict[str, Any]])(
            self.list_jurisdictions
        )
        self.router.get(
            "/jurisdictions/{jurisdiction_id}", response_model=dict[str, Any]
        )(self.get_jurisdiction)
        self.router.get(
            "/jurisdictions/by-name/{name}", response_model=list[dict[str, Any]]
        )(self.find_jurisdictions_by_name)
        self.router.get(
            "/jurisdictions/hierarchy/{jurisdiction_id}",
            response_model=list[dict[str, Any]],
        )(self.get_jurisdiction_hierarchy)

        # Regulation endpoints
        self.router.post("/regulations", response_model=dict[str, Any])(
            self.create_regulation
        )
        self.router.get("/regulations", response_model=list[dict[str, Any]])(
            self.list_regulations
        )
        self.router.get("/regulations/{regulation_id}", response_model=dict[str, Any])(
            self.get_regulation
        )
        self.router.get(
            "/regulations/jurisdiction/{jurisdiction_id}",
            response_model=list[dict[str, Any]],
        )(self.get_regulations_by_jurisdiction)

        # Regulatory framework endpoints
        self.router.post("/frameworks", response_model=dict[str, Any])(
            self.create_regulatory_framework
        )
        self.router.get("/frameworks", response_model=list[dict[str, Any]])(
            self.list_regulatory_frameworks
        )

        # Geospatial endpoints
        self.router.post(
            "/spatial/jurisdictions-at-point", response_model=list[dict[str, Any]]
        )(self.get_jurisdictions_by_point)
        self.router.post(
            "/spatial/regulations-at-point", response_model=list[dict[str, Any]]
        )(self.get_regulations_by_point)
        self.router.get("/spatial/export", response_model=dict[str, Any])(
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
            raise ValueError(f"Invalid geometry: {str(e)}") from e

    def _jurisdiction_to_dict(self, jurisdiction: Jurisdiction) -> dict[str, Any]:
        """
        Convert a Jurisdiction object to a dictionary for API response.

        Args:
            jurisdiction: Jurisdiction object

        Returns:
            Dictionary representation
        """
        geometry_dict = None
        if jurisdiction.geometry:
            geometry_dict = json.loads(
                gpd.GeoSeries([jurisdiction.geometry], crs="EPSG:4326").to_json()
            )["features"][0]["geometry"]

        return {
            "id": jurisdiction.id,
            "name": jurisdiction.name,
            "level": jurisdiction.level,
            "description": jurisdiction.description,
            "code": jurisdiction.code,
            "parent_id": jurisdiction.parent_id,
            "geometry": geometry_dict,
        }

    def _regulation_to_dict(self, regulation: Regulation) -> dict[str, Any]:
        """
        Convert a Regulation object to a dictionary for API response.

        Args:
            regulation: Regulation object

        Returns:
            Dictionary representation
        """
        return {
            "id": regulation.id,
            "name": regulation.name,
            "description": regulation.description,
            "code": regulation.code,
            "category": regulation.category,
            "applicable_jurisdictions": regulation.applicable_jurisdictions,
            "effective_date": regulation.effective_date.isoformat()
            if regulation.effective_date
            else None,
            "expiration_date": regulation.expiration_date.isoformat()
            if regulation.expiration_date
            else None,
            "superseded_regulation_id": regulation.superseded_regulation_id,
            "source_url": regulation.source_url,
            "tags": regulation.tags,
        }

    def _framework_to_dict(self, framework: RegulatoryFramework) -> dict[str, Any]:
        """
        Convert a RegulatoryFramework object to a dictionary for API response.

        Args:
            framework: RegulatoryFramework object

        Returns:
            Dictionary representation
        """
        return {
            "id": framework.id,
            "name": framework.name,
            "description": framework.description,
            "authority": framework.authority,
            "sector": framework.sector,
            "regulation_ids": framework.regulation_ids,
        }

    # Jurisdiction endpoints

    async def create_jurisdiction(
        self, jurisdiction_data: JurisdictionCreate
    ) -> dict[str, Any]:
        """
        Create a new jurisdiction.

        Args:
            jurisdiction_data: The jurisdiction data

        Returns:
            Success message with jurisdiction ID
        """
        try:
            # Convert geometry model to shapely geometry
            geometry = self._geometry_from_model(jurisdiction_data.geometry)

            # Create Jurisdiction object
            jurisdiction = Jurisdiction(
                id=f"jur-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
                name=jurisdiction_data.name,
                level=jurisdiction_data.level,
                description=jurisdiction_data.description or "",
                code=jurisdiction_data.code,
                parent_id=jurisdiction_data.parent_id,
                geometry=geometry,
            )

            # Add to framework and handler
            self.legal_framework.add_jurisdiction(jurisdiction)
            self.jurisdiction_handler.add_jurisdiction(jurisdiction)

            return {
                "status": "success",
                "message": "Jurisdiction created successfully",
                "jurisdiction_id": jurisdiction.id,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def list_jurisdictions(
        self,
        level: str | None = Query(None, description="Filter by jurisdiction level"),
        with_geometry: bool = Query(False, description="Include geometry in response"),
    ) -> list[dict[str, Any]]:
        """
        List all jurisdictions, optionally filtered by level.

        Args:
            level: Optional level to filter by
            with_geometry: Whether to include geometry in response

        Returns:
            List of jurisdictions
        """
        try:
            jurisdictions = self.legal_framework.jurisdictions

            if level:
                jurisdictions = [j for j in jurisdictions if j.level == level]

            result = []
            for jurisdiction in jurisdictions:
                jurisdiction_dict = {
                    "id": jurisdiction.id,
                    "name": jurisdiction.name,
                    "level": jurisdiction.level,
                    "description": jurisdiction.description,
                    "code": jurisdiction.code,
                    "parent_id": jurisdiction.parent_id,
                }

                if with_geometry and jurisdiction.geometry:
                    geometry_dict = json.loads(
                        gpd.GeoSeries(
                            [jurisdiction.geometry], crs="EPSG:4326"
                        ).to_json()
                    )["features"][0]["geometry"]
                    jurisdiction_dict["geometry"] = geometry_dict

                result.append(jurisdiction_dict)

            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def get_jurisdiction(
        self, jurisdiction_id: str = Path(..., description="ID of the jurisdiction")
    ) -> dict[str, Any]:
        """
        Get a jurisdiction by ID.

        Args:
            jurisdiction_id: The ID of the jurisdiction

        Returns:
            Jurisdiction details
        """
        try:
            jurisdiction = self.legal_framework._jurisdiction_index.get(jurisdiction_id)

            if not jurisdiction:
                raise HTTPException(
                    status_code=404,
                    detail=f"Jurisdiction with ID {jurisdiction_id} not found",
                )

            return self._jurisdiction_to_dict(jurisdiction)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def find_jurisdictions_by_name(
        self,
        name: str = Path(..., description="Name to search for"),
        partial_match: bool = Query(
            False, description="Whether to allow partial matching"
        ),
    ) -> list[dict[str, Any]]:
        """
        Find jurisdictions by name.

        Args:
            name: The name to search for
            partial_match: Whether to allow partial matching

        Returns:
            List of matching jurisdictions
        """
        try:
            jurisdictions = self.jurisdiction_handler.find_jurisdictions_by_name(
                name=name, partial_match=partial_match
            )

            return [self._jurisdiction_to_dict(j) for j in jurisdictions]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def get_jurisdiction_hierarchy(
        self, jurisdiction_id: str = Path(..., description="ID of the jurisdiction")
    ) -> list[dict[str, Any]]:
        """
        Get the hierarchical chain of jurisdictions.

        Args:
            jurisdiction_id: The ID of the jurisdiction

        Returns:
            List of jurisdictions representing the hierarchy
        """
        try:
            hierarchy = self.jurisdiction_handler.get_jurisdiction_hierarchy(
                jurisdiction_id
            )

            if not hierarchy:
                raise HTTPException(
                    status_code=404,
                    detail=f"Jurisdiction with ID {jurisdiction_id} not found",
                )

            return [self._jurisdiction_to_dict(j) for j in hierarchy]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    # Regulation endpoints

    async def create_regulation(
        self, regulation_data: RegulationCreate
    ) -> dict[str, Any]:
        """
        Create a new regulation.

        Args:
            regulation_data: The regulation data

        Returns:
            Success message with regulation ID
        """
        try:
            # Create Regulation object
            regulation = Regulation(
                id=f"reg-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
                name=regulation_data.name,
                description=regulation_data.description or "",
                regulation_type=regulation_data.category or "",
                issuing_authority="",
                code=regulation_data.code,
                category=regulation_data.category,
                applicable_jurisdictions=regulation_data.applicable_jurisdictions,
                effective_date=regulation_data.effective_date or datetime.date.today(),
                expiration_date=regulation_data.expiration_date,
                superseded_regulation_id=regulation_data.superseded_regulation_id,
                source_url=regulation_data.source_url,
                tags=regulation_data.tags or [],
            )

            # Add to framework
            self.legal_framework.add_regulation(regulation)

            return {
                "status": "success",
                "message": "Regulation created successfully",
                "regulation_id": regulation.id,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def list_regulations(
        self,
        category: str | None = Query(None, description="Filter by regulation category"),
        tag: str | None = Query(None, description="Filter by tag"),
    ) -> list[dict[str, Any]]:
        """
        List all regulations, optionally filtered.

        Args:
            category: Optional category to filter by
            tag: Optional tag to filter by

        Returns:
            List of regulations
        """
        try:
            regulations = self.legal_framework.regulations

            if category:
                regulations = [r for r in regulations if r.category == category]

            if tag:
                regulations = [r for r in regulations if tag in (r.tags or [])]

            return [self._regulation_to_dict(r) for r in regulations]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def get_regulation(
        self, regulation_id: str = Path(..., description="ID of the regulation")
    ) -> dict[str, Any]:
        """
        Get a regulation by ID.

        Args:
            regulation_id: The ID of the regulation

        Returns:
            Regulation details
        """
        try:
            regulation = self.legal_framework._regulation_index.get(regulation_id)

            if not regulation:
                raise HTTPException(
                    status_code=404,
                    detail=f"Regulation with ID {regulation_id} not found",
                )

            return self._regulation_to_dict(regulation)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def get_regulations_by_jurisdiction(
        self, jurisdiction_id: str = Path(..., description="ID of the jurisdiction")
    ) -> list[dict[str, Any]]:
        """
        Get all regulations applicable to a specific jurisdiction.

        Args:
            jurisdiction_id: The ID of the jurisdiction

        Returns:
            List of applicable regulations
        """
        try:
            regulations = self.legal_framework.get_regulations_by_jurisdiction(
                jurisdiction_id
            )

            return [self._regulation_to_dict(r) for r in regulations]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    # Regulatory framework endpoints

    async def create_regulatory_framework(
        self, framework_data: RegulatoryFrameworkCreate
    ) -> dict[str, Any]:
        """
        Create a new regulatory framework.

        Args:
            framework_data: The framework data

        Returns:
            Success message with framework ID
        """
        try:
            # Create RegulatoryFramework object
            framework = RegulatoryFramework(
                id=f"framework-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
                name=framework_data.name,
                description=framework_data.description or "",
                authority=framework_data.authority,
                sector=framework_data.sector,
                regulation_ids=framework_data.regulation_ids or [],
            )

            # In a real implementation, this would be added to a framework registry

            return {
                "status": "success",
                "message": "Regulatory framework created successfully",
                "framework_id": framework.id,
                "framework": self._framework_to_dict(framework),
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def list_regulatory_frameworks(
        self, sector: str | None = Query(None, description="Filter by sector")
    ) -> list[dict[str, Any]]:
        """
        List all regulatory frameworks.

        Args:
            sector: Optional sector to filter by

        Returns:
            List of regulatory frameworks
        """
        # This would query a framework registry in a real implementation
        # For now, return a deterministic response
        return [
            {
                "id": "framework-example",
                "name": "Example Regulatory Framework",
                "description": "An example framework for demonstration",
                "authority": "Example Authority",
                "sector": "general",
                "regulation_ids": [],
            }
        ]

    # Geospatial endpoints

    async def get_jurisdictions_by_point(
        self, point: PointLocation
    ) -> list[dict[str, Any]]:
        """
        Get all jurisdictions that contain a specific geographic point.

        Args:
            point: Geographic point location

        Returns:
            List of jurisdictions containing the point
        """
        try:
            # Convert to shapely Point
            shapely_point = Point(point.lon, point.lat)

            # Get jurisdictions containing the point
            jurisdictions = self.legal_framework.get_jurisdictions_by_point(
                shapely_point
            )

            return [self._jurisdiction_to_dict(j) for j in jurisdictions]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def get_regulations_by_point(
        self, point: PointLocation
    ) -> list[dict[str, Any]]:
        """
        Get all regulations applicable to a specific geographic point.

        Args:
            point: Geographic point location

        Returns:
            List of applicable regulations
        """
        try:
            # Convert to shapely Point
            shapely_point = Point(point.lon, point.lat)

            # Get regulations applicable to the point
            regulations = self.legal_framework.get_regulations_by_point(shapely_point)

            return [self._regulation_to_dict(r) for r in regulations]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    async def export_to_geojson(
        self,
        with_regulations: bool = Query(
            False, description="Include regulation count in properties"
        ),
    ) -> dict[str, Any]:
        """
        Export the legal framework's jurisdictions to GeoJSON.

        Args:
            with_regulations: Whether to include regulation counts

        Returns:
            GeoJSON data
        """
        try:
            # Export to GeoDataFrame
            gdf = self.legal_framework.export_to_geodataframe()

            if gdf.empty:
                return {
                    "status": "warning",
                    "message": "No jurisdictions with geometry found for GeoJSON export",
                    "geojson": None,
                }

            # Convert to GeoJSON
            geojson = json.loads(gdf.to_json())

            return {
                "status": "success",
                "message": "Jurisdictions exported to GeoJSON successfully",
                "feature_count": len(geojson["features"]),
                "geojson": geojson,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
