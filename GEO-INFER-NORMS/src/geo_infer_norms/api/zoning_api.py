"""
Zoning API module for geospatial zoning analysis and land use management.

This module provides API endpoints for interacting with zoning regulations,
land use classifications, and zoning district analysis.
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

from geo_infer_norms.core.zoning_analysis import ZoningAnalyzer, LandUseClassifier
from geo_infer_norms.models.zoning import ZoningCode, LandUseType, ZoningDistrict


# Pydantic models for API request/response
class GeometryModel(BaseModel):
    """Model for GeoJSON geometry data"""

    type: str
    coordinates: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ZoningCodeCreate(BaseModel):
    """Request model for creating a zoning code"""

    code: str = Field(..., description="Zoning code identifier")
    name: str = Field(..., description="Name of the zoning code")
    description: str | None = Field(None, description="Description of the zoning code")
    category: str = Field(
        ..., description="Category of zoning (residential, commercial, etc.)"
    )
    allowed_uses: list[str] | None = Field(
        None, description="List of allowed land uses"
    )
    conditional_uses: list[str] | None = Field(
        None, description="List of conditional land uses"
    )
    prohibited_uses: list[str] | None = Field(
        None, description="List of prohibited land uses"
    )
    max_height: float | None = Field(
        None, description="Maximum building height in meters"
    )
    max_density: float | None = Field(
        None, description="Maximum density (units/hectare)"
    )
    min_lot_size: float | None = Field(
        None, description="Minimum lot size in square meters"
    )
    max_lot_coverage: float | None = Field(
        None, description="Maximum lot coverage as percentage"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "R-1",
                "name": "Single Family Residential",
                "description": "Low density residential zoning for single family homes",
                "category": "residential",
                "allowed_uses": [
                    "single_family_dwelling",
                    "parks",
                    "religious_facilities",
                ],
                "conditional_uses": ["schools", "community_center"],
                "prohibited_uses": [
                    "commercial",
                    "industrial",
                    "multi_family_dwelling",
                ],
                "max_height": 10.5,
                "max_density": 12,
                "min_lot_size": 500,
                "max_lot_coverage": 40.0,
            }
        }
    )


class ZoningDistrictCreate(BaseModel):
    """Request model for creating a zoning district"""

    name: str = Field(..., description="Name of the zoning district")
    description: str | None = Field(None, description="Description of the district")
    zoning_code: str = Field(..., description="Zoning code applied to this district")
    jurisdiction_id: str | None = Field(
        None, description="ID of jurisdiction this district is in"
    )
    effective_date: datetime.datetime | None = Field(
        None, description="When the zoning takes effect"
    )
    geometry: GeometryModel = Field(..., description="GeoJSON geometry of the district")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "North Heights R-1 District",
                "description": "Single family residential district in North Heights neighborhood",
                "zoning_code": "R-1",
                "jurisdiction_id": "city-001",
                "effective_date": "2020-01-01T00:00:00Z",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                },
            }
        }
    )


class LandUseTypeCreate(BaseModel):
    """Request model for creating a land use type"""

    name: str = Field(..., description="Name of the land use type")
    category: str = Field(..., description="Category of land use")
    description: str | None = Field(
        None, description="Description of the land use type"
    )
    intensity: float | None = Field(
        None, ge=0.0, le=1.0, description="Intensity score (0.0 to 1.0)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Single Family Residential",
                "category": "residential",
                "description": "Detached single family homes on individual lots",
                "intensity": 0.3,
            }
        }
    )


class ZoningChangeRequest(BaseModel):
    """Request model for evaluating a zoning change"""

    district_id: str = Field(..., description="ID of the district to change")
    new_code: str = Field(..., description="New zoning code to apply")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"district_id": "district-001", "new_code": "R-2"}
        }
    )


class PointLocation(BaseModel):
    """Model for a geographic point location"""

    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude")


class LandClassificationRequest(BaseModel):
    """Request model for land use classification"""

    geojson_features: dict[str, Any] = Field(
        ..., description="GeoJSON features to classify"
    )
    feature_columns: list[str] = Field(
        ..., description="Feature columns to use for classification"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "geojson_features": {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [
                                    [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
                                ],
                            },
                            "properties": {
                                "building_count": 15,
                                "population_density": 6000,
                                "business_count": 2,
                            },
                        }
                    ],
                },
                "feature_columns": [
                    "building_count",
                    "population_density",
                    "business_count",
                ],
            }
        }
    )


class ZoningAPI:
    """API for zoning analysis and land use management"""

    def __init__(
        self,
        zoning_analyzer: ZoningAnalyzer | None = None,
        land_use_classifier: LandUseClassifier | None = None,
    ):
        """
        Initialize the ZoningAPI.

        Args:
            zoning_analyzer: Optional ZoningAnalyzer instance to use
            land_use_classifier: Optional LandUseClassifier instance to use
        """
        self.zoning_analyzer = zoning_analyzer or ZoningAnalyzer()
        self.land_use_classifier = land_use_classifier or LandUseClassifier()
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up API routes"""
        # Zoning code endpoints
        self.router.post("/codes", response_model=dict[str, Any])(
            self.create_zoning_code
        )
        self.router.get("/codes", response_model=list[dict[str, Any]])(
            self.list_zoning_codes
        )
        self.router.get("/codes/{code_id}", response_model=dict[str, Any])(
            self.get_zoning_code
        )

        # Zoning district endpoints
        self.router.post("/districts", response_model=dict[str, Any])(
            self.create_zoning_district
        )
        self.router.get("/districts", response_model=list[dict[str, Any]])(
            self.list_zoning_districts
        )
        self.router.get("/districts/{district_id}", response_model=dict[str, Any])(
            self.get_zoning_district
        )

        # Land use type endpoints
        self.router.post("/land-uses", response_model=dict[str, Any])(
            self.create_land_use_type
        )
        self.router.get("/land-uses", response_model=list[dict[str, Any]])(
            self.list_land_use_types
        )

        # Analysis endpoints
        self.router.post("/analyze/boundaries", response_model=dict[str, Any])(
            self.analyze_zoning_boundaries
        )
        self.router.post("/analyze/zoning-change", response_model=dict[str, Any])(
            self.evaluate_zoning_change
        )
        self.router.post(
            "/analyze/districts-at-point", response_model=list[dict[str, Any]]
        )(self.get_districts_at_point)

        # Land use classification endpoints
        self.router.post("/classify/land-use", response_model=dict[str, Any])(
            self.classify_land_use
        )

        # Compatibility endpoints
        self.router.get("/compatibility", response_model=float)(
            self.calculate_compatibility
        )

        # Export endpoints
        self.router.get("/export/geojson", response_model=dict[str, Any])(
            self.export_to_geojson
        )

    # Helper methods

    def _geometry_from_model(self, geometry_model: GeometryModel) -> BaseGeometry:
        """
        Convert a GeometryModel to a Shapely geometry object.

        Args:
            geometry_model: GeometryModel to convert

        Returns:
            Shapely geometry object
        """
        geojson = {
            "type": geometry_model.type,
            "coordinates": geometry_model.coordinates,
        }

        try:
            return shape(geojson)
        except Exception as e:
            raise ValueError(f"Invalid geometry: {str(e)}")

    def _zoning_code_to_dict(self, zoning_code: ZoningCode) -> dict[str, Any]:
        """
        Convert a ZoningCode object to a dictionary for API response.

        Args:
            zoning_code: ZoningCode object

        Returns:
            Dictionary representation
        """
        return {
            "code": zoning_code.code,
            "name": zoning_code.name,
            "description": zoning_code.description,
            "category": zoning_code.category,
            "allowed_uses": zoning_code.allowed_uses,
            "conditional_uses": zoning_code.conditional_uses,
            "prohibited_uses": zoning_code.prohibited_uses,
            "max_height": zoning_code.max_height,
            "max_density": zoning_code.max_density,
            "min_lot_size": zoning_code.min_lot_size,
            "max_lot_coverage": zoning_code.max_lot_coverage,
        }

    def _zoning_district_to_dict(self, district: ZoningDistrict) -> dict[str, Any]:
        """
        Convert a ZoningDistrict object to a dictionary for API response.

        Args:
            district: ZoningDistrict object

        Returns:
            Dictionary representation
        """
        geometry_dict = None
        if district.geometry:
            geometry_dict = json.loads(
                gpd.GeoSeries([district.geometry], crs="EPSG:4326").to_json()
            )["features"][0]["geometry"]

        return {
            "id": district.id,
            "name": district.name,
            "description": district.description,
            "zoning_code": district.zoning_code,
            "jurisdiction_id": district.jurisdiction_id,
            "effective_date": district.effective_date.isoformat()
            if district.effective_date
            else None,
            "geometry": geometry_dict,
        }

    def _land_use_type_to_dict(self, land_use_type: LandUseType) -> dict[str, Any]:
        """
        Convert a LandUseType object to a dictionary for API response.

        Args:
            land_use_type: LandUseType object

        Returns:
            Dictionary representation
        """
        return {
            "id": land_use_type.id,
            "name": land_use_type.name,
            "category": land_use_type.category,
            "description": land_use_type.description,
            "intensity": land_use_type.intensity,
        }

    # Zoning code endpoints

    async def create_zoning_code(self, code_data: ZoningCodeCreate) -> dict[str, Any]:
        """
        Create a new zoning code.

        Args:
            code_data: The zoning code data

        Returns:
            Success message with code ID
        """
        try:
            # Create ZoningCode object
            zoning_code = ZoningCode(
                code=code_data.code,
                name=code_data.name,
                description=code_data.description or "",
                category=code_data.category,
                jurisdiction_id="",
                allowed_uses=code_data.allowed_uses or [],
                conditional_uses=code_data.conditional_uses or [],
                prohibited_uses=code_data.prohibited_uses or [],
                max_height=code_data.max_height,
                max_density=code_data.max_density,
                min_lot_size=code_data.min_lot_size,
                max_lot_coverage=code_data.max_lot_coverage,
            )

            # Add to analyzer
            self.zoning_analyzer.add_zoning_code(zoning_code)

            return {
                "status": "success",
                "message": "Zoning code created successfully",
                "code": zoning_code.code,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def list_zoning_codes(
        self, category: str | None = Query(None, description="Filter by category")
    ) -> list[dict[str, Any]]:
        """
        List all zoning codes, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            List of zoning codes
        """
        try:
            codes = self.zoning_analyzer.zoning_codes

            if category:
                codes = [c for c in codes if c.category == category]

            return [self._zoning_code_to_dict(c) for c in codes]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def get_zoning_code(
        self, code_id: str = Path(..., description="Zoning code identifier")
    ) -> dict[str, Any]:
        """
        Get a zoning code by ID.

        Args:
            code_id: The zoning code identifier

        Returns:
            Zoning code details
        """
        try:
            code = self.zoning_analyzer.get_code_by_id(code_id)

            if not code:
                raise HTTPException(
                    status_code=404, detail=f"Zoning code {code_id} not found"
                )

            return self._zoning_code_to_dict(code)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Zoning district endpoints

    async def create_zoning_district(
        self, district_data: ZoningDistrictCreate
    ) -> dict[str, Any]:
        """
        Create a new zoning district.

        Args:
            district_data: The zoning district data

        Returns:
            Success message with district ID
        """
        try:
            # Convert geometry model to shapely geometry
            geometry = self._geometry_from_model(district_data.geometry)

            # Create ZoningDistrict object
            district = ZoningDistrict(
                id=f"district-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
                name=district_data.name,
                description=district_data.description or "",
                zoning_code=district_data.zoning_code,
                jurisdiction_id=district_data.jurisdiction_id or "",
                effective_date=district_data.effective_date,
                geometry=geometry,
            )

            # Add to analyzer
            self.zoning_analyzer.add_zoning_district(district)

            return {
                "status": "success",
                "message": "Zoning district created successfully",
                "district_id": district.id,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def list_zoning_districts(
        self,
        jurisdiction_id: str | None = Query(
            None, description="Filter by jurisdiction ID"
        ),
        zoning_code: str | None = Query(None, description="Filter by zoning code"),
        with_geometry: bool = Query(False, description="Include geometry in response"),
    ) -> list[dict[str, Any]]:
        """
        List all zoning districts, optionally filtered.

        Args:
            jurisdiction_id: Optional jurisdiction ID to filter by
            zoning_code: Optional zoning code to filter by
            with_geometry: Whether to include geometry in response

        Returns:
            List of zoning districts
        """
        try:
            districts = self.zoning_analyzer.zoning_districts

            if jurisdiction_id:
                districts = [
                    d for d in districts if d.jurisdiction_id == jurisdiction_id
                ]

            if zoning_code:
                districts = [d for d in districts if d.zoning_code == zoning_code]

            result = []
            for district in districts:
                district_dict = {
                    "id": district.id,
                    "name": district.name,
                    "description": district.description,
                    "zoning_code": district.zoning_code,
                    "jurisdiction_id": district.jurisdiction_id,
                    "effective_date": district.effective_date.isoformat()
                    if district.effective_date
                    else None,
                }

                if with_geometry and district.geometry:
                    geometry_dict = json.loads(
                        gpd.GeoSeries([district.geometry], crs="EPSG:4326").to_json()
                    )["features"][0]["geometry"]
                    district_dict["geometry"] = geometry_dict

                result.append(district_dict)

            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def get_zoning_district(
        self, district_id: str = Path(..., description="ID of the zoning district")
    ) -> dict[str, Any]:
        """
        Get a zoning district by ID.

        Args:
            district_id: The ID of the zoning district

        Returns:
            Zoning district details
        """
        try:
            district = self.zoning_analyzer.get_district_by_id(district_id)

            if not district:
                raise HTTPException(
                    status_code=404,
                    detail=f"Zoning district with ID {district_id} not found",
                )

            return self._zoning_district_to_dict(district)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Land use type endpoints

    async def create_land_use_type(
        self, land_use_data: LandUseTypeCreate
    ) -> dict[str, Any]:
        """
        Create a new land use type.

        Args:
            land_use_data: The land use type data

        Returns:
            Success message with land use type ID
        """
        try:
            # Create LandUseType object
            land_use_type = LandUseType(
                id=f"land-use-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
                name=land_use_data.name,
                category=land_use_data.category,
                description=land_use_data.description or "",
                intensity=land_use_data.intensity or 0.5,
            )

            # Add to classifier
            self.land_use_classifier.add_land_use_type(land_use_type)

            return {
                "status": "success",
                "message": "Land use type created successfully",
                "land_use_type_id": land_use_type.id,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def list_land_use_types(
        self, category: str | None = Query(None, description="Filter by category")
    ) -> list[dict[str, Any]]:
        """
        List all land use types, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            List of land use types
        """
        try:
            land_use_types = self.land_use_classifier.land_use_types

            if category:
                land_use_types = [
                    lt for lt in land_use_types if lt.category == category
                ]

            return [self._land_use_type_to_dict(lt) for lt in land_use_types]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Analysis endpoints

    async def analyze_zoning_boundaries(self) -> dict[str, Any]:
        """
        Analyze zoning district boundaries for potential conflicts.

        Returns:
            Analysis results
        """
        try:
            results = self.zoning_analyzer.analyze_zoning_boundaries()

            return results
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def evaluate_zoning_change(
        self, change_request: ZoningChangeRequest
    ) -> dict[str, Any]:
        """
        Evaluate the impact of changing a district's zoning code.

        Args:
            change_request: The zoning change request

        Returns:
            Evaluation results
        """
        try:
            results = self.zoning_analyzer.evaluate_zoning_change(
                district_id=change_request.district_id, new_code=change_request.new_code
            )

            return results
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def get_districts_at_point(
        self, point: PointLocation
    ) -> list[dict[str, Any]]:
        """
        Get all zoning districts that contain a specific point.

        Args:
            point: Geographic point location

        Returns:
            List of zoning districts containing the point
        """
        try:
            # Convert to shapely Point
            shapely_point = Point(point.lon, point.lat)

            # Get districts containing the point
            districts = self.zoning_analyzer.get_zoning_at_point(shapely_point)

            return [self._zoning_district_to_dict(d) for d in districts]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Land use classification endpoints

    async def classify_land_use(
        self, classification_request: LandClassificationRequest
    ) -> dict[str, Any]:
        """
        Classify land use based on features.

        Args:
            classification_request: The land use classification request

        Returns:
            Classification results
        """
        try:
            # Convert GeoJSON to GeoDataFrame
            gdf = gpd.GeoDataFrame.from_features(
                classification_request.geojson_features["features"], crs="EPSG:4326"
            )

            # Classify land use
            result_gdf = self.land_use_classifier.classify_land_use(
                features_gdf=gdf, feature_columns=classification_request.feature_columns
            )

            # Convert back to GeoJSON
            result_geojson = json.loads(result_gdf.to_json())

            return {
                "status": "success",
                "message": "Land use classification completed",
                "feature_count": len(result_geojson["features"]),
                "geojson": result_geojson,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Compatibility endpoints

    async def calculate_compatibility(
        self,
        code1: str = Query(..., description="First zoning code"),
        code2: str = Query(..., description="Second zoning code"),
    ) -> float:
        """
        Calculate the compatibility score between two zoning codes.

        Args:
            code1: The first zoning code
            code2: The second zoning code

        Returns:
            Compatibility score (0-1)
        """
        try:
            compatibility = self.zoning_analyzer.calculate_compatibility(code1, code2)

            return compatibility
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Export endpoints

    async def export_to_geojson(
        self,
        include_codes: bool = Query(
            True, description="Include zoning codes in properties"
        ),
    ) -> dict[str, Any]:
        """
        Export zoning districts to GeoJSON.

        Args:
            include_codes: Whether to include zoning code information

        Returns:
            GeoJSON data
        """
        try:
            # Create a GeoDataFrame from districts
            districts = self.zoning_analyzer.zoning_districts

            if not districts:
                return {
                    "status": "warning",
                    "message": "No zoning districts found for GeoJSON export",
                    "geojson": None,
                }

            data = []
            for district in districts:
                if district.geometry is not None:
                    district_dict = {
                        "id": district.id,
                        "name": district.name,
                        "zoning_code": district.zoning_code,
                        "jurisdiction_id": district.jurisdiction_id,
                        "geometry": district.geometry,
                    }

                    if include_codes:
                        code = self.zoning_analyzer.get_code_by_id(district.zoning_code)
                        if code:
                            district_dict["code_name"] = code.name
                            district_dict["code_category"] = code.category

                    data.append(district_dict)

            if not data:
                return {
                    "status": "warning",
                    "message": "No zoning districts with geometry found for GeoJSON export",
                    "geojson": None,
                }

            gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

            # Convert to GeoJSON
            geojson = json.loads(gdf.to_json())

            return {
                "status": "success",
                "message": "Zoning districts exported to GeoJSON successfully",
                "feature_count": len(geojson["features"]),
                "geojson": geojson,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
