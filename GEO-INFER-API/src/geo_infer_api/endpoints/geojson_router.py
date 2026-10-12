"""
API endpoints for GeoJSON polygon operations.

Implements OGC API Features compatible endpoints for working with GeoJSON polygons.
"""

import math
from typing import Any

from fastapi import APIRouter, Depends, Query, Path, status
from pydantic import BaseModel

from geo_infer_api.core.config import get_settings, Settings
from geo_infer_api.core.exceptions import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    ValidationError,
)
from geo_infer_api.models.geojson import (
    GeoJSONType,
    PolygonFeature,
    PolygonFeatureCollection,
)
from geo_infer_api.utils.geojson_helpers import (
    calculate_polygon_area,
    polygon_contains_point,
    simplify_polygon,
    create_buffer,
    calculate_intersection,
    calculate_union,
    calculate_distance,
)

# Create router
router = APIRouter()

# In-memory storage for demo purposes
# In a production application, replace with a persistent database backend.
POLYGON_FEATURES: dict[str, PolygonFeature] = {}


# ---------------------------------------------------------------------------
# Request body models for operation endpoints
# ---------------------------------------------------------------------------


class MultiPolygonRequest(BaseModel):
    """Request body containing multiple polygon features for set operations."""

    polygons: list[PolygonFeature]


class DistanceRequest(BaseModel):
    """Request body for polygon distance calculation."""

    polygon1: PolygonFeature
    polygon2: PolygonFeature


# ---------------------------------------------------------------------------
# OGC API Features – Collections
# ---------------------------------------------------------------------------


@router.get("/collections", summary="List available feature collections")
async def list_collections(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    List available feature collections.

    Follows the OGC API Features standard for listing collections.
    """
    return {
        "collections": [
            {
                "id": "polygons",
                "title": "GeoJSON Polygons",
                "description": "Collection of polygon features",
                "links": [
                    {
                        "href": f"{settings.api_prefix}/collections/polygons",
                        "rel": "self",
                        "type": "application/json",
                        "title": "This collection",
                    },
                    {
                        "href": f"{settings.api_prefix}/collections/polygons/items",
                        "rel": "items",
                        "type": "application/geo+json",
                        "title": "Items in this collection",
                    },
                ],
            }
        ],
        "links": [
            {
                "href": f"{settings.api_prefix}/collections",
                "rel": "self",
                "type": "application/json",
                "title": "Collections",
            }
        ],
    }


@router.get("/collections/polygons", summary="Get polygon collection metadata")
async def get_polygon_collection(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Get metadata about the polygon collection.

    Follows the OGC API Features standard for describing a collection.
    """
    return {
        "id": "polygons",
        "title": "GeoJSON Polygons",
        "description": "Collection of polygon features",
        "extent": {
            "spatial": {"bbox": [[-180, -90, 180, 90]]},
            "temporal": {"interval": [["2020-01-01T00:00:00Z", None]]},
        },
        "links": [
            {
                "href": f"{settings.api_prefix}/collections/polygons",
                "rel": "self",
                "type": "application/json",
                "title": "This collection",
            },
            {
                "href": f"{settings.api_prefix}/collections/polygons/items",
                "rel": "items",
                "type": "application/geo+json",
                "title": "Items in this collection",
            },
        ],
    }


@router.get(
    "/collections/polygons/items",
    response_model=PolygonFeatureCollection,
    response_model_exclude_none=True,
    summary="List polygon features",
)
async def list_polygon_features(
    bbox: str | None = Query(
        None,
        description="Bounding box (minLon,minLat,maxLon,maxLat)",
    ),
    limit: int = Query(
        10, ge=1, le=1000, description="Maximum number of features to return"
    ),
) -> PolygonFeatureCollection:
    """
    List polygon features with optional bounding-box filtering.

    Follows the OGC API Features standard for listing items in a collection.
    """
    features = list(POLYGON_FEATURES.values())

    if bbox:
        try:
            values = [float(value.strip()) for value in bbox.split(",")]
            if len(values) != 4 or not all(math.isfinite(value) for value in values):
                raise ValueError
            min_lon, min_lat, max_lon, max_lat = values
            if not (
                -180 <= min_lon <= max_lon <= 180 and -90 <= min_lat <= max_lat <= 90
            ):
                raise ValueError
        except ValueError:
            raise BadRequestError(
                "Invalid bbox format. Expected 'minLon,minLat,maxLon,maxLat'",
                field="bbox",
            ) from None

        def polygon_intersects_bbox(polygon_feature: PolygonFeature) -> bool:
            coordinates = [
                position
                for ring in polygon_feature.geometry.coordinates
                for position in ring
            ]
            if not coordinates:
                return False
            polygon_lons = [position[0] for position in coordinates]
            polygon_lats = [position[1] for position in coordinates]
            return (
                min(polygon_lons) <= max_lon
                and max(polygon_lons) >= min_lon
                and min(polygon_lats) <= max_lat
                and max(polygon_lats) >= min_lat
            )

        features = [f for f in features if polygon_intersects_bbox(f)]

    features = features[:limit]

    return PolygonFeatureCollection(
        type=GeoJSONType.FEATURE_COLLECTION,
        features=features,
    )


@router.get(
    "/collections/polygons/items/{feature_id}",
    response_model=PolygonFeature,
    response_model_exclude_none=True,
    summary="Get a specific polygon feature",
)
async def get_polygon_feature(
    feature_id: str = Path(..., description="ID of the feature to retrieve"),
) -> PolygonFeature:
    """Get a specific polygon feature by ID."""
    if feature_id not in POLYGON_FEATURES:
        raise NotFoundError("Polygon feature", feature_id)
    return POLYGON_FEATURES[feature_id]


@router.post(
    "/collections/polygons/items",
    response_model=PolygonFeature,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new polygon feature",
)
async def create_polygon_feature_endpoint(
    feature: PolygonFeature,
    settings: Settings = Depends(get_settings),
) -> PolygonFeature:
    """Create a new polygon feature.

    The in-memory store is capped at ``settings.polygon_store_max_size``
    entries; creation beyond the cap is rejected with HTTP 409 rather than
    letting the store grow without bound.
    """
    if feature.id is None:
        raise ValidationError("Feature must have an ID", field="id")

    if feature.id in POLYGON_FEATURES:
        raise ConflictError("Polygon feature", "already exists", str(feature.id))

    if len(POLYGON_FEATURES) >= settings.polygon_store_max_size:
        raise ConflictError(
            "Polygon feature store",
            "at maximum capacity",
            str(feature.id),
            additional_info={"max_size": settings.polygon_store_max_size},
        )

    POLYGON_FEATURES[str(feature.id)] = feature
    return feature


@router.put(
    "/collections/polygons/items/{feature_id}",
    response_model=PolygonFeature,
    summary="Update a polygon feature",
)
async def update_polygon_feature(
    feature: PolygonFeature,
    feature_id: str = Path(..., description="ID of the feature to update"),
) -> PolygonFeature:
    """Update an existing polygon feature."""
    if feature_id not in POLYGON_FEATURES:
        raise NotFoundError("Polygon feature", feature_id)

    if feature.id and str(feature.id) != feature_id:
        raise ValidationError(
            "Feature ID in body does not match path parameter", field="id"
        )

    feature_copy = feature.model_copy(update={"id": feature_id})
    POLYGON_FEATURES[feature_id] = feature_copy
    return feature_copy


@router.delete(
    "/collections/polygons/items/{feature_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a polygon feature",
)
async def delete_polygon_feature(
    feature_id: str = Path(..., description="ID of the feature to delete"),
) -> None:
    """Delete a polygon feature."""
    if feature_id not in POLYGON_FEATURES:
        raise NotFoundError("Polygon feature", feature_id)
    del POLYGON_FEATURES[feature_id]
    return None


# ---------------------------------------------------------------------------
# Polygon geometry operations
# ---------------------------------------------------------------------------


@router.post("/operations/polygon/area", summary="Calculate polygon area")
async def calculate_area(feature: PolygonFeature) -> dict[str, Any]:
    """
    Calculate the approximate area of a polygon in square kilometers.

    Uses a planar shoelace approximation; accurate for small areas.
    """
    area = calculate_polygon_area(feature.geometry)
    return {
        "area_sq_km": area,
        "feature_id": feature.id,
        "method": "planar",
    }


@router.post(
    "/operations/polygon/simplify",
    response_model=PolygonFeature,
    summary="Simplify a polygon",
)
async def simplify_polygon_endpoint(
    feature: PolygonFeature,
    tolerance: float = Query(
        0.01, ge=0.001, le=1.0, description="Simplification tolerance"
    ),
) -> PolygonFeature:
    """
    Simplify a polygon using the Ramer-Douglas-Peucker algorithm.

    Higher tolerance values produce simpler polygons.
    """
    simplified_geometry = simplify_polygon(feature.geometry, tolerance)
    return PolygonFeature(
        type=GeoJSONType.FEATURE,
        geometry=simplified_geometry,
        properties=feature.properties,
        id=feature.id,
    )


@router.post(
    "/operations/polygon/contains", summary="Check if a polygon contains a point"
)
async def check_polygon_contains_point(
    feature: PolygonFeature,
    lon: float = Query(..., ge=-180, le=180, description="Longitude of the point"),
    lat: float = Query(..., ge=-90, le=90, description="Latitude of the point"),
) -> dict[str, Any]:
    """Check if a polygon contains a point using the ray casting algorithm."""
    contains = polygon_contains_point(feature.geometry, (lon, lat))
    return {
        "contains": contains,
        "feature_id": feature.id,
        "point": [lon, lat],
        "method": "ray_casting",
    }


@router.post(
    "/operations/polygon/buffer",
    response_model=PolygonFeature,
    summary="Create a buffer around a polygon",
)
async def create_buffer_endpoint(
    feature: PolygonFeature,
    distance: float = Query(
        ..., ge=0, description="Buffer distance in the specified unit"
    ),
    unit: str = Query(
        "kilometers",
        enum=["meters", "kilometers", "miles"],
        description="Unit for the buffer distance",
    ),
    segments: int = Query(
        16, ge=8, le=100, description="Segments for buffer approximation"
    ),
) -> PolygonFeature:
    """
    Create a bounding-box buffer zone around a polygon at a specified distance.

    Returns an axis-aligned rectangle expanded by `distance` on each side.
    """
    buffered_geometry = create_buffer(feature.geometry, distance, unit, segments)
    base_props = feature.properties or {}
    return PolygonFeature(
        type=GeoJSONType.FEATURE,
        geometry=buffered_geometry,
        properties={
            **base_props,
            "buffer_distance": distance,
            "buffer_unit": unit,
            "buffer_segments": segments,
        },
        id=f"{feature.id}_buffer" if feature.id else None,
    )


@router.post(
    "/operations/polygon/intersection",
    response_model=PolygonFeature,
    summary="Calculate bounding-box intersection of multiple polygons",
)
async def calculate_intersection_endpoint(
    request: MultiPolygonRequest,
) -> PolygonFeature:
    """
    Calculate the bounding-box intersection of multiple polygon features.

    Raises HTTP 422 if the polygons do not overlap.
    """
    if len(request.polygons) < 2:
        raise ValidationError("At least 2 polygons required for intersection")

    try:
        intersection_geometry = calculate_intersection(
            [pf.geometry for pf in request.polygons]
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    return PolygonFeature(
        type=GeoJSONType.FEATURE,
        geometry=intersection_geometry,
        properties={"operation": "intersection"},
        id="intersection_result",
    )


@router.post(
    "/operations/polygon/union",
    response_model=PolygonFeature,
    summary="Calculate bounding-box union of multiple polygons",
)
async def calculate_union_endpoint(request: MultiPolygonRequest) -> PolygonFeature:
    """
    Calculate the bounding-box union of multiple polygon features.
    """
    if len(request.polygons) < 2:
        raise ValidationError("At least 2 polygons required for union")

    union_geometry = calculate_union([pf.geometry for pf in request.polygons])

    return PolygonFeature(
        type=GeoJSONType.FEATURE,
        geometry=union_geometry,
        properties={"operation": "union"},
        id="union_result",
    )


@router.post(
    "/operations/polygon/distance",
    summary="Calculate centroid distance between two polygons",
)
async def calculate_distance_endpoint(request: DistanceRequest) -> dict[str, Any]:
    """
    Calculate the centroid-to-centroid distance between two polygon features.
    """
    distance = calculate_distance(
        request.polygon1.geometry, request.polygon2.geometry, method="centroid"
    )
    return {
        "distance_km": distance,
        "unit": "kilometers",
        "method": "centroid",
    }
