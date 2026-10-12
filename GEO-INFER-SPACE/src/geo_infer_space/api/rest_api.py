"""
FastAPI REST API for GEO-INFER-SPACE spatial services.

This module provides HTTP endpoints for accessing spatial analysis capabilities
with automatic documentation, validation, and error handling.
"""

import logging
from typing import Any, cast
import math
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import geopandas as gpd
import pandas as pd
import json
from shapely.geometry import shape, mapping

from .schemas import (
    SpatialAnalysisResponse,
    BufferAnalysisRequest,
    ProximityAnalysisRequest,
    InterpolationRequest,
    ClusteringRequest,
    HotspotRequest,
    NetworkAnalysisRequest,
    H3AnalysisRequest,
    ErrorResponse,
)
from .. import __version__

from ..analytics import (
    proximity_analysis,
    spatial_interpolation,
    clustering_analysis,
    hotspot_detection,
    service_area,
    network_connectivity,
    shortest_path,
    routing_analysis,
    accessibility_analysis,
)

from ..utils.h3_utils import (
    compact_cells,
    grid_disk,
    get_resolution,
    is_valid_cell,
    polygon_to_cells,
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GEO-INFER-SPACE API",
    description="Advanced spatial analysis and processing services",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["spatial"])


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(  # type: ignore[call-arg]
            error="InternalServerError",
            message="An internal server error occurred",
            details={"exception": str(exc)},
        ).model_dump(),
    )


def geojson_to_gdf(
    geojson_data: dict[str, Any], crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """Convert GeoJSON data to GeoDataFrame."""
    try:
        if geojson_data.get("type") == "FeatureCollection":
            return gpd.GeoDataFrame.from_features(geojson_data["features"], crs=crs)
        elif geojson_data.get("type") == "Feature":
            return gpd.GeoDataFrame.from_features([geojson_data], crs=crs)
        else:
            # Assume it's a geometry
            geom = shape(geojson_data)
            return gpd.GeoDataFrame([{"geometry": geom}], crs=crs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid GeoJSON data: {e}") from e


def gdf_to_geojson(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """Convert GeoDataFrame to GeoJSON."""
    try:
        return cast(dict[str, Any], json.loads(gdf.to_json()))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to convert to GeoJSON: {e}"
        ) from e


def _records_to_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert tabular network results to JSON-safe records."""
    records = cast(list[dict[str, Any]], frame.to_dict(orient="records"))
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, float) and not math.isfinite(value):
                record[key] = None
    return records


def _reproject_geometry_series(
    series: gpd.GeoSeries, source_crs: Any, target_crs: Any
) -> gpd.GeoSeries:
    """Reproject a geometry series without NumPy scalar warnings.

    The installed GeoPandas/pyproj combination passes one-element NumPy
    arrays into ``Transformer.transform``. Recent NumPy versions warn on the
    scalar conversion performed by that older path. Supplying ordinary Python
    lists keeps the transformation vectorized and warning-free.
    """
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    def transform_coordinates(x: Any, y: Any, z: Any = None) -> tuple[Any, Any]:
        try:
            x_values = [float(value) for value in x]
            y_values = [float(value) for value in y]
        except TypeError:
            return transformer.transform(float(x), float(y))

        transformed_x, transformed_y = transformer.transform(x_values, y_values)
        return transformed_x, transformed_y

    return series.map(
        lambda geometry: shapely_transform(transform_coordinates, geometry)
    ).set_crs(target_crs)


def _buffer_geometry(
    gdf: gpd.GeoDataFrame, distance: float
) -> tuple[gpd.GeoSeries, dict[str, Any]]:
    """Buffer safely, using meters for geographic input CRS values.

    Shapely buffers coordinates in a planar coordinate system. Calling it
    directly on longitude/latitude raises a warning (and the repository test
    contract promotes that warning to an error), while also producing a
    misleading result. Project geographic input to a local UTM CRS, buffer in
    meters, and transform the result back to the request CRS.
    """
    if gdf.crs is None or not gdf.crs.is_geographic:
        return gdf.geometry.buffer(distance), {
            "buffer_crs": str(gdf.crs) if gdf.crs is not None else None,
            "distance_units": "crs units",
        }

    # Derive the local projection from bounds instead of GeoPandas'
    # ``estimate_utm_crs`` helper. Older GeoPandas releases implement that
    # helper through a NumPy scalar conversion that is now a warning/error
    # under the repository's strict warning policy.
    wgs84_geometry = _reproject_geometry_series(gdf.geometry, gdf.crs, "EPSG:4326")
    minx, miny, maxx, maxy = wgs84_geometry.total_bounds
    longitude = float((minx + maxx) / 2.0)
    latitude = float((miny + maxy) / 2.0)
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Geographic geometries have invalid bounds")

    if -80 <= latitude <= 84:
        zone = min(60, max(1, int((longitude + 180) // 6) + 1))
        epsg = (32600 if latitude >= 0 else 32700) + zone
        metric_crs = f"EPSG:{epsg}"
    else:
        # UTM does not cover the polar regions; a local azimuthal equidistant
        # projection keeps the buffer distance metric there as well.
        metric_crs = (
            f"+proj=aeqd +lat_0={latitude} +lon_0={longitude} "
            "+datum=WGS84 +units=m +no_defs"
        )

    projected_geometry = _reproject_geometry_series(gdf.geometry, gdf.crs, metric_crs)
    buffered = projected_geometry.buffer(distance)
    return _reproject_geometry_series(buffered, metric_crs, gdf.crs), {
        "buffer_crs": str(metric_crs),
        "distance_units": "meters",
    }


@router.post("/buffer", response_model=SpatialAnalysisResponse)
async def buffer_analysis_endpoint(
    request: BufferAnalysisRequest,
) -> SpatialAnalysisResponse:
    """
    Perform buffer analysis on geometries.

    Creates buffers around input geometries with specified distance.
    Optionally dissolves overlapping buffers into single features.
    """
    try:
        # Convert GeoJSON to GeoDataFrame
        gdf = geojson_to_gdf(request.data.model_dump(), cast(str, request.crs))

        # Project geographic input locally so that buffers are metric and
        # accurate instead of operating directly on longitude/latitude.
        buffered_gdf = gdf.copy()
        buffered_gdf["geometry"], buffer_metadata = _buffer_geometry(
            gdf, request.buffer_distance
        )

        # Dissolve if requested
        if request.dissolve:
            from shapely.ops import unary_union

            dissolved_geom = unary_union(buffered_gdf.geometry.tolist())
            buffered_gdf = gpd.GeoDataFrame([{"geometry": dissolved_geom}], crs=gdf.crs)

        # Convert back to GeoJSON
        result_geojson = gdf_to_geojson(buffered_gdf)

        return SpatialAnalysisResponse(
            success=True,
            result=result_geojson,
            message=f"Buffer analysis completed with distance {request.buffer_distance}",
            metadata={
                "buffer_distance": request.buffer_distance,
                "dissolved": request.dissolve,
                "num_features": len(buffered_gdf),
                **buffer_metadata,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Buffer analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/proximity", response_model=SpatialAnalysisResponse)
async def proximity_analysis_endpoint(
    request: ProximityAnalysisRequest,
) -> SpatialAnalysisResponse:
    """
    Perform proximity analysis between two sets of geometries.

    Calculates distance metrics and identifies nearest features
    between source and target geometries.
    """
    try:
        # Convert GeoJSON to GeoDataFrames
        request_crs = cast(str, request.crs)
        source_gdf = geojson_to_gdf(request.source_data.model_dump(), request_crs)
        target_gdf = geojson_to_gdf(request.target_data.model_dump(), request_crs)

        # Perform proximity analysis
        result_gdf = proximity_analysis(source_gdf, target_gdf, request.max_distance)

        # Convert to GeoJSON
        result_geojson = gdf_to_geojson(result_gdf)

        return SpatialAnalysisResponse(
            success=True,
            result=result_geojson,
            message="Proximity analysis completed",
            metadata={
                "num_source_features": len(source_gdf),
                "num_target_features": len(target_gdf),
                "num_results": len(result_gdf),
                "max_distance": request.max_distance,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Proximity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/interpolation", response_model=SpatialAnalysisResponse)
async def interpolation_endpoint(
    request: InterpolationRequest,
) -> SpatialAnalysisResponse:
    """
    Perform spatial interpolation on point data.

    Creates a continuous surface from discrete point observations
    using various interpolation methods (IDW, Kriging, RBF, etc.).
    """
    try:
        # Convert points to GeoDataFrame
        points_gdf = geojson_to_gdf(request.points.model_dump(), cast(str, request.crs))

        # Validate value column exists
        if request.value_column not in points_gdf.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{request.value_column}' not found in point data",
            )

        # Perform interpolation
        result_gdf = spatial_interpolation(
            points_gdf=points_gdf,
            value_column=request.value_column,
            grid_bounds=cast(tuple[float, float, float, float], tuple(request.bounds)),
            grid_resolution=request.resolution,
            method=request.method,
            **(request.parameters or {}),
        )

        # Convert to GeoJSON
        result_geojson = gdf_to_geojson(result_gdf)

        return SpatialAnalysisResponse(
            success=True,
            result=result_geojson,
            message=f"Spatial interpolation completed using {request.method}",
            metadata={
                "method": request.method,
                "resolution": request.resolution,
                "num_input_points": len(points_gdf),
                "num_grid_points": len(result_gdf),
                "bounds": request.bounds,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/clustering", response_model=SpatialAnalysisResponse)
async def clustering_endpoint(
    request: ClusteringRequest,
) -> SpatialAnalysisResponse:
    """
    Perform spatial clustering analysis on point data.

    Groups points into clusters based on spatial proximity
    using various clustering algorithms (DBSCAN, K-means, etc.).
    """
    try:
        # Convert points to GeoDataFrame
        points_gdf = geojson_to_gdf(request.points.model_dump(), cast(str, request.crs))

        # Perform clustering
        result_gdf = clustering_analysis(
            points_gdf=points_gdf,
            method=request.method,
            **(request.parameters or {}),
        )

        # Convert to GeoJSON
        result_geojson = gdf_to_geojson(result_gdf)

        # Calculate cluster statistics
        cluster_stats = {}
        if "cluster" in result_gdf.columns:
            cluster_counts = result_gdf["cluster"].value_counts()
            cluster_stats = {
                "num_clusters": len(cluster_counts),
                "largest_cluster": int(cluster_counts.max()),
                "smallest_cluster": int(cluster_counts.min()),
                "noise_points": int(cluster_counts.get(-1, 0)),
            }

        return SpatialAnalysisResponse(
            success=True,
            result=result_geojson,
            message=f"Clustering analysis completed using {request.method}",
            metadata={
                "method": request.method,
                "num_points": len(points_gdf),
                "cluster_statistics": cluster_stats,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clustering analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/hotspots", response_model=SpatialAnalysisResponse)
async def hotspot_detection_endpoint(
    request: HotspotRequest,
) -> SpatialAnalysisResponse:
    """
    Detect spatial hotspots and coldspots in point data.

    Identifies statistically significant clusters of high or low values
    using spatial statistics methods (Getis-Ord Gi*, Local Moran's I, etc.).
    """
    try:
        # Convert points to GeoDataFrame
        points_gdf = geojson_to_gdf(request.points.model_dump(), cast(str, request.crs))

        # Validate value column if provided
        if request.value_column and request.value_column not in points_gdf.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{request.value_column}' not found in point data",
            )

        # Perform hotspot detection
        result_gdf = hotspot_detection(
            points_gdf=points_gdf,
            value_column=request.value_column,
            method=request.method,
            **(request.parameters or {}),
        )

        # Convert to GeoJSON
        result_geojson = gdf_to_geojson(result_gdf)

        # Calculate hotspot statistics
        hotspot_stats = {}
        if "hotspot_type" in result_gdf.columns:
            hotspot_counts = result_gdf["hotspot_type"].value_counts()
            hotspot_stats = {
                "hot_spots": int(hotspot_counts.get("Hot Spot", 0)),
                "cold_spots": int(hotspot_counts.get("Cold Spot", 0)),
                "not_significant": int(hotspot_counts.get("Not Significant", 0)),
            }

        return SpatialAnalysisResponse(
            success=True,
            result=result_geojson,
            message=f"Hotspot detection completed using {request.method}",
            metadata={
                "method": request.method,
                "value_column": request.value_column,
                "num_points": len(points_gdf),
                "hotspot_statistics": hotspot_stats,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hotspot detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/network", response_model=SpatialAnalysisResponse)
async def network_analysis_endpoint(
    request: NetworkAnalysisRequest,
) -> SpatialAnalysisResponse:
    """
    Perform network analysis operations.

    Analyzes transportation networks for routing, accessibility,
    and connectivity using graph-based algorithms.
    """
    try:
        # Convert network to GeoDataFrame
        request_crs = cast(str, request.crs)
        network_gdf = geojson_to_gdf(request.network.model_dump(), request_crs)
        parameters = request.parameters or {}

        if request.analysis_type == "connectivity":
            # Network connectivity analysis
            result = network_connectivity(network_gdf=network_gdf, **parameters)

            return SpatialAnalysisResponse(
                success=True,
                result=result,
                message="Network connectivity analysis completed",
                metadata={
                    "analysis_type": request.analysis_type,
                    "num_edges": len(network_gdf),
                },
            )

        elif request.analysis_type == "service_area":
            # Service area analysis requires center point
            if not request.origins or len(request.origins.features) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Service area analysis requires origin points",
                )

            origins_gdf = geojson_to_gdf(request.origins.model_dump(), request_crs)
            center_point = origins_gdf.geometry.iloc[0]

            max_distance = parameters.get("max_distance", 1000)
            weight_column = parameters.get("weight_column", "length")

            result_gdf = service_area(
                network_gdf=network_gdf,
                center_point=center_point,
                max_distance=max_distance,
                weight_column=weight_column,
            )

            result_geojson = gdf_to_geojson(result_gdf)

            return SpatialAnalysisResponse(
                success=True,
                result=result_geojson,
                message="Service area analysis completed",
                metadata={
                    "analysis_type": request.analysis_type,
                    "max_distance": max_distance,
                    "num_areas": len(result_gdf),
                },
            )

        elif request.analysis_type == "shortest_path":
            if not request.origins or not request.destinations:
                raise HTTPException(
                    status_code=400,
                    detail="shortest_path requires origins and destinations",
                )
            origins_gdf = geojson_to_gdf(request.origins.model_dump(), request_crs)
            destinations_gdf = geojson_to_gdf(
                request.destinations.model_dump(), request_crs
            )
            if origins_gdf.empty or destinations_gdf.empty:
                raise HTTPException(
                    status_code=400,
                    detail="shortest_path requires at least one origin and destination",
                )
            weight_column = parameters.get("weight_column", "length")
            result = shortest_path(
                network_gdf=network_gdf,
                start_point=origins_gdf.geometry.iloc[0],
                end_point=destinations_gdf.geometry.iloc[0],
                weight_column=weight_column,
                impedance_factor=parameters.get("impedance_factor", 1.0),
            )
            if result.get("path_geometry") is not None:
                result["path_geometry"] = mapping(result["path_geometry"])
            return SpatialAnalysisResponse(
                success=True,
                result=result,
                message="Shortest path analysis completed",
                metadata={
                    "analysis_type": request.analysis_type,
                    "weight_column": weight_column,
                },
            )

        elif request.analysis_type == "routing":
            if not request.origins or not request.destinations:
                raise HTTPException(
                    status_code=400,
                    detail="routing requires origins and destinations",
                )
            origins_gdf = geojson_to_gdf(request.origins.model_dump(), request_crs)
            destinations_gdf = geojson_to_gdf(
                request.destinations.model_dump(), request_crs
            )
            weight_column = parameters.get("weight_column", "length")
            result = routing_analysis(
                network_gdf=network_gdf,
                origins=list(origins_gdf.geometry),
                destinations=list(destinations_gdf.geometry),
                weight_column=weight_column,
            )
            return SpatialAnalysisResponse(
                success=True,
                result={"records": _records_to_json(result)},
                message="Routing analysis completed",
                metadata={
                    "analysis_type": request.analysis_type,
                    "weight_column": weight_column,
                },
            )

        elif request.analysis_type == "accessibility":
            if not request.origins or not request.destinations:
                raise HTTPException(
                    status_code=400,
                    detail="accessibility requires origins and destinations",
                )
            origins_gdf = geojson_to_gdf(request.origins.model_dump(), request_crs)
            destinations_gdf = geojson_to_gdf(
                request.destinations.model_dump(), request_crs
            )
            weight_column = parameters.get("weight_column", "length")
            result = accessibility_analysis(
                network_gdf=network_gdf,
                origins=list(origins_gdf.geometry),
                destinations=list(destinations_gdf.geometry),
                max_distance=parameters.get("max_distance", 1000),
                weight_column=weight_column,
            )
            return SpatialAnalysisResponse(
                success=True,
                result={"records": _records_to_json(result)},
                message="Accessibility analysis completed",
                metadata={
                    "analysis_type": request.analysis_type,
                    "max_distance": parameters.get("max_distance", 1000),
                    "weight_column": weight_column,
                },
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Analysis type '{request.analysis_type}' is unsupported",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Network analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/h3", response_model=SpatialAnalysisResponse)
async def h3_analysis_endpoint(
    request: H3AnalysisRequest,
) -> SpatialAnalysisResponse:
    """
    Perform H3 hexagonal grid operations.

    Converts geometries to H3 cells, performs grid operations,
    and provides H3-based spatial indexing capabilities.
    """
    try:
        parameters = request.parameters or {}
        if request.operation == "polygon_to_cells":
            # Convert geometry to H3 cells
            if request.geometry is None:
                raise HTTPException(
                    status_code=400,
                    detail="polygon_to_cells requires a polygon geometry",
                )
            geom_dict = request.geometry.model_dump()

            h3_cells = polygon_to_cells(geom_dict, request.resolution)

            # Create GeoJSON features for each cell
            from ..utils.h3_utils import cell_to_latlng_boundary

            features = []
            for cell in h3_cells:
                boundary = cell_to_latlng_boundary(cell)
                # H3 returns (lat, lng); GeoJSON uses closed [lng, lat] rings.
                ring = [[lng, lat] for lat, lng in boundary]
                if ring and ring[0] != ring[-1]:
                    ring.append(ring[0])
                coords = [ring]

                feature = {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": coords},
                    "properties": {"h3_index": cell, "resolution": request.resolution},
                }
                features.append(feature)

            result = {"type": "FeatureCollection", "features": features}

            return SpatialAnalysisResponse(
                success=True,
                result=result,
                message="H3 polygon conversion completed",
                metadata={
                    "operation": request.operation,
                    "resolution": request.resolution,
                    "num_cells": len(h3_cells),
                },
            )

        elif request.operation == "grid_disk":
            # Grid disk operation requires center cell
            center_cell = parameters.get("center_cell")
            k = parameters.get("k", 1)

            if not center_cell:
                raise HTTPException(
                    status_code=400,
                    detail="Grid disk operation requires 'center_cell' parameter",
                )
            if not is_valid_cell(center_cell):
                raise HTTPException(
                    status_code=400, detail="center_cell is not a valid H3 cell"
                )
            if isinstance(k, bool) or not isinstance(k, int) or k < 0:
                raise HTTPException(
                    status_code=400, detail="k must be a non-negative integer"
                )

            disk_cells = grid_disk(center_cell, k)

            return SpatialAnalysisResponse(
                success=True,
                result={"cells": disk_cells},
                message="H3 grid disk completed",
                metadata={
                    "operation": request.operation,
                    "center_cell": center_cell,
                    "k": k,
                    "num_cells": len(disk_cells),
                },
            )

        elif request.operation == "compact_cells":
            cells = parameters.get("cells")
            if not isinstance(cells, list) or not cells:
                raise HTTPException(
                    status_code=400,
                    detail="compact_cells requires a non-empty 'cells' list",
                )
            invalid = [
                cell
                for cell in cells
                if not isinstance(cell, str) or not is_valid_cell(cell)
            ]
            if invalid:
                raise HTTPException(
                    status_code=400,
                    detail=f"cells contains invalid H3 indexes: {invalid[:3]}",
                )
            resolutions = {get_resolution(cell) for cell in cells}
            if len(resolutions) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="compact_cells requires cells at one common resolution",
                )
            try:
                compacted = compact_cells(cells)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"compact_cells could not compact the supplied cells: {exc}",
                ) from exc
            return SpatialAnalysisResponse(
                success=True,
                result={"cells": compacted},
                message="H3 cell compaction completed",
                metadata={
                    "operation": request.operation,
                    "input_count": len(cells),
                    "num_cells": len(compacted),
                },
            )

        elif request.operation == "cell_to_boundary":
            center_cell = parameters.get("center_cell")
            if not isinstance(center_cell, str) or not is_valid_cell(center_cell):
                raise HTTPException(
                    status_code=400,
                    detail="cell_to_boundary requires a valid 'center_cell' H3 index",
                )
            from ..utils.h3_utils import cell_to_latlng_boundary

            ring = [[lng, lat] for lat, lng in cell_to_latlng_boundary(center_cell)]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
            return SpatialAnalysisResponse(
                success=True,
                result={
                    "cell": center_cell,
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                },
                message="H3 cell boundary completed",
                metadata={"operation": request.operation, "cell": center_cell},
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"H3 operation '{request.operation}' is unsupported",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"H3 analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "GEO-INFER-SPACE"}


@router.get("/capabilities")
async def get_capabilities() -> dict[str, list[str]]:
    """Get available analysis capabilities."""
    return {
        "vector_operations": [
            "buffer",
            "overlay",
            "proximity",
            "spatial_join",
            "geometric_calculations",
        ],
        "raster_operations": [
            "terrain_analysis",
            "map_algebra",
            "focal_statistics",
            "zonal_statistics",
        ],
        "network_analysis": [
            "shortest_path",
            "service_area",
            "connectivity",
            "routing",
            "accessibility",
        ],
        "geostatistics": [
            "interpolation",
            "clustering",
            "hotspot_detection",
            "autocorrelation",
        ],
        "h3_operations": [
            "polygon_to_cells",
            "grid_disk",
            "compact_cells",
            "cell_boundaries",
        ],
        "point_cloud": [
            "filtering",
            "feature_extraction",
            "classification",
            "surface_generation",
        ],
    }


# Include router in app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
