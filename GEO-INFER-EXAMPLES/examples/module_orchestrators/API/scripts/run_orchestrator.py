#!/usr/bin/env python3
"""GEO-INFER-API module orchestrator.

Runs one documented end-to-end API operation: build synthetic GeoJSON
payloads with the real pydantic models (Point, Polygon, PolygonFeature,
PolygonFeatureCollection) from ``geo_infer_api.models.geojson``, verify
round-trip serialization and rejection of an out-of-range coordinate, then
exercise the exported FastAPI ``main_app`` through the in-process
``TestClient`` transport (no socket bind): health endpoints, polygon
creation, listing, area, and point-containment operations.
"""

from __future__ import annotations

import sys
from typing import Any

from geo_infer_examples.orchestration import run_module_orchestrator


def _operation() -> dict[str, Any]:
    import os

    # The API's Settings fails closed without a signing key; the orchestrator
    # supplies an in-process synthetic key for this demo run.
    os.environ.setdefault("SECRET_KEY", "synthetic-orchestrator-demo-secret-key")

    from fastapi.testclient import TestClient
    from geo_infer_api import (
        Feature,
        FeatureCollection,
        Point,
        Polygon,
        PolygonFeature,
        PolygonFeatureCollection,
        get_settings,
    )
    from geo_infer_api.app import main_app
    from pydantic import ValidationError

    settings = get_settings()

    # 1. Synthetic GeoJSON payloads through the real pydantic models.
    site_point = Point(coordinates=[-124.05, 44.05])
    site_polygon = Polygon(
        coordinates=[
            [
                [-124.10, 44.00],
                [-123.95, 44.00],
                [-123.95, 44.12],
                [-124.10, 44.12],
                [-124.10, 44.00],
            ]
        ]
    )
    polygon_feature = PolygonFeature(
        geometry=site_polygon,
        properties={"site": "synthetic-plot-01", "survey": "2026-Q3"},
        id="polygon-001",
    )
    collection = PolygonFeatureCollection(
        features=[
            polygon_feature,
            PolygonFeature(
                geometry=Polygon(
                    coordinates=[
                        [
                            [-123.80, 44.00],
                            [-123.70, 44.00],
                            [-123.70, 44.10],
                            [-123.80, 44.10],
                            [-123.80, 44.00],
                        ]
                    ]
                ),
                properties={"site": "synthetic-plot-02"},
                id="polygon-002",
            ),
        ]
    )
    generic_collection = FeatureCollection(
        features=[
            Feature(
                geometry={"type": "Point", "coordinates": [-124.05, 44.05]},
                properties={"kind": "benchmark"},
            )
        ]
    )

    roundtrip_dict = polygon_feature.model_dump()
    roundtrip_ok = (
        roundtrip_dict["type"] == "Feature"
        and roundtrip_dict["geometry"]["type"] == "Polygon"
        and Point.model_validate(site_point.model_dump()).coordinates
        == site_point.coordinates
    )

    rejected = False
    rejection_message = ""
    try:
        Point(coordinates=[200.0, 44.05])  # longitude out of range
    except ValidationError as exc:
        rejected = True
        rejection_message = str(exc.errors()[0].get("msg", ""))

    # 2. In-process API exercise via TestClient (no socket bind).
    with TestClient(main_app) as client:
        health = client.get("/health")
        health_detailed = client.get("/health/detailed")
        create_1 = client.post(
            "/api/v1/collections/polygons/items",
            json=polygon_feature.model_dump(),
        )
        create_2 = client.post(
            "/api/v1/collections/polygons/items",
            json=collection.features[1].model_dump(),
        )
        listed = client.get("/api/v1/collections/polygons/items")
        area_response = client.post(
            "/api/v1/operations/polygon/area",
            json=polygon_feature.model_dump(),
        )
        contains_response = client.post(
            "/api/v1/operations/polygon/contains",
            params={"lon": -124.02, "lat": 44.06},
            json=polygon_feature.model_dump(),
        )

    status_codes: dict[str, int] = {
        "health": health.status_code,
        "health_detailed": health_detailed.status_code,
        "create_polygon_1": create_1.status_code,
        "create_polygon_2": create_2.status_code,
        "list_polygons": listed.status_code,
        "polygon_area": area_response.status_code,
        "contains_point": contains_response.status_code,
    }

    area_sq_km = area_response.json().get("area_sq_km")
    contains_point = contains_response.json().get("contains")
    listed_count = len(listed.json().get("features", []))
    polygons: list[PolygonFeature] = collection.features

    return {
        "operation": "geojson_model_and_api_validation",
        "settings_app_name": settings.app_name,
        "settings_app_version": settings.app_version,
        "models_built": {
            "point": site_point.coordinates,
            "polygon_vertices": len(site_polygon.coordinates[0]),
            "collection_features": len(polygons),
            "generic_collection_features": len(generic_collection.features),
        },
        "model_roundtrip_ok": roundtrip_ok,
        "invalid_coordinate_rejected": rejected,
        "rejection_message": rejection_message,
        "endpoint_status_codes": status_codes,
        "polygon_area_sq_km": area_sq_km,
        "polygon_contains_point": contains_point,
        "listed_polygon_count": listed_count,
        "all_endpoints_ok": all(code in (200, 201) for code in status_codes.values()),
    }


if __name__ == "__main__":
    sys.exit(run_module_orchestrator("API", _operation))
