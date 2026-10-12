# GeoJSON API Reference

## Introduction

This document describes the GeoJSON endpoints provided by GEO-INFER-API for
handling geospatial polygon data. The routes are generated from the
implemented application (`create_app()` in `src/geo_infer_api/app.py`) and
match `docs/openapi_spec.yaml`, which is regenerated from the same source.

> **Storage warning:** polygon features created through this API are held in a
> process-local in-memory store. Data is not persisted and is not shared
> between multiple uvicorn workers — restarts and multi-worker deployments
> silently lose POSTed features. Use an external store for production.

## Base URL

```
http://localhost:8000/api/v1
```

All GeoJSON and algorithm endpoints are mounted under the `/api/v1` prefix;
the health endpoints alone sit at the application root (`/health`,
`/health/detailed`).

## Endpoints

### Collections (OGC API Features style)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/collections` | List available feature collections |
| `GET` | `/api/v1/collections/polygons` | Polygon collection metadata |
| `GET` | `/api/v1/collections/polygons/items` | List polygon features (`bbox`, `limit` query params) |
| `POST` | `/api/v1/collections/polygons/items` | Create a polygon feature (201; 409 when the ID exists or the store is at capacity) |
| `GET` | `/api/v1/collections/polygons/items/{feature_id}` | Get one polygon feature |
| `PUT` | `/api/v1/collections/polygons/items/{feature_id}` | Update one polygon feature |
| `DELETE` | `/api/v1/collections/polygons/items/{feature_id}` | Delete one polygon feature (204) |

Feature bodies are `PolygonFeature` documents (`type`, `geometry`, optional
`id` and `properties`); `POST` requires an `id`.

### Polygon operations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/operations/polygon/area` | Planar shoelace area in square kilometers |
| `POST` | `/api/v1/operations/polygon/simplify` | Ramer-Douglas-Peucker simplification (`tolerance` query) |
| `POST` | `/api/v1/operations/polygon/contains` | Ray-casting point-in-polygon test (`lon`, `lat` query) |
| `POST` | `/api/v1/operations/polygon/buffer` | Bounding-box buffer zone (`distance`, `unit`, `segments` query) |
| `POST` | `/api/v1/operations/polygon/intersection` | Bounding-box intersection of ≥ 2 polygons |
| `POST` | `/api/v1/operations/polygon/union` | Bounding-box union of multiple polygons |
| `POST` | `/api/v1/operations/polygon/distance` | Centroid distance between two polygons |

Set operations take a body of the form `{"polygons": [ ... ]}`; distance takes
`{"polygon1": ..., "polygon2": ...}`. All operation endpoints accept a
`PolygonFeature` as the primary body.

The registered processing-algorithm registry lives at `/api/v1/algorithms`
and is documented in `algorithms_api.md`.

## Response Formats

- `application/json` for collection metadata and operation results
- `application/geo+json` semantics for feature and feature-collection bodies
  (`PolygonFeature`, `PolygonFeatureCollection`)

## Error Responses

Application errors return the middleware error envelope:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid bbox format. Expected 'minLon,minLat,maxLon,maxLat'",
    "status_code": 422,
    "field": "bbox"
  }
}
```

`VALIDATION_ERROR` maps to 422, `NOT_FOUND_ERROR` to 404, and
`CONFLICT_ERROR` to 409.
