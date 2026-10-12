# Processing Algorithm Registry API

GEO-INFER-API exposes the GeoLibre-style processing algorithm registry
(`geo_infer_space.core.algorithm_registry`) as a read-only REST surface. This
lets SPACE/API/APP clients list, describe, and run registered spatial tools
uniformly.

## Endpoints

All endpoints are under the `/api/v1` prefix.

| Method | Path | Summary |
| --- | --- | --- |
| `GET` | `/api/v1/algorithms` | List registered algorithms and their parameters |
| `GET` | `/api/v1/algorithms/{algorithm_id}` | Describe one algorithm |
| `POST` | `/api/v1/algorithms/{algorithm_id}/run` | Run one algorithm against supplied layers/parameters |

## Example

```bash
# List algorithms
curl http://localhost:8000/api/v1/algorithms

# Run the reference "count-features" algorithm
curl -X POST http://localhost:8000/api/v1/algorithms/count-features/run \
  -H 'Content-Type: application/json' \
  -d '{"layers": [{"id": "grid", "geojson": {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": null}]}}], "parameters": {"layer": "grid"}}'
```

Response:

```json
{"algorithm_id": "count-features", "result": 1, "logs": ["Feature count: 1"]}
```

## Reference algorithms

The reference registry ships with:

- `calculate-bounds` — compute the bounding box of a GeoJSON layer.
- `count-features` — count features in a GeoJSON layer.

Domain modules register their own algorithms on the same
`AlgorithmRegistry`/`ProcessingAlgorithm` contract.

## Availability

The registry is imported gracefully. When the full uv workspace is synced
(`uv sync --all-packages --all-extras --all-groups`), `geo_infer_space` provides the
reference registry. If `geo_infer_space` is not importable the endpoints
return HTTP 503 with a clear message instead of failing at import time.

## Tests

`GEO-INFER-API/tests/unit/test_algorithms_router.py` exercises both branches:
the live reference registry and the graceful-unavailable path (no skipped
tests, per the repository's strict pytest policy).
