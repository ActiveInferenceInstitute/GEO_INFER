## GEO-INFER-API — API Services and Ecosystem Interoperability

**Purpose.** GEO-INFER-API provides comprehensive API development and integration services enabling interoperability across the GEO-INFER ecosystem and external systems. Its README positions it as the service layer through which module capabilities become network-accessible, with FastAPI declared as the primary dependency (`fastapi>=0.100.0`).

**Public API.** The `geo_infer_api` package exports application configuration (`Settings`, `get_settings`, `main_app`) alongside a complete GeoJSON type system: `Feature`, `FeatureCollection`, `Geometry`, `Point`, `MultiPoint`, `LineString`, `MultiLineString`, `MultiPolygon`, `Polygon`, `PolygonFeature`, `PolygonFeatureCollection`, and `GeoJSONType` — the vocabulary with which spatial data crosses service boundaries. The subpackage layout (`core/`, `endpoints/`, `models/`, `utils/`) separates server configuration, route definitions, wire models, and helpers.

The tests exercise endpoint and data-model behavior. The shared repository runner selects this module with `--module API`.

**Theme role.** Infrastructure: API is the interoperability spine of the framework, translating the geospatial core's data structures into service contracts so that domain science and agent modules can be consumed by external systems and by the APP layer.
