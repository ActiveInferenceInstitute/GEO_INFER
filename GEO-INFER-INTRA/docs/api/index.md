# API Documentation

GEO-INFER is a module workspace, not a single hosted API product. API behavior
is owned by the module that exposes it. This page explains how to find and
validate those surfaces.

## Python APIs

Start with the module root README and inspect its package exports:

```python
import geo_infer_act
import geo_infer_space

print(geo_infer_act.__all__)
print(geo_infer_space.__all__)
```

Use the module-local SKILL.md and source __init__.py for current public
imports. The conceptual module pages are listed in the [module catalog](../modules/index.md).

## GEO-INFER-API

GEO-INFER-API provides a FastAPI application and GeoJSON models when its web
dependencies are installed:

```python
from geo_infer_api import Feature, FeatureCollection
from geo_infer_api.app import create_app
```

`create_app()` and access to the lazy `main_app` construct the application and
require a nonempty `SECRET_KEY` in the process environment. Importing the models
or the factory alone does not construct it. See the
[local startup example](../deployment/index.md#running-a-module-api). The current
application has no authentication middleware; supplying the required secret does
not create an authenticated service.

The source-backed API material is:

- [GEO-INFER-API README](../../../GEO-INFER-API/README.md)
- [GeoJSON API guide](../../../GEO-INFER-API/docs/geojson_api.md)
- [OpenAPI specification](../../../GEO-INFER-API/docs/openapi_spec.yaml)
- [API examples](../../../GEO-INFER-API/examples/README.md)

Do not infer a public deployment URL, authentication scheme, or endpoint from
this repository unless a module's current source and configuration define it.

## CLI and validation APIs

Repository commands are documented in the root [README](../../../README.md) and
the [GEO-INFER-TEST command matrix](../../../GEO-INFER-TEST/docs/index.md).
Important executable surfaces include:

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --list-modules
uv run python GEO-INFER-TEST/run_unified_tests.py --module API
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language
```

## Integration rules

- Import public exports instead of internal implementation paths when possible.
- Declare optional web/API dependencies in the owning module.
- Validate GeoJSON coordinates as finite WGS84 longitude/latitude values.
- Document HTTP routes, authentication, response schemas, and error codes only
  when they are implemented and covered by the module tests.
- Keep API examples local, deterministic, and runnable without an unconfigured
  external service.
