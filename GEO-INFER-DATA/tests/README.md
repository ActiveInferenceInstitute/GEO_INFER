# GEO-INFER-DATA/tests

Tests workspace within `GEO-INFER-DATA`.

## Contents

- `fixtures/`
- `integration/`
- `performance/`
- `unit/`
- `conftest.py`

## Public Interface

- `conftest.py:owned_event_loop` (function)
- `conftest.py:sample_coordinates` (function)
- `conftest.py:sample_geodataframe` (function)
- `conftest.py:tmp_output_dir` (function)
- `conftest.py:sample_csv_path` (function)
- `conftest.py:sample_geojson_path` (function)
- `conftest.py:data_source_config` (function)

## Module Metadata

- Module: `GEO-INFER-DATA`
- Package: `geo_infer_data`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-data`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module DATA`

## Dependencies

- `aiohttp>=3.8.0`
- `aiomqtt>=2.4.0`
- `boto3>=1.28.0`
- `fastapi>=0.100.0`
- `starlette>=0.27.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `minio>=7.1.0`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `psutil>=5.9.0`
- `psycopg2-binary>=2.9.0`
- `pydantic>=2.0.0`
- `pyproj>=3.5.0`
- `pyyaml>=6.0.0`
- `rasterio>=1.3.0`
- `redis>=4.5.0`
- `requests>=2.31.0`
- `rtree>=1.0.0`
- `shapely>=2.0.0`
- `sqlalchemy>=2.0.0`
- `urllib3>=2.0.6`
- `uvicorn[standard]>=0.23.0`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-DATA` module's current behavior through unit,
  integration, system, and performance test surfaces.
- Primary marker: tests receive exactly one primary marker from their canonical
  directory; additive domain markers remain allowed.
- Required fixtures: local `tests/conftest.py` fixtures and shared
  `geo_infer_test.testing` fixtures for deterministic RNG, filesystem, HTTP,
  SQLite, service, model, and artifact boundaries.
- Dependencies: required test/runtime dependencies are installed by
  `uv sync --all-packages --all-extras`; missing backends are failures.
- Expected artifacts: JUnit XML under `.geo-infer-test-results/`; model and
  visualization outputs require finite statistics, sidecars, hashes, and a
  manifest.
- Failure triage: `env -u VIRTUAL_ENV uv run pytest -c pyproject.toml -q
  GEO-INFER-DATA/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module DATA
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
