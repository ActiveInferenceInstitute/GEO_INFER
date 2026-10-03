# Agent Instructions: GEO-INFER-DATA/tests/unit

## Scope

- Owning module: `GEO-INFER-DATA`
- Python package: `geo_infer_data`
- Directory role: Unit workspace within `GEO-INFER-DATA`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_data` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_api.py`
- `test_api_connectors.py`
- `test_archive_safety.py`
- `test_caching.py`
- `test_cloud_connectors.py`
- `test_compression.py`
- `test_duckdb_provisioning.py`
- `test_duckdb_spatial.py`
- `test_error_handling.py`
- `test_file_connector.py`
- `test_format_detection.py`
- `test_geospatial_validation.py`
- `test_identifiers.py`
- `test_indexing.py`
- `test_ingestion.py`
- `test_optional_backend_boundaries.py`
- `test_performance.py`
- `test_pipeline.py`
- `test_rest_api_error_handling.py`
- `test_schemas.py`
- `test_storage.py`
- `test_stream_connectors.py`
- `test_timestamp_contracts.py`
- `test_validation.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-DATA/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
