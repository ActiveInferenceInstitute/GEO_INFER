# Agent Instructions: GEO-INFER-API/tests/unit

## Scope

- Owning module: `GEO-INFER-API`
- Python package: `geo_infer_api`
- Directory role: Unit workspace within `GEO-INFER-API`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_api` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_agent_sessions.py`
- `test_agent_stdio.py`
- `test_agent_surfaces.py`
- `test_algorithms_router.py`
- `test_config.py`
- `test_exceptions.py`
- `test_geojson_helpers.py`
- `test_geojson_helpers_extended.py`
- `test_geojson_router.py`
- `test_geojson_visualization.py`
- `test_middleware.py`
- `test_models_geojson.py`
- `test_security_config.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-API/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
