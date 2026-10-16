# Agent Instructions: GEO-INFER-PLACE/tests/integration

## Scope

- Owning module: `GEO-INFER-PLACE`
- Python package: `geo_infer_place`
- Directory role: Integration workspace within `GEO-INFER-PLACE`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_place` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_cascadia_boundary.py`
- `test_cascadia_geojson_pipeline.py`
- `test_cascadia_hydrography.py`
- `test_location_configs.py`
- `test_regional_download_worker.py`
- `test_regional_layer_acquisition.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-PLACE/tests/integration
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
