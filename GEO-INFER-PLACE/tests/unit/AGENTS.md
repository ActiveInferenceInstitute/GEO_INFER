# Agent Instructions: GEO-INFER-PLACE/tests/unit

## Scope

- Owning module: `GEO-INFER-PLACE`
- Python package: `geo_infer_place`
- Directory role: Unit workspace within `GEO-INFER-PLACE`.

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

- `test_api_clients.py`
- `test_bioregion_visualization.py`
- `test_caching.py`
- `test_california_capture.py`
- `test_california_execution_binding.py`
- `test_california_geometry.py`
- `test_california_inference.py`
- `test_california_monitoring.py`
- `test_cascadia_flowlines.py`
- `test_comprehensive_dashboard.py`
- `test_crescent_city_intel.py`
- `test_dashboard_advanced.py`
- `test_data_sources.py`
- `test_del_norte_analyzers.py`
- `test_del_norte_dashboard_packaged_config.py`
- `test_del_norte_demo_orchestration.py`
- `test_h3_operations.py`
- `test_hydrography_ingestion.py`
- `test_integration_wrappers.py`
- `test_module_bridge.py`
- `test_place_analyzer.py`
- `test_place_interface.py`
- `test_place_lazy_exports.py`
- `test_unified_backend.py`
- `test_unified_backend_packaged_config.py`
- `test_visualization_engine.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-PLACE/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
