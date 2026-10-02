# Agent Instructions: GEO-INFER-SPACE/tests/unit

## Scope

- Owning module: `GEO-INFER-SPACE`
- Python package: `geo_infer_space`
- Directory role: Unit workspace within `GEO-INFER-SPACE`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_space` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_algorithm_registry.py`
- `test_analytics_comprehensive.py`
- `test_analytics_context_contract.py`
- `test_analytics_warnings.py`
- `test_api_schemas.py`
- `test_backends_comprehensive.py`
- `test_base_module.py`
- `test_config_loader_packaged_config.py`
- `test_core.py`
- `test_data_integrator.py`
- `test_degradation_warnings.py`
- `test_dispatch_comprehensive.py`
- `test_geolibre_projects.py`
- `test_gis_submodule.py`
- `test_gpu_acceleration.py`
- `test_h3_enhanced.py`
- `test_h3_operations_runtime.py`
- `test_h3_policy.py`
- `test_h3_predicate_logging.py`
- `test_io_modules.py`
- `test_ml_integration_tables.py`
- `test_morans_i_variance_parity.py`
- `test_nested_analytics.py`
- `test_nested_comprehensive.py`
- `test_nested_h3_contract.py`
- `test_nested_messaging_routing.py`
- `test_place_analyzer.py`
- `test_place_analyzer_synthetic.py`
- `test_raster_expression_security.py`
- `test_sparse_transition.py`
- `test_spatial_methods.py`
- `test_spatial_processor.py`
- `test_spatial_statistics.py`
- `test_spatial_utils.py`
- `test_spatiotemporal.py`
- `test_spatiotemporal_alignment.py`
- `test_state_space.py`
- `test_temporal_analytics.py`
- `test_unified_backend.py`
- `test_unified_backend_geojson_seam.py`
- `test_unified_comprehensive.py`
- `test_visualization_engine.py`
- `test_visualization_receipts.py`
- `test_weight_matrix_backend_contract.py`
- `test_whitebox_bridge.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-SPACE/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
