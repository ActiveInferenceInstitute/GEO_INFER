# Agent Instructions: GEO-INFER-IOT/tests/unit

## Scope

- Owning module: `GEO-INFER-IOT`
- Python package: `geo_infer_iot`
- Directory role: Unit workspace within `GEO-INFER-IOT`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_iot` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_calibration_topology_sensor_api.py`
- `test_data_ingestion.py`
- `test_fixwave_regressions.py`
- `test_inference_behavior.py`
- `test_ingest_error_and_latency.py`
- `test_ingestion.py`
- `test_optional_visualization.py`
- `test_performance_monitor.py`
- `test_quality_control.py`
- `test_quality_control_history.py`
- `test_radiation_monitoring.py`
- `test_registry.py`
- `test_sensor_api.py`
- `test_sensor_data.py`
- `test_streaming_forward.py`
- `test_timestamp_spatial_contracts.py`
- `test_visualization.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-IOT/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
