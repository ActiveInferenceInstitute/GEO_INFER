# Agent Instructions: GEO-INFER-LOG/tests/unit

## Scope

- Owning module: `GEO-INFER-LOG`
- Python package: `geo_infer_log`
- Directory role: Unit workspace within `GEO-INFER-LOG`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_log` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_api_endpoints.py`
- `test_api_error_handling.py`
- `test_api_state.py`
- `test_core.py`
- `test_delivery.py`
- `test_delivery_behavior.py`
- `test_geo_utils.py`
- `test_log_init.py`
- `test_log_lifecycle.py`
- `test_observability.py`
- `test_optimization.py`
- `test_optimization_ortools.py`
- `test_supply_chain.py`
- `test_supply_chain_milp_contract.py`
- `test_transport.py`
- `test_transport_behavior.py`
- `test_transport_network_api.py`
- `test_visualization.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-LOG/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
