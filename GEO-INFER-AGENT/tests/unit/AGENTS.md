# Agent Instructions: GEO-INFER-AGENT/tests/unit

## Scope

- Owning module: `GEO-INFER-AGENT`
- Python package: `geo_infer_agent`
- Directory role: Unit workspace within `GEO-INFER-AGENT`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_agent` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `models/`
- `test_agent_base.py`
- `test_agent_communication.py`
- `test_agent_start_task_retention.py`
- `test_api_endpoints.py`
- `test_bdi_agent_and_services.py`
- `test_bdi_placeholders.py`
- `test_cli.py`
- `test_coordination.py`
- `test_core_active_inference.py`
- `test_data_collector.py`
- `test_data_collector_agent.py`
- `test_hybrid.py`
- `test_hybrid_agent.py`
- `test_interface_and_cli.py`
- `test_llm_proxy.py`
- `test_messaging.py`
- `test_package_import_hygiene.py`
- `test_planning.py`
- `test_registry_task_lifecycle.py`
- `test_rl_state_index.py`
- `test_rl_unit.py`
- `test_rule_based.py`
- `test_rule_based_agent.py`
- `test_task_management.py`
- `test_telemetry.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-AGENT/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
