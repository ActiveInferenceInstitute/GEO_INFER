# Agent Instructions: GEO-INFER-TEST/tests/integration

## Scope

- Owning module: `GEO-INFER-TEST`
- Python package: `geo_infer_test`
- Directory role: Integration workspace within `GEO-INFER-TEST`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_test` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_act_agent_ant_coordination.py`
- `test_ai_space_domain_integration.py`
- `test_cross_module.py`
- `test_cross_module_civic_intel.py`
- `test_cross_module_workflows.py`
- `test_ecosystem_health.py`
- `test_h3_space_time_bayes_risk_act_composition.py`
- `test_module_imports.py`
- `test_sec_api_app_security.py`
- `test_space_time_data_integration.py`

## Validation

```bash
uv sync --all-packages --all-extras --all-groups
uv run python -m pytest GEO-INFER-TEST/tests/integration
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
