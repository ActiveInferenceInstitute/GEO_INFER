# Agent Instructions: GEO-INFER-EXAMPLES/tests/integration

## Scope

- Owning module: `GEO-INFER-EXAMPLES`
- Python package: `geo_infer_examples`
- Directory role: Integration workspace within `GEO-INFER-EXAMPLES`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_examples` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_example_orchestration.py`
- `test_local_composition.py`
- `test_module_orchestrator_execution.py`
- `test_orchestrator_scripts_execute.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-EXAMPLES/tests/integration
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
