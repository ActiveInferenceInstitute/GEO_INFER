# Agent Instructions: GEO-INFER-ACT/tests/integration

## Scope

- Owning module: `GEO-INFER-ACT`
- Python package: `geo_infer_act`
- Directory role: Integration workspace within `GEO-INFER-ACT`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_act` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_active_sensing_trajectories.py`
- `test_bayeux_backend.py`
- `test_h3_example_smoke.py`
- `test_integration.py`
- `test_space_integration.py`
- `test_stigmergy.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-ACT/tests/integration
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
