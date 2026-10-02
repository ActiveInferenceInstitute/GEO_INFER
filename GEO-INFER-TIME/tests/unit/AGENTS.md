# Agent Instructions: GEO-INFER-TIME/tests/unit

## Scope

- Owning module: `GEO-INFER-TIME`
- Python package: `geo_infer_time`
- Directory role: Unit workspace within `GEO-INFER-TIME`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_time` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_action_schedule.py`
- `test_core.py`
- `test_inference_schedule.py`
- `test_lazy_public_imports.py`
- `test_linear_trend_contract.py`
- `test_timestamps.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-TIME/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
