# Agent Instructions: GEO-INFER-ANT

## Scope

- Owning module: `GEO-INFER-ANT`
- Python package: `geo_infer_ant`
- Directory role: Comprehensive swarm intelligence and complex adaptive systems modeling using Active Inference principles for emergent collective behavior in geospatial contexts.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_ant` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `config/`
- `docs/`
- `examples/`
- `src/`
- `tests/`
- `run_tests.py`
- `IMPLEMENTATION_STATUS.md`
- `SKILL.md`
- `pyproject.toml`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ANT
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
