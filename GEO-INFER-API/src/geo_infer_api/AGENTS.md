# Agent Instructions: GEO-INFER-API/src/geo_infer_api

## Scope

- Owning module: `GEO-INFER-API`
- Python package: `geo_infer_api`
- Directory role: Geo Infer Api workspace within `GEO-INFER-API`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_api` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `core/`
- `endpoints/`
- `models/`
- `utils/`
- `__init__.py`
- `agent_benchmarks.py`
- `agent_catalog.py`
- `agent_cli.py`
- `agent_diagnostics.py`
- `agent_mcp.py`
- `agent_runtime.py`
- `agent_schema.py`
- `agent_skills.py`
- `agent_stdio.py`
- `app.py`
- `agent_skills.json`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module API
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
