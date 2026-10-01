# Agent Instructions: GEO-INFER-PLACE

## Scope

- Owning module: `GEO-INFER-PLACE`
- Python package: `geo_infer_place`
- Directory role: Comprehensive place-based analysis framework providing deep insights into specific geographic locations and regional systems.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_place` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `config/`
- `data/`
- `docs/`
- `examples/`
- `locations/`
- `src/`
- `tests/`
- `SKILL.md`
- `pyproject.toml`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module PLACE
```


## Visualization Guidance

- Validate H3 resolution, bounds, dashboard result mappings, and user-facing map
  options before building a dashboard.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
