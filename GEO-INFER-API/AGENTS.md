# Agent Instructions: GEO-INFER-API

## Scope

- Owning module: `GEO-INFER-API`
- Python package: `geo_infer_api`
- Directory role: Comprehensive API development and integration services enabling interoperability across the GEO-INFER ecosystem and external systems.

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

- `docs/`
- `examples/`
- `src/`
- `tests/`
- `SKILL.md`
- `pyproject.toml`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module API
```


## GeoJSON Contracts

- GeoJSON positions must be finite WGS84 longitude/latitude values.
- Polygon bbox filtering uses geometry extents, so containing and crossing
  polygons are not missed when no vertex lies inside the query bbox.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
