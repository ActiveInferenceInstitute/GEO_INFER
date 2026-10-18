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
- Integer and string polygon IDs share canonical storage identity; updates
  validate any supplied body ID, including zero.

## Agent Interfaces

- `geo-infer` and `python -m geo_infer_api.agent_cli` share the script
  `geo_infer_api.agent_runtime.AgentSession` execution contract.
- `geo-infer-mcp` uses the optional `mcp` extra and the official SDK over stdio.
- Import-free catalogs cover all 45 modules; runtime calls retain native
  dependencies and authority. Native object handles are process-local.
- Failed retention after context entry unwinds that new context immediately.
- See `GEO-INFER-INTRA/docs/integration/agent_interfaces.md` from the repository
  root for request schemas, codecs, composition and failure semantics.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
