# Agent Instructions: GEO-INFER-APP

## Scope

- Owning module: `GEO-INFER-APP`
- Python package: `geo_infer_app`
- Directory role: Human-computer interaction layer providing accessible geospatial applications, dashboards, and UI components.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_app` and the owning module's public contracts.

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
uv run python GEO-INFER-TEST/run_unified_tests.py --module APP
```


## Visualization Guidance

- Validate geographic coordinates before creating map features and keep emitted
  metadata JSON-safe for downstream GeoJSON/dashboard clients.
- Preserve dashboard widget schemas when adding agent state visualizations.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
