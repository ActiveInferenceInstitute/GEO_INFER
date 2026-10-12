# Agent Instructions: GEO-INFER-NORMS/examples/output

## Scope

- Owning module: `GEO-INFER-NORMS`
- Python package: `geo_infer_norms`
- Directory role: Output workspace within `GEO-INFER-NORMS`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_norms` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `historical_zoning_changes.png`
- `zoning_adjacency_network.png`
- `zoning_change_visualization.png`
- `zoning_compatibility_matrix.png`
- `zoning_distribution_by_category.png`
- `zoning_districts.png`
- `zoning_districts_highlight.png`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module NORMS
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
