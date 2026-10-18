# Agent Instructions: GEO-INFER-INTRA/docs/geospatial/data_formats/h3

## Scope

- Owning module: `GEO-INFER-INTRA`
- Python package: `geo_infer_intra`
- Directory role: H3 workspace within `GEO-INFER-INTRA`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_intra` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `ecosystem.md`
- `h3_api_reference.md`
- `h3_architecture.md`
- `h3_code_examples.md`
- `h3_comparative_analysis.md`
- `h3_database_integration.md`
- `h3_mobility_analysis.md`
- `h3_performance_optimization.md`
- `h3_programming_interfaces.md`
- `h3_readme.md`
- `h3_report.md`
- `h3_resolution_system.md`
- `h3_spatial_analysis.md`
- `h3_use_cases.md`
- `h3_visualization_techniques.md`
- `index.md`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
