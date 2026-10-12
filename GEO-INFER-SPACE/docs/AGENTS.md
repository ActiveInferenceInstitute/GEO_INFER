# Agent Instructions: GEO-INFER-SPACE/docs

## Scope

- Owning module: `GEO-INFER-SPACE`
- Python package: `geo_infer_space`
- Directory role: Docs workspace within `GEO-INFER-SPACE`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_space` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `references/`
- `CLI_TOOLS.md`
- `CROSS_MODULE_COMPOSITION.md`
- `GPU_ACCELERATION.md`
- `H3_DATASET_INTEGRATION_GUIDE.md`
- `H3_DEMO_README.md`
- `H3_MODULE_CONFIGURATION_GUIDE.md`
- `H3_V4_MIGRATION_GUIDE.md`
- `METHOD_CONTRACTS.md`
- `TESTING.md`
- `api_schema.yaml`
- `h3_advanced_methods.md`
- `sparse_state_transitions.md`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
