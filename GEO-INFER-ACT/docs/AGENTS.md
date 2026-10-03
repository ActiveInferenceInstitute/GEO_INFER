# Agent Instructions: GEO-INFER-ACT/docs

## Scope

- Owning module: `GEO-INFER-ACT`
- Python package: `geo_infer_act`
- Directory role: Docs workspace within `GEO-INFER-ACT`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_act` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `0_4_method_review.md`
- `active_inference_overview.md`
- `api_schema.yaml`
- `categorical_inference.md`
- `fep_lean_notation_bridge.md`
- `free_energy_principle.md`
- `geospatial_applications.md`
- `gnn_factored_interchange.md`
- `gnn_interchange.md`
- `mathematical_framework.md`
- `method_inventory.md`
- `references.md`
- `world_systems_modeling.md`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
