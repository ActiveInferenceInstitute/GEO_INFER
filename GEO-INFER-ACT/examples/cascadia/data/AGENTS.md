# Agent Instructions: GEO-INFER-ACT/examples/cascadia/data

## Scope

- Owning module: `GEO-INFER-ACT`
- Python package: `geo_infer_act`
- Directory role: Data workspace within `GEO-INFER-ACT`.

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

- `cascadia_seed_outcomes_20261012.svg`
- `cascadia_seed_outcomes_20261012_svg_v2.svg`
- `cascadia_seed_study_20261012.zip`
- `cascadia_seed_study_20261012_manifest.json`
- `cascadia_seed_study_20261012_summary.json`
- `cascadia_seed_study_20261012_svg_v2.zip`
- `cascadia_seed_study_20261012_svg_v2_manifest.json`
- `cascadia_study_diagnostics_20261013.html`
- `cascadia_study_diagnostics_20261013.json`
- `cascadia_study_explorer_20261013.html`
- `cascadia_study_explorer_20261013_v2.html`
- `model_matrix_20261014.html`
- `model_matrix_20261014.svg`
- `model_matrix_20261014.zip`
- `model_pairs_20261013.html`
- `model_pairs_20261013.svg`
- `model_pairs_20261013.zip`
- `smith_river_homing.domain.json`
- `smith_river_homing.export.json`
- `smith_river_homing.geo-infer.json`
- `smith_river_homing.gnn.md`
- `smith_river_homing.state_ids.json`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
