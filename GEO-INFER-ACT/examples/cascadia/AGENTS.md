# Agent Instructions: GEO-INFER-ACT/examples/cascadia

## Scope

- Owning module: `GEO-INFER-ACT`
- Python package: `geo_infer_act`
- Directory role: Cascadia workspace within `GEO-INFER-ACT`.

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

- `data/`
- `cascadia_homing.py`
- `cascadia_study.py`
- `plot_model_pairs.py`
- `run_model_matrix.py`
- `run_model_pair.py`
- `study_contract.py`
- `study_custody.py`
- `study_diagnostics.py`
- `study_explorer.py`
- `study_figure.py`
- `validate_study.py`
- `cascadia_homing.yaml`
- `model_matrix_design.json`
- `study_design.json`
- `study_explorer.html`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
