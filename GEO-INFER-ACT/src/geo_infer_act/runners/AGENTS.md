# Agent Instructions: GEO-INFER-ACT/src/geo_infer_act/runners

## Scope

- Owning module: `GEO-INFER-ACT`
- Python package: `geo_infer_act`
- Directory role: Runners workspace within `GEO-INFER-ACT`.

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

- `__init__.py`
- `cli.py`
- `closed_loop.py`
- `contracts.py`
- `gallery.py`
- `h3.py`
- `io.py`
- `model_matrix.py`
- `model_matrix_design.py`
- `model_pair.py`
- `model_pair_explorer.py`
- `model_pair_plot.py`
- `scenarios.py`
- `study_diagnostics.py`
- `study_visualization.py`
- `wrapper.py`
- `model_pair_explorer.html`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
