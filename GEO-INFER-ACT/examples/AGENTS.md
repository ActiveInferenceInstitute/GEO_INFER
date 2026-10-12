# Agent Instructions: GEO-INFER-ACT/examples

## Scope

- Owning module: `GEO-INFER-ACT`
- Python package: `geo_infer_act`
- Directory role: Examples workspace within `GEO-INFER-ACT`.

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

- `cascadia/`
- `output/`
- `__init__.py`
- `ecological_model.py`
- `h3_active_inference.py`
- `modern_active_inference.py`
- `run_all_examples.py`
- `simple_model.py`
- `spatial_active_inference_gallery.py`
- `spatial_inference_demo.py`
- `urban_planning.py`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
