# Agent Instructions: GEO-INFER-MATH/src/geo_infer_math/core

## Scope

- Owning module: `GEO-INFER-MATH`
- Python package: `geo_infer_math`
- Directory role: Core workspace within `GEO-INFER-MATH`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_math` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `information_theory/`
- `theorem_proving/`
- `__init__.py`
- `circulation.py`
- `geometry.py`
- `gpu_acceleration.py`
- `graph_theory.py`
- `interpolation.py`
- `linalg_tensor.py`
- `numerical_methods.py`
- `optimization.py`
- `spatial_statistics.py`
- `symbolic_math.py`
- `transforms.py`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
