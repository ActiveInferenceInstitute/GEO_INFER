# Agent Instructions: GEO-INFER-MATH/tests/unit

## Scope

- Owning module: `GEO-INFER-MATH`
- Python package: `geo_infer_math`
- Directory role: Unit workspace within `GEO-INFER-MATH`.

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

- `test_ai_convenience.py`
- `test_bayes_convenience.py`
- `test_circulation.py`
- `test_clustering.py`
- `test_config_contract.py`
- `test_convenience_api.py`
- `test_free_energy.py`
- `test_geometry.py`
- `test_gpu_acceleration.py`
- `test_graph_theory.py`
- `test_information_theory.py`
- `test_integration_convenience.py`
- `test_interpolation.py`
- `test_linalg_tensor.py`
- `test_numerical_methods.py`
- `test_optimization.py`
- `test_regression.py`
- `test_spatial_statistics.py`
- `test_symbolic_math.py`
- `test_theorem_proving.py`
- `test_transforms.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-MATH/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
