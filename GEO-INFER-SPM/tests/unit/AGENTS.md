# Agent Instructions: GEO-INFER-SPM/tests/unit

## Scope

- Owning module: `GEO-INFER-SPM`
- Python package: `geo_infer_spm`
- Directory role: Unit workspace within `GEO-INFER-SPM`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_spm` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_acceptance_spm.py`
- `test_advanced_models.py`
- `test_api_endpoints.py`
- `test_bayesian.py`
- `test_compiler_profile.py`
- `test_contrasts.py`
- `test_data_io.py`
- `test_glm.py`
- `test_helper_input_contracts.py`
- `test_helpers.py`
- `test_helpers_reproducibility.py`
- `test_package_exports.py`
- `test_preprocessing.py`
- `test_rft.py`
- `test_rft_fwe_contract.py`
- `test_spatial_analysis.py`
- `test_temporal_analysis.py`
- `test_validation.py`
- `test_visualization.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-SPM/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
