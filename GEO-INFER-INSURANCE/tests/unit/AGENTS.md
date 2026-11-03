# Agent Instructions: GEO-INFER-INSURANCE/tests/unit

## Scope

- Owning module: `GEO-INFER-INSURANCE`
- Python package: `geo_infer_insurance`
- Directory role: Unit workspace within `GEO-INFER-INSURANCE`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_insurance` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `test_compliance_engine.py`
- `test_data_integration_fetches.py`
- `test_identifier_generation.py`
- `test_models.py`
- `test_package_safety.py`
- `test_policy_lifecycle.py`
- `test_portfolio_management.py`
- `test_pricing_and_claims.py`
- `test_silent_fabrication_fixes.py`
- `test_underwriting_decisions.py`
- `test_underwriting_rules.py`

## Validation

```bash
uv run python -m pytest GEO-INFER-INSURANCE/tests/unit
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
