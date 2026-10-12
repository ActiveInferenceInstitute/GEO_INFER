# GEO-INFER-PEP/examples

Examples workspace within `GEO-INFER-PEP`.

## Contents

- `basic_crm_example.py`
- `basic_hr_example.py`
- `onboarding_workflow_example.py`

## Public Interface

- `basic_crm_example.py:create_sample_crm_data` (function)
- `basic_crm_example.py:main` (function)
- `basic_hr_example.py:create_sample_hr_data` (function)
- `basic_hr_example.py:main` (function)
- `onboarding_workflow_example.py:create_sample_talent_data` (function)
- `onboarding_workflow_example.py:demonstrate_onboarding_workflow` (function)
- `onboarding_workflow_example.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-PEP`
- Package: `geo_infer_pep`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-pep`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module PEP`

## Dependencies

- `fastapi>=0.100.0`
- `starlette>=0.27.0`
- `uvicorn[standard]>=0.23.2`
- `pydantic>=2.0`
- `pandas>=2.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.13.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module PEP
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
