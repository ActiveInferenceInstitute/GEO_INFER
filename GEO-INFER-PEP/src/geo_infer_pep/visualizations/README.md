# GEO-INFER-PEP/src/geo_infer_pep/visualizations

Visualizations workspace within `GEO-INFER-PEP`.

## Contents

- `__init__.py`
- `crm_visuals.py`
- `hr_visuals.py`
- `talent_visuals.py`

## Public Interface

- `crm_visuals.py:plot_customer_distribution_by_status` (function)
- `crm_visuals.py:plot_customer_distribution_by_source` (function)
- `hr_visuals.py:plot_headcount_by_department` (function)
- `hr_visuals.py:plot_gender_distribution` (function)
- `talent_visuals.py:plot_candidate_pipeline_by_status` (function)
- `talent_visuals.py:plot_time_to_hire_distribution` (function)

## Module Metadata

- Module: `GEO-INFER-PEP`
- Package: `geo_infer_pep`
- Version: `0.3.0`
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
