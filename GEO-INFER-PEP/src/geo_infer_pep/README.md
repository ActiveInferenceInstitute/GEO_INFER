# GEO-INFER-PEP/src/geo_infer_pep

Geo Infer Pep workspace within `GEO-INFER-PEP`.

## Contents

- `api/`
- `core/`
- `crm/`
- `hr/`
- `models/`
- `reporting/`
- `talent/`
- `utils/`
- `visualizations/`
- `__init__.py`
- `methods.py`
- `performance_store.py`

## Public Interface

- `methods.py:process_employee_onboarding_workflow` (function)
- `methods.py:generate_quarterly_people_report` (function)
- `methods.py:import_hr_data_from_csv` (function)
- `methods.py:import_crm_data_from_csv` (function)
- `methods.py:import_talent_data_from_csv` (function)
- `methods.py:generate_comprehensive_hr_dashboard` (function)
- `methods.py:generate_comprehensive_crm_dashboard` (function)
- `methods.py:generate_comprehensive_talent_dashboard` (function)
- `methods.py:get_all_employees` (function)
- `methods.py:get_all_candidates` (function)
- `methods.py:get_all_customers` (function)
- `methods.py:clear_all_data` (function)
- `performance_store.py:PerformanceReviewStore` (class)

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
