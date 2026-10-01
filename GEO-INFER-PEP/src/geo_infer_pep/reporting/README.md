# GEO-INFER-PEP/src/geo_infer_pep/reporting

Reporting workspace within `GEO-INFER-PEP`.

## Contents

- `__init__.py`
- `crm_reports.py`
- `generic_report_generator.py`
- `hr_reports.py`
- `talent_reports.py`

## Public Interface

- `crm_reports.py:generate_customer_segmentation_report` (function)
- `crm_reports.py:generate_lead_conversion_report` (function)
- `crm_reports.py:get_quarterly_metrics` (function)
- `generic_report_generator.py:create_quarterly_overview` (function)
- `hr_reports.py:generate_headcount_report` (function)
- `hr_reports.py:generate_diversity_report` (function)
- `hr_reports.py:get_quarterly_metrics` (function)
- `talent_reports.py:generate_candidate_pipeline_report` (function)
- `talent_reports.py:calculate_time_to_hire` (function)
- `talent_reports.py:get_quarterly_metrics` (function)

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
