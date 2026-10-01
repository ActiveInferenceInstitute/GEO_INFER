# GEO-INFER-PEP/src/geo_infer_pep/api

Api workspace within `GEO-INFER-PEP`.

## Contents

- `__init__.py`
- `crm_endpoints.py`
- `errors.py`
- `hr_endpoints.py`
- `talent_endpoints.py`

## Public Interface

- `__init__.py:health_check` (function)
- `__init__.py:system_status` (function)
- `__init__.py:system_dashboard` (function)
- `__init__.py:create_onboarding_workflow` (function)
- `__init__.py:get_workflow_status` (function)
- `__init__.py:execute_workflow` (function)
- `__init__.py:create_performance_review` (function)
- `__init__.py:get_performance_reviews` (function)
- `__init__.py:create_learning_course` (function)
- `__init__.py:get_learning_courses` (function)
- `__init__.py:enroll_employee` (function)
- `__init__.py:get_learning_enrollments` (function)
- `__init__.py:create_conflict_case` (function)
- `__init__.py:get_conflict_cases` (function)
- `__init__.py:update_conflict_case` (function)
- `__init__.py:create_survey` (function)
- `__init__.py:get_survey_responses` (function)
- `__init__.py:submit_survey_response` (function)
- `__init__.py:validate_employee_data` (function)
- `__init__.py:validate_customer_data` (function)

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
