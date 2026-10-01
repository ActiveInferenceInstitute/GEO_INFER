# GEO-INFER-PEP/src/geo_infer_pep/models

Models workspace within `GEO-INFER-PEP`.

## Contents

- `__init__.py`
- `conflict_models.py`
- `crm_models.py`
- `hr_models.py`
- `learning_models.py`
- `survey_models.py`
- `talent_models.py`

## Public Interface

- `conflict_models.py:ConflictCase` (class)
- `crm_models.py:InteractionLog` (class)
- `crm_models.py:Address` (class)
- `crm_models.py:Customer` (class)
- `hr_models.py:EmploymentStatus` (class)
- `hr_models.py:Gender` (class)
- `hr_models.py:Compensation` (class)
- `hr_models.py:JobHistoryEntry` (class)
- `hr_models.py:PerformanceReview` (class)
- `hr_models.py:Employee` (class)
- `learning_models.py:LearningCourse` (class)
- `learning_models.py:LearningEnrollment` (class)
- `survey_models.py:Survey` (class)
- `survey_models.py:SurveyResponse` (class)
- `talent_models.py:JobRequisitionStatus` (class)
- `talent_models.py:CandidateStatus` (class)
- `talent_models.py:InterviewType` (class)
- `talent_models.py:InterviewFeedback` (class)
- `talent_models.py:Interview` (class)
- `talent_models.py:Offer` (class)

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
