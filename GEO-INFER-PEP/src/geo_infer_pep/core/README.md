# GEO-INFER-PEP/src/geo_infer_pep/core

Core workspace within `GEO-INFER-PEP`.

## Contents

- `__init__.py`
- `data_store.py`
- `orchestrator.py`
- `pep_engine.py`
- `validator.py`

## Public Interface

- `data_store.py:PEPDataManager` (class)
- `orchestrator.py:WorkflowStatus` (class)
- `orchestrator.py:WorkflowStep` (class)
- `orchestrator.py:PEPOrchestrator` (class)
- `pep_engine.py:PEPEngine` (class)
- `validator.py:ValidationResult` (class)
- `validator.py:PEPValidator` (class)

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
