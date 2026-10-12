# GEO-INFER-AGENT/src/geo_infer_agent/models/bdi

Bdi workspace within `GEO-INFER-AGENT`.

## Contents

- `__init__.py`
- `agent.py`
- `belief.py`
- `desire.py`
- `intention.py`
- `plan.py`

## Public Interface

- `agent.py:BDIState` (class)
- `agent.py:BDIAgent` (class)
- `belief.py:Belief` (class)
- `belief.py:BeliefBase` (class)
- `desire.py:DesireState` (class)
- `desire.py:Desire` (class)
- `desire.py:DesireSet` (class)
- `intention.py:IntentionStatus` (class)
- `intention.py:Intention` (class)
- `intention.py:IntentionStructure` (class)
- `plan.py:PlanStatus` (class)
- `plan.py:Plan` (class)
- `plan.py:PlanLibrary` (class)

## Module Metadata

- Module: `GEO-INFER-AGENT`
- Package: `geo_infer_agent`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-agent`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module AGENT`

## Dependencies

- `numpy>=1.23.5`
- `torch>=2.0.0`
- `pyyaml>=6.0`
- `requests>=2.28.2`
- `fastapi>=0.104.0`
- `pydantic>=2.5.0`
- `pandas>=1.3.0`
- `uvicorn>=0.24.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module AGENT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
