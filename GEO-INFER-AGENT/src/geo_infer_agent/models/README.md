# GEO-INFER-AGENT/src/geo_infer_agent/models

Models workspace within `GEO-INFER-AGENT`.

## Contents

- `bdi/`
- `schemas/`
- `__init__.py`
- `active_inference.py`
- `hybrid.py`
- `rl.py`
- `rule_based.py`

## Public Interface

- `active_inference.py:GenerativeModel` (class)
- `active_inference.py:ActiveInferenceState` (class)
- `active_inference.py:ActiveInferenceAgent` (class)
- `hybrid.py:SubAgentWrapper` (class)
- `hybrid.py:HybridState` (class)
- `hybrid.py:HybridAgent` (class)
- `rl.py:Experience` (class)
- `rl.py:QTable` (class)
- `rl.py:ReplayBuffer` (class)
- `rl.py:RLState` (class)
- `rl.py:RLAgent` (class)
- `rule_based.py:Rule` (class)
- `rule_based.py:RuleSet` (class)
- `rule_based.py:RuleBasedState` (class)
- `rule_based.py:RuleBasedAgent` (class)

## Module Metadata

- Module: `GEO-INFER-AGENT`
- Package: `geo_infer_agent`
- Version: `0.3.0`
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
