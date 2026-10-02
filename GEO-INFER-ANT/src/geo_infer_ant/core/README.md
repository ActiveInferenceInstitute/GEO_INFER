# GEO-INFER-ANT/src/geo_infer_ant/core

Core workspace within `GEO-INFER-ANT`.

## Contents

- `__init__.py`
- `agent_base.py`
- `digital_stigmergy.py`
- `population.py`
- `stigmergy.py`

## Public Interface

- `agent_base.py:SensoryInput` (class)
- `agent_base.py:ActionDecision` (class)
- `agent_base.py:SwarmAgent` (class)
- `digital_stigmergy.py:DigitalTrace` (class)
- `digital_stigmergy.py:InformationQuery` (class)
- `digital_stigmergy.py:DigitalStigmergy` (class)
- `population.py:PopulationConfig` (class)
- `population.py:EnvironmentalState` (class)
- `population.py:SimulationResults` (class)
- `population.py:AgentPopulation` (class)
- `stigmergy.py:PheromoneType` (class)
- `stigmergy.py:PheromoneDeposit` (class)
- `stigmergy.py:PheromoneField` (class)
- `stigmergy.py:PheromoneSystem` (class)

## Module Metadata

- Module: `GEO-INFER-ANT`
- Package: `geo_infer_ant`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-ant`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ANT`

## Dependencies

- `jsonschema>=4.0.0`
- `networkx>=2.8`
- `numpy>=1.21.0`
- `pyyaml>=6.0`
- `scikit-learn>=1.1.0`
- `scipy>=1.7.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ANT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
