# GEO-INFER-APP/src/geo_infer_app/models

Models workspace within `GEO-INFER-APP`.

## Contents

- `interfaces/`
- `__init__.py`
- `agent_configuration.py`
- `agent_factory.py`
- `agent_interface.py`
- `agent_visualization.py`

## Public Interface

- `agent_configuration.py:ConfigFieldType` (class)
- `agent_configuration.py:ConfigField` (class)
- `agent_configuration.py:AgentConfigSchema` (class)
- `agent_configuration.py:AgentConfiguration` (class)
- `agent_factory.py:AgentFactory` (class)
- `agent_interface.py:AgentType` (class)
- `agent_interface.py:AgentState` (class)
- `agent_interface.py:AgentInterface` (class)
- `agent_visualization.py:VisualizationType` (class)
- `agent_visualization.py:VisualizationConfig` (class)
- `agent_visualization.py:AgentVisualization` (class)

## Module Metadata

- Module: `GEO-INFER-APP`
- Package: `geo_infer_app`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-app`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module APP`

## Dependencies

- Dependencies are declared in `pyproject.toml` or inherited from the workspace.


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module APP
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
