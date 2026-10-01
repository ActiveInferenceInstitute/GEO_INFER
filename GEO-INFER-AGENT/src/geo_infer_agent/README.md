# GEO-INFER-AGENT/src/geo_infer_agent

Geo Infer Agent workspace within `GEO-INFER-AGENT`.

## Contents

- `agents/`
- `api/`
- `core/`
- `models/`
- `__init__.py`
- `cli.py`

## Public Interface

- `cli.py:setup_logging` (function)
- `cli.py:load_config` (function)
- `cli.py:load_agent_class` (function)
- `cli.py:run_agent` (function)
- `cli.py:list_agents_command` (function)
- `cli.py:create_config_command` (function)
- `cli.py:main` (function)

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
