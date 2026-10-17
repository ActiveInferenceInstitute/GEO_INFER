# GEO-INFER-API/src/geo_infer_api

Geo Infer Api workspace within `GEO-INFER-API`.

## Contents

- `core/`
- `endpoints/`
- `models/`
- `utils/`
- `__init__.py`
- `agent_benchmarks.py`
- `agent_catalog.py`
- `agent_cli.py`
- `agent_diagnostics.py`
- `agent_mcp.py`
- `agent_runtime.py`
- `agent_schema.py`
- `agent_skills.py`
- `agent_stdio.py`
- `app.py`
- `agent_skills.json`

## Public Interface

- `agent_benchmarks.py:benchmark_agent` (function)
- `agent_benchmarks.py:main` (function)
- `agent_catalog.py:package_name` (function)
- `agent_catalog.py:module_inventory` (function)
- `agent_catalog.py:definitions` (function)
- `agent_catalog.py:catalog` (function)
- `agent_cli.py:main` (function)
- `agent_diagnostics.py:runtime_diagnostics` (function)
- `agent_mcp.py:create_server` (function)
- `agent_mcp.py:main` (function)
- `agent_runtime.py:resolve` (function)
- `agent_runtime.py:describe` (function)
- `agent_runtime.py:AgentSession` (class)
- `agent_schema.py:validate_request` (function)
- `agent_schema.py:request_schema` (function)
- `agent_skills.py:build_bundle` (function)
- `agent_skills.py:bundle_bytes` (function)
- `agent_skills.py:read_guidance` (function)
- `agent_skills.py:skill_inventory` (function)
- `agent_skills.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-API`
- Package: `geo_infer_api`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-api`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module API`

## Dependencies

- `starlette>=0.27.0`
- `fastapi>=0.100.0`
- `pydantic>=2.0.0`
- `pydantic-settings>=2.0.0`
- `uvicorn>=0.21.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module API
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
