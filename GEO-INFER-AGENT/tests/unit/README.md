# GEO-INFER-AGENT/tests/unit

Unit workspace within `GEO-INFER-AGENT`.

## Contents

- `models/`
- `test_agent_base.py`
- `test_agent_communication.py`
- `test_agent_start_task_retention.py`
- `test_api_endpoints.py`
- `test_bdi_agent_and_services.py`
- `test_bdi_placeholders.py`
- `test_cli.py`
- `test_coordination.py`
- `test_core_active_inference.py`
- `test_data_collector.py`
- `test_data_collector_agent.py`
- `test_hybrid.py`
- `test_hybrid_agent.py`
- `test_interface_and_cli.py`
- `test_llm_proxy.py`
- `test_messaging.py`
- `test_package_import_hygiene.py`
- `test_planning.py`
- `test_rl_state_index.py`
- `test_rl_unit.py`
- `test_rule_based.py`
- `test_rule_based_agent.py`
- `test_task_management.py`
- `test_telemetry.py`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-AGENT`
- Package: `geo_infer_agent`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-agent`
- Tests: `uv run python -m pytest GEO-INFER-AGENT/tests/unit`

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
uv run python -m pytest GEO-INFER-AGENT/tests/unit
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
