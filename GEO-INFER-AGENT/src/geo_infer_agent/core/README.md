# GEO-INFER-AGENT/src/geo_infer_agent/core

Core workspace within `GEO-INFER-AGENT`.

## Contents

- `active_inference.py`
- `agent_base.py`
- `agent_registry.py`
- `llm_proxy.py`

## Public Interface

- `active_inference.py:ActiveInferenceConfig` (class)
- `active_inference.py:GenerativeModel` (class)
- `active_inference.py:ActiveInferenceAgent` (class)
- `agent_base.py:AgentState` (class)
- `agent_base.py:BaseAgent` (class)
- `agent_base.py:ExampleAgent` (class)
- `agent_registry.py:AgentRegistry` (class)
- `llm_proxy.py:LLMProxyPolicyError` (class)
- `llm_proxy.py:LLMProxyPolicy` (class)
- `llm_proxy.py:TokenBucket` (class)
- `llm_proxy.py:check_allowed_model` (function)
- `llm_proxy.py:check_request_size` (function)
- `llm_proxy.py:check_output_tokens` (function)
- `llm_proxy.py:enforce_llm_proxy_policy` (function)

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
