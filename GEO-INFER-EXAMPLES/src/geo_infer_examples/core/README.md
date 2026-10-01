# GEO-INFER-EXAMPLES/src/geo_infer_examples/core

Core workspace within `GEO-INFER-EXAMPLES`.

## Contents

- `__init__.py`
- `module_orchestrator.py`

## Public Interface

- `module_orchestrator.py:ConfigManager` (class)
- `module_orchestrator.py:setup_logging` (function)
- `module_orchestrator.py:APIConnector` (class)
- `module_orchestrator.py:PerformanceMonitor` (class)
- `module_orchestrator.py:ExecutionStrategy` (class)
- `module_orchestrator.py:ModuleStatus` (class)
- `module_orchestrator.py:WorkflowExecution` (class)
- `module_orchestrator.py:ModuleOrchestrator` (class)
- `module_orchestrator.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-EXAMPLES`
- Package: `geo_infer_examples`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-examples`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES`

## Dependencies

- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
