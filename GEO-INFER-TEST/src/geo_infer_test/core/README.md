# GEO-INFER-TEST/src/geo_infer_test/core

Core workspace within `GEO-INFER-TEST`.

## Contents

- `__init__.py`
- `log_integration.py`
- `module_health.py`
- `performance_monitor.py`
- `test_discoverer.py`
- `test_orchestrator.py`
- `test_runner.py`
- `validators.py`

## Public Interface

- `log_integration.py:TestLogEntry` (class)
- `log_integration.py:ModuleTestSummary` (class)
- `log_integration.py:LogIntegration` (class)
- `log_integration.py:LoggingTestReporter` (class)
- `log_integration.py:LogAnalyzer` (class)
- `module_health.py:HealthMetrics` (class)
- `module_health.py:ModuleHealthChecker` (class)
- `module_health.py:SystemValidator` (class)
- `module_health.py:DependencyChecker` (class)
- `performance_monitor.py:PerformanceMonitor` (class)
- `performance_monitor.py:BenchmarkRunner` (class)
- `performance_monitor.py:LoadTester` (class)
- `performance_monitor.py:MetricsCollector` (class)
- `performance_monitor.py:PerformanceAnalyzer` (class)
- `validators.py:BaseValidator` (class)
- `validators.py:DataQualityValidator` (class)
- `validators.py:SpatialValidator` (class)
- `validators.py:IoTValidator` (class)
- `validators.py:BayesianValidator` (class)
- `validators.py:PerformanceValidator` (class)

## Module Metadata

- Module: `GEO-INFER-TEST`
- Package: `geo_infer_test`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-test`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module TEST`

## Dependencies

- `coverage[toml]>=7.0.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `hypothesis>=6.0.0`
- `matplotlib>=3.5.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `psutil>=5.9.0`
- `pytest>=7.0.0`
- `pytest-benchmark>=4.0.0`
- `pytest-cov>=4.0.0`
- `pytest-html>=3.1.0`
- `pytest-mock>=3.10.0`
- `pytest-timeout>=2.1.0`
- `pytest-xdist>=3.0.0`
- `pyyaml>=6.0`


## Validation

```bash
uv sync --all-packages --all-extras --all-groups
uv run python GEO-INFER-TEST/run_unified_tests.py --module TEST
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
