# GEO-INFER-ANT/src/geo_infer_ant/utils

Utils workspace within `GEO-INFER-ANT`.

## Contents

- `__init__.py`
- `config.py`
- `integration.py`
- `logging.py`
- `spatial.py`

## Public Interface

- `config.py:SwarmConfig` (class)
- `config.py:AlgorithmConfig` (class)
- `config.py:StigmergyConfig` (class)
- `config.py:SpatialConfig` (class)
- `config.py:PerformanceConfig` (class)
- `config.py:LoggingConfig` (class)
- `config.py:AntModuleConfig` (class)
- `config.py:load_config` (function)
- `config.py:validate_config` (function)
- `config.py:config_to_dict` (function)
- `config.py:save_config` (function)
- `config.py:get_default_config` (function)
- `config.py:update_config` (function)
- `integration.py:IntegrationManager` (class)
- `logging.py:setup_logging` (function)
- `spatial.py:parse_h3_resolution` (function)
- `spatial.py:validate_bounds` (function)
- `spatial.py:validate_numeric_matrix` (function)

## Module Metadata

- Module: `GEO-INFER-ANT`
- Package: `geo_infer_ant`
- Version: `0.3.0`
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
