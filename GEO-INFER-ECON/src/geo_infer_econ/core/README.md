# GEO-INFER-ECON/src/geo_infer_econ/core

Core workspace within `GEO-INFER-ECON`.

## Contents

- `__init__.py`
- `econometrics_engine.py`
- `modeling_engine.py`
- `policy_engine.py`

## Public Interface

- `econometrics_engine.py:SpatialWeightsConfig` (class)
- `econometrics_engine.py:EconometricResults` (class)
- `econometrics_engine.py:SpatialEconometricsEngine` (class)
- `modeling_engine.py:ModelConfiguration` (class)
- `modeling_engine.py:EconomicModelingEngine` (class)
- `policy_engine.py:PolicyType` (class)
- `policy_engine.py:PolicyScenario` (class)
- `policy_engine.py:PolicyImpact` (class)
- `policy_engine.py:PolicyComparison` (class)
- `policy_engine.py:PolicyAnalysisEngine` (class)

## Module Metadata

- Module: `GEO-INFER-ECON`
- Package: `geo_infer_econ`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-econ`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ECON`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `geopandas>=0.13.0`
- `shapely>=2.0.0`
- `scikit-learn>=1.0.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.12.0`
- `h3>=4.5.0,<5`
- `requests>=2.28.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ECON
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
