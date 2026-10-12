# GEO-INFER-ECON/examples

Examples workspace within `GEO-INFER-ECON`.

## Contents

- `comprehensive_economic_analysis.py`
- `integration_example.py`

## Public Interface

- `comprehensive_economic_analysis.py:create_sample_bioregion` (function)
- `comprehensive_economic_analysis.py:analyze_consumer_behavior` (function)
- `comprehensive_economic_analysis.py:analyze_regional_growth` (function)
- `comprehensive_economic_analysis.py:analyze_ecosystem_services` (function)
- `comprehensive_economic_analysis.py:integrated_analysis` (function)
- `comprehensive_economic_analysis.py:main` (function)
- `integration_example.py:example_spatial_economic_analysis` (function)
- `integration_example.py:example_temporal_economic_analysis` (function)
- `integration_example.py:example_data_integration` (function)
- `integration_example.py:example_integrated_analysis` (function)
- `integration_example.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-ECON`
- Package: `geo_infer_econ`
- Version: `0.4.0`
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
