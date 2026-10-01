# GEO-INFER-RISK/examples

Examples workspace within `GEO-INFER-RISK`.

## Contents

- `basic_risk_assessment.py`
- `comprehensive_risk_assessment.py`

## Public Interface

- `basic_risk_assessment.py:build_loss_table` (function)
- `basic_risk_assessment.py:main` (function)
- `comprehensive_risk_assessment.py:build_parser` (function)
- `comprehensive_risk_assessment.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-RISK`
- Package: `geo_infer_risk`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-risk`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module RISK`

## Dependencies

- `jsonschema>=4.17.0`
- `pyyaml>=6.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `geopandas>=0.13.0`
- `shapely>=2.0.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module RISK
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
