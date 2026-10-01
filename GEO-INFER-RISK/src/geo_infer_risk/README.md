# GEO-INFER-RISK/src/geo_infer_risk

Geo Infer Risk workspace within `GEO-INFER-RISK`.

## Contents

- `config/`
- `core/`
- `utils/`
- `__init__.py`
- `civic_intel.py`

## Public Interface

- `__init__.py:create_risk_analysis` (function)
- `civic_intel.py:CrescentCityBounds` (class)
- `civic_intel.py:CrescentCityAnchor` (class)
- `civic_intel.py:MunicipalCodeSection` (class)
- `civic_intel.py:CivicHazardDomain` (class)
- `civic_intel.py:CrescentCityHazardIntel` (class)
- `civic_intel.py:parse_crescent_city_hazard` (function)
- `civic_intel.py:load_crescent_city_hazard` (function)
- `civic_intel.py:crescent_city_hazard_weights` (function)

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
