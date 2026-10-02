# GEO-INFER-WATER/src/geo_infer_water/core

Core workspace within `GEO-INFER-WATER`.

## Contents

- `__init__.py`
- `flood_drought.py`
- `hydrology.py`
- `infiltration.py`
- `water_balance.py`
- `water_infrastructure.py`
- `water_quality.py`
- `watershed_delineation.py`

## Public Interface

- `flood_drought.py:FloodDroughtAnalyzer` (class)
- `hydrology.py:HydrologicalModeler` (class)
- `infiltration.py:InfiltrationModeler` (class)
- `water_balance.py:WaterBalanceModeler` (class)
- `water_infrastructure.py:WaterInfrastructurePlanner` (class)
- `water_quality.py:WaterBodyType` (class)
- `water_quality.py:PollutantType` (class)
- `water_quality.py:WaterSample` (class)
- `water_quality.py:WaterQualityAssessor` (class)
- `watershed_delineation.py:WatershedDelineator` (class)

## Module Metadata

- Module: `GEO-INFER-WATER`
- Package: `geo_infer_water`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-water`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module WATER`

## Dependencies

- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `xarray>=0.19.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module WATER
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
