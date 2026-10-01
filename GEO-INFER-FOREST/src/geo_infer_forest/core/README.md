# GEO-INFER-FOREST/src/geo_infer_forest/core

Core workspace within `GEO-INFER-FOREST`.

## Contents

- `__init__.py`
- `canopy_analysis.py`
- `carbon_sequestration.py`
- `deforestation.py`
- `fire_risk.py`
- `forest_health.py`
- `forest_inventory.py`
- `wildfire_risk.py`

## Public Interface

- `canopy_analysis.py:CanopyAnalyzer` (class)
- `carbon_sequestration.py:CarbonSequestrationModeler` (class)
- `deforestation.py:DeforestationDetector` (class)
- `fire_risk.py:FireRiskAssessor` (class)
- `forest_health.py:ForestHealthMonitor` (class)
- `forest_inventory.py:ForestInventory` (class)
- `wildfire_risk.py:FireDangerRating` (class)
- `wildfire_risk.py:FuelType` (class)
- `wildfire_risk.py:FireWeatherObservation` (class)
- `wildfire_risk.py:FireIncident` (class)
- `wildfire_risk.py:WildfireRiskAnalyzer` (class)

## Module Metadata

- Module: `GEO-INFER-FOREST`
- Package: `geo_infer_forest`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-forest`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module FOREST`

## Dependencies

- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `xarray>=0.19.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module FOREST
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
