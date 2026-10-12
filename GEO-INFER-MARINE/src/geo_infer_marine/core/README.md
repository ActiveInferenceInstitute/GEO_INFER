# GEO-INFER-MARINE/src/geo_infer_marine/core

Core workspace within `GEO-INFER-MARINE`.

## Contents

- `__init__.py`
- `coastal_analysis.py`
- `coral_reef.py`
- `marine_ecosystems.py`
- `marine_spatial_planning.py`
- `ocean_currents.py`
- `oceanographic_data.py`
- `sea_level.py`
- `water_quality.py`

## Public Interface

- `coastal_analysis.py:CoastalAnalyzer` (class)
- `coral_reef.py:CoralReefAssessor` (class)
- `marine_ecosystems.py:MarineHabitatType` (class)
- `marine_ecosystems.py:SpeciesData` (class)
- `marine_ecosystems.py:MarineEcosystemModeler` (class)
- `marine_spatial_planning.py:MarineSpatialPlanner` (class)
- `ocean_currents.py:OceanCurrentModeler` (class)
- `oceanographic_data.py:OceanographicDataProcessor` (class)
- `sea_level.py:SeaLevelAnalyzer` (class)
- `water_quality.py:MarineWaterQuality` (class)

## Module Metadata

- Module: `GEO-INFER-MARINE`
- Package: `geo_infer_marine`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-marine`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module MARINE`

## Dependencies

- `numpy>=1.20.0`
- `xarray>=0.19.0`
- `netcdf4>=1.5.8`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module MARINE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
