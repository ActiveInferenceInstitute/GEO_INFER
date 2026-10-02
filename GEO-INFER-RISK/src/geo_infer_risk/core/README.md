# GEO-INFER-RISK/src/geo_infer_risk/core

Core workspace within `GEO-INFER-RISK`.

## Contents

- `__init__.py`
- `catastrophe_models.py`
- `exposure_model.py`
- `hazard_model.py`
- `insurance_models.py`
- `risk_engine.py`
- `risk_models.py`
- `vulnerability_model.py`

## Public Interface

- `catastrophe_models.py:CatastropheConfig` (class)
- `catastrophe_models.py:MultiHazardInteractionMatrix` (class)
- `catastrophe_models.py:calculate_compound_exceedance_probability` (function)
- `catastrophe_models.py:EnhancedCatastropheModel` (class)
- `catastrophe_models.py:EnhancedEarthquakeModel` (class)
- `catastrophe_models.py:EnhancedHurricaneModel` (class)
- `catastrophe_models.py:EnhancedFloodModel` (class)
- `catastrophe_models.py:create_enhanced_earthquake_model` (function)
- `catastrophe_models.py:create_enhanced_hurricane_model` (function)
- `catastrophe_models.py:create_enhanced_flood_model` (function)
- `catastrophe_models.py:CatastropheModelManager` (class)
- `exposure_model.py:EnhancedExposureModel` (class)
- `exposure_model.py:EnhancedPropertyExposureModel` (class)
- `exposure_model.py:EnhancedPopulationExposureModel` (class)
- `exposure_model.py:EnhancedInfrastructureExposureModel` (class)
- `exposure_model.py:create_enhanced_property_exposure_model` (function)
- `exposure_model.py:create_enhanced_population_exposure_model` (function)
- `exposure_model.py:create_enhanced_infrastructure_exposure_model` (function)
- `hazard_model.py:EnhancedHazardModel` (class)
- `hazard_model.py:EnhancedFloodModel` (class)

## Module Metadata

- Module: `GEO-INFER-RISK`
- Package: `geo_infer_risk`
- Version: `0.4.0`
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
