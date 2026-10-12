# GEO-INFER-PLACE/src/geo_infer_place/locations/del_norte_county

Del Norte County workspace within `GEO-INFER-PLACE`.

## Contents

- `config/`
- `dashboard/`
- `data/`
- `__init__.py`
- `coastal_resilience_analyzer.py`
- `comprehensive_dashboard.py`
- `crescent_city_intel.py`
- `fire_risk_assessor.py`
- `forest_health_monitor.py`
- `seismic_hazard_analyzer.py`

## Public Interface

- `coastal_resilience_analyzer.py:CoastalResilienceAnalyzer` (class)
- `comprehensive_dashboard.py:DelNorteComprehensiveDashboard` (class)
- `crescent_city_intel.py:MunicipalGeoIntelMapper` (class)
- `crescent_city_intel.py:CrescentCityIntelMapper` (class)
- `fire_risk_assessor.py:FireRiskAssessor` (class)
- `forest_health_monitor.py:ForestHealthMonitor` (class)
- `seismic_hazard_analyzer.py:SeismicHazardAnalyzer` (class)

## Module Metadata

- Module: `GEO-INFER-PLACE`
- Package: `geo_infer_place`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-place`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module PLACE`

## Dependencies

- `urllib3>=2.0.6`
- `geopandas>=0.13.0`
- `networkx>=2.6.0`
- `shapely>=2.0.0`
- `h3>=4.5.0,<5`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `pyyaml>=6.0`
- `folium>=0.14.0`
- `plotly>=5.0.0`
- `matplotlib>=3.5.0`
- `branca>=0.6.0`
- `requests>=2.28.0`
- `geo-infer-space>=0.4.0`
- `geo-infer-time>=0.4.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module PLACE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
