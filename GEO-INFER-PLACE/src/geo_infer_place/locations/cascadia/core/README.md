# GEO-INFER-PLACE/src/geo_infer_place/locations/cascadia/core

Core workspace within `GEO-INFER-PLACE`.

## Contents

- `visualization/`
- `__init__.py`
- `analysis_engine.py`
- `data_cleanup.py`
- `data_processor.py`
- `download_empirical_data.py`
- `empirical_data_assessment.py`
- `enhanced_config.py`
- `enhanced_data_manager.py`
- `enhanced_h3_fusion.py`
- `enhanced_logging.py`
- `geo_infer_integrations.py`
- `real_data_acquisition.py`
- `reporting_engine.py`
- `setup_manager.py`

## Public Interface

- `analysis_engine.py:perform_enhanced_spatial_analysis` (function)
- `analysis_engine.py:run_comprehensive_analysis` (function)
- `data_cleanup.py:DataCleanupManager` (class)
- `data_cleanup.py:create_data_cleanup_manager` (function)
- `data_processor.py:initialize_modules` (function)
- `data_processor.py:create_shared_backend` (function)
- `data_processor.py:export_results` (function)
- `data_processor.py:validate_data_acquisition` (function)
- `download_empirical_data.py:EmpiricalDataDownloader` (class)
- `download_empirical_data.py:main` (function)
- `empirical_data_assessment.py:EmpiricalDataAssessor` (class)
- `empirical_data_assessment.py:main` (function)
- `enhanced_config.py:AnalysisConfig` (class)
- `enhanced_config.py:VisualizationConfig` (class)
- `enhanced_config.py:DataConfig` (class)
- `enhanced_config.py:ModuleConfig` (class)
- `enhanced_config.py:CascadiaConfig` (class)
- `enhanced_config.py:EnhancedConfigManager` (class)
- `enhanced_config.py:create_enhanced_config_manager` (function)
- `enhanced_data_manager.py:EnhancedDataManager` (class)

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
