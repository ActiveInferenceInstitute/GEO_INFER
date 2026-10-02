# GEO-INFER-SPACE/src/geo_infer_space/nested/analytics

Analytics workspace within `GEO-INFER-SPACE`.

## Contents

- `__init__.py`
- `flow_analysis.py`
- `hierarchy_metrics.py`
- `pattern_detection.py`
- `performance_metrics.py`

## Public Interface

- `flow_analysis.py:FlowType` (class)
- `flow_analysis.py:FlowDirection` (class)
- `flow_analysis.py:FlowPattern` (class)
- `flow_analysis.py:FlowVector` (class)
- `flow_analysis.py:FlowField` (class)
- `flow_analysis.py:FlowAnalysisResult` (class)
- `flow_analysis.py:H3FlowAnalyzer` (class)
- `hierarchy_metrics.py:HierarchyMetric` (class)
- `hierarchy_metrics.py:HierarchyStructure` (class)
- `hierarchy_metrics.py:HierarchyNode` (class)
- `hierarchy_metrics.py:HierarchyMetrics` (class)
- `hierarchy_metrics.py:HierarchyAnalysisResult` (class)
- `hierarchy_metrics.py:H3HierarchyAnalyzer` (class)
- `pattern_detection.py:PatternType` (class)
- `pattern_detection.py:PatternScale` (class)
- `pattern_detection.py:DetectionMethod` (class)
- `pattern_detection.py:Pattern` (class)
- `pattern_detection.py:PatternDetectionResult` (class)
- `pattern_detection.py:H3PatternDetector` (class)
- `performance_metrics.py:PerformanceMetric` (class)

## Module Metadata

- Module: `GEO-INFER-SPACE`
- Package: `geo_infer_space`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-space`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE`

## Dependencies

- `geo-infer-time>=0.4.0`
- `fastapi>=0.100.0`
- `fiona>=1.8.0`
- `geojson-pydantic>=2.0.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `networkx>=2.6.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `psutil>=5.9.0`
- `pydantic>=2.0.0`
- `pyproj>=3.3.0`
- `python-multipart>=0.0.5`
- `pyyaml>=6.0`
- `requests>=2.28.0`
- `rasterio>=1.3.0`
- `scikit-learn>=1.0.0`
- `scipy>=1.7.0`
- `shapely>=2.0.0`
- `uvicorn>=0.15.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
