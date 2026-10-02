# GEO-INFER-BIO/src/geo_infer_bio/api

Api workspace within `GEO-INFER-BIO`.

## Contents

- `__init__.py`
- `graphql_api.py`
- `rest_api.py`

## Public Interface

- `graphql_api.py:SpatialData` (class)
- `graphql_api.py:SequenceData` (class)
- `graphql_api.py:AnalysisResult` (class)
- `graphql_api.py:VisualizationData` (class)
- `graphql_api.py:SpatialDataInput` (class)
- `graphql_api.py:SequenceDataInput` (class)
- `graphql_api.py:AnalysisResultInput` (class)
- `graphql_api.py:Query` (class)
- `rest_api.py:SpatialData` (class)
- `rest_api.py:SequenceData` (class)
- `rest_api.py:AnalysisResult` (class)
- `rest_api.py:root` (function)
- `rest_api.py:analyze_sequence` (function)
- `rest_api.py:analyze_file` (function)
- `rest_api.py:visualize_spatial` (function)
- `rest_api.py:health_check` (function)

## Module Metadata

- Module: `GEO-INFER-BIO`
- Package: `geo_infer_bio`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-bio`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module BIO`

## Dependencies

- `biopython>=1.79`
- `fastapi>=0.100.0`
- `geopandas>=0.13.0`
- `graphql-core>=3.1.0`
- `matplotlib>=3.4.0`
- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `pydantic>=2.0.0`
- `python-multipart>=0.0.7`
- `requests>=2.28`
- `seaborn>=0.11.0`
- `shapely>=2.0.0`
- `strawberry-graphql>=0.96.0`
- `uvicorn>=0.15.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module BIO
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
