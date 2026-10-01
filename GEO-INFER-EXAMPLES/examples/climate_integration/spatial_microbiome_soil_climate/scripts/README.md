# GEO-INFER-EXAMPLES/examples/climate_integration/spatial_microbiome_soil_climate/scripts

Scripts workspace within `GEO-INFER-EXAMPLES`.

## Contents

- `output/`
- `run_example.py`
- `run_spatial_integration.py`

## Public Interface

- `run_example.py:setup_logging` (function)
- `run_example.py:ClimateAnalysisSystem` (class)
- `run_example.py:main` (function)
- `run_spatial_integration.py:SpatialMicrobiomeIntegrator` (class)
- `run_spatial_integration.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-EXAMPLES`
- Package: `geo_infer_examples`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-examples`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES`

## Dependencies

- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
