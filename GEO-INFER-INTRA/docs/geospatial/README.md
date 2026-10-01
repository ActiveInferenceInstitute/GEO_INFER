# GEO-INFER-INTRA/docs/geospatial

Geospatial workspace within `GEO-INFER-INTRA`.

## Contents

- `algorithms/`
- `analysis/`
- `case_studies/`
- `concepts/`
- `data_formats/`
- `getting_started/`
- `standards/`
- `visualization/`
- `index.md`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-INTRA`
- Package: `geo_infer_intra`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-intra`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA`

## Dependencies

- `h3>=4.5.0,<5`
- `jsonschema>=4.0.0`
- `pillow>=10.0.0`
- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
