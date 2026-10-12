# GEO-INFER-TRANSPORT/examples

Examples workspace within `GEO-INFER-TRANSPORT`.

## Contents

- `multimodal_analysis.py`
- `traffic_simulation.py`

## Public Interface

- `multimodal_analysis.py:build_sample_network` (function)
- `multimodal_analysis.py:routing_example` (function)
- `multimodal_analysis.py:traffic_example` (function)
- `multimodal_analysis.py:transit_example` (function)
- `multimodal_analysis.py:main` (function)
- `traffic_simulation.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-TRANSPORT`
- Package: `geo_infer_transport`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-transport`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module TRANSPORT`

## Dependencies

- `networkx>=2.6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module TRANSPORT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
