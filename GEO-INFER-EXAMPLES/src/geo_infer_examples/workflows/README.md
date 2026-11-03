# GEO-INFER-EXAMPLES/src/geo_infer_examples/workflows

Workflows workspace within `GEO-INFER-EXAMPLES`.

## Contents

- `health_surveillance_basic.yaml`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-EXAMPLES`
- Package: `geo_infer_examples`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-examples`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES`

## Dependencies

- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES
```


## Workflow Definitions

- `health_surveillance_basic.yaml` is a packaged sequential workflow definition
  consumed by the examples module's orchestration surfaces.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
