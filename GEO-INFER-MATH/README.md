# GEO-INFER-MATH

Core mathematical and statistical engine providing geometric operations, spatial statistics, and numerical methods for geospatial analysis.

## Contents

- `docs/`
- `examples/`
- `src/`
- `tests/`
- `CHANGELOG.md`
- `IMPROVEMENTS.md`
- `SKILL.md`
- `pyproject.toml`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-MATH`
- Package: `geo_infer_math`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-math`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH`

## Dependencies

- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `sympy>=1.9.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH
```


## Information Theory Contracts

- MIMO rates honor explicit total power divided among transmitters; `None`
  preserves unit power per transmitter. Singular-mode evaluation avoids forming
  overflowing Gram matrices or determinants.
- Waterfilling uses power gains and leaves zero-gain channels unallocated.
- Spatial information-theory entropy preserves its selected logarithm base.
- See [communication and entropy conventions](docs/channel_capacity.md).

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
