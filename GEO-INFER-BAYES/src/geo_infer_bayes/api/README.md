# GEO-INFER-BAYES/src/geo_infer_bayes/api

Api workspace within `GEO-INFER-BAYES`.

## Contents

- `__init__.py`
- `pymc_interface.py`
- `stan_interface.py`
- `tfp_interface.py`

## Public Interface

- `pymc_interface.py:PyMCInterface` (class)
- `stan_interface.py:StanInterface` (class)
- `tfp_interface.py:TFPInterface` (class)

## Module Metadata

- Module: `GEO-INFER-BAYES`
- Package: `geo_infer_bayes`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-bayes`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module BAYES`

## Dependencies

- `arviz>=0.12.0`
- `matplotlib>=3.5.0`
- `numpy>=1.20.0,<2.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `tqdm>=4.60.0`
- `xarray>=2022.3.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module BAYES
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
