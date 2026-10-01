# GEO-INFER-SPM

Statistical parametric mapping methodology adapted for geospatial analysis to identify significant patterns in spatial-temporal data.

## Contents

- `config/`
- `docs/`
- `examples/`
- `src/`
- `tests/`
- `SKILL.md`
- `pyproject.toml`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-SPM`
- Package: `geo_infer_spm`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-SPM`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SPM`

## Dependencies

- `geopandas>=0.13.0`
- `h5py>=3.6.0`
- `matplotlib>=3.5.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `plotly>=5.0.0`
- `rasterio>=1.2.0`
- `scikit-learn>=1.0.0`
- `scipy>=1.7.0`
- `xarray>=0.20.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPM
```


## Random Field Theory Contracts

- `RandomFieldTheory` computes the complete Gaussian Euler characteristic from
  zero- through top-dimensional resel counts and exposes full-EC peak FWE
  thresholds.
- Cluster inference labels excursion components, measures extent in resels,
  and returns Poisson-clumping maximum-cluster FWE p-values; the default
  cluster-forming Gaussian tail is one-sided `p=0.001`.

## Visualization Contracts

- Statistical and interactive maps reject invalid contrast indices, empty or
  non-finite coordinates/statistics, and misaligned significance arrays.
- Diagnostic leverage and Cook's distance use a numerically stable hat-matrix
  calculation, and the package-level interactive-map export is unambiguous.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
