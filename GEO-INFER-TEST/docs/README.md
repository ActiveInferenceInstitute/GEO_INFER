# GEO-INFER-TEST/docs

Docs workspace within `GEO-INFER-TEST`.

## Contents

- `examples/`
- `api_reference.md`
- `benchmark_baseline_2026-09-10.md`
- `getting_started.md`
- `gnn_continuation_2026_09.md`
- `gnn_space_time_2026_09.md`
- `import_latency_2026_09.md`
- `index.md`
- `perf01-act-importtime.txt`
- `perf01-pandas-importtime.txt`
- `pin_review_2026-Q3.md`
- `secret_scan_policy.md`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-TEST`
- Package: `geo_infer_test`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-test`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module TEST`

## Dependencies

- `coverage[toml]>=7.0.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `hypothesis>=6.0.0`
- `matplotlib>=3.5.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `psutil>=5.9.0`
- `pytest>=7.0.0`
- `pytest-benchmark>=4.0.0`
- `pytest-cov>=4.0.0`
- `pytest-html>=3.1.0`
- `pytest-mock>=3.10.0`
- `pytest-timeout>=2.1.0`
- `pytest-xdist>=3.0.0`
- `pyyaml>=6.0`


## Validation

```bash
uv sync --all-packages --all-extras --all-groups
uv run python GEO-INFER-TEST/run_unified_tests.py --module TEST
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
