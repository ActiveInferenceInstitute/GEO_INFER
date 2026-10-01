# GEO-INFER-TEST/demo

Demo workspace within `GEO-INFER-TEST`.

## Contents

- `__init__.py`
- `crescent_city_civic_intel_demo.py`

## Public Interface

- `crescent_city_civic_intel_demo.py:bundled_contract_path` (function)
- `crescent_city_civic_intel_demo.py:load_bundled_contract` (function)
- `crescent_city_civic_intel_demo.py:geo_views_agree` (function)
- `crescent_city_civic_intel_demo.py:build_iso_geo_parity` (function)
- `crescent_city_civic_intel_demo.py:build_geo_parity` (function)
- `crescent_city_civic_intel_demo.py:build_summary` (function)
- `crescent_city_civic_intel_demo.py:render_summary` (function)
- `crescent_city_civic_intel_demo.py:main` (function)

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
uv sync --all-packages --all-extras
uv run python GEO-INFER-TEST/run_unified_tests.py --module TEST
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
