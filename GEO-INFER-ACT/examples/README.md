# GEO-INFER-ACT/examples

Examples workspace within `GEO-INFER-ACT`.

## Contents

- `output/`
- `__init__.py`
- `ecological_model.py`
- `h3_active_inference.py`
- `modern_active_inference.py`
- `run_all_examples.py`
- `simple_model.py`
- `spatial_active_inference_gallery.py`
- `spatial_inference_demo.py`
- `urban_planning.py`

## Public Interface

- `ecological_model.py:main` (function)
- `h3_active_inference.py:main` (function)
- `modern_active_inference.py:main` (function)
- `run_all_examples.py:main` (function)
- `simple_model.py:main` (function)
- `spatial_active_inference_gallery.py:main` (function)
- `spatial_inference_demo.py:main` (function)
- `urban_planning.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-ACT`
- Package: `geo_infer_act`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-act`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT`

## Dependencies

- `matplotlib>=3.4.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `pyyaml>=6.0`
- `requests>=2.25.0`
- `scipy>=1.7.0`
- `seaborn>=0.11.0`
- `inferactively-pymdp==1.0.3`
- `h3>=4.5.0,<5`
- `geo-infer-bayes>=0.3.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
