# GEO-INFER-MATH/src/geo_infer_math/integration/ai

Ai workspace within `GEO-INFER-MATH`.

## Contents

- `__init__.py`
- `gradient_helpers.py`
- `loss_functions.py`
- `optimization_bridges.py`
- `spatial_attention.py`
- `tensor_operations.py`

## Public Interface

- `gradient_helpers.py:AIGradientHelpers` (class)
- `loss_functions.py:SpatialLossFunctions` (class)
- `optimization_bridges.py:OptimizationBridges` (class)
- `spatial_attention.py:SpatialAttention` (class)
- `tensor_operations.py:SpatialTensorOperations` (class)

## Module Metadata

- Module: `GEO-INFER-MATH`
- Package: `geo_infer_math`
- Version: `0.3.0`
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


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
