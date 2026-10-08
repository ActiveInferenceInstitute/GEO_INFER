# GEO-INFER-MATH/src/geo_infer_math/core

Core workspace within `GEO-INFER-MATH`.

## Contents

- `information_theory/`
- `theorem_proving/`
- `__init__.py`
- `circulation.py`
- `geometry.py`
- `gpu_acceleration.py`
- `graph_theory.py`
- `interpolation.py`
- `linalg_tensor.py`
- `numerical_methods.py`
- `optimization.py`
- `spatial_statistics.py`
- `symbolic_math.py`
- `transforms.py`

## Public Interface

- `circulation.py:skew_part` (function)
- `circulation.py:probability_current` (function)
- `circulation.py:circulation_part` (function)
- `circulation.py:current_divergence` (function)
- `circulation.py:satisfies_detailed_balance` (function)
- `circulation.py:GraphHodgeDecomposition` (class)
- `circulation.py:graph_hodge_decomposition` (function)
- `geometry.py:Point` (class)
- `geometry.py:LineString` (class)
- `geometry.py:Polygon` (class)
- `geometry.py:haversine_distance` (function)
- `geometry.py:vincenty_distance` (function)
- `geometry.py:bearing` (function)
- `geometry.py:destination_point` (function)
- `geometry.py:point_in_polygon` (function)
- `geometry.py:points_in_polygon_vectorized` (function)
- `geometry.py:buffer_point` (function)
- `geometry.py:line_intersection` (function)
- `geometry.py:polygon_area_spherical` (function)
- `geometry.py:great_circle_distance` (function)

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


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
