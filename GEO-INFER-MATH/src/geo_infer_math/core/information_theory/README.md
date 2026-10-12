# GEO-INFER-MATH/src/geo_infer_math/core/information_theory

Information Theory workspace within `GEO-INFER-MATH`.

## Contents

- `__init__.py`
- `channel_capacity.py`
- `entropy.py`
- `information_geometry.py`
- `kl_divergence.py`
- `mutual_information.py`
- `spatial_coding.py`

## Public Interface

- `channel_capacity.py:channel_capacity` (function)
- `channel_capacity.py:spatial_channel_capacity` (function)
- `channel_capacity.py:awgn_channel_capacity` (function)
- `channel_capacity.py:mimo_channel_capacity` (function)
- `channel_capacity.py:waterfilling_power_allocation` (function)
- `channel_capacity.py:ChannelCapacityCalculator` (class)
- `entropy.py:shannon_entropy` (function)
- `entropy.py:renyi_entropy` (function)
- `entropy.py:tsallis_entropy` (function)
- `entropy.py:spatial_entropy` (function)
- `entropy.py:conditional_entropy` (function)
- `entropy.py:joint_entropy` (function)
- `entropy.py:EntropyCalculator` (class)
- `information_geometry.py:fisher_information_matrix` (function)
- `information_geometry.py:information_metric` (function)
- `information_geometry.py:geodesic_distance` (function)
- `information_geometry.py:information_distance` (function)
- `information_geometry.py:spatial_fisher_information` (function)
- `information_geometry.py:InformationGeometryCalculator` (class)
- `kl_divergence.py:kl_divergence` (function)

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
