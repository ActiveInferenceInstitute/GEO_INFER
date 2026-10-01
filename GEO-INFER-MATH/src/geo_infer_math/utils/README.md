# GEO-INFER-MATH/src/geo_infer_math/utils

Utils workspace within `GEO-INFER-MATH`.

## Contents

- `__init__.py`
- `caching.py`
- `constants.py`
- `conversion.py`
- `decorators.py`
- `exceptions.py`
- `parallel.py`
- `rng.py`
- `validation.py`

## Public Interface

- `caching.py:cache_result` (function)
- `caching.py:ComputationCache` (class)
- `constants.py:get_constant` (function)
- `constants.py:list_constants` (function)
- `conversion.py:degrees_to_radians` (function)
- `conversion.py:radians_to_degrees` (function)
- `conversion.py:celsius_to_fahrenheit` (function)
- `conversion.py:fahrenheit_to_celsius` (function)
- `conversion.py:kelvin_to_celsius` (function)
- `conversion.py:celsius_to_kelvin` (function)
- `conversion.py:meters_to_feet` (function)
- `conversion.py:feet_to_meters` (function)
- `conversion.py:meters_to_miles` (function)
- `conversion.py:miles_to_meters` (function)
- `conversion.py:meters_to_kilometers` (function)
- `conversion.py:kilometers_to_meters` (function)
- `conversion.py:square_meters_to_square_feet` (function)
- `conversion.py:square_feet_to_square_meters` (function)
- `conversion.py:square_meters_to_acres` (function)
- `conversion.py:acres_to_square_meters` (function)

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
