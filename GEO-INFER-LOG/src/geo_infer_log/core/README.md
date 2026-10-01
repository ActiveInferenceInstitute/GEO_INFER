# GEO-INFER-LOG/src/geo_infer_log/core

Core workspace within `GEO-INFER-LOG`.

## Contents

- `__init__.py`
- `delivery.py`
- `routing.py`
- `secure_serialization.py`
- `supply_chain.py`
- `transport.py`

## Public Interface

- `delivery.py:LastMileRouter` (class)
- `delivery.py:DeliveryScheduler` (class)
- `delivery.py:ServiceAreaAnalyzer` (class)
- `routing.py:save_gpickle` (function)
- `routing.py:RouteOptimizer` (class)
- `routing.py:FleetManager` (class)
- `routing.py:VehicleRouter` (class)
- `routing.py:TravelTimeEstimator` (class)
- `routing.py:MultiObjectiveOptimizer` (class)
- `routing.py:RealTimeTracker` (class)
- `secure_serialization.py:PayloadSecurityError` (class)
- `secure_serialization.py:SigningKeyUnavailableError` (class)
- `secure_serialization.py:MalformedEnvelopeError` (class)
- `secure_serialization.py:UnsignedPayloadError` (class)
- `secure_serialization.py:SignatureMismatchError` (class)
- `secure_serialization.py:clear_signing_key_cache` (function)
- `secure_serialization.py:default_key_path` (function)
- `secure_serialization.py:resolve_signing_key` (function)
- `secure_serialization.py:derive_context_key` (function)
- `secure_serialization.py:is_signed_envelope` (function)

## Module Metadata

- Module: `GEO-INFER-LOG`
- Package: `geo_infer_log`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-log`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module LOG`

## Dependencies

- `numpy>=1.20.0`
- `starlette>=0.27.0`
- `pandas>=1.3.0`
- `geopandas>=0.13.0`
- `networkx>=2.6.0`
- `pulp>=2.7.0,<3`
- `shapely>=2.0.0`
- `pydantic>=2.0.0`
- `fastapi>=0.100.0`
- `scipy>=1.9.0`
- `matplotlib>=3.5.0`
- `folium>=0.14.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module LOG
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
