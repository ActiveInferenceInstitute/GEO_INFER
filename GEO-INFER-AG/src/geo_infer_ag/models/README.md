# GEO-INFER-AG/src/geo_infer_ag/models

Models workspace within `GEO-INFER-AG`.

## Contents

- `__init__.py`
- `base.py`
- `carbon_sequestration.py`
- `crop_yield.py`
- `secure_serialization.py`
- `soil_health.py`
- `water_usage.py`

## Public Interface

- `base.py:write_signed_payload` (function)
- `base.py:read_verified_payload` (function)
- `base.py:AgricultureModel` (class)
- `carbon_sequestration.py:CarbonSequestrationModel` (class)
- `crop_yield.py:CropYieldModel` (class)
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
- `secure_serialization.py:sign_payload` (function)
- `secure_serialization.py:verify_payload` (function)
- `secure_serialization.py:sign_payload_text` (function)
- `secure_serialization.py:verify_payload_text` (function)
- `secure_serialization.py:dumps_signed` (function)

## Module Metadata

- Module: `GEO-INFER-AG`
- Package: `geo_infer_ag`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-ag`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module AG`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `geopandas>=0.13.0`
- `shapely>=2.0.0`
- `scikit-learn>=1.0.0`
- `rasterio>=1.2.0`
- `pyproj>=3.0.0`
- `matplotlib>=3.3.0`
- `scipy>=1.6.0`
- `xarray>=0.18.0`
- `joblib>=1.0.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module AG
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
