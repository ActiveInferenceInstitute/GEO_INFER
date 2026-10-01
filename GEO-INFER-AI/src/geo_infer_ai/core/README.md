# GEO-INFER-AI/src/geo_infer_ai/core

Core workspace within `GEO-INFER-AI`.

## Contents

- `__init__.py`
- `explainability.py`
- `model_evaluation.py`
- `secure_serialization.py`
- `training.py`

## Public Interface

- `explainability.py:ModelExplainer` (class)
- `model_evaluation.py:GeospatialModelEvaluator` (class)
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
- `secure_serialization.py:loads_signed` (function)
- `secure_serialization.py:dumps_signed_text` (function)
- `secure_serialization.py:loads_signed_text` (function)

## Module Metadata

- Module: `GEO-INFER-AI`
- Package: `geo_infer_ai`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-ai`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module AI`

## Dependencies

- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `h3>=4.5.0,<5`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module AI
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
