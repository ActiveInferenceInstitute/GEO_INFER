# GEO-INFER-NORMS/src/geo_infer_norms/api

Api workspace within `GEO-INFER-NORMS`.

## Contents

- `__init__.py`
- `compliance_api.py`
- `errors.py`
- `legal_api.py`
- `normative_api.py`
- `policy_api.py`
- `zoning_api.py`

## Public Interface

- `compliance_api.py:ComplianceStatusCreate` (class)
- `compliance_api.py:ComplianceMetricCreate` (class)
- `compliance_api.py:EvaluationData` (class)
- `compliance_api.py:GeoPoint` (class)
- `compliance_api.py:ReportParams` (class)
- `compliance_api.py:ComplianceAPI` (class)
- `errors.py:ErrorHandlerMiddleware` (class)
- `errors.py:register_error_handlers` (function)
- `legal_api.py:GeometryModel` (class)
- `legal_api.py:JurisdictionCreate` (class)
- `legal_api.py:RegulationCreate` (class)
- `legal_api.py:RegulatoryFrameworkCreate` (class)
- `legal_api.py:PointLocation` (class)
- `legal_api.py:LegalAPI` (class)
- `normative_api.py:GeometryModel` (class)
- `normative_api.py:SocialNormCreate` (class)
- `normative_api.py:NormDiffusionRequest` (class)
- `normative_api.py:NormativeInferenceRequest` (class)
- `normative_api.py:NormPolicyImpactRequest` (class)
- `normative_api.py:PointLocation` (class)

## Module Metadata

- Module: `GEO-INFER-NORMS`
- Package: `geo_infer_norms`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-norms`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module NORMS`

## Dependencies

- `fastapi>=0.95.0,<1`
- `starlette>=0.27.0`
- `geopandas>=0.13.0,<2`
- `matplotlib>=3.7.0,<4`
- `networkx>=2.6.0,<4`
- `numpy>=1.24.0,<3`
- `pandas>=2.0.0,<3`
- `pydantic>=2.0.0,<3`
- `shapely>=2.0.0,<3`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module NORMS
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
