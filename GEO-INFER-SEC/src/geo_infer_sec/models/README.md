# GEO-INFER-SEC/src/geo_infer_sec/models

Models workspace within `GEO-INFER-SEC`.

## Contents

- `__init__.py`
- `risk_assessment.py`
- `security_models.py`

## Public Interface

- `risk_assessment.py:RiskSeverity` (class)
- `risk_assessment.py:RiskLikelihood` (class)
- `risk_assessment.py:RiskCategory` (class)
- `risk_assessment.py:GeospatialSecurityRisk` (class)
- `risk_assessment.py:RiskAssessment` (class)
- `risk_assessment.py:create_common_geospatial_risks` (function)
- `security_models.py:ThreatLevel` (class)
- `security_models.py:SecurityEventCategory` (class)
- `security_models.py:SecurityEvent` (class)
- `security_models.py:SecurityAlert` (class)
- `security_models.py:ThreatIntelligence` (class)
- `security_models.py:SecurityAsset` (class)
- `security_models.py:SecurityPolicy` (class)
- `security_models.py:SecurityCompliance` (class)
- `security_models.py:SecurityMetrics` (class)
- `security_models.py:RiskAssessmentRecord` (class)
- `security_models.py:SecurityIncidentWorkflow` (class)
- `security_models.py:SecurityConfiguration` (class)
- `security_models.py:SecurityModelUtils` (class)

## Module Metadata

- Module: `GEO-INFER-SEC`
- Package: `geo_infer_sec`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-sec`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module SEC`

## Dependencies

- `cryptography>=36.0.0`
- `flask>=2.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `pyjwt>=2.3.0`
- `pyyaml>=6.0`
- `shapely>=2.0.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SEC
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
