# GEO-INFER-SEC/src/geo_infer_sec/core

Core workspace within `GEO-INFER-SEC`.

## Contents

- `__init__.py`
- `access_control.py`
- `anonymization.py`
- `audit.py`
- `authentication.py`
- `authorization.py`
- `cognitive_security.py`
- `compliance.py`
- `digital_security.py`
- `encryption.py`
- `integrated_security.py`
- `physical_security.py`

## Public Interface

- `access_control.py:SpatialPermission` (class)
- `access_control.py:Role` (class)
- `access_control.py:GeospatialAccessManager` (class)
- `anonymization.py:GeospatialAnonymizer` (class)
- `audit.py:AuditEventType` (class)
- `audit.py:AuditEventSeverity` (class)
- `audit.py:AuditEvent` (class)
- `audit.py:AuditLogger` (class)
- `authentication.py:generate_totp` (function)
- `authentication.py:verify_totp` (function)
- `authentication.py:UserCredentials` (class)
- `authentication.py:TokenInfo` (class)
- `authentication.py:AuthenticationManager` (class)
- `authorization.py:PermissionType` (class)
- `authorization.py:AuthorizationManager` (class)
- `cognitive_security.py:BehaviorType` (class)
- `cognitive_security.py:ThreatHuntingType` (class)
- `cognitive_security.py:LearningMode` (class)
- `cognitive_security.py:BehaviorProfile` (class)
- `cognitive_security.py:CognitiveThreat` (class)

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
