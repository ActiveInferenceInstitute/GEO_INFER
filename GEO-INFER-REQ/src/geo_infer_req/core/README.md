# GEO-INFER-REQ/src/geo_infer_req/core

Core workspace within `GEO-INFER-REQ`.

## Contents

- `__init__.py`
- `requirements.py`
- `traceability.py`
- `validation.py`

## Public Interface

- `requirements.py:RequirementType` (class)
- `requirements.py:RequirementStatus` (class)
- `requirements.py:PriorityLevel` (class)
- `requirements.py:Requirement` (class)
- `requirements.py:DependencyGraph` (class)
- `requirements.py:CompletenessReport` (class)
- `requirements.py:RequirementsAnalyzer` (class)
- `traceability.py:ArtifactType` (class)
- `traceability.py:TraceLink` (class)
- `traceability.py:TraceMatrixEntry` (class)
- `traceability.py:CoverageReport` (class)
- `traceability.py:ImpactReport` (class)
- `traceability.py:TraceabilityManager` (class)
- `validation.py:ConflictType` (class)
- `validation.py:ValidationSeverity` (class)
- `validation.py:ValidationIssue` (class)
- `validation.py:ConflictDetectionResult` (class)
- `validation.py:ConsistencyReport` (class)
- `validation.py:FeasibilityAssessment` (class)
- `validation.py:RequirementSpec` (class)

## Module Metadata

- Module: `GEO-INFER-REQ`
- Package: `geo_infer_req`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-req`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module REQ`

## Dependencies

- Dependencies are declared in `pyproject.toml` or inherited from the workspace.


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module REQ
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
