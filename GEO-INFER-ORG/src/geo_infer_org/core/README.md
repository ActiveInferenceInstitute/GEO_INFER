# GEO-INFER-ORG/src/geo_infer_org/core

Core workspace within `GEO-INFER-ORG`.

## Contents

- `__init__.py`
- `collaboration.py`
- `governance.py`
- `organization.py`

## Public Interface

- `collaboration.py:CollaborationType` (class)
- `collaboration.py:CollaborationEdge` (class)
- `collaboration.py:TeamMember` (class)
- `collaboration.py:NetworkMetrics` (class)
- `collaboration.py:TeamFormationResult` (class)
- `collaboration.py:CollaborationNetwork` (class)
- `collaboration.py:TeamFormation` (class)
- `governance.py:VotingMethod` (class)
- `governance.py:DecisionStatus` (class)
- `governance.py:Vote` (class)
- `governance.py:Proposal` (class)
- `governance.py:VotingResult` (class)
- `governance.py:VotingEngine` (class)
- `governance.py:ConsensusModel` (class)
- `organization.py:OrgStructureType` (class)
- `organization.py:RoleLevel` (class)
- `organization.py:OrgUnit` (class)
- `organization.py:Role` (class)
- `organization.py:Resource` (class)
- `organization.py:OrgMetrics` (class)

## Module Metadata

- Module: `GEO-INFER-ORG`
- Package: `geo_infer_org`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-org`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ORG`

## Dependencies

- Dependencies are declared in `pyproject.toml` or inherited from the workspace.


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ORG
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
