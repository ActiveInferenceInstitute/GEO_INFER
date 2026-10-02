# GEO-INFER-CIV/src/geo_infer_civ/core

Core workspace within `GEO-INFER-CIV`.

## Contents

- `__init__.py`
- `civic_engagement.py`
- `participation.py`
- `policy_analysis.py`

## Public Interface

- `civic_engagement.py:MeetingType` (class)
- `civic_engagement.py:CommentCategory` (class)
- `civic_engagement.py:MeetingRecord` (class)
- `civic_engagement.py:PublicComment` (class)
- `civic_engagement.py:AttendanceTrend` (class)
- `civic_engagement.py:CommentAnalysis` (class)
- `civic_engagement.py:AttendanceTracker` (class)
- `civic_engagement.py:PublicCommentAnalyzer` (class)
- `civic_engagement.py:VoterTurnoutModel` (class)
- `participation.py:ParticipationMethod` (class)
- `participation.py:ParticipantRecord` (class)
- `participation.py:EngagementScore` (class)
- `participation.py:RepresentationReport` (class)
- `participation.py:ParticipationAnalyzer` (class)
- `policy_analysis.py:ImpactLevel` (class)
- `policy_analysis.py:PolicyDomain` (class)
- `policy_analysis.py:CostBenefitItem` (class)
- `policy_analysis.py:StakeholderImpact` (class)
- `policy_analysis.py:CostBenefitResult` (class)
- `policy_analysis.py:EquityScore` (class)

## Module Metadata

- Module: `GEO-INFER-CIV`
- Package: `geo_infer_civ`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-civ`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module CIV`

## Dependencies

- Dependencies are declared in `pyproject.toml` or inherited from the workspace.


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module CIV
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
