# GEO-INFER-COG/src/geo_infer_cog/decision

Decision workspace within `GEO-INFER-COG`.

## Contents

- `__init__.py`
- `support.py`

## Public Interface

- `support.py:DecisionStrategy` (class)
- `support.py:DecisionAlternative` (class)
- `support.py:DecisionRecommendation` (class)
- `support.py:SpatialDecisionSupport` (class)

## Module Metadata

- Module: `GEO-INFER-COG`
- Package: `geo_infer_cog`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-cog`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module COG`

## Dependencies

- `numpy>=1.20.0`
- `networkx>=2.6`
- `pyyaml>=5.4`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module COG
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
