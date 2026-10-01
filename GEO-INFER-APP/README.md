# GEO-INFER-APP

Human-computer interaction layer providing accessible geospatial applications, dashboards, and UI components.

## Contents

- `docs/`
- `examples/`
- `src/`
- `tests/`
- `SKILL.md`
- `pyproject.toml`

## Public Interface

- No public Python symbols are defined directly in this directory.

## Module Metadata

- Module: `GEO-INFER-APP`
- Package: `geo_infer_app`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-APP`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module APP`

## Dependencies

- Dependencies are declared in `pyproject.toml` or inherited from the workspace.


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module APP
```


## Visualization Contracts

- Agent map features validate finite longitude/latitude values and geographic
  bounds, and normalize metadata to JSON-safe values.
- Active-inference prediction and reinforcement-learning reward series are
  exposed in dashboard widgets when present in agent metadata.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
