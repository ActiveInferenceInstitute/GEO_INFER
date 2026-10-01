# GEO-INFER-COG

Human-centered geospatial tools that model perception, reasoning, and spatial cognition for intuitive interfaces.

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

- Module: `GEO-INFER-COG`
- Package: `geo_infer_cog`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-COG`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module COG`

## Dependencies

- `numpy>=1.20.0`
- `networkx>=2.6`
- `pyyaml>=5.4`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module COG
```


## Visualization Contracts

- Human-centered visualization IDs are deterministic per visualizer instance,
  and proximity thresholds and color counts are validated at construction.
- Proximity grouping uses connected components and similarity grouping returns
  explicit geometry groups with confidence metadata.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
