# GEO-INFER-INTRA

Reproducible documentation previews and configuration utilities for the GEO-INFER ecosystem.

## Contents

- `assessment_results/`
- `config/`
- `docs/`
- `examples/`
- `scripts/`
- `src/`
- `templates/`
- `tests/`
- `assess_repository.py`
- `SKILL.md`
- `pyproject.toml`
- `requirements-dev.txt`

## Public Interface

- `assess_repository.py:RepositoryAssessment` (class)
- `assess_repository.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-INTRA`
- Package: `geo_infer_intra`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-INTRA`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA`

## Dependencies

- `h3>=4.5.0,<5`
- `jsonschema>=4.0.0`
- `pillow>=10.0.0`
- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module INTRA
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
