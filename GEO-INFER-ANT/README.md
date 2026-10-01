# GEO-INFER-ANT

Comprehensive swarm intelligence and complex adaptive systems modeling using Active Inference principles for emergent collective behavior in geospatial contexts.

## Contents

- `config/`
- `docs/`
- `examples/`
- `src/`
- `tests/`
- `run_tests.py`
- `IMPLEMENTATION_STATUS.md`
- `SKILL.md`
- `pyproject.toml`

## Public Interface

- `run_tests.py:run_command` (function)
- `run_tests.py:run_unit_tests` (function)
- `run_tests.py:run_integration_tests` (function)
- `run_tests.py:run_performance_tests` (function)
- `run_tests.py:run_examples` (function)
- `run_tests.py:run_coverage_analysis` (function)
- `run_tests.py:run_quick_tests` (function)
- `run_tests.py:generate_test_report` (function)
- `run_tests.py:print_final_summary` (function)
- `run_tests.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-ANT`
- Package: `geo_infer_ant`
- Version: `0.3.0`
- Install: `uv pip install -e ./GEO-INFER-ANT`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module ANT`

## Dependencies

- `jsonschema>=4.0.0`
- `networkx>=2.8`
- `numpy>=1.21.0`
- `pyyaml>=6.0`
- `scikit-learn>=1.1.0`
- `scipy>=1.7.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ANT
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
