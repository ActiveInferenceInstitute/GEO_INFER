# GEO-INFER-EXAMPLES/scripts

Scripts workspace within `GEO-INFER-EXAMPLES`.

## Contents

- `comprehensive_assessment.py`
- `run_all_examples.py`

## Public Interface

- `comprehensive_assessment.py:discover_examples` (function)
- `comprehensive_assessment.py:assess_example` (function)
- `comprehensive_assessment.py:build_report` (function)
- `comprehensive_assessment.py:markdown` (function)
- `comprehensive_assessment.py:save_report` (function)
- `comprehensive_assessment.py:main` (function)
- `run_all_examples.py:ExampleResult` (class)
- `run_all_examples.py:AssessmentReport` (class)
- `run_all_examples.py:IntegrationExampleRunner` (class)
- `run_all_examples.py:main` (function)

## Module Metadata

- Module: `GEO-INFER-EXAMPLES`
- Package: `geo_infer_examples`
- Version: `0.3.0`
- Install: `uv sync --package geo-infer-examples`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES`

## Dependencies

- `pyyaml>=6.0`


## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module EXAMPLES
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
