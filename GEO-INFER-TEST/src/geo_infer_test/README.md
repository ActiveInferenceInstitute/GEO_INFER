# GEO-INFER-TEST/src/geo_infer_test

Geo Infer Test workspace within `GEO-INFER-TEST`.

## Contents

- `core/`
- `models/`
- `__init__.py`
- `act_research_oracles.py`
- `coverage.py`
- `doc_examples.py`
- `execution.py`
- `process.py`
- `selection.py`
- `testing.py`
- `wheel_evidence.py`
- `wheel_profiles.py`

## Public Interface

- `act_research_oracles.py:research_profile_reference` (function)
- `act_research_oracles.py:research_posterior_reference` (function)
- `act_research_oracles.py:research_policy_reference` (function)
- `act_research_oracles.py:assert_research_policy` (function)
- `coverage.py:measure_module` (function)
- `coverage.py:junit_failure_names` (function)
- `coverage.py:junit_failure_details` (function)
- `coverage.py:main` (function)
- `doc_examples.py:load_manifest` (function)
- `doc_examples.py:example_code` (function)
- `doc_examples.py:verify_page` (function)
- `doc_examples.py:main` (function)
- `execution.py:CommandResult` (class)
- `execution.py:Module` (class)
- `execution.py:SuiteReport` (class)
- `execution.py:discover_geo_infer_modules` (function)
- `execution.py:discover_workspace_test_targets` (function)
- `execution.py:profile_selection_args` (function)
- `execution.py:ensure_results_dir` (function)
- `execution.py:run_results_dir` (function)

## Module Metadata

- Module: `GEO-INFER-TEST`
- Package: `geo_infer_test`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-test`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module TEST`

## Dependencies

- `coverage[toml]>=7.0.0`
- `geopandas>=0.13.0`
- `h3>=4.5.0,<5`
- `hypothesis>=6.0.0`
- `matplotlib>=3.5.0`
- `numpy>=1.20.0`
- `pandas>=1.3.0`
- `psutil>=5.9.0`
- `pytest>=7.0.0`
- `pytest-benchmark>=4.0.0`
- `pytest-cov>=4.0.0`
- `pytest-html>=3.1.0`
- `pytest-mock>=3.10.0`
- `pytest-timeout>=2.1.0`
- `pytest-xdist>=3.0.0`
- `pyyaml>=6.0`


## Validation

```bash
uv sync --all-packages --all-extras --all-groups
uv run python GEO-INFER-TEST/run_unified_tests.py --module TEST
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
