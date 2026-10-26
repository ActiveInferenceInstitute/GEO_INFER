# GEO-INFER-TEST/src/geo_infer_test

Geo Infer Test workspace within `GEO-INFER-TEST`.

## Contents

- `core/`
- `models/`
- `__init__.py`
- `_template_worker.py`
- `act_research_oracles.py`
- `companion_releases.py`
- `coverage.py`
- `dependency_audit.py`
- `doc_examples.py`
- `documentation_contracts.py`
- `execution.py`
- `manuscript_rendering.py`
- `process.py`
- `profiling.py`
- `research_evidence.py`
- `selection.py`
- `testing.py`
- `torch_variant.py`
- `wheel_evidence.py`
- `wheel_profiles.py`
- `torch_cu126_variant.json`

## Public Interface

- `act_research_oracles.py:research_profile_reference` (function)
- `act_research_oracles.py:research_posterior_reference` (function)
- `act_research_oracles.py:research_policy_reference` (function)
- `act_research_oracles.py:assert_research_policy` (function)
- `companion_releases.py:validate_companion_releases` (function)
- `coverage.py:measure_module` (function)
- `coverage.py:junit_failure_names` (function)
- `coverage.py:junit_failure_details` (function)
- `coverage.py:run_coverage_measurements` (function)
- `coverage.py:main` (function)
- `dependency_audit.py:canonical_name` (function)
- `dependency_audit.py:advisory_inventory` (function)
- `dependency_audit.py:locked_inventory` (function)
- `dependency_audit.py:requirement_batches` (function)
- `dependency_audit.py:validate_response` (function)
- `dependency_audit.py:audit` (function)
- `dependency_audit.py:main` (function)
- `doc_examples.py:load_manifest` (function)
- `doc_examples.py:example_code` (function)
- `doc_examples.py:verify_page` (function)

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
