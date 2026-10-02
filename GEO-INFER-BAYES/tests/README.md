# GEO-INFER-BAYES/tests

Tests workspace within `GEO-INFER-BAYES`.

## Contents

- `integration/`
- `unit/`
- `conftest.py`

## Public Interface

- `conftest.py:sample_coordinates` (function)
- `conftest.py:sample_geodataframe` (function)
- `conftest.py:tmp_output_dir` (function)
- `conftest.py:gp_kernel_config` (function)
- `conftest.py:mcmc_samples` (function)
- `conftest.py:prior_params` (function)
- `conftest.py:synthetic_spatial_data` (function)

## Module Metadata

- Module: `GEO-INFER-BAYES`
- Package: `geo_infer_bayes`
- Version: `0.4.0`
- Install: `uv sync --package geo-infer-bayes`
- Tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --module BAYES`

## Dependencies

- `arviz>=0.12.0`
- `matplotlib>=3.5.0`
- `numpy>=1.20.0,<2.0`
- `pandas>=1.3.0`
- `scipy>=1.7.0`
- `tqdm>=4.60.0`
- `xarray>=2022.3.0`


## Strict Test Inventory

- Purpose: validate the `GEO-INFER-BAYES` module's current behavior through unit,
  integration, system, and performance test surfaces.
- Primary marker: tests receive exactly one primary marker from their canonical
  directory; additive domain markers remain allowed.
- Required fixtures: local `tests/conftest.py` fixtures and shared
  `geo_infer_test.testing` fixtures for deterministic RNG, filesystem, HTTP,
  SQLite, service, model, and artifact boundaries.
- Dependencies: required test/runtime dependencies are installed by
  `uv sync --all-packages --all-extras --all-groups`; missing backends are failures.
- Expected artifacts: JUnit XML under `.geo-infer-test-results/`; model and
  visualization outputs require finite statistics, sidecars, hashes, and a
  manifest.
- Failure triage: `env -u VIRTUAL_ENV uv run pytest -c pyproject.toml -q
  GEO-INFER-BAYES/tests`, followed by
  `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`.

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module BAYES
```


## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
