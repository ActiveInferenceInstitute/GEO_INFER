# GEO-INFER-TEST

Unified testing framework for quality assurance across all GEO-INFER modules with automated testing, performance benchmarks, and integration validation.

## Contents

- `config/`
- `demo/`
- `docs/`
- `examples/`
- `src/`
- `tests/`
- `tools/`
- `_validator_common.py`
- `build_package_wheels.py`
- `check_coverage_floor.py`
- `coverage_baseline_metric.py`
- `import_probe.py`
- `measure_module_coverage.py`
- `orchestrator_coverage_metric.py`
- `preview_receipt_metric.py`
- `record_validation.py`
- `render_lane_metric.py`
- `rewrite_readme_agents.py`
- `run_model_audit.py`
- `run_unified_tests.py`
- `secret_scan_metric.py`
- `stale_assessment_metric.py`
- `tests_lint_metric.py`
- `validate_act_geospatial_contract.py`
- `validate_act_script_orchestration.py`
- `validate_active_inference_contract.py`
- `validate_doc_examples.py`
- `validate_doc_imports.py`
- `validate_documentation.py`
- `validate_gnn_interchange.py`
- `validate_h3_active_inference_contract.py`
- `validate_logging_hygiene.py`
- `validate_model_contracts.py`
- `validate_packaging.py`
- `validate_repo_contracts.py`
- `validate_research_evidence.py`
- `validate_skills.py`
- `validate_test_contracts.py`
- `validate_wheel_receipt.py`
- `water_surface_metric.py`
- `.gitignore`
- `SKILL.md`
- `TESTING.md`
- `coverage_baseline.json`
- `doc_examples.json`
- `pyproject.toml`

## Public Interface

- `build_package_wheels.py:BuildResult` (class)
- `build_package_wheels.py:BuildSummary` (class)
- `build_package_wheels.py:validate_wheel_contents` (function)
- `build_package_wheels.py:build_wheel` (function)
- `build_package_wheels.py:verify_wheels` (function)
- `build_package_wheels.py:verify_profiles` (function)
- `build_package_wheels.py:install_and_verify` (function)
- `build_package_wheels.py:main` (function)
- `check_coverage_floor.py:main` (function)
- `coverage_baseline_metric.py:main` (function)
- `import_probe.py:run_import_probe` (function)
- `orchestrator_coverage_metric.py:main` (function)
- `preview_receipt_metric.py:main` (function)
- `rewrite_readme_agents.py:ModuleInfo` (class)
- `rewrite_readme_agents.py:git_ls_files` (function)
- `rewrite_readme_agents.py:tracked_files` (function)
- `rewrite_readme_agents.py:requirement_lines` (function)
- `rewrite_readme_agents.py:declared_dependencies` (function)
- `rewrite_readme_agents.py:discover_modules` (function)
- `rewrite_readme_agents.py:module_for` (function)

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


## Strict Testing Contracts

- `src/geo_infer_test/testing.py` exports deterministic RNG, local filesystem,
  HTTP, SQLite, and service fixtures plus finite/probability/matrix/model and
  visualization-manifest assertions.
- `validate_test_contracts.py --strict` validates every module inventory,
  primary marker, forbidden pytest control, syntax tree, and behavior-test
  docstring.
- `validate_documentation.py --strict` validates the maintained documentation
  hub's relative links and rejects known stale current-state claims.
- `validate_model_contracts.py` checks representative ACT model contracts;
  `run_model_audit.py` emits finite statistics, a PNG visualization, SHA-256
  sidecars, and a deterministic manifest.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
