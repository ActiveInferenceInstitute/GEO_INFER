# Agent Instructions: GEO-INFER-TEST

## Scope

- Owning module: `GEO-INFER-TEST`
- Python package: `geo_infer_test`
- Directory role: Unified testing framework for quality assurance across all GEO-INFER modules with automated testing, performance benchmarks, and integration validation.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_test` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

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
- `validate_doc_imports.py`
- `validate_documentation.py`
- `validate_gnn_interchange.py`
- `validate_h3_active_inference_contract.py`
- `validate_logging_hygiene.py`
- `validate_model_contracts.py`
- `validate_packaging.py`
- `validate_repo_contracts.py`
- `validate_skills.py`
- `validate_test_contracts.py`
- `water_surface_metric.py`
- `.gitignore`
- `SKILL.md`
- `TESTING.md`
- `coverage_baseline.json`
- `pyproject.toml`

## Validation

```bash
uv sync --all-packages --all-extras --all-groups
uv run python GEO-INFER-TEST/run_unified_tests.py --module TEST
```


## Strict Testing Contracts

- Reuse `geo_infer_test.testing` fixtures and assertions for local boundaries,
  model contracts, and visualization artifacts.
- Missing dependencies, unavailable backends, warnings, skips, xfails, and
  empty selections are failures; do not hide them with warning filters or
  conditional pytest controls.
- Keep `validate_test_contracts.py`, `validate_model_contracts.py`, and
  `run_model_audit.py` synchronized with the documented commands and output
  schemas.
- Keep `validate_documentation.py` synchronized with the maintained
  authoritative documentation paths when the hub moves.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
