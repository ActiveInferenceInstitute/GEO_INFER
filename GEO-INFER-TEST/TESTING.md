# GEO-INFER strict testing program

The repository test contract is zero-warning, zero-failure, and zero-skip. The shared root configuration enables strict markers/configuration, importlib discovery, `pytest-asyncio` auto mode, and `-W error`. The runner treats missing tests, collection errors, pytest exit code 5, skips, xfails, and xpasses as failures.

## Canonical gate

Run from the repository root:

```bash
uv sync --all-packages --all-extras --all-groups
uv run python -m compileall GEO-INFER-*/src GEO-INFER-*/examples
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language
uv run python GEO-INFER-TEST/validate_packaging.py --strict
uv run python GEO-INFER-TEST/validate_logging_hygiene.py
uv run python GEO-INFER-TEST/validate_documentation.py --strict
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs
uv run python GEO-INFER-TEST/validate_active_inference_contract.py
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
uv run python GEO-INFER-TEST/validate_test_contracts.py --strict
uv run python GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs
uv run python GEO-INFER-TEST/run_unified_tests.py --category unit
uv run python GEO-INFER-TEST/run_unified_tests.py --category integration
uv run python GEO-INFER-TEST/run_unified_tests.py --category performance
uv run python GEO-INFER-TEST/run_unified_tests.py --h3-migration
uv run python GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible
```

Use `env -u VIRTUAL_ENV` before `uv` when a different shell environment is active. This avoids an environment-selection warning and ensures the workspace `.venv` is used.

## Test taxonomy and fixtures

Every test receives exactly one primary marker based on its canonical directory: `unit`, `integration`, `system`, or `performance`. Markers such as `api`, `core`, `geospatial`, `model`, `reporting`, `reproducibility`, `spatial`, and `artifact` are additive. The unified runner discovers category tests from those canonical directories; a filename containing `performance` does not move a unit test into the performance gate. `geo_infer_test.testing` provides deterministic RNG, local filesystem, localhost HTTP, SQLite, in-process service, finite-value, probability-vector, stochastic-matrix, model-contract, and artifact-manifest helpers.

Pytest test files may use either `test_*.py` or `*_test.py`; inventories, health checks, and runners count and execute both patterns.

Required external behavior is represented by local fixtures. A test must not use `pytest.skip`, `skipif`, `importorskip`, `xfail`, or warning suppression. Missing dependencies and unavailable required backends are explicit failures.

## Model and artifact contracts

Model checks require finite outputs, declared shapes and dtypes, normalized probabilities, valid stochastic matrices, deterministic seeded replay, reset restoration, and explicit invalid-input errors. `validate_model_contracts.py` exercises representative ACT categorical, Gaussian, climate, ecological, urban, resource, and multi-agent models.

`run_model_audit.py` writes `model_contracts.json`, `statistics.json`, `model_audit.png`, and `manifest.json` beneath `.geo-infer-test-results/model-audit/`. The manifest records schema version, finite statistics, byte counts, SHA-256 hashes, and a deterministic manifest hash. The shared `assert_visualization_manifest` helper verifies every sidecar.

## Failure triage

- Contract failure: run `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict` and fix the reported source file.
- Warning or collection failure: run the affected module with `uv run pytest -c pyproject.toml -W error -vv path/to/test.py`.
- Model failure: run `uv run python GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42`.
- Artifact failure: run `uv run python GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible` and inspect the manifest sidecars.
- Category failure: re-run with `--show-failures` to print each failed
  suite's failing test names in the final summary verdict (summary.json
  records them regardless), and inspect the corresponding JUnit XML and
  `summary.json` under `.geo-infer-test-results/`.
