# Getting Started with GEO-INFER-TEST

GEO-INFER-TEST is installed as part of the shared workspace. Run commands from
the repository root with `uv run`.

## Setup

```bash
uv sync --all-packages --all-extras --all-groups
uv run python -c "import geo_infer_test; print(geo_infer_test.__version__)"
```

## Essential commands

```bash
# discover module names
uv run python GEO-INFER-TEST/run_unified_tests.py --list-modules

# focus one module
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE

# run release categories
uv run python GEO-INFER-TEST/run_unified_tests.py --category unit
uv run python GEO-INFER-TEST/run_unified_tests.py --category integration
uv run python GEO-INFER-TEST/run_unified_tests.py --category performance

# run H3/Active Inference contracts
uv run python GEO-INFER-TEST/run_unified_tests.py --h3-migration
```

The `--module` value is the uppercase suffix without `GEO-INFER-`.
The runner writes summaries and JUnit reports below
`.geo-infer-test-results/`.

## Direct pytest

Use direct pytest when narrowing a failure:

```bash
uv run python -m pytest -c pyproject.toml -W error \
  GEO-INFER-SPACE/tests/unit -q
uv run python -m pytest -c pyproject.toml -W error \
  GEO-INFER-ACT/tests/unit/test_h3_active_inference.py -q
```

The repository uses strict markers and treats warnings as errors. A test run
that collects no tests, skips, xfails, or relies on an unavailable dependency
does not satisfy the release contract.

## Programmatic runner

```python
from geo_infer_test import GeoInferTestRunner, TestConfiguration

config = TestConfiguration(
    modules_to_test=["SPACE"],
    test_types=["unit"],
    parallel_execution=False,
)
runner = GeoInferTestRunner(config)
print(runner.discover_tests())
print(runner.run_all_tests())
```

For reusable assertions use the package exports such as `assert_finite`,
`assert_probability`, `assert_seed_replay`, and
`assert_visualization_manifest`.

## Contract bundle

```bash
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs
uv run python GEO-INFER-TEST/validate_test_contracts.py --strict
uv run python GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42
uv run python GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible
```

For the full explanation of categories, strict policy, artifacts, and triage,
read the [GEO-INFER-TEST module guide](index.md) and the repository
[testing guide](../../GEO-INFER-INTRA/docs/developer_guide/testing_guide.md).
