## GEO-INFER-TEST — Unified Testing and Validation

GEO-INFER-TEST is the unified testing framework for quality assurance across all GEO-INFER modules, with automated testing, performance benchmarks, and integration validation (per its `README.md`). Uniquely, its module root is itself a toolchest: some thirty `validate_*.py` and `*_metric.py` scripts (for example `validate_repo_contracts.py`, `validate_documentation.py`, `validate_model_contracts.py`, `coverage_baseline.json`, `check_coverage_floor.py`), alongside `run_unified_tests.py`, `tools/`, `demo/`, and `TESTING.md`. The package `geo_infer_test` adds `core/`, `models/`, and `testing.py` at roughly 3,629 lines of Python.

The public interface, verified from `__init__.py`, exports twenty-two symbols. Execution: `GeoInferTestRunner`, `TestConfiguration`, `TestOutcome`, `run_full_system_test`, and `LocalService`.

The test census counts 46 test files. Shared execution, selection, and receipt handling preserve module isolation while accounting for collected and executed tests. Missing or malformed JUnit reports, empty aggregate lanes, and incomplete execution fail the gate; assertion failures are not automatically retried.

Under the root README's Module Themes, TEST anchors Infrastructure & Validation with INTRA, LOG, GIT, EXAMPLES, and BIO. Its role is the evidence engine: the manuscript's reproducibility section cites its validator fleet, and every module section's claims ultimately trace to gates TEST defines and enforces.
