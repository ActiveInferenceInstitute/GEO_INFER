## GEO-INFER-TEST — Unified Testing and Validation

GEO-INFER-TEST is the unified testing framework for quality assurance across all GEO-INFER modules, with automated testing, performance benchmarks, and integration validation. Its module root contains repository validators, metric and coverage tools, and thin execution entrypoints. The owning `geo_infer_test` package supplies shared execution, selection, process, receipt, installed-wheel and analytical-reference behavior.

The public interface in `__init__.py` includes execution components: `GeoInferTestRunner`, `TestConfiguration`, `TestOutcome`, `run_full_system_test`, and `LocalService`.

Shared execution, selection, and receipt handling preserve module isolation while accounting for collected and executed tests. Missing or malformed JUnit reports, empty aggregate lanes, and incomplete execution fail the gate; assertion failures are not automatically retried.

Under the root README's Module Themes, TEST anchors Infrastructure & Validation with INTRA, LOG, GIT, EXAMPLES, and BIO. Its role is the evidence engine: the manuscript's reproducibility section cites its validator fleet, and every module section's claims ultimately trace to gates TEST defines and enforces.
