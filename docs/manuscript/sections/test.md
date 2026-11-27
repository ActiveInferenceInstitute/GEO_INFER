## GEO-INFER-TEST — Unified Testing and Validation

GEO-INFER-TEST is the unified testing framework for quality assurance across all GEO-INFER modules, with automated testing, performance benchmarks, and integration validation. Its module root contains repository validators, metric and coverage tools, and thin execution entrypoints. The owning `geo_infer_test` package supplies shared execution, selection, process, receipt, installed-wheel and analytical-reference behavior.

The public interface in `__init__.py` includes execution components: `GeoInferTestRunner`, `TestConfiguration`, `TestOutcome`, `run_full_system_test`, and `LocalService`.

Shared execution, selection, and receipt handling preserve module isolation while accounting for collected and executed tests. Missing or malformed JUnit reports, empty aggregate lanes, and incomplete execution fail the gate; assertion failures are not automatically retried.

The owning submodule `geo_infer_test.core.performance_monitor` exposes `BenchmarkRunner` and `LoadTester`. Their asynchronous methods await operations through completion; the load tester uses a bounded set of owned workers, records request errors, and drains workers on cancellation. Synchronous methods reject awaitable work rather than timing coroutine construction as successful execution. Benchmark warmup and iterations are explicit, and a timing summary describes the chosen callable and environment rather than a framework-wide performance result.

Profiling helpers retain interpreter and command provenance and distinguish deterministic tracing from sampled stacks. Native Python 3.15 profiler availability is a property of that runtime; optional scientific dependencies must be qualified separately. A successful transport or standard-library workload cannot be generalized to an unexecuted scientific module.

Checked subprocess failures preserve native exception types, captured output and process-ownership evidence. A failed or interrupted profiling attempt can be inspected after receipt and artifact verification even when no profile bytes were produced; its summary remains absent and its failure disposition remains explicit. Missing or corrupt successful profiles still fail validation. These reports diagnose an attempt without rerunning its target or turning a permission denial into a successful measurement.

Under the root README's Module Themes, TEST anchors Infrastructure & Validation with INTRA, LOG, GIT, EXAMPLES, and BIO. Its role is the evidence engine: the manuscript's reproducibility section cites its validator fleet, and every module section's claims ultimately trace to gates TEST defines and enforces.
