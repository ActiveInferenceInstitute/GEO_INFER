#!/usr/bin/env bash
# Benchmark harness for the ROOT-01 CI render-lane workstream.
#
# Workload (deterministic, offline):
#   1. collect the root manuscript test battery from tests/;
#   2. parse every pytest selection in the tracked GitHub Actions
#      workflows and measure, as the union of collected test ids, how
#      many root tests CI actually runs (the ROOT-01 acceptance line);
#   3. run the repository's own render path when present
#      (scripts/render_manuscript_pdf.py) and execute the 7
#      render-dependent tests against its artifacts.
# 4. measure the HYG-04 tests-suite dead-import surface
#    (ruff F401/F841/F811 over GEO-INFER-*/tests);
# 5. measure the SEC-02 unresolved secret-scan findings
#    (gitleaks over the full git history, committed policy);
# 6. measure the DOCS-03 stale assessment artifacts
#    (tracked assessment files missing their historical-artifact banner);
# 7. measure the TEST-03 coverage-baseline completeness
#    (modules lacking an entry in coverage_baseline.json);
# 8. measure the EXAMPLES-01 orchestrator coverage
#    (ledger-named modules lacking scripts/run_orchestrator.py);
# 9. measure the DOCS-01 preview-verification acceptance
#    (open checks in the committed receipt + recomputed artifact hashes).
#
# Metrics are printed as "METRIC name=value" lines; diagnostics as
# "ASI key=value" lines.  Prerequisite: the shared uv workspace is synced
# (uv sync --all-packages --all-extras --all-groups); the harness itself never touches
# the network (uv run --no-sync).
 set -euo pipefail
 cd "$(dirname "$0")"
uv run --no-sync python GEO-INFER-TEST/render_lane_metric.py "$@"
uv run --no-sync python GEO-INFER-TEST/tests_lint_metric.py "$@"
uv run --no-sync python GEO-INFER-TEST/secret_scan_metric.py "$@"
uv run --no-sync python GEO-INFER-TEST/stale_assessment_metric.py "$@"
uv run --no-sync python GEO-INFER-TEST/coverage_baseline_metric.py "$@"
uv run --no-sync python GEO-INFER-TEST/orchestrator_coverage_metric.py "$@"
uv run --no-sync python GEO-INFER-TEST/preview_receipt_metric.py "$@"
uv run --no-sync python GEO-INFER-TEST/water_surface_metric.py "$@"
