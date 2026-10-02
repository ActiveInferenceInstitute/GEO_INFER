# GEO-INFER-GIT: Git & Orchestration

`GEO-INFER-GIT` owns the `geo_infer_git` package under `GEO-INFER-GIT/src/`.
GEO-INFER-GIT - Git repository cloning tools for GEO-INFER framework.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_git` is the installed package namespace; inspect its owning source modules for behavior APIs.

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_git
assert geo_infer_git.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module GIT --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-GIT/src/geo_infer_git/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-GIT/README.md)
- [Module operating contract](../../../GEO-INFER-GIT/AGENTS.md)
- [Regression: test_report_workflow.py](../../../GEO-INFER-GIT/tests/integration/test_report_workflow.py)
- [Regression: test_advanced_git.py](../../../GEO-INFER-GIT/tests/unit/test_advanced_git.py)
- [Regression: test_cli.py](../../../GEO-INFER-GIT/tests/unit/test_cli.py)
- [Example source: integration_with_ai.py](../../../GEO-INFER-GIT/examples/integration_with_ai.py)
- [Example source: integration_with_data.py](../../../GEO-INFER-GIT/examples/integration_with_data.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
