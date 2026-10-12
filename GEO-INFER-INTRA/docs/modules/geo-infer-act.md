# GEO-INFER-ACT: Active Inference Engine

`GEO-INFER-ACT` owns the `geo_infer_act` package under `GEO-INFER-ACT/src/`.
GEO-INFER-ACT: Active Inference modeling module for GEO-INFER framework.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_act.ActiveInferenceModel`
- `geo_infer_act.ActiveInferenceStepResult`
- `geo_infer_act.FreeEnergyBreakdown`
- `geo_infer_act.H3BeliefUpdateResult`
- `geo_infer_act.H3CellDiagnostics`
- `geo_infer_act.H3EdgeDiagnostics`
- `geo_infer_act.H3GridInferenceResult`
- `geo_infer_act.H3LevelDiagnostics`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_act
from geo_infer_act import ActiveInferenceModel, ActiveInferenceStepResult, FreeEnergyBreakdown, H3BeliefUpdateResult
assert all(value is not None for value in (ActiveInferenceModel, ActiveInferenceStepResult, FreeEnergyBreakdown, H3BeliefUpdateResult,))
assert geo_infer_act.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module ACT --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-ACT/src/geo_infer_act/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-ACT/README.md)
- [Module operating contract](../../../GEO-INFER-ACT/AGENTS.md)
- [Regression: test_active_sensing_trajectories.py](../../../GEO-INFER-ACT/tests/integration/test_active_sensing_trajectories.py)
- [Regression: test_h3_example_smoke.py](../../../GEO-INFER-ACT/tests/integration/test_h3_example_smoke.py)
- [Regression: test_integration.py](../../../GEO-INFER-ACT/tests/integration/test_integration.py)
- [Example source: __init__.py](../../../GEO-INFER-ACT/examples/__init__.py)
- [Example source: ecological_model.py](../../../GEO-INFER-ACT/examples/ecological_model.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
