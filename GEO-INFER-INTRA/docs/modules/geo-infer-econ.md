# GEO-INFER-ECON: Economic Analysis

`GEO-INFER-ECON` owns the `geo_infer_econ` package under `GEO-INFER-ECON/src/`.
GEO-INFER-ECON: Spatial Economic Modeling, Analysis, and Policy Evaluation

## Public import surface

These names are exported by the current owning package:

- `geo_infer_econ.EconomicModelingEngine`
- `geo_infer_econ.SpatialEconometricsEngine`
- `geo_infer_econ.PolicyAnalysisEngine`
- `geo_infer_econ.ConsumerChoiceModels`
- `geo_infer_econ.ProducerTheoryModels`
- `geo_infer_econ.MarketStructureAnalysis`
- `geo_infer_econ.GameTheoryModels`
- `geo_infer_econ.BehavioralEconomicsEngine`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_econ
from geo_infer_econ import EconomicModelingEngine, SpatialEconometricsEngine, PolicyAnalysisEngine, ConsumerChoiceModels
assert all(value is not None for value in (EconomicModelingEngine, SpatialEconometricsEngine, PolicyAnalysisEngine, ConsumerChoiceModels,))
assert geo_infer_econ.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module ECON --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-ECON/src/geo_infer_econ/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-ECON/README.md)
- [Module operating contract](../../../GEO-INFER-ECON/AGENTS.md)
- [Regression: test_integrations.py](../../../GEO-INFER-ECON/tests/integration/test_integrations.py)
- [Regression: test_behavioral_economics.py](../../../GEO-INFER-ECON/tests/unit/test_behavioral_economics.py)
- [Regression: test_bioregional_economics.py](../../../GEO-INFER-ECON/tests/unit/test_bioregional_economics.py)
- [Example source: comprehensive_economic_analysis.py](../../../GEO-INFER-ECON/examples/comprehensive_economic_analysis.py)
- [Example source: integration_example.py](../../../GEO-INFER-ECON/examples/integration_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
