# GEO-INFER-ENERGY: Energy Systems Module

`GEO-INFER-ENERGY` owns the `geo_infer_energy` package under `GEO-INFER-ENERGY/src/`.
GEO-INFER-ENERGY: Energy Systems Analysis Module.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_energy.RenewableResourceAssessor`
- `geo_infer_energy.RenewableType`
- `geo_infer_energy.SuitabilityClass`
- `geo_infer_energy.RenewableSite`
- `geo_infer_energy.EnergyGridOptimizer`
- `geo_infer_energy.EnergyDemandForecaster`
- `geo_infer_energy.EnergyInfrastructurePlanner`
- `geo_infer_energy.CarbonFootprintAnalyzer`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_energy
from geo_infer_energy import RenewableResourceAssessor, RenewableType, SuitabilityClass, RenewableSite
assert all(value is not None for value in (RenewableResourceAssessor, RenewableType, SuitabilityClass, RenewableSite,))
assert geo_infer_energy.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module ENERGY --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-ENERGY/src/geo_infer_energy/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-ENERGY/README.md)
- [Module operating contract](../../../GEO-INFER-ENERGY/AGENTS.md)
- [Regression: test_energy_integration.py](../../../GEO-INFER-ENERGY/tests/integration/test_energy_integration.py)
- [Regression: test_acceptance_energy.py](../../../GEO-INFER-ENERGY/tests/unit/test_acceptance_energy.py)
- [Regression: test_carbon_footprint.py](../../../GEO-INFER-ENERGY/tests/unit/test_carbon_footprint.py)
- [Example source: __init__.py](../../../GEO-INFER-ENERGY/examples/__init__.py)
- [Example source: basic_energy_analysis.py](../../../GEO-INFER-ENERGY/examples/basic_energy_analysis.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
