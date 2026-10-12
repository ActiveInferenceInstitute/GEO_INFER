# GEO-INFER-IOT: Internet of Things Integration

`GEO-INFER-IOT` owns the `geo_infer_iot` package under `GEO-INFER-IOT/src/`.
GEO-INFER-IOT

## Public import surface

These names are exported by the current owning package:

- `geo_infer_iot.IoTDataIngestion`
- `geo_infer_iot.SensorRegistry`
- `geo_infer_iot.RadiationMonitoringSystem`
- `geo_infer_iot.GlobalMonitoringSystem`
- `geo_infer_iot.IoTSystem`
- `geo_infer_iot.BayesianSpatialInference`
- `geo_infer_iot.MultiModalFusion`
- `geo_infer_iot.AdaptiveSampling`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_iot
from geo_infer_iot import IoTDataIngestion, SensorRegistry, RadiationMonitoringSystem, GlobalMonitoringSystem
assert all(value is not None for value in (IoTDataIngestion, SensorRegistry, RadiationMonitoringSystem, GlobalMonitoringSystem,))
assert geo_infer_iot.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module IOT --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-IOT/src/geo_infer_iot/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-IOT/README.md)
- [Module operating contract](../../../GEO-INFER-IOT/AGENTS.md)
- [Regression: test_iot_integration.py](../../../GEO-INFER-IOT/tests/integration/test_iot_integration.py)
- [Regression: test_calibration_topology_sensor_api.py](../../../GEO-INFER-IOT/tests/unit/test_calibration_topology_sensor_api.py)
- [Regression: test_data_ingestion.py](../../../GEO-INFER-IOT/tests/unit/test_data_ingestion.py)
- [Example source: radiation_surveillance.py](../../../GEO-INFER-IOT/examples/radiation_surveillance.py)
- [Example source: smart_sensor_network.py](../../../GEO-INFER-IOT/examples/smart_sensor_network.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
