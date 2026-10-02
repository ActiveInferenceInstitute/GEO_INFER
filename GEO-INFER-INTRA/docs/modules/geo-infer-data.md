# GEO-INFER-DATA: Data Management Engine

`GEO-INFER-DATA` owns the `geo_infer_data` package under `GEO-INFER-DATA/src/`.
GEO-INFER-DATA: Geospatial Data Management, ETL, and Storage Optimization

## Public import surface

These names are exported by the current owning package:

- `geo_infer_data.MultiSourceDataIngestion`
- `geo_infer_data.IntelligentETLPipeline`
- `geo_infer_data.AdaptiveDataStorage`
- `geo_infer_data.DataQualityManager`
- `geo_infer_data.Dataset`
- `geo_infer_data.DatasetMetadata`
- `geo_infer_data.DataQualityReport`
- `geo_infer_data.initialize_data_system`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_data
from geo_infer_data import MultiSourceDataIngestion, IntelligentETLPipeline, AdaptiveDataStorage, DataQualityManager
assert all(value is not None for value in (MultiSourceDataIngestion, IntelligentETLPipeline, AdaptiveDataStorage, DataQualityManager,))
assert geo_infer_data.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module DATA --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-DATA/src/geo_infer_data/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-DATA/README.md)
- [Module operating contract](../../../GEO-INFER-DATA/AGENTS.md)
- [Regression: test_end_to_end.py](../../../GEO-INFER-DATA/tests/integration/test_end_to_end.py)
- [Regression: test_pipeline_integration.py](../../../GEO-INFER-DATA/tests/integration/test_pipeline_integration.py)
- [Regression: test_benchmarks.py](../../../GEO-INFER-DATA/tests/performance/test_benchmarks.py)
- [Example source: api_example.py](../../../GEO-INFER-DATA/examples/api_example.py)
- [Example source: basic_ingestion_example.py](../../../GEO-INFER-DATA/examples/basic_ingestion_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
