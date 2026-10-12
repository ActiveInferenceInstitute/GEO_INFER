# GEO-INFER-BIO: Biological Systems

`GEO-INFER-BIO` owns the `geo_infer_bio` package under `GEO-INFER-BIO/src/`.
GEO-INFER-BIO: A bioinformatics module for the GEO-INFER framework.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_bio.SequenceAnalyzer`
- `geo_infer_bio.DataValidator`
- `geo_infer_bio.BioVisualizer`
- `geo_infer_bio.ClimateDataProcessor`
- `geo_infer_bio.ClimateDataset`
- `geo_infer_bio.MicrobiomeDataLoader`
- `geo_infer_bio.MicrobiomeDataset`
- `geo_infer_bio.SoilDataIntegrator`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_bio
from geo_infer_bio import SequenceAnalyzer, DataValidator, BioVisualizer, ClimateDataProcessor
assert all(value is not None for value in (SequenceAnalyzer, DataValidator, BioVisualizer, ClimateDataProcessor,))
assert geo_infer_bio.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module BIO --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-BIO/src/geo_infer_bio/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-BIO/README.md)
- [Module operating contract](../../../GEO-INFER-BIO/AGENTS.md)
- [Regression: test_bio_integration.py](../../../GEO-INFER-BIO/tests/integration/test_bio_integration.py)
- [Regression: test_climate.py](../../../GEO-INFER-BIO/tests/unit/test_climate.py)
- [Regression: test_graphql_api.py](../../../GEO-INFER-BIO/tests/unit/test_graphql_api.py)
- [Example source: sequence_analysis_example.py](../../../GEO-INFER-BIO/examples/sequence_analysis_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
