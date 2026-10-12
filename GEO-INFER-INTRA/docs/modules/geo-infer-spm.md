# GEO-INFER-SPM: Spatial Process Modeling

`GEO-INFER-SPM` owns the `geo_infer_spm` package under `GEO-INFER-SPM/src/`.
GEO-INFER-SPM: Statistical Parametric Mapping for Geospatial Analysis

## Public import surface

These names are exported by the current owning package:

- `geo_infer_spm.GeneralLinearModel`
- `geo_infer_spm.fit_glm`
- `geo_infer_spm.RandomFieldTheory`
- `geo_infer_spm.compute_spm`
- `geo_infer_spm.Contrast`
- `geo_infer_spm.contrast`
- `geo_infer_spm.SpatialAnalyzer`
- `geo_infer_spm.TemporalAnalyzer`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_spm
from geo_infer_spm import GeneralLinearModel, fit_glm, RandomFieldTheory, compute_spm
assert all(value is not None for value in (GeneralLinearModel, fit_glm, RandomFieldTheory, compute_spm,))
assert geo_infer_spm.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module SPM --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-SPM/src/geo_infer_spm/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-SPM/README.md)
- [Module operating contract](../../../GEO-INFER-SPM/AGENTS.md)
- [Regression: test_correctness.py](../../../GEO-INFER-SPM/tests/integration/test_correctness.py)
- [Regression: test_full_pipeline.py](../../../GEO-INFER-SPM/tests/integration/test_full_pipeline.py)
- [Regression: test_real_data.py](../../../GEO-INFER-SPM/tests/integration/test_real_data.py)
- [Example source: spatiotemporal_climate_analysis.py](../../../GEO-INFER-SPM/examples/advanced_analysis/spatiotemporal_climate_analysis.py)
- [Example source: spatial_trend_analysis.py](../../../GEO-INFER-SPM/examples/basic_usage/spatial_trend_analysis.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
