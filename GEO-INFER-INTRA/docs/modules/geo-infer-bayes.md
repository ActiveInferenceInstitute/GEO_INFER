# GEO-INFER-BAYES: Bayesian Inference Engine

`GEO-INFER-BAYES` owns the `geo_infer_bayes` package under `GEO-INFER-BAYES/src/`.
GEO-INFER-BAYES: Bayesian inference for geospatial applications ======================================================================

## Public import surface

These names are exported by the current owning package:

- `geo_infer_bayes.SpatialGP`
- `geo_infer_bayes.SparseSpatialGP`
- `geo_infer_bayes.BayesianInference`
- `geo_infer_bayes.PosteriorAnalysis`
- `geo_infer_bayes.GaussianProcess`
- `geo_infer_bayes.SpatialCovariance`
- `geo_infer_bayes.VariationalInference`
- `geo_infer_bayes.MCMCSampler`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_bayes
from geo_infer_bayes import SpatialGP, SparseSpatialGP, BayesianInference, PosteriorAnalysis
assert all(value is not None for value in (SpatialGP, SparseSpatialGP, BayesianInference, PosteriorAnalysis,))
assert geo_infer_bayes.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module BAYES --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-BAYES/src/geo_infer_bayes/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-BAYES/README.md)
- [Module operating contract](../../../GEO-INFER-BAYES/AGENTS.md)
- [Regression: test_bayesian_workflow.py](../../../GEO-INFER-BAYES/tests/integration/test_bayesian_workflow.py)
- [Regression: test_abc_smc.py](../../../GEO-INFER-BAYES/tests/unit/test_abc_smc.py)
- [Regression: test_api_interfaces.py](../../../GEO-INFER-BAYES/tests/unit/test_api_interfaces.py)
- [Example source: spatial_bayesian_analysis.py](../../../GEO-INFER-BAYES/examples/spatial_bayesian_analysis.py)
- [Example source: spatial_gp_example.py](../../../GEO-INFER-BAYES/examples/spatial_gp_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
