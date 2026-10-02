# GEO-INFER-SEC: Security

`GEO-INFER-SEC` owns the `geo_infer_sec` package under `GEO-INFER-SEC/src/`.
GEO-INFER-SEC provides security and privacy frameworks for sensitive geospatial information.

## Public import surface

These names are exported by the current owning package:

- `geo_infer_sec.AuthenticationManager`
- `geo_infer_sec.UserCredentials`
- `geo_infer_sec.TokenInfo`
- `geo_infer_sec.GeospatialAccessManager`
- `geo_infer_sec.AccessManager`
- `geo_infer_sec.Role`
- `geo_infer_sec.SpatialPermission`
- `geo_infer_sec.GeospatialEncryption`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_sec
from geo_infer_sec import AuthenticationManager, UserCredentials, TokenInfo, GeospatialAccessManager
assert all(value is not None for value in (AuthenticationManager, UserCredentials, TokenInfo, GeospatialAccessManager,))
assert geo_infer_sec.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module SEC --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-SEC/src/geo_infer_sec/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-SEC/README.md)
- [Module operating contract](../../../GEO-INFER-SEC/AGENTS.md)
- [Regression: test_sec_integration.py](../../../GEO-INFER-SEC/tests/integration/test_sec_integration.py)
- [Regression: test_serialization_security.py](../../../GEO-INFER-SEC/tests/integration/test_serialization_security.py)
- [Regression: test_anonymization.py](../../../GEO-INFER-SEC/tests/test_anonymization.py)
- [Example source: anonymization_example.py](../../../GEO-INFER-SEC/examples/anonymization_example.py)
- [Example source: comprehensive_security_example.py](../../../GEO-INFER-SEC/examples/comprehensive_security_example.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
