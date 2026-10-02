--

`GEO-INFER-CIV` owns the `geo_infer_civ` package under `GEO-INFER-CIV/src/`.
GEO-INFER-CIV: Civic Engagement and Participatory Mapping

## Public import surface

These names are exported by the current owning package:

- `geo_infer_civ.ParticipationAnalyzer`
- `geo_infer_civ.ParticipationMethod`
- `geo_infer_civ.ParticipantRecord`
- `geo_infer_civ.EngagementScore`
- `geo_infer_civ.RepresentationReport`
- `geo_infer_civ.AttendanceTracker`
- `geo_infer_civ.PublicCommentAnalyzer`
- `geo_infer_civ.VoterTurnoutModel`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_civ
from geo_infer_civ import ParticipationAnalyzer, ParticipationMethod, ParticipantRecord, EngagementScore
assert all(value is not None for value in (ParticipationAnalyzer, ParticipationMethod, ParticipantRecord, EngagementScore,))
assert geo_infer_civ.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module CIV --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-CIV/src/geo_infer_civ/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-CIV/README.md)
- [Module operating contract](../../../GEO-INFER-CIV/AGENTS.md)
- [Regression: test_integration.py](../../../GEO-INFER-CIV/tests/integration/test_integration.py)
- [Regression: test_acceptance_civ.py](../../../GEO-INFER-CIV/tests/unit/test_acceptance_civ.py)
- [Regression: test_civ_init.py](../../../GEO-INFER-CIV/tests/unit/test_civ_init.py)
- [Example source: basic_civic_engagement.py](../../../GEO-INFER-CIV/examples/basic_civic_engagement.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
