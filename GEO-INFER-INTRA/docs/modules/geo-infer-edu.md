# GEO-INFER-EDU: Educational Technology Module

`GEO-INFER-EDU` owns the `geo_infer_edu` package under `GEO-INFER-EDU/src/`.
GEO-INFER-EDU: Educational Technology Module

## Public import surface

These names are exported by the current owning package:

- `geo_infer_edu.CurriculumDesigner`
- `geo_infer_edu.ExerciseGenerator`
- `geo_infer_edu.ProgressTracker`
- `geo_infer_edu.PersonalizedLearning`
- `geo_infer_edu.ProfessionalDevelopment`

The small example below verifies the installed import contract. It does not infer
scientific validity or service availability from successful imports; the owning
regression tests exercise behavior with concrete inputs.

```python
import geo_infer_edu
from geo_infer_edu import CurriculumDesigner, ExerciseGenerator, ProgressTracker, PersonalizedLearning
assert all(value is not None for value in (CurriculumDesigner, ExerciseGenerator, ProgressTracker, PersonalizedLearning,))
assert geo_infer_edu.__version__ == "0.4.0"
```

Run examples in the shared, locked workspace environment. Constructor inputs,
optional backends, and result shapes belong to the referenced source and tests.
Cross-module callers should pass explicit spatial state ordering and timezone-aware
instants when those fields are part of their data contract.

## Verification

From the repository root:

```bash
uv run --no-sync python GEO-INFER-TEST/run_unified_tests.py --module EDU --timeout 600 --workers 2
```

The module command includes its owned test files and registered nested test roots.
The fleet's separate unit and slow categories cover complementary marker selections;
release CI requires unit, slow, integration, performance, system, and H3 lanes on
Python 3.11 and 3.12. Results include immutable attempt receipts under
`.geo-infer-test-results/runs/`, with logs, current JUnit, selection inventories,
interpreter and source custody. A missing optional dependency must be addressed by
the declared package extra rather than by omitting its tests.

## Source and examples

- [Owning package](../../../GEO-INFER-EDU/src/geo_infer_edu/README.md)
- [Module inventory and dependencies](../../../GEO-INFER-EDU/README.md)
- [Module operating contract](../../../GEO-INFER-EDU/AGENTS.md)
- [Regression: test_education_workflow.py](../../../GEO-INFER-EDU/tests/integration/test_education_workflow.py)
- [Regression: test_curriculum.py](../../../GEO-INFER-EDU/tests/test_curriculum.py)
- [Regression: test_exercise_generator.py](../../../GEO-INFER-EDU/tests/test_exercise_generator.py)
- [Example source: curriculum_design.py](../../../GEO-INFER-EDU/examples/curriculum_design.py)
- [Example source: interactive_learning.py](../../../GEO-INFER-EDU/examples/interactive_learning.py)

See the [cross-module integration guide](../../../GEO-INFER-EXAMPLES/docs/INTEGRATION_GUIDE.md)
for actual DATA, SPACE, TIME, BAYES, and ACT composition checks.
