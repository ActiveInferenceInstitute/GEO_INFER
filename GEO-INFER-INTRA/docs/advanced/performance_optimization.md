# Performance Optimization

Profile the workload and preserve its numerical contract before optimizing it.
Spatial grids can produce quadratic dense transition tensors, while an explicit
space-time alignment costs the number of cells multiplied by the time-axis length.
Use sparse transitions for large domains and set allocation budgets at boundaries.

## Profile a real H3 operation

The example profiles sparse movement construction and independently checks that
both operators conserve probability. It also exercises the dense allocation gate.

```python
import cProfile
import h3
import numpy as np
from geo_infer_space import H3StateSpace

center = h3.latlng_to_cell(41.75, -124.2, 8)
space = H3StateSpace(sorted(h3.grid_disk(center, 1)))
profiler = cProfile.Profile()
operators = profiler.runcall(space.transitions)
for operator in operators:
    np.testing.assert_allclose(np.asarray(operator.sum(axis=0)).ravel(), 1.0)
try:
    space.dense_transition_tensor(max_entries=1)
except ValueError:
    pass
else:
    raise AssertionError("dense allocation budget was ignored")
assert profiler.getstats()
```

## Measurement and optimization choices

Record interpreter, lock hash, input sizes, H3 resolution, peak memory, cold/warm
state, and repeated elapsed measurements. Measure source import costs separately
from the steady-state operation. TIME's scalar timestamp helper imports without
loading its analytical engines; accessing an analysis export loads that component.

Vectorized arithmetic can reduce Python overhead, but wrapping an H3 loop in a
NumPy array does not make H3 indexing vectorized. Cache immutable neighborhood
lookups when a workload revisits cells, stream transport records when a finite
batch suffices, and compare the output against the original numerical oracle.
Choose resolution from measurement support and cost rather than maximum detail.

Test speed improvements must retain full collected/selected/executed inventories.
The [TEST guide](../modules/geo-infer-test.md) describes the required lanes; a
small contract run is early feedback, while the full category gates remain release
checks. See [SPACE composition](../../../GEO-INFER-SPACE/docs/CROSS_MODULE_COMPOSITION.md).
