# GEO-INFER-MATH: numeric and spatial primitives

MATH supplies geometry, spatial statistics, interpolation, optimization, linear
algebra and numerical methods through `geo_infer_math`. Algorithms own their
mathematical units and assumptions; higher-level applications choose coordinate
reference systems, observational error models and domain constraints.

## Great-circle distance

`haversine_distance` takes latitude then longitude in degrees and returns
kilometers. On its spherical Earth model, one quarter of the equator spans
approximately 10,008 kilometers. It is not an ellipsoidal geodesic calculation.

```python
from geo_infer_math import haversine_distance
from geo_infer_math.utils import MemoryConstraintError

assert haversine_distance(0., 0., 0., 0.) == 0.
quarter_equator = haversine_distance(0., 0., 0., 90.)
assert 10000. < quarter_equator < 10010.
assert issubclass(MemoryConstraintError, Exception)
```

Use the owning `core.geometry` and `core.spatial_statistics` APIs for coordinate
and statistical operations, and `core.interpolation` for IDW, RBF and kriging
models. Import names and constructor arguments are documented in the module
SKILL and checked by the owning tests; a universal spatial analyzer facade is not
part of this surface.

## Resource and GPU boundaries

The custom memory exception is `MemoryConstraintError` in 0.4.0, so it cannot be
confused with Python's builtin `MemoryError`. Importing MATH does not initialize
GPU runtimes. Explicit GPU calls probe available accelerator libraries; CPU
success does not establish hardware acceleration or backend equivalence.

Run `uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH` for numeric
references and input contracts. See the
[0.4.0 migration guide](../releases/0.4.0_migration.md).
