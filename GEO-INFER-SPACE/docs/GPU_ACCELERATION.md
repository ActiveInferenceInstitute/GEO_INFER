# Numeric acceleration and host H3 topology

`geo_infer_space.backends.gpu` provides NumPy reference kernels for great-circle
and Euclidean distances. Optional CuPy, PyTorch CUDA, or JAX GPU installations
can execute the same float64 array formulas. Accelerator libraries are not
imported when this package or `H3Backend` is imported or constructed.

```python
from geo_infer_space.backends.gpu import (
    get_backend_diagnostics,
    gpu_spatial_join_by_distance,
    pairwise_haversine_kernel,
)

diagnostics = {}
pairs, unmatched_a, unmatched_b = gpu_spatial_join_by_distance(
    [[37.7749, -122.4194], [34.0522, -118.2437]],
    [[37.8044, -122.2712]],
    max_distance_km=30,
    backend="cpu",
    chunk_size=256,
    diagnostics=diagnostics,
)
assert pairs == [(0, 0)]
assert unmatched_a == [1]
assert diagnostics["used_backends"] == ["cpu"]
```

## Selecting and diagnosing execution

All numeric kernels accept keyword arguments `backend`, `chunk_size`, and
`diagnostics`. `backend="cpu"` performs no accelerator discovery.
`backend="auto"` (the default) probes CuPy, PyTorch, then JAX and chooses the
first usable GPU. If none works, CPU execution records a reason and logs an
informational message. If a selected GPU fails during a tile, automatic mode
logs a warning, computes that tile on CPU, and uses CPU for remaining tiles.
Explicit `"cupy"`, `"torch"`, or `"jax"` requests raise
`AcceleratorUnavailableError` when unavailable; execution failures raise
`RuntimeError` with the original exception as their cause.

The optional `diagnostics` dictionary is cleared at call start and records:

- `requested_backend`: the requested selection.
- `backend`: the actual backend, `"mixed"` after successful GPU tiles followed
  by CPU fallback, or `"none"` when empty inputs require no numeric computation.
- `used_backends`: backends that successfully produced output, in execution order.
- `fallback_reason`: discovery or runtime reason, or `None`.

`get_backend_diagnostics()` separates library installation from usable GPU
execution, including import/driver failures. Probes validate device allocation,
a numeric operation, and a float64 result. Results are cached; call
`get_backend_diagnostics(refresh=True)` after changing device availability or
configuration. JAX requires a GPU and `JAX_ENABLE_X64=1` configured before its
initialization; CPU-only JAX is not advertised as GPU acceleration. The library
does not change JAX global precision settings. Use `is_accelerator_available()`
or `get_available_backends()` for capability checks; the module exposes no
import-time `HAS_*` availability flags.

GPU packages and drivers are provisioned separately for the target machine.
They are not mandatory SPACE dependencies. This change establishes CPU and
simulated failure-path correctness; real GPU parity and performance measurements
require hardware verification. No speedup is promised.

## Numerical and memory contracts

Geographic inputs are `(N, 2)` arrays of `(latitude, longitude)` degrees. Values
must be finite, latitude must be in `[-90, 90]`, and longitude in `[-180, 180]`.
An empty geographic list is accepted. Earth radii and join thresholds must be
finite and strictly positive. The haversine intermediate is clamped to `[0, 1]`
to prevent roundoff near antipodes from producing NaN. Euclidean inputs must
have matching positive dimension counts and finite values. Output is float64;
unrepresentable nonfinite results raise instead of returning misleading values.

`chunk_size` is a positive integer (default 1024), bounding the number of points
from either input in a tile. Matrix APIs necessarily retain the `(N, M)` result,
but bound intermediate arrays. The distance join never builds the entire
pairwise distance matrix: working distance storage is bounded by the tile size
squared. Dense match outputs can still require quadratic memory. Euclidean
kernels accumulate dimensions without allocating an `(N, M, D)` tensor.

## H3 integration and migrations

`H3Backend.geodesic_distance_matrix()` and `geodesic_spatial_join()` accept
keyword-only `backend` (default `"auto"`; `"cpu"` forces CPU without probing
accelerators) and `chunk_size`, and return diagnostics. Invalid centroid cell
identifiers raise; they are never converted to `(0, 0)`.

H3 grid distance (`compute_distance_matrix()`) and topology joins
(`spatial_join()`) always use host H3 and take no backend argument; metadata
reports `backend="cpu"` and `accelerator=[]`. H3 `intersects` means identical or distance-1 adjacent cells,
and `contains`/`within` mean strict H3 ancestry; these are not arbitrary polygon
intersection predicates. Join matches preserve input order; unmatched cell IDs
are unique in input order. Invalid cells in topology joins raise.
`spatial_join_kernel()` infers ancestry from each cell's actual resolution. Grid-distance matrices retain the
documented `-1` sentinel for incomparable or invalid pairs.

For distance joins using `label_offsets_a`/`label_offsets_b`, both matched pairs
and unmatched outputs now use the supplied integer labels. Previously unmatched
outputs incorrectly returned row indices. Labels must have exactly one integer
per input point; float and boolean labels are rejected. A group is matched if
any of its points matches. Pairs and unmatched labels are unique and sorted.
Without labels they remain row indices.

## Validation

```bash
uv run --no-sync python -m pytest \
  GEO-INFER-SPACE/tests/unit/test_gpu_acceleration.py \
  GEO-INFER-SPACE/tests/integration/test_gpu_spatial_integration.py --no-cov
```

Tests cover independent scalar references, antipodes, empty inputs, input
validation, bounded tile execution, label groups, unavailable and failing
backends, mixed execution diagnostics, real H3 topology, and imports with all
accelerator imports forbidden. Actual available GPUs are compared with CPU in
the capability-dependent parity test. Without a GPU it verifies diagnosed CPU
fallback; real hardware parity remains explicitly deferred, not inferred from
that passing CPU test. The repository test contract forbids skipped tests.

Hardware discovery on 2026-09-04 in the development environment (macOS 26.6.2,
arm64, Python 3.12.13) found no CuPy installation, PyTorch 2.8.0 with zero CUDA
devices, and JAX 0.5.3 with only `TFRT_CPU_0`. PyTorch reported Apple MPS
availability, but the existing accelerator contract requires float64 execution
and supports CUDA rather than MPS. This probe does not verify physical GPU
parity. `[DEFERRED-VERIFY]`: run the documented capability-dependent parity
test on provisioned CUDA hardware and retain the device, driver, float64,
boundary and allocation-failure receipts before closing hardware verification.
