---
name: geo-infer-test
description: Unified tests, async benchmarks, source-bound profiling and optional template PDF rendering for GEO-INFER. Use when running cross-module categories, discovering tests, measuring awaited workloads, reviewing profiling receipts or rendering a prepared manuscript with docxology/template.
prerequisites:
  required: []
  recommended: []
difficulty: beginner
estimated_time: 30min
examples_dir: ../GEO-INFER-EXAMPLES/examples/
---

# GEO-INFER-TEST

## Instructions

### Core Capabilities

- **Unified test runner**: `run_unified_tests.py` for all 45 modules
- **Category filtering**: unit, integration, system, performance
- **Module filtering**: Test individual or groups of modules
- **Result reporting**: JUnit XML + HTML reports per module
- **Fixtures**: Shared spatial test data and coordinate generators

### Usage

```bash
# Run all tests
uv run python GEO-INFER-TEST/run_unified_tests.py

# Run specific module
uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH

# Run by category
uv run python GEO-INFER-TEST/run_unified_tests.py --category integration

# Canonical runner
uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH
```

### Pytest Markers

`unit`, `integration`, `system`, `performance`, `geospatial`, `api`, `slow`, `fast`

## Examples

```python
# Spatial test fixture: generate realistic test coordinates
import numpy as np

def make_test_coordinates(n=100, center=(45.5, -122.6), spread=0.1):
    """Generate n random lat/lng pairs around a center point."""
    lats = np.random.normal(center[0], spread, n)
    lngs = np.random.normal(center[1], spread, n)
    return list(zip(lats.clip(-90, 90), lngs.clip(-180, 180)))

coords = make_test_coordinates(200)
```

```python
# Property-based test example with Hypothesis
from hypothesis import given, strategies as st

@given(
    lat=st.floats(min_value=-90, max_value=90),
    lng=st.floats(min_value=-180, max_value=180),
    resolution=st.integers(min_value=0, max_value=15)
)
def test_h3_roundtrip(lat, lng, resolution):
    """The cell center re-encodes to the same cell at the same resolution."""
    import h3
    cell = h3.latlng_to_cell(lat, lng, resolution)
    rlat, rlng = h3.cell_to_latlng(cell)
    assert h3.latlng_to_cell(rlat, rlng, resolution) == cell
```

```bash
# Run specific test categories
uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```

### Agent interfaces

Use `uv run --frozen geo-infer catalog TEST` to discover public source
APIs, then `geo-infer describe` before calling a selected target. Scripts use
`geo_infer_api.agent_runtime.AgentSession`; the optional stdio MCP server is
`geo-infer-mcp`. See [requests, native objects and composition](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md).
The adapter retains this module's dependencies and authorization requirements.

## Guidelines

- Use real objects in tests
- Property-based tests (Hypothesis) in ≥10 modules
- Retain the runner's immutable attempt receipts, selection and JUnit files
  under `.geo-infer-test-results/runs/`; discovery does not prove execution.

### Integrations

- **EXAMPLES** → Example code treated as integration tests
- **OPS** → CI/CD test results feed monitoring
- **All modules** → Unified test runner spans all 45 modules

### Profile actual backend workloads

Use `geo-infer-profile --module geo_infer_api.agent_benchmarks --output
.geo-infer-test-results/profiles/agent-run --timeout 120` for tracing. Select
`--python python3.15 --engine sampling` only in a target environment with its
dependencies and OS inspection permission. Script and MCP target is
`geo_infer_test.profiling:profile_command`; check receipt `success`,
`source_bound` and `source_unchanged` as well as transport `ok`. Retain failed
runs. Read [profiling controls and native evidence](../GEO-INFER-INTRA/docs/integration/backend_profiling.md)
for sampler modes, runtime/JIT facts, artifact custody and timing interpretation.
Read retained evidence through `geo-infer-profile --report RECEIPT --sort-by own`
or `geo_infer_test.profiling:summarize_profile` in Python, CLI or MCP. The report
verifies sealed bytes without rerunning targets; preserve historical success
and source identity. Use an independently retained receipt hash through
`--receipt-sha256` / `expected_receipt_sha256` when authenticity matters. Trusted
local receipts remain required; artifact hashes alone do not authenticate a
rewritten manifest. Sampling ranks weighted stacks, tracing ranks function
time/calls, and overlapping cumulative times are not additive percentages.
Use `PerformanceMonitor.section` for balanced tracer ownership and the bounded
`LoadTester` for concurrent local workloads.

Await `BenchmarkRunner.run_async` or `LoadTester.run_async` for async functions,
callable objects and wrappers returning awaitables. Measurements include awaited
completion. Async load owns at most `concurrency` workers and drains them on
cancellation; ordinary request errors count in metrics. Benchmark failures
propagate. Synchronous `run` rejects async work instead of timing coroutine
creation. Async methods do not offload blocking synchronous functions.

`TestDiscoverer` includes async test definitions and reports malformed files,
actual file totals, ROOT categories, extra roots and cross-module imports.
Discovery metadata alone does not prove tests executed; retain runner receipts
and JUnit results for qualification.

### Render with docxology/template

After a clean publication build, use the explicit optional template lane in
`scripts/render_manuscript_pdf.py --template-root PATH --template-python PYTHON
--template-revision FULL_COMMIT`. The separate interpreter must already have
template's dependencies. The template checkout must be clean. Script, CLI and
MCP target `geo_infer_test.manuscript_rendering:render_template_pdf` consumes
the prepared combined Markdown and records template source hashes, runtime,
inputs, owned-process evidence and the final PDF hash. This runs trusted local
code and does not install dependencies or replace publication verification.
See [template rendering and cover checks](../GEO-INFER-INTRA/docs/integration/manuscript_rendering.md).


## Linux GRIB/GIS workloads

After installing the complete frozen workspace, use the explicit
[native I/O startup profile](docs/native_io_profiles.md) for Linux runtime
commands and programmatic Python consumers:

```bash
uv run --frozen python GEO-INFER-TEST/native_io_profile.py -- python -c 'import xarray; print(xarray.backends.list_engines())'
```

The profile configures native library lookup before the child process starts.
Its description is advisory; real GRIB/GeoTIFF, backend-discovery and projection
tests provide the scoped interoperability checks. Retain full unit/coverage
results separately. Other platforms keep their declared native profiles.
