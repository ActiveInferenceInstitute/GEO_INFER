# Backend performance profiling

The owning implementation is `geo_infer_test.profiling.profile_command`.
The installed `geo-infer-profile` command and thin
`GEO-INFER-TEST/profile_backend.py` share it. Select an actual script or module,
an explicit Python executable, a fresh output directory and a finite timeout.
Targets run once with the current user's authority and environment.

```bash
uv run --frozen geo-infer-profile --module geo_infer_api.agent_benchmarks \
  --output .geo-infer-test-results/profiles/agent-tracing --timeout 120 \
  -- --iterations 1000 --repeats 5 --warmup 1
uv run --frozen python GEO-INFER-TEST/profile_backend.py \
  --python python3.15 --engine sampling --mode cpu \
  --module geo_infer_api.agent_benchmarks \
  --output .geo-infer-test-results/profiles/agent-sampling --timeout 120 \
  -- --iterations 3000 --repeats 3 --warmup 1
```

The selected interpreter must contain the target dependencies; selecting 3.15
does not upgrade or qualify the shared scientific environment. A minimal API
installation can execute the control benchmark without importing scientific
backends. Use separate environments for different interpreter qualifications.

Tracing uses Python 3.15's `profiling.tracing`, and `cProfile` on older qualified
interpreters. Native `profile.pstats` and bounded cumulative hotspots contain
instrumented function times and call counts. Sampling uses Python 3.15's
[`profiling.sampling`](https://docs.python.org/3.15/library/profiling.sampling.html)
owned-child runner, all threads and native frames; the collapsed profile
contains weighted thread stacks, not function call counts. Select wall, CPU,
GIL or exception sampling mode explicitly. Memory inspection needs OS
permission; denied sampling fails with evidence. There is no elevation,
arbitrary PID attachment, fallback or retry. JIT availability and activation
are recorded independently; compare explicit `PYTHON_JIT` configurations in
separate runs. See [Python 3.15 changes](https://docs.python.org/3.15/whatsnew/3.15.html).

Receipts record actual target runtime, executable, completion token/PID,
command, elapsed time, source/lock identity before and after, raw stdout/stderr,
profile summary and SHA-256 of artifacts. Existing output directories are
never overwritten. Target exceptions, timeouts, missing completion, empty or
oversized profiles and changed source fail `success`. The deadline covers
runtime probing and target execution; final source auditing has a separate
bounded ten-second budget. The profile artifact limit defaults to 128 MiB;
stdout/stderr capture shares the existing process runner and is not a memory
quota. Use process resource limits for large-output targets.

Output inside a Git checkout must be ignored to avoid self-induced source
drift. Non-Git targets can execute, but `source_bound` is false and
`source_unchanged` is null; they provide no repository qualification claim.
Retain failed receipts alongside successful runs. Artifacts can contain target
paths and output; share them under the target data's normal access policy.

All execution surfaces use this same callable:

```python
from geo_infer_test.profiling import profile_command

receipt = profile_command(
    module="geo_infer_api.agent_benchmarks",
    args=["--iterations", "1000", "--repeats", "5"],
    output=".geo-infer-test-results/profiles/script-tracing",
    timeout=120,
)
assert receipt["success"] and receipt["source_bound"] and receipt["source_unchanged"]
```

CLI `geo-infer call geo_infer_test.profiling:profile_command --kwargs ...`
and MCP `geo_call` accept the same keyword fields; use tagged path values or
plain path strings. Check the returned receipt's `success` in addition to the
transport envelope. Library calls return failed receipts; the profiling CLI
also exits 1. There is no automatic repair or re-execution of failed targets.

For throughput comparisons, run the benchmark without a profiler:

```bash
uv run --frozen python -m geo_infer_api.agent_benchmarks \
  --iterations 1000 --repeats 5 --warmup 1
```

`benchmark_agent` is async and composable through `AgentSession` and MCP. Its
schema, paginated catalog, native direct-call and 25-step batch cases retain
raw sample seconds, medians, ranges and operation counts. All 45 source
inventories are scanned before measured warmed requests; catalog scan time is
reported separately and may be warm if the process already cached inventory.
Compare identical workloads on the same host/runtime/source conditions;
instrumented timings are diagnostic and must not be presented as throughput
improvements. CI executes real tracing on 3.11/3.12/3.15 and real 3.15 sampling
on Linux, checking completion, nonempty samples and unchanged source custody.

`PerformanceMonitor.section` balances cleanup after exceptions and preserves
external or overlapping tracemalloc ownership. Its memory peak is the observed
process-wide traced peak, not a section's exclusive allocation. `BenchmarkRunner`
reports nearest-rank p95/p99 and throughput; `LoadTester` keeps only concurrency-
bounded pending work and counts exceptions even when their messages are empty.
