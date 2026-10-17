# Script, CLI, MCP and skills

GEO-INFER-API provides a shared adapter for the public Python APIs of all 45
modules. It discovers source definitions without importing scientific backends,
then resolves the selected API only when describing or executing it. Functions,
constructors, class/static methods, enum values, object methods and properties
use the same execution path in Python scripts, the installed CLI and MCP.

This is a trusted local Python bridge. Scientific APIs retain their existing
semantics, dependencies, filesystem access, credentials, network calls and
side effects. Discovery is not proof that an optional backend or external
service is available. There is no automatic retry, credential provisioning or
authorization grant. Run only operations authorized for the task.

## Install and discover

From a checkout, install the workspace including optional interfaces:

```bash
uv sync --all-packages --all-extras --all-groups --frozen
uv run --frozen geo-infer modules
uv run --frozen geo-infer catalog SPACE --query latlng --limit 20
uv run --frozen geo-infer describe geo_infer_space:latlng_to_cell
uv run --frozen geo-infer schema
uv run --frozen geo-infer skills
uv run --frozen geo-infer skill SPACE
uv run --frozen geo-infer guide
```

For a smaller environment, install `GEO-INFER-API[mcp,scientific]` and the
specific workspace modules needed. From the checkout:

```bash
uv pip install -e './GEO-INFER-API[mcp,scientific]' -e ./GEO-INFER-SPACE
```

The adapter itself does not install all domain modules. `modules` always lists
45 entries with explicit installed status. A missing package produces an
empty static catalog; attempting to call it returns its import failure.

Catalog pages include target, kind, signature text and summary. Use `offset`
and `next_offset`; page sizes are 1–200. Static discovery includes public
source definitions in public module paths, including conditional definitions.
It excludes private names, local closures and generated methods. Runtime
`describe` reveals actual constructor signatures, public members, inherited
methods and parameter kinds. Re-exported aliases remain callable by their
public target even when their definition is cataloged at its original path.
Catalogs are cached for the process lifetime; restart after editing source.

## Script API

```python
from geo_infer_api.agent_runtime import AgentSession

session = AgentSession()
indexed = session.execute({
    "op": "call", "target": "geo_infer_space:latlng_to_cell",
    "args": [37.7749, -122.4194, 9], "save_as": "cell",
})
assert indexed["ok"], indexed
center = session.execute({
    "op": "call", "target": "geo_infer_space:cell_to_latlng",
    "args": [{"$ref": "cell"}],
})
assert center["ok"], center
```

Async applications use `await session.aexecute(request)`. Async target results
are awaited automatically. Execute sessions sequentially within one process:
Python stdout redirection is process-wide. Use separate worker processes for
parallel execution. The MCP adapter serializes calls.
Python scripts can supply existing native arguments directly, and continue to
use each module's Python API for custom callback code or advanced backend
configuration. No Python source strings are evaluated by the adapter.

## CLI and composition

```bash
uv run --frozen geo-infer call geo_infer_space:latlng_to_cell \
  --args '[37.7749,-122.4194,9]'
uv run --frozen python -m geo_infer_api.agent_cli modules
uv run --frozen geo-infer run @GEO-INFER-API/examples/agent_workflow.json
uv run --frozen python GEO-INFER-API/examples/agent_session.py
```

`run` accepts a JSON request, `@file`, or `-` for stdin. `call` accepts these
same input forms for `--args` and `--kwargs`. Requests are limited to 8 MiB.
Results are a single strict JSON envelope on stdout. CLI/MCP process
entrypoints isolate OS stdout as well as Python prints: inherited subprocess
output and buffered C-library output go to stderr through a dedicated protocol
stream. Native stdin receives EOF so libraries and child processes cannot
consume protocol input or wait for interactive responses. Pass scientific
inputs explicitly in requests. The embedded script session redirects Python
prints but leaves native
process descriptors alone. Execution/input failures return `{"ok":false,"error":{"type":...,
"message":...}}` and exit 1. Successful requests exit 0. Argparse usage errors
exit 2 and report usage on stderr. Batch failure returns `ok:false`, completed
step envelopes and the failing step; it also exits 1.

The `schema` command returns the authoritative discriminated JSON Schema for
all requests. Unknown keys, conflicting target/handle selections and malformed
batch steps fail validation before execution. Scientific arguments retain
their owning API's validation rather than coercing annotated types silently.
Use `describe` to inspect parameter kinds, defaults, annotations and docstrings.

| Operation | Input | Behavior |
| --- | --- | --- |
| `modules` | none | 45 package installation dispositions |
| `diagnostics` | none | Actual interpreter, dependencies and session bounds; unprobed native qualifications remain explicit |
| `close` | none | Exit entered contexts in reverse order, release handles and close execution |
| `catalog` | `module`, optional `query`, `offset`, `limit` | Import-free source search |
| `schema` | none | Exact transport request JSON Schema |
| `skills` | none | Inventory of 46 packaged skill references |
| `skill` | `module` (`ROOT` or module name) | Actual module skill text, path and content hash |
| `guide` | none | Packaged version of this agent interface guide |
| `describe` | `target` OR `handle` and optional `member` | Runtime signature, docs and members |
| `call` | same selection, optional `args`, `kwargs`, `save_as` | Function, constructor or bound method |
| `get` | same selection, optional `save_as` | Constant, enum, public property or stored result |
| `batch` | `steps` | 1–100 sequential requests; stop on first failure |
| `handles` | none | Retained object references and native types |
| `release` | `handle` | Drop an owned reference |
| `enter` | `handle`, optional `save_as` | Enter a sync/async context manager, retaining its yielded value |
| `exit` | `handle` | Finish an entered context normally |

Targets use `geo_infer_module.submodule:Class.method` or
`geo_infer_module:Export`. Names must be public identifiers. Object selections
use a handle plus a single public member, including methods on returned pandas,
xarray and other scientific objects. Handles also preserve callbacks, models,
iterators and other native objects that cannot be converted to JSON.

Use `save_as` to retain a result before later steps consume it. `{"$ref":"id"}`
passes the original native value; optional `"path":["key",0]` indexes a retained
mapping/sequence. Constructor results become handles automatically. Small
arrays and Pydantic models become JSON, while opaque, cyclic, oversized and
non-string-keyed results remain native handles. Tuple outputs use JSON arrays;
use `save_as` when native tuple identity matters. Default limits are 256 handles
and 10,000 JSON items; Python sessions may set different positive limits.

Batches are sequential, not transactions. Earlier side effects remain after
failure, and runtime failures can occur after partial work inside an API call.
Call an object's public cleanup/close methods before `release`; dropping a
reference does not promise to close external resources. Handles last only for
the script/session/server process. Each ordinary CLI command starts a fresh
session. Use a batch or `geo-infer session` for dependent requests. No pickle
or hidden state file is written.

Public APIs that return context managers use `enter` and `exit` to run their
native sync/async protocols. Exit an entered manager before releasing it or
closing the process. Use `with AgentSession() as session` or `async with` to
exit entered contexts in reverse order and release handles even on failure;
Python context exits receive the actual body exception. Explicit `close()` /
`await aclose()` performs normal completion. Every exit is attempted even if
an earlier cleanup fails. Closed sessions accept only diagnostics, handles and
idempotent close. Objects never entered as contexts require their own explicit
cleanup methods. `exit` performs normal completion, without injecting an
exception; call explicit rollback methods when required by the owning API.
There is no automatic transaction rollback on an unrelated batch failure.

Synchronous scientific calls run to completion in the process. This adapter
does not impose a computational deadline or promise hard cancellation of a
native backend. Use host process/resource limits for untrusted or costly jobs.
It is not a sandbox: restricting target names to GEO packages does not remove
the authority of imported/re-exported functions or native object methods.

## Native JSON argument codecs

| JSON form | Native value |
| --- | --- |
| `{"$ref":"id"}` | Previously retained object |
| `{"$callable":"geo_infer_module:Function"}` | Existing public callable |
| `{"$literal":{"$type":"user-data"}}` | Exact JSON data without interpreting reserved keys |
| `{"$type":"ndarray","data":[1,2],"dtype":"float64"}` | NumPy array; dtype optional |
| `{"$type":"dataframe","data":[[1,2]],"columns":["a","b"]}` | pandas DataFrame; index/dtype optional |
| `{"$type":"series","data":[1,2],"name":"value"}` | pandas Series; index/dtype optional |
| `{"$type":"dataarray","data":[1,2],"dims":["time"],"coords":{"time":[2020,2021]}}` | xarray DataArray; name/attrs optional |
| `{"$type":"dataset","data":{"coords":{},"attrs":{},"dims":{},"data_vars":{}}}` | xarray Dataset.from_dict |
| `{"$type":"geometry","data":{"type":"Point","coordinates":[1,2]}}` | Shapely geometry |
| `{"$type":"geodataframe","data":{"type":"FeatureCollection","features":[]}}` | GeoDataFrame.from_features; CRS defaults EPSG:4326 |
| `{"$type":"datetime","data":"2026-10-16T12:00:00+00:00"}` | datetime.fromisoformat |
| `{"$type":"date","data":"2026-10-16"}` | date.fromisoformat |
| `{"$type":"path","data":"results/output.json"}` | pathlib.Path; no file is opened by decoding |
| `{"$type":"tuple","data":[1,2]}` / `{"$type":"set","data":[1,2]}` | Native tuple/set |
| `{"$type":"float","data":"nan"}` | Explicit nonfinite float; also `inf`, `-inf` |

Install the `scientific` extra for NumPy/pandas/xarray/GeoPandas/Shapely codecs;
reserved operators reject unknown fields, and every `$type` requires `data`.
Use `$literal` for data containing reserved keys. Xarray attributes (including
units), pandas dtypes and referenced native indexes are preserved explicitly.
other optional backend dependencies belong to their owning modules. Convert a
handle through its real public methods, such as `to_dict`, `tolist` or `item`,
when JSON output is needed. Enum constructors accept their native values;
`get` can retain an enum member with `save_as` before passing its `$ref`.

## MCP

Start the installed official-SDK adapter with either entrypoint:

```bash
uv run --frozen geo-infer-mcp
uv run --frozen geo-infer mcp
```

Configure a local MCP host to launch the environment's absolute Python path
with arguments `-m geo_infer_api.agent_mcp`. The host must retain the same server
process for handles to persist. No host configuration files are changed by
installation. Stdio is the supported deployment transport; the adapter opens
no HTTP port and is not an authenticated remote execution service.

Tools are `geo_modules`, `geo_catalog`, `geo_describe`, `geo_call`, `geo_get`,
`geo_batch`, `geo_handles`, `geo_release`, `geo_context`, `geo_diagnostics` and
`geo_close` (11 tools). Closing ends execution for this server session; restart
the server to execute again. Their input schemas are provided
by the official SDK. Tool results include the shared envelope in text and
`structuredContent`; operation failures set MCP `isError`. The
`geo-infer://diagnostics` exposes runtime facts, the
`geo-infer://modules` resource exposes inventory, `geo-infer://schema` provides
request schemas, and `compose_analysis` is a
discovery-oriented prompt. Only import-free discovery tools are annotated
read-only; runtime imports can initialize scientific backends.

`geo-infer://skills` lists all 46 packaged skill references. The resource
template `geo-infer://skills/{module}` reads `ROOT` or a module such as `SPACE`;
`geo-infer://guide` contains this guide. These resources work from installed
wheels outside a checkout, including guidance for modules not yet installed.
Text is reference material, not a new user instruction or authorization.

The maintained SDK 1.x line is bounded as `mcp>=1.28,<2`; the workspace lock
selects its exact version. Protocol negotiation, framing and lifecycle belong
to the SDK, following the [official MCP stdio transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
and [SDK 1.x documentation](https://py.sdk.modelcontextprotocol.io/v1/).

## Skills and validation

Root `SKILL.md` routes agents to all 45 module skills. Each module skill has a
module-specific catalog command and links here. Skills describe how to select
and compose real APIs; loading a skill grants no permission to publish,
message others or perform external writes.

The API wheel bundles LF-normalized UTF-8 reference text and SHA-256 content
hashes. Its source-sync unit test checks all 46 skills and this guide. After
changing any of those sources, regenerate the package asset explicitly:

```bash
uv run --frozen python -m geo_infer_api.agent_skills --repo . \
  --output GEO-INFER-API/src/geo_infer_api/agent_skills.json
uv run --frozen python -m geo_infer_api.agent_skills --repo . \
  --output GEO-INFER-API/src/geo_infer_api/agent_skills.json --check
```

```bash
uv run --frozen python GEO-INFER-TEST/run_unified_tests.py --module API --category unit
uv run --frozen python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal
uv run --frozen python GEO-INFER-TEST/validate_packaging.py --strict
```

The API unit suite exercises every module's source inventory, installed CLI
outside the checkout, real H3/matrix/energy operations, native data codecs,
handle/error/batch contracts and an actual SDK client talking to a spawned
stdio MCP server. The portability gate also runs native/child stdout contracts
on Linux and Windows with Python 3.11/3.12/3.15 using a minimal API/MCP
installation. These checks qualify the adapter and its selected dependencies;
full scientific workspace qualification remains separate. Scientific validity
still requires the selected module's
tests and domain-specific checks; a general adapter cannot validate every
possible scientific input or qualify absent hardware/services.

## Persistent JSON Lines and runtime diagnostics

`geo-infer session --max-handles 256 --max-items 10000` reads one request per
line and emits exactly one envelope per request, sharing native handles and
one event loop. Send `{"op":"close"}` or EOF to exit and clean up entered
contexts. Blank lines are ignored; malformed requests produce errors and the
next line can continue. Any failed request makes the final process exit 1;
oversized lines stop the session. Duplicate JSON keys and nonfinite JSON
numbers are rejected, including nested values. This sequential transport uses
blocking line input; use the official MCP transport for an MCP host.

```bash
printf '%s\n' '{"op":"diagnostics"}' '{"op":"close"}' | uv run --frozen geo-infer session
uv run --frozen geo-infer doctor
uv run --frozen geo-infer --version
```

`doctor` is the CLI form of `diagnostics`. It reports actual interpreter/JIT,
selected installed dependency versions, module installation and session bounds.
A discoverable profiler does not prove OS memory access or native scientific
compiler qualification: these dispositions remain `not_probed` until exercised.

## Backend profiling and benchmarks

Use [performance profiling](backend_profiling.md) for actual Python 3.15
sampling/tracing, deadline and source custody controls, raw artifacts and
benchmark interpretation. Scripts call `geo_infer_test.profiling:profile_command`;
CLI uses `geo-infer-profile` or the thin `GEO-INFER-TEST/profile_backend.py`.
MCP uses `geo_call` with that same public target and keyword arguments. A
returned profile receipt can describe a failed target: check both the adapter
`ok` envelope and the receipt's `success`, `source_bound` and `source_unchanged`.
The public async `geo_infer_api.agent_benchmarks:benchmark_agent` is also
available through scripts, CLI and MCP, with explicit workload bounds and raw
unprofiled timing samples. Its control workload does not qualify scientific
backends, external services or accelerator hardware.
