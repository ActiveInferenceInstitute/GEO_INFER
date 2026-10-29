# Linux native I/O startup profile

The frozen GRIB and GIS wheels include different PROJ libraries. On Linux,
findlibs' package lookup recursively preloads ecCodes' eckit dependencies with
global symbol visibility. Loading Rasterio or PyProj in the same interpreter
can then resolve symbols from a different PROJ build and crash. The retained
diagnostic comparisons isolate the failure using both import orders and real files.

Use the explicit process-startup profile when running the complete GRIB/GIS
environment on Linux:

```bash
uv sync --all-packages --all-extras --all-groups
uv run --frozen python GEO-INFER-TEST/native_io_profile.py -- python -c 'import xarray; print(xarray.backends.list_engines())'
uv run --frozen python GEO-INFER-TEST/native_io_profile.py -- geo-infer modules
uv run --frozen python GEO-INFER-TEST/native_io_profile.py -- python GEO-INFER-TEST/run_unified_tests.py --category unit
```

This also supports programmatic workloads: launch their Python interpreter
through the profile before importing the native backends. The launcher replaces
itself with the requested command, forwarding its arguments and environment.
Command failures and signals remain failures. It does not change the calling
shell or an already-running interpreter. Library lookup paths must be configured
before Python starts; changing them after loading a backend cannot repair its
native state.

The profile derives paths from the active interpreter's installed eccodeslib
and eckitlib wheels. It requires exactly one supported lib/lib64 directory for
each native library, directs findlibs to ecCodes' explicit home, and provides
the matching eckit dependency directory to the process loader. Existing library
search paths follow those selected directories. It uses findlibs' documented
lookup controls; it neither preloads arbitrary libraries nor chooses an import
order. Missing, ambiguous or unrepresentable wheel layouts fail before launch.

The lookup controls apply to all findlibs calls in the launched process. The
current frozen workspace uses findlibs for ecCodes. A workload adding another
findlibs consumer must configure its native home and qualify that environment
separately. This profile is Linux-only; other platforms retain their declared
native providers and compiler profiles.

Inspect the configuration without executing a backend:

```bash
uv run --frozen python GEO-INFER-TEST/native_io_profile.py --describe
```

The description explicitly reports `execution_verified: false`. Path resolution
does not establish native interoperability. The permanent tests perform actual
GRIB2 and GeoTIFF writes/reads, xarray backend discovery and coordinate transforms
in separate interpreters for both import orders. File-layout tests qualify only
the environment builder. Full unit, coverage and platform acceptance remain
separate checks.

The shared CPU-workspace CI action appends this same profile to `GITHUB_ENV`
after frozen installation, before subsequent Python processes start. The
`--github-env PATH` mode is for that runner file, not for shell sourcing.
CI retains its ordinary unit selections, coverage floors and process deadlines.
