# Getting Started

This path takes a new contributor or analyst from a clean checkout to a small,
validated spatial-inference workflow. The commands assume the repository root
is the current working directory.

## Runtime defaults and required profiles

The historical host observations below remain scoped to their recorded uv and
interpreter versions. Each candidate needs its own locked/source/native gates;
later runtime support is not inferred from an earlier receipt. Linux service
profiles have their separate build and actual-health gate described below.

Use the latest stable CPython by default (currently 3.15) and exact uv 0.13.1.
This checkout's unchanged locked base has a named local runtime exception:
`.python-version` selects the latest 3.12 patch. Isolated frozen base installs on
macOS passed on 3.12; 3.14 and 3.15 attempted the locked Fiona 1.10.1 source
build and failed because `gdal-config`/GDAL were unavailable. On 3.13, locked
pandas 2.1.4 failed compiling its generated `_PyLong_AsByteArray` calls. These
are locked dependency and host prerequisites, not a claim that CPython 3.15
cannot run GEO-INFER. The declared Python 3.11 compatibility floor is unchanged.
Optional AI/Torch/JAX, all-workspace, native sampling and scientific publication
profiles require their own dependency and native qualification.

To remove this exception, provision/review GDAL and the other native build
prerequisites, qualify the unchanged or explicitly reviewed lock on 3.15, then
run the selected source/native profile gates before moving the default pin.
Preserve the lock, paired revisions, captured publisher bytes, generated
signposts and scientific receipts throughout that review. CI retains the
3.11/3.12 compatibility and native publication lanes; standalone formatter and
import-probe tools select 3.15 without installing the geospatial runtime.
Explicit `UV_PYTHON` continues to select a named compatibility/profile runtime.

### Linux container profiles

BIO, OPS, and IoT use the named frozen CPU profile: Python 3.12 and exact
uv 0.13.1. Build each image with the workspace root as its context so its
selected dependencies come from the existing `uv.lock`, exported with hashes.
The runtime installs only the selected module and required workspace siblings;
there is no unconstrained gunicorn or base-dependency resolution at image build.
The root `.dockerignore` admits only package sources, metadata and owned service
configuration. Repository credentials, environments, results and Git metadata
are excluded. Base image updates still require fresh build/runtime evidence.

```bash
docker build -f GEO-INFER-BIO/Dockerfile -t geo-bio .
docker build -f GEO-INFER-OPS/Dockerfile -t geo-ops .
docker build -f GEO-INFER-IOT/deployment/Dockerfile -t geo-iot .
```

The `Frozen service containers` workflow builds all three images, checks installed
dependency consistency and the selected Python, then starts each actual image
command and verifies its HTTP health response. It retains build, image, runtime,
service and health evidence on success or failure. No images are published.
Container health does not qualify broker/database, hardware or sampling work.

IoT starts `geo_infer_iot.api.sensor_api:create_app` with uvicorn's `--factory`.
Each worker owns independent real registry/ingestion state. An explicit
`CONFIG_PATH` must name a readable YAML mapping; malformed/missing selections
fail startup. The thin entrypoint executes the API as PID 1 so its lifecycle
receives container signals; it does not start monitoring in a short-lived helper.
Protocol consumers and performance monitoring remain explicit application work.

The former Python 3.15 BIO/OPS and 3.13 IoT draft images were not qualified under
uv 0.13.1. Their source is retained in Git history. Moving a runtime beyond this
frozen profile needs separately reviewed dependency/native-build qualification.

## Five-minute path

```bash
uv sync --all-packages --all-extras --all-groups
uv run python -c "import geo_infer_space, geo_infer_act; print('GEO-INFER imports are ready')"
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE
```

Then continue with [Your First Analysis](first_analysis.md).

## Select a learning path

### Spatial analysis

1. [Installation](installation_guide.md)
2. [Spatial analysis basics](spatial_analysis_basics.md)
3. [Your first map](first_map.md)
4. [H3 v4 reference](../geospatial/data_formats/h3/index.md)

### Active Inference

1. [Active Inference basics](active_inference_basics.md)
2. [Active Inference guide](../active_inference_guide.md)
3. [ACT module guide](../modules/geo-infer-act.md)
4. [ACT validation contracts](../../../GEO-INFER-TEST/validate_active_inference_contract.py)

### Framework development

1. [Developer guide](../developer_guide/index.md)
2. [Testing guide](../developer_guide/testing_guide.md)
3. [Module catalog](../modules/index.md)
4. [Contributing](../../../CONTRIBUTING.md)

## Core concepts

- **Module**: an independently packaged `GEO-INFER-*` workspace with source,
  tests, metadata, and operational documentation.
- **H3 cell**: a real H3 v4 index used as a stable spatial key. H3 resolutions
  range from 0 through 15; choose a resolution based on analysis scale and
  cell-count budget.
- **Belief**: a normalized probability distribution or model state maintained
  by an inference component.
- **Contract validator**: a repository executable that checks structural,
  numerical, documentation, or artifact invariants.

## After the first workflow

- Inspect [examples](../examples/README.md) for domain-oriented starting points.
- Read [architecture](../architecture/index.md) before adding a cross-module
  integration.
- Use [support](../support/index.md) when dependency, H3, or output validation
  fails.
