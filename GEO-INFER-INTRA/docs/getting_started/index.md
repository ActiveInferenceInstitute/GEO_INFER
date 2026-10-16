# Getting Started

This path takes a new contributor or analyst from a clean checkout to a small,
validated spatial-inference workflow. The commands assume the repository root
is the current working directory.

## Runtime defaults and required profiles

The profile observations below come from earlier uv 0.13.0 development notes.
They have not been revalidated for uv 0.13.1 and do not qualify this draft.
Fresh locked installs, container builds and declared gates remain required.

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

The BIO and OPS standalone base-service containers select Python 3.15 with
exact uv 0.13.1. Earlier uv 0.13.0 notes reported successful source-image
and offline import/runtime checks. The uv 0.13.1 images remain unqualified;
workspace integrations, service startup and scientific gates remain separate.
Their published Python 3.14 base supplies system libraries; uv installs the
selected executable runtime.

The IoT container selects Python 3.13 for its unchanged frozen workspace
closure. Locked pydantic-core 2.33.2 embeds PyO3 0.24.1, whose actual native
build rejects Python 3.14 and 3.15. Its Linux closure selects pandas 2.3.1;
the macOS pandas 2.1.4 failure above is a separate host profile. NumPy 1.26.4
successfully built on Linux Python 3.15 and does not itself justify this
exception. GDAL, PROJ and Fortran build prerequisites are also separate.
The IoT source image uses the workspace-root build context to include its
BAYES, SPACE and TIME distributions while preserving producer metadata and
lock bytes. Offline source-image qualification does not establish running
API, broker/database, sensor, sampling, hosted or scientific acceptance.

To move IoT to 3.15, separately review a producer dependency/lock update that
removes the PyO3 refusal, rebuild the selected closure and run its source/native
gates. Preserve prior image/campaign/publication receipts and paired revisions.

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
