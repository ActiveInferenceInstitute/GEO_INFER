# Getting Started

This path takes a new contributor or analyst from a clean checkout to a small,
validated spatial-inference workflow. The commands assume the repository root
is the current working directory.

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
