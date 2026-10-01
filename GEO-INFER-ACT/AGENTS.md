# Agent Instructions: GEO-INFER-ACT

## Scope

- Owning module: `GEO-INFER-ACT`
- Python package: `geo_infer_act`
- Directory role: Advanced Active Inference framework implementing Free Energy Principle for geospatial decision-making, perception, and learning.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_act` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `config/`
- `docs/`
- `examples/`
- `src/`
- `tests/`
- `debug_models.py`
- `verify_comprehensive.py`
- `verify_pipeline.py`
- `SKILL.md`
- `pyproject.toml`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```


## Current H3 Contracts

- Production ACT H3 runtime paths must use
  `geo_infer_act.utils.pymdp_adapter` and fail if `inferactively-pymdp` is not
  exactly `1.0.3`.
- Keep flat H3 method signatures and dictionary return shapes backward-compatible.
- Keep trace diagnostics JSON-safe and backed by typed result classes:
  `H3CellDiagnostics`, `H3EdgeDiagnostics`, `H3LevelDiagnostics`, and
  `SpatialInferenceTrace`.
- Use nested methods only for opt-in nested H3 behavior:
  `enable_nested_h3_spatial`, `update_nested_h3_beliefs`,
  `infer_over_nested_h3_grid`, `trace_over_nested_h3_grid`, `step_nested`,
  `trace_nested_step`, and `simulate_nested_h3_lattice`.
- Nested inference must reject invalid cells, observations outside the enabled
  hierarchy, mixed unexpected resolutions, non-finite observations, and empty
  belief vectors.
- Runner nested mode must keep outputs under the configured output directory and
  list generated hierarchy, diagnostics, and visualization files in the manifest.
- Runner flat and nested H3 modes must emit pymdp posterior, negative-EFE,
  free-energy, entropy, and backend-version diagnostics through manifest-linked
  sidecars.
- Runner flat and nested H3 modes must emit spatial trace JSON/CSV outputs plus
  belief-flux, policy-surface, policy-transition, spatial-autocorrelation,
  entropy/free-energy phase, and research-report HTML visualizations through
  the manifest sidecar pipeline.
- Research-profile and gallery runs must remain deterministic, real-H3, and
  non-degenerate: policy probabilities, entropy, local coherence, and belief
  flux should show finite variation above the validator thresholds.
- Use `uv run` for ACT/H3 commands. A system Python installation with a legacy
  `inferactively-pymdp` distribution is outside the supported contract.
- `create_h3_spatial_model` enforces a default 100,000-cell budget; callers
  with intentionally larger domains must pass `config["max_cells"]` explicitly.
- `infer_over_h3_grid` is read-only with respect to the attached generative
  model; preserve this contract when adding grid diagnostics.
- Optional Python model-source integrations (Bayeux, PyMC, and Pyro) require
  `config["allow_dynamic_code"] = True` and execute in per-call namespaces;
  keep this opt-in boundary when adding integrations.
- Preserve the finite-input, solve-based Gaussian, factor-table variational,
  policy information-gain, and multi-agent perception/action contracts when
  extending ACT core behavior.

## Failure Triage

- If nested belief tests fail, inspect `geo_infer_act.core.generative_model`
  before changing agents or runners.
- If artifact validation fails, inspect `geo_infer_act.runners.scenarios` and
  `geo_infer_act.runners.io` together.
- Re-run `uv run python GEO-INFER-TEST/run_unified_tests.py --h3-migration`
  after changing any H3 or nested spatial inference path.

## Visualization Guidance

- Keep plotting inputs finite and shape-aligned; use local figure styling so a
  library call cannot mutate the caller's matplotlib or seaborn configuration.
- H3 visualization writers must create output directories and handle constant
  metric ranges without emitting invalid colorbar limits.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
