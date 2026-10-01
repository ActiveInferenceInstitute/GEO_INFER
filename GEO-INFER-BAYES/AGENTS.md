# Agent Instructions: GEO-INFER-BAYES

## Scope

- Owning module: `GEO-INFER-BAYES`
- Python package: `geo_infer_bayes`
- Directory role: Comprehensive Bayesian inference framework with probabilistic modeling, uncertainty quantification, and computational methods for geospatial data.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_bayes` and the owning module's public contracts.

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
- `SKILL.md`
- `mcmc_traces.png`
- `mean_prediction.png`
- `posterior_distributions.png`
- `pyproject.toml`
- `spatial_data.png`
- `uncertainty.png`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module BAYES
```


## Visualization Guidance

- Validate finite aligned spatial arrays and confidence levels before plotting.
- Normalize single-axis layouts before indexing axes so optional uncertainty
  panels work for one and many spatial predictions.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
