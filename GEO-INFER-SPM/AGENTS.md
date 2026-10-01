# Agent Instructions: GEO-INFER-SPM

## Scope

- Owning module: `GEO-INFER-SPM`
- Python package: `geo_infer_spm`
- Directory role: Statistical parametric mapping methodology adapted for geospatial analysis to identify significant patterns in spatial-temporal data.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_spm` and the owning module's public contracts.

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
- `pyproject.toml`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPM
```


## Random Field Theory Guidance

- Preserve every Gaussian EC density and boundary resel term in peak inference;
  do not replace the full expected EC with only its top-dimensional term.
- Keep cluster extent in resel units, correct the maximum-cluster probability,
  and verify peak and cluster FWE with deterministic known-null simulations.

## Visualization Guidance

- Reject invalid contrast/statistic/coordinate inputs before map construction.
- Keep diagnostic leverage calculations numerically stable and maintain one
  canonical package-level interactive-map export.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
