# Agent Instructions: GEO-INFER-ECON

## Scope

- Owning module: `GEO-INFER-ECON`
- Python package: `geo_infer_econ`
- Directory role: Spatial economic modeling, market analysis, policy evaluation, and economic impact assessment with geospatial dimensions.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_econ` and the owning module's public contracts.

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
uv run python GEO-INFER-TEST/run_unified_tests.py --module ECON
```


## Visualization Guidance

- Validate finite nonempty chart inputs and use call-local figure saving; do not
  mutate process-wide matplotlib/seaborn style from a visualizer constructor.
- Keep dashboard output paths operational and format optional metrics safely.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
