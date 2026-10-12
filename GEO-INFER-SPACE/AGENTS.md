# Agent Instructions: GEO-INFER-SPACE

## Scope

- Owning module: `GEO-INFER-SPACE`
- Python package: `geo_infer_space`
- Directory role: H3 v4 spatial indexing and comprehensive geospatial analysis framework with advanced spatial methods and coordinate transformations.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_space` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `docs/`
- `examples/`
- `output/`
- `reports/`
- `src/`
- `test_output/`
- `tests/`
- `.gitignore`
- `SKILL.md`
- `pyproject.toml`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SPACE
```


## Current Nested H3 Contracts

- Default H3 dispatch loads only H3. SRAI initializes on an explicit request,
  default selection, or full backend inspection. Missing SRAI is reported by
  its interface; broken installed dependencies propagate when it is loaded.
- `dispatcher.backends` contains realized implementations. Use
  `get_available_backends()` or `get_backend_info()` for full discovery.


- `NestedH3Grid` owns H3 parent/child closure, validation, same-resolution
  neighbor maps, and child-to-parent aggregation.
- Build hierarchies with ordered real `h3>=4.5.0,<5` resolutions and
  deterministic cell ordering; do not pass synthetic cell IDs to nested H3
  paths.
- Validate orphan counts, parent/child membership, resolution matches, and
  finite aggregate values before handing hierarchies to ACT.

## Failure Triage

- If hierarchy validation fails, inspect
  `geo_infer_space.nested.core.nested_grid.NestedH3Grid` first.
- If ACT nested contracts fail after SPACE edits, run the SPACE nested unit test
  and then `uv run python GEO-INFER-TEST/validate_h3_active_inference_contract.py`.

## Visualization Guidance

- Validate H3 resolution, bounds, dashboard result mappings, and user-facing map
  options before building a dashboard.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
