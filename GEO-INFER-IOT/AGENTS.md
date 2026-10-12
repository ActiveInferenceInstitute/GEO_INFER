# Agent Instructions: GEO-INFER-IOT

## Scope

- Owning module: `GEO-INFER-IOT`
- Python package: `geo_infer_iot`
- Directory role: IoT sensor networks, real-time geospatial data streams, and sensor data fusion for environmental monitoring.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_iot` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `config/`
- `deployment/`
- `docs/`
- `examples/`
- `src/`
- `tests/`
- `SKILL.md`
- `pyproject.toml`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module IOT
```


## Visualization Guidance

- Validate sensor/interpolation coordinates and aligned values before passing
  data to Folium or Matplotlib; create parents before writing artifacts.
- Close saved figures and preserve explicit error dictionaries for invalid input.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
