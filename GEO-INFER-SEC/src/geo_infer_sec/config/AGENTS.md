# Agent Instructions: GEO-INFER-SEC/src/geo_infer_sec/config

## Scope

- Owning module: `GEO-INFER-SEC`
- Python package: `geo_infer_sec`
- Directory role: Config workspace within `GEO-INFER-SEC`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_sec` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `threat_indicators.yaml`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module SEC
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
