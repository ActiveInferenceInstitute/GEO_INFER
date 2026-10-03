# Agent Instructions: GEO-INFER-DATA/src/geo_infer_data/utils

## Scope

- Owning module: `GEO-INFER-DATA`
- Python package: `geo_infer_data`
- Directory role: Utils workspace within `GEO-INFER-DATA`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_data` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `_duckdb_extension_worker.py`
- `caching.py`
- `compression.py`
- `dependencies.py`
- `duckdb_spatial.py`
- `format_detection.py`
- `identifiers.py`
- `indexing.py`
- `performance.py`
- `secure_serialization.py`
- `timestamps.py`
- `validation.py`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module DATA
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
