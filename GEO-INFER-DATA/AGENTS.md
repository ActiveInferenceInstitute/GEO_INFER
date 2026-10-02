# Agent Instructions: GEO-INFER-DATA

## Scope

- Owning module: `GEO-INFER-DATA`
- Python package: `geo_infer_data`
- Directory role: Foundational data backbone providing ETL pipelines, storage optimization, and data quality assurance for geospatial datasets.

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

- `config/`
- `docs/`
- `examples/`
- `src/`
- `tests/`
- `provision_duckdb_spatial.py`
- `SKILL.md`
- `pyproject.toml`

## Validation

```bash
uv run python GEO-INFER-TEST/run_unified_tests.py --module DATA
```


## Data Boundary Contracts

- Persistent cache filenames are derived from SHA-256 digests of logical keys;
  never reconstruct cache paths directly from caller-provided strings.
- Cache timestamps are normalized to UTC, and `ttl=0` means immediate expiry.
- Large DataFrame compression uses in-memory Parquet via a file-like reader;
  preserve this round-trip behavior when changing serializers.
- Temporal validators accept both timezone-naive and timezone-aware pandas
  datetime columns without mixing comparison timezones.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
