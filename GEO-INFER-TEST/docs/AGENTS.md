# Agent Instructions: GEO-INFER-TEST/docs

## Scope

- Owning module: `GEO-INFER-TEST`
- Python package: `geo_infer_test`
- Directory role: Docs workspace within `GEO-INFER-TEST`.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through `geo_infer_test` and the owning module's public contracts.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `examples/`
- `api_reference.md`
- `benchmark_baseline_2026-09-10.md`
- `getting_started.md`
- `gnn_continuation_2026_09.md`
- `gnn_space_time_2026_09.md`
- `import_latency_2026_09.md`
- `index.md`
- `perf01-act-importtime.txt`
- `perf01-pandas-importtime.txt`
- `pin_review_2026-Q3.md`
- `secret_scan_policy.md`

## Validation

```bash
uv sync --all-packages --all-extras --all-groups
uv run python GEO-INFER-TEST/run_unified_tests.py --module TEST
```


## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
