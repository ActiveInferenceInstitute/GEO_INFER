# .github/actions/sync-cpu-workspace

Sync Cpu Workspace workspace within GEO-INFER.

## Contents

- `action.yml`

## Public Interface

- No public Python symbols are defined directly in this directory.


## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## Composite Action

- Installs every locked workspace member, extra, and dependency group on CPU
  runners; a cache hit never substitutes for the locked sync.
- Keep the sync command aligned with root `AGENTS.md` Standard Commands.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
