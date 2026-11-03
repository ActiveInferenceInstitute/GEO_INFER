# .github/actions/provision-pytensor-blas

Provision Pytensor Blas workspace within GEO-INFER.

## Contents

- `action.yml`

## Public Interface

- No public Python symbols are defined directly in this directory.


## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## Composite Action

- Installs and selects the Linux reference BLAS runtime used by CI model checks.
- Change the action together with the workflow jobs that consume it, then re-run
  the model-contract gate.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
