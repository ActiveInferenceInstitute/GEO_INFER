# .github/workflows

Workflows workspace within GEO-INFER.

## Contents

- `backend-profiling.yml`
- `ci.yml`
- `companion-releases.yml`
- `container-profiles.yml`
- `dependency-audit.yml`
- `format-check.yml`
- `gnn-interchange.yml`
- `import-probes.yml`
- `place-download-windows.yml`
- `release.yml`

## Public Interface

- No public Python symbols are defined directly in this directory.


## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## CI Workflows

- Ten tracked workflows cover repository validation, model contracts, formatting,
  import probes, companion releases, container profiles, and wheel releases.
- Validate workflow syntax with `actionlint`; CI runs the `workflow-lint` job.
- Do not add a workflow that weakens the strict repository, test, model, or
  source-hygiene gates.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
