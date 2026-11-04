# .github/ISSUE_TEMPLATE

Issue Template workspace within GEO-INFER.

## Contents

- `bug_report.md`
- `documentation.md`
- `feature_request.md`

## Public Interface

- No public Python symbols are defined directly in this directory.


## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## Issue Templates

- Three templates structure bug, feature, and documentation reports.
- Keep required fields aligned with commands and diagnostics that exist in this
  checkout.

## Documentation Notes

This README describes current repository state only. Keep examples and claims tied to importable code, tracked files, or validation commands.
