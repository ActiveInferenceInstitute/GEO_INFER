# Agent Instructions: .github

## Scope

- Owning module: none (repository-level directory outside the GEO-INFER-* module fleet)
- Directory role: .Github workspace within GEO-INFER.

## Capabilities

- Maintains the tracked files and subdirectories listed below for this workspace.
- Validates behavior with the command in the Validation section.
- Integrates through the repository's module-level public contracts; this directory defines no Python package.

## Working Rules

- Keep changes scoped to this directory unless an import, test, or documented command requires a coordinated edit.
- Prefer existing module patterns and public exports over new orchestration layers.
- Do not add planned, fake, mock, stub, or placeholder behavior to user-facing docs.
- If external services are involved, keep deterministic local validation available.

## Local Contents

- `ISSUE_TEMPLATE/`
- `actions/`
- `workflows/`
- `PULL_REQUEST_TEMPLATE.md`
- `companion-releases.json`
- `dependabot.yml`
- `gnn-pair.json`

## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## GitHub Configuration

- `README.md` here is the generated GitHub landing page, not a folder guide; edit
  the generator instead of this file.
- Workflows, composite actions, and issue templates live in the subdirectories
  listed above.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
