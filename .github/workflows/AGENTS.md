# Agent Instructions: .github/workflows

## Scope

- Owning module: none (repository-level directory outside the GEO-INFER-* module fleet)
- Directory role: Workflows workspace within GEO-INFER.

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

## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## Workflow Rules

- Validate edits with `actionlint`; do not weaken strict repository, test, model,
  or source-hygiene gates.
- Keep action pins on stable releases and update `.github/dependabot.yml` when the
  pinned action set changes.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
