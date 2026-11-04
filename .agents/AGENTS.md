# Agent Instructions: .agents

## Scope

- Owning module: none (repository-level directory outside the GEO-INFER-* module fleet)
- Directory role: .Agents workspace within GEO-INFER.

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

- `documentation.md`
- `implementation.md`
- `integration.md`
- `navigation.md`
- `principles.md`
- `requirements.md`
- `standards.md`
- `structure.md`
- `testing.md`
- `workflow.md`

## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## Framework Rules

- These documents define framework-wide development rules referenced by the root
  `AGENTS.md`.
- Keep them operational and cross-module; module-specific contracts belong in
  each module's `AGENTS.md` and `SKILL.md`.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
