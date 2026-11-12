# Agent Instructions: research/measurement-quality

## Scope

- Owning module: none (repository-level directory outside the GEO-INFER-* module fleet)
- Directory role: Measurement Quality workspace within GEO-INFER.

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

- `baseline-scenarios.json`
- `fixture.json`

## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## Probe Custody

- `fixture.json` and `baseline-scenarios.json` are retained synthetic inputs;
  never hand-edit them or substitute observed data.
- The harness creates its output directory exclusively; use a new path per run
  and never commit runtime paths or generated result bundles.
- Keep claims at implementation-boundary strength: the fixture establishes no
  field benefit, ecological observation or strategy superiority.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
