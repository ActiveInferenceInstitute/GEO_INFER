# Agent Instructions: manuscript/sections

## Scope

- Owning module: none (repository-level directory outside the GEO-INFER-* module fleet)
- Directory role: Sections workspace within GEO-INFER.

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

- `act.md`
- `ag.md`
- `agent.md`
- `ai.md`
- `ant.md`
- `api.md`
- `app.md`
- `art.md`
- `bayes.md`
- `bio.md`
- `civ.md`
- `climate.md`
- `cog.md`
- `comms.md`
- `data.md`
- `econ.md`
- `edu.md`
- `emergency.md`
- `energy.md`
- `examples.md`
- `forest.md`
- `git.md`
- `health.md`
- `insurance.md`
- `intra.md`
- `iot.md`
- `log.md`
- `marine.md`
- `math.md`
- `metagov.md`
- `norms.md`
- `ops.md`
- `org.md`
- `pep.md`
- `place.md`
- `req.md`
- `risk.md`
- `sec.md`
- `sim.md`
- `space.md`
- `spm.md`
- `test.md`
- `time.md`
- `transport.md`
- `water.md`

## Validation

```bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --skip-import-smoke
```


## Section Editing Rules

- Edit only the tracked section Markdown; the resolved manuscript is generated.
- After changing a section, run
  `uv run python manuscript/generate_research_artifacts.py` and commit
  `manuscript/config.yaml` separately when the source date moves.

## Integration Notes

- Update this AGENTS.md and the sibling README.md when commands, exports, dependencies, or generated outputs change.
- Keep cross-module references anchored to real package imports and tracked files.
