# Contributing to GEO-INFER

Thank you for improving GEO-INFER. This repository values small, source-backed
changes with explicit tests, reproducible commands, and documentation that
describes the current checkout.

## Before you start

Read:

- [Code of Conduct](../../../CODE_OF_CONDUCT.md)
- [Security policy](../../../SECURITY.md)
- [Repository architecture](../architecture/index.md)
- [Testing guide](testing_guide.md)
- [Documentation guide](../documentation_guide.md)

Use an issue for planned work and keep implementation claims out of generated
signposts until the code and tests exist.

## Development setup

```bash
git clone https://github.com/ActiveInferenceInstitute/GEO_INFER.git
cd GEO_INFER
uv sync --all-packages --all-extras --all-groups
```

Check the working tree before editing. In a shared checkout, preserve unrelated
changes and avoid destructive Git commands.

## Implement a change

1. Identify the owning `GEO-INFER-*` module.
2. Inspect its public exports, tests, `pyproject.toml`, README, and SKILL.
3. Add behavior under `src/`; keep examples and scripts as orchestration.
4. Add a focused behavior test and update the module test inventory if needed.
5. Update conceptual docs when the workflow, contract, or architecture changes.
6. Refresh generated signposts with:
   `uv run python GEO-INFER-TEST/rewrite_readme_agents.py`.
7. Inspect the diff and run the strict checks.

## Validation checklist

```bash
uv run python -m compileall GEO-INFER-*/src GEO-INFER-*/examples
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs
uv run python GEO-INFER-TEST/validate_test_contracts.py --strict
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
uv run python GEO-INFER-TEST/validate_documentation.py --strict
uv run python GEO-INFER-TEST/run_unified_tests.py --module MODULE
git diff --check
```

For cross-module or release work, run the unit, integration, performance, model,
source-hygiene, and H3 gates listed in the root README and CI workflow.

## Code conventions

- Python target: 3.11+; use the shared uv workspace.
- Package imports are lowercase `geo_infer_<module>`.
- Use type hints and Google-style docstrings for public behavior.
- Validate finite numeric inputs, coordinate bounds, CRS, units, and output paths
  at public boundaries.
- Use module loggers; configure handlers only in CLI entrypoints.
- Keep random state local and seedable.
- Do not add legacy H3 v3 calls or fake/mock/stub/placeholder behavior.

## Documentation conventions

- Prefer links to real repository files over invented services or endpoints.
- Mark code blocks with their language and make examples executable.
- State the working directory, install command, coordinate order, CRS, units,
  optional dependencies, output files, and validation command.
- Put future work in an issue or root TODO, not in current-state README claims.
- Run generated-doc parity checks after changes to source, exports, tests, or
  commands.

## Pull requests

Use a focused branch and describe:

- what changed and why;
- the owning module and public imports;
- tests and validation commands with outcomes;
- documentation and generated-signpost changes;
- any optional dependency or artifact implications.

Never claim that an issue is addressed on GitHub until the fix is committed,
pushed, and merged. The local worktree and the remote main branch are separate
states.
