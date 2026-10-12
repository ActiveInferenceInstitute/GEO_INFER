# Developer Guide

This guide describes how to change GEO-INFER safely in a shared uv workspace.
The repository is a monorepo: source and tests belong to a module, while
cross-module policy, architecture, and user guidance belong in
`GEO-INFER-INTRA/docs/`.

## Developer path

1. [Installation](../getting_started/installation_guide.md)
2. [Repository architecture](../architecture/index.md)
3. [Code structure](code_structure.md)
4. [Testing guide](testing_guide.md)
5. [Documentation guide](../documentation_guide.md)
6. [Contributing](contributing.md)

## Change ownership

| Change | Source of truth | Required follow-up |
| --- | --- | --- |
| Module behavior | `GEO-INFER-*/src/` | Focused test and module docs |
| Public export | module `__init__.py` | README/SKILL and import validation |
| CLI or validator | module script or `GEO-INFER-TEST/` | Command docs and contract test |
| Cross-module contract | owning module plus INTRA docs | Integration test and architecture note |
| Generated signpost | `rewrite_readme_agents.py` | Regenerate and run `--check` |
| CI behavior | `.github/workflows/ci.yml` | README/AGENTS command matrix |

## Standard loop

```bash
# inspect before editing
git status --short
uv sync --all-packages --all-extras --all-groups

# run the narrowest check first
uv run python -m pytest GEO-INFER-MODULE/tests/unit -q

# then validate contracts and documentation
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language
uv run python GEO-INFER-TEST/validate_test_contracts.py --strict
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
git diff --check
```

Replace `MODULE` with the uppercase module suffix, for example `ACT` or
`SPACE`. Use `GEO-INFER-TEST/run_unified_tests.py --module MODULE` when the
module's complete gate is required.

## Shared-worktree safety

Other agents may be editing the same checkout. Before changing a file:

- inspect its diff and preserve unrelated hunks;
- avoid reset, checkout, clean, or broad formatting commands;
- keep generated documentation changes explainable;
- do not stage or publish another agent's work without explicit authorization.

## Quality gates

The full CI-equivalent sequence is documented in the root
[README.md](../../../README.md) and executed by
[.github/workflows/ci.yml](../../../.github/workflows/ci.yml). The canonical
gates cover repository contracts, skills, generated docs, syntax, test
contracts, model determinism, source runtime hygiene, unit/integration/
performance suites, and H3 migration.

## Common review questions

- Does the public import path exist and appear in `__all__` where appropriate?
- Are input bounds, finite values, CRS, units, and output schemas documented?
- Does the test assert behavior rather than only object existence?
- Are optional dependencies explicit, and are missing backends actionable?
- Does the change preserve output-directory isolation and deterministic seeds?
- Are generated docs current without hiding source or test changes?
