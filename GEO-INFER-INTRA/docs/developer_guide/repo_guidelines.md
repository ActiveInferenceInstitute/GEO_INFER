# Repository Guidelines

These guidelines apply to source, tests, examples, documentation, and repository
automation.

## Scope and ownership

- The root is a uv workspace with 45 GEO-INFER-* modules.
- Importable behavior belongs under a module's src/ package.
- Tests belong to the owning module unless they validate a repository-wide
  contract, in which case they belong in GEO-INFER-TEST.
- Cross-module tutorials and architecture guidance belong in
  GEO-INFER-INTRA/docs/.

## Working safely

`bash
git status --short
uv sync --all-packages --all-extras --all-groups
uv run python -m pytest GEO-INFER-MODULE/tests/unit -q
`

Inspect diffs before staging. Preserve unrelated work in a shared checkout and
do not use destructive reset, checkout, or clean commands as a shortcut.

## Source and test rules

- Python target is 3.11+ and package imports use lowercase
  geo_infer_<module>.
- Use type hints and docstrings for public functions and classes.
- Validate finite numeric values, coordinate bounds, CRS, units, and output paths.
- Use deterministic local RNGs when a public operation is stochastic.
- Tests must assert observable behavior and real output invariants.
- Do not hide warnings, skips, missing dependencies, collection failures, or
  empty test selections.

## Documentation rules

- Examples use real imports from the current checkout.
- State the working directory, setup command, coordinate order, CRS, units,
  optional dependencies, output paths, and verification command.
- Keep module README/AGENTS files synchronized with the generator.
- Put future work in issues or TODO.md, not current-state API docs.
- Avoid introducing deprecated API names or unverifiable status claims.

## Validation

`bash
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language
uv run python GEO-INFER-TEST/validate_test_contracts.py --strict
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
git diff --check
`

Use the root README and CI workflow for the full unit, integration, performance,
model, source-hygiene, and H3 release gates.
