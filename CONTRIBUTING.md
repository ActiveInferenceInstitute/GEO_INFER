# Contributing to GEO-INFER

Thank you for improving GEO-INFER. Contributions should be source-backed,
reproducible, and scoped to the module that owns the behavior.

## Requirements

- Python 3.11+
- Git
- `uv`
- Familiarity with pytest and the module's local README/SKILL

Start with the [documentation hub](GEO-INFER-INTRA/docs/index.md) and the
[developer guide](GEO-INFER-INTRA/docs/developer_guide/index.md).

## Setup

```bash
git clone https://github.com/ActiveInferenceInstitute/GEO_INFER.git
cd GEO_INFER
uv sync --all-packages --all-extras
uv run python -c "import geo_infer_space, geo_infer_act; print('workspace ready')"
```

## Implementation workflow

1. Inspect `git status --short` and the owning module before editing.
2. Keep importable behavior in `GEO-INFER-*/src/`.
3. Add a focused behavioral test under the owning module's `tests/`.
4. Keep examples and scripts as thin orchestration surfaces.
5. Update public exports, README/SKILL, conceptual docs, and generated signposts
   when behavior or commands change.
6. Thread a `seed`/`rng` through the module's `utils/rng.py` `resolve_rng`
   (a `numpy.random.Generator`; never the global `numpy.random` stream), and
   validate finite values, coordinate bounds, CRS, units, and output paths.
7. Declare dependencies only in the module's `pyproject.toml`
   (`uv add --package geo-infer-<module> <dep>` updates it and `uv.lock`);
   module-root `setup.py`, `setup.cfg` and `requirements.txt` are rejected.
8. Preserve unrelated changes when another agent is using the checkout.

## Focused validation

Replace `MODULE` with an uppercase module suffix such as `ACT`, `ANT`, or
`SPACE`:

```bash
uv run python -m pytest GEO-INFER-MODULE/tests/unit -q
uv run python GEO-INFER-TEST/run_unified_tests.py --module MODULE
uv run --with 'ruff>=0.15.6,<0.16' ruff check GEO-INFER-MODULE/
uv run --with 'ruff>=0.15.6,<0.16' ruff format --check GEO-INFER-MODULE/
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language --strict-import-smoke
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal
uv run python GEO-INFER-TEST/validate_test_contracts.py --strict
git diff --check
```

Ruff is the only lint and format tool; it is configured once in the root
`pyproject.toml` (`[tool.ruff]`, `[tool.ruff.lint]`).

Use the full command matrix in [README.md](README.md) and the CI definition in
[.github/workflows/ci.yml](.github/workflows/ci.yml) for cross-module or release
changes.

## Code conventions

- Import packages as lowercase `geo_infer_<module>`.
- Use type hints in modern syntax (`list[int]`, `X | None`) and Google-style
  docstrings for public behavior.
- Use module loggers; configure handlers only from CLI entrypoints.
- Use current H3 v4 names and preserve explicit latitude/longitude order.
- Do not add fake, mock, stub, placeholder, or legacy behavior to production
  paths or user-facing examples.
- Keep optional dependencies explicit and actionable when unavailable.

## Documentation conventions

Examples must use real imports and state prerequisites, coordinate systems,
units, output locations, and expected validation. Future work belongs in an
issue or root [TODO.md](TODO.md), not in current-state module signposts.

## Pull requests

A pull request should explain:

- the owning module and public import paths;
- the behavior and reason for the change;
- focused and repository-wide validation commands;
- documentation, generated signpost, dependency, and artifact changes;
- any remaining environment-only limitations.

Keep the change focused, inspect the final diff, and never claim that a GitHub
issue is closed until the fix is committed, pushed, and merged.
