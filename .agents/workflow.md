# Development Workflow

## Environment Management with uv

**All package management uses `uv`** against the root workspace
(`pyproject.toml`, `uv.lock`, `.python-version`):

```bash
# Synchronize the locked workspace with every package and extra
uv sync --all-packages --all-extras

# Synchronize one workspace package for a focused check
uv sync --package geo-infer-module

# Add a dependency to the owning module's pyproject.toml and refresh uv.lock
uv add --package geo-infer-module package-name

# Run a Python script or a module's tests
uv run python script.py
uv run python -m pytest GEO-INFER-MODULE/tests/
```

Dependencies are declared only in each module's `pyproject.toml` and resolved
in the root `uv.lock`; module-level `setup.py`, `setup.cfg` and
`requirements.txt` files are rejected by
`GEO-INFER-TEST/validate_repo_contracts.py`.

**Never use**: bare `pip install`, `python -m pip`, or `conda install`.

## Before Writing Code

1. **Understand the Module**: Read the module's README.md and AGENTS.md
2. **Check Dependencies**: Review `pyproject.toml` for module dependencies
3. **Review Examples**: Look at `examples/` for usage patterns
4. **Plan Integration**: Consider data flow with other modules
5. **Check TODO.md**: See if the work is tracked in the repository TODO

## While Writing Code

1. **Follow Existing Patterns**: Maintain consistency with existing code style
2. **Document as You Go**: Write docstrings and type hints simultaneously
3. **Test Incrementally**: Write tests for each function as you implement it
4. **Log Appropriately**: Use `logging.getLogger(__name__)` (see `implementation.md`)
5. **Validate Data**: Implement input validation and error handling
6. **No Placeholders**: Every method must have real, working logic

## After Writing Code

1. **Run Tests**: `uv run python -m pytest GEO-INFER-MODULE/tests/` — all must pass
2. **Check Coverage**: `uv run --with pytest-cov python GEO-INFER-TEST/check_coverage_floor.py --base main --head HEAD`
   — touched modules must stay at or above their floor in `GEO-INFER-TEST/coverage_baseline.json`
3. **Lint**: `uv run --with 'ruff>=0.15.6,<0.16' ruff check GEO-INFER-MODULE/`
4. **Format**: `uv run --with 'ruff>=0.15.6,<0.16' ruff format GEO-INFER-MODULE/`
5. **Update Docs**: Update hand-written docs and SKILL.md, then refresh generated
   README.md/AGENTS.md signposts with `uv run python GEO-INFER-TEST/rewrite_readme_agents.py`
6. **Create Examples**: Add working examples to `examples/`

Ruff is the only lint and format tool. It is configured once in the root
`pyproject.toml` (`[tool.ruff]`: target `py311`, line length 88;
`[tool.ruff.lint]`: `E4`, `E7`, `E9`, `F`, `UP`, `B`, `NPY`, with `NPY002`
allowed in tests, examples and scripts).

## CI/CD Pipeline

### GitHub Actions (`.github/workflows/ci.yml`)

`ci.yml` runs on pushes and pull requests to `main` and `develop`, on `v*`
tags, on manual dispatch, and weekly for the slow lane. The `validate`, `test`,
`manuscript` and `slow-scheduled` jobs install with
`uv sync --locked --all-packages --all-extras` (CUDA/native-only extras
excluded on the CPU runners).

| Job / step | Command | Gate |
|------|---------|------|
| `validate`: code quality (changed `*.py` only) | `ruff check` (root `[tool.ruff.lint]` contract) and `ruff format --check` | 0 findings, formatted |
| `validate`: repository contracts | `validate_repo_contracts.py --strict-source-language --strict-import-smoke`, `validate_packaging.py --strict`, `validate_logging_hygiene.py`, `validate_documentation.py --strict`, `validate_skills.py --check-xrefs --warnings-fatal`, `validate_active_inference_contract.py`, `rewrite_readme_agents.py --check` | all pass |
| `validate`: syntax and test contracts | `python -m compileall GEO-INFER-*/src GEO-INFER-*/examples`; `validate_test_contracts.py --strict` | all pass |
| `validate`: model contracts | `validate_model_contracts.py --strict --seed 42`; `run_model_audit.py --seed 42 --reproducible` | all pass |
| `validate`: source runtime hygiene | `ruff check .` (whole tree, root `[tool.ruff.lint]` contract) | 0 findings |
| `validate`: secrets | `gitleaks detect --source . --config .gitleaks.toml --redact --verbose` | 0 leaks |
| `validate`: coverage floors | `check_coverage_floor.py --base <base> --head <head>` | touched modules meet their recorded floor |
| `build-smoke` | `python GEO-INFER-TEST/build_package_wheels.py` | one wheel per module builds |
| `workflow-lint` | `actionlint` | 0 findings |
| `test` (Python 3.11 and 3.12) | `run_unified_tests.py --category {unit,integration,performance,system} --timeout 600`; `run_unified_tests.py --h3-migration` | all pass |
| `manuscript` | `manuscript/generate_research_artifacts.py --verify` and `--check`; `scripts/render_manuscript_pdf.py`; root `tests/` | all pass |
| `slow-scheduled` (weekly) | `run_unified_tests.py --category slow --timeout 600` | all pass |

All validator scripts live in `GEO-INFER-TEST/` and run through `uv run python`.
There is no CI mypy gate and no repository-wide percentage coverage threshold.
`.github/workflows/format-check.yml` runs `ruff format --check .` repository-wide
on a weekly schedule.

## Branch Strategy

- **`main`**: Protected, always deployable
- **`develop`**: Integration branch; CI runs on pushes and pull requests to it
- **`feature/*`**: Feature branches from `main`
- **`fix/*`**: Bug fix branches
- **`release/*`**: Release preparation branches

Workflow: `feature/add-spatial-stats` → PR → review → merge to `main`

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with changes (Keep a Changelog format)
3. Run full test suite: `uv run python GEO-INFER-TEST/run_unified_tests.py`
4. Verify the CI gates above pass on the release commit
5. Tag release: `git tag -a v0.X.0 -m "Release v0.X.0"`
6. Push: `git push origin main --tags`; `.github/workflows/release.yml` waits
   for a successful `ci.yml` run on the tagged commit before uploading wheels

## Semantic Versioning

All modules follow SemVer (`MAJOR.MINOR.PATCH`):

- **MAJOR**: Breaking API changes
- **MINOR**: New features that keep the public API
- **PATCH**: Bug fixes

The workspace version is declared in the root `pyproject.toml`.
