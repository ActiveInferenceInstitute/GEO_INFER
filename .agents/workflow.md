# Development Workflow

## Environment Management with uv

**All package management uses `uv`**:

```bash
# Install a module in editable mode
uv pip install -e ./GEO-INFER-MODULE

# Install a dependency
uv pip install package-name

# Synchronize the locked workspace (pyproject.toml + uv.lock)
uv sync --all-packages --all-extras

# Run a Python script
uv run python script.py

# Run tests
uv run pytest tests/
```

**Never use**: bare `pip install`, `python -m pip`, or `conda install`.

**Rationale**: `uv` provides faster, reproducible package management and is the standard tool for GEO-INFER.

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

1. **Run Tests**: `uv run pytest tests/` — all must pass
2. **Check Coverage**: `uv run pytest --cov` — must meet threshold
3. **Format Code**: `black . && isort .`
4. **Lint**: `ruff check --fix .`
5. **Update Docs**: Update README.md, AGENTS.md, and docstrings
6. **Create Examples**: Add working examples to `examples/`

## CI/CD Pipeline

### GitHub Actions (`.github/workflows/ci.yml`)

The CI pipeline runs on every pull request:

| Step | Command | Gate |
|------|---------|------|
| Install | `uv pip install -e .` | — |
| Lint | `ruff check .` | 0 errors |
| Format | `black --check .` | All formatted |
| Type Check | `mypy --strict src/` (core modules) | 0 errors |
| Tests | `uv run pytest tests/ --cov` | ≥80% coverage |

### Pre-commit Hooks

Configure `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks: [{ id: black }]
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks: [{ id: isort }]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks: [{ id: ruff, args: [--fix] }]
```

## Branch Strategy

- **`main`**: Protected, always deployable
- **`feature/*`**: Feature branches from `main`
- **`fix/*`**: Bug fix branches
- **`release/*`**: Release preparation branches

Workflow: `feature/add-spatial-stats` → PR → review → merge to `main`

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with changes (Keep a Changelog format)
3. Update `TODO.md` progress metrics
4. Run full test suite: `uv run python GEO-INFER-TEST/run_unified_tests.py`
5. Verify all release gates pass (see `TODO.md` Release Criteria)
6. Tag release: `git tag -a v0.X.0 -m "Release v0.X.0"`
7. Push: `git push origin main --tags`

## Semantic Versioning

All modules follow SemVer (`MAJOR.MINOR.PATCH`):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, backward-compatible

Current version: **0.2.0** (Beta)
