# Excellence Standards & Code Review

## Code Review Checklist

### Functionality

- [ ] No mock or placeholder methods
- [ ] All functions fully implemented with real logic
- [ ] Proper error handling (specific exceptions, logging)
- [ ] Mathematical correctness validated with tests
- [ ] Data processing pipelines complete and tested
- [ ] Graceful degradation for optional dependencies

### Documentation

- [ ] Google-style docstrings for all public APIs
- [ ] Type hints for all parameters, returns, and attributes
- [ ] README.md updated if module behaviour changed
- [ ] AGENTS.md updated if key files or patterns changed
- [ ] Examples provided and verified to run
- [ ] Mathematical foundations documented with citations
- [ ] Language is concise and professional

### Integration

- [ ] Follows established module patterns
- [ ] Uses standardised data models (Pydantic at boundaries)
- [ ] Properly handles cross-module dependencies
- [ ] Uses H3 v4 API exclusively for spatial operations
- [ ] Respects layered architecture (Foundation → Data → Domain → App)

### Quality

- [ ] `ruff format` clean and `ruff check` reports 0 findings (root config)
- [ ] Touched modules meet their coverage floor (`GEO-INFER-TEST/coverage_baseline.json`)
- [ ] Library randomness threads a `seed`/`rng` through `utils/rng.py` `resolve_rng`
- [ ] Performance tested with realistic data volumes
- [ ] Security: no hardcoded secrets, inputs validated
- [ ] Structured logging (no `print()` statements)

## Code Formatting

Ruff is the only lint and format tool; its configuration lives once in the
root `pyproject.toml` (`[tool.ruff]` target `py311`, line length 88;
`[tool.ruff.lint]` selects `E4`, `E7`, `E9`, `F`, `UP`, `B`, `NPY`).

```bash
# Format
uv run --with 'ruff>=0.15.6,<0.16' ruff format GEO-INFER-MODULE/

# Lint
uv run --with 'ruff>=0.15.6,<0.16' ruff check GEO-INFER-MODULE/

# Optional type check against the root [tool.mypy] configuration
uv run mypy GEO-INFER-MODULE/src/
```

## Commit Message Conventions

```
<type>(<scope>): <description>

<body>
```

| Type | Usage |
|------|-------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or correcting tests |
| `chore` | Build process, tooling, dependencies |

Examples:

- `feat(RISK): implement spatial autocorrelation (Moran's I)`
- `fix(COMMS): resolve subscriber lookup returning empty list`
- `docs(AGENT): update AGENTS.md with telemetry patterns`

## Current Priorities

Planned work and its acceptance criteria are tracked in the root
[`TODO.md`](../TODO.md); do not duplicate status snapshots here.

## Release Checklist

Before tagging any version release:

- [ ] All tests pass: `uv run python GEO-INFER-TEST/run_unified_tests.py`
- [ ] 0 placeholder/stub implementations in source code
- [ ] 0 `pass` stubs (excluding `__init__.py`, `except`, abstract methods)
- [ ] `ruff format --check .` and the CI `ruff check` selections clean
- [ ] Every module meets its coverage floor (`check_coverage_floor.py`)
- [ ] Generated README.md + AGENTS.md current: `uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check`
- [ ] CHANGELOG.md entries for this version
- [ ] `pyproject.toml` version updated

---

Every line of code should reflect production-quality engineering: mathematical rigour, functional completeness, structured logging, and precise documentation. Use technical accuracy over promotional language.

## Dependency Floor Policy (2026-09-11, GS-025)

Shared geospatial/HTTP stack families must agree on one support surface across
the root pyproject and all module pyprojects:

- `shapely>=2.0.0` fleet-wide (1.x/2.x API split is behavioral; every consumer
  passes current 2.x tests). No `shapely>=1.*` declarations anywhere.
- `urllib3>=2.0.6` wherever `urllib3` is declared (matches root).
- `numpy` floors may vary per module (they reflect each module's own needs),
  but `<2.0` caps are permitted only with a documented incompatibility reason
  (today: BAYES, SPACE).
- `requests`/`uvicorn` floors may vary per module; they are transport-only and
  are not behaviorally version-split.

Rule: when raising or lowering a floor for a family above, change it in the
root pyproject and every module that declares it, or do not change it.
