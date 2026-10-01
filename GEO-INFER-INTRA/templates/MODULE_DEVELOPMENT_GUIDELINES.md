---
title: "GEO-INFER Module Development Guidelines"
description: "Guidelines for developing GEO-INFER modules"
purpose: "Ensure consistency, quality, and proper integration of modules"
---

# GEO-INFER Module Development Guidelines

## Overview

This document describes how to develop a module within the GEO-INFER uv
workspace so that it integrates with the shared environment, CI gates and
generated documentation.

## Development Phases

### Phase 1: Planning

#### 1.1 Module Proposal

- Create a proposal document from [MODULE_PROPOSAL_TEMPLATE.md](./MODULE_PROPOSAL_TEMPLATE.md).
- Define module scope and boundaries.
- Identify dependencies and integration points.
- Estimate development time and resources.

#### 1.2 Requirements Analysis

- Document functional requirements.
- Identify data sources and formats.
- Define API interfaces.
- Plan integration with dependent modules.

#### 1.3 Design Review

- Review the proposal with framework maintainers.
- Validate dependencies and the integration approach.
- Confirm the module fits the framework architecture.
- Get approval to proceed.

### Phase 2: Infrastructure Setup

#### 2.1 Directory Structure

Create the following structure. Packaging is `pyproject.toml`-only: do not add
`setup.py`, `setup.cfg` or a module-level `requirements.txt`;
`GEO-INFER-TEST/validate_repo_contracts.py` rejects them.

```text
GEO-INFER-{MODULE}/
├── pyproject.toml              # Sole packaging + dependency declaration (required)
├── SKILL.md                    # Hand-written Claude Code skill (required)
├── README.md                   # Generated signpost (rewrite_readme_agents.py)
├── AGENTS.md                   # Generated agent signpost (rewrite_readme_agents.py)
├── config/                     # Configuration files (optional)
│   └── example.yaml
├── src/                        # Source code (required)
│   └── geo_infer_{module}/
│       ├── __init__.py         # Public exports and __version__
│       ├── core/               # Core functionality
│       │   ├── __init__.py
│       │   └── {main_classes}.py
│       ├── api/                # API interfaces (optional)
│       │   ├── __init__.py
│       │   └── {api_classes}.py
│       └── utils/              # Utility functions
│           ├── __init__.py
│           └── rng.py          # Shared resolve_rng helper, if the module draws randomness
├── tests/                      # Test suite (required, at least four test files)
│   ├── conftest.py
│   ├── unit/
│   │   └── test_{module}.py
│   └── integration/
│       └── test_{module}_integration.py
├── examples/                   # Working examples (thin orchestration)
│   └── basic_usage.py
└── docs/                       # Module-specific hand-written documentation
```

#### 2.2 Register the Workspace Member

1. Start `pyproject.toml` from [pyproject.toml.template](./pyproject.toml.template):
   `setuptools.build_meta` backend, PEP 639 license
   `license = "CC-BY-NC-SA-4.0"`, `requires-python = ">=3.11"`, the workspace
   version, and `[tool.setuptools.package-data]` globs.
2. The root `[tool.uv.workspace]` `members = ["GEO-INFER-*"]` glob picks up the
   new directory automatically.
3. Declare third-party dependencies under `[project.dependencies]`; declare
   workspace siblings in an optional extra and map them in `[tool.uv.sources]`
   with `{ workspace = true }`.
4. Refresh the lock and environment:

   ```bash
   uv lock
   uv sync --all-packages --all-extras
   ```

   Add later dependencies with `uv add --package geo-infer-{module} <dep>`.

Do not add `[tool.ruff]`, `[tool.pytest.ini_options]` or formatter sections to
the module `pyproject.toml`; lint, format and pytest settings live in the root
`pyproject.toml`.

#### 2.3 Documentation Surfaces

- `README.md` and `AGENTS.md` are generated from repository files by
  `uv run python GEO-INFER-TEST/rewrite_readme_agents.py`; do not edit them by
  hand.
- `SKILL.md` is hand-written: YAML front matter with `name` and `description`,
  plus `## Instructions`, `## Examples` and `## Guidelines` (with an
  `### Integrations` subsection). Validate with
  `uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal`.
- Conceptual guides and integration notes go in the module `docs/` or in
  `GEO-INFER-INTRA/docs/`.

### Phase 3: Core Implementation

#### 3.1 Code Standards

- **Type hints**: every parameter and return value, in modern syntax
  (`list[int]`, `dict[str, Any]`, `X | None`).
- **Docstrings**: Google-style docstrings for all public classes and functions.
- **Error handling**: specific exceptions; no bare `except`.
- **No placeholders**: no `pass`-only bodies, `NotImplementedError`, mocks or
  stubs in production code.
- **Logging**: `logger = logging.getLogger(__name__)`; configure handlers only
  in CLI entrypoints.
- **Randomness**: accept a `seed`/`rng` argument and resolve it with the
  module's `utils/rng.py` `resolve_rng` (returns `numpy.random.Generator`;
  `numpy.random.RandomState` is rejected). Never draw from the global
  `numpy.random` stream in `src/`.
- **Optional dependencies**: guard imports with `try/except ImportError` and
  expose a `HAS_<DEP>` flag.

#### 3.2 Testing Requirements

- **Unit tests** for all core functionality.
- **Integration tests** for interactions with dependent modules.
- **Coverage**: once the module is recorded in
  `GEO-INFER-TEST/coverage_baseline.json`, CI fails a change that drops it
  below its floor.
- **Markers**: use only markers declared in the root `pyproject.toml`; pytest
  runs with `--strict-markers` and `-W error`.

Example test:

```python
# tests/unit/test_{module}.py
from geo_infer_{module} import {MainClass}


class Test{MainClass}:
    def test_{method}_returns_expected_value(self) -> None:
        instance = {MainClass}()
        result = instance.{method}({known_input})
        assert result == {expected_value}
```

#### 3.3 Examples

- Create runnable examples in `examples/`; keep them thin orchestration over
  `src/` APIs.
- Examples import the installed package (`import geo_infer_{module}`); do not
  add `sys.path` manipulation.
- Include integration examples with other modules where relevant.

### Phase 4: Documentation

- Document all public classes and functions in docstrings and `SKILL.md`.
- Show integration patterns with dependent modules.
- Add the module to [GEO-INFER-INTRA/docs/modules/index.md](../docs/modules/index.md).
- Add the module to the theme list in
  `GEO-INFER-TEST/rewrite_readme_agents.py` so the root README "Module Themes"
  table includes it, then regenerate signposts.

### Phase 5: Integration and Validation

Run the focused tests, lint and repository validators before review:

```bash
uv run python -m pytest GEO-INFER-{MODULE}/tests/
uv run --with 'ruff>=0.15.6,<0.16' ruff check GEO-INFER-{MODULE}/
uv run --with 'ruff>=0.15.6,<0.16' ruff format --check GEO-INFER-{MODULE}/
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language --strict-import-smoke
uv run python GEO-INFER-TEST/validate_packaging.py --strict
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal
uv run python GEO-INFER-TEST/validate_test_contracts.py --strict
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
```

The full CI gate list is in `.github/workflows/ci.yml` and root `AGENTS.md`
"Standard Commands".

## Code Quality Standards

- Ruff is the only lint and format tool, configured in the root
  `pyproject.toml` (`[tool.ruff]` target `py311`, line length 88;
  `[tool.ruff.lint]` selects `E4`, `E7`, `E9`, `F`, `UP`, `B`, `NPY`).
- Type hints are required for all functions.
- Docstrings document parameters, return values, raised exceptions and
  non-obvious algorithms.

## Integration Checklist

Before marking a module as complete:

- [ ] `pyproject.toml` is the only packaging file; module is a workspace member
      and `uv.lock` is current
- [ ] `SKILL.md` passes `validate_skills.py --check-xrefs --warnings-fatal`
- [ ] Generated `README.md`/`AGENTS.md` are current (`rewrite_readme_agents.py --check`)
- [ ] At least four test files; tests pass under the root pytest configuration
- [ ] `ruff check` and `ruff format --check` are clean
- [ ] Working examples in `examples/`
- [ ] Integration examples with dependent modules
- [ ] Module listed in `GEO-INFER-INTRA/docs/modules/index.md` and the README theme list
- [ ] Code and documentation review completed

## Common Pitfalls to Avoid

1. **Placeholder code**: never use `pass` or `NotImplementedError` in production code.
2. **Mock methods**: implement real functionality.
3. **Missing tests**: every module needs at least four test files.
4. **Second dependency declaration**: `setup.py` or `requirements.txt` copies drift
   from `pyproject.toml` and are rejected.
5. **Global randomness**: `np.random.*` in `src/` fails ruff `NPY002`.
6. **Hardcoded values**: use configuration files or parameters.
7. **Hand-edited signposts**: generated `README.md`/`AGENTS.md` edits are overwritten.

## Resources

- [Module Proposal Template](./MODULE_PROPOSAL_TEMPLATE.md)
- [pyproject.toml template](./pyproject.toml.template)
- [Documentation Standards](../docs/DOCUMENTATION_STANDARDS.md)
- [Module Integration Guide](../docs/guides/MODULE_INTEGRATION_GUIDE.md)
- [GEO-INFER Framework README](../../README.md)
