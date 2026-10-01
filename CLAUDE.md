# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this GEO-INFER repository.

## Project Overview

GEO-INFER is a 45-module geospatial inference framework implementing Active Inference principles for ecological, civic, and commercial applications. It is a Python monorepo using `uv` as the package manager, with Python 3.11+ required.

### Current repository inventory

- **45 modules**; source, test and generated-document counts are maintained in
  the root [README](README.md#current-repository-facts) by `rewrite_readme_agents.py`.
- Nested README/AGENTS signposts are generated from repository files.
- All package directories follow PEP 8 lowercase naming: `geo_infer_<module>` (including `geo_infer_forest`, `geo_infer_marine`, `geo_infer_energy`, `geo_infer_water`). Mixed-case directory normalization is complete.
- Repo contract checks live in `GEO-INFER-TEST/validate_repo_contracts.py`; source-language debt is reported by default and can be made fatal with `--strict-source-language`.
- The same contract validator also enforces root uv workspace hygiene, per-module test inventory, source/test task-marker hygiene, and library logging configuration.
- Every module has a minimum of 4 test files.

## Build & Development Commands

```bash
# Synchronize the shared workspace, all package extras and dependency groups
uv sync --all-packages --all-extras --all-groups

# Synchronize one workspace package with its test dependency group, for an
# isolated check (modules declare test-only deps in [dependency-groups] test)
uv sync --package geo-infer-ops --group test
```

## Testing

```bash
# Run unified test suite (all modules)
uv run python GEO-INFER-TEST/run_unified_tests.py

# Run tests for a specific module
uv run python GEO-INFER-TEST/run_unified_tests.py --module MATH

# Run by category (unit, integration, system, performance, coverage, all)
uv run python GEO-INFER-TEST/run_unified_tests.py --category integration
uv run python GEO-INFER-TEST/run_unified_tests.py --category system
uv run python GEO-INFER-TEST/run_unified_tests.py --category performance
uv run python GEO-INFER-TEST/run_unified_tests.py --category coverage --timeout 900

# Run tests directly with pytest for a single module
uv run python -m pytest GEO-INFER-MATH/tests/ -v

# Run a single test file
uv run python -m pytest GEO-INFER-MATH/tests/unit/test_spatial_statistics.py -v

# Run with coverage
uv run python -m pytest GEO-INFER-MATH/tests/ --cov=GEO-INFER-MATH/src --cov-report=html

# Validate repo-wide contracts and Active Inference API contracts (CI flags)
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language --strict-import-smoke
uv run python GEO-INFER-TEST/validate_documentation.py --strict
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal
uv run python GEO-INFER-TEST/validate_active_inference_contract.py
```

Pytest markers (declared in root `pyproject.toml` `[tool.pytest.ini_options]`, the source of truth): `slow`, `integration`, `unit`, `performance`, `system`, `core`, `geospatial`, `api`, `reporting`, `fast`, `model`, `reproducibility`, `artifact`, `spatial`.

## Code Quality

```bash
# Lint and format (Ruff is the only lint/format tool)
uv run --with 'ruff>=0.15.6,<0.16' ruff check GEO-INFER-MODULE/
uv run --with 'ruff>=0.15.6,<0.16' ruff format GEO-INFER-MODULE/

# Optional type check against the root [tool.mypy] configuration (not a CI gate)
uv run mypy GEO-INFER-MODULE/src/
```

Ruff is configured once in the root `pyproject.toml`: `[tool.ruff]` targets
`py311` with line length 88; `[tool.ruff.lint]` selects `E4`, `E7`, `E9`, `F`,
`UP`, `B` and `NPY`, with `NPY002` allowed in tests, examples and scripts.
Modules carry no Ruff, Black, isort or pydocstyle configuration. Use modern
typing (PEP 585/604: `list[int]`, `X | None`). CI runs `ruff check .` over the
whole tree and `ruff check` + `ruff format --check` on changed files; see
`.github/workflows/ci.yml` and root `AGENTS.md` "Standard Commands".

## Architecture

### Module Layout

Most modules use a structure similar to the following; inspect the owning
module before assuming an optional directory or export exists:

```text
GEO-INFER-MODULE/
├── src/geo_infer_module/
│   ├── __init__.py      # Public exports (module-specific)
│   ├── core/            # Core algorithms and logic
│   ├── models/          # Data models
│   ├── api/             # API endpoints/interfaces
│   └── utils/           # Helpers
├── tests/
│   ├── unit/
│   └── integration/
├── examples/            # Examples; verify each one before running
├── pyproject.toml       # Sole packaging + dependency declaration (dependencies locked in root uv.lock)
├── README.md            # Generated signpost (rewrite_readme_agents.py)
├── AGENTS.md            # Generated agent signpost (rewrite_readme_agents.py)
└── SKILL.md             # Hand-written Claude Code skill (auto-discovered)
```

### Module Themes

Matches the root README "Module Themes" module-for-module (45 modules):

- **Spatial & Place-based**: SPACE, PLACE, TIME, MARINE, WATER, FOREST, CLIMATE, ENERGY, TRANSPORT, EMERGENCY
- **Bayesian & Active Inference**: BAYES, SIM, SPM, COG, ACT, MATH
- **Agents & AI Orchestration**: AGENT, AG, AI, ANT, OPS, COMMS
- **Governance, Risk & Domain**: INSURANCE, METAGOV, NORMS, ECON, PEP, REQ, SEC, CIV, HEALTH, ORG, RISK
- **Data, API & Applications**: API, APP, DATA, IOT, ART, EDU
- **Infrastructure & Validation**: INTRA, TEST, LOG, GIT, EXAMPLES, BIO

### Data Flow

```text
Data Sources → DATA → SPACE/TIME → MATH/BAYES/ACT → AI/AGENT → Domain Modules → API/APP
```

Foundation modules (MATH) have no dependencies. Core modules (BAYES, ACT) depend on MATH. Infrastructure (DATA, SPACE, TIME) is consumed by analytics (AI) and domain modules (AG, HEALTH, etc.). API and APP are the top-level consumers.

### Key Technical Decisions

- **H3 v4**: H3-enabled runtime surfaces require `h3>=4.5.0,<5`; the
  lockfile currently resolves the latest official `4.5.0` release. Use
  `latlng_to_cell`, `cell_to_latlng`, `cell_to_boundary`, and explicit
  `[lng, lat]` GeoJSON conversion rather than legacy APIs.
- **Backend-agnostic pattern**: SPACE module uses a dispatcher/interface pattern for H3 vs SRAI backends.
- **Graceful degradation**: `__init__.py` files use `try/except` for optional dependency imports, with module-level `HAS_<DEP>` flags consumed by call sites.
- **Package directory casing**: All 45 modules use `geo_infer_<module>` (lowercase) naming.
- **Packaging**: `pyproject.toml` is the only packaging and dependency declaration per module, resolved by the root `uv.lock`; module-root `setup.py`, `setup.cfg` and `requirements.txt` are rejected by `validate_repo_contracts.py`.
- **Randomness**: Library code takes a `seed`/`rng` argument and resolves it through the module's `utils/rng.py` `resolve_rng`, which returns a `numpy.random.Generator` (`numpy.random.RandomState` is rejected); ruff `NPY002` forbids the global `numpy.random` stream outside tests, examples and scripts.
- **Real implementations only**: BAYES GaussianProcess uses Cholesky decomposition. Model comparison uses real LOO/WAIC/DIC/BIC/AIC. ACT free energy exposes typed breakdowns and uses `complexity - accuracy` for categorical variational free energy.
- **Active Inference contract**: `GEO-INFER-ACT` exports `FreeEnergyBreakdown`, `PolicyEvaluation`, and `ActiveInferenceStepResult`. `PolicySelector(selection_mode="deterministic")` selects the lowest expected free energy; stochastic selection is seedable.

## Critical Development Rules

These rules are from `.agents/` and apply to all modules:

1. **NO MOCK METHODS**: Never create placeholder, stub, or mock implementations. Every function must have real logic. Use proper error handling instead of `pass` or `NotImplementedError`.

2. **Active Inference First**: Ground implementations in Active Inference mathematical principles (free energy minimization, Bayesian inference, perception-action loops).

3. **Concise Professional Language**: Avoid unnecessary adjectives and marketing hyperbole ("advanced", "sophisticated", "comprehensive" when not adding value). Use precise, technical language.

4. **Type Hints Everywhere**: Full type annotations on all function parameters and return values.

5. **Module-specific agent guidance**: Individual modules maintain domain-specific workflows and contracts in their `AGENTS.md` and `SKILL.md` files. Check `GEO-INFER-MODULE/AGENTS.md` before working on a module.

6. **Docs track code**: Every code change must keep `SKILL.md` and hand-written docs in sync with the implementation, and refresh the generated `README.md`/`AGENTS.md` signposts with `uv run python GEO-INFER-TEST/rewrite_readme_agents.py` (CI runs it with `--check`).

7. **Modular hygiene is centralized**: Root `pyproject.toml`, `uv.lock`, and `.python-version` define the uv environment; planned work belongs in root `TODO.md` or an issue; module source and tests must not carry local task markers.

8. **Library logging is passive**: Importable modules should create loggers with `logging.getLogger(__name__)`. Configure process-wide handlers only in CLI entrypoints.

## Key Files & Resources

- `GEO-INFER-TEST/run_unified_tests.py` - Cross-module unified test runner
- `GEO-INFER-TEST/validate_repo_contracts.py` - Module inventory, signposting, casing, pyproject-only packaging, uv workspace hygiene, and source-language debt report
- `GEO-INFER-TEST/validate_active_inference_contract.py` - Executable ACT API contract check
- `GEO-INFER-INTRA/docs/` - Central documentation hub (guides, tutorials, integration docs)
- `GEO-INFER-EXAMPLES/examples/` - Working examples including module orchestrators
- `SKILL.md` - Root Claude Code skill (ecosystem overview)
- `GEO-INFER-*/SKILL.md` - Module-level Claude Code skills (45 files)
- `.agents/` - Framework-wide development rules and agent guidance
- `AGENTS.md` - Repository-level operating contract for automated agents (scope, required workflow, standard commands, hygiene and documentation contracts)
- `PAI.md` - PAI Algorithm integration and development methodology
- `ISA.md` - Current ideal-state criteria and verification targets
- `CLAUDE.md` - This file (Claude Code guidance)

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **GEO_INFER** (68334 symbols, 99008 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/GEO_INFER/context` | Codebase overview, check index freshness |
| `gitnexus://repo/GEO_INFER/clusters` | All functional areas |
| `gitnexus://repo/GEO_INFER/processes` | All execution flows |
| `gitnexus://repo/GEO_INFER/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
