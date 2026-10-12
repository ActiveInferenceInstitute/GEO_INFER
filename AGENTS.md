# GEO-INFER Agent Instructions

Use this file as the repository-level operating contract for automated agents working in GEO-INFER.

## Cold Start (agent orientation)

New to this repo? In order:

1. **What this is**: a multi-module geospatial inference monorepo (uv workspace, Python 3.11+); see [README.md](README.md) "What's inside".
2. **Module map**: [README.md Module Index](README.md#module-index) (source/test counts per module) and [GEO-INFER-INTRA/docs/modules/index.md](GEO-INFER-INTRA/docs/modules/index.md) (conceptual catalog).
3. **Where to change what**: module behavior in `GEO-INFER-*/src/` (owning package); cross-module docs in `GEO-INFER-INTRA/docs/`; tests/validation in `GEO-INFER-TEST/`.
4. **Health in one command**: `uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check` (generated-signpost drift) plus the validators under Standard Commands.
5. **Backlog with acceptance lines**: [TODO.md](TODO.md).
6. **GNN companion pin**: to bump the paired-interchange GNN revision, follow
   the [pair-pin bump procedure](GEO-INFER-TEST/docs/gnn_space_time_2026_09.md#gnn-pair-pin-bump-procedure).

## Repository Scope

- Root path: repository root (`.`; resolve it from the checkout in use)
- Modules: `GEO-INFER-ACT`, `GEO-INFER-AG`, `GEO-INFER-AGENT`, `GEO-INFER-AI`, `GEO-INFER-ANT`, `GEO-INFER-API`, `GEO-INFER-APP`, `GEO-INFER-ART`, `GEO-INFER-BAYES`, `GEO-INFER-BIO`, `GEO-INFER-CIV`, `GEO-INFER-CLIMATE`, `GEO-INFER-COG`, `GEO-INFER-COMMS`, `GEO-INFER-DATA`, `GEO-INFER-ECON`, `GEO-INFER-EDU`, `GEO-INFER-EMERGENCY`, `GEO-INFER-ENERGY`, `GEO-INFER-EXAMPLES`, `GEO-INFER-FOREST`, `GEO-INFER-GIT`, `GEO-INFER-HEALTH`, `GEO-INFER-INSURANCE`, `GEO-INFER-INTRA`, `GEO-INFER-IOT`, `GEO-INFER-LOG`, `GEO-INFER-MARINE`, `GEO-INFER-MATH`, `GEO-INFER-METAGOV`, `GEO-INFER-NORMS`, `GEO-INFER-OPS`, `GEO-INFER-ORG`, `GEO-INFER-PEP`, `GEO-INFER-PLACE`, `GEO-INFER-REQ`, `GEO-INFER-RISK`, `GEO-INFER-SEC`, `GEO-INFER-SIM`, `GEO-INFER-SPACE`, `GEO-INFER-SPM`, `GEO-INFER-TEST`, `GEO-INFER-TIME`, `GEO-INFER-TRANSPORT`, `GEO-INFER-WATER`
- Package manager: `uv`
- Python target: 3.11+

## Required Workflow

1. Inspect the relevant module before editing.
2. Keep functionality in the owning module under `src/`.
3. Keep scripts and examples as thin orchestration surfaces.
4. Update README.md and AGENTS.md when behavior, commands, exports, or dependencies change.
5. Run the narrowest relevant test first, then the repo contract validators.

## Standard Commands

```bash
uv sync --all-packages --all-extras --all-groups
python -m compileall GEO-INFER-*/src GEO-INFER-*/examples
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language --strict-import-smoke
uv run python GEO-INFER-TEST/validate_logging_hygiene.py
uv run python GEO-INFER-TEST/validate_documentation.py --strict
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal
uv run python manuscript/generate_research_artifacts.py
uv run python GEO-INFER-TEST/run_unified_tests.py --category unit
uv run python GEO-INFER-TEST/run_unified_tests.py --category integration
uv run python GEO-INFER-TEST/run_unified_tests.py --category performance
uv run python GEO-INFER-TEST/run_unified_tests.py --h3-migration
uv run python GEO-INFER-TEST/validate_test_contracts.py --strict
uv run python GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42
uv run python GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible
uv run --with 'ruff>=0.15.6,<0.16' ruff check .
gitleaks detect --source . --config .gitleaks.toml --redact --verbose
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
```

On macOS 27, native PyMC checks require the explicit repository compiler
profile described in [SPM native sampling profiles](GEO-INFER-SPM/docs/native_sampling_profiles.md).
Prefix validation commands with
`PYTENSOR_FLAGS="cxx=$PWD/GEO-INFER-SPM/pytensor_clang++_macos27.py"`.
Keep native compilation enabled and retain the chosen profile in evidence.

## Modular Hygiene Contract

- Use root `pyproject.toml`, `uv.lock`, and `.python-version` as the shared uv environment contract.
- Preserve `uv.lock` LF bytes through root `.gitattributes`; receipt custody uses raw file hashes on every platform.
- Preserve captured Cascadia publisher bytes and LF replay outputs through the exact paths in root `.gitattributes`; do not normalize hash-bound archives.
- Sync the shared workspace with `uv sync --all-packages --all-extras --all-groups`.
- Keep module behavior in the owning `GEO-INFER-*` package under `src/`; keep scripts and examples as orchestration surfaces.
- Treat `manuscript/generate_research_artifacts.py` as the only producer of manuscript variables, figure captions, figure registries, and resolved manuscript copies; never hand-edit ignored `output/`.
- Keep every module's local test inventory above the minimum release gate of four pytest files.
- Put planned work in root `TODO.md` or a tracked issue; do not leave task markers in module source or tests.
- Use module loggers in libraries and configure handlers only from CLI entrypoints.

## Documentation Contract

Agent-facing documentation must be operational: current paths, commands, package names, public exports, test surfaces, and failure triage. Do not advertise planned APIs in AGENTS.md; use issues, roadmaps, or implementation status files for future work.

## Documentation Workflow

- Put conceptual, cross-module, and user-facing guidance in
  `GEO-INFER-INTRA/docs/`.
- Keep module READMEs focused on the current filesystem, imports, dependencies,
  and verification commands; they are generated from tracked repository facts.
- Keep `SKILL.md` action-oriented and synchronized with real public APIs.
- Validate relative documentation links and generated signposts before handoff.
- Preserve existing work from other agents in a shared checkout; inspect the
  diff before staging generated documentation.

## Documentation and Release Gate

- Run `uv run python GEO-INFER-TEST/rewrite_readme_agents.py` after changing tracked module files, public exports, tests, or validation commands.
- Treat the generated README/AGENTS diff as a review surface: it should reflect the intended source, test, dependency, and signpost changes only.
- Before integrating to `main`, run the strict repository, test, model, skill, source-hygiene, unit, integration, performance, and H3 gates and record any environment-only setup warnings separately from repository failures.
