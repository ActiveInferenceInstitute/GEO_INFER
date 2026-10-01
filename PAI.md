# PAI Integration with GEO-INFER

## Overview

GEO-INFER uses the PAI (Personal AI Infrastructure) Algorithm as a planning
and verification vocabulary. The algorithm is a development methodology, not a
runtime dependency of the Python packages:

1. **OBSERVE** — identify the request, repository state, and constraints.
2. **THINK** — test assumptions and identify risks.
3. **PLAN** — define scoped, verifiable criteria and an execution order.
4. **BUILD** — implement the source, tests, and documentation.
5. **EXECUTE** — run the relevant commands and integrations.
6. **VERIFY** — check every criterion against real tool output.
7. **LEARN** — record residual risks and reusable improvements.

## Applying the method

For a GEO-INFER change:

- Start with `git status --short` and inspect the owning module.
- Express completion as observable state (for example, “the SPACE H3 helper
  test passes”), not as an intention (“improve spatial indexing”).
- Keep implementation in `GEO-INFER-*/src/`, tests in the owning module, and
  conceptual guidance in `GEO-INFER-INTRA/docs/`.
- Run the narrowest relevant test first, then the repository contract and
  documentation validators.
- Record environment-only limitations separately from repository failures.

## Repository verification surfaces

```bash
uv sync --all-packages --all-extras --all-groups
uv run python GEO-INFER-TEST/run_unified_tests.py --module MODULE
uv run python GEO-INFER-TEST/validate_repo_contracts.py
uv run python GEO-INFER-TEST/validate_documentation.py --strict
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs
```

Replace `MODULE` with an uppercase module suffix such as `ACT`, `MATH`, or
`SPACE`. See [ISA.md](ISA.md) for the current ideal-state criteria and
[CONTRIBUTING.md](CONTRIBUTING.md) for the contribution contract.

## Ideal-state criteria

GEO-INFER uses criteria such as:

- All 45 workspace modules have the required generated signposts.
- Package directories use lowercase `geo_infer_<module>` names.
- The ACT typed result contracts and policy-selection behavior pass their
  executable contract validator.
- Documentation claims match current source exports and tracked paths.
- Planned work is recorded in `TODO.md` or a tracked issue, not left as source
  or test task markers.

These are verification targets. A dated report or previous pass must not be
read as current test evidence unless its commands are rerun.

## Active Inference grounding

The canonical implementation is `GEO-INFER-ACT`. Its current public contracts
and verification commands are described in `ISA.md` and the ACT module
README/SKILL. Other modules should call those contracts or document any
intentional equivalent semantics rather than copying an incompatible API.

## No-fabrication policy

Examples must use imports and symbols that exist in the current checkout. If a
page describes a conceptual or historical integration, it must say so clearly
and link readers to a source-backed module example.
