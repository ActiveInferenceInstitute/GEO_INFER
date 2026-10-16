# GEO-INFER Documentation

This is the documentation hub for the GEO-INFER repository. GEO-INFER is a
uv-managed Python 3.11+ monorepo containing 45 modules for spatial indexing,
probabilistic inference, Active Inference, domain analysis, applications, and
repository validation.

The documentation has three layers:

1. **Repository guidance** in the root `README.md`, `AGENTS.md`,
   `CONTRIBUTING.md`, `SECURITY.md`, and `CHANGELOG.md`.
2. **Conceptual and workflow guidance** in this `GEO-INFER-INTRA/docs/` hub.
3. **Executable module guidance** in each module's `README.md`, `SKILL.md`,
   `examples/`, `tests/`, and public source exports.

If a page conflicts with importable code or a passing contract validator, treat
the source and validator as authoritative and open a documentation issue.

## Start here

- [Repository overview](overview.md) — architecture, module layers, and data flow.
- [Installation](getting_started/installation_guide.md) — reproducible uv setup.
- [First analysis](getting_started/first_analysis.md) — a runnable H3 plus Active Inference workflow.
- [Active Inference basics](getting_started/active_inference_basics.md) — categorical beliefs, free energy, and policy selection.
- [Research-grade inference contracts](research_grade_inference_contracts.md) — executable ACT, BAYES, and RISK behavior, uncertainty, and verification.
- [H3 v4 guide](geospatial/data_formats/h3/index.md) — current H3 naming and hierarchy contracts.

## Choose a path

### Users and analysts

- [Getting Started](getting_started/index.md)
- [Examples](examples/README.md)
- [User guide](user_guide/index.md)
- [Spatial concepts](geospatial/concepts/index.md)
- [Temporal analysis](temporal_analysis_guide.md)
- [Support and troubleshooting](support/index.md)

### Developers and reviewers

- [Developer guide](developer_guide/index.md)
- [Testing guide](developer_guide/testing_guide.md)
- [Documentation guide](documentation_guide.md)
- [Architecture](architecture/index.md)
- [Module catalog](modules/index.md)
- [Integration guide](integration/index.md)
- [Contributing](../../CONTRIBUTING.md)
- [Security policy](../../SECURITY.md)

### Researchers and model builders

- [Active Inference guide](active_inference_guide.md)
- [Bayesian inference guide](bayesian_inference_guide.md)
- [Advanced model guidance](advanced/index.md)
- [Workflow patterns](workflows/index.md)
- [Data dictionary](data_dictionary.md)
- [Terminology](terminology.md)

## Operational references

- [GEO-INFER-TEST documentation](../../GEO-INFER-TEST/docs/index.md)
- [Unified test runner API](../../GEO-INFER-TEST/docs/api_reference.md)
- [Module READMEs](modules/index.md)
- [API documentation](api/index.md)
- [Deployment and environments](deployment/index.md)
- [Security guidance](security/index.md)
- [Release history](../../CHANGELOG.md)
- [Release readiness and acceptance](releases/0.4.0_readiness.md)
- [Release TODO and evidence ledger](../../TODO.md)

## Documentation rules

- Commands use `uv run` from the repository root unless a page explicitly says
  otherwise.
- Examples must import real symbols from the current package exports and must
  state coordinate systems, units, and expected artifact locations.
- H3 examples use the v4 API (`latlng_to_cell`, `cell_to_latlng`,
  `grid_disk`, and related names); v3 names are not supported.
- Test and model claims must be backed by a current command and should not be
  copied from historical assessment artifacts.
- Generated `README.md` and `AGENTS.md` files are refreshed by
  `GEO-INFER-TEST/rewrite_readme_agents.py`; edit that generator when the
  generated documentation contract changes.

## Documentation validation

From the repository root:

```bash
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
uv run python GEO-INFER-TEST/validate_documentation.py --strict
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs
git diff --check
```

For a focused change, run the affected module gate after the documentation
checks. Keep generated output under `.geo-infer-test-results/` or an explicit
temporary directory; do not commit repository-root runtime artifacts.
