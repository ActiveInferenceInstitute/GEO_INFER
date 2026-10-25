# GEO-INFER Framework

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](../pyproject.toml)
[![uv workspaces](https://img.shields.io/badge/uv-workspace-4C65F6?logo=astral&logoColor=white)](../pyproject.toml)
[![CI](https://img.shields.io/github/actions/workflow/status/ActiveInferenceInstitute/GEO_INFER/ci.yml?branch=main&label=CI)](../.github/workflows/ci.yml)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](../LICENSE)
[![Active Inference Institute](https://img.shields.io/badge/Active_Inference_Institute-6C3483?style=flat)](https://activeinference.org)

**GEO-INFER** connects geospatial data, spatial and temporal analysis, and
Active Inference in one Python workspace. Its 45 modules cover H3
indexing, Bayesian models, agents, and domain workflows for ecological, civic,
and commercial applications. Developed by the
[Active Inference Institute](https://activeinference.org).

[Get started](#quick-start) · [Documentation](../GEO-INFER-INTRA/docs/index.md) ·
[Module index](#module-index) · [Contribute](../CONTRIBUTING.md) ·
[Open work](../TODO.md)

## Start with a workflow

| Goal | Modules | Starting point |
| --- | --- | --- |
| Index locations and align observations over time | SPACE, TIME, DATA | [First spatial analysis](../GEO-INFER-INTRA/docs/getting_started/first_analysis.md) |
| Infer hidden states and evaluate actions | ACT, BAYES, MATH | [Active Inference guide](../GEO-INFER-INTRA/docs/active_inference_guide.md) |
| Analyze places and regional indicators | PLACE, SPACE, domain modules | [PLACE documentation](../GEO-INFER-PLACE/README.md) |
| Compose applications and agents | EXAMPLES, API, APP, AGENT | [Integration guide](../GEO-INFER-INTRA/docs/integration/index.md) |

Agent automation: [script, CLI, MCP and skills](../GEO-INFER-INTRA/docs/integration/agent_interfaces.md)
share discovery and execution across all 45 modules. After a full
workspace sync, use `uv run --frozen geo-infer modules` and
`uv run --frozen geo-infer catalog SPACE`.

Companion releases: [GNN 5.0.1](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v5.0.1)
and [FEP_Formal 2.1.0](https://github.com/ActiveInferenceInstitute/fep_formal/releases/tag/v2.1.0).
The [compatibility guide](../GEO-INFER-INTRA/docs/integration/companion_releases.md)
describes exact source pins, supported interchange contracts and separate proof boundaries.

## Quick Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone
the repository and start with one workspace package:

```bash
git clone https://github.com/ActiveInferenceInstitute/GEO_INFER.git
cd GEO_INFER
uv sync --package geo-infer-space
uv run --package geo-infer-space python - <<'PY'
from geo_infer_space import cell_to_latlng, latlng_to_cell

cell = latlng_to_cell(41.7558, -124.2026, resolution=9)  # Crescent City, CA
print(cell)
print(cell_to_latlng(cell))  # Cell center as (latitude, longitude)
PY
```

Coordinates are WGS84 degrees, in latitude/longitude order. Continue with the
[SPACE → ACT walkthrough](../GEO-INFER-INTRA/docs/getting_started/first_analysis.md)
to attach beliefs, actions, and free-energy terms to a cell. For cross-module
development, use the [full workspace profile](#choose-an-installation-profile).

Python 3.11+ is supported; `.python-version` pins the local development runtime.
The packages are under active development. Module tests describe implemented
behavior; [release readiness](../GEO-INFER-INTRA/docs/releases/0.4.0_readiness.md)
and the [issue ledger](../GEO-INFER-INTRA/docs/releases/issue_reconciliation_2026_10.md)
record evidence and outstanding acceptance requirements.

## What's inside

- 🧭 **Spatial & place-based analysis** — geospatial data, H3 grids, place & time modeling, and Earth-system domains (water, marine, forest, climate, energy, transport, emergency).
- 🧠 **Active Inference & Bayesian modeling** — Active-Inference agents and Bayesian models (Bayes, simulation, SPM, cognition, math).
- 🤖 **Agent & AI orchestration** — agent workflows, AI/LLM integration, communications, and operations.
- 🏛️ **Governance, risk & domain modeling** — risk, meta-governance, norms, economics, policy, security, health, and civil domains.
- 🗄️ **Data, API & applications** — data pipelines, APIs, applications, IoT, art, and education.
- 🛠️ **Infrastructure & validation** — documentation hub (INTRA), the validation & test harness, logging, git, examples, and bio.

## Documentation Map

- User documentation hub: [`GEO-INFER-INTRA/docs/index.md`](../GEO-INFER-INTRA/docs/index.md)
- Installation and first workflow: [`GEO-INFER-INTRA/docs/getting_started/index.md`](../GEO-INFER-INTRA/docs/getting_started/index.md)
- Framework architecture: [`GEO-INFER-INTRA/docs/overview.md`](../GEO-INFER-INTRA/docs/overview.md)
- Module catalog: [`GEO-INFER-INTRA/docs/modules/index.md`](../GEO-INFER-INTRA/docs/modules/index.md)
- Developer workflow: [`GEO-INFER-INTRA/docs/developer_guide/index.md`](../GEO-INFER-INTRA/docs/developer_guide/index.md)
- Test and release gates: [`GEO-INFER-TEST/docs/index.md`](../GEO-INFER-TEST/docs/index.md)
- Manuscript generation and evidence: [`manuscript/README.md`](../manuscript/README.md)
- Reusable figures and artifact verification: [`research artifacts`](../GEO-INFER-INTRA/docs/integration/research_artifacts.md)
- Active Inference reference: [`GEO-INFER-INTRA/docs/active_inference_guide.md`](../GEO-INFER-INTRA/docs/active_inference_guide.md)
- Spatial/H3 reference: [`GEO-INFER-INTRA/docs/geospatial/data_formats/h3/index.md`](../GEO-INFER-INTRA/docs/geospatial/data_formats/h3/index.md)
- Contribution rules: [`CONTRIBUTING.md`](../CONTRIBUTING.md)
- Security reporting: [`SECURITY.md`](../SECURITY.md)
- Release history: [`CHANGELOG.md`](../CHANGELOG.md)

Conceptual tutorials, integration guidance, and policy live in the INTRA
documentation hub. Each module owns its implementation and tests. The root
README, GitHub entrypoint, and module signposts are generated together from
`GEO-INFER-TEST/rewrite_readme_agents.py`.

## Choose an Installation Profile

The repository is a uv workspace. Use the full sync when working across module
boundaries, or sync a single package when developing one module:

```bash
uv sync --all-packages --all-extras --all-groups
uv sync --package geo-infer-act
uv sync --package geo-infer-space
uv sync --package geo-infer-ant
```

`--all-extras` installs optional scientific, Bayesian, web, IoT, performance,
quality, and documentation dependencies. CI intentionally omits native-only
extras that cannot build on its CPU runner; see `.github/workflows/ci.yml` for
the exact reproducible exception list.

## Module Themes

| Theme | Modules |
| --- | --- |
| 🌍 Spatial & Place-based | [GEO-INFER-SPACE](../GEO-INFER-SPACE/), [GEO-INFER-PLACE](../GEO-INFER-PLACE/), [GEO-INFER-TIME](../GEO-INFER-TIME/), [GEO-INFER-MARINE](../GEO-INFER-MARINE/), [GEO-INFER-WATER](../GEO-INFER-WATER/), [GEO-INFER-FOREST](../GEO-INFER-FOREST/), [GEO-INFER-CLIMATE](../GEO-INFER-CLIMATE/), [GEO-INFER-ENERGY](../GEO-INFER-ENERGY/), [GEO-INFER-TRANSPORT](../GEO-INFER-TRANSPORT/), [GEO-INFER-EMERGENCY](../GEO-INFER-EMERGENCY/) |
| 🧠 Bayesian & Active Inference | [GEO-INFER-BAYES](../GEO-INFER-BAYES/), [GEO-INFER-SIM](../GEO-INFER-SIM/), [GEO-INFER-SPM](../GEO-INFER-SPM/), [GEO-INFER-COG](../GEO-INFER-COG/), [GEO-INFER-ACT](../GEO-INFER-ACT/), [GEO-INFER-MATH](../GEO-INFER-MATH/) |
| 🤖 Agents & AI Orchestration | [GEO-INFER-AGENT](../GEO-INFER-AGENT/), [GEO-INFER-AG](../GEO-INFER-AG/), [GEO-INFER-AI](../GEO-INFER-AI/), [GEO-INFER-ANT](../GEO-INFER-ANT/), [GEO-INFER-OPS](../GEO-INFER-OPS/), [GEO-INFER-COMMS](../GEO-INFER-COMMS/) |
| 🏛️ Governance, Risk & Domain | [GEO-INFER-RISK](../GEO-INFER-RISK/), [GEO-INFER-INSURANCE](../GEO-INFER-INSURANCE/), [GEO-INFER-METAGOV](../GEO-INFER-METAGOV/), [GEO-INFER-NORMS](../GEO-INFER-NORMS/), [GEO-INFER-ECON](../GEO-INFER-ECON/), [GEO-INFER-PEP](../GEO-INFER-PEP/), [GEO-INFER-REQ](../GEO-INFER-REQ/), [GEO-INFER-SEC](../GEO-INFER-SEC/), [GEO-INFER-CIV](../GEO-INFER-CIV/), [GEO-INFER-HEALTH](../GEO-INFER-HEALTH/), [GEO-INFER-ORG](../GEO-INFER-ORG/) |
| 🗄️ Data, API & Applications | [GEO-INFER-API](../GEO-INFER-API/), [GEO-INFER-APP](../GEO-INFER-APP/), [GEO-INFER-DATA](../GEO-INFER-DATA/), [GEO-INFER-IOT](../GEO-INFER-IOT/), [GEO-INFER-ART](../GEO-INFER-ART/), [GEO-INFER-EDU](../GEO-INFER-EDU/) |
| 🛠️ Infrastructure & Validation | [GEO-INFER-INTRA](../GEO-INFER-INTRA/), [GEO-INFER-TEST](../GEO-INFER-TEST/), [GEO-INFER-LOG](../GEO-INFER-LOG/), [GEO-INFER-GIT](../GEO-INFER-GIT/), [GEO-INFER-EXAMPLES](../GEO-INFER-EXAMPLES/), [GEO-INFER-BIO](../GEO-INFER-BIO/) |

## Module Index

| Module | Package | Source files | Test files |
| --- | --- | ---: | ---: |
| [GEO-INFER-ACT](../GEO-INFER-ACT/) | `geo_infer_act` | 66 | 67 |
| [GEO-INFER-AG](../GEO-INFER-AG/) | `geo_infer_ag` | 16 | 12 |
| [GEO-INFER-AGENT](../GEO-INFER-AGENT/) | `geo_infer_agent` | 26 | 28 |
| [GEO-INFER-AI](../GEO-INFER-AI/) | `geo_infer_ai` | 18 | 21 |
| [GEO-INFER-ANT](../GEO-INFER-ANT/) | `geo_infer_ant` | 24 | 11 |
| [GEO-INFER-API](../GEO-INFER-API/) | `geo_infer_api` | 23 | 14 |
| [GEO-INFER-APP](../GEO-INFER-APP/) | `geo_infer_app` | 13 | 9 |
| [GEO-INFER-ART](../GEO-INFER-ART/) | `geo_infer_art` | 20 | 13 |
| [GEO-INFER-BAYES](../GEO-INFER-BAYES/) | `geo_infer_bayes` | 38 | 28 |
| [GEO-INFER-BIO](../GEO-INFER-BIO/) | `geo_infer_bio` | 12 | 10 |
| [GEO-INFER-CIV](../GEO-INFER-CIV/) | `geo_infer_civ` | 5 | 7 |
| [GEO-INFER-CLIMATE](../GEO-INFER-CLIMATE/) | `geo_infer_climate` | 11 | 12 |
| [GEO-INFER-COG](../GEO-INFER-COG/) | `geo_infer_cog` | 21 | 16 |
| [GEO-INFER-COMMS](../GEO-INFER-COMMS/) | `geo_infer_comms` | 19 | 20 |
| [GEO-INFER-DATA](../GEO-INFER-DATA/) | `geo_infer_data` | 30 | 28 |
| [GEO-INFER-ECON](../GEO-INFER-ECON/) | `geo_infer_econ` | 36 | 18 |
| [GEO-INFER-EDU](../GEO-INFER-EDU/) | `geo_infer_edu` | 7 | 7 |
| [GEO-INFER-EMERGENCY](../GEO-INFER-EMERGENCY/) | `geo_infer_emergency` | 8 | 9 |
| [GEO-INFER-ENERGY](../GEO-INFER-ENERGY/) | `geo_infer_energy` | 9 | 9 |
| [GEO-INFER-EXAMPLES](../GEO-INFER-EXAMPLES/) | `geo_infer_examples` | 7 | 11 |
| [GEO-INFER-FOREST](../GEO-INFER-FOREST/) | `geo_infer_forest` | 9 | 8 |
| [GEO-INFER-GIT](../GEO-INFER-GIT/) | `geo_infer_git` | 22 | 19 |
| [GEO-INFER-HEALTH](../GEO-INFER-HEALTH/) | `geo_infer_health` | 18 | 15 |
| [GEO-INFER-INSURANCE](../GEO-INFER-INSURANCE/) | `geo_infer_insurance` | 21 | 12 |
| [GEO-INFER-INTRA](../GEO-INFER-INTRA/) | `geo_infer_intra` | 9 | 10 |
| [GEO-INFER-IOT](../GEO-INFER-IOT/) | `geo_infer_iot` | 19 | 18 |
| [GEO-INFER-LOG](../GEO-INFER-LOG/) | `geo_infer_log` | 21 | 20 |
| [GEO-INFER-MARINE](../GEO-INFER-MARINE/) | `geo_infer_marine` | 12 | 10 |
| [GEO-INFER-MATH](../GEO-INFER-MATH/) | `geo_infer_math` | 67 | 24 |
| [GEO-INFER-METAGOV](../GEO-INFER-METAGOV/) | `geo_infer_metagov` | 21 | 16 |
| [GEO-INFER-NORMS](../GEO-INFER-NORMS/) | `geo_infer_norms` | 20 | 15 |
| [GEO-INFER-OPS](../GEO-INFER-OPS/) | `geo_infer_ops` | 17 | 18 |
| [GEO-INFER-ORG](../GEO-INFER-ORG/) | `geo_infer_org` | 5 | 7 |
| [GEO-INFER-PEP](../GEO-INFER-PEP/) | `geo_infer_pep` | 40 | 16 |
| [GEO-INFER-PLACE](../GEO-INFER-PLACE/) | `geo_infer_place` | 103 | 32 |
| [GEO-INFER-REQ](../GEO-INFER-REQ/) | `geo_infer_req` | 5 | 7 |
| [GEO-INFER-RISK](../GEO-INFER-RISK/) | `geo_infer_risk` | 15 | 24 |
| [GEO-INFER-SEC](../GEO-INFER-SEC/) | `geo_infer_sec` | 22 | 20 |
| [GEO-INFER-SIM](../GEO-INFER-SIM/) | `geo_infer_sim` | 11 | 9 |
| [GEO-INFER-SPACE](../GEO-INFER-SPACE/) | `geo_infer_space` | 85 | 52 |
| [GEO-INFER-SPM](../GEO-INFER-SPM/) | `geo_infer_spm` | 27 | 23 |
| [GEO-INFER-TEST](../GEO-INFER-TEST/) | `geo_infer_test` | 27 | 59 |
| [GEO-INFER-TIME](../GEO-INFER-TIME/) | `geo_infer_time` | 20 | 23 |
| [GEO-INFER-TRANSPORT](../GEO-INFER-TRANSPORT/) | `geo_infer_transport` | 7 | 10 |
| [GEO-INFER-WATER](../GEO-INFER-WATER/) | `geo_infer_water` | 9 | 9 |

## Current Repository Facts

| Metric | Value |
| --- | ---: |
| Modules | 45 |
| Python source files | 1041 |
| Python test files | 826 |
| Repository README.md files | 812 |
| Repository AGENTS.md files | 808 |

These are file inventories, not executed-test or coverage counts. Regenerate
them with `uv run python GEO-INFER-TEST/rewrite_readme_agents.py`.

## Modular Hygiene

- Root `pyproject.toml`, `uv.lock`, and `.python-version` are the canonical uv environment surfaces.
- Keep `uv.lock` as LF text through root `.gitattributes` so raw receipt hashes agree across platforms.
- Sync the full workspace with `uv sync --all-packages --all-extras --all-groups` before repo-wide validation.
- Each module owns importable behavior under `src/` and keeps at least four pytest files under `tests/`.
- Planned work belongs in root `TODO.md` or a tracked issue, not source or test task markers.
- Importable libraries use `logging.getLogger(__name__)`; process-wide logging configuration belongs in CLI entrypoints.

## Validation

- Repository contracts: `uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language --strict-import-smoke`
- Documentation links and current-state claims: `uv run python GEO-INFER-TEST/validate_documentation.py --strict`
- Syntax gate: `python -m compileall GEO-INFER-*/src GEO-INFER-*/examples`
- Skill contracts: `uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal`
- Secrets: `gitleaks detect --source . --config .gitleaks.toml --redact --verbose` (local replication: GEO-INFER-TEST/docs/secret_scan_policy.md)
- Unit tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --category unit`
- Integration tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --category integration`
- System tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --category system`
- Performance tests: `uv run python GEO-INFER-TEST/run_unified_tests.py --category performance`
- Coverage gate: `uv run python GEO-INFER-TEST/run_unified_tests.py --category coverage --timeout 900`
- H3 contracts: `uv run python GEO-INFER-TEST/run_unified_tests.py --h3-migration`
- Test contract: `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict`
- Model contract: `uv run python GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42`
- Reproducible model audit: `uv run python GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible`
- Lint contract: `uv run --with 'ruff>=0.15.6,<0.16' ruff check .`
- Manuscript variables, figures, captions, and resolved copies: `uv run python manuscript/generate_research_artifacts.py`
- After source commits, refresh metadata with `uv run python manuscript/generate_research_artifacts.py --refresh-config`, then commit only `manuscript/config.yaml` separately before generating evidence.
- Verify delivered PNG/SVG/JSON/CSV and resolved manuscript bytes with `uv run python manuscript/generate_research_artifacts.py --check`.
- For an actual docxology/template PDF with the graphical-abstract cover, use the explicit optional [template rendering lane](../GEO-INFER-INTRA/docs/integration/manuscript_rendering.md); retain its template commit, interpreter and render receipt separately from publication verification.

## Repo-wide Change Workflow

1. Inspect the owning module and keep behavior in its `src/` package.
2. Add or update a focused test in the owning module's `tests/` directory.
3. Run the focused test, then compile and run the contract validators.
4. Refresh generated signposts with `uv run python GEO-INFER-TEST/rewrite_readme_agents.py`.
5. Confirm generated documentation is stable with `uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check`.

On macOS 27, native PyMC checks require the explicit
[SPM compiler profile](../GEO-INFER-SPM/docs/native_sampling_profiles.md).
Keep native compilation enabled and retain the chosen profile with results.

## Artifact and Output Hygiene

- Test reports belong under `.geo-infer-test-results/`.
- Model-audit artifacts are emitted under `.geo-infer-test-results/model-audit/`.
- The manuscript pipeline is the only approved writer to ignored repository-root
  `output/`; scenario and other visualization outputs must use an explicit
  output directory and must not write there.
- Generated signposts must describe tracked files only; local caches and build
  products are intentionally excluded.

README.md and AGENTS.md files below the repository root are generated signposts.
The generator derives their contents from tracked files, public symbols, module
metadata, validation commands, and test inventories; update the generator when
the documentation contract itself changes.

## Failure Triage

- `validate_repo_contracts.py`: source layout, language, dependency, logger, and documentation contract.
- `validate_logging_hygiene.py`: passive library-logging contract (no root-logger mutation; module-local `getLogger(__name__)` loggers only).
- `validate_test_contracts.py`: test inventories, markers, fixtures, skips, and warning policy.
- `run_unified_tests.py`: module behavior by unit, integration, performance, or H3 category.
- `validate_model_contracts.py` and `run_model_audit.py`: deterministic model outputs and reproducibility artifacts.
- `rewrite_readme_agents.py --check`: generated README/AGENTS drift; rerun the generator after intentional tracked-file changes.

## Zero-warning test policy

The shared pytest policy treats warnings as errors, requires strict markers/configuration, assigns exactly one primary marker to every test, and rejects skips, xfails, xpasses, collection errors, missing dependencies, missing fixtures, and empty selections. Every module has a test inventory at `GEO-INFER-*/tests/README.md`; the inventory records purpose, fixtures, dependencies, artifacts, and triage commands.

## Documentation Policy

README.md and AGENTS.md files describe current, discoverable repository state. Do not add aspirational APIs to these files unless the implementation, export path, and validation command exist in this checkout. Keep module-local public exports and test commands synchronized through the generator.
