# Architecture Overview

This document provides a high-level overview of the GEO-INFER architecture:
the module layers, shared environment, documentation hub, and validation
contracts. Detailed component documentation lives in the
[Architecture index](index.md).

## System Architecture

GEO-INFER is a uv-managed Python 3.11+ monorepo of 45 modules. Each module is
a `GEO-INFER-*` directory owning an importable package under `src/`, tests
under `tests/`, examples, and generated `README.md`/`AGENTS.md` signposts.

The major layers:

1. **Spatial and statistical core** — GEO-INFER-SPACE (H3 v4 indexing and
   geospatial analysis), GEO-INFER-MATH (spatial statistics and numerical
   methods), GEO-INFER-TIME (temporal analysis), GEO-INFER-BAYES and
   GEO-INFER-SPM (probabilistic inference).
2. **Active Inference** — GEO-INFER-ACT (generative models, belief updating,
   policy selection, H3 spatial inference) and GEO-INFER-AGENT (agents for
   perception, decision, and action).
3. **Domain modules** — agriculture, climate, water, marine, forest, energy,
   transport, health, risk, economy, and more; each implements domain models
   on top of the core layers.
4. **Applications and integration** — GEO-INFER-API, GEO-INFER-APP,
   GEO-INFER-IOT, GEO-INFER-COMMS, GEO-INFER-OPS, GEO-INFER-EXAMPLES.
5. **Repository intelligence** — GEO-INFER-INTRA (documentation hub, ontology,
   workflows, knowledge base), GEO-INFER-TEST (validation contracts, unified
   test runner, signpost generator).

See the [Module Catalog](module_catalog.md) for the full inventory and
[Module Structure](module_structure.md) for the per-module layout.

## Data Flow

The typical data flow through the system:

1. **Ingestion** — data is loaded through the module I/O layers (GEO-INFER-DATA
   and module-specific connectors).
2. **Spatial processing** — H3 v4 cell indexing, geometric operations, and
   spatial analytics in GEO-INFER-SPACE.
3. **Modeling** — probabilistic and active inference modeling in ACT, BAYES,
   SPM, and domain modules.
4. **Analysis and visualization** — results are validated, rendered, and
   reported; deterministic visualization receipts are emitted under
   `.geo-infer-test-results/`.
5. **Validation** — repository contracts, tests, and generated signposts are
   checked by GEO-INFER-TEST validators.

See [Data Flow](data_flow.md) and [Component Diagram](component_diagram.md)
for detailed diagrams.

## Documentation Architecture

- The repository root hosts `README.md`, `AGENTS.md`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CHANGELOG.md`, and `TODO.md`.
- The [GEO-INFER-INTRA docs hub](../index.md) hosts conceptual, tutorial, and
  reference documentation.
- Every directory carries generated `README.md`/`AGENTS.md` signposts derived
  from tracked files by `GEO-INFER-TEST/rewrite_readme_agents.py`.
- Documentation accuracy is enforced by
  `GEO-INFER-TEST/validate_documentation.py --strict`.

## Design Principles

### Modularity

Clearly defined modules with specific responsibilities and public interfaces.
Each module owns its behavior under `src/`; scripts and examples are thin
orchestration surfaces.

### Contract-Driven Validation

Behavior is enforced by validators and tests:

- `validate_repo_contracts.py` — source layout, language, dependencies,
  loggers, documentation.
- `validate_test_contracts.py` — test inventories, markers, fixtures, skips,
  warning policy.
- `validate_model_contracts.py` / `run_model_audit.py` — deterministic model
  outputs and reproducibility.
- `run_unified_tests.py` — module behavior by unit, integration, performance,
  or H3 category.

### FAIR Principles

The system adheres to FAIR principles (Findable, Accessible, Interoperable,
Reusable):

- **Findable**: content is indexed in the documentation hub.
- **Accessible**: content is available through standardized interfaces.
- **Interoperable**: standard formats (GeoJSON, H3, etc.) are used.
- **Reusable**: content includes metadata and licensing information
  (CC-BY-4.0).

### Security by Design

- Token authentication and validation in GEO-INFER-SEC.
- Sandboxed evaluation for SPACE raster and MATH symbolic eval paths.
- Safe-extract guards for archive handling; pickle loads are trusted-only.

## Technical Stack

- **Python 3.11+** with the **uv** workspace manager (root `pyproject.toml`,
  `uv.lock`, `.python-version`).
- **H3 v4** (h3>=4.5.0,<5) for hierarchical spatial indexing.
- **pytest** with a zero-warning, zero-skip policy; **ruff** for linting.
- Scientific stack (numpy, scipy, pandas) plus optional extras (PyMC, arviz,
  pymdp, web, IoT, performance, documentation).

## Deployment Options

GEO-INFER modules are Python packages; deployment depends on the module:

1. **Workspace usage** — `uv sync --all-packages --all-extras --all-groups` from the
   repository root.
2. **Single-package usage** — `uv sync --package geo-infer-<module>`.
3. **Containerized deployment** — modules can be packaged as containers; CI
   runs on CPU runners (see `.github/workflows/ci.yml`).

See [Deployment Architecture](../deployment/index.md) for environments and
scaling guidance.

## Integration Points

GEO-INFER-INTRA integrates with other modules through their public packages:

- **GEO-INFER-SPACE** (`geo_infer_space`) — spatial data management and
  analysis.
- **GEO-INFER-TIME** (`geo_infer_time`) — temporal data management and
  analysis.
- **GEO-INFER-ACT** (`geo_infer_act`) — active inference models.
- **GEO-INFER-API** (`geo_infer_api`) — external API gateway.
- **GEO-INFER-OPS** (`geo_infer_ops`) — operational management.
- **GEO-INFER-APP** (`geo_infer_app`) — user-facing applications.

Each integration point is documented in the
[Integration guide](../integration/index.md).

## Future Architecture

Planned directions are tracked as open work in the root
[TODO.md](../../../TODO.md) and repository issues; they are not described here
as current capabilities.
