# Module Structure Standards

Modules follow this layout; optional directories (`config/`, `docs/`,
`examples/`, `api/`, `models/`) exist only where the module needs them:

```
GEO-INFER-MODULE/
├── config/                 # Configuration files (YAML/JSON)
│   ├── default.yaml        # Default configuration
│   └── schema.json         # Configuration validation schema
├── docs/                   # Documentation (markdown, API specs)
│   ├── api_schema.yaml     # OpenAPI documentation
│   ├── architecture.md     # Module architecture
│   └── tutorials/          # Step-by-step tutorials
├── examples/               # Working examples and demonstrations
│   ├── basic_example.py    # Basic usage
│   └── advanced_example.py # Advanced workflows
├── src/                    # Source code
│   └── geo_infer_module/   # Main package (lowercase, underscored)
│       ├── __init__.py     # Package init with version and exports
│       ├── api/            # API definitions and routes
│       │   ├── __init__.py
│       │   ├── rest_api.py
│       │   └── schemas.py
│       ├── core/           # Core functionality and algorithms
│       │   ├── __init__.py
│       │   ├── main_engine.py
│       │   └── algorithms.py
│       ├── models/         # Data models and schemas
│       │   ├── __init__.py
│       │   └── data_models.py
│       └── utils/          # Utility functions and helpers
│           ├── __init__.py
│           ├── helpers.py
│           └── validation.py
├── tests/                  # Comprehensive test suite
│   ├── conftest.py         # Shared fixtures
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── performance/        # Performance benchmarks
├── pyproject.toml          # Sole packaging + dependency declaration
├── AGENTS.md               # Generated agent signpost for this module
├── SKILL.md                # Hand-written Claude Code skill
└── README.md               # Generated module signpost
```

## Required Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata and dependencies (no `setup.py`, `setup.cfg` or `requirements.txt`; lint, format and pytest config live in the root `pyproject.toml`) |
| `src/geo_infer_*/` | Source package (PEP 8 lowercase) |
| `tests/` | At least four pytest files, with unit + integration subdirs |
| `SKILL.md` | Hand-written Claude Code skill; validated by `validate_skills.py` |
| `README.md`, `AGENTS.md` | Generated signposts (contents, public interface, metadata, dependencies, validation) |

## Generated Signposts

Module `README.md` and `AGENTS.md` files are generated from repository files by
`uv run python GEO-INFER-TEST/rewrite_readme_agents.py`; CI fails when they
drift (`--check`). Do not hand-edit them. Put conceptual guidance, tutorials
and integration notes in `GEO-INFER-INTRA/docs/` or the module's `docs/`.

## Module Themes

These are the root README "Module Themes" (45 modules).

| Theme | Modules |
|-------|---------|
| 🌍 **Spatial & Place-based** | SPACE, PLACE, TIME, MARINE, WATER, FOREST, CLIMATE, ENERGY, TRANSPORT, EMERGENCY |
| 🧠 **Bayesian & Active Inference** | BAYES, SIM, SPM, COG, ACT, MATH |
| 🤖 **Agents & AI Orchestration** | AGENT, AG, AI, ANT, OPS, COMMS |
| 🏛️ **Governance, Risk & Domain** | INSURANCE, METAGOV, NORMS, ECON, PEP, REQ, SEC, CIV, HEALTH, ORG, RISK |
| 🗄️ **Data, API & Applications** | API, APP, DATA, IOT, ART, EDU |
| 🛠️ **Infrastructure & Validation** | INTRA, TEST, LOG, GIT, EXAMPLES, BIO |

## Module-Specific Agent Guidance

Modules provide domain guidance in their module-level `AGENTS.md` file.

### Best Practices

- Keep module instructions focused on domain-specific requirements in `AGENTS.md`
- Don't duplicate root rules — reference `.agents/` and root `AGENTS.md` instead
- Update module `AGENTS.md` when adding new domain capabilities or workflows
