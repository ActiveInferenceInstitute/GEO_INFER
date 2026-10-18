# Code Structure

GEO-INFER is a monorepo of independently packaged GEO-INFER-* workspaces.
The root pyproject.toml and uv.lock provide the shared environment;
module pyproject.toml files provide package metadata and local extras.

## Repository layout

```text
GEO-INFER/
├── GEO-INFER-*/src/             # importable module behavior
├── GEO-INFER-*/tests/           # module tests and inventories
├── GEO-INFER-*/examples/        # thin runnable orchestration
├── GEO-INFER-*/docs/            # module-specific conceptual docs
├── GEO-INFER-*/SKILL.md         # agent-facing current API guidance
├── GEO-INFER-INTRA/docs/        # cross-module documentation hub
├── GEO-INFER-TEST/              # unified runner and validators
├── .github/workflows/           # CI definitions
├── pyproject.toml               # root workspace and shared tooling
├── uv.lock                      # resolved dependency graph
└── .python-version              # supported interpreter selection
```

## Inside a module

```text
GEO-INFER-MODULE/
├── src/geo_infer_module/
│   ├── __init__.py               # public exports and package metadata
│   ├── core/                     # primary algorithms and contracts
│   ├── models/                   # domain/model objects
│   ├── api/                      # optional service surfaces
│   └── utils/                    # bounded reusable helpers
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── performance/
│   └── README.md                 # generated test inventory
├── examples/
├── README.md                     # generated operational signpost
├── AGENTS.md                     # generated local instructions
├── SKILL.md
└── pyproject.toml
```

Not every module uses every subdirectory. Follow the actual directory and
public exports instead of copying a template blindly.

## Ownership rules

- Put behavior in the owning module's src/.
- Keep cross-module calls on public exports or documented adapters.
- Put validation close to the boundary that owns the invariant.
- Keep scripts and examples thin; do not duplicate library algorithms there.
- Use module loggers and configure handlers only at CLI boundaries.
- Keep generated signposts synchronized through
  GEO-INFER-TEST/rewrite_readme_agents.py.

## Reading a module

1. Read the module root README and SKILL.
2. Inspect src/<package>/__init__.py and its __all__.
3. Read the relevant tests before changing behavior.
4. Check pyproject.toml for dependencies and extras.
5. Run a focused test and then the module gate.

## Cross-module changes

Document the owning module, input/output schema, coordinate system, optional
dependencies, and validation command. Update architecture or integration docs
only after the contract is implemented and tested.
