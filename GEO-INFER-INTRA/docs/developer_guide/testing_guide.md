# GEO-INFER Testing Guide

GEO-INFER uses pytest plus repository-level executable contracts. The goal is a
meaningful, deterministic signal: warnings, skips, xfails, missing dependencies,
empty selections, and generated-artifact drift are treated as failures at the
strict gates.

## Test ownership

Each `GEO-INFER-*` module owns its tests under `tests/`. Keep behavior tests
next to the owning module; use `GEO-INFER-TEST/tests/` for the test framework
itself and for repository-contract behavior.

Typical layout:

```text
GEO-INFER-MODULE/
├── src/geo_infer_module/       # importable behavior
├── tests/
│   ├── unit/                   # focused behavior
│   ├── integration/            # cross-component behavior
│   ├── performance/            # measured or benchmarked behavior
│   ├── README.md               # generated inventory and triage
│   └── conftest.py
├── examples/                   # thin runnable orchestration
├── README.md
├── AGENTS.md
├── SKILL.md
└── pyproject.toml
```

The repository release contract requires at least four pytest files per module.
The generated `tests/README.md` records the current inventory.

## Environment

From the repository root:

```bash
uv sync --all-packages --all-extras --all-groups
uv run python -c "import sys; print(sys.executable)"
```

Use `uv run` for every command in this guide. This avoids accidentally
loading a system interpreter or a conflicting H3/pymdp installation.

## Focused development loop

Run the smallest useful check first:

```bash
uv run python -m pytest GEO-INFER-ACT/tests/unit/test_h3_active_inference.py -q
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```

For direct pytest use, retain the repository configuration and strict warnings:

```bash
uv run python -m pytest -c pyproject.toml -W error GEO-INFER-SPACE/tests/unit -q
```

## Canonical repository gates

```bash
# generated docs and structural contracts
uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check
uv run python GEO-INFER-TEST/validate_documentation.py --strict
uv run python GEO-INFER-TEST/validate_doc_examples.py
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language --strict-import-smoke
uv run python GEO-INFER-TEST/validate_packaging.py --strict
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs --warnings-fatal
uv run python GEO-INFER-TEST/validate_test_contracts.py --strict

# syntax, models, and runtime hygiene
uv run python -m compileall GEO-INFER-*/src GEO-INFER-*/examples
uv run python GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42
uv run python GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible
uv run --with 'ruff>=0.15.6,<0.16' ruff check .

# behavioral suites
uv run python GEO-INFER-TEST/run_unified_tests.py --category unit
uv run python GEO-INFER-TEST/run_unified_tests.py --category slow
uv run python GEO-INFER-TEST/run_unified_tests.py --category integration
uv run python GEO-INFER-TEST/run_unified_tests.py --category performance
uv run python GEO-INFER-TEST/run_unified_tests.py --category system
uv run python GEO-INFER-TEST/run_unified_tests.py --h3-migration
```

CI runs the same categories on Python 3.11 and 3.12. Hosted CI omits a small
set of native-only optional packages; see the workflow for the exact install
flags.

### GitHub Actions behavior

The maintained workflow is
[`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml). It runs on
pushes and pull requests targeting `main` or `develop`, and can also be
started with `workflow_dispatch`. Tag and scheduled runs use the same category
matrix. Each Python matrix entry is evaluated
independently (`fail-fast: false`) so a failure on one interpreter does not
hide the result from the other. Superseded runs for the same branch or pull
request are cancelled through workflow concurrency.
CPU CI sets BLAS/OpenMP libraries to one thread per isolated module process;
the runner defaults to two module workers. Local runs can set the native thread
environment variables and `--workers` explicitly for their available resources.

The workflow uses uv 0.10.7 with the locked `uv.lock` resolution, caches uv
artifacts per Python version, and pins third-party Actions to reviewed commit
SHAs. The
checkout has read-only repository permissions and does not persist credentials.
On every outcome, `.geo-infer-test-results/` is uploaded as a short-lived CI
artifact when available; this includes the unified summary and JUnit reports
needed to diagnose a failure. Deleted Python paths are excluded from the
changed-file formatter/linter step because they are not present in the
checkout. Changed-file and repository-wide Ruff checks use the root lint
configuration; `ruff format --check` is the formatting gate. Coverage change
selection includes deleted source/test paths, dependency files and shared
configuration. Coverage uses the same registered test roots as execution.

When local and hosted results differ, first compare Python versions and the
native-only dependency exclusions, then inspect the uploaded summary and
JUnit artifacts. Do not relax strict warnings, skips, empty selections, or
reproducibility gates to accommodate a single environment difference.

## Test categories

The unified runner supports:

| Command | Scope |
| --- | --- |
| `--module ACT` | all discoverable tests for one module |
| `--category unit` | unit and direct module tests, excluding declared slow cases |
| `--category slow` | explicitly marked slow cases in registered test roots |
| `--category integration` | canonical integration directories |
| `--category performance` | canonical performance directories |
| `--category system` | canonical system directories |
| `--category coverage` | coverage analysis |
| `--h3-migration` | ACT/SPACE H3 contract validators |
| `--list-modules` | discovered module names |
| `--timeout 600` | per-command timeout override |

Discovery includes all 45 modules, PLACE's registered Cascadia tests and the
root manuscript tests. Explicit backend profiles install their dependencies and
record the selected, deselected and executed inventory. Tests for absent extras
remain separate from tests that exercise installed backend behavior.

Every execution preserves an immutable attempt directory containing receipts,
complete UTF-8 logs and applicable JUnit under `.geo-infer-test-results/runs/`.
The summary path is a latest-run pointer. Receipts bind revision, dirty source
bytes, interpreter, lock and artifact hashes. Missing, stale, malformed or empty
JUnit reports fail pytest attempts; ordinary validators record their failure
without requiring JUnit. Planned empty module selections are recorded, while an
entirely empty lane fails. Assertion failures are not automatically retried.

Monotonic deadlines bound the command and ownership census. Cleanup terminates
and waits for descendants that inherit the launch environment, including detached
children. Native POSIX inspection supports both BSD and GNU `ps`; Windows uses
process identities. Console text escapes characters a restricted encoding cannot
represent; retained UTF-8 artifacts preserve the original decoded output. Read
those artifacts with `encoding="utf-8"` independently of the machine's locale.

Temporary Git fixtures override inherited `core.fsmonitor` configuration so they
cannot launch persistent user-configured monitor daemons. This isolation applies
to test repositories; production Git behavior remains caller-configured.

## Writing strong tests

A useful test asserts an observable invariant:

- probabilities are finite, non-negative, and normalized;
- H3 cells have the expected resolution and valid parent/child closure;
- outputs preserve ordering, coordinate systems, units, and shape;
- stochastic algorithms replay with the same seed and vary with a different seed;
- visualizations create the requested artifact and a JSON-safe manifest;
- invalid input fails at the public boundary with an actionable exception.

Avoid asserting only that an object exists or that a dictionary is nonempty.
Avoid mocks, fake return values, or placeholder tests for the implementation under
test. Use deterministic local fixtures and real small inputs.

## Shared testing helpers

`geo_infer_test.testing` exports reusable helpers for finite arrays,
probabilities, stochastic matrices, model contracts, seed replay, visualization
manifests, local filesystem boundaries, and local HTTP/SQLite/service fixtures.
Prefer these helpers when they make the invariant clearer.

```python
from geo_infer_test import assert_probability, assert_seed_replay

assert_probability([0.2, 0.3, 0.5])
assert_seed_replay(run_once, seed=42)
```

## H3 and numerical tests

- Use H3 v4 names such as `latlng_to_cell`, `cell_to_latlng`, and
  `grid_disk`.
- Keep WGS84 coordinate order explicit: latitude, then longitude.
- Test constant-valued and boundary cases for color scales and maps.
- Check finite covariance, positive-definite solves, and normalized posteriors.
- For nested H3, validate orphan counts, resolution ordering, parent/child maps,
  and aggregation conservation.

## Troubleshooting

- **Import mismatch**: print `sys.executable` under `uv run`; inspect package
  metadata with `uv pip show <package>`.
- **H3 API error**: verify `h3>=4.5.0,<5` and use v4 names.
- **No tests collected**: confirm the category directory and filename match
  `test_*.py` or `*_test.py`; the strict runner treats exit code 5 as failure.
- **Warning failure**: remove warning filters and fix the underlying deprecation,
  optional-backend, or input-validation problem.
- **Artifact failure**: inspect `.geo-infer-test-results/` and ensure the test
  uses a temporary output directory rather than repository-root paths.
- **Generated-doc failure**: run the generator once, inspect the diff, then use
  `--check` to confirm stability.
