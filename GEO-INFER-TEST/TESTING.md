# GEO-INFER strict testing program

The repository test contract is zero-warning, zero-failure, and zero-skip. The shared root configuration enables strict markers/configuration, importlib discovery, `pytest-asyncio` auto mode, and `-W error`. The runner treats missing tests, collection errors, pytest exit code 5, skips, xfails, and xpasses as failures.

## Canonical gate

`scripts/gate.py` is the one entry point; each arm runs the commands one CI job
runs, copied from the workflow definition. Run from the repository root:

```bash
uv sync --all-packages --all-extras --all-groups
uv run --frozen python scripts/gate.py all
```

Run one arm while iterating; `all` runs each arm exactly once. The previous
inline list here ran `validate_skills.py` twice, omitted the `--warnings-fatal`
and `--strict-import-smoke` flags CI enforces, and had no fast, GNN or render
surface, so it was replaced rather than kept as a second copy of the gate.

| Arm | Purpose |
| --- | --- |
| `lint` | `ruff check .` and `ruff format --check .` under the root lint contract |
| `docs` | authoritative-documentation links/claims and generated-signpost drift |
| `fast` | the eight-file execution, runner, workflow and SPACE TIME selection |
| `gates` | repository, packaging, logging, doc-example, skill, Active Inference, syntax, test-contract, model-contract, model-audit and secret-scan gates |
| `test-<category>` | one matrix category (`unit`, `integration`, `performance`, `system`, `slow`, `h3`) |
| `tests` | every `test-<category>` arm |
| `gnn` | paired GNN conformance against the pinned companion revision |
| `manuscript-artifacts` / `manuscript-convergence` / `manuscript-render` | producer verification, convergence check, PDF render and `ROOT` profiles |
| `manuscript` | those three arms in order |

`ci.yml` calls the same arms (`scripts/gate.py <arm>`) rather than repeating
their commands, so a gate command is written once in this repository.

Arm contents, host requirements, measured wall time and the CI surfaces no arm
covers are in [the gate guide](docs/gate.md).

Use `env -u VIRTUAL_ENV` before `uv` when a different shell environment is active. This avoids an environment-selection warning and ensures the workspace `.venv` is used.

## Test taxonomy and fixtures

Every test receives exactly one primary marker based on its canonical directory: `unit`, `integration`, `system`, or `performance`. Markers such as `api`, `core`, `geospatial`, `model`, `reporting`, `reproducibility`, `spatial`, and `artifact` are additive. The unified runner discovers category tests from those canonical directories; a filename containing `performance` does not move a unit test into the performance gate. `geo_infer_test.testing` provides deterministic RNG, local filesystem, localhost HTTP, SQLite, in-process service, finite-value, probability-vector, stochastic-matrix, model-contract, and artifact-manifest helpers.

Pytest test files may use either `test_*.py` or `*_test.py`; inventories, health checks, and runners count and execute both patterns.

Required external behavior is represented by local fixtures. A test must not use `pytest.skip`, `skipif`, `importorskip`, `xfail`, or warning suppression. Missing dependencies and unavailable required backends are explicit failures.

## Model and artifact contracts

Model checks require finite outputs, declared shapes and dtypes, normalized probabilities, valid stochastic matrices, deterministic seeded replay, reset restoration, and explicit invalid-input errors. `validate_model_contracts.py` exercises representative ACT categorical, Gaussian, climate, ecological, urban, resource, and multi-agent models.

`run_model_audit.py` writes `model_contracts.json`, `statistics.json`, `model_audit.png`, and `manifest.json` beneath `.geo-infer-test-results/model-audit/`. The manifest records schema version, finite statistics, byte counts, SHA-256 hashes, and a deterministic manifest hash. The shared `assert_visualization_manifest` helper verifies every sidecar.

## Failure triage

- Contract failure: run `uv run python GEO-INFER-TEST/validate_test_contracts.py --strict` and fix the reported source file.
- Warning or collection failure: run the affected module with `uv run pytest -c pyproject.toml -W error -vv path/to/test.py`.
- Model failure: run `uv run python GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42`.
- Artifact failure: run `uv run python GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible` and inspect the manifest sidecars.
- Category failure: re-run with `--show-failures` to print each failed
  suite's failing test names in the final summary verdict (summary.json
  records them regardless), and inspect the corresponding JUnit XML and
  `summary.json` under `.geo-infer-test-results/`.

## CI setup and timing

The CPU validation, fast-contract, category and manuscript jobs use
`.github/actions/sync-cpu-workspace/action.yml` for the same locked workspace
sync. It installs all members, extras and dependency groups with the four
explicit hosted CPU exclusions (CuPy, Mayavi, Vaex and Vaex Core). Each job
constructs its own environment; a cache hit never replaces `uv sync --locked`.

These jobs share the uv download/build cache per Python interpreter, keyed by
all workspace manifests, the lock and `.python-version`. Only each interpreter's
unit job saves that key; the other jobs restore it without competing uploads.
The installed-wheel job retains its separate cache and installation isolation.

The baseline green run [37734618009](https://github.com/ActiveInferenceInstitute/GEO_INFER/actions/runs/37734618009)
spent 65–106 seconds in each category's dependency sync. System tests took
15–16 seconds and unit tests 494–525 seconds. Compare the same step timings in
the next green run before claiming a speedup. A new cache key still starts cold;
no jobs wait for another category to populate it. Workflow contracts parse each
workflow once and return independent document copies to prevent test coupling.
Per-module execution receipts and `--durations=10` retain expensive runtime paths
for subsequent profiling without narrowing selections or relaxing deadlines.
