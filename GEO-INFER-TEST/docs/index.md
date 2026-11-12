# GEO-INFER-TEST

GEO-INFER-TEST is the repository-wide validation module. It provides the
unified pytest runner, strict test-contract checks, model reproducibility
checks, documentation parity, source hygiene, and H3/Active Inference
contract validators.

## Install and run

```bash
uv sync --all-packages --all-extras --all-groups
uv run python GEO-INFER-TEST/run_unified_tests.py --list-modules
uv run python GEO-INFER-TEST/run_unified_tests.py --module ACT
```

Use `uv run` so the command uses the shared workspace interpreter.

## Canonical command matrix

The gate is `uv run --frozen python scripts/gate.py <arm>`; each arm runs the
same commands as one CI job. See the [gate guide](gate.md) for arm contents,
host requirements, measured wall time and the CI surfaces no arm covers.

| Command | Purpose |
| --- | --- |
| `scripts/gate.py lint` | Whole-tree `ruff check` and `ruff format --check`. |
| `scripts/gate.py docs` | Authoritative documentation links/claims and generated-signpost drift. |
| `scripts/gate.py fast` | Fast contract selection (execution, runner, workflow, SPACE TIME). |
| `scripts/gate.py gates` | Interpreter-independent validators, model contracts and secret scan. |
| `scripts/gate.py tests` | Every test category the CI matrix runs. |
| `scripts/gate.py gnn` | Paired GNN conformance against the pinned revision. |
| `scripts/gate.py manuscript` | Research-artifact producer, render and `ROOT` profiles. |
| `run_unified_tests.py --module NAME` | Run all discoverable tests for one module. |
| `run_unified_tests.py --category unit` | Run unit tests module by module. |
| `run_unified_tests.py --category integration` | Run integration tests module by module. |
| `run_unified_tests.py --category performance` | Run performance directories only. |
| `run_unified_tests.py --category coverage` | Run the coverage analysis path. |
| `run_unified_tests.py --h3-migration` | Run ACT/SPACE H3 contract validators. |
| `validate_repo_contracts.py --strict-source-language` | Check repository structure, source hygiene, and signposting. |
| `validate_test_contracts.py --strict` | Check test inventories, markers, skips, and warnings. |
| `validate_model_contracts.py --strict --seed 42` | Check finite model outputs and typed contracts. |
| `run_model_audit.py --seed 42 --reproducible` | Emit deterministic model statistics and visual artifacts. |
| `rewrite_readme_agents.py --check` | Fail if generated README/AGENTS files drift. |
| `validate_documentation.py --strict` | Check authoritative documentation links and stale claims. |
| `validate_skills.py --check-xrefs --warnings-fatal` | Check SKILL metadata and referenced paths; warnings fail. |

All report-producing commands write under `.geo-infer-test-results/` unless an
explicit output directory is part of the command contract.

## Test categories and strict policy

Tests are discovered from `test_*.py` and `*_test.py` files under each
module's `tests/`. The strict repository policy rejects:

- warnings emitted during tests;
- skipped, xfailed, or xpassed tests;
- missing dependencies, fixtures, or markers;
- pytest collection failures;
- empty selections and exit code 5;
- root-level generated output artifacts;
- non-finite model statistics and stale generated documentation.

Modules retain their local test layouts, but each module must keep at least four
pytest files and a generated `tests/README.md` inventory.

## Programmatic API

```python
from geo_infer_test import GeoInferTestRunner, TestConfiguration

config = TestConfiguration(
    modules_to_test=["ACT"],
    test_types=["unit"],
    parallel_execution=False,
)
runner = GeoInferTestRunner(config)
discovered = runner.discover_tests()
report = runner.run_all_tests()
print(discovered)
print(report)
```

For lower-level reusable assertions, import from `geo_infer_test.testing` or
the package exports: finite arrays, normalized probabilities, stochastic
matrices, model contracts, seed replay, and visualization manifests.

## GNN companion conformance

The paired-interchange conformance check is owned by
`.github/workflows/gnn-interchange.yml`; `ci.yml` calls it as the
`paired-interchange` job. The job key is `interchange`, and its `3.12` matrix
leg publishes the required check name `gnn conformance (pinned)` while the
`3.11` leg publishes `gnn interchange (3.11)`.

- Pin file: `.github/gnn-pair.json`, schema `gnn-importer-pin/v1`, pinning
  `ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.1.0`
  (`75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf`).
- Reader: `python -m gnn.importers check --repo . --require-pointer
  --workflow .github/workflows/gnn-interchange.yml --job interchange`, run from
  the pinned checkout's interpreter. The arm carries no pin shape or pointer
  assertion of its own.
- What the job parses, and what it deliberately does not cover:
  [`.github/GNN.md`](../../.github/GNN.md).
- Validator: `validate_gnn_interchange.py`; run it locally with
  `scripts/gate.py gnn` and a GNN checkout at the pinned revision. The
  [gate guide](gate.md) lists the exact interpreter requirement.
- Raising the pin is a reviewed change to `.github/gnn-pair.json`; the
  procedure is in
  [the pair-pin bump section](gnn_space_time_2026_09.md#gnn-pair-pin-bump-procedure).

## Results and triage

- `.geo-infer-test-results/summary.json` contains command-level status,
  durations, and output tails.
- Module JUnit reports are written beside the summary.
- Model-audit images and sidecars are under
  `.geo-infer-test-results/model-audit/`.
- GitHub Actions uploads this directory as a per-Python-version artifact on
  every CI outcome when reports exist; use it to inspect failures without
  rerunning the full matrix locally.
- A module failure should be reproduced with the module command first, then the
  relevant contract validator.
- A docs failure should be checked with
  `uv run python GEO-INFER-TEST/rewrite_readme_agents.py --check` and
  `git diff --check`.

See the [API reference](api_reference.md) and the repository
[testing guide](../../GEO-INFER-INTRA/docs/developer_guide/testing_guide.md).
