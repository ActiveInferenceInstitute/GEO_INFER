# Local gate

`scripts/gate.py` is the repository's one local gate entry point. An arm is a
named list of the commands one GitHub Actions job runs, copied from the
workflow sources rather than re-derived. A local arm and CI therefore measure
the same thing, and `GEO-INFER-TEST/tests/unit/test_gate_contracts.py` fails
when an arm and its workflow disagree.

```bash
uv run --frozen python scripts/gate.py --list
uv run --frozen python scripts/gate.py lint
uv run --frozen python scripts/gate.py docs
uv run --frozen python scripts/gate.py fast
uv run --frozen python scripts/gate.py gates
uv run --frozen python scripts/gate.py tests
uv run --frozen python scripts/gate.py gnn
uv run --frozen python scripts/gate.py manuscript
uv run --frozen python scripts/gate.py all
```

Every arm runs against the single locked workspace environment
(`uv sync --all-packages --all-extras --all-groups`); no arm installs a second
environment. `scripts/gate.py` puts `.venv/bin` first on `PATH`, so a bare
`python` in a copied CI command means the workspace interpreter, matching the
interpreter CI selects with `actions/setup-python`. An arm stops at its first
failing command and prints that command.

## Arms

| Arm | CI source | Commands |
| --- | --- | --- |
| `lint` | `ci.yml`, `validate` job, `Run source runtime hygiene` | `ruff check .`, `ruff format --check .` |
| `docs` | `ci.yml`, `validate` job, `Validate repository contracts` | `validate_documentation.py --strict`, `rewrite_readme_agents.py --check` |
| `fast` | `ci.yml`, `fast-contracts` job | the eight-file `--paths` selection |
| `gates` | `ci.yml`, `validate` job | repository contracts, packaging, logging hygiene, doc examples, skills, Active Inference contract, syntax, test contracts, model contracts, model audit, secret scan |
| `tests` | `ci.yml`, `test` job | `unit`, `integration`, `performance`, `system`, `--h3-migration`, `slow` |
| `gnn` | `gnn-interchange.yml`, `interchange` job | the paired GNN conformance validator |
| `manuscript` | `ci.yml`, `manuscript` job | producer verification, convergence, PDF render, `ROOT` manuscript profiles |
| `all` | every arm above | each arm exactly once, in the order listed |

`gates` wraps each validator in `GEO-INFER-TEST/record_validation.py`, exactly
as the workflow does, so a local run leaves the same receipts under
`.geo-infer-test-results/` that CI uploads.

There is no `types` arm. `pyproject.toml` declares `[tool.mypy]` and a `mypy`
dependency, but no workflow runs a type check, so a type arm would invent a
contract instead of mirroring one.

## Measured wall time

One run of each arm on the host that prepared this entry point (macOS, arm64,
workspace `.venv` already synced, uv package cache warm). "Before" is the
command sequence the repository documented before this entry point existed:
the `## Canonical gate` block of `GEO-INFER-TEST/TESTING.md`, which listed the
same validators unwrapped, ran `validate_skills.py --check-xrefs` twice, and
had no fast or GNN arm at all. "After" is the arm, CI-wrapped, in one process.

| Arm | Before | After | What the difference is |
| --- | ---: | ---: | --- |
| `lint` | 1.02 s | 0.19 s | CI paid a whole-tree `ruff check .` (0.71 s) and a diff-scoped `ruff check`/`ruff format --check` (0.31 s); the arm pays one whole-tree check plus a whole-tree format check. |
| `docs` | 4.65 s, failing | 8.08 s, green | The two validators now run once, CI-wrapped (`record_validation.py` writes receipts), and the stale `.github/AGENTS.md` signpost they reported is fixed. The extra seconds are the receipt wrapper. |
| `fast` | 177.4 s | 155.5 s | The removed duplicate `validate_test_contracts.py --strict` was 1.90 s; the eight-file selection that dominates the arm measured 175.5 s before and 155.5 s after, so per-run variance (155–176 s observed) is larger than the dedupe saving. |
| `gates` | 341.5 s for the six validators alone | bounded, see below | The arm runs the same commands, so there is no speed claim to make. |
| `gnn` | no local arm | 8.2 s | A new arm, not a faster one: it runs the pinned conformance validator against a GNN checkout at `v5.0.3`. |
| `manuscript` | no local arm | 147.1 s | Convergence check 1.1 s, PDF render 15.7 s, `ROOT manuscript-render` profile 6.2 s, `ROOT manuscript` profile 124.1 s. |
| `tests` | no local arm | not measured | The full matrix is a CI lane (45 modules x six categories); it was not run here. |

Bounded runs are recorded with the exact command, not as a green arm. The
`gates` arm was stopped at `timeout 600` inside its first validator while other
work on the host held every core, so it is not reported as green. Each of its
commands was then run alone under `timeout 300` (600 for the model gates):

| Command | Wall time | Result |
| --- | ---: | --- |
| `python -m compileall GEO-INFER-*/src GEO-INFER-*/examples` | 0.2 s | green |
| `validate_repo_contracts.py --strict-source-language --strict-import-smoke` | 76.2 s | green |
| `validate_packaging.py --strict` | 5.9 s | green |
| `validate_logging_hygiene.py` | 3.8 s | green |
| `validate_doc_examples.py` | 218.2 s | green |
| `validate_skills.py --check-xrefs --warnings-fatal` | 0.1 s | green |
| `validate_active_inference_contract.py` | 11.6 s | green |
| `validate_test_contracts.py --strict` | 5.1 s | green |
| `validate_model_contracts.py --strict --seed 42` | 4.9 s | green |
| `run_model_audit.py --seed 42 --reproducible` | 3.8 s | green |
| `gitleaks detect --source . --config .gitleaks.toml --redact --verbose` | > 600 s | not completed |

The secret scan does not finish inside a ten-minute local bound on this host;
CI allows the same command 1200 s. Two other lanes were not completed locally
for the same reason: the `tests` matrix and the `manuscript` arm's `--verify
--rerun-verification` group sweep. The four render legs (convergence, render,
`ROOT manuscript-render`, `ROOT manuscript`) were completed, and the render
produced `output/pdf/GEO-INFER_combined.pdf` from the 11 published sections.


## What an arm does not cover

A hosted runner supplies things this host has or has not. These CI surfaces
have no arm because they need a hosted-only dependency; they are listed so a
green local gate is never read as a green CI:

- `test` job coverage floors (`check_coverage_floor.py --base/--head`) need the
  event's base commit; the module-list measurement is a CI event artifact.
- `build-smoke` builds and installs all wheels outside the checkout, with both
  interpreter legs.
- `process-contracts-windows` runs on `windows-latest`.
- `optional-data-services` needs disposable PostGIS, Redis and MinIO services.
- `import-probes.yml`, `dependency-audit.yml`, `companion-releases.yml`,
  `container-profiles.yml` and `release.yml` own their own environments.
- `workflow-lint` runs `actionlint`, which CI installs from a pinned release.
- The secret-scan command is in `gates`, but the pinned `gitleaks` binary is a
  host prerequisite; the arm fails with the required version rather than
  skipping the scan.

## GNN companion conformance

The `gnn` arm mirrors the required fleet check, not a new contract:

- Pin file: `.github/gnn-pair.json`, schema `gnn-importer-pin/v1`.
- Pin: `ActiveInferenceInstitute/Generalized_Notation_Notation` at `v5.0.3`,
  revision `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`.
- Workflow: `.github/workflows/gnn-interchange.yml`; job key `interchange`;
  the required check name `gnn conformance (pinned)` is the `3.12` matrix leg,
  and the `3.11` leg reports as `gnn interchange (3.11)`.
- Caller: `ci.yml` job `paired-interchange` (`uses: ./.github/workflows/gnn-interchange.yml`).
- Validator: `GEO-INFER-TEST/validate_gnn_interchange.py --gnn-repo … --gnn-python … --output …`.
- [`.github/GNN.md`](../../.github/GNN.md) records what the job parses and what
  it deliberately does not cover.
- `.github/companion-releases.json` (schema `geo-infer-companion-releases/v1`)
  is the release-facing companion identity (`5.0.1`), not the importer pin. A
  reviewed importer pin may run ahead of the published companion, and neither
  file substitutes for the other.

The arm needs a GNN checkout at the pinned revision and that checkout's own
interpreter with the `geo-infer` extra:

```bash
export GEO_INFER_GNN_REPO=/path/to/Generalized_Notation_Notation
uv sync --project "$GEO_INFER_GNN_REPO" --locked --python 3.11 --extra dev --extra geo-infer
uv run --frozen python scripts/gate.py gnn
```

It verifies the pin file, the checkout revision, the receipt's success flag,
both recorded revisions, execution custody, every retained artifact digest, and
a clean worktree pair. `--allow-dirty` records a non-publication run instead of
failing on a dirty tree.

This repository's workflows use `astral-sh/setup-uv` directly; they do not use
the shared `ActiveInferenceInstitute/.github/.github/actions/setup-python-uv`
composite action, so this gate documents no pin for it.

## Manuscript render

The `manuscript` arm runs the repository's own producer and render path:

- `manuscript/generate_research_artifacts.py` is the only producer of
  variables, captions, the figure registry and resolved manuscript copies.
- `scripts/render_manuscript_pdf.py` is the only render script; it hydrates the
  evidence bundle, combines the published sections in `manuscript/SYNTAX.md`
  order, and runs Pandoc, `pandoc-crossref` and XeLaTeX fail-closed.
- The arm requires `pandoc`, `pandoc-crossref` and `xelatex` on `PATH`; the
  render script names any missing tool and fails.
- The two `ROOT` profiles run after the PDF exists: `manuscript-render` reads
  the produced artifacts, `manuscript` is the ordinary regression profile.
  Running them before a render measures the CI job's own ordering, so the arm
  keeps it.

## Adding an arm

Add the commands to `ARMS` in `scripts/gate.py` with the workflow step that
owns them, then extend
`GEO-INFER-TEST/tests/unit/test_gate_contracts.py`. A command that is not in
the workflow, or a gate that appears in two jobs, fails that test.
