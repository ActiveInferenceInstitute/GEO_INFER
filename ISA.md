---
phase: verifying
---
# GEO-INFER Ideal State Artifact

> History note (2026-09-07): the published history was rewritten to re-attribute
> hum-side personal commit identities to docxology
> <docxology@users.noreply.github.com>. Every commit SHA recorded in this
> artifact before that rewrite refers to pre-rewrite history and no longer
> resolves; the recorded evidence and run results remain valid, only the
> identifiers changed.

## Problem

At fc62502c, streaming adapters emit generated records without opening network
connections; GPU selection can lose precision or misreport hardware; required
hydrography data is absent; previews depict schematic geometry as H3 cells.

## Out of Scope

Production deployment, full-region downloads, and changes to HumOS.
Publication was authorized separately on 2026-09-04.
Real GPU execution remains a separately reported hardware verification item.

## Principles

Preserve the existing scientific and repository contracts below. Fix demonstrated
behavior with regression tests; do not suppress warnings or weaken assertions.


This artifact defines the ideal state for GEO-INFER hardening work. It is not a
roadmap by itself; it is the verification target that implementation plans and
release gates should measure against.

### Active Inference Spine

The canonical Active Inference implementation is `GEO-INFER-ACT`.

Ideal state criteria:

- ACT exposes stable typed result contracts: `FreeEnergyBreakdown`, `PolicyEvaluation`, and `ActiveInferenceStepResult`.
- Categorical free energy reports `F = complexity - accuracy` with finite normalized probability terms.
- Expected free energy reports epistemic, pragmatic, entropy, risk, and selection probability terms.
- Policy selection evaluates all candidates by expected free energy; deterministic mode selects the lowest value.
- Stochastic policy selection is seedable through `PolicySelector(random_seed=...)`.
- `ActiveInferenceModel.step(..., return_result=True)` returns a typed result without breaking tuple-return compatibility.
- BAYES and MATH support ACT instead of duplicating incompatible Active Inference contracts.
- AGENT and SIM adapters either call ACT or document equivalent expected-free-energy semantics.

Verification:

```bash
uv run python GEO-INFER-TEST/validate_active_inference_contract.py
uv run --package geo-infer-act --extra dev python -m pytest GEO-INFER-ACT/tests -q
```

### Bayesian And Math Foundations

Ideal state criteria:

- `GEO-INFER-BAYES` mean-field and full-rank variational inference paths run without guarded `NotImplementedError`.
- Full-rank VI uses a Cholesky covariance factor and finite posterior samples.
- `GEO-INFER-MATH` convenience imports do not require Flask or other web dependencies.
- Flask-backed MATH APIs are reachable through the documented `web` extra.

Verification:

```bash
uv run --package geo-infer-bayes --extra dev python -m pytest GEO-INFER-BAYES/tests -q
uv run --package geo-infer-math --extra dev python -m pytest GEO-INFER-MATH/tests -q
```

### Repo-Wide Contracts

Ideal state criteria:

- Exactly 45 `GEO-INFER-*` modules are present.
- Every module has `README.md`, `AGENTS.md`, `SKILL.md`, and `pyproject.toml`.
- Python package directories use lowercase `geo_infer_<module>` casing.
- `pyproject.toml` is the only packaging surface; module-root `setup.py`, `setup.cfg` and `requirements.txt` mirrors are retired and rejected by `validate_repo_contracts.py`.
- Optional dependency failures degrade gracefully or are reported as warnings by contract checks.
- Source-language implementation debt is tracked and driven down.

### Modular Hygiene

Ideal state criteria:

- Root `pyproject.toml`, `uv.lock`, and `.python-version` define the shared uv environment.
- Root `[tool.uv.workspace]` covers all `GEO-INFER-*` modules.
- Each module keeps at least four pytest files under `tests/`.
- Planned work is tracked in root `TODO.md` or issues, not source/test task markers.
- Importable libraries never configure process-wide logging; CLI entrypoints may configure handlers.

Verification:

```bash
uv run python GEO-INFER-TEST/validate_skills.py --check-xrefs
uv run python GEO-INFER-TEST/validate_repo_contracts.py
uv run python GEO-INFER-TEST/validate_repo_contracts.py --strict-source-language
```

### Documentation And Signposting

Ideal state criteria:

- Root `README.md`, `CLAUDE.md`, `PAI.md`, `TODO.md`, and INTRA docs describe verified commands.
- Active Inference tutorials use current `geo_infer_act` APIs and run as written.
- Module docs do not claim package casing or package capabilities that conflict with the filesystem.
- Generated assessment outputs are clearly treated as historical artifacts unless regenerated.

### Release Evidence

Each hardening pass should record:

- Baseline failures observed before changes.
- Files changed for source, tests, validators, and docs.
- Exact commands run and pass/fail outcomes.
- Known residual warnings, especially optional dependency import warnings and source-language debt.

## Goal

Complete the four open backlog capabilities and review all 45 modules against
runtime, scientific, packaging, documentation, and security contracts.

## Criteria

- [x] ISC-2026-01: Record every module baseline and final result in the review ledger.
- [x] ISC-2026-02: Real WebSocket ingestion and explicit replay pass TIME tests.
- [x] ISC-2026-03: Kafka processing acknowledges only accepted records; broker smoke is recorded separately.
- [x] ISC-2026-04: SPACE CPU kernels match references, report actual backends, and bound join memory.
- [x] ISC-2026-05: PLACE pilot ingestion preserves official IDs, geometry, topology, and checksums.
- [x] ISC-2026-06: PLACE resources and public imports work from an installed wheel outside the checkout.
- [x] ISC-2026-07: All 45 preview bundles are deterministic, use real H3 geometry, and identify illustrative content.
- [x] ISC-2026-08: Shared validators have bounded import probes and substantive wheel checks.
- [x] ISC-2026-09: Anti: raster expressions cannot access filesystem or arbitrary Python attributes.
- [x] ISC-2026-10: Changed APIs, callers, dependencies, docs, and generated signposts agree.
- [x] ISC-2026-11: Required release gates and fresh-context review have recorded outcomes.
- [x] ISC-2026-12: Anti: unavailable services or hardware are never represented as passing verification.

## Test Strategy

Run focused behavioral regressions, then all-module unit/integration/performance,
H3, model, reproducibility, packaging, source hygiene, documentation, skill, and
signpost gates. Record exact commands and distinguish assertions from environment
failures. Verify wheels outside the source checkout. Recheck timing failures in
isolation before attributing them to code.

## Decisions

- 2026-09-04: User accepted hardening plus all four backlog additions; breaking APIs
  allowed with migrations; CPU plus local services; Smith River pilot and full loader.
- 2026-09-04: Outstanding baseline runs use a detached fc62502c worktree and assert
  every package import origin. Already captured focused results remain baseline
  evidence; asynchronous timeout measurements are load-conditional.

## Verification

- Initial checkout: fc62502cce7111398b34363e9928536f9837006d, clean.
- Python 3.12.11 locked workspace installed (529 distributions); CPU exclusions match CI.
- All 45 workspace wheels built using uv build --all-packages --wheel.
- Focused baseline: TIME 423 pass; SPACE 30 pass/1 failure; PLACE 13 pass/4 failures;
  INTRA previews 8 pass. Full baseline logs: /tmp/geo-infer-baseline-fc62502c/.
- Interceptor browser probe: no extensions connected; visual browser verification deferred.


### Pre-merge September implementation evidence

- All 44 modules then in the workspace have completed baseline results: 8,229 tests, 16 failures. This includes
  completed pristine reruns for the original four timeouts. The DATA timing
  failure is load-conditional; no performance fix is claimed.
- Final complete module suites: 8,413 pass on Python 3.12.11 and 8,413 pass on 3.11.15;
  zero failures, errors or skips. The final timeout, descendant-cleanup and
  early-exit regressions were followed by complete TEST-module reruns on both
  versions (1,378 tests each).
- All 45 wheel source-code/resource inventories match current source and canonical
  metadata. All 45 wheels passed isolated installation/imports on both Python
  versions on macOS and Linux ARM64. macOS used an explicit 600-second import
  bound after recorded native 120-second timeouts; Linux required container-only
  GDAL build dependencies. Owned containers were removed.
- Real Kafka 4.3.1 broker lifecycle/delivery checks passed; owned container removed.
- USGS lower Smith River pilot: 34 native reaches, 132,920 bytes, SHA256
  `2e8c7aee125cb5a01f0f9cbd8bc9137f7ebbf6880337b4856d597afcac479247`.
- All 45 tracked preview bundles match independent regeneration byte-for-byte (44
  existing regenerated byte-identical; the new GEO-INFER-INSURANCE bundle added).
- Repository, package, documentation, skill, test, source hygiene, generated
  signpost, model, H3/geospatial and orchestration gates pass. Both model audits
  have identical deterministic hashes. Changed Python files pass Ruff formatting.
- The final fresh-context review closed early-zero-exit false passes and
  descendant leaks in both import validators, with 12 new regression cases.
  Earlier independent review findings were also fixed; migrations, all 45
  module results, commands and verification limits are preserved in the
  CHANGELOG September 5 hardening close-out section.
- [DEFERRED-VERIFY] Real GPU execution requires supported hardware. Live Leaflet
  rendering/keyboard checks require a connected Interceptor browser extension.
- [DEFERRED-VERIFY] Windows process-tree cleanup is implemented but requires a
  Windows runner; POSIX child cleanup is verified. GitNexus had no project index,
  so impact review used direct source/caller inspection with reduced confidence.
- Full regional downloads and acquisition of four unbundled regional geographic
  layers remain explicit backlog work, outside the agreed pilot delivery.


## 2026-09-02 Reconciliation

- `codex/act-categorical-runtime` was merged into `main` (branch
  `agent-ergonomics-round2`); its categorical-runtime hardening is part of the
  active state.
- A repository-wide fix wave applied the repo contract (real implementations
  instead of simplified/stub paths, deterministic RNG threading, passive
  library logging, H3 v4 API usage) across all modules; per-module outcomes are
  recorded in `CHANGELOG.md` under `[0.2.0]` (the September
  `[Unreleased]` section was folded into that release).
- The contract validators were run against the reconciled state:
  `GEO-INFER-TEST/validate_test_contracts.py --strict`,
  `GEO-INFER-TEST/validate_model_contracts.py --strict --seed 42`,
  `GEO-INFER-TEST/run_model_audit.py --seed 42 --reproducible`, and
  `GEO-INFER-TEST/validate_active_inference_contract.py`.
- Residual known gaps: full-resolution NHDPlus HR vector extracts are not
  committed (see `TODO.md` PLACE-01 evidence), and optional heavy backends
  (CuPy, Mayavi, Vaex) remain excluded from hosted-runner validation.

## Publication integration

- 2026-09-04: Public push authorized. Local hardening preserved as `3a68ce80`;
  remote `main` advanced to `b7e3a8ea` with seven commits. Merge integration
  preserves both histories and reruns validation on the combined tree.
- Earlier September test and wheel results describe the pre-merge tree;
  combined-tree evidence is recorded separately in the review ledger.
- [x] PUBLISH-01: Resolve conflicts while preserving substantive upstream behavior.
- [x] PUBLISH-02: Combined-tree tests, wheels and required gates have recorded outcomes.
- [x] PUBLISH-03: Remaining TODO items have prerequisites and acceptance evidence.
- [x] PUBLISH-04: Public outgoing diff reviewed, normal push succeeds, remote SHA matches.

- Combined module suites: 8,607 pass per interpreter (3.11.15 and 3.12.11),
  zero failures/errors/skips. Twelve merged gates and manuscript verification
  pass. Initial ANT memory-order and missing DATA aiohttp findings were corrected;
  their original failed measurements remain in the review evidence.
- Final merged wheels: 45 fresh source-matching builds and 45 completed isolated
  imports on each supported Python version. Public outgoing content audit found
  no credential patterns or added private local paths.

- Publication receipt: normal push of `b45f10832db9c4815455b79246f88a20735aca1b`
  to public `origin/main` succeeded; `git ls-remote origin refs/heads/main`
  matched local HEAD and the worktree was clean. Hosted verification remains
  the recurring CI-01 gate in TODO.md; local passes are not a hosted-CI claim.


## GNN, space and time extension — September 2026

Current extension criteria (publication receipt above remains historical):

- [x] Gaussian diagnostics satisfy F = KL - expected log likelihood and agree with analytic evidence after exact filtering; invalid dimensions/covariances fail atomically.
- [x] H3 state order is stable, sparse movement conserves probability at ordinary/pentagon/boundary cells, and dense conversion is bounded.
- [x] UTC inference records preserve sequence and reject gaps, duplicates and naive timestamps before execution.
- [x] GNN exports explicit single-factor A–E artifacts; ACT rejects unsupported semantics, preserves policy priors and conditions/propagates exactly once per timestamp.
- [x] Separate GNN and GEO environments complete an artifact round trip with a source digest, real pymdp execution and repeatable traces.
- [x] Task commits exclude concurrent unrelated edits; isolated affected-module tests, focused reruns and repository contracts have recorded evidence; remaining scope is concrete in TODO.

Anti-criteria: no source execution, silent matrix repair, invented spatial labels,
implicit timestep units, automatic conversion of continuous generators to discrete
transitions, cross-checkout import paths, or publication of unrelated GNN changes.

Verification receipt: [space/time/GNN review](GEO-INFER-TEST/docs/gnn_space_time_2026_09.md).
# Paired contract expansion and PR verification

- phase: verifying
- Baseline: GEO `634b61b987f744ffc2b5499c3a840824891c4f3d`, GNN
  `92255b125b46fd64de45e2628fce4c8e9c5f89d2`; both isolated worktrees clean.
  Fresh affected GEO and full GNN baseline suites started before implementation.
- [x] ISC-GNN-7: legacy policy evaluation conditions each observation once;
  analytic timing regressions and full ACT suite pass.
- [x] ISC-GNN-8: separate Gaussian/factored artifacts preserve declared axes,
  dynamics and policy cardinality; independent producer/consumer checks pass.
- [x] ISC-GNN-9: sparse spatial transfers conserve mass and irregular schedules
  preserve explicit action counts; bounded-allocation tests pass.
- [x] ISC-GNN-10: opt-in Step 7 preserves source provenance and existing defaults;
  missing physical metadata fails visibly.
- [x] ISC-GNN-11: pinned paired CI and Windows import-probe workflows run on
  the published revisions; actual hosted results are recorded separately.
- [x] ISC-GNN-12: both informative PRs identify ancestry, validation and remaining
  external dependencies; pushed branch SHAs match remote refs.
- Anti: preserve concurrent original checkouts, reject silent matrix repair,
  do not equate CPU fallback with physical GPU validation, and never mark an
  unavailable external verification complete.

Verification: [continuation evidence](GEO-INFER-TEST/docs/gnn_continuation_2026_09.md).

## Main integration and acquisition deadline

- Baseline: GEO `cee1b5f08c2acad435b7d8864db4461b3be58c88`, GNN
  `ffebd394b62fe300f43f5cbc99af4d454bd85098`, isolated worktrees clean.
  GEO hosted unit/integration/performance passed 8825 tests on each Python
  3.11/3.12 plus both H3 validators; GNN CI passed 3612 tests and skipped
  14 on each Python 3.11/3.12/3.13. Both remote main refs are ancestors.
- [x] ISC-MAIN-1: a parent-enforced regional acquisition deadline stops a
  slow-drip request and reaps its worker; byte/feature caps and replay pass.
- [x] ISC-MAIN-2: the merged GNN revision is pinned and both independent
  environments reproduce all four contracts at the integration revisions.
- [x] ISC-MAIN-3: required gates and hosted checks pass for the outgoing GEO
  revision; paired and portability workflows also run on pushes to main.
- [x] ISC-MAIN-4: both PRs are merged and remote main contains the reviewed
  commits; isolated local integration revisions match the remote refs.
- Anti: original dirty checkouts remain untouched; do not force-push, weaken
  tests, publish packages or mark external verification complete.

- Worker review: 51 focused acquisition tests pass on Python 3.12. Independent
  review also verified real KeyboardInterrupt cleanup and connection closure.
  The installed-wheel and complete PLACE results are recorded in the PR.

- Main integration verified: GNN `903b9c3391ce277292fd1f3d5626e5fbd66d354a` and GEO
  `6f15c1000c258accdc765c6d05be7cd49ac0b285` are the actual PR merge commits and matched
  remote refs. GEO's merge tree is identical to reviewed `2241e645`; its nine
  hosted checks passed, with 8849 tests and both H3 validators per interpreter.
  The 44-wheel workflow passed on that source tree. Subsequent documentation
  closeout retains the same packaged source; exact-main workflow evidence is
  retained by CI and linked from the merged PR.
