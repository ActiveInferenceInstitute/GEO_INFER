# GEO-INFER Open Task & Backlog Ledger

> Last reviewed: 2026-10-05
> Scope: Multi-package repository (`GEO-INFER`) across workspace packages and 45 domain modules.
> Centralization Rule: All planned, open, or deferred engineering work across all modules is tracked exclusively in this ledger. Module source code and tests must never carry local task markers (`TODO`, `FIXME`, `XXX`, `HACK`).
> History note (2026-09-07): the published history was rewritten to re-attribute
> hum-side personal commit identities to docxology
> <docxology@users.noreply.github.com>. Every commit SHA recorded in this
> ledger, in CHANGELOG.md and in ISA.md before that rewrite refers to
> pre-rewrite history and no longer resolves; the recorded evidence and run
> results remain valid, only the identifiers changed. Future contributions
> from this checkout are authored as docxology.

---

## Open work and acceptance criteria

Completed implementation and deferred verification are tracked separately below.
Module names identify the responsible area, not an assigned person. Acquisition
items require an explicit area, source and resource budget before running them;
this ledger does not authorize full-region downloads or a package release.
Rows are tiered: **Major** needs external resources or is a release-scale
decision, **Medium** is self-serve multi-session engineering, **Minor** is a
bounded single-session change. Re-tiered 2026-09-07 from a nine-scout ledger
audit: the previously open rows verified current against the tree, delivered
claims were spot-checked, and module/package surfaces were swept for untracked
next steps.

### 0.4.0 continuation — current scope (2026-10-05)

**Execution resumed:** on 2026-10-05 the owner authorized comprehensive issue
fixes, verification and an ordinary main push. Candidate acceptance remains
required before integration. Tagging, release attachments, registry publication
and deployment retain their separate authorization boundary; accounts,
credentials, configured models and hooks remain under human control.

At the resumed verification checkpoint, preparation branch `codex/0.4-readiness`
was pushed at `a254a2856e4c1489ee31ea6c0183542d5db18967`; remote `main` remains
`510f1008e698b45aedc4306d65caf898da6cda04`. The local legacy sweep ancestor
`cbe6b9c59975254487452d697c2ea961a19f2286` must be preserved. **Readiness is
NO GO.** The last candidate's hosted TEST coverage was 72.22955% against the
unchanged 85% floor; other successes remain bound to their original revision.

At implementation resume there were 39 pending implementation paths: root and 36 member coverage
configurations, the TEST tracing regression, and the owning testing guide.
Their correction removes only the erroneous `*/test_*` source omission.
The 37-configuration × three-filename matrix passed 111 cases in each of two
narrow recording profiles. The original narrow inner SQLite traces were not
retained; the later whole TEST diagnostic retained all 111 fresh traces.

The whole diagnostic **completed naturally with FAIL**, before a requested stop
could be sent: **1,815 passed / one failed / 1,816 executed**. The ordinary
validator control
`test_validator_arguments_do_not_advertise_pytest_or_report_promises[arguments1]`
recorded a real TIMEOUT (5-second envelope, 5.799 seconds elapsed, return code
`None`) despite emitting its expected output. This establishes a timeout to
investigate, not a classifier defect or an environmental cause. Independently,
raw coverage was **2,529/3,032 = 83.41029%**, still below 85%. The attempt is
diagnostic evidence, not accepted TEST/floor evidence. No signal was issued;
the observed owned processes were absent after completion. Original manuscript
output remains an older failed-candidate bundle.

Continuation repairs now separate census failures from pipe polling, enforce the
original deadline around final ownership inspection, retain admitted parallel
worker outcomes through repeated shutdown interruption, and reject empty runner
selections. Focused regressions establish these defects independently of the
historical timeout, whose cause remains unresolved. The manuscript custody path
now rejects root/directory/file replacement during validation; its focused
independent review accepted the declared boundary. Native Windows PLACE workflow
implementation is present, with hosted runtime acceptance still pending. These
are implementation and narrow-test results; whole-package coverage and final
candidate acceptance remain required.

The resumed clean-candidate whole TEST attempt reached its 1,800-second deadline
without final JUnit or raw coverage output. Source stayed fixed, all 111 policy
traces were retained, and its owned processes exited. This is failed acceptance;
the progress stream contains failures whose complete dispositions require fresh
diagnosis. Aggregate native census time was 1,517 seconds over 1,107 calls.
A synthetic parser assessment demonstrated avoidable token-negative regex work;
the resulting prefilter passed 53 focused ownership tests and independent
review; renewed whole-package acceptance is still required. These observations
do not establish the historical ordinary-validator timeout's cause.

Both initial native Windows PLACE legs failed before collection because the
canonical runner injected uninstalled optional DATA source into the minimal
PLACE profile, exposing its absent SQLAlchemy dependency. The corrected workflow
uses the existing execution API with observer-only `PYTHONPATH`, a separate real
poisoning negative control and both complete acceptance files. Its 22 structural
and custody tests passed. The separate locked macOS Python 3.12 profile reproduced
the negative failure and passed all 51 positive cases plus custody. Independent
source review accepted both scoped repairs; native runtime proof remains pending.
Hosted artifact reconciliation also identified Windows LF-to-CRLF lock conversion;
an explicit LF attribute and real Git checkout regression preserve raw lock
custody. The updated workflow/metadata suite passed 68 tests. Fresh native receipt
hashes must still be checked on the renewed candidate.

The next whole TEST attempt completed all 1,953 selected cases with source
unchanged and all 111 policy traces retained: 1,951 passed and two deadline/census
controls failed. Raw coverage was 2,921/3,325 = 87.84962%, above the unchanged
85% floor, but failed tests still reject the measurement. The controls assumed
the child exited before its first 250 ms pipe poll. Their correction observes
actual target completion, distinguishes preliminary/final/cleanup scans and
forces a real incomplete poll; production deadlines remain unchanged. All four
focused cases passed canonical receipt acceptance. Fresh independent review
accepted the repair and confirmed that a retry mutation fails both parameter
cases. Renewed whole TEST proof remains required.

The next native Windows PLACE legs reached all 51 cases on Python 3.11/3.12:
48 passed and three failed on converted fixture bytes or default text decoding.
Both sealed artifacts reconciled with no hash errors and the exact raw lock;
all ten observed workers per leg were reaped with closed streams. Exact archived
source attributes and LF replay attributes preserve the existing hash contracts;
the renderer test now reads UTF-8 explicitly and exercises a cp1252 default.
Local testcases passed, but canonical acceptance rejected an owned-process leak
in the new temporary Git checkout regression. A focused diagnostic identified
the inherited Git fsmonitor setting: the test started an owned background daemon.
Command-local fsmonitor disabling in this temporary checkout preserved real
autocrlf conversion and passed canonical acceptance with no owned processes or
daemon artifacts. Final local locked-profile acceptance passed all 52 cases
with sealed receipt hashes and all ten observed workers reaped with closed
streams. Fresh final independent review accepted the scoped repair; renewed
native Windows acceptance remains required.

On clean candidate `69ed0121601f026bc16debbe736d657199468de9`, whole TEST
passed all 1,954 cases with 2,922/3,325 = 87.879699% raw coverage, all 111 policy
traces, historical inventory retention and unchanged source. Both native Windows
PLACE legs also passed. Their artifact reconciliation remains separate. Full CI
exposed one forced-poll process-control assertion requiring a remaining budget
strictly below the original budget; the preliminary scan can occur before the
monotonic clock advances. Its test-only correction permits equality while still
rejecting budget extension. All four focused cases passed canonical acceptance.
Fresh independent review accepted the scoped bound and confirmed prior retry
mutation evidence still applies. Renewed exact-candidate acceptance remains required.

The [readiness tracker (#61)](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/61)
links the complete next-work graph. Dependencies govern completion; preparation
may overlap after a stable source freeze with bounded resources and explicit
source/output ownership. Resumed work starts with the two independent blockers
below; their historical failed evidence remains unchanged.

| ID | Owning area / status | Complete bounded next step | Acceptance / dependencies |
| --- | --- | --- | --- |
| **R04-00** | Repository / NO GO tracker | [#61: exact-candidate readiness](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/61) coordinates every required child and separate deferred scope. | One clean revision binds local, hosted, wheel, paired, security and manuscript evidence; successful required children precede main integration. |
| **R04-01** | TEST / demonstrated timeout, cause unresolved | [#62: ordinary-validator timeout](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/62): inspect actual child/parent timing, ownership census, cleanup and receipts; repair only a demonstrated contract problem. | Real command outcomes, fresh nonzero JUnit where promised, validators without JUnit, invalid/empty selections, POSIX/Windows descendants and interruption; no automatic retries, suppressed errors or production-budget workaround. Fresh independent shared-infrastructure review. |
| **R04-02** | TEST / source omission fixed locally; coverage below floor | [#63: real runner behavior and 85% floor](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/63): review policy fix; add meaningful public runner/error/cancellation behavior coverage. | Preserve the 116 existing core behavior identities and 111 policy controls; fully passing whole TEST measurement with genuine raw coverage ≥85%. Complete all 46 final package/ROOT dispositions and inventory reconciliation. Depends on R04-01. |
| **R04-03** | Repository, SPACE/TIME/ACT and connected modules / final local proof pending | [#64: strict and analytical acceptance](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/64): freeze reviewed source and complete all declared local contracts/categories, real composition oracles, security and reproducibility; reduce measured fixture/setup cost without losing coverage. | UTC/DST/ns/step/gap/order/ownership/allocation contracts; real DATA→SPACE→TIME, BAYES/ACT permutations, seven-cell/pentagon transitions/posteriors/policy, IOT coordinate oracle, ART hostile input, OPS port cleanup and async lifecycle. Strict repository/package/docs/skills/test/model/logging/lint/format/workflow/secret gates. Depends on R04-01/R04-02. |
| **R04-04** | Packaging and GNN / fresh exact-candidate proof pending | [#65: all wheels, optional boundaries and paired interchange](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/65). | 45 wheels, 45 base and 20 operation profiles outside checkout on 3.11/3.12; metadata/resources/extra parity, real signed DuckDB fast path; exact pinned GNN four-family/exporter/replay evidence. Stable candidate from R04-03. |
| **R04-05 / CODE-01** | Documentation, manuscript, index / convergence and full proof pending | [#66: executable docs and manuscript](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/66): finish truthful API/migration/configuration guidance, generated signposts, 0.4.0 parity, Git-derived metadata fixed point and source-current GitNexus. | Coordinator-only full eleven-group producer, independent whole-output custody/snapshot, real native PDF/figure inspection, generator drift checks and fresh ROOT inventory (currently 142+7). Retain producer manifest/inventory/variables/figure-registry JSON or state hosted upload limits. No old-output or default-seven-tier passing claim. Depends on stable R04-03 source. |
| **R04-06** | CI / fresh corrected-candidate evidence pending | [#67: complete hosted evidence](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/67): run declared workflows once on the reviewed candidate and audit actual receipts/artifact bytes. | Required unit/integration/performance/system/slow/H3 matrix on 3.11/3.12, native Windows and early contracts; actual profile/inventory reconciliation, all 46 floors, wheels/paired/ROOT/models/security and complete uploaded artifacts. Derive fresh counts; preserve failures. Depends on R04-01 through R04-05. |
| **R04-07** | Git/main / held, not integrated | [#68: accepted main integration](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/68): when implementation is resumed and candidate GO exists, recheck remote/protection/ancestry, integrate ordinarily and verify exact published/main-triggered evidence. | No force-push, ancestry rewrite, stale-SHA acceptance or protection bypass; preserve rollback and unrelated work. Main SHA parity plus actual same-SHA hosted results. Depends on R04-03 through R04-06. |
| **SPACE-01** | SPACE / owner-deferred GPU work | [#69: physical backend validation](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/69). GPU experiments are on hold; future work must be usable from a Mac laptop. | Establish a backend compatible with the Mac target and the declared precision contract before physical numerical parity, failure/chunk boundaries, memory and cold/warm timings. Preserve laptop CPU operation; H3 remains host CPU. No CPU-fallback hardware claim. |
| **PLACE-V14** | PLACE / deferred licensed data | [#70: complete Cascadia boundary](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/70). | Authorized source/license/extent/resource budget, provenance/checksum/WGS84/stable identities and real renderer/integration; missing layer remains explicit and fail-closed. |
| **PLACE-04** | PLACE / deferred native Windows worker proof | [#71: real Windows download worker](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/71). | Locked PLACE runtime, stalled-header/slow-drip deadlines, termination/pipe closure, batch failure and replay. Shared-process Windows CI does not close this separate item. |
| **TEST-GNN-01** | SPACE/TIME/ACT / historical cause unresolved | [#72: PROJ SQLite I/O causality](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/72). | Bounded evidence-driven reproduction with version/import/file state; respect prior negative lock probes and distinguish non-recurrence from explanation. Do not suppress CRS tests. |
| **TEST-04** | TEST / deferred configured-advisor review | [#73: advisor review](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/73). | Human-owned account/authentication decisions; actual configured-service review bound to the candidate when available. No auth/key/model/billing/hook changes or review claim from service errors. |
| **REL-04 / REL-01** | Release / separate future authorization | [#74: 0.4.0 release decision and execution](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/74). | Explicit tag/release/attachment/registry authorization after exact-main readiness; verify destination bytes and retain rollback. Main preparation does not authorize publication or deployment. |
| **ACT-MATH-58** | ACT/MATH / separate future feature | [Existing #58: reviewed skew/circulation/Helmholtz constructs](https://github.com/ActiveInferenceInstitute/GEO_INFER/issues/58). | Scope actual numerical APIs, transition/rate/current conventions and independent mathematical fixtures before notation-bridge symbol binding. Current diagnostics/internal symmetrization do not establish those APIs. Not a required 0.4.0 dependency. |

Implemented preparation changes remain recorded in [CHANGELOG.md](CHANGELOG.md);
do not reopen completed work without a concrete source/behavior finding.
The older standing rows below retain their historical evidence. The current
CODE-01 next step is R04-05; the current release boundary is R04-07/REL-04,
not the historical 0.2.x/0.3.x actions described in REL-01.

### Major — external resources or release-scale decisions

| ID | Area / status | Bounded next step | Acceptance evidence / dependencies |
| --- | --- | --- | --- |
| **SPACE-01** | SPACE / owner-deferred GPU verification | [DEFERRED-VERIFY] GPU work is on hold by owner direction (2026-10-06). Future design must be runnable from a Mac laptop, retaining CPU operation. First establish a compatible backend and precision contract; then run numeric distance and grouped-reduction parity for each backend claimed as supported, including empty inputs, float64 precision, chunk boundaries and allocation failure. Keep H3 topology labeled as host CPU. | Record device/driver/library versions, actual backend diagnostics, CPU-reference tolerances, peak memory and separate cold/warm timings. Publish speed claims only for measured workloads; do not infer support from CPU fallback or claim Apple GPU float64 support without evidence. [Guide](GEO-INFER-SPACE/docs/GPU_ACCELERATION.md). No external GPU machine or budget is requested while work is on hold. |
| **PLACE-V14** | PLACE / regional layer acquisition open | Three source-backed layers are delivered (13 HU4 display polygons, 24 volcanoes, one convergent boundary). Obtain the remaining complete licensed `cascadia_bioregion_boundary.geojson`; retain the documented per-layer extent and interpretation. | Validate WGS84, required geometry types, stable feature identifiers, provenance and checksums; run actual-data renderer/integration checks (missing-layer behavior is fail-closed and pinned at `test_regional_layer_acquisition.py:73,81`). Keep missing-layer behavior explicit until data exists. Do not restore the former 12-volcano or earthquake-probability claims without evidence. 2026-09-15 scope pass: fail-closed pin re-verified — the assertions now sit at `GEO-INFER-PLACE/tests/integration/test_regional_layer_acquisition.py:76-80` (this row's `:73,81` citation has drifted); boundary geojson still absent from the repo; remains open on licensed-data acquisition. |
| **PLACE-04** | PLACE / deferred Windows verification | [DEFERRED-VERIFY] Run the real regional download-worker loopback tests on Windows with the locked PLACE runtime. | Prove stalled-header/slow-drip deadlines, native process termination, pipe closure, batch failure preservation and exact replay on Windows; retain interpreter/OS versions. POSIX termination is verified and the worker starts no child processes. 2026-09-15 scope pass: POSIX worker surfaces verified present (`GEO-INFER-PLACE/tests/integration/test_regional_download_worker.py`); remains open, Windows-runtime-blocked. |
| **CODE-01** | Repository / recurring index refresh | [REFRESHED 2026-09-15] Incremental `node .gitnexus/run.cjs analyze` on `main` at `0e6a47d4` (~100 s, incremental path healthy this time — no invalid-UTF-8 failure, no forced rebuild): index updated in the worktree `.gitnexus/` store (gitignored; 67,245 nodes, 96,848 edges, 1,824 clusters, 300 flows, no embeddings; same three files skipped >512 KB) with `.gitnexus/meta.json` `lastCommit` now at the tip. The analyze's block injection into root `AGENTS.md` was again stripped by hand per the cadence note, and `CLAUDE.md`'s committed block was count-refreshed only (65872/93811 → 67245/96848). | Indexed/current-commit parity plus correct explicit-file Gaussian-contract (the GNN-repo exporter is outside this index) and sparse-transition lookups verified at the receipt SHA as recorded; direct source/caller review remains the documented fallback while no index exists. Recurring cadence: re-run `gitnexus analyze .` after major refactors or when `gitnexus status` reports stale, restoring generator-owned AGENTS.md/CLAUDE.md afterwards. 2026-09-15 scope pass: freshness residual — the tip has advanced 9 commits past the indexed `0e6a47d4` (`git rev-list --count 0e6a47d4..HEAD`), including ErrorHandlerMiddleware source in DATA/NORMS/GIT/PEP (`3f67344b`, `2cbb6782`, `aff9776e`), starlette dependency declarations (`1037bc7b`), LOG endpoint tests (`ef3db992`) and two test-flake repairs (`8d3cb5e2`, `f5e5baf1`), so the index no longer reflects the source surface; refresh due before the next index-reliant task. |
| **REL-01** | Repository / release authorization and execution | Prep complete 2026-09-10: the `[Unreleased]` content folded into `## [0.2.0] - 2026-09-10` (Keep-a-Changelog fold; February content retained as a subsection; link refs updated), GEO-INFER-INSURANCE promoted 0.1.0→0.2.0 (pyproject `version`; Development Status kept at 3-Alpha to match the fleet classifier majority; the `KNOWN_VERSION_DEVIATIONS` entry removed), `validate_packaging --strict` green (45 modules, 0 errors/warnings). 2026-09-15 truth correction: the release happened — local tag `v0.2.0` was created 2026-09-11 20:01:41 -0700 at `0a30df63` (merge of PR #31; contained in `main` and `origin/main`), the pushed tag fired `release.yml` to completion (run [34669262158](https://github.com/ActiveInferenceInstitute/GEO_INFER/actions/runs/34669262158), head `v0.2.0`, concluded success 2026-09-12T03:01Z after an earlier failed attempt at `db400e3b` fixed by PR #31), and the prior "tagging remains withheld pending go/no-go" text was stale. Remaining act: the deliberate go/no-go on the post-tag delta — 18 commits have landed since `v0.2.0` (`git rev-list --count v0.2.0..HEAD`), including fleet-wide ErrorHandlerMiddleware source in DATA/NORMS/GIT/PEP (`3f67344b`, `2cbb6782`, `aff9776e`), starlette dependency declarations (`1037bc7b`), LOG endpoint contract tests (`ef3db992`) and two statistical-flake repairs (`8d3cb5e2`, `f5e5baf1`) — either cut `v0.2.1` (fold the `[Unreleased]` CHANGELOG entry, bump versions per the fleet classifier, `validate_packaging --strict`, tag) or record the decision to carry the delta to a later release. 2026-09-15 GO decision (GS15-03): PATCH RELEASE STAGED, TAG DEFERRED TO ORCHESTRATOR — the 18-commit delta since `v0.2.0` is 5 user-facing bug-fix commits (LOG-EXC-01 router narrowing `0125d397`; ErrorHandlerMiddleware + detail-leak 500 elimination in DATA/NORMS/GIT/PEP `3f67344b`/`2cbb6782`/`aff9776e`), packaging dep declarations (`1037bc7b`) and test/docs-only commits; no features, no breaking changes, so SemVer patch. Staged: `[Unreleased]` folded into `## [0.2.1] - 2026-09-15` (delta summarized, `[0.2.1]` compare ref added), all 45 member pyprojects + root bumped 0.2.0→0.2.1 per the fleet-uniform classifier rule (`validate_version_uniformity`; the 0.2.0 precedent promoted the single outlier INSURANCE to the fleet majority — a new uniform fleet version moves all members together, and partial bumps are warnings that `--strict` turns into errors), `uv.lock` re-locked offline (46 member-version lines; CI runs `uv sync --locked`), generated README/AGENTS version metadata regenerated (`rewrite_readme_agents.py --check` green: 1629 files current), `validate_packaging --strict` green (45 modules, 0 errors/0 warnings). No tag created — `git tag -l v0.2.1` is empty; tagging, push and the `release.yml` run remain with the orchestrator. 2026-09-26 audited correction: the staged `v0.2.1` was subsequently tagged and its release run succeeded (run [35053351237](https://github.com/ActiveInferenceInstitute/GEO_INFER/actions/runs/35053351237), 2026-09-16) but left no GitHub release object; **v0.3.0 (tag `07a5fe31`, 2026-09-17) published NO wheels** — release.yml run [35274969824](https://github.com/ActiveInferenceInstitute/GEO_INFER/actions/runs/35274969824) concluded failure (its ci-gate observed the same-SHA tag-push CI run 35274969748 fail the coverage-floor step; root cause = the tag-event diff-scope divergence, spec CI-02), and the v0.3.0 release object still carries only the manuscript PDF as a binary asset (re-verified 2026-09-26 via the GitHub API). Durable in-tree fixes landed (GS19-01 version-only hunk filter + bounded FAILED-SUITE retry `9267f33d`; CI-02 tag/zero-SHA empty-tree base `74c17323`). 2026-09-27 CLOSURE (owner-authorized): the tag-push validate job re-run (attempt 2) → success; release run 35274969824 re-run → success (ci-gate + release); the 45 wheels were published as the run artifact `geo-infer-wheels` — release.yml uploads wheels ONLY as a workflow artifact, never to the release object (same design at v0.2.0, so that release's 45 assets were attached owner-side) — and the coordinator attached them to the v0.3.0 release per the owner's authorization: 46 assets (45 `.whl` + manuscript PDF). **CI-01 ≙ GS19-01 CLOSED.** Residual pipeline gap filed as **REL-02**. [SCOPE-2026-09-19.md](SCOPE-2026-09-19.md) Disposition (2026-09-26). |

### Medium — self-serve, multi-session

| ID | Area / status | Bounded next step | Acceptance evidence / dependencies |
| --- | --- | --- | --- |
| **TEST-GNN-01** | TEST / Python 3.12 PROJ SQLite disk-I/O cause | Investigate the failure observed during the combined ACT/SPACE/TIME test process; a fresh integrity probe and all 587 SPACE tests passed separately, and the continuation receipt records non-recurrence with versions (Python 3.12.13, pyproj 3.7.1, PROJ 9.5.1, SQLite 3.53.1) — establish the historical cause, not just current success. | Minimal import/order reproduction, loaded PROJ/GDAL/SQLite versions and file-descriptor state; correct a reproducible cause without suppressing CRS tests or declaring an unverified environment fix. 2026-09-10 bounded investigation delivered (dated section in the continuation receipt): the exact historical version combination reproduced clean across 400 CRS iterations, concurrent-process stress clean — leading hypothesis transient concurrent-access contention on the shared embedded proj.db; exact trigger honestly unexplained. 2026-09-11 follow-up probe (dated section in the receipt): combined ACT/SPACE/TIME CRS subsets in one pytest process ×8 (1,557 tests), intra-process thread hammering (~42,800 CRS ops) and a simultaneous pytest+thread storm — all clean on the current build (Python 3.12.11, SQLite 3.50.4); this weakens the internal-concurrency half of the hypothesis, while `journal_mode=delete` on the shared in-worktree proj.db plus live-observed co-tenant worktree edits during the clean runs keep external interference a demonstrated always-present factor; 2026-09-15 induced-lock probe (dated section in the receipt): a verified external SQLite `BEGIN EXCLUSIVE` lock on the shared embedded proj.db did NOT reproduce — database-bound CRS creation and authority lookups succeeded under the lock while ordinary SQLite readers in the same process were blocked (`database is locked`), so SQLite-level lock contention is ruled out as the trigger for PROJ CRS reads on this build (libproj's lock-bypass mechanism itself unresolved) and the causal chain narrows to I/O-class/file-content-level external interference (the historical text is a disk-I/O/SQLITE_IOERR-class error, orthogonal to the SQLITE_BUSY class); still open (no reproducer; induced-lock minimal case negative). [Receipt](GEO-INFER-TEST/docs/gnn_continuation_2026_09.md). |

### Minor — bounded single-session changes

| ID | Area / status | Bounded next step | Acceptance evidence / dependencies |
| --- | --- | --- | --- |
| **TEST-04** | TEST / advisor review repeat | [DEFERRED-VERIFY] The configured advisor exited with an error during the September GNN campaign so no advisor review ran (gnn_space_time_2026_09.md): repeat the advisor review when the service is available. | 2026-09-10 probe: the configured advisor is Cato via Codex CLI 0.153.2; ChatGPT-account auth rejects the gpt-5.2 slug (HTTP 400) and no OPENAI_API_KEY is configured, so no review could run — probe appended to the continuation receipt. Unblocking is an account-level auth decision (codex login --with-api-key). 2026-09-11 re-check: environment still exposes no OPENAI_API_KEY and the CLI auth state is unchanged — remains blocked on the account-level decision. 2026-09-15 re-check: environment still exposes no OPENAI_API_KEY — remains blocked, unchanged. |

## Completed-record reset (2026-09-11)

The delivered-capabilities table, the completed-audit log and the dated
receipt sections were cleared on 2026-09-11 as part of the repo-wide
scoping reset that precedes the fresh minor/medium/major pass over the
package and all 45 modules. All prior records remain restorable from git
history, and standing receipts live in `GEO-INFER-TEST/docs/` and the
CHANGELOG.
The fresh pass itself lands as `SCOPE-2026-09-11.md`.

## Campaign execution record (2026-09-11)

The full 189-item spec was executed the same day in six reviewed waves,
each pushed with CI green (probe-confirmed bug fixes; 82 silent-fabrication
fixes across 36 modules; packaging coherence with a single integration
re-lock; 21 test-gap closures; 29 docs/API-honesty fixes including the new
`validate_doc_imports.py` gate; 31 CI/structural items via
[PR #28](https://github.com/ActiveInferenceInstitute/GEO_INFER/pull/28)).
The campaign is recorded in CHANGELOG under
`## [0.2.0] - 2026-09-11 - repo-wide quality campaign`. All spec items are
delivered; the ledger's open rows below are the survivors.

---


## GNN interoperability follow-up (September 2026)

TEST-GNN-01 (Python 3.12 PROJ SQLite disk-I/O cause) is tiered **Medium** in
the open-work table above with the continuation-receipt evidence; this section
is retained only as a pointer for existing links.


## Scope pass (2026-09-15)

Docs-only truth pass at `main @ f5e5baf1` (clean tree; no code changes):
every open row above was re-verified against the tree and re-stated with
current evidence. Method, per-row dispositions and the fresh scoping items
live in [SCOPE-2026-09-15.md](SCOPE-2026-09-15.md). Disposition summary:
SPACE-01, PLACE-V14 and PLACE-04 verified current and remain open on their
external blockers (hardware, licensed data, Windows runtime); CODE-01
verified current at `0e6a47d4` with a recorded 9-commit freshness residual;
REL-01 corrected — the `v0.2.0` tag exists and its release run succeeded,
so the residual is the go/no-go on the 18-commit post-tag delta; TEST-GNN-01
already carried the 2026-09-15 induced-lock probe outcome (SQLite lock
contention ruled out, hypothesis narrowed to I/O-class faults) and needed no
edit; TEST-04 re-checked and still blocked on the account-level advisor auth
decision. No rows were cleared this pass.

## Scope pass (2026-09-19)

Fresh 14-lane read-only scoping swarm against `main @ 07a5fe31` (v0.3.0 tip;
CI green at tip, run 35268910816): five package-level lenses (CI/gates,
packaging, docs-truth, test-estate, pipeline/perf/security) + nine module
batches over all 45 packages. Method, items, tiering and probes live in
[SCOPE-2026-09-19.md](SCOPE-2026-09-19.md) — 78 items (1 Major, 14 Medium,
63 Minor); no prior-scope items re-opened; no TODO rows cleared or edited
this pass (the CI-01/REL-01 row update below lands with the first wave's
ledger commit).

Headline: **REL-01 residual changed.** v0.3.0 (tag at `07a5fe31`, local +
remote; CHANGELOG/CITATION.cff/manuscript consistent) published **no
wheels**: release.yml run 35274969824 concluded failure because its ci-gate
observed the tag-push CI (run 35274969748, same SHA) fail the
coverage-floor step — root cause is the tag-event `BASE_SHA` diff-scope
divergence (spec item CI-02). The v0.3.0 release object carries only the
manuscript PDF vs v0.2.0's 45 wheel assets, and tag `v0.2.1` has no GitHub
release object at all. Bounded path: land the CI-02 fix, then a
user-authorized release.yml `workflow_dispatch` re-run at `07a5fe31` (or a
dated no-wheel decision). Also recorded this pass: GNN pair-pin drift
(CI-05 — bump only through the SC-22 paired-custody ritual), test-contract
validator scan gaps (TST-01..03), and the full 78-item inventory.

## Wave closeout (2026-09-22)

The CI-01/REL-01 row update promised above, plus the supplement-wave
disposition, recorded at branch `wave/scope-2026-09-19` after reconciling
`origin/main` (merge `7f0dcdf7`).

- **REL-01/CI-01 (release)**: the durable half is landed — GS19-01's
  version-only hunk filter plus one bounded FAILED-SUITE retry in
  `GEO-INFER-TEST/check_coverage_floor.py` (commit `9267f33d`), and CI-02's
  tag-event diff-scope fix (commit `74c17323`). The recovery half remains
  open and owner-gated: v0.3.0 still ships zero wheels (run 35274969824,
  failure, no re-attempt); unblocking requires an authorized re-run of the
  failed validate job then release.yml's ci-gate, now complicated by the
  org migration (below). Row stays open on that external action.
- **Org migration completed in-tree**: the repository is
  `ActiveInferenceInstitute/GEO_INFER` (GitHub PR #34, merge `c3c8854a`);
  this branch swept the residual old-slug references (commit `771e1a42`),
  repointed the paired GNN interchange to
  `Generalized_Notation_Notation` @ `4b50307cb` through the merge, and the
  local remote now points at the underscore slug.
- **SCOPE-2026-09-19 supplement executed**: all 88 GS19 items are landed or
  verifiably already satisfied — the final residue lane (13 minor items +
  TST-08/M4-06) and the closeout wave (GS19-15/16/17/18/19/20/65/67/71/88
  plus the previously-missing halves of 38/51/63/84) landed in commits
  `3a1a135c`..`8f3e0154` on this branch. Wave P (GS19-04/05/28/29/34/36/39/74
  + the scheduled slow lane) landed via PR #35 (merge `6f8fd263`); the wave
  branch itself was never pushed as a branch and is obsolete — 2026-09-26
  reconciliation verified its only tree delta vs `main` is the stale
  `.github/gnn-pair.json` pin (main's newer cycle-29 pin wins).
- **Still open, external-blocked**: SPACE-01 (hardware), PLACE-V14
  (licensed data), PLACE-04 (Windows runtime), TEST-04 (advisor auth);
  TEST-GNN-01 remains Medium with its recorded investigation.

## Reconciliation pass (2026-09-26)

Fresh audit of both 2026-09-19 specs (78 main items + 88 GS19 items) against
`origin/main @ 5a1e3801` by two read-only audit lanes. Verdicts: 158 of 166
item-records DELIVERED (73 + 85; unique set is smaller after the addendum
overlap mapping). Six unique OPEN items and one EXTERNAL action remained;
all six OPEN items landed the same day (receipts below):

- **RESOLVED the same pass (2026-09-26/27)** — all six OPEN items landed:
  PR #40 (merge `a55a0460`) = this ledger + the DOC-05 stamp drop (main
  ci.yml run 36285058396 success); PR #41 (merge `fbf5681b`, lane t-0001
  `test-guards`) = TST-03 + M1-05 (155/155 touched-file tests green; local
  CI-gate battery replicated: diff-scoped ruff check + format, repo-wide
  src/tests surfaces, `validate_test_contracts --strict`); PR #42
  (merge `565aa438`, lane t-0002 `dep-truth`) = M5-03 + GS19-50 + GS19-87
  (cli import + 15 CLI tests + `SystemValidator.MIN_PYTHON == (3, 11)`;
  `validate_test_contracts --strict` green; `rewrite_readme_agents --check`
  green, 1617 files current). Both lane PRs had green pre-merge CI (runs
  36285562519 / 36285588084); their main push runs (36286371848 /
  36286450117) were CANCELLED by ci.yml cancel-in-progress when PR #43
  (GNN pair bump c30) pushed at 01:57Z; the successor run at tip
  `0b482cfa` (36287083986) re-measured the union — all substantive jobs
  success, slow-scheduled failed once on the ART VGG19 live-fetch flake
  (below) and passed on the bounded rerun (attempt 2, 02:32:42Z).
- **RESOLVED 2026-09-27 (owner-authorized)**: CI-01 ≙ GS19-01 v0.3.0 wheel
  re-release — tag-push validate attempt 2 green (run 35274969748), release
  run 35274969824 rerun green (ci-gate + release), the 45 wheels built as
  the run artifact `geo-infer-wheels` and attached to the release per the
  owner's go: `v0.3.0` now carries 46 assets (45 `.whl` + the manuscript
  PDF). Pipeline gap (wheels never reach the release object without a
  manual step) filed as **REL-02**.
- **Branch hygiene**: `wave/scope-2026-09-19` fully delivered via PR #35
  (`6f8fd263`); obsolete, retained for history. No repo TASKS.md exists
  (`git ls-tree -r main`); this TODO.md is the sole task ledger. No
  completed TODO rows were cleared this pass — REL-01 closed 2026-09-27
  (below); the remaining pre-existing open rows (SPACE-01, PLACE-V14,
  PLACE-04, CODE-01, TEST-GNN-01, TEST-04) are all externally blocked and
  verified still open; the three new self-serve rows (ANT-DEP-01,
  ART-FLAKE-01, REL-02) are filed in the Minor table.
- **Execution findings → new Minor rows `ANT-DEP-01` and `ART-FLAKE-01`
  (below)**: ANT's undeclared `geo_infer_math` test dep (the deleted
  availability guard masked it; resolves via workspace-wide sync today), and
  the ART `test_style_transfer.py` live VGG19 GCS download that flaked the
  scheduled slow lane (run 36287083986 attempt 1, network reset; rerun
  green).
- **Superseded 2026-09-27**: the cli.py:256/:340 geopandas sites (try/log-
  reraise wrappers) were recorded here as a verified non-defect; ModuleSweep
  and an independent advisory reclassified them as a half-finished M5-03
  cutover — filed as **HEALTH-01** (finish the cutover to match the landed
  :441/:572 treatment).

## Scope pass (2026-09-27)

Six-lane read-only scout fleet (DepSweep, FlakeHunter, WorkflowLens,
DocsTruth, ModuleSweep, AuditVerifier) against `main @ 27557432` plus two
implementation-verification passes; the integrated spec with per-item
evidence, probes and execution order is
[SCOPE-2026-09-27.md](SCOPE-2026-09-27.md) — 19 items (3 Medium, 16 Minor).
AuditVerifier independently confirmed 13/14 of the 2026-09-26 dispositions;
its 1 REFUTE (DOC-05's bold-variant stamp family, invisible to the
plain-phrase probe) was fixed same-day. A 6/6 inline spot-check of other
DELIVERED verdicts also confirmed.

**Landed the same wave:**

- **ANT-DEP-01**: PR #47 (merge `4e592897`) — `geo-infer-math>=0.3.0` in
  ANT's integrations extra + workspace pin; `uv.lock` +2 lines; row deleted.
- **ART-FLAKE-01**: PR #46 (merge `a5dd2ccf`) — offline VGG19 fixture (real
  architecture, seeded local weights, URL/socket blockers assert); zero
  live-fetch sites left in ART tests; row deleted.
- **REL-02**: PR #48 — job-scoped `contents: write` on the release job;
  attach-wheels step gated on tag refs (warn+skip when the release object is
  absent — release-object creation stays an owner action); receipt check
  fails on a short asset count; contract pins mutation-verified. Residual:
  the manuscript PDF attach (REL-03). Row deleted.
- **CI-06**: PR #49 — scheduled runs get their own concurrency bucket
  (landed before the first scheduled fire, Mon 2026-09-28 06:00 UTC).
- **DOC-05 residue**: PR #50 (merge `f844c21e`) — all 8 bold-variant
  `**Last Updated**: 2026-02-24` stamps removed (5 already false per git
  log); `2026-02-24` now has zero hits repo-wide. Follow-up DOC-09 filed
  for the other-date stamp families.
- **FLK-01** (filed from FlakeHunter after `test_import_smoke_timeout_stops_descendants`
  blocked three CI runs on 2026-09-28 — runs 36361629171 ×2, 36363203677):
  PR #51 (merge `1f050175`) — 30 s latency-tolerant poll; the termination
  property is asserted in both spawn outcomes.

**Queued (15 rows below)**: REL-03, DEP-02 (Medium); CI-07, CI-08, CI-09,
DEP-03, FLK-02, FLK-03, FLK-04, TST-09, TST-10, HEALTH-01, DOC-06, DOC-07,
DOC-08, DOC-09 (Minor). HEALTH-01 supersedes the 2026-09-26
"verified non-defect" note below: ModuleSweep reclassified the
cli.py:256/:340 sites as a half-finished M5-03 cutover, and consistency
with the landed :441/:572 treatment argues for finishing it.

## Legacy sweep and closures (2026-10-01)

Full review of `main @ 510f1008` (local `main` == `origin/main`), followed by a
repository-wide legacy removal and modernization branch
(`review/2026-10-01-legacy-sweep`); details in `CHANGELOG.md` `[Unreleased]`;
prioritized next steps with file paths in [HANDOFF-2026-10-01.md](HANDOFF-2026-10-01.md).

**Closed rows (removed from the tables above):**

- **CI-07, CI-08, CI-09, REL-03**: delivered by PR #53 (tag-scoped release
  gate, target-keyed release queue, manuscript PDF attach, no import-probe
  push leg); contract pins present in `test_ci_workflow_contracts.py`.
- **FLK-02, FLK-03, FLK-04**: delivered by PRs #54/#59; the sweep further
  replaced OPS's sleep-assert with a deadline poll and made the PEP talent
  importer clock injectable.
- **TST-09, HEALTH-01**: delivered by PR #56. **TST-10**: PR #56 plus the
  sweep (SPACE io dead `HAS_*` guards and the anti-pattern docstring removed).
- **DEP-03**: retired at the root — module `requirements.txt` mirrors (and
  `setup.py` shims) are deleted and `validate_repo_contracts.py` now rejects
  them; `pyproject.toml` + `uv.lock` are the only dependency declaration.
- **DEP-02 / TST-11**: `validate_packaging.py` gains
  `validate_test_import_parity` (every third-party or sibling import under a
  module's `tests/`, including guarded, function-local and string-literal
  dynamic imports, must be declared as a runtime dep, an extra, or a PEP 735
  `[dependency-groups]` entry); the remaining undeclared test imports (HEALTH
  pyogrio, OPS and TEST workspace siblings) live in `test` groups with
  `[tool.uv.sources]` workspace pins. CI and the canonical sync command add
  `--all-groups`.
- **DOC-06**: `[Unreleased]` CHANGELOG section opened. **DOC-07**: ISA
  cross-reference repointed at `[0.2.0]`. **DOC-08**: CLAUDE.md "Module
  Themes" now mirrors the generated README theme table, which the generator
  enforces lists every module exactly once (RISK was missing).

**Still open:** DOC-09 (other-date
stamps), and the externally blocked Major/Medium rows.

## DOC-09 closure (2026-10-05)

Removed the 11 manually maintained `Last Updated` footers named in DOC-09.
Seven were stale after meaningful content changes; the other four were removed
to avoid maintaining dates separately from Git history. Narrative content,
examples, citations and links remain unchanged.

The INTRA template `last_updated` fields had already been removed in the
2026-10-01 documentation cleanup; those templates required no further change.
The October 1 open-status note above is retained as a historical statement.
