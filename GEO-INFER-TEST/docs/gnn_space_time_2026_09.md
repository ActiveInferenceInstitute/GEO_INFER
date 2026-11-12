# Space, time and GNN integration verification

Historical first-phase receipt. See [the continuation](gnn_continuation_2026_09.md)
for later capabilities, passing combined suites and current verification limits.

This work extends GEO-INFER from `f098b292` and GNN from `64d49355` in isolated
checkouts. Both original checkouts were changing concurrently; this receipt does
not describe their evolving uncommitted trees. The GNN baseline includes existing
fleet commits `3f2694d3a` and `64d49355a`; its topic branch retains that ancestry.

> History note (2026-09-07): the published history was rewritten to re-attribute
> hum-side personal commit identities to docxology
> <docxology@users.noreply.github.com>. Every commit SHA recorded in this
> receipt before that rewrite refers to pre-rewrite history and no longer
> resolves; the recorded evidence and run results remain valid, only the
> identifiers changed.

## Delivered behavior

- GNN exports an opt-in `gnn-geo-infer/1` categorical JSON contract with explicit
  A–E matrices, physical timestep, exact state order and source provenance.
- SPACE supplies sparse H3 stay/diffuse operators with real pentagon topology,
  reflecting domain boundaries and a bounded dense conversion.
- TIME validates aware UTC schedules and rejects duplicate, reversed or missing
  model intervals before inference begins.
- ACT preserves E in real pymdp policy inference, conditions and propagates once
  per timestamp, rejects impossible observations and validates matrix shape
  before allocating NumPy arrays.
- Continuous ACT filtering uses the correct Gaussian KL and accuracy sign,
  Gaussian entropy, solve/Joseph updates, conditional mutual information and
  explicit continuous versus discrete dynamics.
- GNN B-axis diagnostics and canonicalization support unequal state/action
  dimensions; canonicalization copies data and remains idempotent.

## Evidence

| Check | Result |
| --- | --- |
| Python 3.11 affected module suite | 1,573 passed, no failures/errors/skips |
| Python 3.12 affected module suite | ACT and TIME passed; 32 SPACE failures/errors reported a PROJ SQLite disk-I/O error |
| Python 3.12 SPACE fresh-process rerun | All 587 passed; separate SQLite quick-check and coordinate transform passed |
| Final Gaussian/artifact regressions | 35 passed on each Python version, including the additional allocation-order regression |
| GNN isolated export/extractor suite | 104 passed |
| GNN strict documentation audit | Zero link, anchor or directory-signpost issues |
| Cross-repository real conformance | Gridworld and seven-cell H3 round trips passed on both GEO Python versions using a separate locked GNN Python 3.11 environment; complete traces matched |
| Packaging | ACT, SPACE and TIME wheels validated byte-for-byte against source, installed and imported from fresh site-packages |
| Repository gates | All ten passed, including all 44 package import probes with zero errors/warnings (explicit 120-second import bound), packaging, logging, docs, skills, active inference, signposts, test contracts, model contracts and reproducibility |
| Code quality | Changed Python files formatted; targeted critical lint and whitespace checks passed |

The affected suite contains 1,573 tests before the final allocation-order
regression. With that additional passing test, 1,574 unique affected tests have
passing evidence per Python version across the suite and focused reruns. The
Python 3.12 combined-process result is retained as a limitation; no CRS assertion
was weakened. `TEST-GNN-01` scopes its investigation.

Cross-environment artifact digests:

- Gridworld: `0bcd9d7dc78951754533a24f11d757d9d001491eb8a3772ce0af063995d9647d`
- H3: `caa97b84f98946c6b459a4b0fbe526fcf4fca2a862eaf1797e0622ecc5d67019`

[DEFERRED-VERIFY] The configured advisor exited with an error; no advisor review ran. Repeat advisor review when the configured service is available. [DEFERRED-VERIFY] Neither repo
had an available GitNexus index. Build/check repository-specific indexes before relying on indexed impact reports. Direct source/caller review and numerical tests
provide the evidence above; indexed impact analysis remains unavailable.

## Reproduce and integrate

See [the ACT integration guide](../../GEO-INFER-ACT/docs/gnn_interchange.md) for
setup and the executable two-environment conformance command. GNN's producer
contract is `src/gnn/export/geo_infer_contract.md` in its repository
(GNN revisions before its 2026-09 package reorganization, including pinned pairing commit 903b9c339, use `src/export/geo_infer_contract.md`).

The topic branches are `codex/gnn-space-time` (GEO) and
`codex/geo-infer-interchange` (GNN). Main-branch integration must reconcile
concurrent changes before repeating conformance and remote-SHA checks. The TODO
files scope Gaussian/multifactor interchange, longer policies, irregular sampling,
sparse/multiresolution grids, Step 7 integration and legacy ACT timing alignment.

GNN topic publication verified at `8005e37c668e91b85f6d54b1a989983098ee99a8`.

## GNN pair-pin bump procedure

The paired interchange pins the companion GNN revision in
`.github/gnn-pair.json` (`revision`) and records the reviewed release label
alongside it (`tag`). The file carries the canonical importer schema
`gnn-importer-pin/v1` together with `repository` and `recorded` fields; the
current pin is `v5.0.3` at `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`.
`.github/workflows/gnn-interchange.yml` checks out GEO at the PR head SHA and
GNN at that pin in two locked environments (Python 3.11 and 3.12); the job key
is `interchange`, and its `3.12` leg publishes the required check name
`gnn conformance (pinned)` while the `3.11` leg publishes
`gnn interchange (3.11)`. `.github/GNN.md` records what that job parses and
what it deliberately does not cover.

The revision is authoritative; the tag is a review label and CI rejects a pin
without a valid `vMAJOR.MINOR.PATCH` tag. The importer pin is not the
release-facing companion identity in `.github/companion-releases.json`
(schema `geo-infer-companion-releases/v1`, GNN `5.0.1` and FEP_Formal `2.1.0`
for GEO-INFER `0.4.0`): a reviewed importer pin may run ahead of the published
companion, and neither file substitutes for the other. GNN `main` usually runs
ahead of the pin, so bumping is a deliberate, reviewed companion-revision
decision, never automatic:

1. Diff GNN from the current pin to the target revision
   (`git diff <pin>..<target>` in the GNN checkout) and review the change.
2. Confirm contract-version compatibility: the consumer boundary
   `geo_infer_act.core.gnn_contract` (and its Gaussian and factored siblings)
   must still accept the `schema_version` values the target GNN exporters emit
   (`gnn-geo-infer/1`, `gnn-geo-infer/2`, `gnn-geo-infer/factored/1`).
3. Open a PR that updates **only** `.github/gnn-pair.json` to the target
   revision and its release tag; never mix a pin bump with consumer-contract
   code changes. The published-companion identity check verifies the pin's
   shape, records whether it still equals the published companion and accepts a
   reviewed pin ahead of it.
4. Record the bounded lineage receipt
   (`validate_gnn_pin_freshness.py --gnn-root <gnn> --require-current`) in the
   PR, and let the paired interchange CI validate the new pair on the PR (the
   workflow triggers on the pin path); merge once it is green.

## Publication receipt

- GEO implementation: `e028aa9060e05f765762224499f5e2c714cf25a3`, pushed to `codex/gnn-space-time` with remote SHA parity verified.
- GNN topic: `92255b125b46fd64de45e2628fce4c8e9c5f89d2`, pushed to `codex/geo-infer-interchange` with remote SHA parity verified.
- Main-branch integration remains the scoped follow-up; original shared checkouts were not reset or cleaned.
