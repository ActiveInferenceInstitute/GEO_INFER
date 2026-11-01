# GEO-INFER 0.4.0 companion compatibility

The release identities in `.github/companion-releases.json` connect published
[GEO-INFER 0.4.0](https://github.com/ActiveInferenceInstitute/GEO_INFER/releases/tag/v0.4.0)
to published GNN 5.0.1 and FEP_Formal 2.1.0. GEO's released source is
`71ce1f77418c6f8c6edc8f83c7682c8eab207e01`, tree
`46a4febd0a5a794c275afc5f74f4412bca8885b4`, annotated tag object
`891d4bb6774c31028891180d4dc1d0a90a28f127`. These identities and pins do
not qualify subsequent source, numerical behavior, manuscript output or
hosted results; later changes require their own acceptance.

| Project | Published tag | Exact source commit | Exact annotated tag object | Automatic source DOI |
| --- | --- | --- | --- | --- |
| GNN | `v5.0.1` | `e54813b4197df36f7ae0054d5e79f3ad2d5db0ff` | `35e71d1a6f109f6e0d08fda2acd5a94376482cf4` | [10.5281/zenodo.23631909](https://doi.org/10.5281/zenodo.23631909) |
| FEP_Formal | `v2.1.0` | `96bf714d91c088e79b9dea54af0f9151f70de64f` | `6bc61ffa58428190055245781328820c7c86abac` | [10.5281/zenodo.23631895](https://doi.org/10.5281/zenodo.23631895) |

The published release pages are [GNN 5.0.1](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases/tag/v5.0.1)
and [FEP_Formal 2.1.0](https://github.com/ActiveInferenceInstitute/fep_formal/releases/tag/v2.1.0).

The DOI links identify automatic software source archives. Verifying the
public metadata, downloaded source bytes, separate distribution assets and
manuscript publication remains a separate release procedure. The offline
identity check performs no network requests or DOI resolution.

## Direction of the pins

The manifest records the published release identities above: GEO 0.4.0 was
released with GNN 5.0.1 and FEP_Formal 2.1.0, and that GNN release's own
`.github/fep-lean-pair.json` selects the exact FEP_Formal source above. GNN
5.0.1's own GEO pin remains the accepted historical GEO commit
`21b65a26dd15288d6e578baef86a8bcfd6a7b5db`. A new GEO candidate must retain
that commit as an ancestor and obtain its own paired interchange evidence.
GNN's already published tag does not pin a later GEO release.

GEO's `.github/gnn-pair.json` is the current interchange pin and its only home.
It advances through the documented
[pair-pin bump procedure](../../../GEO-INFER-TEST/docs/gnn_space_time_2026_09.md#gnn-pair-pin-bump-procedure),
so it may select a later GNN revision than the published companion. The identity
check verifies the pin's shape and records whether it still equals the published
companion; the revision pair's conformance is owned by the paired GNN
interchange workflow, which exports in a locked GNN environment and consumes in
GEO.

| Date | Interchange pin | Basis |
| --- | --- | --- |
| 2026-11-01 | `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1` (GNN v5.0.3) | Advanced from `e54813b4…` (v5.0.1) after reviewing the delta: `src/gnn/export/`, `src/gnn/parsers/` and `src/gnn/extract/` are byte-identical between the two revisions and the schema roster is unchanged, so the exported contract surface does not move. Supersedes the earlier `codex/gnn-companion-pin-20261028` attempt, whose target `4582f99c…` is an ancestor of v5.0.3. |

The supported interchange schemas remain `gnn-geo-infer/1`,
`gnn-geo-infer/2` and `gnn-geo-infer/factored/1`. The existing
[two-environment interchange procedure](../../../GEO-INFER-TEST/docs/gnn_space_time_2026_09.md#gnn-pair-pin-bump-procedure)
and [ACT interchange guide](../../../GEO-INFER-ACT/docs/gnn_interchange.md)
own exporter/consumer execution. Source identity checks do not execute
categorical, H3, Gaussian or factored models and do not establish numerical
equivalence.

## Run the offline identity check

Use three clean, separate local checkouts. The companion checkouts must be at
the exact released commits and contain their actual annotated tag objects;
the GEO checkout must contain the committed manifest and current pin.
No scientific dependencies, environment synchronization or native proof build
is needed for this stdlib-only command on Python 3.11 or newer:

```bash
PYTHONPATH=GEO-INFER-TEST/src python -B GEO-INFER-TEST/validate_companion_releases.py \
  --geo-root /absolute/path/to/GEO_INFER \
  --gnn-root /absolute/path/to/Generalized_Notation_Notation \
  --fep-root /absolute/path/to/fep_formal \
  --expected-geo-revision FULL_ACTUAL_GEO_COMMIT
```

The result binds exact HEAD/tree/tag identities, project and citation versions,
the released GNN pins, GEO's current interchange pin and whether it still equals
the published companion revision, manifest bytes, GEO baseline ancestry and
unchanged before/after inputs. A dirty checkout, lightweight or moved companion
tag, wrong version, disconnected GEO history or malformed interchange pin fails
the check. The local check trusts the reviewed pin and manifest and the supplied
Git objects; it does not authenticate a remote origin, resolve an advanced pin's
revision or establish publication acceptance.

## Review the interchange pin's release lineage

The identity check verifies the pin's shape and records whether it equals the
published companion; it does not resolve an advanced revision. The separate
bounded check reads the pin's lineage from a local GNN checkout without
fetching, and reports the release tags at the pin, how far the checkout has
moved past it and whether the pin is behind the newest release tag.
`--require-current` turns a pin behind the newest release tag into a refusal:

```bash
PYTHONPATH=GEO-INFER-TEST/src python -B GEO-INFER-TEST/validate_gnn_pin_freshness.py \
  --gnn-root /absolute/path/to/Generalized_Notation_Notation --require-current
```

The pin must exist in the checkout and retain the published companion revision
as an ancestor. The receipt is local Git evidence only; it does not fetch,
authenticate tags or execute any model.

The standalone `companion-releases.yml` workflow checks out the actual GEO
event SHA and the two literal companion SHAs. It retains the resulting JSON
disposition. This complements the scientific two-environment workflow; it
does not replace it. Its remote checkouts provide the hosted source context
that is absent from a purely local invocation.

## Formal and historical evidence

FEP_Formal 2.1.0's catalogue contains 168 topics (`fep-001` through `fep-168`)
in 22 declared families. Topic statements, assumptions, semantic maturity and
native compilation receipts retain the scope of their actual FEP source.
The [notation bridge](../../../GEO-INFER-ACT/docs/fep_lean_notation_bridge.md)
is a correspondence guide; it does not prove GEO code or extend an instance
proof to later GEO source.

The retained Cascadia reference fixture provenance names historical GEO
`2e20fe2e86d19c1f5dfb6184ae88b5f224731a9c`. Its slice-local native receipt
uses Lean 4.34.1. These source and toolchain identities remain historical even
when the files are included in FEP_Formal 2.1.0. New source, variants or
sampled trajectories cannot inherit that receipt. See the existing
[Cascadia guide](../../../GEO-INFER-ACT/docs/cascadia_river_homing.md) for the
finite-model and implementation boundaries.

## Current GEO release evidence

The published [0.4.0 evidence](../releases/0.4.0_readiness.md) separates exact
source identity, paired numerical interchange, native proof, installed profiles,
manuscript review and publication. All 52 GitHub assets passed anonymous and
authenticated byte verification; the checksum file lists the other 51 assets.
The 46 successful tag-qualification jobs and separate two operational release
jobs precede the explicit asset publication and do not substitute for it.
Hardware, configured-advisor and historical-cause limitations remain open.

For later releases, follow the [research artifact contracts](research_artifacts.md):
freeze source, refresh generator-owned config in a separate commit when required,
and collect fresh local and hosted evidence. The 0.4.0 wheel-operation registry
defines 26 independent profiles; derive a future required roster from its actual
registry rather than reusing a historical count. Preserve the published sources,
annotated tags, assets and original receipts.

## Next cross-repository handoff: E4 and S1

E4 and S1 are GNN roadmap identifiers, not new GEO interchange schemas or claims
that the linked releases completed those workstreams. The
[maintained GNN roadmap](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/TO-DO.md)
and its [bounded contracts](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/development/remaining_roadmap_contracts.md)
own their current implementation scope; these links can advance beyond the
published 5.0.1 source. FEP's [maintained backlog](https://github.com/ActiveInferenceInstitute/fep_formal/blob/main/TODO.md)
retains its independently governed proof and empirical lanes.

| Workstream | Ownership and first handoff | Acceptance before extending claims |
| --- | --- | --- |
| E4: formal-to-numerical semantics | GNN extraction/semantic-witness owners, GEO ACT/SPACE/TIME consumer owners and the relevant FEP theorem owner agree one explicitly admitted finite or Gaussian fixture. Record exact source identities, independent extraction, quantities, axes, conditioning/time, authored actions, units and precision. | Compare independently specified numerical references and actual native results. Include source, axis, action-time, conditioning and covariance mismatch controls. Bind any formal statement to its actual assumptions and current native compiler receipt separately; matching hashes, shapes or values and successful runners do not prove general equivalence. |
| S1: source-preserving multi-request processing | GNN Step 13 scheduler/provider owners define the synthesis/checkpoint contract. A participating GEO workflow supplies an immutable manifest of every selected source byte and required prompt, plus cross-part questions whose answer requires the full selection. | Verify provider/model/context admission, genuine terminal completion and joint synthesis for every required task, cross-part evidence, source-preserving resume, cancellation/deadline behavior and explicit unfinished/refused work. Independent fragment answers or displayed saved receipts do not establish joint interpretation or new model execution. |

Before implementation, agree the bounded contract, owning files, source baselines,
required negative controls and which companion owners change. An extension beyond
the admitted model grammar or proof assumptions needs a new explicit contract.
After implementation, collect fresh native, numerical, formal and custody evidence
for their separate scopes, and use the
[paired-revision procedure](../../../GEO-INFER-TEST/docs/gnn_space_time_2026_09.md#gnn-pair-pin-bump-procedure)
when source/pin changes require it. This documentation handoff changes no pins,
formal statements or runtime behavior. GNN's historical GEO `21b65a26` baseline
and the historical Cascadia GEO `2e20fe2e` / Lean 4.34.1 receipt remain historical;
neither is refreshed by this roadmap.
